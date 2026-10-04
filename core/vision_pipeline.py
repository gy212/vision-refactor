from __future__ import annotations

import os
import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Reduce noisy native logs (glog/TFLite) in console/UI.
# Use hard set (not setdefault) so user env doesn't accidentally re-enable spammy logs.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import cv2
import mediapipe as mp
import numpy as np


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC (in degrees) using 2D points a,b,c."""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = float(np.dot(ba, bc) / denom)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def _pt2d(lm, w: int, h: int) -> np.ndarray:
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def _lm2d(lm, w: int, h: int) -> np.ndarray:
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def _visibility(lm) -> float:
    # Tasks NormalizedLandmark usually includes visibility; keep a safe fallback.
    return float(getattr(lm, "visibility", 1.0))


def _finger_extended(lm, w: int, h: int, mcp: int, pip: int, dip: int) -> bool:
    # Use joint angle at PIP: closer to 180deg => straighter finger.
    a = _lm2d(lm[mcp], w, h)
    b = _lm2d(lm[pip], w, h)
    c = _lm2d(lm[dip], w, h)
    return _angle_deg(a, b, c) > 150.0


def _finger_curled(lm, w: int, h: int, mcp: int, pip: int, dip: int) -> bool:
    a = _lm2d(lm[mcp], w, h)
    b = _lm2d(lm[pip], w, h)
    c = _lm2d(lm[dip], w, h)
    return _angle_deg(a, b, c) < 150.0


def _is_v_sign(hand_lm, w: int, h: int) -> bool:
    """
    Very lightweight "V sign" rule using only hand landmarks:
      - index & middle extended
      - ring & pinky curled
      - index tip and middle tip reasonably separated
    """
    if hand_lm is None:
        return False

    lm = hand_lm
    # Indices: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
    idx_ext = _finger_extended(lm, w, h, 5, 6, 7)
    mid_ext = _finger_extended(lm, w, h, 9, 10, 11)
    ring_cur = _finger_curled(lm, w, h, 13, 14, 15)
    pinky_cur = _finger_curled(lm, w, h, 17, 18, 19)

    if not (idx_ext and mid_ext and ring_cur and pinky_cur):
        return False

    index_tip = _lm2d(lm[8], w, h)
    middle_tip = _lm2d(lm[12], w, h)
    wrist = _lm2d(lm[0], w, h)
    middle_mcp = _lm2d(lm[9], w, h)
    hand_scale = float(np.linalg.norm(middle_mcp - wrist) + 1e-6)

    return float(np.linalg.norm(index_tip - middle_tip)) > 0.25 * hand_scale


def _classify_pose_actions(landmarks, w: int, h: int) -> list[str]:
    if landmarks is None:
        return []

    lm = landmarks

    # MediaPipe pose landmark indices (BlazePose, 33 points)
    L_SHOULDER = 11
    R_SHOULDER = 12
    L_WRIST = 15
    R_WRIST = 16
    L_HIP = 23
    R_HIP = 24
    L_KNEE = 25
    R_KNEE = 26
    L_ANKLE = 27
    R_ANKLE = 28

    # Normalized coordinates: x,y in [0,1], y smaller => higher on screen.
    l_wrist = lm[L_WRIST]
    r_wrist = lm[R_WRIST]
    l_sh = lm[L_SHOULDER]
    r_sh = lm[R_SHOULDER]

    actions: list[str] = []

    left_hand_up = _visibility(l_wrist) > 0.5 and _visibility(l_sh) > 0.5 and (l_wrist.y < l_sh.y - 0.05)
    right_hand_up = _visibility(r_wrist) > 0.5 and _visibility(r_sh) > 0.5 and (r_wrist.y < r_sh.y - 0.05)
    if left_hand_up and right_hand_up:
        actions.append("HANDS_UP")
    elif left_hand_up:
        actions.append("LEFT_HAND_UP")
    elif right_hand_up:
        actions.append("RIGHT_HAND_UP")

    # Squat (very rough): both knee angles small + hip close to knee vertically.
    def knee_ok(side: str) -> bool:
        if side == "L":
            hip_lm = lm[L_HIP]
            knee_lm = lm[L_KNEE]
            ankle_lm = lm[L_ANKLE]
        else:
            hip_lm = lm[R_HIP]
            knee_lm = lm[R_KNEE]
            ankle_lm = lm[R_ANKLE]

        v_ok = _visibility(hip_lm) > 0.5 and _visibility(knee_lm) > 0.5 and _visibility(ankle_lm) > 0.5
        if not v_ok:
            return False

        hip = _pt2d(hip_lm, w, h)
        knee = _pt2d(knee_lm, w, h)
        ankle = _pt2d(ankle_lm, w, h)

        ang = _angle_deg(hip, knee, ankle)  # 180 ~= straight
        hip_knee_dist = abs((hip[1] - knee[1]) / max(h, 1))  # normalized vertical distance
        return ang < 125.0 and hip_knee_dist < 0.18

    if knee_ok("L") and knee_ok("R"):
        actions.append("SQUAT")

    return actions


_ACTION_POSE_UNSET = object()


def draw_pose_frame(
    frame_bgr: np.ndarray,
    pose_landmarks,
    hands,
    *,
    draw_face: bool,
    draw_joint_angles: bool,
    action_pose_landmarks=_ACTION_POSE_UNSET,
) -> tuple[np.ndarray, list[str]]:
    """无状态绘制：在帧副本上画 pose/hands 并返回 ``(annotated, actions)``。

    不依赖 MediaPipe 模型实例，供并行预览路径在重排出口的主线程直接绘制平滑后的 landmarks
    （避免在消费线程再构造一个加载模型的管线）。``action_pose_landmarks`` 可单独指定用于
    动作分类的 raw landmarks，使平滑结果只影响骨架绘制；省略时保持旧行为。逻辑与
    ``MediaPipePipeline.annotate`` 的绘制步骤一致。
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    action_pose = (
        pose_landmarks
        if action_pose_landmarks is _ACTION_POSE_UNSET
        else action_pose_landmarks
    )
    actions = _classify_pose_actions(action_pose, w, h)

    MediaPipePipeline._draw_pose(out, pose_landmarks, w, h, draw_face=draw_face)
    if draw_joint_angles:
        MediaPipePipeline._draw_joint_angles(out, pose_landmarks, w, h)

    MediaPipePipeline._draw_hands(out, hands, w, h)
    if any(_is_v_sign(hand, w, h) for hand in hands):
        actions.insert(0, "V_SIGN")

    return out, actions


def _pose_model_url(variant: str) -> str:
    # 模型清单已收口到 core.model_manager，避免 URL 在多处漂移。
    from core.model_manager import MEDIAPIPE_MODELS

    key = {"lite": "pose_lite", "full": "pose_full", "heavy": "pose_heavy"}.get(
        (variant or "").strip().lower(), "pose_lite"
    )
    for spec in MEDIAPIPE_MODELS:
        if spec.key == key:
            return spec.url
    return ""


def _ensure_file(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model to: {path}")
    urllib.request.urlretrieve(url, path)  # nosec - official model asset


def _normalize_delegate(delegate: str) -> str:
    delegate_norm = (delegate or "cpu").strip().lower()
    if delegate_norm in {"cpu", "gpu"}:
        return delegate_norm
    raise ValueError(f"Unsupported MediaPipe delegate: {delegate!r} (use 'cpu' or 'gpu')")


def _base_options(model_path: Path, *, delegate: str) -> mp.tasks.BaseOptions:
    delegate_norm = _normalize_delegate(delegate)
    # Windows 原生模型加载器不能可靠打开中文路径；Python 读字节避开其窄字符文件接口。
    asset = ({"model_asset_buffer": model_path.read_bytes()}
             if os.name == "nt" and not str(model_path).isascii()
             else {"model_asset_path": str(model_path)})
    if delegate_norm == "cpu":
        return mp.tasks.BaseOptions(**asset)
    if delegate_norm == "gpu":
        delegate_enum = getattr(mp.tasks.BaseOptions, "Delegate", None)
        gpu_delegate = getattr(delegate_enum, "GPU", None) if delegate_enum is not None else None
        if gpu_delegate is None:
            raise RuntimeError("MediaPipe BaseOptions.Delegate.GPU is not available in this mediapipe build")
        return mp.tasks.BaseOptions(**asset, delegate=gpu_delegate)
    raise AssertionError("unreachable delegate branch")


@dataclass(frozen=True)
class PipelineConfig:
    pose_variant: str = "full"  # lite/full/heavy
    num_poses: int = 1
    num_hands: int = 2
    running_mode: str = "video"  # "video" (tracking) or "image" (per-frame, parallel-friendly)
    draw_pose_face: bool = False  # Disable by default to avoid visual confusion when hands are near the face.
    draw_joint_angles: bool = False  # Overlay knee/elbow joint angles on the annotated preview (UI opt-in).
    enable_hands: bool = True  # Set False for pose-only extraction (faster), e.g. template matching.
    delegate: str = "cpu"  # Explicit opt-in only: "cpu" keeps the legacy BaseOptions path, "gpu" probes GPU delegate.
    delegate_fallback_to_cpu: bool = True
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_pose_tracking_confidence: float = 0.5
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_hand_tracking_confidence: float = 0.5


class MediaPipePipeline:
    def __init__(self, *, models_dir: Path, cfg: PipelineConfig = PipelineConfig()) -> None:
        self.cfg = cfg
        self.models_dir = models_dir
        self.requested_delegate = _normalize_delegate(cfg.delegate)
        self.active_delegate = self.requested_delegate
        self.delegate_fallback_reason: str | None = None

        pose_path = self.models_dir / f"pose_landmarker_{cfg.pose_variant}.task"
        _ensure_file(_pose_model_url(cfg.pose_variant), pose_path)
        hand_path: Path | None = None
        if cfg.enable_hands:
            hand_path = self.models_dir / "hand_landmarker.task"
            _ensure_file(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/latest/hand_landmarker.task",
                hand_path,
            )

        mode = (cfg.running_mode or "video").lower()
        if mode not in ("video", "image"):
            raise ValueError(f"Unsupported running_mode: {cfg.running_mode!r} (use 'video' or 'image')")
        self.running_mode = mode
        mp_mode = mp.tasks.vision.RunningMode.VIDEO if mode == "video" else mp.tasks.vision.RunningMode.IMAGE

        self.pose_landmarker = None
        self.hand_landmarker = None
        try:
            self._create_landmarkers(
                pose_path=pose_path,
                hand_path=hand_path,
                mp_mode=mp_mode,
                delegate=self.requested_delegate,
            )
        except Exception as exc:
            if self.requested_delegate != "gpu" or not bool(cfg.delegate_fallback_to_cpu):
                raise
            self.delegate_fallback_reason = str(exc)
            self.active_delegate = "cpu"
            try:
                self._create_landmarkers(
                    pose_path=pose_path,
                    hand_path=hand_path,
                    mp_mode=mp_mode,
                    delegate="cpu",
                )
            except Exception as cpu_exc:
                raise RuntimeError(
                    "MediaPipe GPU delegate failed and CPU fallback also failed"
                ) from cpu_exc

        self._t0 = time.monotonic()
        self._frame_index = 0

    def _create_landmarkers(
        self,
        *,
        pose_path: Path,
        hand_path: Path | None,
        mp_mode: object,
        delegate: str,
    ) -> None:
        cfg = self.cfg
        self.pose_landmarker = None
        self.hand_landmarker = None
        try:
            self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
                mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=_base_options(pose_path, delegate=delegate),
                    running_mode=mp_mode,
                    num_poses=cfg.num_poses,
                    min_pose_detection_confidence=cfg.min_pose_detection_confidence,
                    min_pose_presence_confidence=cfg.min_pose_presence_confidence,
                    min_tracking_confidence=cfg.min_pose_tracking_confidence,
                )
            )
            if cfg.enable_hands and hand_path is not None:
                self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
                    mp.tasks.vision.HandLandmarkerOptions(
                        base_options=_base_options(hand_path, delegate=delegate),
                        running_mode=mp_mode,
                        num_hands=cfg.num_hands,
                        min_hand_detection_confidence=cfg.min_hand_detection_confidence,
                        min_hand_presence_confidence=cfg.min_hand_presence_confidence,
                        min_tracking_confidence=cfg.min_hand_tracking_confidence,
                    )
                )
        except Exception:
            self.close()
            raise

    def next_timestamp_ms(self, *, is_file: bool, fps_for_ts: float) -> int:
        if is_file:
            return int(self._frame_index * 1000.0 / max(1e-6, fps_for_ts))
        return int((time.monotonic() - self._t0) * 1000)

    def annotate(self, frame_bgr: np.ndarray, *, timestamp_ms: int | None = None) -> tuple[np.ndarray, list[str]]:
        pose_landmarks, hands = self.infer(frame_bgr, timestamp_ms=timestamp_ms)
        out, actions = self.draw(frame_bgr, pose_landmarks, hands)
        self._frame_index += 1
        return out, actions

    def draw(
        self,
        frame_bgr: np.ndarray,
        pose_landmarks,
        hands,
        *,
        action_pose_landmarks=_ACTION_POSE_UNSET,
    ) -> tuple[np.ndarray, list[str]]:
        """在 ``frame_bgr`` 上绘制给定 landmarks，返回 ``(annotated, actions)``。

        从 ``annotate()`` 抽出的可复用绘制步骤（不含 infer、不动 ``_frame_index``），供
        「infer → 重排 → 预览平滑 → draw」这条单线程预览路径在重排出口调用。传入平滑后的
        landmarks 即得平滑预览；传入 raw 则与旧 ``annotate()`` 行为逐字节一致。
        """
        return draw_pose_frame(
            frame_bgr,
            pose_landmarks,
            hands,
            draw_face=self.cfg.draw_pose_face,
            draw_joint_angles=self.cfg.draw_joint_angles,
            action_pose_landmarks=action_pose_landmarks,
        )

    def infer(self, frame_bgr: np.ndarray, *, timestamp_ms: int | None = None):
        """Run landmarkers and return raw landmarks (pose_landmarks, hands_list)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.running_mode == "video":
            if timestamp_ms is None:
                raise ValueError("timestamp_ms is required in VIDEO mode")
            pose_res = self.pose_landmarker.detect_for_video(mp_image, int(timestamp_ms))
            hand_res = self.hand_landmarker.detect_for_video(mp_image, int(timestamp_ms)) if self.hand_landmarker else None
        else:
            pose_res = self.pose_landmarker.detect(mp_image)
            hand_res = self.hand_landmarker.detect(mp_image) if self.hand_landmarker else None

        pose_landmarks = pose_res.pose_landmarks[0] if getattr(pose_res, "pose_landmarks", None) else None
        hands = hand_res.hand_landmarks if (hand_res is not None and getattr(hand_res, "hand_landmarks", None)) else []
        return pose_landmarks, hands

    def close(self) -> None:
        if self.pose_landmarker is not None:
            self.pose_landmarker.close()
        if self.hand_landmarker is not None:
            self.hand_landmarker.close()

    @staticmethod
    def _draw_pose(out_bgr: np.ndarray, landmarks, w: int, h: int, *, draw_face: bool) -> None:
        if landmarks is None:
            return

        # PoseLandmarker uses BlazePose's 33 landmarks:
        # indices 0..10 are face-related points (nose/eyes/ears/mouth).
        # Drawing them often creates confusing visuals when hands occlude the face.
        face_max = 10

        # Draw connections first for nicer layering.
        for c in mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS:
            i, j = int(c.start), int(c.end)
            if (not draw_face) and (i <= face_max or j <= face_max):
                continue
            a, b = landmarks[i], landmarks[j]
            if _visibility(a) < 0.3 or _visibility(b) < 0.3:
                continue
            ax, ay = int(a.x * w), int(a.y * h)
            bx, by = int(b.x * w), int(b.y * h)
            cv2.line(out_bgr, (ax, ay), (bx, by), (0, 140, 255), 2, cv2.LINE_AA)

        for idx, p in enumerate(landmarks):
            if (not draw_face) and idx <= face_max:
                continue
            if _visibility(p) < 0.3:
                continue
            x, y = int(p.x * w), int(p.y * h)
            cv2.circle(out_bgr, (x, y), 3, (0, 255, 0), -1, cv2.LINE_AA)

    @staticmethod
    def _draw_joint_angles(out_bgr: np.ndarray, landmarks, w: int, h: int) -> None:
        """Overlay knee (hip-knee-ankle) and elbow (shoulder-elbow-wrist) angles."""
        if landmarks is None:
            return

        # (label_anchor_idx, a, b, c): angle is at vertex b.
        joints = [
            (25, 23, 25, 27),  # left knee
            (26, 24, 26, 28),  # right knee
            (13, 11, 13, 15),  # left elbow
            (14, 12, 14, 16),  # right elbow
        ]
        for vtx, ia, ib, ic in joints:
            a, b, c = landmarks[ia], landmarks[ib], landmarks[ic]
            if _visibility(a) < 0.3 or _visibility(b) < 0.3 or _visibility(c) < 0.3:
                continue
            pa, pb, pc = _pt2d(a, w, h), _pt2d(b, w, h), _pt2d(c, w, h)
            ang = _angle_deg(pa, pb, pc)

            # 加粗重画构成夹角的两条边与三个节点（覆盖在 _draw_pose 的细线之上）。
            # 用与整副骨架（橙 0,140,255 / 绿点）明显不同的高对比色，便于区分夹角。
            ipa = (int(pa[0]), int(pa[1]))
            ipb = (int(pb[0]), int(pb[1]))
            ipc = (int(pc[0]), int(pc[1]))
            cv2.line(out_bgr, ipb, ipa, (255, 0, 255), 5, cv2.LINE_AA)  # 品红边
            cv2.line(out_bgr, ipb, ipc, (255, 0, 255), 5, cv2.LINE_AA)
            for pt in (ipa, ipb, ipc):
                cv2.circle(out_bgr, pt, 6, (0, 0, 255), -1, cv2.LINE_AA)  # 红节点

            # 在顶点 b 处用一段圆弧连接 b→a 与 b→c 两条边（取两向量间的内角短弧）。
            start = math.degrees(math.atan2(pa[1] - pb[1], pa[0] - pb[0]))
            end = math.degrees(math.atan2(pc[1] - pb[1], pc[0] - pb[0]))
            sweep = (end - start + 180.0) % 360.0 - 180.0  # 归一化到 (-180, 180] 走短边
            radius = max(8, int(0.018 * math.hypot(w, h)))
            cv2.ellipse(
                out_bgr,
                (int(pb[0]), int(pb[1])),
                (radius, radius),
                0.0,
                start,
                start + sweep,
                (255, 255, 0),
                4,
                cv2.LINE_AA,
            )

            vx, vy = int(landmarks[vtx].x * w), int(landmarks[vtx].y * h)
            cv2.putText(
                out_bgr,
                f"{ang:.0f}",
                (vx + 6, vy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_hands(out_bgr: np.ndarray, hands, w: int, h: int) -> None:
        if not hands:
            return

        for hand in hands:
            for c in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                i, j = int(c.start), int(c.end)
                a, b = hand[i], hand[j]
                ax, ay = int(a.x * w), int(a.y * h)
                bx, by = int(b.x * w), int(b.y * h)
                cv2.line(out_bgr, (ax, ay), (bx, by), (200, 255, 0), 2, cv2.LINE_AA)
            for p in hand:
                x, y = int(p.x * w), int(p.y * h)
                cv2.circle(out_bgr, (x, y), 2, (0, 255, 255), -1, cv2.LINE_AA)
