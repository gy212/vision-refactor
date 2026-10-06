# -*- coding: utf-8 -*-
"""MediaPipe 推理引擎的唯一构造入口：``EngineSpec``、``create_engine``、``PoseEngine`` 与缺模型提示（设计文档 §6.1）。

``PoseEngine`` 是原 ``MediaPipePipeline`` 的推理部分（``infer``、``next_timestamp_ms``、``close``）；
``annotate()`` 是"推理 + ``render.render_pose``"的薄组合，文件模式下内部帧序号自增的语义不变。
缺模型的提示文案只在本模块定义（``MISSING_MODEL_HINTS``）。
"""
from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.foundation.paths import models_dir as default_models_dir
from core.pose.mediapipe_runtime import mediapipe
from core.pose.models import MEDIAPIPE_MODELS
from core.pose.render import UNSET_ACTION_POSE, render_pose

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# 缺模型时给用户看的提示；各入口按场景取对应条目，不在别处另写文案。
MISSING_MODEL_HINTS = {
    "practice_variant": "缺少 MediaPipe {variant}，请在设置的模型管理中安装。",
    "student_lite": "缺少 Lite 模型，请先在设置→模型管理中安装。",
    "exam_heavy": "缺少 pose_landmarker_heavy.task，请在模型管理中安装。",
    "exam_occupancy_lite": "缺少 pose_landmarker_lite.task（占用检测需要）。",
    "postprocess_heavy": "缺少 pose_landmarker_heavy.task，请先在模型管理中安装 heavy 模型",
    "pose_path": "缺少 MediaPipe pose 模型：{path}",
}


def missing_model_message(key: str, **fields) -> str:
    return MISSING_MODEL_HINTS[key].format(**fields)


class ModelMissing(FileNotFoundError):
    """所需模型文件不存在。``hint`` 是给用户看的提示。"""

    def __init__(self, model_key: str, hint: str) -> None:
        super().__init__(hint)
        self.model_key = model_key
        self.hint = hint

    def __reduce__(self):
        # 默认 __reduce__ 只用 self.args（即 hint）重建，缺 model_key 这个必填参数会抛 TypeError。
        # 显式传回两个构造参数，使 pickle 往返保真。
        return (self.__class__, (self.model_key, self.hint))


@dataclass(frozen=True)
class EngineSpec:
    """推理引擎规格（原 ``PipelineConfig``，迁移期保留别名，字段名不变）。"""

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


PipelineConfig = EngineSpec


def pose_model_path(variant: str, directory: Path) -> Path:
    return Path(directory) / f"pose_landmarker_{variant}.task"


def _pose_model_url(variant: str) -> str:
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


def _base_options(model_path: Path, *, delegate: str):
    mp = mediapipe()
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


class PoseEngine:
    """MediaPipe Pose（可选 Hands）landmarker 的封装；只通过 ``create_engine`` 构造。"""

    def __init__(self, *, models_dir: Path, cfg: EngineSpec = EngineSpec()) -> None:
        self.cfg = cfg
        self.models_dir = models_dir
        self.requested_delegate = _normalize_delegate(cfg.delegate)
        self.active_delegate = self.requested_delegate
        self.delegate_fallback_reason: str | None = None

        pose_path = pose_model_path(cfg.pose_variant, self.models_dir)
        _ensure_file(_pose_model_url(cfg.pose_variant), pose_path)
        hand_path: Path | None = None
        if cfg.enable_hands:
            hand_path = self.models_dir / "hand_landmarker.task"
            _ensure_file(HAND_MODEL_URL, hand_path)

        mode = (cfg.running_mode or "video").lower()
        if mode not in ("video", "image"):
            raise ValueError(f"Unsupported running_mode: {cfg.running_mode!r} (use 'video' or 'image')")
        self.running_mode = mode
        mp = mediapipe()
        mp_mode = mp.tasks.vision.RunningMode.VIDEO if mode == "video" else mp.tasks.vision.RunningMode.IMAGE

        self.pose_landmarker = None
        self.hand_landmarker = None
        try:
            self._create_landmarkers(
                pose_path=pose_path, hand_path=hand_path, mp_mode=mp_mode, delegate=self.requested_delegate,
            )
        except Exception as exc:
            if self.requested_delegate != "gpu" or not bool(cfg.delegate_fallback_to_cpu):
                raise
            self.delegate_fallback_reason = str(exc)
            self.active_delegate = "cpu"
            try:
                self._create_landmarkers(pose_path=pose_path, hand_path=hand_path, mp_mode=mp_mode, delegate="cpu")
            except Exception as cpu_exc:
                raise RuntimeError(
                    "MediaPipe GPU delegate failed and CPU fallback also failed"
                ) from cpu_exc

        self._t0 = time.monotonic()
        self._frame_index = 0

    def _create_landmarkers(self, *, pose_path: Path, hand_path: Path | None, mp_mode: object,
                            delegate: str) -> None:
        cfg = self.cfg
        mp = mediapipe()
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

    def draw(self, frame_bgr: np.ndarray, pose_landmarks, hands, *,
             action_pose_landmarks=UNSET_ACTION_POSE) -> tuple[np.ndarray, list[str]]:
        """按本引擎的绘制选项调用 ``render.render_pose``（不含 infer、不动内部帧序号）。"""
        return render_pose(
            frame_bgr,
            pose_landmarks,
            hands,
            draw_face=self.cfg.draw_pose_face,
            draw_joint_angles_overlay=self.cfg.draw_joint_angles,
            action_pose_landmarks=action_pose_landmarks,
        )

    def infer(self, frame_bgr: np.ndarray, *, timestamp_ms: int | None = None):
        """Run landmarkers and return raw landmarks (pose_landmarks, hands_list)."""
        mp = mediapipe()
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


MediaPipePipeline = PoseEngine


def model_available(variant: str, directory: Path | None = None) -> bool:
    """姿态模型文件是否已就位（不下载）。"""
    return pose_model_path(variant, directory or default_models_dir()).is_file()


def create_engine(spec: EngineSpec, *, models_dir: Path | None = None, download: bool = True,
                  missing_hint: str = "practice_variant") -> PoseEngine:
    """构造推理引擎。``download=False`` 时模型不在就抛 ``ModelMissing``（提示取 ``missing_hint`` 条目）；
    默认缺模型时自动下载（与旧 ``MediaPipePipeline`` 一致）。"""
    directory = Path(models_dir) if models_dir is not None else default_models_dir()
    if not download and not model_available(spec.pose_variant, directory):
        raise ModelMissing(
            f"pose_{spec.pose_variant}",
            missing_model_message(missing_hint, variant=spec.pose_variant,
                                  path=pose_model_path(spec.pose_variant, directory)),
        )
    return PoseEngine(models_dir=directory, cfg=spec)
