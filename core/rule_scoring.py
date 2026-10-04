from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np

from .pose_features import (
    DEFAULT_VALID_CONF_THR,
    MEDIAPIPE_VALIDITY_POLICY,
    derive_valid_mask,
    landmark_capabilities,
    landmark_names,
    landmarks_missing_for_capabilities,
)
from .paths import models_dir
from .vision_pipeline import MediaPipePipeline, PipelineConfig


# ---------------------------------------------------------------------------
# 结构化三态（YOLO 迁移 S4 / Issue #11）
# ---------------------------------------------------------------------------
# 此前规则三态靠 detail 中文字符串拼（（未评估）/（合格）），下游无法稳定判断。
# 本期补结构化字段：state ∈ {evaluated, skipped}、skip_reason ∈
# {missing_landmarks, low_confidence, insufficient_valid_frames}。detail 中文保留
# 供 UI，下游判断只读 state / skip_reason。
RuleState = Literal["evaluated", "skipped"]
SkipReason = Literal["missing_landmarks", "low_confidence", "insufficient_valid_frames"]

RULE_STATE_EVALUATED: RuleState = "evaluated"
RULE_STATE_SKIPPED: RuleState = "skipped"

SKIP_MISSING_LANDMARKS: SkipReason = "missing_landmarks"
SKIP_LOW_CONFIDENCE: SkipReason = "low_confidence"
SKIP_INSUFFICIENT_VALID_FRAMES: SkipReason = "insufficient_valid_frames"


# 规则评分总体思路：
# 1) 用原始 Pose33 关键点（归一化坐标）计算几何量，如关节角、相对距离等。
# 2) 对每条规则统计“违规帧占比”（只统计可见度足够的帧）。
# 3) 当违规占比 >= 触发阈值时，扣一次该条分数并记录明细。

# BlazePose 33 关键点索引（部分）
NOSE = 0
MOUTH_L = 9
MOUTH_R = 10
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_HIP = 23
R_HIP = 24
L_KNEE = 25
R_KNEE = 26
L_ANKLE = 27
R_ANKLE = 28
L_HEEL = 29
R_HEEL = 30
L_FOOT_INDEX = 31
R_FOOT_INDEX = 32


@dataclass(frozen=True)
class Rule:
    rule_id: str
    name: str
    penalty: int
    view: str  # "front" | "side" | "any"
    action: str  # "stance" | "punch" | "both"
    trigger_ratio: float
    check_fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, str]]
    # YOLO 迁移 S4 / Issue #11：声明该规则依赖的关键点索引（BlazePose33）。
    # required_landmarks / required_capabilities 由 source_indices 派生（见 _build_rules）。
    required_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class RuleViolation:
    rule_id: str
    name: str
    penalty: int
    violation_ratio: float
    valid_frames: int
    total_frames: int
    detail: str
    # YOLO 迁移 S4 / Issue #11：结构化三态。detail（中文）保留供 UI；
    # 下游稳定判断只读 state / skip_reason，不再解析中文。
    state: RuleState = RULE_STATE_EVALUATED
    skip_reason: SkipReason | None = None
    # 该规则依赖的关键点名 + 能力分组（声明）；missing_landmarks 为运行时缺失的点名。
    required_landmarks: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    missing_landmarks: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuleScore:
    score: int
    total_deduction: int
    violations: tuple[RuleViolation, ...]


@dataclass(frozen=True)
class Pose33RawSeries:
    landmarks: np.ndarray
    meta: dict


def _lm_xy(lm: np.ndarray, idx: int) -> np.ndarray:
    return lm[idx, :2].astype(np.float32)


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """计算 ABC 夹角（度），适用于 2D 点。"""
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cosang = float(np.dot(ba, bc) / denom)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _torso_len(lm: np.ndarray) -> float:
    sh = 0.5 * (_lm_xy(lm, L_SHOULDER) + _lm_xy(lm, R_SHOULDER))
    hip = 0.5 * (_lm_xy(lm, L_HIP) + _lm_xy(lm, R_HIP))
    return float(np.linalg.norm(sh - hip) + 1e-9)


def _valid_frame(mask_row: np.ndarray, idxs: tuple[int, ...]) -> bool:
    """该帧给定关键点是否全部可用。

    读取由 ``derive_valid_mask`` 产出的单帧 ``(33,)`` bool 切片，不再现场比较
    ``lm[idx, 3] >= thr``（阈值只存在于 ``derive_valid_mask``，YOLO 迁移 Issue #5）。
    """
    return all(bool(mask_row[i]) for i in idxs)


def _valid_frames(valid_mask: np.ndarray, idxs: tuple[int, ...]) -> np.ndarray:
    """Vectorized equivalent of applying ``_valid_frame`` to every row."""
    return np.asarray(valid_mask[:, idxs], dtype=bool).all(axis=1)


def _raw_meta(
    *,
    video_path: Path,
    pose_variant: str,
    fps: float,
    frame_count: int,
    width: int,
    height: int,
    start_frame: int,
    landmarks: np.ndarray,
) -> dict:
    valid_mask = derive_valid_mask(landmarks, DEFAULT_VALID_CONF_THR)
    return {
        "video": str(video_path),
        "fps": float(fps),
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
        "pose_variant": str(pose_variant),
        "start_frame": int(start_frame),
        "end_frame": int(start_frame + landmarks.shape[0] - 1),
        "landmark_layout": "pose33_normalized_xyzw(visibility)",
        # valid_mask 契约（YOLO 迁移 Issue #5）：MediaPipe 侧 0.5 视为已标定。
        "validity_policy": MEDIAPIPE_VALIDITY_POLICY,
        "valid_conf_thr": float(DEFAULT_VALID_CONF_THR),
        "valid_mask": valid_mask,
    }


def _extract_pose_raw_prefix(
    video_path: Path,
    *,
    pose_variant: str,
    end_frame: int | None = None,
) -> Pose33RawSeries:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    end_i = int(end_frame) if end_frame is not None else (n_frames - 1 if n_frames > 0 else 10**9)
    end_i = max(0, end_i)

    models_dir_path = models_dir()
    pipe = MediaPipePipeline(
        models_dir=models_dir_path,
        cfg=PipelineConfig(pose_variant=pose_variant, running_mode="video", enable_hands=False),
    )

    out: list[np.ndarray] = []
    last: np.ndarray | None = None

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(i * 1000.0 / fps)
        pose_landmarks, _hands = pipe.infer(frame, timestamp_ms=ts)
        if pose_landmarks is None:
            arr = last.copy() if last is not None else np.zeros((33, 4), dtype=np.float32)
        else:
            arr = np.zeros((33, 4), dtype=np.float32)
            for j in range(min(33, len(pose_landmarks))):
                lm = pose_landmarks[j]
                arr[j, 0] = float(getattr(lm, "x", 0.0))
                arr[j, 1] = float(getattr(lm, "y", 0.0))
                arr[j, 2] = float(getattr(lm, "z", 0.0))
                arr[j, 3] = float(getattr(lm, "visibility", 0.0))
            last = arr

        out.append(arr)
        if i >= end_i:
            break
        i += 1

    cap.release()
    if not out:
        raise RuntimeError(f"未能从视频提取姿态：{video_path}")

    landmarks = np.stack(out, axis=0).astype(np.float32, copy=False)
    meta = _raw_meta(
        video_path=video_path,
        pose_variant=pose_variant,
        fps=fps,
        frame_count=n_frames,
        width=w,
        height=h,
        start_frame=0,
        landmarks=landmarks,
    )
    return Pose33RawSeries(landmarks=landmarks, meta=meta)


def slice_pose_raw_series(
    series: Pose33RawSeries,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Return an ``extract_pose_raw``-compatible slice from a pre-extracted series."""
    base = np.asarray(series.landmarks, dtype=np.float32)
    if base.ndim != 3 or base.shape[1:] != (33, 4):
        raise ValueError(f"Pose33RawSeries.landmarks 形状必须为 (T,33,4)，实际 {base.shape}")
    if base.shape[0] == 0:
        raise RuntimeError(f"未能从视频提取姿态：{series.meta.get('video', '')}")

    start_i = int(start_frame) if start_frame is not None else 0
    end_i = int(end_frame) if end_frame is not None else int(base.shape[0] - 1)
    start_i = max(0, start_i)
    end_i = max(start_i, end_i)
    end_i = min(end_i, int(base.shape[0] - 1))
    if start_i >= int(base.shape[0]):
        raise RuntimeError(f"未能从视频提取姿态：{series.meta.get('video', '')}")

    landmarks = base[start_i : end_i + 1].astype(np.float32, copy=False)
    meta = dict(series.meta)
    meta["start_frame"] = int(start_i)
    meta["end_frame"] = int(start_i + landmarks.shape[0] - 1)
    meta["valid_mask"] = derive_valid_mask(landmarks, float(meta.get("valid_conf_thr") or DEFAULT_VALID_CONF_THR))
    return landmarks, meta


def extract_pose_raw_series(
    video_path: Path,
    *,
    pose_variant: str,
) -> Pose33RawSeries:
    """Extract full-video raw Pose33 once for downstream read-only slicing."""
    return _extract_pose_raw_prefix(Path(video_path), pose_variant=pose_variant, end_frame=None)


def extract_pose_raw(
    video_path: Path,
    *,
    pose_variant: str,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    提取原始 Pose33 关键点（标准化坐标）。
    返回 (landmarks[T,33,4], meta)。
    """
    start_i = int(start_frame) if start_frame is not None else 0
    if end_frame is None:
        n_frames_probe = cv2.VideoCapture(str(video_path))
        try:
            n_frames = int(n_frames_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if n_frames_probe.isOpened() else 0
        finally:
            n_frames_probe.release()
        end_i = n_frames - 1 if n_frames > 0 else 10**9
    else:
        end_i = int(end_frame)
    start_i = max(0, start_i)
    end_i = max(start_i, end_i)
    series = _extract_pose_raw_prefix(Path(video_path), pose_variant=pose_variant, end_frame=end_i)
    return slice_pose_raw_series(series, start_frame=start_i, end_frame=end_i)


def _rule_elbow_front_arm_range(landmarks: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """前手大小臂夹角 90-135 度（允许左右任一手满足）。"""
    valid = _valid_frames(valid_mask, (L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST))
    viol = np.zeros_like(valid)
    for i, lm in enumerate(landmarks):
        if not bool(valid[i]):
            continue
        ang_l = _angle_deg(_lm_xy(lm, L_SHOULDER), _lm_xy(lm, L_ELBOW), _lm_xy(lm, L_WRIST))
        ang_r = _angle_deg(_lm_xy(lm, R_SHOULDER), _lm_xy(lm, R_ELBOW), _lm_xy(lm, R_WRIST))
        ok = (90.0 <= ang_l <= 135.0) or (90.0 <= ang_r <= 135.0)
        viol[i] = not ok
    return viol, valid, "前手肘角需在90-135度"


def _rule_fist_height_near_nose(landmarks: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """拳峰高度与鼻尖同高（允许误差，使用归一化距离近似）。"""
    valid = _valid_frames(valid_mask, (NOSE, L_WRIST, R_WRIST))
    viol = np.zeros_like(valid)
    for i, lm in enumerate(landmarks):
        if not bool(valid[i]):
            continue
        nose_y = _lm_xy(lm, NOSE)[1]
        lw_y = _lm_xy(lm, L_WRIST)[1]
        rw_y = _lm_xy(lm, R_WRIST)[1]
        torso = _torso_len(lm)
        tol = 0.08 * torso  # 近似 3cm 宽容（按画面比例）
        ok = (abs(lw_y - nose_y) <= tol) or (abs(rw_y - nose_y) <= tol)
        viol[i] = not ok
    return viol, valid, "拳峰高度需与鼻尖接近"


def _rule_back_arm_close(landmarks: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """后手贴近肋骨与下颌（近似：手腕靠近嘴部，肘部靠近躯干）。"""
    valid = _valid_frames(valid_mask, (MOUTH_L, MOUTH_R, L_ELBOW, L_WRIST, R_ELBOW, R_WRIST, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP))
    viol = np.zeros_like(valid)
    for i, lm in enumerate(landmarks):
        if not bool(valid[i]):
            continue
        mouth = 0.5 * (_lm_xy(lm, MOUTH_L) + _lm_xy(lm, MOUTH_R))
        torso = _torso_len(lm)
        center = 0.5 * (_lm_xy(lm, L_SHOULDER) + _lm_xy(lm, R_SHOULDER) + _lm_xy(lm, L_HIP) + _lm_xy(lm, R_HIP))
        l_ok = (np.linalg.norm(_lm_xy(lm, L_WRIST) - mouth) <= 0.45 * torso) and (
            np.linalg.norm(_lm_xy(lm, L_ELBOW) - center) <= 0.5 * torso
        )
        r_ok = (np.linalg.norm(_lm_xy(lm, R_WRIST) - mouth) <= 0.45 * torso) and (
            np.linalg.norm(_lm_xy(lm, R_ELBOW) - center) <= 0.5 * torso
        )
        viol[i] = not (l_ok or r_ok)
    return viol, valid, "后手应贴近下颌并靠近躯干"


def _rule_knee_slight_bend(landmarks: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """双膝微曲（膝角过直视为违规）。"""
    valid = _valid_frames(valid_mask, (L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE))
    viol = np.zeros_like(valid)
    for i, lm in enumerate(landmarks):
        if not bool(valid[i]):
            continue
        ang_l = _angle_deg(_lm_xy(lm, L_HIP), _lm_xy(lm, L_KNEE), _lm_xy(lm, L_ANKLE))
        ang_r = _angle_deg(_lm_xy(lm, R_HIP), _lm_xy(lm, R_KNEE), _lm_xy(lm, R_ANKLE))
        viol[i] = (ang_l > 170.0) or (ang_r > 170.0)
    return viol, valid, "膝关节应微曲（过直扣分）"


def _rule_stance_width(landmarks: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """两脚间距≈肩宽（允许一定比例波动）。"""
    valid = _valid_frames(valid_mask, (L_ANKLE, R_ANKLE, L_SHOULDER, R_SHOULDER))
    viol = np.zeros_like(valid)
    for i, lm in enumerate(landmarks):
        if not bool(valid[i]):
            continue
        foot_w = float(np.linalg.norm(_lm_xy(lm, L_ANKLE) - _lm_xy(lm, R_ANKLE)))
        shoulder_w = float(np.linalg.norm(_lm_xy(lm, L_SHOULDER) - _lm_xy(lm, R_SHOULDER)))
        if shoulder_w < 1e-6:
            viol[i] = True
            continue
        ratio = foot_w / shoulder_w
        viol[i] = not (0.8 <= ratio <= 1.2)
    return viol, valid, "站距应接近肩宽"


def _rule_feet_parallel(landmarks: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """两脚基本平行（用脚尖方向夹角近似）。"""
    valid = _valid_frames(valid_mask, (L_HEEL, L_FOOT_INDEX, R_HEEL, R_FOOT_INDEX))
    viol = np.zeros_like(valid)
    for i, lm in enumerate(landmarks):
        if not bool(valid[i]):
            continue
        v_l = _lm_xy(lm, L_FOOT_INDEX) - _lm_xy(lm, L_HEEL)
        v_r = _lm_xy(lm, R_FOOT_INDEX) - _lm_xy(lm, R_HEEL)
        if np.linalg.norm(v_l) < 1e-6 or np.linalg.norm(v_r) < 1e-6:
            viol[i] = True
            continue
        cosang = float(np.dot(v_l, v_r) / (np.linalg.norm(v_l) * np.linalg.norm(v_r)))
        cosang = float(np.clip(cosang, -1.0, 1.0))
        ang = float(np.degrees(np.arccos(cosang)))
        viol[i] = ang > 35.0
    return viol, valid, "双脚方向应基本平行"


def _rule_punch_elbow_straight(landmarks: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """直拳出拳时肘应接近伸直（只在“出拳帧”评估）。"""
    visible = _valid_frames(valid_mask, (L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST, L_HIP, R_HIP))
    valid = np.zeros((landmarks.shape[0],), dtype=bool)
    viol = np.zeros_like(valid)
    for i, lm in enumerate(landmarks):
        if not bool(visible[i]):
            continue
        torso = _torso_len(lm)
        ext_l = np.linalg.norm(_lm_xy(lm, L_WRIST) - _lm_xy(lm, L_SHOULDER)) / torso
        ext_r = np.linalg.norm(_lm_xy(lm, R_WRIST) - _lm_xy(lm, R_SHOULDER)) / torso
        # 仅在出拳伸展明显时进行判定
        if max(ext_l, ext_r) < 1.05:
            continue
        valid[i] = True
        if ext_l >= ext_r:
            ang = _angle_deg(_lm_xy(lm, L_SHOULDER), _lm_xy(lm, L_ELBOW), _lm_xy(lm, L_WRIST))
        else:
            ang = _angle_deg(_lm_xy(lm, R_SHOULDER), _lm_xy(lm, R_ELBOW), _lm_xy(lm, R_WRIST))
        viol[i] = ang < 160.0
    return viol, valid, "出拳时肘应接近伸直"


def _rule_guard_hand(landmarks: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """直拳时另一手应保持护颌（近似：手腕靠近嘴部）。"""
    visible = _valid_frames(valid_mask, (MOUTH_L, MOUTH_R, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP))
    valid = np.zeros((landmarks.shape[0],), dtype=bool)
    viol = np.zeros_like(valid)
    for i, lm in enumerate(landmarks):
        if not bool(visible[i]):
            continue
        torso = _torso_len(lm)
        ext_l = np.linalg.norm(_lm_xy(lm, L_WRIST) - _lm_xy(lm, L_SHOULDER)) / torso
        ext_r = np.linalg.norm(_lm_xy(lm, R_WRIST) - _lm_xy(lm, R_SHOULDER)) / torso
        if max(ext_l, ext_r) < 1.05:
            continue
        valid[i] = True
        mouth = 0.5 * (_lm_xy(lm, MOUTH_L) + _lm_xy(lm, MOUTH_R))
        if ext_l >= ext_r:
            guard_dist = float(np.linalg.norm(_lm_xy(lm, R_WRIST) - mouth))
        else:
            guard_dist = float(np.linalg.norm(_lm_xy(lm, L_WRIST) - mouth))
        viol[i] = guard_dist > 0.6 * torso
    return viol, valid, "出拳时另一手应靠近下颌护防"


def _build_rules() -> list[Rule]:
    # 触发比例：当违规帧占比 >= trigger_ratio，扣该条分数一次。
    # required_indices 必须与各 check_fn 内 _valid_frame(...) 的关键点集合一致
    # （YOLO 迁移 S4 / Issue #11：用于声明 required_landmarks/capabilities 与缺点判定）。
    return [
        Rule("stance_elbow", "前手肘角", 2, "front", "stance", 0.3, _rule_elbow_front_arm_range,
             required_indices=(L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST)),
        Rule("stance_fist_height", "拳峰高度", 2, "front", "stance", 0.3, _rule_fist_height_near_nose,
             required_indices=(NOSE, L_WRIST, R_WRIST)),
        Rule("stance_back_arm", "后手贴近", 2, "front", "stance", 0.3, _rule_back_arm_close,
             required_indices=(MOUTH_L, MOUTH_R, L_ELBOW, L_WRIST, R_ELBOW, R_WRIST, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)),
        Rule("stance_knee", "双膝微曲", 2, "any", "stance", 0.3, _rule_knee_slight_bend,
             required_indices=(L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE)),
        Rule("stance_width", "站距合理", 2, "front", "stance", 0.3, _rule_stance_width,
             required_indices=(L_ANKLE, R_ANKLE, L_SHOULDER, R_SHOULDER)),
        Rule("stance_feet", "脚尖方向", 2, "front", "stance", 0.3, _rule_feet_parallel,
             required_indices=(L_HEEL, L_FOOT_INDEX, R_HEEL, R_FOOT_INDEX)),
        Rule("punch_elbow", "出拳伸直", 15, "any", "punch", 0.3, _rule_punch_elbow_straight,
             required_indices=(L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST, L_HIP, R_HIP)),
        Rule("punch_guard", "护手位置", 15, "any", "punch", 0.3, _rule_guard_hand,
             required_indices=(MOUTH_L, MOUTH_R, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP)),
    ]


_RULES: tuple[Rule, ...] = tuple(_build_rules())


def score_rules(
    landmarks: np.ndarray,
    *,
    view: str,
    action_scope: str,
    min_valid: int = 5,
    valid_mask: np.ndarray | None = None,
    supported_capabilities=None,
) -> RuleScore:
    """
    对单段视频关键点进行规则评分。
    - view: "front" | "side"，用于过滤只适用于某视角的规则
    - action_scope: "stance" | "punch" | "both"
    - min_valid: 单条规则至少需要的有效帧数（不足则标记“未评估”）
    - valid_mask: ``(T,33)`` 有效性掩码（YOLO 迁移 Issue #5）。``None`` 时由
      ``derive_valid_mask`` 现场推导（等价旧式 ``visibility >= 0.5``），保证旧调用方
      （含 ``compare_video_to_dual_templates`` 内部 ``extract_pose_raw`` 调用）行为不变。
    - supported_capabilities: 后端结构性支持的能力集合（YOLO 迁移 S4 / Issue #11）。
      ``None`` 表示「全部支持」（MediaPipe full，默认），此时缺点判定不生效，行为与
      旧版逐位一致。传入受限集合（如 ``COCO17_SUPPORTED_CAPABILITIES``）时，依赖
      不支持能力的规则直接 ``state=skipped`` + ``skip_reason=missing_landmarks``，
      不当合格/不合格。本期 #10 结论为「仅预览」，YOLO partial eval 未启用，该形参
      仅供未来 YOLO 路径与测试使用。

    结构化三态（Issue #11）：每条 ``RuleViolation`` 都带 ``state`` /（``skip_reason``）/
    ``required_landmarks`` / ``required_capabilities`` / ``missing_landmarks``。
    ``detail`` 中文保留供 UI，下游稳定判断只读结构化字段。
    """
    view = (view or "front").lower()
    action_scope = (action_scope or "both").lower()

    if valid_mask is None:
        valid_mask = derive_valid_mask(landmarks, DEFAULT_VALID_CONF_THR)

    deductions = 0
    violations: list[RuleViolation] = []
    total_frames = int(landmarks.shape[0])
    for rule in _RULES:
        if rule.view not in ("any", view):
            continue
        if action_scope != "both" and rule.action != action_scope:
            continue

        req_landmarks = landmark_names(rule.required_indices)
        req_caps = landmark_capabilities(rule.required_indices)

        # 缺点判定（YOLO 迁移 S4 / Issue #11）：后端结构性缺失所需关键点时，直接
        # 跳过评估、标 skipped/missing_landmarks，不混入合格/不合格。
        backend_missing = landmarks_missing_for_capabilities(
            rule.required_indices, supported_capabilities
        )
        if backend_missing:
            violations.append(
                RuleViolation(
                    rule_id=rule.rule_id,
                    name=rule.name,
                    penalty=0,
                    violation_ratio=0.0,
                    valid_frames=0,
                    total_frames=int(total_frames),
                    detail=f"{rule.name}（后端缺少关键点，未评估）",
                    state=RULE_STATE_SKIPPED,
                    skip_reason=SKIP_MISSING_LANDMARKS,
                    required_landmarks=req_landmarks,
                    required_capabilities=req_caps,
                    missing_landmarks=tuple(backend_missing),
                )
            )
            continue

        viol, valid, detail = rule.check_fn(landmarks, valid_mask)
        valid_cnt = int(valid.sum())
        if valid_cnt < min_valid:
            # 有效帧不足：valid_cnt==0 视为关键点置信度始终不足（low_confidence），
            # 0<valid_cnt<min_valid 视为有效帧数不足（insufficient_valid_frames）。
            # detail 中文保持与旧版逐位一致（golden 不漂移），仅结构化字段区分。
            skip_reason = (
                SKIP_LOW_CONFIDENCE if valid_cnt == 0 else SKIP_INSUFFICIENT_VALID_FRAMES
            )
            violations.append(
                RuleViolation(
                    rule_id=rule.rule_id,
                    name=rule.name,
                    penalty=0,
                    violation_ratio=0.0,
                    valid_frames=int(valid_cnt),
                    total_frames=int(total_frames),
                    detail=f"{detail}（有效帧不足，未评估）",
                    state=RULE_STATE_SKIPPED,
                    skip_reason=skip_reason,
                    required_landmarks=req_landmarks,
                    required_capabilities=req_caps,
                )
            )
            continue
        ratio = float(viol[valid].sum() / max(1, valid_cnt))
        if ratio >= float(rule.trigger_ratio):
            deductions += int(rule.penalty)
            violations.append(
                RuleViolation(
                    rule_id=rule.rule_id,
                    name=rule.name,
                    penalty=int(rule.penalty),
                    violation_ratio=ratio,
                    valid_frames=int(valid_cnt),
                    total_frames=int(total_frames),
                    detail=detail,
                    state=RULE_STATE_EVALUATED,
                    skip_reason=None,
                    required_landmarks=req_landmarks,
                    required_capabilities=req_caps,
                )
            )
        else:
            violations.append(
                RuleViolation(
                    rule_id=rule.rule_id,
                    name=rule.name,
                    penalty=0,
                    violation_ratio=ratio,
                    valid_frames=int(valid_cnt),
                    total_frames=int(total_frames),
                    detail=f"{detail}（合格）",
                    state=RULE_STATE_EVALUATED,
                    skip_reason=None,
                    required_landmarks=req_landmarks,
                    required_capabilities=req_caps,
                )
            )

    score = max(0, 100 - int(deductions))
    return RuleScore(score=score, total_deduction=int(deductions), violations=tuple(violations))
