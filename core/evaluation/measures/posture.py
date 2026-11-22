"""通用人体姿势测量：关节角、高度差、肘贴身距离、低头、侧倾、前后倾与翻胯。"""

from __future__ import annotations

import numpy as np

from core.evaluation.measures.arms import distance_to, segment_lengths, series_angles
from core.evaluation.measures.motion import relative_position
from core.pose.landmarks import L_HIP, L_SHOULDER, NOSE, R_HIP, R_SHOULDER


def joint_angle(
    lm: np.ndarray, mask: np.ndarray, joints: tuple[int, int, int], scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """测量关节角并拒绝缺点或零长肢段。"""
    first, joint, last = joints
    values = series_angles(lm, first, joint, last)
    valid = mask[:, joints].all(axis=1) & (scale > 1e-4)
    valid &= segment_lengths(lm, first, joint) > 1e-4
    valid &= segment_lengths(lm, last, joint) > 1e-4
    return values, valid


def relative_height(
    lm: np.ndarray, mask: np.ndarray, moving: int, reference: int, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """纵向差除以身体尺度；正值表示被测点低于参照点。"""
    position, valid = relative_position(lm, mask, scale, reference, moving)
    return position[:, 1], valid


def elbow_body_distance(
    lm: np.ndarray,
    mask: np.ndarray,
    shoulder: int,
    elbow: int,
    hip: int,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """肘距同侧肩髋中点的距离，使用同一尺度归一。"""
    center = (lm[:, shoulder, :2] + lm[:, hip, :2]) / 2
    values = distance_to(lm, elbow, center) / np.maximum(scale, 1e-4)
    valid = mask[:, [shoulder, elbow, hip]].all(axis=1) & (scale > 1e-4)
    return values, valid


def measure_head_down(
    lm: np.ndarray,
    mask: np.ndarray,
    torso: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    base_torso: np.ndarray,
    max_drop: float = 0.030,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """低头检查：侧视鼻相对双肩中点垂向位置比基线低超过阈值判低头。"""
    base_ok = baseline_mask[:, [NOSE, L_SHOULDER, R_SHOULDER]].all(axis=1) & (base_torso > 1e-4)
    if not base_ok.any():
        empty = np.zeros(len(lm), dtype=bool)
        return empty, empty.copy(), {}
    base_mid = 0.5 * (baseline[base_ok, L_SHOULDER, 1] + baseline[base_ok, R_SHOULDER, 1])
    base_med = float(np.median((baseline[base_ok, NOSE, 1] - base_mid) / base_torso[base_ok]))
    cur_ok = mask[:, [NOSE, L_SHOULDER, R_SHOULDER]].all(axis=1) & (torso > 1e-4)
    cur_mid = 0.5 * (lm[:, L_SHOULDER, 1] + lm[:, R_SHOULDER, 1])
    drop = (lm[:, NOSE, 1] - cur_mid) / np.maximum(torso, 1e-4) - base_med
    return drop > max_drop, cur_ok, {"鼻相对肩中点下压增量": (drop, "躯干长度")}


def measure_body_lean(
    lm: np.ndarray,
    mask: np.ndarray,
    max_angle: float = 35.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """正面侧倾角度：双肩连线与水平线夹角超过 max_angle 度判上体侧倾。"""
    dy = np.abs(lm[:, L_SHOULDER, 1] - lm[:, R_SHOULDER, 1])
    dx = np.maximum(np.abs(lm[:, L_SHOULDER, 0] - lm[:, R_SHOULDER, 0]), 1e-6)
    angles = np.degrees(np.arctan2(dy, dx))
    valid = (
        mask[:, L_SHOULDER]
        & mask[:, R_SHOULDER]
        & (segment_lengths(lm, L_SHOULDER, R_SHOULDER) > 1e-4)
    )
    return angles > max_angle, valid, {"正面双肩倾斜角": (angles, "度")}


def measure_trunk_lean_sagittal(
    lm: np.ndarray,
    mask: np.ndarray,
    max_angle: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """侧面躯干前后倾角：肩中点到髋中点连线相对竖直线前倾或后仰超过 max_angle 度判违规。"""
    sh_mid = 0.5 * (lm[:, L_SHOULDER, :2] + lm[:, R_SHOULDER, :2])
    hp_mid = 0.5 * (lm[:, L_HIP, :2] + lm[:, R_HIP, :2])
    vec = sh_mid - hp_mid
    length = np.linalg.norm(vec, axis=1)
    dx = np.abs(vec[:, 0])
    dy = hp_mid[:, 1] - sh_mid[:, 1]
    angles = np.degrees(np.arctan2(dx, dy))
    valid = mask[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]].all(axis=1) & (length > 1e-4)
    return angles > max_angle, valid, {"侧面躯干相对竖直倾角": (angles, "度")}


def measure_hip_turn_width(
    lm: np.ndarray,
    mask: np.ndarray,
    torso: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    base_torso: np.ndarray,
    max_ratio: float = 0.80,
    *,
    label: str = "正面两髋水平宽比值",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """翻胯/转胯判定：相对基线双髋水平宽比值仍大于 max_ratio 判未充分转胯。"""
    base_ok = baseline_mask[:, L_HIP] & baseline_mask[:, R_HIP] & (base_torso > 1e-4)
    if not base_ok.any():
        empty = np.zeros(len(lm), dtype=bool)
        return empty, empty.copy(), {}
    base_w = np.abs(baseline[base_ok, R_HIP, 0] - baseline[base_ok, L_HIP, 0]) / base_torso[base_ok]
    base_med = max(float(np.median(base_w)), 1e-4)
    cur_ok = mask[:, L_HIP] & mask[:, R_HIP] & (torso > 1e-4)
    ratio = np.abs(lm[:, R_HIP, 0] - lm[:, L_HIP, 0]) / np.maximum(torso, 1e-4) / base_med
    return ratio > max_ratio, cur_ok, {label: (ratio, "倍基线")}
