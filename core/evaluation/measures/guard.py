"""公共护手几何测量：掉肩、护颚、抬肘高度、正面开肘夹角与肘退至身体后方。"""

from __future__ import annotations

import numpy as np

from core.evaluation.measures.arms import distance_to, segment_lengths, series_mouth
from core.evaluation.measures.posture import joint_angle, relative_height
from core.evaluation.measures.torso import body_forward
from core.pose.landmarks import (
    L_HIP,
    L_SHOULDER,
    MOUTH_L,
    MOUTH_R,
    NOSE,
    R_HIP,
    R_SHOULDER,
)


def measure_guard_shoulder_drop(
    lm: np.ndarray,
    mask: np.ndarray,
    torso: np.ndarray,
    guard_wrist: int,
    guard_shoulder: int,
    max_drop: float,
    *,
    label: str = "护手腕相对肩下垂",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """公共护手掉肩测量：护手腕低于同侧肩超过 max_drop 倍躯干长判掉落。"""
    drop, valid = relative_height(lm, mask, guard_wrist, guard_shoulder, torso)
    return drop > max_drop, valid, {label: (drop, "躯干长度")}


def measure_guard_jaw_distance(
    lm: np.ndarray,
    mask: np.ndarray,
    torso: np.ndarray,
    guard_wrist: int,
    max_dist: float = 0.34,
    *,
    label: str = "护手腕到下颚距离",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """公共护手护颚测量：手腕距嘴中点超过 max_dist 倍躯干长判未护住下颚。"""
    mouth = series_mouth(lm)
    dist = distance_to(lm, guard_wrist, mouth) / np.maximum(torso, 1e-4)
    valid = (
        mask[:, [guard_wrist, MOUTH_L, MOUTH_R]].all(axis=1) & (torso > 1e-4) & np.isfinite(dist)
    )
    return dist > max_dist, valid, {label: (dist, "躯干长度")}


def measure_guard_elbow_height(
    lm: np.ndarray,
    mask: np.ndarray,
    torso: np.ndarray,
    guard_shoulder: int,
    guard_elbow: int,
    max_raise: float,
    *,
    label: str = "肘高于肩",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """公共护手抬肘高度测量：肘相对肩高度超过 max_raise 倍躯干长判抬肘。"""
    height, valid = relative_height(lm, mask, guard_shoulder, guard_elbow, torso)
    return height > max_raise, valid, {label: (height, "躯干长度")}


def measure_guard_elbow_flare(
    lm: np.ndarray,
    mask: np.ndarray,
    torso: np.ndarray,
    guard_elbow: int,
    guard_shoulder: int,
    guard_hip: int,
    max_flare: float = 45.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """公共护手正面开肘测量：大臂与躯干夹角大于 max_flare 度判开肘亮肋。"""
    angles, valid = joint_angle(lm, mask, (guard_elbow, guard_shoulder, guard_hip), torso)
    return angles > max_flare, valid, {"护手大臂与躯干夹角": (angles, "度")}


def measure_guard_elbow_behind_side(
    lm: np.ndarray,
    mask: np.ndarray,
    torso: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    base_torso: np.ndarray,
    guard_shoulder: int,
    guard_elbow: int,
    guard_hip: int,
    max_back: float,
    min_valid: int,
    issues: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """侧面护手后拉测量：量肘到肩髋线后方的距离，随身体倾斜修正前后方向。"""
    base_valid = baseline_mask[:, [NOSE, guard_hip]].all(axis=1) & (base_torso > 1e-4)
    if base_valid.sum() < min_valid:
        if issues is not None:
            issues.append(
                f"起始护位方向参照可用帧不足（{base_valid.sum()}/{len(baseline)}帧，至少{min_valid}帧）"
            )
        return np.zeros(len(lm), bool), np.zeros(len(lm), bool), {}
    direction = np.median(
        (baseline[base_valid, NOSE, 0] - baseline[base_valid, guard_hip, 0])
        / base_torso[base_valid]
    )
    if abs(direction) < 0.1:
        if issues is not None:
            issues.append("起始护位的鼻与髋部位置不足以确定面朝方向")
        return np.zeros(len(lm), bool), np.zeros(len(lm), bool), {}
    body = lm[:, guard_hip, :2] - lm[:, guard_shoulder, :2]
    body_length = segment_lengths(lm, guard_shoulder, guard_hip)
    forward = np.stack((body[:, 1], -body[:, 0]), axis=1) * np.sign(direction)
    forward /= np.maximum(body_length[:, None], 1e-4)
    values = np.sum((lm[:, guard_shoulder, :2] - lm[:, guard_elbow, :2]) * forward, axis=1)
    values /= np.maximum(torso, 1e-4)
    valid = (
        mask[:, [guard_shoulder, guard_elbow, guard_hip]].all(axis=1)
        & (torso > 1e-4)
        & (body_length > 1e-4)
    )
    return values > max_back, valid, {"护手肘退到身体后方": (values, "躯干长度")}


def measure_guard_elbow_behind_front(
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    guard_shoulder: int,
    guard_elbow: int,
    required: tuple[int, ...],
    max_back: float,
    min_valid: int,
    issues: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """正面护手后拉测量：看清远侧护手肘后，按三维前向投影试算肘退至身体后方距离。"""
    base_direction, base_length, _ = body_forward(baseline)
    base_valid = baseline_mask[:, [NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]].all(axis=1) & (
        base_length > 1e-4
    )
    if base_valid.sum() < min_valid:
        if issues is not None:
            issues.append(
                f"起始护位方向参照可用帧不足（{base_valid.sum()}/{len(baseline)}帧，至少{min_valid}帧）"
            )
        return np.zeros(len(lm), bool), np.zeros(len(lm), bool), {}
    hips = (baseline[:, L_HIP, :3] + baseline[:, R_HIP, :3]) / 2
    facing = np.sign(
        np.median(np.sum((baseline[:, NOSE, :3] - hips) * base_direction, axis=1)[base_valid])
    )
    if facing == 0 and issues is not None:
        issues.append("起始护位的鼻与髋部位置不足以确定面朝方向")
    direction, length, torso = body_forward(lm)
    values = np.sum((lm[:, guard_shoulder, :3] - lm[:, guard_elbow, :3]) * direction, axis=1)
    values *= facing / np.maximum(torso, 1e-4)
    valid = mask[:, required].all(axis=1) & (length > 1e-4) & (torso > 1e-4) & (facing != 0)
    return values > max_back, valid, {"估计护手肘退到身体后方": (values, "身体长度")}
