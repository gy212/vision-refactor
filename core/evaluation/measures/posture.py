"""姿势检查共用的关节角、高度差与肘贴身距离测量。"""

from __future__ import annotations

import numpy as np

from core.evaluation.measures.arms import distance_to, segment_lengths, series_angles
from core.evaluation.measures.motion import relative_position


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
    lm: np.ndarray, mask: np.ndarray, shoulder: int, elbow: int, hip: int, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """肘距同侧肩髋中点的距离，使用同一尺度归一。"""
    center = (lm[:, shoulder, :2] + lm[:, hip, :2]) / 2
    values = distance_to(lm, elbow, center) / np.maximum(scale, 1e-4)
    valid = mask[:, [shoulder, elbow, hip]].all(axis=1) & (scale > 1e-4)
    return values, valid
