# -*- coding: utf-8 -*-
"""脚部公共测量：两脚内外八、支撑脚碾脚外旋投影及基础脚向向量。"""

from __future__ import annotations

import math

import numpy as np

from core.evaluation.measures.arms import segment_lengths
from core.foundation.geometry import landmark_xy
from core.pose.landmarks import L_FOOT_INDEX, L_HEEL, R_FOOT_INDEX, R_HEEL

FEET = (L_HEEL, L_FOOT_INDEX, R_HEEL, R_FOOT_INDEX)


def foot_vectors(lm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """左、右脚的跟→脚尖向量。"""
    v_l = landmark_xy(lm, L_FOOT_INDEX) - landmark_xy(lm, L_HEEL)
    v_r = landmark_xy(lm, R_FOOT_INDEX) - landmark_xy(lm, R_HEEL)
    return v_l, v_r


def vector_lengths(v_l: np.ndarray, v_r: np.ndarray) -> tuple[float, float]:
    return np.linalg.norm(v_l), np.linalg.norm(v_r)


def vector_angle_deg(v_l: np.ndarray, v_r: np.ndarray) -> float:
    """两向量夹角（度）；调用方先排除零长向量。"""
    cosang = float(np.dot(v_l, v_r) / (np.linalg.norm(v_l) * np.linalg.norm(v_r)))
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def measure_toe_spread(lm: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """比较两脚尖与两脚跟的距离，沿用原来两脚方向相差 35° 的容许范围。"""
    toes = segment_lengths(lm, L_FOOT_INDEX, R_FOOT_INDEX)
    heels = segment_lengths(lm, L_HEEL, R_HEEL)
    left = segment_lengths(lm, L_HEEL, L_FOOT_INDEX)
    right = segment_lengths(lm, R_HEEL, R_FOOT_INDEX)
    valid = mask[:, FEET].all(axis=1) & (left > 1e-4) & (right > 1e-4)
    spread = (toes - heels) / np.maximum(left + right, 1e-4)
    limit = math.sin(math.radians(35 / 2))
    return np.abs(spread) > limit, valid, {"脚尖间距减脚跟间距": (spread, "两脚长度之和")}


def measure_pivot_heel(
    lm: np.ndarray,
    safe_mask: np.ndarray,
    torso: np.ndarray,
    sup_heel: int,
    sup_toe: int,
    facing: float | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """支撑脚外旋/碾脚：脚尖仍在脚跟攻击前方（未转到重合或负向）判未达成帧。"""
    dir_x = float(facing) if facing else 1.0
    valid = (
        safe_mask[:, [sup_heel, sup_toe]].all(axis=1)
        & (torso > 1e-4)
        & bool(facing is not None and facing != 0)
    )
    safe_t = np.where(torso > 1e-4, torso, 1.0)
    proj = (lm[:, sup_toe, 0] - lm[:, sup_heel, 0]) * dir_x / safe_t
    return proj > 0.0, valid, {"支撑脚尖相对脚跟前向投影": (proj, "躯干长度")}
