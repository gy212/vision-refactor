# -*- coding: utf-8 -*-
"""腿部测量：膝角、髋轴转角；数值口径逐位复现迁移前的规则。"""
from __future__ import annotations

import math

import numpy as np

from core.foundation.geometry import angle_deg, landmark_xy
from core.pose.landmarks import L_ANKLE, L_HIP, L_KNEE, R_ANKLE, R_HIP, R_KNEE


def knee_angles(lm: np.ndarray) -> tuple[float, float]:
    """左、右膝角（髋–膝–踝，度）。"""
    ang_l = angle_deg(landmark_xy(lm, L_HIP), landmark_xy(lm, L_KNEE), landmark_xy(lm, L_ANKLE))
    ang_r = angle_deg(landmark_xy(lm, R_HIP), landmark_xy(lm, R_KNEE), landmark_xy(lm, R_ANKLE))
    return ang_l, ang_r


def hip_axis_rotation(lm: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """逐帧髋轴（左髋→右髋，x–z 平面）相对基线中位方向的转角绝对值（度，深度为估计值）。"""
    axis = lm[:, R_HIP, [0, 2]] - lm[:, L_HIP, [0, 2]]
    initial = baseline[:, R_HIP, [0, 2]] - baseline[:, L_HIP, [0, 2]]
    base_angle = math.atan2(float(np.median(initial[:, 1])), float(np.median(initial[:, 0])))
    delta = (np.arctan2(axis[:, 1], axis[:, 0]) - base_angle + np.pi) % (2 * np.pi) - np.pi
    return np.abs(np.degrees(delta))
