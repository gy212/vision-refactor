# -*- coding: utf-8 -*-
"""脚部测量：两脚方向夹角（跟→脚尖）、膝–踝–脚尖角（脚背绷直）；数值口径逐位复现迁移前的规则。"""
from __future__ import annotations

import numpy as np

from core.foundation.geometry import landmark_xy
from core.pose.landmarks import L_FOOT_INDEX, L_HEEL, R_FOOT_INDEX, R_HEEL


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
