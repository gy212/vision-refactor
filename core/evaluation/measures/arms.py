# -*- coding: utf-8 -*-
"""手臂测量：逐帧关节角、线段长度、口部中点与关节到给定点的距离（规则目录 ``sanda_v5`` 用）。

输入 ``(T, 33, C)`` 宽高比修正坐标；数值口径逐位复现迁移前的写法。
"""
from __future__ import annotations

import numpy as np

from core.foundation.geometry import angle_deg
from core.pose.landmarks import MOUTH_L, MOUTH_R


def series_angles(lm: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    """逐帧 ∠abc（度），取点不转 dtype。"""
    return np.array([angle_deg(p[a, :2], p[b, :2], p[c, :2]) for p in lm])


def segment_lengths(lm: np.ndarray, a: int, b: int) -> np.ndarray:
    """逐帧 a、b 两点的距离。"""
    return np.linalg.norm(lm[:, a, :2] - lm[:, b, :2], axis=1)


def series_mouth(lm: np.ndarray) -> np.ndarray:
    return (lm[:, MOUTH_L, :2] + lm[:, MOUTH_R, :2]) / 2


def distance_to(lm: np.ndarray, joint: int, points: np.ndarray) -> np.ndarray:
    """逐帧关节到给定点序列的距离。"""
    return np.linalg.norm(lm[:, joint, :2] - points, axis=1)
