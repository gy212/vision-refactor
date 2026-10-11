# -*- coding: utf-8 -*-
"""手臂测量：肘角、伸展比、腕–嘴距离、肘–躯干中心距离、拳峰与鼻尖的高度差、出拳与肩的高度差、护手位移。

单帧函数输入 ``(33, C)``，序列函数输入 ``(T, 33, C)``；正式规则用单帧（归一化坐标），练习用序列
（宽高比修正坐标）。数值口径逐位复现迁移前各规则里的写法（float32 取点、先算距离再比较）。
"""
from __future__ import annotations

import numpy as np

from core.foundation.geometry import angle_deg, landmark_xy, torso_length
from core.pose.landmarks import (
    L_ELBOW,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    MOUTH_L,
    MOUTH_R,
    NOSE,
    R_ELBOW,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
)

# --------------------------------------------------------------------------- #
# 单帧（正式规则）
# --------------------------------------------------------------------------- #


def elbow_angles(lm: np.ndarray) -> tuple[float, float]:
    """左、右肘角（肩–肘–腕，度）。"""
    ang_l = angle_deg(landmark_xy(lm, L_SHOULDER), landmark_xy(lm, L_ELBOW), landmark_xy(lm, L_WRIST))
    ang_r = angle_deg(landmark_xy(lm, R_SHOULDER), landmark_xy(lm, R_ELBOW), landmark_xy(lm, R_WRIST))
    return ang_l, ang_r


def elbow_angle(lm: np.ndarray, side: str) -> float:
    """``side`` 为 "L"/"R" 的肘角（度）。"""
    if side == "L":
        return angle_deg(landmark_xy(lm, L_SHOULDER), landmark_xy(lm, L_ELBOW), landmark_xy(lm, L_WRIST))
    return angle_deg(landmark_xy(lm, R_SHOULDER), landmark_xy(lm, R_ELBOW), landmark_xy(lm, R_WRIST))


def wrist_nose_height_gaps(lm: np.ndarray) -> tuple[float, float, float]:
    """左、右腕与鼻尖的纵向差（绝对值），以及中点法躯干长度。"""
    nose_y = landmark_xy(lm, NOSE)[1]
    lw_y = landmark_xy(lm, L_WRIST)[1]
    rw_y = landmark_xy(lm, R_WRIST)[1]
    return abs(lw_y - nose_y), abs(rw_y - nose_y), torso_length(lm)


def mouth_center(lm: np.ndarray) -> np.ndarray:
    return 0.5 * (landmark_xy(lm, MOUTH_L) + landmark_xy(lm, MOUTH_R))


def torso_center(lm: np.ndarray) -> np.ndarray:
    """双肩与双髋四点的中点（除以 2 的写法沿用正式规则，不是四点平均）。"""
    return 0.5 * (landmark_xy(lm, L_SHOULDER) + landmark_xy(lm, R_SHOULDER) + landmark_xy(lm, L_HIP)
                  + landmark_xy(lm, R_HIP))


def guard_distances(lm: np.ndarray, side: str, mouth: np.ndarray, center: np.ndarray) -> tuple[float, float]:
    """``side`` 手的腕–嘴距离与肘–躯干中心距离。"""
    wrist, elbow = (L_WRIST, L_ELBOW) if side == "L" else (R_WRIST, R_ELBOW)
    return (np.linalg.norm(landmark_xy(lm, wrist) - mouth), np.linalg.norm(landmark_xy(lm, elbow) - center))


def arm_extensions(lm: np.ndarray, torso: float) -> tuple[float, float]:
    """左、右腕到同侧肩的距离 ÷ 躯干长度（伸展比）。"""
    ext_l = np.linalg.norm(landmark_xy(lm, L_WRIST) - landmark_xy(lm, L_SHOULDER)) / torso
    ext_r = np.linalg.norm(landmark_xy(lm, R_WRIST) - landmark_xy(lm, R_SHOULDER)) / torso
    return ext_l, ext_r


def wrist_mouth_distance(lm: np.ndarray, side: str, mouth: np.ndarray) -> float:
    wrist = L_WRIST if side == "L" else R_WRIST
    return float(np.linalg.norm(landmark_xy(lm, wrist) - mouth))


# --------------------------------------------------------------------------- #
# 序列（学生练习）
# --------------------------------------------------------------------------- #


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
