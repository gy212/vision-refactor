# -*- coding: utf-8 -*-
"""单帧 landmark 的统一读取：BlazePose 33 点的索引名，MediaPipe landmark 对象与 ``(33, 4)`` 数组行共用的访问函数。"""
from __future__ import annotations

import numpy as np

LANDMARK_COUNT = 33

# BlazePose 33 点索引（规则、侧别、技术指标共用；原先 rule_scoring、kick_quality、tech_eval 各写一份）。
NOSE = 0
L_EAR, R_EAR = 7, 8
MOUTH_L, MOUTH_R = 9, 10
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT_INDEX, R_FOOT_INDEX = 31, 32


def as_rows(landmarks) -> np.ndarray | None:
    """返回 ``(N, >=2)`` 数组视图；输入是 landmark 对象序列时返回 None（调用方按对象读取）。"""
    if isinstance(landmarks, np.ndarray):
        return landmarks
    return None


def visibility(lm) -> float:
    """landmark 对象的 visibility；缺省按 1.0（与旧绘制、分类、视角分数一致）。"""
    return float(getattr(lm, "visibility", 1.0))


def landmarks_to_array(pose_landmarks) -> np.ndarray:
    """MediaPipe 单帧 landmark → ``(33, 4)`` float64 ``(x, y, z, visibility)``；缺省值为 0。

    float64 存储 float32/Python float 无损，派生视图再按旧实现的 dtype 输出。
    """
    arr = np.zeros((LANDMARK_COUNT, 4), dtype=np.float64)
    for j in range(min(LANDMARK_COUNT, len(pose_landmarks))):
        lm = pose_landmarks[j]
        arr[j, 0] = float(getattr(lm, "x", 0.0))
        arr[j, 1] = float(getattr(lm, "y", 0.0))
        arr[j, 2] = float(getattr(lm, "z", 0.0))
        arr[j, 3] = float(getattr(lm, "visibility", 0.0))
    return arr


class RowLandmark:
    """把 ``(33, 4)`` 数组的一行包装成带 ``x/y/z/visibility`` 属性的只读对象（绘制与分类复用对象接口）。"""

    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, row: np.ndarray) -> None:
        self.x = float(row[0])
        self.y = float(row[1])
        self.z = float(row[2]) if row.shape[0] > 2 else 0.0
        self.visibility = float(row[3]) if row.shape[0] > 3 else 1.0


def as_landmark_objects(landmarks) -> list | None:
    """数组输入转成 ``RowLandmark`` 列表；对象序列与 None 原样返回。"""
    rows = as_rows(landmarks)
    if rows is None:
        return landmarks
    return [RowLandmark(rows[i]) for i in range(rows.shape[0])]
