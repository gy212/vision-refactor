# -*- coding: utf-8 -*-
"""躯干测量：投影肩宽（躯干朝向）、可见侧肩髋链的躯干长度。"""

from __future__ import annotations

import numpy as np

from core.foundation.geometry import torso_length
from core.pose.landmarks import L_SHOULDER, R_SHOULDER


def chain_torso(lm: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """逐帧躯干长度：两侧都可见的同侧肩髋链求平均，都不可见为 0（练习用）。"""
    return torso_length(lm, mask, "visible_side_chain")


def shoulder_width(lm: np.ndarray) -> np.ndarray:
    """逐帧两肩间距（投影肩宽：正对镜头时宽，侧身时窄）。"""
    return np.linalg.norm(lm[:, L_SHOULDER, :2] - lm[:, R_SHOULDER, :2], axis=1)


def body_forward(lm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从肩线和身体上下方向估计前向，返回单位方向、方向长度及身体长度。"""
    shoulders = (lm[:, 11, :3] + lm[:, 12, :3]) / 2
    hips = (lm[:, 23, :3] + lm[:, 24, :3]) / 2
    up = shoulders - hips
    forward = np.cross(lm[:, 12, :3] - lm[:, 11, :3], up)
    length = np.linalg.norm(forward, axis=1)
    return forward / np.maximum(length[:, None], 1e-4), length, np.linalg.norm(up, axis=1)
