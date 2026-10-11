# -*- coding: utf-8 -*-
"""躯干测量：投影肩宽（躯干朝向）、可见侧肩髋链的躯干长度。"""
from __future__ import annotations

import numpy as np

from core.foundation.geometry import torso_length
from core.pose.landmarks import L_SHOULDER, R_SHOULDER


def midpoint_torso(lm: np.ndarray) -> float:
    """单帧躯干长度：双肩中点到双髋中点 + 1e-9（正式规则用）。"""
    return torso_length(lm)


def chain_torso(lm: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """逐帧躯干长度：两侧都可见的同侧肩髋链求平均，都不可见为 0（练习用）。"""
    return torso_length(lm, mask, "visible_side_chain")


def shoulder_width(lm: np.ndarray) -> np.ndarray:
    """逐帧两肩间距（投影肩宽：正对镜头时宽，侧身时窄）。"""
    return np.linalg.norm(lm[:, L_SHOULDER, :2] - lm[:, R_SHOULDER, :2], axis=1)
