# -*- coding: utf-8 -*-
"""平面几何：关节角、距离、单位向量与躯干长度的唯一实现，输入统一为 ndarray。

数值口径沿用合并前各调用方的实现（eps、dtype 与运算顺序逐位一致），golden 依赖这一点。
"""
from __future__ import annotations

from typing import Literal

import numpy as np

# MediaPipe Pose33 的肩、髋编号（躯干长度用）。
_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP = 11, 12, 23, 24

TorsoMethod = Literal["midpoint", "visible_side_chain"]


def landmark_xy(lm: np.ndarray, idx: int) -> np.ndarray:
    """单帧 ``(33, C)`` 点位中第 ``idx`` 个点的 ``(x, y)``，float32。"""
    return lm[idx, :2].astype(np.float32)


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """∠ABC（度），2D 点。零向量时分母取 1e-9，结果为 90°。"""
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cosang = float(np.dot(ba, bc) / denom)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """两点欧氏距离。"""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """单位向量；模长不超过 ``eps`` 时返回同形状的零向量。"""
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def torso_length(
    lm: np.ndarray,
    mask: np.ndarray | None = None,
    method: TorsoMethod = "midpoint",
):
    """躯干长度的两种定义。

    - ``midpoint``：单帧 ``(33, C)``，双肩中点到双髋中点的距离 + 1e-9（正式规则用），返回 float。
    - ``visible_side_chain``：序列 ``(T, 33, C)`` 与有效性 ``(T, 33)``，只取两侧都可见的
      同侧肩髋链求平均，一侧都不可见时为 0（学生练习用），返回 ``(T,)``。
    """
    if method == "midpoint":
        sh = 0.5 * (landmark_xy(lm, _L_SHOULDER) + landmark_xy(lm, _R_SHOULDER))
        hip = 0.5 * (landmark_xy(lm, _L_HIP) + landmark_xy(lm, _R_HIP))
        return float(np.linalg.norm(sh - hip) + 1e-9)
    if method == "visible_side_chain":
        if mask is None:
            raise ValueError("visible_side_chain 需要 mask")
        shoulders, hips = [_L_SHOULDER, _R_SHOULDER], [_L_HIP, _R_HIP]
        lengths = np.linalg.norm(lm[:, shoulders, :2] - lm[:, hips, :2], axis=2)
        valid = mask[:, shoulders] & mask[:, hips] & np.isfinite(lengths) & (lengths > 1e-4)
        return np.where(valid, lengths, 0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    raise ValueError(f"未知的躯干长度定义：{method!r}")


def _ci_probe_layer_violation():
    """临时探针：L0 向上导入 evaluation，用于验证 CI 门禁会失败（issue #9），随后回退。"""
    from core.evaluation import rules  # noqa: F401
