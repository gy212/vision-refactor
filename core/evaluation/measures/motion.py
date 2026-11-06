"""按身体尺度测量关节点相对位置，供姿势与运动事件共用。"""

from __future__ import annotations

import numpy as np


def relative_position(
    landmarks: np.ndarray,
    valid_mask: np.ndarray,
    scale: np.ndarray,
    anchor: int,
    moving: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回移动点相对参照点的二维位置与有效性，由调用方决定如何聚合。"""
    valid = valid_mask[:, [anchor, moving]].all(axis=1) & (scale > 1e-4)
    relative = (landmarks[:, moving, :2] - landmarks[:, anchor, :2]) / np.maximum(
        scale[:, None], 1e-4
    )
    return relative, valid


def normalized_motion(
    landmarks: np.ndarray,
    valid_mask: np.ndarray,
    torso: np.ndarray,
    anchor: int,
    moving: int,
) -> np.ndarray:
    """运动信号保留无效画面为 NaN，禁止跨缺点画面制造运动。"""
    relative, valid = relative_position(landmarks, valid_mask, torso, anchor, moving)
    return np.where(valid[:, None], relative, np.nan)
