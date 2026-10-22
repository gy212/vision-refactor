# -*- coding: utf-8 -*-
"""画面方向变换的唯一实现：按 90° 倍数旋转帧（``apply_rotation``）与旋转后的尺寸（``rotated_size``），预览与录像共用。"""
from __future__ import annotations

import cv2
import numpy as np

_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def apply_rotation(frame: np.ndarray, degrees: int) -> np.ndarray:
    """按度数旋转帧；0（或非 90 倍数）时原样返回（不复制）。"""
    code = _ROTATE_CODES.get(degrees)
    return cv2.rotate(frame, code) if code is not None else frame


def rotated_size(width: int, height: int, degrees: int) -> tuple[int, int]:
    """旋转后的 ``(width, height)``，供录制登记尺寸与布局使用。"""
    return (height, width) if degrees in (90, 270) else (width, height)


__all__ = ["apply_rotation", "rotated_size"]
