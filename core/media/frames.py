# -*- coding: utf-8 -*-
"""帧有效性的唯一判定：``is_valid_bgr`` 判断一帧能否进入预览、推理或录像；``FrameSet`` 是按角色组织的一组帧。"""
from __future__ import annotations

from typing import Any

import numpy as np

FrameSet = dict[str, np.ndarray]


def is_valid_bgr(frame: Any) -> bool:
    """非空的 ``uint8`` 三通道 BGR 数组才算有效帧。"""
    return (
        isinstance(frame, np.ndarray)
        and frame.dtype == np.uint8
        and frame.ndim == 3
        and frame.shape[2] == 3
        and frame.shape[0] > 0
        and frame.shape[1] > 0
    )


__all__ = ["FrameSet", "is_valid_bgr"]
