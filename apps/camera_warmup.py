# -*- coding: utf-8 -*-
"""旧导入路径的转发门面：摄像头池已迁到 ``core.media.camera.pool``，``CameraWarmupPool`` 更名为 ``CameraPool``（P5）；P9 删除。"""
from core.media import (
    PRIMARY,
    SECONDARY,
    CameraLease,
    CameraWarmupError,
    CameraWarmupStopped,
    CameraWarmupTimeout,
    CaptureSpec,
    WarmFrame,
)
from core.media import CameraPool as CameraWarmupPool

__all__ = [
    "CameraLease",
    "CameraWarmupError",
    "CameraWarmupPool",
    "CameraWarmupStopped",
    "CameraWarmupTimeout",
    "CaptureSpec",
    "PRIMARY",
    "SECONDARY",
    "WarmFrame",
]
