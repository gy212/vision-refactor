# -*- coding: utf-8 -*-
"""摄像头采集的请求与结果定义：角色名、请求时冻结的 ``CaptureSpec``、预热帧 ``WarmFrame`` 和池的错误类型。"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Final, Literal

import numpy as np

PRIMARY: Final = "primary"
SECONDARY: Final = "secondary"
ROLES: Final = (PRIMARY, SECONDARY)
DEFAULT_CAPTURE_SIZE: Final = (1280, 720)
CapturePolicy = Literal["prefer", "require"]
CAPTURE_POLICIES: Final = ("prefer", "require")


class CameraWarmupError(RuntimeError):
    """Base error raised by the camera pool."""


class CameraWarmupTimeout(CameraWarmupError, TimeoutError):
    """The requested cameras did not become ready before their deadline."""


class CameraWarmupStopped(CameraWarmupError):
    """The caller cancelled a wait through its stop event."""


@dataclass(frozen=True)
class CaptureSpec:
    """一次打开请求的完整采集配置；请求时冻结，打开过程中不再读取模式标志。

    ``expected_name``：打开时在采集进程内复核设备名，不符即失败并提示刷新（V-3）。
    ``policy``：驱动给不出请求尺寸时，``prefer`` 接受实际尺寸，``require`` 直接失败（D-6）。
    """

    index: int
    width: int = DEFAULT_CAPTURE_SIZE[0]
    height: int = DEFAULT_CAPTURE_SIZE[1]
    expected_name: str | None = None
    policy: CapturePolicy = "prefer"

    def __post_init__(self) -> None:
        for name in ("index", "width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"camera {name} must be an integer")
        if self.index < 0:
            raise ValueError("camera index must be a non-negative integer")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera size must be positive")
        if self.policy not in CAPTURE_POLICIES:
            raise ValueError(f"camera policy must be one of {CAPTURE_POLICIES!r}")
        if self.expected_name is not None and not isinstance(self.expected_name, str):
            raise ValueError("camera expected_name must be a string")

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


@dataclass(frozen=True)
class WarmFrame:
    """Latest valid frame captured for one camera role."""

    frame: np.ndarray
    captured_at: float
    sequence: int
    generation: int


__all__ = [
    "CAPTURE_POLICIES",
    "CameraWarmupError",
    "CameraWarmupStopped",
    "CameraWarmupTimeout",
    "CapturePolicy",
    "CaptureSpec",
    "DEFAULT_CAPTURE_SIZE",
    "PRIMARY",
    "ROLES",
    "SECONDARY",
    "WarmFrame",
]
