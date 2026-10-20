# -*- coding: utf-8 -*-
"""帧契约：``is_valid_bgr`` 是帧有效性的唯一判定，``CapturedFrame`` 是采集帧的身份，``capture_clock_ns`` 是采集时钟。

``FrameSet`` 是按角色组织的一组帧。采集时钟用 ``time.perf_counter_ns``：Windows 上 ``time.monotonic`` 是
``GetTickCount64``，分辨率 15.6 ms，约半个 30 fps 帧周期；``perf_counter`` 基于系统范围的 QPC，
采集子进程与主进程取到的值可以直接比较（设计文档 §8.8）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

FrameSet = dict[str, np.ndarray]


def capture_clock_ns() -> int:
    """采集时间戳的时钟（纳秒）；预热帧、会话帧与采集子进程都用它。"""
    return time.perf_counter_ns()


@dataclass(frozen=True)
class CapturedFrame:
    """一路摄像头采到的一帧。``captured_at_ns`` 是主机取帧完成的时刻（不是曝光时刻），
    ``seq`` 在同一 ``generation``（一次设备打开）内递增；``bgr`` 发布后只读，绘制要用副本。"""

    role: str
    seq: int
    captured_at_ns: int
    bgr: np.ndarray
    generation: int = 0


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


__all__ = ["CapturedFrame", "FrameSet", "capture_clock_ns", "is_valid_bgr"]
