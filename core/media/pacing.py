# -*- coding: utf-8 -*-
"""恒定帧率写出的节拍 ``FramePacer``：按帧的采集时刻算出这一帧写几次，使录像时长等于真实时长（D-18）。

实时会话每处理完一帧才写一帧，处理速度（8～20 fps）低于 writer 的名义帧率（30 fps）时，按帧原样写出会让
回放变快。节拍器把采集时刻折算到恒定帧率的时间轴上：慢于名义帧率时重复写当前帧补足空档，快于名义帧率时
跳过已占满时间槽的帧。时刻用 ``time.monotonic()`` 的秒数；时刻为 None 表示不按时间写（视频文件输入）。
"""
from __future__ import annotations

import math


class FramePacer:
    """一个录像文件的时间轴：第 k 帧占 ``[k/fps, (k+1)/fps)``，首帧的采集时刻是零点。"""

    def __init__(self, fps: float) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"帧率必须为正数：{fps}")
        self._fps = float(fps)
        self._origin: float | None = None
        self._written = 0

    @property
    def written(self) -> int:
        """时间轴上已写出的帧数（含重复写的帧）。"""
        return self._written

    def reset(self) -> None:
        """新文件：时间轴清零，下一帧作为零点。"""
        self._origin = None
        self._written = 0

    def resync(self) -> None:
        """暂停后继续：下一帧紧接已写的帧，暂停期间不计入时长。"""
        self._origin = None

    def repeats(self, captured_at: float | None) -> int:
        """采集于 ``captured_at`` 的帧应写入的次数；0 表示它的时间槽已被前一帧占满，跳过。"""
        if captured_at is None:
            self._written += 1
            return 1
        if self._origin is None:
            self._origin = captured_at - self._written / self._fps
            self._written += 1
            return 1
        # 取整到最近的时间槽：名义帧率的采集抖动不会造成成对的丢帧与重复帧。
        slot = math.floor((captured_at - self._origin) * self._fps + 0.5)
        count = max(0, slot - self._written + 1)
        self._written += count
        return count


__all__ = ["FramePacer"]
