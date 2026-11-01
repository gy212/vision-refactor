"""从侧面前伸信号定位出拳事件，区分回收、保持与向下放手。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_GAP_SECONDS = .25
HOLD_SECONDS = .3


@dataclass(frozen=True)
class PunchEvent:
    """事件终点不等于合格护位；held 表示未观察到回收。"""

    start: int
    peak: int
    end: int
    ending: str


def reach_events(
    reach: np.ndarray,
    downward: np.ndarray,
    times: np.ndarray,
    baseline: np.ndarray,
    cfg,
    *,
    rise: float,
    fall: float,
) -> list[PunchEvent]:
    """完整往返沿用原边界；未回落但持续停住的事件保留为 held。

    向下位移大于向前位移的放手不作为直拳事件；此处不检查肘角、
    拳肩是否齐平或收回位置是否正确，质量交给各自的规则判断。
    """
    events = []
    start = last = None
    initial_x = float(np.nanmedian(reach[baseline]))
    initial_y = float(np.nanmedian(downward[baseline]))

    def finish(end: int, ending: str) -> None:
        observed = np.arange(start, end + 1)
        observed = observed[np.isfinite(reach[observed])]
        if len(observed) < cfg.min_valid or times[end] - times[start] < cfg.min_motion_seconds:
            return
        peak = int(observed[np.argmax(reach[observed])])
        if not np.isfinite(downward[peak]):
            return
        forward = reach[peak] - initial_x
        if downward[peak] - initial_y > forward:
            return
        events.append(PunchEvent(start, peak, end, ending))

    for index, value in enumerate(reach):
        if not np.isfinite(value):
            if start is not None and last is not None and times[index] - times[last] > MAX_GAP_SECONDS:
                start = None
            continue
        if start is None and value > rise:
            start = index
        elif start is not None and value < fall:
            finish(index, "returned")
            start = None
        last = index

    # shortcut: 保持容错只补尾部未闭合事件；连续多次不收回时再扩展逐峰分割。
    if start is not None and last is not None:
        peak = start + int(np.nanargmax(reach[start:last + 1]))
        for end in range(peak + 1, last + 1):
            if times[end] - times[peak] < HOLD_SECONDS:
                continue
            window = reach[peak:end + 1]
            if np.isfinite(window).all() and np.ptp(window) <= cfg.return_distance / 2:
                finish(end, "held")
            break
    return events
