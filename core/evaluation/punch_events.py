"""从侧面前伸信号定位出拳事件，区分回收、保持与向下放手。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_GAP_SECONDS = 0.25
HOLD_SECONDS = 0.3


@dataclass(frozen=True)
class PunchEvent:
    """事件终点不等于合格护位；held 表示未观察到回收。"""

    start: int
    peak: int
    end: int
    ending: str


def preceding_punch(events: list[PunchEvent], floor: int, before: int) -> PunchEvent | None:
    """只配对本次后手起动之前唯一闭合的前手事件，不跨上一次后手。"""
    candidates = [
        event
        for event in events
        if event.ending == "returned" and floor <= event.start <= event.peak < event.end <= before
    ]
    return candidates[0] if len(candidates) == 1 else None


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
            if (
                start is not None
                and last is not None
                and times[index] - times[last] > MAX_GAP_SECONDS
            ):
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
        # 按局部稳定窗确认保持，末帧的小抖动不能重置整次出拳。
        for hold_start in range(start, last):
            end = int(np.searchsorted(times, times[hold_start] + HOLD_SECONDS))
            if end > last:
                break
            window = reach[hold_start : end + 1]
            if (
                np.isfinite(window).all()
                and np.min(window) > rise
                and np.ptp(window) <= cfg.return_distance / 2
            ):
                finish(end, "held")
                break
    return events


def settled_returns(
    events: list[PunchEvent],
    reach: np.ndarray,
    times: np.ndarray,
    stable: list[np.ndarray],
    cfg,
    *,
    rise: float,
) -> list[PunchEvent]:
    """已回撤并稳定时提前结束事件，不等待旧护位阈值在退场时才被越过。"""
    result = []
    for event in events:
        end = event.end
        if event.ending == "returned":
            for window in stable:
                if window[0] <= event.peak or window[-1] >= event.end:
                    continue
                candidate = int(window[0])
                observed = np.isfinite(reach[event.start : candidate + 1])
                if (
                    observed.sum() < cfg.min_valid
                    or times[candidate] - times[event.start] < cfg.min_motion_seconds
                    or (reach[window[-1] + 1 : event.end + 1] > rise).any()
                ):
                    continue
                values = reach[window]
                if (
                    np.isfinite(values).all()
                    and (values < rise).all()
                    and reach[event.peak] - values.max() > cfg.return_distance
                ):
                    end = candidate
                    break
        result.append(PunchEvent(event.start, event.peak, end, event.ending))
    return result
