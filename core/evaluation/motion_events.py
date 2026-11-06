"""从肢体离开和返回信号识别有明确边界的运动事件。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class EventConfig(Protocol):
    min_valid: int
    min_motion_seconds: float
    max_gap_seconds: float
    return_distance: float


@dataclass(frozen=True)
class MotionEvent:
    """一次观测到的运动；held 表示尚未观察到返回。"""

    start: int
    peak: int
    end: int
    ending: str


@dataclass(frozen=True)
class EventPolicy:
    """区分已批准的起点和返回边界，不改变姿势质量标准。"""

    start_at_rest: bool = False
    inclusive_return: bool = False
    allow_held: bool = True
    hold_seconds: float = 0.3
    max_gap_seconds: float | None = None
    rearm_after_gap: bool = True


def _observed_event(
    signal: np.ndarray,
    times: np.ndarray,
    start: int,
    end: int,
    ending: str,
    cfg: EventConfig,
    forward: np.ndarray | None,
    downward: np.ndarray | None,
) -> MotionEvent | None:
    observed = np.arange(start, end + 1)
    observed = observed[np.isfinite(signal[observed])]
    if len(observed) < cfg.min_valid or times[end] - times[start] < cfg.min_motion_seconds:
        return None
    peak = int(observed[np.argmax(signal[observed])])
    if forward is not None and downward is not None:
        if not np.isfinite(downward[peak]) or downward[peak] > forward[peak]:
            return None
    return MotionEvent(start, peak, end, ending)


def _held_end(
    signal: np.ndarray,
    times: np.ndarray,
    start: int,
    last: int,
    rise: float,
    cfg: EventConfig,
    policy: EventPolicy,
) -> int | None:
    for first in range(start, last):
        end = int(np.searchsorted(times, times[first] + policy.hold_seconds))
        if end > last:
            break
        window = signal[first : end + 1]
        if (
            np.isfinite(window).all()
            and np.min(window) > rise
            and np.ptp(window) <= cfg.return_distance / 2
        ):
            return end
    return None


def excursion_events(
    signal: np.ndarray,
    times: np.ndarray,
    baseline: np.ndarray,
    cfg: EventConfig,
    *,
    rise: float,
    fall: float,
    policy: EventPolicy,
    forward: np.ndarray | None = None,
    downward: np.ndarray | None = None,
) -> list[MotionEvent]:
    """识别往返与末尾保持，缺帧后的起点按明确的既有策略处理。

    forward/downward 为相对初始护位的位移；直拳用二者排除向下放手，
    其他动作不提供方向约束。起止阈值由使用方的明确策略提供。
    """
    events: list[MotionEvent] = []
    last = rest = int(baseline[-1])
    start: int | None = None
    armed = True
    max_gap = cfg.max_gap_seconds if policy.max_gap_seconds is None else policy.max_gap_seconds
    for index in range(last + 1, len(signal)):
        value = signal[index]
        if not np.isfinite(value):
            if times[index] - times[last] > max_gap:
                start = None
                armed = not policy.rearm_after_gap
            continue
        returned = value <= fall if policy.inclusive_return else value < fall
        if start is None:
            if returned:
                armed, rest = True, index
            elif value > rise and armed:
                start = rest if policy.start_at_rest else index
                armed = False
        elif returned:
            event = _observed_event(signal, times, start, index, "returned", cfg, forward, downward)
            if event is not None:
                events.append(event)
            start, armed, rest = None, True, index
        last = index
    if start is not None and policy.allow_held:
        end = _held_end(signal, times, start, last, rise, cfg, policy)
        if end is not None:
            event = _observed_event(signal, times, start, end, "held", cfg, forward, downward)
            if event is not None:
                events.append(event)
    return events
