"""从侧面前伸信号定位出拳事件，区分回收、保持与向下放手。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.evaluation.motion_events import EventPolicy, MotionEvent, excursion_events


@dataclass(frozen=True)
class PunchObservation:
    """一路手臂的前伸信号、护位参照与独立观测事件。"""

    reach: np.ndarray
    baseline: float
    events: list[MotionEvent]


def observe_punch(
    relative: np.ndarray,
    times: np.ndarray,
    baseline: np.ndarray,
    cfg,
    *,
    facing: float,
    rise: float,
    fall: float,
    stable: list[np.ndarray],
) -> PunchObservation:
    """前后手共用前伸识别；阈值与是否按稳定护位结束由调用方明确提供。"""
    reach = relative[:, 0] * facing
    if np.isnan(reach[baseline]).all():
        return PunchObservation(reach, float("nan"), [])
    initial = float(np.nanmedian(reach[baseline]))
    events = reach_events(
        reach, relative[:, 1], times, baseline, cfg, rise=initial + rise, fall=initial + fall
    )
    events = settled_returns(events, reach, times, stable, cfg, rise=initial + rise)
    return PunchObservation(reach, initial, events)


def preceding_punch(events: list[MotionEvent], floor: int, before: int) -> MotionEvent | None:
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
) -> list[MotionEvent]:
    """完整往返沿用原边界；未回落但持续停住的事件保留为 held。

    向下位移大于向前位移的放手不作为直拳事件；此处不检查肘角、
    拳肩是否齐平或收回位置是否正确，质量交给各自的规则判断。
    """
    forward = reach - float(np.nanmedian(reach[baseline]))
    descent = downward - float(np.nanmedian(downward[baseline]))
    return excursion_events(
        reach,
        times,
        baseline,
        cfg,
        rise=rise,
        fall=fall,
        policy=EventPolicy(max_gap_seconds=0.25, rearm_after_gap=False),
        forward=forward,
        downward=descent,
    )


def settled_returns(
    events: list[MotionEvent],
    reach: np.ndarray,
    times: np.ndarray,
    stable: list[np.ndarray],
    cfg,
    *,
    rise: float,
) -> list[MotionEvent]:
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
        result.append(MotionEvent(event.start, event.peak, end, event.ending))
    return result
