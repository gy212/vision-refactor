"""按机位可观测的直拳位移定位事件，区分回收、保持与向下放手。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.evaluation.motion_events import EventPolicy, MotionEvent, excursion_events
from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.motion import normalized_motion

REAR_REACH = 0.30
FRONT_REACH = 0.25
RETURN_REACH = 0.12


@dataclass(frozen=True)
class PunchObservation:
    """一路手臂的可观测位移、护位参照与独立观测事件。"""

    reach: np.ndarray
    baseline: float
    events: list[MotionEvent]
    reason: str = ""


def observe_straight(
    lm: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    torso: np.ndarray,
    baseline: np.ndarray,
    stable: list[np.ndarray],
    role: str,
    stance: str,
    cfg,
    view: str = "side",
) -> PunchObservation:
    """按机位可观测投影定位直拳，单拳与组合共用同一规则。"""
    shoulder, _, wrist, *_ = sides_for_stance(role, stance)
    relative = normalized_motion(lm, mask, torso, shoulder, wrist)
    if view == "front":
        return observe_projected_punch(relative, times, baseline, stable, role, cfg)
    hip_mask = mask[baseline][:, [23, 24]]
    hip_count = hip_mask.sum(axis=1)
    reference_valid = mask[baseline, 0] & (hip_count > 0) & (torso[baseline] > 1e-4)
    if reference_valid.sum() < cfg.min_valid:
        return PunchObservation(
            np.full(len(times), np.nan),
            float("nan"),
            [],
            "起始护位的鼻点或髋点不足，无法确定出拳方向",
        )
    hips = np.where(hip_mask, lm[baseline][:, [23, 24], 0], 0).sum(axis=1)
    hips /= np.maximum(hip_count, 1)
    facing = float(np.sign(np.median((lm[baseline, 0, 0] - hips)[reference_valid])))
    if facing == 0:
        return PunchObservation(
            np.full(len(times), np.nan), float("nan"), [], "起始护位无法确定出拳方向"
        )
    return observe_punch(
        relative,
        times,
        baseline,
        cfg,
        facing=facing,
        rise=FRONT_REACH if role == "front" else REAR_REACH,
        fall=RETURN_REACH,
        stable=stable,
    )


def observe_projected_punch(
    relative: np.ndarray,
    times: np.ndarray,
    baseline: np.ndarray,
    stable: list[np.ndarray],
    role: str,
    cfg,
) -> PunchObservation:
    """正面无法量向前距离，按肩腕投影离开护位识别；向下放手仍排除。"""
    visible = np.isfinite(relative[baseline]).all(axis=1)
    if visible.sum() < cfg.min_valid:
        return PunchObservation(
            np.full(len(times), np.nan), float("nan"), [], "起始护位肩腕点不足，无法确定出拳位移"
        )
    displacement = relative - np.median(relative[baseline][visible], axis=0)
    distance = np.linalg.norm(displacement, axis=1)
    rise = FRONT_REACH if role == "front" else REAR_REACH
    events = excursion_events(
        distance,
        times,
        baseline,
        cfg,
        rise=rise,
        fall=RETURN_REACH,
        policy=EventPolicy(max_gap_seconds=0.25, rearm_after_gap=False),
        forward=np.abs(displacement[:, 0]),
        downward=displacement[:, 1],
    )
    events = settled_returns(events, distance, times, stable, cfg, rise=rise)
    return PunchObservation(distance, 0.0, events)


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
