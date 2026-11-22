# -*- coding: utf-8 -*-
"""摆拳事件观测与打到瞬间峰值精修，支持缺帧鲁棒基线与保持不收回（held）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.arms import segment_lengths
from core.evaluation.measures.motion import normalized_motion
from core.evaluation.motion_events import EventPolicy, MotionEvent, excursion_events
from core.evaluation.punch_events import (
    PunchObservation,
    baseline_facing,
    reach_events,
    settled_returns,
)
from core.evaluation.rules.feedback_config import FeedbackConfig

HOOK_SIDE_FRONT_REACH = 0.18
HOOK_SIDE_REAR_REACH = 0.25
HOOK_FRONT_FRONT_REACH = 0.20
HOOK_FRONT_REAR_REACH = 0.25
HOOK_RETURN_REACH = 0.10


@dataclass(frozen=True)
class HookReference:
    """摆拳观测真正使用的动作点与起始参照条件。"""

    motion_landmarks: tuple[int, int]
    motion_frames: int
    direction_frames: int
    facing: float
    reason: str = ""


def hook_motion_landmarks(role: str, stance: str) -> tuple[int, int]:
    """摆拳位移的固定点与运动点，供选源和观测共用。"""
    shoulder, _, wrist, *_ = sides_for_stance(role, stance)
    return shoulder, wrist


def hook_reference(
    lm: np.ndarray,
    mask: np.ndarray,
    torso: np.ndarray,
    baseline: np.ndarray,
    role: str,
    stance: str,
    view: str,
    cfg: FeedbackConfig,
) -> HookReference:
    """摆拳缺帧鲁棒起始参照：基线仅统计有效肩腕帧，侧面复用统一鼻髋朝向。"""
    points = hook_motion_landmarks(role, stance)
    visible = mask[baseline][:, points].all(axis=1) & (torso[baseline] > 1e-4)
    motion_frames = int(visible.sum())
    minimum = cfg.min_valid if view == "front" else 1
    if motion_frames < minimum:
        return HookReference(points, motion_frames, 0, 0.0, "起始护位肩腕点不足，无法确定摆拳位移")
    if view == "front":
        return HookReference(points, motion_frames, 0, 0.0)
    direction_frames, facing, reason = baseline_facing(lm, mask, torso, baseline, cfg)
    return HookReference(points, motion_frames, direction_frames, facing, reason)


def _refine_hook_peak(
    event: MotionEvent,
    signal: np.ndarray,
    base: float,
    lm: np.ndarray,
    mask: np.ndarray,
    torso: np.ndarray,
    role: str,
    stance: str,
    view: str,
) -> MotionEvent:
    """在有效出拳弧段内精修打到瞬间峰值：正面取最内摆顶点，侧面取小臂投影最短帧。"""
    peak_val = float(signal[event.peak])
    if not np.isfinite(peak_val) or peak_val <= base:
        return event
    arc_thresh = base + 0.65 * (peak_val - base)
    span = np.arange(event.start, event.end + 1)
    arc = span[np.isfinite(signal[span]) & (signal[span] >= arc_thresh)]
    if arc.size == 0:
        return event
    att_sh, el, wr, *_ = sides_for_stance(role, stance)
    if view == "front":
        opp_role = "rear" if role == "front" else "front"
        opp_sh = sides_for_stance(opp_role, stance)[0]
        sh_dx = lm[arc, opp_sh, 0] - lm[arc, att_sh, 0]
        sh_width = segment_lengths(lm[arc], att_sh, opp_sh)
        ok = (
            mask[arc][:, (att_sh, opp_sh, wr)].all(axis=1)
            & (np.abs(sh_dx) > 1e-4)
            & (sh_width > 1e-4)
        )
        if not ok.any():
            return event
        cand = arc[ok]
        mid_x = 0.5 * (lm[cand, att_sh, 0] + lm[cand, opp_sh, 0])
        inward = np.sign(lm[cand, opp_sh, 0] - lm[cand, att_sh, 0])
        cw = segment_lengths(lm[cand], att_sh, opp_sh)
        signed_cross = (lm[cand, wr, 0] - mid_x) * inward / cw
        refined = int(cand[int(np.argmax(signed_cross))])
        return MotionEvent(event.start, refined, event.end, event.ending)
    ok = mask[arc][:, (el, wr)].all(axis=1) & (torso[arc] > 1e-4)
    if not ok.any():
        return event
    cand = arc[ok]
    forearm_norm = segment_lengths(lm[cand], el, wr) / torso[cand]
    refined = int(cand[int(np.argmin(forearm_norm))])
    return MotionEvent(event.start, refined, event.end, event.ending)


def _observe_front_hook(
    lm: np.ndarray,
    mask: np.ndarray,
    relative: np.ndarray,
    torso: np.ndarray,
    times: np.ndarray,
    baseline: np.ndarray,
    stable: list[np.ndarray],
    role: str,
    stance: str,
    cfg: FeedbackConfig,
) -> PunchObservation:
    """正面摆拳往返与保持识别：排除向下放手并在有效弧段精修最内摆峰值。"""
    visible = np.isfinite(relative[baseline]).all(axis=1)
    displacement = relative - np.median(relative[baseline][visible], axis=0)
    distance = np.linalg.norm(displacement, axis=1)
    rise = HOOK_FRONT_FRONT_REACH if role == "front" else HOOK_FRONT_REAR_REACH
    events = excursion_events(
        distance,
        times,
        baseline,
        cfg,
        rise=rise,
        fall=HOOK_RETURN_REACH,
        policy=EventPolicy(max_gap_seconds=0.25, rearm_after_gap=True, allow_held=True),
        forward=np.abs(displacement[:, 0]),
        downward=displacement[:, 1],
    )
    events = settled_returns(events, distance, times, stable, cfg, rise=rise)
    refined = [
        _refine_hook_peak(ev, distance, 0.0, lm, mask, torso, role, stance, "front")
        for ev in events
    ]
    return PunchObservation(distance, 0.0, refined)


def observe_hook(
    lm: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    torso: np.ndarray,
    baseline: np.ndarray,
    stable: list[np.ndarray],
    role: str,
    stance: str,
    cfg: FeedbackConfig,
    view: str = "side",
) -> PunchObservation:
    """按机位可观测投影定位摆拳事件，单拳与组合共用同一识别与峰值精修策略。"""
    reference = hook_reference(lm, mask, torso, baseline, role, stance, view, cfg)
    if reference.reason:
        return PunchObservation(np.full(len(times), np.nan), float("nan"), [], reference.reason)
    relative = normalized_motion(lm, mask, torso, *reference.motion_landmarks)
    if view == "front":
        return _observe_front_hook(
            lm, mask, relative, torso, times, baseline, stable, role, stance, cfg
        )
    reach = relative[:, 0] * reference.facing
    if np.isnan(reach[baseline]).all():
        return PunchObservation(reach, float("nan"), [])
    initial = float(np.nanmedian(reach[baseline]))
    rise = HOOK_SIDE_FRONT_REACH if role == "front" else HOOK_SIDE_REAR_REACH
    events = reach_events(
        reach,
        relative[:, 1],
        times,
        baseline,
        cfg,
        rise=initial + rise,
        fall=initial + HOOK_RETURN_REACH,
    )
    events = settled_returns(events, reach, times, stable, cfg, rise=initial + rise)
    refined = [
        _refine_hook_peak(ev, reach, initial, lm, mask, torso, role, stance, "side")
        for ev in events
    ]
    return PunchObservation(reach, initial, refined)
