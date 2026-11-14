# -*- coding: utf-8 -*-
"""侧视角低鞭腿组合动作相位窗口切分。"""

from __future__ import annotations

import numpy as np

from core.evaluation.action_catalog import (
    SEGMENT_LABELS,
    action_definition,
    action_segments,
)
from core.evaluation.kick_events import segment_kick_events
from core.evaluation.motion_input import MotionInput, prepare_motion_input
from core.evaluation.phase_instances import PhaseInstance, phase_instance

_EXPECTED_REPETITIONS = 2
_PAD_CONTEXT_S = 2.0
_PAD_POST_SETTLE_S = 0.15
_FINISH_PAD_S = 0.12
_RETRACT_TAIL_TRIM_S = 0.03


def _span_indices(
    times: np.ndarray,
    t_lo: float,
    t_hi: float,
    bound_lo: int,
    bound_hi: int,
    min_frames: int,
) -> np.ndarray:
    """将时间窗映射为帧索引，并在事件允许边界内对称扩足至 min_frames。"""
    if bound_hi < bound_lo or len(times) == 0:
        return np.array([], dtype=int)
    lo = max(bound_lo, int(np.searchsorted(times, t_lo, side="left")))
    hi = min(bound_hi, int(np.searchsorted(times, t_hi, side="right")) - 1)
    if hi < lo:
        mid = int(np.searchsorted(times, 0.5 * (t_lo + t_hi)))
        lo = hi = max(bound_lo, min(bound_hi, mid))
    while (hi - lo + 1) < min_frames and (lo > bound_lo or hi < bound_hi):
        if lo > bound_lo:
            lo -= 1
        if (hi - lo + 1) < min_frames and hi < bound_hi:
            hi += 1
    return np.arange(lo, hi + 1, dtype=int)


def _select_start_stance(
    times: np.ndarray,
    motion_in: MotionInput,
    first_start_t: float,
) -> np.ndarray | None:
    """选取起腿前最早的完整稳定护位窗口，不截断滑窗、不拿垂手帧充数。"""
    for idx in motion_in.stable:
        if len(idx) > 0 and float(times[idx[-1]]) < first_start_t:
            return idx
    return None


def _select_end_stance(
    times: np.ndarray,
    motion_in: MotionInput,
    last_event: dict,
) -> np.ndarray | None:
    """仅在末次踢腿闭合落地后选取首个完整稳定护位窗口。"""
    if not last_event.get("closed"):
        return None
    last_end_t = float(last_event["t_end"])
    for idx in motion_in.stable:
        if len(idx) > 0 and float(times[idx[0]]) >= last_end_t:
            return idx
    return None


def _kick_segment_name(action: str) -> str:
    segments = action_segments(action)
    if "front_low_kick" in segments:
        return "front_low_kick"
    if "rear_low_kick" in segments:
        return "rear_low_kick"
    raise ValueError(f"动作 {action!r} 不包含低鞭腿段")


def _append_rep_phases(
    parts: dict[tuple[str, str], list[np.ndarray]],
    instances: list[PhaseInstance],
    times: np.ndarray,
    events: list[dict],
    rep_idx: int,
    kick_seg: str,
    has_pad: bool,
    min_frames: int,
) -> None:
    ev = events[rep_idx - 1]
    s_idx, b_idx = ev["start_idx"], ev["end_idx"]
    t_start, t_peak, t_ext, t_end = (
        ev["t_start"],
        ev["t_peak"],
        ev["t_ext"],
        ev["t_end"],
    )
    hi_peak = max(t_ext, t_peak)

    if has_pad:
        if rep_idx > 1:
            prev_ev = events[rep_idx - 2]
            prev_bound = prev_ev["end_idx"]
            pad_t0 = min(
                t_start,
                max(prev_ev["t_end"] + _PAD_POST_SETTLE_S, t_start - _PAD_CONTEXT_S),
            )
        else:
            prev_bound = 0
            pad_t0 = max(float(times[0]), t_start - _PAD_CONTEXT_S)
        pad_idx = _span_indices(times, pad_t0, t_start, prev_bound, s_idx, min_frames)
        parts.setdefault(("pad_step", "prepare"), []).append(pad_idx)
        instances.append(phase_instance("pad_step", "prepare", rep_idx, pad_idx, times))

    mot_idx = _span_indices(times, t_start, hi_peak, s_idx, b_idx, min_frames)
    fin_lo = max(t_start, t_ext - _FINISH_PAD_S)
    fin_hi = min(t_end, hi_peak + _FINISH_PAD_S)
    fin_idx = _span_indices(times, fin_lo, fin_hi, s_idx, b_idx, min_frames)
    ret_idx = _span_indices(
        times, t_ext, max(t_ext, t_end - _RETRACT_TAIL_TRIM_S), s_idx, b_idx, min_frames
    )

    for phase, idx in (("motion", mot_idx), ("finish", fin_idx), ("retract", ret_idx)):
        parts.setdefault((kick_seg, phase), []).append(idx)
        instances.append(phase_instance(kick_seg, phase, rep_idx, idx, times))


def _collect_rep_outputs(
    events: list[dict],
    kick_seg: str,
    has_pad: bool,
    times: np.ndarray,
    min_frames: int,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, list]]:
    parts: dict[tuple[str, str], list[np.ndarray]] = {}
    instances: list[PhaseInstance] = []
    event_texts: list[str] = []
    event_details: list[dict] = []
    repetitions: list[dict] = []
    incomplete: list[int] = []
    label = SEGMENT_LABELS.get(kick_seg, kick_seg)

    for rep_idx, ev in enumerate(events, 1):
        _append_rep_phases(
            parts, instances, times, events, rep_idx, kick_seg, has_pad, min_frames
        )
        ending = "recovered" if ev["closed"] else "unrecovered"
        if not ev["closed"]:
            incomplete.append(rep_idx)
        event_texts.append(
            f"第{rep_idx}次{label} {ev['t_start']:.2f}～{ev['t_end']:.2f}s"
        )
        item = {
            "id": f"{kick_seg}_{rep_idx}",
            "segment": kick_seg,
            "repetition": rep_idx,
            "start": ev["t_start"],
            "peak": ev["t_peak"],
            "ext": ev["t_ext"],
            "end": ev["t_end"],
        }
        event_details.append(
            {**item, "ending": ending, "recovered": bool(ev["closed"])}
        )
        repetitions.append(
            {**{k: item[k] for k in item if k != "id"}, "closed": bool(ev["closed"])}
        )

    rep_windows = {
        key: np.unique(np.concatenate(chunks))
        for key, chunks in parts.items()
        if chunks
    }
    records = {
        "events": event_texts,
        "eventDetails": event_details,
        "phaseInstances": instances,
        "repetitions": repetitions,
        "incomplete": incomplete,
    }
    return rep_windows, records


def kick_combo_windows(
    series: dict,
    action: str,
    stance: str,
    cfg,
    data: dict | None = None,
    prepared: MotionInput | None = None,
) -> dict[tuple[str, str], np.ndarray]:
    """切分侧视角低鞭腿组合动作的相位窗口，并填充重复次数与诊断信息。"""
    diagnostics = data if data is not None else {}
    motion_in = prepared if prepared is not None else prepare_motion_input(series, cfg)
    times = motion_in.times
    per_rep = action_definition(action).scoring == "per_repetition"
    events = segment_kick_events(series, action, stance, prepared=motion_in)

    if not events:
        diagnostics.update(
            {
                "reason": (
                    "未检测到完整的低鞭腿动作事件"
                    if motion_in.stable
                    else "没有找到持续稳定的起始护位"
                ),
                "events": [],
                "eventDetails": [],
                "phaseInstances": [],
                "repetitions": [],
                "unpairedEventIds": [],
            }
        )
        if per_rep:
            diagnostics["expectedRepetitions"] = _EXPECTED_REPETITIONS
        if motion_in.stable:
            return {
                ("stance", "start"): motion_in.stable[0],
                ("stance", "end"): motion_in.stable[-1],
            }
        return {}

    kick_seg = _kick_segment_name(action)
    has_pad = "pad_step" in action_segments(action)
    min_frames = max(1, int(getattr(cfg, "min_valid", 3)))

    windows: dict[tuple[str, str], np.ndarray] = {}
    start_win = _select_start_stance(times, motion_in, float(events[0]["t_start"]))
    if start_win is not None:
        windows[("stance", "start")] = start_win
    end_win = _select_end_stance(times, motion_in, events[-1])
    if end_win is not None:
        windows[("stance", "end")] = end_win

    rep_windows, records = _collect_rep_outputs(
        events, kick_seg, has_pad, times, min_frames
    )
    windows.update(rep_windows)
    incomplete = records.pop("incomplete")
    diagnostics.update(records)
    diagnostics["unpairedEventIds"] = []
    if incomplete:
        diagnostics["incompleteRecovery"] = incomplete
    if per_rep:
        diagnostics["expectedRepetitions"] = _EXPECTED_REPETITIONS
        diagnostics["reason"] = (
            ""
            if len(events) >= _EXPECTED_REPETITIONS
            else f"侧面只检测到 {len(events)} 次（预期 {_EXPECTED_REPETITIONS} 次）"
        )
    else:
        diagnostics["reason"] = "已检测到完整动作"
    return windows
