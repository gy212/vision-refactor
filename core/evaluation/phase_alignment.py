"""核对各检查阶段的跨机位时间对应，保留独立事件与借用时间的来源。"""

from __future__ import annotations

from collections import Counter

import numpy as np

from core.foundation.signal import contiguous_runs


def _ranges(view: dict, indices: np.ndarray, offset: float = 0.0) -> list[tuple[float, float]]:
    times = np.asarray(view["times"])
    return [
        (float(times[run[0]]) + offset, float(times[run[-1]]) + offset)
        for run in contiguous_runs(indices)
    ]


def phase_correspondence(
    views: dict, windows: dict, offset: float, tolerance: float, clock_reliable: bool
) -> dict:
    """各阶段独立核对范围；无关阶段的计数冲突不改变本站姿的对应关系。"""
    keys = set().union(*(set(part) for part in windows.values()))
    result = {}
    for segment, phase in sorted(keys):
        key = (segment, phase)
        ranges = {
            view: _ranges(series, windows[view][key], offset if view == "side" else 0)
            for view, series in views.items()
            if key in windows[view]
        }
        entry = {"reliable": False, "ranges": ranges, "reason": "缺少对应机位的动作阶段"}
        if set(ranges) == {"front", "side"}:
            front, side = ranges["front"], ranges["side"]
            if not clock_reliable:
                entry["reason"] = "正侧面时间无法稳定对应"
            elif len(front) != len(side):
                entry["reason"] = "正侧面该阶段检测到的动作次数不一致"
            elif any(
                max(abs(a - b) for a, b in zip(f, s)) > tolerance for f, s in zip(front, side)
            ):
                entry["reason"] = "正侧面动作阶段时间不对应，请使用同一次同步录制的视频"
            else:
                entry.update(reliable=True, reason="该阶段正侧面时间可对应")
        result[f"{segment}:{phase}"] = entry
    return result


def event_correspondence(diagnostics: dict, offset: float, tolerance: float) -> dict:
    """仅配对独立观测的同类事件；投影而来的时间另记来源。"""
    observed = {view: data.get("eventDetails", []) for view, data in diagnostics.items()}
    counts = {
        view: dict(Counter(event["segment"] for event in events))
        for view, events in observed.items()
    }
    front_events, side_events = observed.get("front", []), observed.get("side", [])
    front_candidates = {index: [] for index in range(len(front_events))}
    side_candidates = {index: [] for index in range(len(side_events))}
    for front_index, front in enumerate(front_events):
        for side_index, side in enumerate(side_events):
            if (
                side["segment"] == front["segment"]
                and abs(front["peak"] - side["peak"] - offset) <= tolerance
            ):
                front_candidates[front_index].append(side_index)
                side_candidates[side_index].append(front_index)
    pairs, unmatched = [], []
    paired = {"front": set(), "side": set()}
    for front_index, candidates in front_candidates.items():
        if len(candidates) == 1 and len(side_candidates[candidates[0]]) == 1:
            side_index = candidates[0]
            pairs.append(
                {
                    "segment": front_events[front_index]["segment"],
                    "frontIndex": front_index,
                    "sideIndex": side_index,
                }
            )
            paired["front"].add(front_index)
            paired["side"].add(side_index)
    for view, events, candidates in (
        ("front", front_events, front_candidates),
        ("side", side_events, side_candidates),
    ):
        for index, event in enumerate(events):
            if index in paired[view]:
                continue
            reason = "对应候选不唯一" if candidates[index] else "没有时间相符的同类事件"
            unmatched.append(
                {
                    "view": view,
                    "index": index,
                    **event,
                    "reason": reason,
                    "candidateIndices": candidates[index],
                }
            )
    return {"eventCounts": counts, "eventPairs": pairs, "unmatchedEvents": unmatched}


def align_views(views: dict, windows: dict, cfg, diagnostics: dict | None = None) -> dict:
    """估计小时间偏移并分别记录整体冲突与每个检查阶段的对应状态。"""
    alignment = {
        "method": "video_timestamps",
        "sideOffsetSeconds": 0.0,
        "anchors": 0,
        "reliable": set(views) == {"front", "side"},
    }
    offsets, count_conflict = [], False
    if alignment["reliable"]:
        keys = windows["front"].keys() | windows["side"].keys()
        for key in keys:
            if key[1] != "finish":
                continue
            front = contiguous_runs(windows["front"].get(key, np.array([], dtype=int)))
            side = contiguous_runs(windows["side"].get(key, np.array([], dtype=int)))
            if len(front) != len(side):
                both_observed = diagnostics is None or all(
                    diagnostics.get(view, {}).get("eventDetails") for view in ("front", "side")
                )
                count_conflict |= bool(front and side) or both_observed
                continue
            for f, side_run in zip(front, side):
                offsets.append(
                    float(
                        np.median(np.asarray(views["front"]["times"])[f])
                        - np.median(np.asarray(views["side"]["times"])[side_run])
                    )
                )
    clock_reliable = alignment["reliable"]
    alignment["anchors"] = len(offsets)
    if len(offsets) >= 2:
        offset = float(np.median(offsets))
        residual = float(np.max(np.abs(np.asarray(offsets) - offset)))
        if abs(offset) > cfg.phase_tolerance_seconds or residual > 0.1:
            clock_reliable = False
            alignment.update(reliable=False, reason="正侧面峰值时间无法稳定对应")
        else:
            alignment.update(
                method="action_peaks", sideOffsetSeconds=offset, residualSeconds=residual
            )
    if count_conflict:
        alignment.update(reliable=False, reason="正侧面检测到的动作次数不一致")
    alignment["phases"] = phase_correspondence(
        views,
        windows,
        alignment["sideOffsetSeconds"],
        cfg.phase_tolerance_seconds,
        clock_reliable,
    )
    if diagnostics is not None:
        alignment.update(
            event_correspondence(
                diagnostics,
                alignment["sideOffsetSeconds"],
                cfg.phase_tolerance_seconds,
            )
        )
    return alignment


def project_event_times(
    events: list[dict],
    source_offset: float,
    target_offset: float,
    *,
    fields: tuple[str, ...] = ("start", "peak", "end"),
) -> list[dict]:
    """把源机位事件时刻换算到目标原时轴，原观测记录保持不变。"""
    shift = source_offset - target_offset
    return [{**event, **{key: event[key] + shift for key in fields}} for event in events]


def share_phase_timing(views, windows, alignment, diagnostics, cfg):
    """同步录像可共用可见机位的时间段；不借用它的关键点或测量结果。"""
    sources = {view: "own_view" for view in views}
    if set(views) != {"front", "side"} or not alignment["reliable"]:
        return sources
    for target, source in (("front", "side"), ("side", "front")):
        if any(key[1] == "motion" for key in windows[target]) or diagnostics[target].get("events"):
            continue  # 已有相反的次数/顺序证据时不能覆盖它。
        if not any(key[1] == "motion" for key in windows[source]):
            continue
        start = ("stance", "start")
        if start not in windows[target] or start not in windows[source]:
            continue
        times = {
            v: np.asarray(views[v]["times"])
            + (alignment["sideOffsetSeconds"] if v == "side" else 0)
            for v in (target, source)
        }
        if (
            abs(
                np.median(times[target][windows[target][start]])
                - np.median(times[source][windows[source][start]])
            )
            > cfg.phase_tolerance_seconds
        ):
            continue
        windows[target] = project_windows(windows[source], times[source], times[target])
        diagnostics[target]["projectedEventDetails"] = project_event_times(
            diagnostics[source].get("eventDetails", []),
            alignment["sideOffsetSeconds"] if source == "side" else 0.0,
            alignment["sideOffsetSeconds"] if target == "side" else 0.0,
        )
        diagnostics[target]["projectedPhaseInstances"] = project_event_times(
            diagnostics[source].get("phaseInstances", []),
            alignment["sideOffsetSeconds"] if source == "side" else 0.0,
            alignment["sideOffsetSeconds"] if target == "side" else 0.0,
            fields=("start", "end"),
        )
        diagnostics[target]["timingSource"] = source
        sources[target] = source + "_synchronized_timing"
    alignment["phases"] = phase_correspondence(
        views, windows, alignment["sideOffsetSeconds"], cfg.phase_tolerance_seconds, True
    )
    return sources


def project_windows(windows: dict, source_times, target_times) -> dict:
    """把一路的相位窗口按时间投到同步录制的另一路：每段连续帧取另一路同一时间范围内的帧，超出另一路时长的段丢弃。"""
    source_times, target_times = np.asarray(source_times), np.asarray(target_times)
    projected = {}
    for key, indices in windows.items():
        runs = []
        for run in contiguous_runs(indices):
            lo, hi = source_times[run[[0, -1]]]
            if lo < target_times[0] - 1e-6 or hi > target_times[-1] + 1e-6:
                continue
            runs.append(np.flatnonzero((target_times >= lo - 1e-6) & (target_times <= hi + 1e-6)))
        if runs:
            projected[key] = np.unique(np.concatenate(runs))
    return projected
