"""为已观测基础动作生成唯一事件身份、检查阶段与共享组合衔接窗口。"""

from __future__ import annotations

from collections import Counter

import numpy as np

from core.evaluation.action_events import ActionEvent
from core.evaluation.phase_instances import phase_instance
from core.evaluation.punch_events import preceding_punch
from core.evaluation.punch_phases import punch_detail_windows
from core.evaluation.rules.feedback_config import FeedbackConfig


def observed_phase_windows(
    observed: list[ActionEvent],
    times: np.ndarray,
    stable: list[np.ndarray],
    cfg: FeedbackConfig,
    diagnostics: dict,
) -> dict[tuple[str, str], np.ndarray]:
    """每个确认事件先产生自己的阶段，组合只允许引用，不能增删这些事实。"""
    counts: Counter[str] = Counter()
    records = []
    grouped: dict[str, list[ActionEvent]] = {}
    for event in sorted(observed, key=lambda item: item.motion.peak):
        segment = event.segment
        counts[segment] += 1
        records.append(
            {
                **event.to_record(times),
                "eventId": f"{segment}:{counts[segment]}",
                "repetition": counts[segment],
            }
        )
        grouped.setdefault(segment, []).append(event)
    diagnostics["eventDetails"] = records
    diagnostics["events"] = [record["segment"] for record in records]
    diagnostics["incompleteRecovery"] = sum(event.motion.ending == "held" for event in observed)
    instances = diagnostics["phaseInstances"]
    windows = {}
    radius = max(1, int(round(cfg.finish_window_seconds / float(np.median(np.diff(times))))))
    for segment, events in grouped.items():
        parts: dict[str, list[np.ndarray]] = {"motion": [], "finish": []}
        for repetition, event in enumerate(events, 1):
            motion = event.motion
            phases = {
                "motion": np.arange(motion.start, motion.end + 1),
                "finish": np.arange(
                    max(motion.start, motion.peak - radius),
                    min(motion.end, motion.peak + radius) + 1,
                ),
            }
            for phase, indices in phases.items():
                parts[phase].append(indices)
                instances.append(phase_instance(segment, phase, repetition, indices, times))
        for phase, chunks in parts.items():
            if chunks:
                windows[(segment, phase)] = np.unique(np.concatenate(chunks))
        if segment.endswith(("_straight", "_hook")):
            windows.update(
                punch_detail_windows(
                    times,
                    [event.motion for event in events],
                    stable,
                    segment=segment,
                    phase_instances=instances,
                )
            )
    return windows


def combo_transition_windows(
    observed: list[ActionEvent],
    baseline: np.ndarray,
    times: np.ndarray,
    front_seg: str,
    rear_seg: str,
    cfg: FeedbackConfig,
    diagnostics: dict,
) -> dict[tuple[str, str], np.ndarray]:
    """为前后手成对组合生成共享衔接窗口与配对记录，支持边收边出空档<=0正常判定。"""
    front_motions = sorted(
        (
            ev.motion
            for ev in observed
            if ev.segment == front_seg and ev.motion.start > baseline[-1]
        ),
        key=lambda item: item.peak,
    )
    rear_motions = sorted(
        (ev.motion for ev in observed if ev.segment == rear_seg and ev.motion.start > baseline[-1]),
        key=lambda item: item.peak,
    )
    identities = {
        (rec["segment"], rec["start"]): rec["eventId"]
        for rec in diagnostics.get("eventDetails", [])
    }
    pairs: list[dict] = []
    parts: list[np.ndarray] = []
    instances = diagnostics["phaseInstances"]
    previous_end = int(baseline[-1])
    for rep_idx, rear in enumerate(rear_motions, 1):
        paired = preceding_punch(front_motions, previous_end + 1, rear.start, rear_peak=rear.peak)
        previous_end = rear.end
        if paired is None:
            continue
        pairs.append(
            {
                "repetition": rep_idx,
                "frontEventId": identities[(front_seg, float(times[paired.start]))],
                "rearEventId": identities[(rear_seg, float(times[rear.start]))],
                "frontStart": float(times[paired.start]),
                "frontPeak": float(times[paired.peak]),
                "frontEnd": float(times[paired.end]),
                "frontEnding": paired.ending,
                "rearStart": float(times[rear.start]),
                "rearPeak": float(times[rear.peak]),
                "rearEnd": float(times[rear.end]),
                "rearEnding": rear.ending,
            }
        )
        lo, hi = paired.peak, rear.peak
        while hi - lo + 1 < cfg.min_valid and (lo > paired.start or hi < rear.end):
            if lo > paired.start:
                lo -= 1
            if hi - lo + 1 < cfg.min_valid and hi < rear.end:
                hi += 1
        indices = np.arange(lo, hi + 1)
        parts.append(indices)
        instances.append(phase_instance("combo_transition", "motion", rep_idx, indices, times))
    diagnostics["comboPairs"] = pairs
    if not parts:
        return {}
    return {("combo_transition", "motion"): np.unique(np.concatenate(parts))}


def final_stance_window(observed: list[ActionEvent], stable: list[np.ndarray]) -> np.ndarray | None:
    """结束实战式只取全部已观测动作之后的真实回收窗口。"""
    last = max(observed, key=lambda event: event.motion.end).motion
    if last.ending == "returned":
        return next((indices for indices in stable if indices[0] >= last.end), None)
    return None
