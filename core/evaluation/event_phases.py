"""为已观测基础动作生成唯一事件身份及检查阶段，独立于组合配对。"""

from __future__ import annotations

from collections import Counter

import numpy as np

from core.evaluation.action_events import ActionEvent
from core.evaluation.phase_instances import phase_instance
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
        if segment.endswith("_straight"):
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


def final_stance_window(observed: list[ActionEvent], stable: list[np.ndarray]) -> np.ndarray | None:
    """结束实战式只取全部已观测动作之后的真实回收窗口。"""
    last = max(observed, key=lambda event: event.motion.end).motion
    if last.ending == "returned":
        return next((indices for indices in stable if indices[0] >= last.end), None)
    return None
