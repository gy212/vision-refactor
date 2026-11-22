"""引用已观测的前后拳事件组成上步直拳/摆拳组合的逐次切分、衔接与步法窗口。"""

from __future__ import annotations

import numpy as np

from core.evaluation.action_catalog import SEGMENT_LABELS, motion_segments
from core.evaluation.action_events import ActionEvent
from core.evaluation.event_phases import (
    combo_transition_windows,
    final_stance_window,
    observed_phase_windows,
)
from core.evaluation.hook_events import observe_hook
from core.evaluation.motion_events import MotionEvent
from core.evaluation.motion_input import MotionInput, prepare_motion_input
from core.evaluation.phase_instances import phase_instance
from core.evaluation.punch_events import observe_straight, preceding_punch

EXPECTED_REPETITIONS = 2  # 考试每个动作做 2 次（#32 汇总一）
STEP_PRE_LEAD_SECONDS = 0.50
STEP_POST_TAIL_SECONDS = 0.60


def _combination_records(
    punches: list[MotionEvent],
    front_events: list[MotionEvent],
    baseline: np.ndarray,
    times: np.ndarray,
    event_records: list[dict],
    front_seg: str,
    rear_seg: str,
) -> list[dict]:
    """后手为组合计次锚，仅引用实际事件；缺前手不推造阶段或评分起点。"""
    records = []
    identities = {
        (record["segment"], record["start"]): record["eventId"] for record in event_records
    }
    previous_end = int(baseline[-1])
    for index, punch in enumerate(punches, 1):
        paired = preceding_punch(front_events, previous_end + 1, punch.start, rear_peak=punch.peak)
        front_id = None if paired is None else identities[(front_seg, float(times[paired.start]))]
        records.append(
            {
                "index": index,
                "start": float(times[paired.start if paired is not None else punch.start]),
                "frontPeak": None if paired is None else float(times[paired.peak]),
                "rearPeak": float(times[punch.peak]),
                "end": float(times[punch.end]),
                "ending": punch.ending,
                "frontEventId": front_id,
                "rearEventId": identities[(rear_seg, float(times[punch.start]))],
                "missingSegments": [front_seg] if paired is None else [],
            }
        )
        previous_end = punch.end
    return records


def step_footwork_windows(
    repetitions: list[dict],
    times: np.ndarray,
    diagnostics: dict,
) -> dict[tuple[str, str], np.ndarray]:
    """为每次上步组合生成共享步法窗口，向后留足余量覆盖落脚滞后且不越入相邻次。"""
    if not repetitions or len(times) == 0:
        return {}
    parts: list[np.ndarray] = []
    instances = diagnostics["phaseInstances"]
    last_idx = len(times) - 1
    for idx, rep in enumerate(repetitions):
        prev_end = repetitions[idx - 1]["end"] if idx > 0 else float(times[0])
        next_start = (
            repetitions[idx + 1]["start"] if idx + 1 < len(repetitions) else float(times[-1])
        )
        start_t = max(prev_end, float(rep["start"]) - STEP_PRE_LEAD_SECONDS)
        end_t = min(
            next_start,
            max(float(rep["end"]), float(rep["rearPeak"]) + STEP_POST_TAIL_SECONDS),
        )
        lo = min(last_idx, max(0, int(np.searchsorted(times, start_t, side="left"))))
        hi = min(last_idx, max(lo, int(np.searchsorted(times, end_t, side="right")) - 1))
        indices = np.arange(lo, hi + 1)
        if indices.size:
            parts.append(indices)
            instances.append(
                phase_instance("step_footwork", "motion", int(rep["index"]), indices, times)
            )
    if not parts:
        return {}
    return {("step_footwork", "motion"): np.unique(np.concatenate(parts))}


def step_combo_windows(
    series: dict,
    stance: str,
    cfg,
    diagnostics: dict,
    prepared: MotionInput | None = None,
    *,
    action: str = "step_straight_combo",
) -> dict:
    """全部基础事件先产生检查阶段，再登记组合引用、衔接窗口与共享步法窗口。"""
    diagnostics.update(
        reason="没有找到持续稳定的起始护位",
        events=[],
        eventDetails=[],
        phaseInstances=[],
        comboPairs=[],
        repetitions=[],
        expectedRepetitions=EXPECTED_REPETITIONS,
    )
    lm, mask = series["landmarks"], series["valid_mask"]
    prepared = prepared or prepare_motion_input(series, cfg)
    times, torso, stable = prepared.times, prepared.torso, prepared.stable
    if not stable:
        return {}
    baseline = stable[0]
    windows = {("stance", "start"): baseline, ("stance", "end"): stable[-1]}
    front_seg, rear_seg = motion_segments(action)
    observer = observe_hook if front_seg.endswith("_hook") else observe_straight
    rear = observer(lm, mask, times, torso, baseline, stable, "rear", stance, cfg)
    front = observer(lm, mask, times, torso, baseline, stable, "front", stance, cfg)
    punches = [event for event in rear.events if event.start > baseline[-1]]
    front_events = [event for event in front.events if event.start > baseline[-1]]
    observed = [ActionEvent(rear_seg, event) for event in punches]
    observed += [ActionEvent(front_seg, event) for event in front_events]
    windows.update(observed_phase_windows(observed, times, stable, cfg, diagnostics))
    windows.update(
        combo_transition_windows(observed, baseline, times, front_seg, rear_seg, cfg, diagnostics)
    )
    diagnostics["repetitions"] = _combination_records(
        punches, front_events, baseline, times, diagnostics["eventDetails"], front_seg, rear_seg
    )
    windows.update(step_footwork_windows(diagnostics["repetitions"], times, diagnostics))
    referenced = {
        record[key]
        for record in diagnostics["repetitions"]
        for key in ("frontEventId", "rearEventId")
        if record[key] is not None
    }
    diagnostics["unpairedEventIds"] = [
        event["eventId"]
        for event in diagnostics["eventDetails"]
        if event["eventId"] not in referenced
    ]
    if observed:
        after = final_stance_window(observed, stable)
        if after is not None:
            windows[("stance", "end")] = after
        else:
            windows.pop(("stance", "end"), None)
    count = len(punches)
    front_label = SEGMENT_LABELS.get(front_seg, "前手拳")
    rear_label = SEGMENT_LABELS.get(rear_seg, "后手拳")
    if not punches:
        diagnostics["reason"] = rear.reason or f"未检测到{rear_label}"
        if front_events:
            diagnostics["reason"] += "；已观察到的前手仍独立检查"
    else:
        diagnostics["reason"] = (
            "已检测到完整动作"
            if count == EXPECTED_REPETITIONS
            else f"检测到 {count} 次动作（应为 {EXPECTED_REPETITIONS} 次），按检测到的次数判定"
        )
    if any(record["missingSegments"] for record in diagnostics["repetitions"]):
        diagnostics["reason"] += f"；组合缺少可配对的{front_label}，已观察到的动作段仍独立检查"
    if diagnostics["incompleteRecovery"]:
        diagnostics["reason"] += "；存在打出后保持、未观察到收回的动作"
    return windows
