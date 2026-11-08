"""引用已观测的前后直拳事件组成上步组合的顺序与评分区间。"""

from __future__ import annotations

import numpy as np

from core.evaluation.action_events import ActionEvent
from core.evaluation.event_phases import final_stance_window, observed_phase_windows
from core.evaluation.motion_events import MotionEvent
from core.evaluation.punch_events import observe_straight, preceding_punch
from core.evaluation.motion_input import MotionInput, prepare_motion_input

EXPECTED_REPETITIONS = 2  # 考试每个动作做 2 次（#32 汇总一）


def _combination_records(
    punches: list[MotionEvent],
    front_events: list[MotionEvent],
    baseline: np.ndarray,
    times: np.ndarray,
    event_records: list[dict],
) -> list[dict]:
    """后手为组合计次锚，仅引用实际事件；缺前手不推造阶段或评分起点。"""
    records = []
    identities = {
        (record["segment"], record["start"]): record["eventId"] for record in event_records
    }
    previous_end = int(baseline[-1])
    for index, punch in enumerate(punches, 1):
        paired = preceding_punch(front_events, previous_end + 1, punch.start)
        front_id = (
            None if paired is None else identities[("front_straight", float(times[paired.start]))]
        )
        records.append(
            {
                "index": index,
                "start": float(times[paired.start if paired is not None else punch.start]),
                "frontPeak": None if paired is None else float(times[paired.peak]),
                "rearPeak": float(times[punch.peak]),
                "end": float(times[punch.end]),
                "ending": punch.ending,
                "frontEventId": front_id,
                "rearEventId": identities[("rear_straight", float(times[punch.start]))],
                "missingSegments": ["front_straight"] if paired is None else [],
            }
        )
        previous_end = punch.end
    return records


def step_combo_windows(
    series: dict, stance: str, cfg, diagnostics: dict, prepared: MotionInput | None = None
) -> dict:
    """全部基础事件先产生检查阶段，再登记组合引用与缺失成员。"""
    diagnostics.update(
        reason="没有找到持续稳定的起始护位",
        events=[],
        eventDetails=[],
        phaseInstances=[],
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
    rear = observe_straight(lm, mask, times, torso, baseline, stable, "rear", stance, cfg)
    front = observe_straight(lm, mask, times, torso, baseline, stable, "front", stance, cfg)
    punches = [event for event in rear.events if event.start > baseline[-1]]
    front_events = [event for event in front.events if event.start > baseline[-1]]
    observed = [ActionEvent("rear_straight", event) for event in punches]
    observed += [ActionEvent("front_straight", event) for event in front_events]
    windows.update(observed_phase_windows(observed, times, stable, cfg, diagnostics))
    diagnostics["repetitions"] = _combination_records(
        punches, front_events, baseline, times, diagnostics["eventDetails"]
    )
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
    if not punches:
        diagnostics["reason"] = rear.reason or "未检测到后手直拳"
        if front_events:
            diagnostics["reason"] += "；已观察到的前手仍独立检查"
    else:
        diagnostics["reason"] = (
            "已检测到完整动作"
            if count == EXPECTED_REPETITIONS
            else f"检测到 {count} 次动作（应为 {EXPECTED_REPETITIONS} 次），按检测到的次数判定"
        )
    if any(record["missingSegments"] for record in diagnostics["repetitions"]):
        diagnostics["reason"] += "；组合缺少可配对的前手直拳，已观察到的动作段仍独立检查"
    if diagnostics["incompleteRecovery"]:
        diagnostics["reason"] += "；存在打出后保持、未观察到收回的动作"
    return windows
