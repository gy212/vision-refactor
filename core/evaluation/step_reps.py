# -*- coding: utf-8 -*-
"""上步组合拳的逐次切分：按侧面近侧手（后手）直拳的前伸数出每一次，向前找前手直拳，给出各动作段的相位窗口（D-25 部分）。

正面看直拳朝向镜头，二维图像里手腕几乎不动；侧面机位在考生右侧（D-19），左式的前手在远侧、出拳时常被身体挡住，
后手在近侧、相对更可见。所以每一次以后手出拳为锚：后手前伸（手腕相对同侧肩、沿面向方向、按躯干长归一）超过起始
护位 ``REAR_REACH``，再回落到 ``RETURN_REACH`` 或在末尾持续停住时保留事件；向下放手不算。前手直拳取这次后手出拳之前 ``FRONT_LOOKBACK_SECONDS``
内前手前伸（超过 ``FRONT_REACH``）的峰值；若此前存在唯一完整闭合前手事件，优先配对，避免慢速示范停顿超出回看范围。
这一次从前手离开护位算起；前手看不清时这一次只有前手动作过程、
没有前手出拳瞬间。阈值取自第二批样本（标定报告
``docs/上步直拳组合标定报告.md``）：A、B 类 6 段都切出 2 次。正面共用侧面的时间段（同步录制，D-18）。
"""

from __future__ import annotations

import math

import numpy as np

from core.evaluation.action_events import ActionEvent
from core.evaluation.stance_windows import stable_guard_windows
from core.evaluation.punch_events import (
    FRONT_REACH,
    RETURN_REACH,
    observe_straight,
    preceding_punch,
)
from core.evaluation.phase_instances import phase_instance
from core.evaluation.punch_phases import punch_detail_windows
from core.foundation.geometry import torso_length

EXPECTED_REPETITIONS = 2  # 考试每个动作做 2 次（#32 汇总一）
FRONT_LOOKBACK_SECONDS = 1.0  # 前手直拳在后手出拳前这段时间内找


def _concat(parts: list[np.ndarray]) -> np.ndarray | None:
    parts = [p for p in parts if len(p)]
    return np.unique(np.concatenate(parts)) if parts else None


def _combination_windows(
    punches, front_events, front, front_base, baseline, times, dt, cfg, diagnostics
):
    """把已识别的前后手事件配成上步组合，并保留每次组合的阶段。"""
    windows = {}
    radius = max(1, int(round(cfg.finish_window_seconds / dt)))
    parts: dict[tuple[str, str], list[np.ndarray]] = {}
    previous_end = int(baseline[-1])
    for punch in punches:
        start, peak, end = punch.start, punch.peak, punch.end
        floor = previous_end + 1
        rep_start = max(floor, int(np.searchsorted(times, times[start] - FRONT_LOOKBACK_SECONDS)))
        # 前手峰值只在后手出拳之前找：后手伸出后远侧的前手常被认成后手，会冒出假峰值。
        lookback = np.arange(rep_start, start + 1)
        lookback = lookback[~np.isnan(front[lookback])]
        paired = preceding_punch(front_events, floor, start)
        front_end = paired.end if paired is not None else start
        front_peak = paired.peak if paired is not None else None
        if (
            paired is None
            and len(lookback)
            and not math.isnan(front_base)
            and front[lookback].max() - front_base > FRONT_REACH
        ):
            front_peak = int(lookback[np.argmax(front[lookback])])
        if front_peak is not None:
            # 起点往回找到前手离开护位的时刻；示范放慢、前手停留较久时可以早于回看范围。
            rep_start = paired.start if paired is not None else front_peak
            if paired is None:
                while rep_start > floor and front[rep_start - 1] >= front_base + RETURN_REACH:
                    rep_start -= 1
            parts.setdefault(("front_straight", "finish"), []).append(
                np.arange(
                    max(rep_start, front_peak - radius), min(front_end, front_peak + radius) + 1
                )
            )
        parts.setdefault(("front_straight", "motion"), []).append(
            np.arange(rep_start, front_end + 1)
        )
        parts.setdefault(("rear_straight", "motion"), []).append(np.arange(start, end + 1))
        parts.setdefault(("rear_straight", "finish"), []).append(
            np.arange(max(start, peak - radius), min(end, peak + radius) + 1)
        )
        diagnostics["repetitions"].append(
            {
                "index": len(diagnostics["repetitions"]) + 1,
                "start": float(times[rep_start]),
                "frontPeak": None if front_peak is None else float(times[front_peak]),
                "rearPeak": float(times[peak]),
                "end": float(times[end]),
                "ending": punch.ending,
            }
        )
        diagnostics["events"] += ["front_straight", "rear_straight"]
        for (segment, phase), chunks in parts.items():
            if segment == "front_straight" and phase == "finish" and front_peak is None:
                continue
            diagnostics["phaseInstances"].append(
                phase_instance(segment, phase, len(diagnostics["repetitions"]), chunks[-1], times)
            )
        previous_end = end
    for key, chunks in parts.items():
        indices = _concat(chunks)
        if indices is not None and len(indices) >= cfg.min_valid:
            windows[key] = indices
    return windows


def step_combo_windows(series: dict, stance: str, cfg, diagnostics: dict) -> dict:
    """侧面一路的相位窗口 ``{(段, 相位): 帧下标}``；``diagnostics`` 写入结论、事件与每一次的时间（``repetitions``）。"""
    diagnostics.update(
        reason="没有找到持续稳定的起始护位",
        events=[],
        eventDetails=[],
        phaseInstances=[],
        repetitions=[],
        expectedRepetitions=EXPECTED_REPETITIONS,
    )
    lm, mask = series["landmarks"], series["valid_mask"]
    times = np.asarray(series["times"], dtype=float)
    if len(lm) < cfg.min_valid * 2 + 1:
        return {}
    dt = float(np.median(np.diff(times)))
    if not math.isfinite(dt) or dt <= 0:
        return {}
    torso = torso_length(lm, mask, "visible_side_chain")
    stable = stable_guard_windows(
        lm, mask, torso, max(cfg.min_valid, int(math.ceil(cfg.stance_window_seconds / dt))), cfg
    )
    if not stable:
        return {}
    baseline = stable[0]
    # 预摆停住后也可能形成稳定窗；整段固定用最初护位，不能把错误姿势重置为基线。
    windows = {("stance", "start"): baseline, ("stance", "end"): stable[-1]}
    rear_observation = observe_straight(
        lm, mask, times, torso, baseline, stable, "rear", stance, cfg
    )
    front_observation = observe_straight(
        lm, mask, times, torso, baseline, stable, "front", stance, cfg
    )
    if math.isnan(rear_observation.baseline):
        diagnostics["reason"] = rear_observation.reason or "起始护位看不清后手，无法确定出拳方向"
        return windows
    front, front_base = front_observation.reach, front_observation.baseline
    front_events = front_observation.events
    punches = [p for p in rear_observation.events if p.start > baseline[-1]]
    observed = [ActionEvent("rear_straight", event) for event in punches]
    observed += [
        ActionEvent("front_straight", event) for event in front_events if event.start > baseline[-1]
    ]
    diagnostics["eventDetails"] = [
        event.to_record(times) for event in sorted(observed, key=lambda item: item.motion.peak)
    ]
    if not punches:
        diagnostics["reason"] = "未检测到后手直拳"
        return windows
    combined = _combination_windows(
        punches, front_events, front, front_base, baseline, times, dt, cfg, diagnostics
    )
    windows.update(combined)
    after = [ix for ix in stable if ix[0] >= punches[-1].end]
    if after:
        windows[("stance", "end")] = after[0]
    else:
        windows.pop(("stance", "end"), None)
    windows.update(
        punch_detail_windows(
            times,
            punches,
            stable,
            segment="rear_straight",
            phase_instances=diagnostics["phaseInstances"],
        )
    )
    held = [p for p in punches if p.ending == "held"]
    if held:
        diagnostics["incompleteRecovery"] = len(held)
    count = len(punches)
    diagnostics["reason"] = (
        "已检测到完整动作"
        if count == EXPECTED_REPETITIONS
        else f"检测到 {count} 次动作（应为 {EXPECTED_REPETITIONS} 次），按检测到的次数判定"
    )
    if held:
        diagnostics["reason"] = f"检测到 {count} 次出拳；存在打出后保持、未观察到收回的动作"
    return windows
