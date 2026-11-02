# -*- coding: utf-8 -*-
"""上步组合拳的逐次切分：按侧面近侧手（后手）直拳的前伸数出每一次，向前找前手直拳，给出各动作段的相位窗口（D-25 部分）。

正面看直拳朝向镜头，二维图像里手腕几乎不动；侧面机位在考生右侧（D-19），左式的前手在远侧、出拳时常被身体挡住，
后手在近侧、全程可见。所以每一次以后手出拳为锚：后手前伸（手腕相对同侧肩、沿面向方向、按躯干长归一）超过起始
护位 ``REACH_START`` 再回落到 ``REACH_RETURN`` 以内算一次；前手直拳取这次后手出拳之前 ``FRONT_LOOKBACK_SECONDS``
内前手前伸（超过 ``FRONT_REACH``）的峰值，这一次从前手离开护位算起；前手看不清时这一次只有前手动作过程、
没有前手出拳瞬间。阈值取自第二批样本（标定报告
``docs/上步直拳组合标定报告.md``）：A、B 类 6 段都切出 2 次。正面共用侧面的时间段（同步录制，D-18）。
"""
from __future__ import annotations

import math

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.phases import stable_guard_windows
from core.evaluation.punch_phases import punch_detail_windows
from core.foundation.geometry import torso_length

EXPECTED_REPETITIONS = 2  # 考试每个动作做 2 次（#32 汇总一）
REACH_START = .30  # 后手前伸超过起始护位这么多（躯干长度）算出拳
REACH_RETURN = .12  # 回落到起始护位这么多以内算收回
FRONT_REACH = .25  # 前手（远侧、护位时已靠前）前伸超过起始护位这么多算出拳
FRONT_LOOKBACK_SECONDS = 1.0  # 前手直拳在后手出拳前这段时间内找
MAX_GAP_SECONDS = .25  # 出拳过程中关键点短暂缺失可以跨过的时长


def _reach(lm: np.ndarray, mask: np.ndarray, torso: np.ndarray, wrist: int, shoulder: int, facing: float) -> np.ndarray:
    """手腕相对同侧肩沿面向方向的前伸（躯干长度）；点无效的帧为 NaN。"""
    ok = mask[:, wrist] & mask[:, shoulder] & (torso > 1e-4)
    reach = (lm[:, wrist, 0] - lm[:, shoulder, 0]) * facing / np.maximum(torso, 1e-4)
    return np.where(ok, reach, np.nan)


def _excursions(
    signal: np.ndarray,
    times: np.ndarray,
    rise: float,
    fall: float,
    cfg,
) -> list[tuple[int, int, int]]:
    """前伸再回落的各段 ``(起点, 峰值, 终点)``；逐段核对最少有效帧与时长，缺失过久则作废。"""
    found, start, last = [], None, None
    for i, value in enumerate(signal):
        if np.isnan(value):
            if start is not None and last is not None and times[i] - times[last] > MAX_GAP_SECONDS:
                start = None
            continue
        if start is None and value > rise:
            start = i
        elif start is not None and value < fall:
            observed = np.arange(start, i + 1)
            observed = observed[~np.isnan(signal[observed])]
            # 先逐段过滤，不能把多个单帧跳点合起来凑够有效帧数。
            if len(observed) >= cfg.min_valid and times[i] - times[start] >= cfg.min_motion_seconds:
                peak = int(observed[np.argmax(signal[observed])])
                found.append((start, peak, i))
            start = None
        last = i
    return found


def _concat(parts: list[np.ndarray]) -> np.ndarray | None:
    parts = [p for p in parts if len(p)]
    return np.unique(np.concatenate(parts)) if parts else None


def step_combo_windows(series: dict, stance: str, cfg, diagnostics: dict) -> dict:
    """侧面一路的相位窗口 ``{(段, 相位): 帧下标}``；``diagnostics`` 写入结论、事件与每一次的时间（``repetitions``）。"""
    diagnostics.update(reason="没有找到持续稳定的起始护位", events=[], repetitions=[],
                       expectedRepetitions=EXPECTED_REPETITIONS)
    lm, mask = series["landmarks"], series["valid_mask"]
    times = np.asarray(series["times"], dtype=float)
    if len(lm) < cfg.min_valid * 2 + 1:
        return {}
    dt = float(np.median(np.diff(times)))
    if not math.isfinite(dt) or dt <= 0:
        return {}
    torso = torso_length(lm, mask, "visible_side_chain")
    stable = stable_guard_windows(lm, mask, torso, max(cfg.min_valid, int(math.ceil(cfg.stance_window_seconds / dt))), cfg)
    if not stable:
        return {}
    baseline = stable[0]
    # 预摆停住后也可能形成稳定窗；整段固定用最初护位，不能把错误姿势重置为基线。
    windows = {("stance", "start"): baseline, ("stance", "end"): stable[-1]}
    hips = (lm[baseline, 23, 0] + lm[baseline, 24, 0]) / 2
    facing = float(np.sign(np.nanmedian(lm[baseline, 0, 0] - hips)))
    front_sh, _, front_wr, *_ = sides_for_stance("front", stance)
    rear_sh, _, rear_wr, *_ = sides_for_stance("rear", stance)
    rear = _reach(lm, mask, torso, rear_wr, rear_sh, facing)
    front = _reach(lm, mask, torso, front_wr, front_sh, facing)
    if facing == 0 or np.isnan(rear[baseline]).all():
        diagnostics["reason"] = "起始护位看不清后手，无法确定出拳方向"
        return windows
    rear_base = float(np.nanmedian(rear[baseline]))
    front_base = float(np.nanmedian(front[baseline])) if not np.isnan(front[baseline]).all() else math.nan
    excursions = _excursions(rear, times, rear_base + REACH_START, rear_base + REACH_RETURN, cfg)
    punches = [p for p in excursions if p[0] > baseline[-1]]
    if not punches:
        diagnostics["reason"] = "未检测到后手直拳"
        return windows
    radius = max(1, int(round(cfg.finish_window_seconds / dt)))
    parts: dict[tuple[str, str], list[np.ndarray]] = {}
    previous_end = int(baseline[-1])
    for start, peak, end in punches:
        floor = previous_end + 1
        rep_start = max(floor, int(np.searchsorted(times, times[start] - FRONT_LOOKBACK_SECONDS)))
        # 前手峰值只在后手出拳之前找：后手伸出后远侧的前手常被认成后手，会冒出假峰值。
        lookback = np.arange(rep_start, start + 1)
        lookback = lookback[~np.isnan(front[lookback])]
        front_peak = None
        if len(lookback) and not math.isnan(front_base) and front[lookback].max() - front_base > FRONT_REACH:
            front_peak = int(lookback[np.argmax(front[lookback])])
            # 起点往回找到前手离开护位的时刻；示范放慢、前手停留较久时可以早于回看范围。
            rep_start = front_peak
            while rep_start > floor and front[rep_start - 1] >= front_base + REACH_RETURN:  # 前手看不清（NaN）即停
                rep_start -= 1
            parts.setdefault(("front_straight", "finish"), []).append(
                np.arange(max(rep_start, front_peak - radius), min(start, front_peak + radius) + 1))
        parts.setdefault(("front_straight", "motion"), []).append(np.arange(rep_start, start + 1))
        parts.setdefault(("rear_straight", "motion"), []).append(np.arange(start, end + 1))
        parts.setdefault(("rear_straight", "finish"), []).append(np.arange(max(start, peak - radius), min(end, peak + radius) + 1))
        diagnostics["repetitions"].append({
            "index": len(diagnostics["repetitions"]) + 1, "start": float(times[rep_start]),
            "frontPeak": None if front_peak is None else float(times[front_peak]),
            "rearPeak": float(times[peak]), "end": float(times[end])})
        diagnostics["events"] += ["front_straight", "rear_straight"]
        previous_end = end
    for key, chunks in parts.items():
        indices = _concat(chunks)
        if indices is not None and len(indices) >= cfg.min_valid:
            windows[key] = indices
    after = [ix for ix in stable if ix[0] >= punches[-1][2]]
    if after:
        windows[("stance", "end")] = after[0]
    else:
        windows.pop(("stance", "end"), None)
    windows.update(punch_detail_windows(times, punches, stable))
    count = len(punches)
    diagnostics["reason"] = ("已检测到完整动作" if count == EXPECTED_REPETITIONS
                             else f"检测到 {count} 次动作（应为 {EXPECTED_REPETITIONS} 次），按检测到的次数判定")
    return windows
