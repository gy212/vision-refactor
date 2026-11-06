# -*- coding: utf-8 -*-
"""规则判定的相位窗口（start/motion/finish/end）、正侧双视角对齐与同步相位时间共享（原 ``feedback_phases``）。

考试、学生练习与离线处理的规则都按这里的窗口取帧（D-3）；保留原始点及无效帧，不影响模板评分链路。
"""

from __future__ import annotations

import math

import numpy as np

from core.evaluation.action_events import action_events
from core.evaluation.action_catalog import action_segments
from core.evaluation.stance_windows import stable_guard_windows
from core.evaluation.phase_instances import phase_instance
from core.foundation.geometry import torso_length
from core.foundation.signal import contiguous_runs


def phase_windows(series: dict, action: str, stance: str, cfg, diagnostics=None) -> dict:
    """单路序列的相位窗口 ``{(段, 相位): 帧下标}``；``diagnostics`` 写入检测结论与事件序列。"""
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.update(
        reason="没有找到持续稳定的起始护位", events=[], eventDetails=[], phaseInstances=[]
    )
    lm, mask = series["landmarks"], series["valid_mask"]
    times = np.asarray(series["times"], dtype=float)
    n = len(lm)
    if n < cfg.min_valid * 2 + 1:
        return {}
    dt = float(np.median(np.diff(times)))
    if not math.isfinite(dt) or dt <= 0:
        return {}
    edge = max(cfg.min_valid, int(math.ceil(cfg.stance_window_seconds / dt)))
    torso = torso_length(lm, mask, "visible_side_chain")
    stable = stable_guard_windows(lm, mask, torso, edge, cfg)
    if not stable:
        return {}
    baseline = stable[0]
    diagnostics["reason"] = "未检测到完整离开并回收的动作"
    windows = {("stance", "start"): baseline, ("stance", "end"): stable[-1]}
    segments = list(action_segments(action))
    observed = action_events(lm, mask, times, torso, baseline, tuple(segments), stance, cfg)
    diagnostics["eventDetails"] = [event.to_record(times) for event in observed]
    events = [(e.segment, e.motion.start, e.motion.peak, e.motion.end) for e in observed]
    diagnostics["events"] = [e[0] for e in events]
    diagnostics["incompleteRecovery"] = sum(e.motion.ending == "held" for e in observed)
    if not events:
        return windows
    # 多次完整前后组合逐轮保留；错序/缺失不强行拼成一个动作。
    if len(segments) > 1:
        expected = segments * (len(events) // len(segments))
        if [e[0] for e in events] != expected:
            diagnostics["reason"] = "检测到的前后手次数或顺序不完整"
            return windows
    diagnostics["reason"] = "已检测到完整动作"
    for segment in segments:
        selected = [e for e in events if e[0] == segment]
        if not selected:
            continue
        windows[(segment, "motion")] = np.unique(
            np.concatenate([np.arange(s, e + 1) for _, s, _, e in selected])
        )
        peaks = []
        for index, (_, start, peak, end) in enumerate(selected, 1):
            diagnostics["phaseInstances"].append(
                phase_instance(segment, "motion", index, np.arange(start, end + 1), times)
            )
            radius = max(1, int(round(cfg.finish_window_seconds / dt)))
            indices = np.arange(max(start, peak - radius), min(end, peak + radius) + 1)
            if len(indices) >= cfg.min_valid:
                peaks.append(indices)
                diagnostics["phaseInstances"].append(
                    phase_instance(segment, "finish", index, indices, times)
                )
        if peaks:
            windows[(segment, "finish")] = np.unique(np.concatenate(peaks))
    before = [ix for ix in stable if ix[-1] <= events[0][1]]
    after = [ix for ix in stable if ix[0] >= events[-1][3]]
    if before:
        windows[("stance", "start")] = before[-1]
    if after and observed[-1].motion.ending == "returned":
        windows[("stance", "end")] = after[0]
    else:
        windows.pop(("stance", "end"), None)
    if diagnostics["incompleteRecovery"]:
        diagnostics["reason"] = "已检测到出拳；存在打出后保持、尚未观察到收回的动作"
    return windows
