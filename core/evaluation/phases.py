# -*- coding: utf-8 -*-
"""从选定机位识别一次动作及其相位窗口（start/motion/finish/end）。

考试、学生练习与离线处理的规则都按这里的窗口取帧（D-3）；保留原始点及无效帧，不影响模板评分链路。
"""

from __future__ import annotations

from core.evaluation.action_events import action_events
from core.evaluation.action_catalog import action_segments
from core.evaluation.motion_input import MotionInput, prepare_motion_input
from core.evaluation.event_phases import final_stance_window, observed_phase_windows


def phase_windows(
    series: dict,
    action: str,
    stance: str,
    cfg,
    diagnostics=None,
    view: str = "side",
    prepared: MotionInput | None = None,
) -> dict:
    """单路序列的相位窗口 ``{(段, 相位): 帧下标}``；``diagnostics`` 写入检测结论与事件序列。"""
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.update(
        reason="没有找到持续稳定的起始护位", events=[], eventDetails=[], phaseInstances=[]
    )
    lm, mask = series["landmarks"], series["valid_mask"]
    prepared = prepared or prepare_motion_input(series, cfg)
    times, torso, stable = prepared.times, prepared.torso, prepared.stable
    if not stable:
        return {}
    baseline = stable[0]
    diagnostics["reason"] = "未检测到完整离开并回收的动作"
    windows = {("stance", "start"): baseline, ("stance", "end"): stable[-1]}
    segments = list(action_segments(action))
    observed = action_events(
        lm, mask, times, torso, baseline, tuple(segments), stance, cfg, stable, diagnostics, view
    )
    windows.update(observed_phase_windows(observed, times, stable, cfg, diagnostics))
    if not observed:
        return windows
    diagnostics["reason"] = "已检测到完整动作"
    # 组合顺序只影响组合说明，不抹掉已观察到的独立动作段。
    if len(segments) > 1:
        expected = segments * (len(observed) // len(segments))
        if diagnostics["events"] != expected:
            diagnostics["reason"] = "检测到的前后手次数或顺序不完整"
    after = final_stance_window(observed, stable)
    if after is not None:
        windows[("stance", "end")] = after
    else:
        windows.pop(("stance", "end"), None)
    if diagnostics["incompleteRecovery"]:
        diagnostics["reason"] = "已检测到出拳；存在打出后保持、尚未观察到收回的动作"
    return windows
