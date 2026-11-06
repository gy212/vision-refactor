"""为同步录像选择一次动作识别的来源，并共享其次数与阶段时间。"""

from __future__ import annotations

from collections import Counter

from core.evaluation.action_catalog import ActionDefinition, action_definition
from core.evaluation.action_events import motion_landmarks
from core.evaluation.phase_alignment import phase_correspondence, project_windows
from core.evaluation.phases import phase_windows
from core.evaluation.rules.feedback_config import FeedbackConfig
from core.evaluation.stance_windows import settled_stance_windows
from core.evaluation.step_reps import step_combo_windows
from core.foundation.geometry import torso_length

NEEDS_SIDE = "上步组合按侧面视频确定动作阶段，缺少侧面视频"


def _source_coverage(series: dict, definition: ActionDefinition, stance: str) -> tuple[int, float]:
    """仅衡量动作所需点的可见覆盖，不执行事件识别或读取姿势对错。"""
    mask = series["valid_mask"]
    if not len(mask):
        return 0, 0.0
    usable = torso_length(series["landmarks"], mask, "visible_side_chain") > 1e-4
    for segment in definition.segments:
        usable &= mask[:, motion_landmarks(segment, stance)].all(axis=1)
    if not definition.segments:
        usable &= mask[:, [11, 15, 23]].all(axis=1) | mask[:, [12, 16, 24]].all(axis=1)
    return int(usable.sum()), float(usable.mean())


def timing_source(
    views: dict, definition: ActionDefinition, stance: str, cfg: FeedbackConfig
) -> tuple[str | None, dict]:
    """先按所需点可用性和动作机位策略选一路，再调用一次分段。"""
    candidates = [view for view in definition.timing_views if view in views]
    coverage = {view: _source_coverage(views[view], definition, stance) for view in candidates}
    usable = [
        view
        for view in candidates
        if coverage[view][0] >= cfg.min_valid and coverage[view][1] >= cfg.valid_ratio
    ]
    source = usable[0] if usable else max(candidates, key=lambda view: coverage[view], default=None)
    return source, {
        "viewPriority": list(definition.timing_views),
        "selectedView": source,
        "minimumFrames": cfg.min_valid,
        "minimumRatio": cfg.valid_ratio,
        "motionLandmarks": {
            segment: list(motion_landmarks(segment, stance)) for segment in definition.segments
        },
        "bodyReference": "至少一条有效肩髋链",
        "visibility": {
            view: {"validFrames": count, "validRatio": ratio}
            for view, (count, ratio) in coverage.items()
        },
        "reason": (
            "采用达到所需点可见要求的首选机位；仅该机位执行动作识别"
            if usable
            else "没有机位达到完整可见要求，使用可用点最多的机位尝试一次识别；未找到的阶段保留缺失"
        ),
    }


def _project_diagnostics(data: dict, source: str) -> dict:
    """共享事件及阶段，目标机位不冒充独立识别了另一份次数。"""
    return {
        **data,
        "events": [],
        "eventDetails": [],
        "phaseInstances": [],
        "projectedEventDetails": data.get("eventDetails", []),
        "projectedPhaseInstances": data.get("phaseInstances", []),
        "timingSource": source,
    }


def shared_timeline(
    views: dict, action: str, stance: str, cfg: FeedbackConfig
) -> tuple[dict, dict, dict, dict]:
    """输入已同步的时间轴，生成所有入口共用的一份动作时间结果。

    时间戳是调用方提供的同一时钟；不依据两路姿势、峰值或识别数量猜偏移。
    每一路只按共有时段读取自己的有效点，数据不完整由对应检查报告。
    """
    definition = action_definition(action)
    source, selection = timing_source(views, definition, stance, cfg)
    if source is None:
        reason = NEEDS_SIDE if definition.timing == "side_combo" else "缺少动作计时机位"
        return (
            {view: {} for view in views},
            {view: {"reason": reason, "events": []} for view in views},
            {
                "method": "shared_clock",
                "sideOffsetSeconds": 0.0,
                "reliable": False,
                "reason": reason,
                "phases": {},
                "sourceSelection": selection,
            },
            {view: "own_view" for view in views},
        )
    data: dict = {}
    if definition.timing == "side_combo":
        source_windows = step_combo_windows(views[source], stance, cfg, data)
    else:
        source_windows = phase_windows(views[source], action, stance, cfg, data)
    data["timingSource"] = "own_view"
    windows, diagnostics, sources = {}, {}, {}
    for view, series in views.items():
        windows[view] = (
            source_windows
            if view == source
            else project_windows(source_windows, views[source]["times"], series["times"])
        )
        diagnostics[view] = data if view == source else _project_diagnostics(data, source)
        sources[view] = "own_view" if view == source else source + "_synchronized_timing"
        missing = set(source_windows) - set(windows[view])
        if missing:
            diagnostics[view]["reason"] = "本机位录像未覆盖共享阶段的完整时间范围"
        if definition.timing == "side_combo":
            windows[view] = settled_stance_windows(series, windows[view], data["repetitions"], cfg)
    alignment = {
        "method": "shared_clock",
        "sideOffsetSeconds": 0.0,
        "reliable": True,
        "timingSource": source,
        "sourceSelection": selection,
        "eventCounts": {
            source: dict(Counter(event["segment"] for event in data.get("eventDetails", [])))
        },
        "phases": phase_correspondence(views, windows, cfg.phase_tolerance_seconds),
    }
    return windows, diagnostics, alignment, sources
