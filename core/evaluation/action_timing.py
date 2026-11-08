"""为同步录像选择一次动作识别的来源，并共享其次数与阶段时间。"""

from __future__ import annotations

from collections import Counter

from core.evaluation.action_catalog import ActionDefinition, action_definition
from core.evaluation.action_events import motion_landmarks
from core.evaluation.motion_input import MotionInput, prepare_motion_input
from core.evaluation.punch_events import straight_reference
from core.evaluation.phase_alignment import phase_correspondence, project_windows
from core.evaluation.phases import phase_windows
from core.evaluation.rules.feedback_config import FeedbackConfig
from core.evaluation.stance_windows import settled_stance_windows
from core.evaluation.step_reps import step_combo_windows

NEEDS_SIDE = "上步组合按侧面视频确定动作阶段，缺少侧面视频"


def _segment_reference(series, prepared, segment, stance, view, cfg) -> dict:
    """记录识别所需的起始参照，不检测动作或读取姿势结论。"""
    points = motion_landmarks(segment, stance)
    if not prepared.stable:
        return {
            "ready": False,
            "motionLandmarks": list(points),
            "reason": "没有找到持续稳定的起始护位",
        }
    baseline = prepared.stable[0]
    if segment.endswith("_straight"):
        role = "front" if segment.startswith("front_") else "rear"
        reference = straight_reference(
            series["landmarks"],
            series["valid_mask"],
            prepared.torso,
            baseline,
            role,
            stance,
            view,
            cfg,
        )
        return {
            "ready": not reference.reason,
            "motionLandmarks": list(reference.motion_landmarks),
            "baselineFrames": len(baseline),
            "motionReferenceFrames": reference.motion_frames,
            "minimumMotionReferenceFrames": cfg.min_valid if view == "front" else 1,
            "directionReferenceFrames": reference.direction_frames,
            "directionLandmarks": [0, 23, 24] if view == "side" else [],
            "minimumDirectionReferenceFrames": cfg.min_valid if view == "side" else 0,
            "directionRequirement": "鼻点及至少一个髋点" if view == "side" else "无需方向参照",
            "reason": reference.reason,
        }
    valid = series["valid_mask"][baseline][:, points].all(axis=1)
    valid &= prepared.torso[baseline] > 1e-4
    return {
        "ready": bool(valid.all()),
        "motionLandmarks": list(points),
        "baselineFrames": len(baseline),
        "motionReferenceFrames": int(valid.sum()),
        "reason": "" if valid.all() else "起始护位动作点不足，无法确定动作位移",
    }


def _source_coverage(series, prepared, definition, stance, view, cfg) -> dict:
    """可见帧与必要参照分别声明，配对缺成员不抹掉仍可观测的独立段。"""
    mask = series["valid_mask"]
    usable = prepared.torso > 1e-4
    for segment in definition.segments:
        usable &= mask[:, motion_landmarks(segment, stance)].all(axis=1)
    if not definition.segments:
        usable &= mask[:, [11, 15, 23]].all(axis=1) | mask[:, [12, 16, 24]].all(axis=1)
    references = {
        segment: _segment_reference(series, prepared, segment, stance, view, cfg)
        for segment in definition.segments
    }
    ready = sum(item["ready"] for item in references.values())
    return {
        "validFrames": int(usable.sum()),
        "validRatio": float(usable.mean()) if len(usable) else 0.0,
        "referenceReady": bool(prepared.stable) and ready == len(references),
        "readySegments": ready,
        "references": references,
    }


def timing_source(
    views: dict,
    definition: ActionDefinition,
    stance: str,
    cfg: FeedbackConfig,
    prepared: dict[str, MotionInput],
) -> tuple[str | None, dict]:
    """先核对可见性和识别的真实参照条件，再选一路执行一次识别。"""
    candidates = [view for view in definition.timing_views if view in views]
    coverage = {
        view: _source_coverage(views[view], prepared[view], definition, stance, view, cfg)
        for view in candidates
    }
    executable = [view for view in candidates if coverage[view]["referenceReady"]]
    usable = [
        view
        for view in executable
        if coverage[view]["validFrames"] >= cfg.min_valid
        and coverage[view]["validRatio"] >= cfg.valid_ratio
    ]
    source = (
        usable[0]
        if usable
        else max(
            executable or candidates,
            key=lambda view: (
                coverage[view]["readySegments"],
                coverage[view]["validFrames"],
                coverage[view]["validRatio"],
            ),
            default=None,
        )
    )
    return source, {
        "viewPriority": list(definition.timing_views),
        "selectedView": source,
        "minimumFrames": cfg.min_valid,
        "minimumRatio": cfg.valid_ratio,
        "motionLandmarks": {
            segment: list(motion_landmarks(segment, stance)) for segment in definition.segments
        },
        "bodyReference": "至少一条有效肩髋链",
        "visibility": coverage,
        "reason": (
            "采用满足动作点可见性及起始参照条件的首选机位；仅该机位执行动作识别"
            if usable
            else "按起始参照和动作点可用性选择合法机位；仅执行一次识别，缺失的独立段保留原因"
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
    prepared = {view: prepare_motion_input(series, cfg) for view, series in views.items()}
    source, selection = timing_source(views, definition, stance, cfg, prepared)
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
        source_windows = step_combo_windows(views[source], stance, cfg, data, prepared[source])
    else:
        source_windows = phase_windows(
            views[source], action, stance, cfg, data, source, prepared[source]
        )
    data["timingSource"] = "own_view"
    windows, diagnostics, sources = {}, {}, {}
    for view, series in views.items():
        windows[view] = (
            source_windows
            if view == source
            else project_windows(
                source_windows, views[source]["times"], series["times"], data["phaseInstances"]
            )
        )
        diagnostics[view] = data if view == source else _project_diagnostics(data, source)
        sources[view] = "own_view" if view == source else source + "_synchronized_timing"
        missing = set(source_windows) - set(windows[view])
        if missing:
            diagnostics[view]["reason"] = "本机位录像未覆盖共享阶段的完整时间范围"
        boundaries = data["eventDetails"]
        windows[view] = settled_stance_windows(series, windows[view], boundaries, cfg)
    correspondence = phase_correspondence(
        views, windows, source, source_windows, data["phaseInstances"]
    )
    for view in views:
        diagnostics[view]["phaseCoverage"] = {
            scope: phase["views"][view] for scope, phase in correspondence.items()
        }
    alignment = {
        "method": "shared_clock",
        "sideOffsetSeconds": 0.0,
        "reliable": True,
        "timingSource": source,
        "sourceSelection": selection,
        "eventCounts": {
            source: dict(Counter(event["segment"] for event in data.get("eventDetails", [])))
        },
        "phases": correspondence,
    }
    return windows, diagnostics, alignment, sources
