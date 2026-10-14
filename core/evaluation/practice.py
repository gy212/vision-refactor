# -*- coding: utf-8 -*-
"""学生练习分析：逐帧序列提取、相位窗口、经规则引擎的单视角判定与正侧证据融合，产出练习结果字典（设计文档 §7.8）。

复刻 v4（原 ``feedback_geometry.extract_series``/``evaluate_series``/``analyze_geometry``）：丢失帧全无效，
绝不复用上一帧；结论只是待教师确认的问题说明，``scoreAuthorized=False``。
"""
from __future__ import annotations

from dataclasses import asdict
import time
from pathlib import Path
from typing import Callable

import numpy as np

from core.evaluation.phases import align_views, phase_windows, share_phase_timing
from core.evaluation.rules.adapters import to_practice_check
from core.evaluation.rules.engine import evaluate_rules
from core.evaluation.rules.practice_v4 import (
    RULE_VERSION,
    FeedbackConfig,
    build_checks,
    check_view_weights,
    load_feedback_config,
    practice_profile,
)
from core.evaluation.rules.spec import RuleContext, RuleOutcome, RuleProfile
from core.foundation.geometry import torso_length
from core.foundation.paths import models_dir
from core.foundation.signal import contiguous_runs
from core.foundation.tasks import Cancelled, from_stopped
from core.media.video_source import UndecodableFrame, VideoFpsUnavailable, VideoOpenError
from core.pose.engine import create_engine
from core.pose.extract import extract_pose_sequence, practice_engine, practice_v4_profile
from core.pose.sequence import Coords, GapPolicy
from core.pose.validity import derive_valid_mask

_VIEW_LABELS = {"front": "正面", "side": "侧面"}


def create_practice_engine(cfg: FeedbackConfig | None = None):
    """练习分析用的 MediaPipe 引擎（不联网下载模型）。"""
    cfg = cfg or FeedbackConfig()
    return create_engine(practice_engine(cfg.pose_variant), models_dir=models_dir(), download=False)


def extract_series(path: Path, cfg: FeedbackConfig, stopped: Callable[[], bool], pipeline_factory=None) -> dict:
    """默认原帧率逐帧推理，只保留关键点。丢失帧全无效，绝不复用上一帧。

    ``PRACTICE_V4`` 提取（长边上限、抽帧取自 ``cfg``）+ ``raw(ZERO, ASPECT_CORRECTED)``：x、z 统一为图像高度单位。
    """
    started = time.monotonic()
    profile = practice_v4_profile(sample_fps=cfg.sample_fps, max_edge=cfg.max_edge)
    try:
        seq = extract_pose_sequence(
            path, practice_engine(cfg.pose_variant), profile, cancel=from_stopped(stopped),
            engine_factory=pipeline_factory or (lambda: create_practice_engine(cfg)),
        )
    except VideoOpenError:
        raise ValueError(f"视频无法读取，请重新录制：{path.name}") from None
    except VideoFpsUnavailable:
        raise ValueError("视频帧率不可用，请重新录制") from None
    except UndecodableFrame:
        raise ValueError("视频存在不可解码的帧，请重新录制") from None
    except Cancelled:
        raise InterruptedError("分析已取消") from None
    if len(seq) == 0:
        raise ValueError("视频没有可读取的帧，请重新录制")
    lm = seq.raw(GapPolicy.ZERO, Coords.ASPECT_CORRECTED).copy()
    w, h = (int(v) for v in seq.source_wh[-1])
    fps = seq.meta.fps_reported
    frames = [int(i) for i in seq.frame_index]
    mask = derive_valid_mask(lm) & np.isfinite(lm[:, :, :3]).all(axis=2)
    mask &= (lm[:, :, 0] >= 0) & (lm[:, :, 0] <= w / h) & (lm[:, :, 1] >= 0) & (lm[:, :, 1] <= 1)
    inference_w, inference_h = seq.meta.inference_size
    return {"landmarks": lm, "valid_mask": mask, "frames": frames, "times": [i / fps for i in frames],
            "capture": {"sourceFps": fps, "sourceFrames": seq.meta.frames_scanned, "analyzedFrames": len(frames),
                        "stride": seq.meta.stride, "width": w, "height": h,
                        "inferenceWidth": inference_w, "inferenceHeight": inference_h,
                        "analysisSeconds": round(time.monotonic() - started, 3)}}


def _observe_view(c: dict, view: str, series: dict, view_windows: dict, view_diagnostics: dict, alignment: dict,
                  weights: dict, cfg: FeedbackConfig, evidence: dict, outcome: RuleOutcome | None):
    """一路视角对一个检查项的观测：``(该阶段时间段或 None, (是否有问题, 证据权重) 或 None)``。

    ``outcome`` 是 ``evaluate_rules`` 对该项在本视角的结论；该项不适用本视角时为 None。
    """
    indices = view_windows.get((c["segment"], c["phase"]))
    if indices is None or len(indices) < cfg.min_valid:
        c["viewProblems"][view] = "未找到该阶段；" + view_diagnostics.get("reason", "阶段不足")
        return None, None
    # 共用的只能是同步时间段；每路仍须独立提供有效人体和规则所需的点。
    ranges = None
    body_valid = torso_length(series["landmarks"][indices], series["valid_mask"][indices], "visible_side_chain") > 1e-4
    if body_valid.mean() >= cfg.valid_ratio and body_valid.sum() >= cfg.min_valid:
        offset = alignment["sideOffsetSeconds"] if view == "side" else 0
        ranges = [(float(series["times"][run[0]]) + offset, float(series["times"][run[-1]]) + offset)
                  for run in contiguous_runs(indices)]
    else:
        c["viewProblems"][view] = f"该阶段可见肩髋链不足（{body_valid.sum()}/{len(indices)}帧）"
    if outcome is None:  # 该项只适用另一视角的公式（基础权重为 0），本视角只核对阶段
        return ranges, None
    return ranges, to_practice_check(c, view, outcome, weights[view], series, evidence)


def _fuse(c: dict, views: dict, ranges: dict, outcomes: dict, weights: dict, alignment: dict, cfg: FeedbackConfig) -> None:
    """正侧观测合成一个检查项的状态与原因；任一前提不满足时保持 ``unable`` 并写明原因。"""
    if set(views) != {"front", "side"}:
        c["reason"] = "缺少正面或侧面视频；仅保留单视角观测，不能给出联合结论"
        return
    if set(ranges) != {"front", "side"}:
        c["reason"] = "；".join(_VIEW_LABELS[v] + "：" + c["viewProblems"].get(v, "阶段不足")
                              for v in ("front", "side") if v not in ranges)
        return
    if not alignment["reliable"]:
        c["reason"] = alignment.get("reason", "正侧面时间无法对应")
        return
    if len(ranges["front"]) != len(ranges["side"]) or any(
        max(abs(a - b) for a, b in zip(front, side)) > cfg.phase_tolerance_seconds
        for front, side in zip(ranges["front"], ranges["side"])
    ):
        c["reason"] = "正侧面动作阶段时间不对应，请使用同一次同步录制的视频"
        return
    if any(weight > 0 and view not in outcomes for view, weight in weights.items()):
        c["reason"] = "；".join(_VIEW_LABELS[v] + "：" + problem for v, problem in c["viewProblems"].items())
        return
    total = sum(value[1] for value in outcomes.values())
    effective = {view: value[1] / total for view, value in outcomes.items()}
    support = sum(effective[view] * value[0] for view, value in outcomes.items())
    c["fusion"].update(effectiveWeights=effective, support=float(support))
    if support >= cfg.fusion_support - 1e-9:
        c["status"] = "candidate"
    elif support <= 1 - cfg.fusion_support + 1e-9:
        c["status"] = "not_observed"
    else:
        c["reason"] = "正侧加权证据接近，暂不能确定，请教师复核"
        return
    c["reason"] = ("按动作与检查项权重、有效帧比例融合；权重待标定"
                   if c["fusion"]["method"] == "weighted_support" else
                   "该公式仅适用" + ("正面" if weights["front"] else "侧面") + "，另一视角只核对阶段；不作为双视角一致结论")


def _evaluate_check(c: dict, views: dict, windows: dict, diagnostics: dict, alignment: dict, outcomes_by_view: dict,
                    cfg: FeedbackConfig, evidence: dict) -> None:
    c.update(status="unable", reason="关键点、视角或动作阶段不足，请重新录制", evidence=[], review="pending")
    if c["blockedReason"]:
        c.update(status="pending_rule", reason=c["blockedReason"])
        return
    weights = check_view_weights(c, cfg)
    c["fusion"] = {"method": "weighted_support" if all(weights.values()) else "primary_view",
                   "baseWeights": weights, "effectiveWeights": {}, "support": None,
                   "threshold": cfg.fusion_support, "calibrationStatus": "unvalidated"}
    c["measurements"] = []
    c["viewProblems"] = {}
    outcomes = {}
    ranges = {}
    for view in ("front", "side"):
        if view not in views:
            continue
        view_ranges, outcome = _observe_view(c, view, views[view], windows[view], diagnostics[view], alignment,
                                             weights, cfg, evidence, outcomes_by_view[view].get(c["id"]))
        if view_ranges is not None:
            ranges[view] = view_ranges
        if outcome is not None:
            outcomes[view] = outcome
    _fuse(c, views, ranges, outcomes, weights, alignment, cfg)


def _view_outcomes(view: str, series: dict, view_windows: dict, profile: RuleProfile, action: str, stance: str,
                   cfg: FeedbackConfig) -> dict[str, RuleOutcome]:
    """一路视角上全部检查项经 ``evaluate_rules`` 的结论（以检查项 id 为键）；基线取起始实战式窗口，缺失时取开头几帧。"""
    baseline = view_windows.get(("stance", "start"), np.arange(min(cfg.min_valid, len(series["times"]))))
    ctx = RuleContext(view=view, action_scope=action, stance=stance,
                      phase_windows={f"{segment}:{phase}": idx for (segment, phase), idx in view_windows.items()},
                      baseline=baseline, params=cfg)
    outcomes = evaluate_rules(series["landmarks"], series["valid_mask"], profile, view=view, action_scope="both",
                              context=ctx)
    return {o.spec.code: o for o in outcomes}


def _view_diagnostics(views: dict, diagnostics: dict, phase_sources: dict) -> dict:
    return {view: {**diagnostics[view], "phaseSource": phase_sources[view],
                   "bodyValidFrames": int((torso_length(s["landmarks"], s["valid_mask"], "visible_side_chain") > 1e-4).sum()),
                   "totalFrames": len(s["times"]),
                   "jointValidRatios": s["valid_mask"].mean(axis=0).tolist()}
            for view, s in views.items()}


def _phase_ranges(views: dict, windows: dict) -> dict:
    return {view: [{"segment": segment, "phase": phase,
                    "ranges": [[float(series["times"][run[0]]), float(series["times"][run[-1]])]
                               for run in contiguous_runs(indices)]}
                   for (segment, phase), indices in windows[view].items()]
            for view, series in views.items()}


def evaluate_series(action: str, stance: str, views: dict, cfg: FeedbackConfig | None = None) -> dict:
    """``views``（``{"front"/"side": extract_series 的结果}``）上的练习结果字典（``schemaVersion=1``）。"""
    cfg = cfg or FeedbackConfig()
    checks = build_checks(action, stance)
    diagnostics = {view: {} for view in views}
    windows = {view: phase_windows(s, action, stance, cfg, diagnostics[view]) for view, s in views.items()}
    alignment = align_views(views, windows, cfg)
    phase_sources = share_phase_timing(views, windows, alignment, diagnostics, cfg)
    profile = practice_profile(action, stance, cfg)
    outcomes_by_view = {view: _view_outcomes(view, s, windows[view], profile, action, stance, cfg)
                        for view, s in views.items()}
    evidence = {}
    for c in checks:
        _evaluate_check(c, views, windows, diagnostics, alignment, outcomes_by_view, cfg, evidence)
    unable = any(c["status"] == "unable" for c in checks)
    issues = any(c["status"] == "candidate" for c in checks)
    return {"schemaVersion": 1, "ruleVersion": RULE_VERSION, "action": action, "stance": stance,
            "backend": "mediapipe", "poseVariant": cfg.pose_variant, "delegate": "cpu", "actionSource": "user_selected",
            "fusionMethod": "action_rule_quality_weighted_v1", "inputViews": sorted(views),
            "capture": {view: series.get("capture", {}) for view, series in views.items()}, "alignment": alignment,
            "diagnostics": _view_diagnostics(views, diagnostics, phase_sources),
            "phaseWindows": _phase_ranges(views, windows),
            "config": asdict(cfg), "status": "partial" if any(c["status"] in {"unable", "pending_rule"} for c in checks) else "completed",
            "needsRerecord": unable, "scoreAuthorized": False, "calibrationStatus": "unvalidated",
            "summary": "规则检查发现待教师确认的问题" if issues else "证据不足，请重新录制" if unable else "已检查项目未发现明显问题，不代表全部项目合格",
            "checks": checks, "evidence": list(evidence.values()), "reviewHistory": []}


def analyze_practice(action, stance, front, side, *, stopped=lambda: False, config=None, pipeline_factory=None) -> dict:
    """正、侧两路练习视频（可缺一路）→ 练习结果字典；``config=None`` 时读 ``feedback_config.json``。"""
    build_checks(action, stance)
    cfg = config or load_feedback_config()
    if front is not None and side is not None and Path(front).samefile(Path(side)):
        raise ValueError("正侧面不能使用同一个视频，请选择同次录制的两路视频")
    views = {}
    for view, path in (("front", front), ("side", side)):
        if stopped():
            raise InterruptedError("分析已取消")
        if path is not None:
            views[view] = extract_series(Path(path), cfg, stopped, pipeline_factory)
    if stopped():
        raise InterruptedError("分析已取消")
    return evaluate_series(action, stance, views, cfg)
