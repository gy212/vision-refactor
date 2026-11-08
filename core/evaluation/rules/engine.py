# -*- coding: utf-8 -*-
"""唯一的规则引擎 ``evaluate_rules``：按视角筛规则，判后端缺点、相位窗口与有效帧，再按 ``PhaseVerdict`` 出结论（设计文档 §7.2）。

考试、学生练习与离线处理都经这里执行同一套规则目录（``sanda_v5``，D-3）；结论带证据帧与测量值摘要。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.evaluation.phase_alignment import missing_phase_coverage
from core.evaluation.rules.spec import (
    RULE_STATE_PASS,
    RULE_STATE_PENDING_RULE,
    RULE_STATE_UNABLE,
    RULE_STATE_VIOLATION,
    SKIP_INSUFFICIENT_VALID_FRAMES,
    SKIP_MISSING_LANDMARKS,
    SKIP_PHASE_NOT_FOUND,
    RuleContext,
    RuleEvidence,
    RuleOutcome,
    RuleProfile,
    RuleSpec,
)
from core.pose.validity import landmark_names, landmarks_missing_for_capabilities
from core.foundation.signal import contiguous_runs


def _measurement_summaries(details: dict, valid: np.ndarray) -> tuple[dict, ...]:
    """测量值序列在有效且有限的帧上的中位数、最小、最大；没有可用帧的测量不列出。"""
    summaries = []
    for name, (values, unit) in details.items():
        observed = values[valid & np.isfinite(values)]
        if observed.size:
            summaries.append(
                {
                    "name": name,
                    "unit": unit,
                    "median": float(np.median(observed)),
                    "min": float(observed.min()),
                    "max": float(observed.max()),
                }
            )
    return tuple(summaries)


def _evaluate_phase(
    spec: RuleSpec,
    landmarks: np.ndarray,
    valid_mask: np.ndarray,
    ctx: RuleContext,
    supported_capabilities,
) -> RuleOutcome:
    """一条规则在其相位窗口里的单视角结论；证据帧为问题帧（无问题时为有效帧）的首/中/末帧。"""
    if spec.authorization == "blocked":
        return RuleOutcome(spec, RULE_STATE_PENDING_RULE, None, 0.0, 0, 0, spec.blocked_reason)
    indices = ctx.phase_windows.get(spec.scope)
    # 后端结构性缺少所需关键点：直接判证据不足，不混入合格/不合格（Issue #11）。
    backend_missing = landmarks_missing_for_capabilities(spec.required, supported_capabilities)
    if backend_missing:
        return RuleOutcome(
            spec,
            RULE_STATE_UNABLE,
            SKIP_MISSING_LANDMARKS,
            0.0,
            0,
            0,
            detail="后端缺少所需关键点：" + "、".join(backend_missing),
            missing_landmarks=tuple(backend_missing),
            evidence_frames=()
            if indices is None
            else tuple(item.frame for item in _diagnostic_evidence(indices, ctx, spec)),
            evidence_instances=() if indices is None else _diagnostic_evidence(indices, ctx, spec),
        )
    agg = spec.aggregation
    if indices is None or len(indices) < agg.min_valid:
        return RuleOutcome(
            spec,
            RULE_STATE_UNABLE,
            SKIP_PHASE_NOT_FOUND,
            0.0,
            0,
            0 if indices is None else len(indices),
            detail="未找到该阶段"
            if indices is None
            else f"阶段仅有{len(indices)}帧，至少需要{agg.min_valid}帧",
            evidence_frames=() if indices is None else tuple(int(i) for i in indices),
            evidence_instances=() if indices is None else _diagnostic_evidence(indices, ctx, spec),
        )
    ctx = replace(ctx, measurement_issues=[])
    baseline = ctx.baseline
    bad, valid, details = spec.check(
        landmarks[indices], valid_mask[indices], landmarks[baseline], valid_mask[baseline], ctx
    )
    count = int(valid.sum())
    total = len(indices)
    violation = float((bad & valid).sum() / count) if count else 0.0
    measurements = _measurement_summaries(details, valid)
    if count < agg.min_valid or count / total < agg.valid_ratio:
        evidence = _diagnostic_evidence(indices, ctx, spec)
        return RuleOutcome(
            spec,
            RULE_STATE_UNABLE,
            SKIP_INSUFFICIENT_VALID_FRAMES,
            violation,
            count,
            total,
            detail=_insufficient_detail(spec, valid_mask[indices], count, ctx),
            evidence_frames=tuple(item.frame for item in evidence),
            measurements=measurements,
            evidence_instances=evidence,
        )
    # "未做某动作"只能在整个阶段均未出现该动作时成立。
    issue = bool(bad[valid].all()) if agg.absence else violation >= agg.trigger_ratio
    evidence = _phase_evidence(indices, valid, bad, issue, ctx.phase_instances.get(spec.scope, ()))
    return RuleOutcome(
        spec,
        RULE_STATE_VIOLATION if issue else RULE_STATE_PASS,
        None,
        violation,
        count,
        total,
        evidence_frames=tuple(dict.fromkeys(item.frame for item in evidence)),
        measurements=measurements,
        evidence_instances=evidence,
    )


def _diagnostic_evidence(indices, ctx: RuleContext, spec: RuleSpec) -> tuple[RuleEvidence, ...]:
    """保留实际选中的阶段画面，失效点不会被当成可用测量。"""
    samples = _phase_evidence(
        indices,
        np.ones(len(indices), bool),
        np.zeros(len(indices), bool),
        False,
        ctx.phase_instances.get(spec.scope, ()),
    )
    return tuple(replace(sample, purpose="diagnostic") for sample in samples)


def _point_label(index: int) -> str:
    """将所需点的稳定名称转成报告可读的身体部位。"""
    name = landmark_names((index,))[0]
    for english, chinese in (
        ("foot_index", "脚尖"),
        ("shoulder", "肩"),
        ("elbow", "肘"),
        ("wrist", "腕"),
        ("hip", "髋"),
        ("knee", "膝"),
        ("ankle", "踝"),
        ("heel", "脚跟"),
        ("mouth", "嘴角"),
        ("nose", "鼻"),
        ("left", "左"),
        ("right", "右"),
    ):
        name = name.replace(english, chinese)
    return name.replace("_", "")


def _insufficient_detail(spec: RuleSpec, mask: np.ndarray, count: int, ctx: RuleContext) -> str:
    """解释测量卡在所需点还是必需参照，不把所有失败归为录像缺帧。"""
    total = len(mask)
    required = spec.aggregation
    reason = (
        f"该项测量有效帧不足（{count}/{total}帧）；"
        f"至少需要{required.min_valid}帧且占比达到{required.valid_ratio:.0%}"
    )
    absent = [
        f"{_point_label(index)}{int(mask[:, index].sum())}/{total}帧"
        for index in spec.required
        if not mask[:, index].all()
    ]
    if absent:
        reason += "；关键点可用情况：" + "、".join(absent)
    if ctx.measurement_issues:
        reason += "；" + "；".join(ctx.measurement_issues)
    elif not absent:
        reason += "；所需点已识别，但身体尺度或测量参照无法计算"
    return reason


def _phase_evidence(indices, valid, bad, issue, instances) -> tuple[RuleEvidence, ...]:
    """按已识别的每次阶段取图；连续或共用边界不改变次数，聚合判断不参与重算。"""
    evidence = []
    covered = np.zeros(len(indices), bool)
    groups = []
    for repetition, frames in instances:
        inside = np.isin(indices, frames)
        covered |= inside
        groups.append((repetition, inside))
    for window in contiguous_runs(indices[~covered]):
        groups.append((None, (indices >= window[0]) & (indices <= window[-1]) & ~covered))
    for repetition, inside in groups:
        has_issue = issue and bool((inside & valid & bad).any())
        observed = indices[inside & valid & (bad if has_issue else True)]
        if observed.size:
            selected = observed[np.unique([0, len(observed) // 2, len(observed) - 1])]
            evidence.extend(
                RuleEvidence(int(i), repetition, "violation" if has_issue else "observation")
                for i in selected
            )
    return tuple(evidence)


def evaluate_rules(
    landmarks: np.ndarray,
    valid_mask: np.ndarray,
    profile: RuleProfile,
    *,
    context: RuleContext,
    supported_capabilities=None,
) -> list[RuleOutcome]:
    """``profile`` 中适用于 ``context.view`` 的规则逐条出结论，保持目录顺序。

    ``supported_capabilities=None`` 表示后端全部支持（MediaPipe），缺点判定不生效。
    """
    outcomes = []
    for spec in profile.rules:
        if spec.view not in ("any", context.view):
            continue
        outcome = _evaluate_phase(spec, landmarks, valid_mask, context, supported_capabilities)
        missing = missing_phase_coverage(
            context.phase_coverage, spec.scope, spec.required_views or (context.view,)
        )
        if (
            missing
            and outcome.state != RULE_STATE_PENDING_RULE
            and outcome.skip_reason != SKIP_MISSING_LANDMARKS
        ):
            outcome = replace(
                outcome, state=RULE_STATE_UNABLE, skip_reason=SKIP_PHASE_NOT_FOUND, detail=missing
            )
        outcomes.append(outcome)
    return outcomes
