# -*- coding: utf-8 -*-
"""唯一的规则引擎 ``evaluate_rules``：按视角筛规则，判后端缺点、相位窗口与有效帧，再按 ``PhaseVerdict`` 出结论（设计文档 §7.2）。

考试、学生练习与离线处理都经这里执行同一套规则目录（``sanda_v5``，D-3）；结论带证据帧与测量值摘要。
"""
from __future__ import annotations

import numpy as np

from core.evaluation.rules.spec import (
    RULE_STATE_PASS,
    RULE_STATE_PENDING_RULE,
    RULE_STATE_UNABLE,
    RULE_STATE_VIOLATION,
    SKIP_INSUFFICIENT_VALID_FRAMES,
    SKIP_MISSING_LANDMARKS,
    SKIP_PHASE_NOT_FOUND,
    RuleContext,
    RuleOutcome,
    RuleProfile,
    RuleSpec,
)
from core.pose.validity import landmarks_missing_for_capabilities


def _measurement_summaries(details: dict, valid: np.ndarray) -> tuple[dict, ...]:
    """测量值序列在有效且有限的帧上的中位数、最小、最大；没有可用帧的测量不列出。"""
    summaries = []
    for name, (values, unit) in details.items():
        observed = values[valid & np.isfinite(values)]
        if observed.size:
            summaries.append({"name": name, "unit": unit, "median": float(np.median(observed)),
                              "min": float(observed.min()), "max": float(observed.max())})
    return tuple(summaries)


def _evaluate_phase(spec: RuleSpec, landmarks: np.ndarray, valid_mask: np.ndarray, ctx: RuleContext,
                    supported_capabilities) -> RuleOutcome:
    """一条规则在其相位窗口里的单视角结论；证据帧为问题帧（无问题时为有效帧）的首/中/末帧。"""
    if spec.authorization == "blocked":
        return RuleOutcome(spec, RULE_STATE_PENDING_RULE, None, 0.0, 0, 0, spec.blocked_reason)
    # 后端结构性缺少所需关键点：直接判证据不足，不混入合格/不合格（Issue #11）。
    backend_missing = landmarks_missing_for_capabilities(spec.required, supported_capabilities)
    if backend_missing:
        return RuleOutcome(spec, RULE_STATE_UNABLE, SKIP_MISSING_LANDMARKS, 0.0, 0, 0,
                           missing_landmarks=tuple(backend_missing))
    agg = spec.aggregation
    indices = ctx.phase_windows.get(spec.scope)
    if indices is None or len(indices) < agg.min_valid:
        return RuleOutcome(spec, RULE_STATE_UNABLE, SKIP_PHASE_NOT_FOUND, 0.0, 0,
                           0 if indices is None else len(indices))
    baseline = ctx.baseline
    bad, valid, details = spec.check(landmarks[indices], valid_mask[indices],
                                     landmarks[baseline], valid_mask[baseline], ctx)
    count = int(valid.sum())
    total = len(indices)
    violation = float((bad & valid).sum() / count) if count else 0.0
    measurements = _measurement_summaries(details, valid)
    if count < agg.min_valid or count / total < agg.valid_ratio:
        return RuleOutcome(spec, RULE_STATE_UNABLE, SKIP_INSUFFICIENT_VALID_FRAMES, violation, count, total,
                           measurements=measurements)
    # "未做某动作"只能在整个阶段均未出现该动作时成立。
    issue = bool(bad[valid].all()) if agg.absence else violation >= agg.trigger_ratio
    picked = indices[valid & bad] if issue else indices[valid]
    evidence = tuple(int(i) for i in picked[np.unique([0, len(picked) // 2, len(picked) - 1])])
    return RuleOutcome(spec, RULE_STATE_VIOLATION if issue else RULE_STATE_PASS, None, violation, count, total,
                       evidence_frames=evidence, measurements=measurements)


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
    return [_evaluate_phase(spec, landmarks, valid_mask, context, supported_capabilities)
            for spec in profile.rules if spec.view in ("any", context.view)]
