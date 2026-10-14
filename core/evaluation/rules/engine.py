# -*- coding: utf-8 -*-
"""唯一的规则引擎 ``evaluate_rules``：按视角与动作范围筛规则，判缺点、有效帧，再按聚合方式出结论（设计文档 §7.2）。

正式 profile（``RatioDeduction``）在整段序列上逐帧判定；练习 profile（``PhaseVerdict``，D-16）在 ``RuleContext``
给出的相位窗口里判定，并带回证据帧与测量值摘要。
"""
from __future__ import annotations

import numpy as np

from core.evaluation.rules.spec import (
    RULE_STATE_EVALUATED,
    RULE_STATE_PASS,
    RULE_STATE_PENDING_RULE,
    RULE_STATE_SKIPPED,
    RULE_STATE_UNABLE,
    RULE_STATE_VIOLATION,
    SKIP_INSUFFICIENT_VALID_FRAMES,
    SKIP_LOW_CONFIDENCE,
    SKIP_MISSING_LANDMARKS,
    SKIP_PHASE_NOT_FOUND,
    PhaseVerdict,
    RuleContext,
    RuleOutcome,
    RuleProfile,
    RuleSpec,
)
from core.pose.validity import landmarks_missing_for_capabilities


def _applies(spec: RuleSpec, view: str, action_scope: str) -> bool:
    if spec.view not in ("any", view):
        return False
    return action_scope == "both" or spec.scope == action_scope


def _evaluate_one(spec: RuleSpec, landmarks: np.ndarray, valid_mask: np.ndarray, *, min_valid: int,
                  supported_capabilities, total_frames: int) -> RuleOutcome:
    # 后端结构性缺少所需关键点：直接跳过，不混入合格/不合格（Issue #11）。
    backend_missing = landmarks_missing_for_capabilities(spec.required, supported_capabilities)
    if backend_missing:
        return RuleOutcome(spec, RULE_STATE_SKIPPED, SKIP_MISSING_LANDMARKS, 0, 0.0, 0, total_frames,
                           f"{spec.title}（后端缺少关键点，未评估）", tuple(backend_missing))

    viol, valid, detail = spec.check(landmarks, valid_mask)
    valid_cnt = int(valid.sum())
    if valid_cnt < min_valid:
        # 0 帧有效视为置信度始终不足，0 < 有效 < min_valid 视为有效帧数不足；detail 两者相同（golden 不漂移）。
        reason = SKIP_LOW_CONFIDENCE if valid_cnt == 0 else SKIP_INSUFFICIENT_VALID_FRAMES
        return RuleOutcome(spec, RULE_STATE_SKIPPED, reason, 0, 0.0, valid_cnt, total_frames,
                           f"{detail}（有效帧不足，未评估）")
    ratio = float(viol[valid].sum() / max(1, valid_cnt))
    agg = spec.aggregation
    if ratio >= float(agg.trigger_ratio):
        return RuleOutcome(spec, RULE_STATE_EVALUATED, None, int(agg.points), ratio, valid_cnt, total_frames, detail)
    return RuleOutcome(spec, RULE_STATE_EVALUATED, None, 0, ratio, valid_cnt, total_frames, f"{detail}（合格）")


def _measurement_summaries(details: dict, valid: np.ndarray) -> tuple[dict, ...]:
    """测量值序列在有效且有限的帧上的中位数、最小、最大；没有可用帧的测量不列出。"""
    summaries = []
    for name, (values, unit) in details.items():
        observed = values[valid & np.isfinite(values)]
        if observed.size:
            summaries.append({"name": name, "unit": unit, "median": float(np.median(observed)),
                              "min": float(observed.min()), "max": float(observed.max())})
    return tuple(summaries)


def _evaluate_phase(spec: RuleSpec, landmarks: np.ndarray, valid_mask: np.ndarray, ctx: RuleContext) -> RuleOutcome:
    """练习检查项在其相位窗口里的单视角结论；证据帧为问题帧（无问题时为有效帧）的首/中/末帧。"""
    if spec.authorization == "blocked":
        return RuleOutcome(spec, RULE_STATE_PENDING_RULE, None, 0, 0.0, 0, 0, spec.blocked_reason)
    agg = spec.aggregation
    indices = ctx.phase_windows.get(spec.scope)
    if indices is None or len(indices) < agg.min_valid:
        return RuleOutcome(spec, RULE_STATE_UNABLE, SKIP_PHASE_NOT_FOUND, 0, 0.0, 0,
                           0 if indices is None else len(indices), "")
    baseline = ctx.baseline
    bad, valid, details = spec.check(landmarks[indices], valid_mask[indices],
                                     landmarks[baseline], valid_mask[baseline], ctx)
    count = int(valid.sum())
    total = len(indices)
    violation = float((bad & valid).sum() / count) if count else 0.0
    measurements = _measurement_summaries(details, valid)
    if count < agg.min_valid or count / total < agg.valid_ratio:
        return RuleOutcome(spec, RULE_STATE_UNABLE, SKIP_INSUFFICIENT_VALID_FRAMES, 0, violation, count, total, "",
                           measurements=measurements)
    # "未做某动作"只能在整个阶段均未出现该动作时成立。
    issue = bool(bad[valid].all()) if agg.absence else violation >= agg.trigger_ratio
    picked = indices[valid & bad] if issue else indices[valid]
    evidence = tuple(int(i) for i in picked[np.unique([0, len(picked) // 2, len(picked) - 1])])
    return RuleOutcome(spec, RULE_STATE_VIOLATION if issue else RULE_STATE_PASS, None, 0, violation, count, total, "",
                       evidence_frames=evidence, measurements=measurements)


def evaluate_rules(
    landmarks: np.ndarray,
    valid_mask: np.ndarray,
    profile: RuleProfile,
    *,
    view: str,
    action_scope: str,
    min_valid: int = 5,
    supported_capabilities=None,
    context: RuleContext | None = None,
) -> list[RuleOutcome]:
    """``profile`` 中适用于 ``view``/``action_scope`` 的规则逐条出结论，保持目录顺序。

    ``supported_capabilities=None`` 表示后端全部支持（MediaPipe），缺点判定不生效。``PhaseVerdict`` 规则
    需要 ``context``（相位窗口与基线），``min_valid`` 只用于 ``RatioDeduction`` 规则。
    """
    total_frames = int(landmarks.shape[0])
    outcomes = []
    for spec in profile.rules:
        if not _applies(spec, view, action_scope):
            continue
        if isinstance(spec.aggregation, PhaseVerdict):
            if context is None:
                raise ValueError(f"{profile.name} 的相位规则需要 RuleContext")
            outcomes.append(_evaluate_phase(spec, landmarks, valid_mask, context))
        else:
            outcomes.append(_evaluate_one(spec, landmarks, valid_mask, min_valid=min_valid,
                                          supported_capabilities=supported_capabilities, total_frames=total_frames))
    return outcomes
