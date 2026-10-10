# -*- coding: utf-8 -*-
"""唯一的规则引擎 ``evaluate_rules``：按视角与动作范围筛规则，判缺点、有效帧，再按聚合方式出结论（设计文档 §7.2）。"""
from __future__ import annotations

import numpy as np

from core.evaluation.rules.spec import (
    RULE_STATE_EVALUATED,
    RULE_STATE_SKIPPED,
    SKIP_INSUFFICIENT_VALID_FRAMES,
    SKIP_LOW_CONFIDENCE,
    SKIP_MISSING_LANDMARKS,
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


def evaluate_rules(
    landmarks: np.ndarray,
    valid_mask: np.ndarray,
    profile: RuleProfile,
    *,
    view: str,
    action_scope: str,
    min_valid: int = 5,
    supported_capabilities=None,
) -> list[RuleOutcome]:
    """``profile`` 中适用于 ``view``/``action_scope`` 的规则逐条出结论，保持目录顺序。

    ``supported_capabilities=None`` 表示后端全部支持（MediaPipe），缺点判定不生效。
    """
    total_frames = int(landmarks.shape[0])
    return [
        _evaluate_one(spec, landmarks, valid_mask, min_valid=min_valid,
                      supported_capabilities=supported_capabilities, total_frames=total_frames)
        for spec in profile.rules
        if _applies(spec, view, action_scope)
    ]
