# -*- coding: utf-8 -*-
"""规则结论的对外结构：``RuleOutcome`` → 现有的 ``RuleScore``/``RuleViolation``（考试、batch、golden 的下游不用改）。"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from core.evaluation.rules.spec import RULE_STATE_EVALUATED, RuleOutcome, RuleState, SkipReason
from core.pose.validity import landmark_capabilities, landmark_names


@dataclass(frozen=True)
class RuleViolation:
    rule_id: str
    name: str
    penalty: int
    violation_ratio: float
    valid_frames: int
    total_frames: int
    detail: str
    # 结构化三态（Issue #11）：detail（中文）只供界面，下游判断只读 state / skip_reason。
    state: RuleState = RULE_STATE_EVALUATED
    skip_reason: SkipReason | None = None
    # 规则依赖的关键点名与能力分组（声明）；missing_landmarks 为后端缺失的点名。
    required_landmarks: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    missing_landmarks: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuleScore:
    score: int
    total_deduction: int
    violations: tuple[RuleViolation, ...]


def to_rule_violation(o: RuleOutcome) -> RuleViolation:
    return RuleViolation(
        rule_id=o.spec.code, name=o.spec.title, penalty=int(o.deduction), violation_ratio=o.violation_ratio,
        valid_frames=int(o.valid_frames), total_frames=int(o.total_frames), detail=o.detail, state=o.state,
        skip_reason=o.skip_reason, required_landmarks=landmark_names(o.spec.required),
        required_capabilities=landmark_capabilities(o.spec.required), missing_landmarks=o.missing_landmarks,
    )


def to_rule_score(outcomes: Iterable[RuleOutcome]) -> RuleScore:
    violations = tuple(to_rule_violation(o) for o in outcomes)
    deductions = sum(int(v.penalty) for v in violations)
    return RuleScore(score=max(0, 100 - int(deductions)), total_deduction=int(deductions), violations=violations)
