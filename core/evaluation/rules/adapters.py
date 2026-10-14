# -*- coding: utf-8 -*-
"""规则结论的对外结构：``RuleOutcome`` → 现有的 ``RuleScore``/``RuleViolation``，以及练习 ``checks`` 字典里的单视角观测（下游不用改）。"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from core.evaluation.rules.spec import (
    RULE_STATE_EVALUATED,
    RULE_STATE_UNABLE,
    RULE_STATE_VIOLATION,
    RuleOutcome,
    RuleState,
    SkipReason,
)
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


def to_practice_check(check: dict, view: str, outcome: RuleOutcome, base_weight: float, series: Mapping,
                      evidence: dict) -> tuple[bool, float] | None:
    """把一路视角的练习结论写进现有 ``checks`` 字典（测量记录、视角问题、证据引用），返回融合输入。

    返回 ``(是否有问题, 基础权重 × 有效帧比例)``；有效帧不足时返回 None。证据帧连同原始骨架登记进
    ``evidence``（以 ``"视角:帧号"`` 去重），供教师直接核对，不保存视频图像。
    """
    count, total = outcome.valid_frames, outcome.total_frames
    quality = count / total
    measurement = {"view": view, "validFrames": count, "totalFrames": total, "validRatio": quality,
                   "violationRatio": outcome.violation_ratio if count else None, "baseWeight": base_weight,
                   "values": list(outcome.measurements)}
    check["measurements"].append(measurement)
    if outcome.state == RULE_STATE_UNABLE:
        check["viewProblems"][view] = f"该项测量有效帧不足（{count}/{total}帧），见关键点有效比例"
        return None
    issue = outcome.state == RULE_STATE_VIOLATION
    measurement["issueObserved"] = issue
    for index in outcome.evidence_frames:
        ref = f"{view}:{series['frames'][index]}"
        evidence[ref] = {"id": ref, "view": view, "frame": series["frames"][index], "timeSeconds": series["times"][index],
                         "landmarks": np.round(series["landmarks"][index], 5).tolist(),
                         "validMask": series["valid_mask"][index].tolist()}
        check["evidence"].append(ref)
    return issue, base_weight * quality
