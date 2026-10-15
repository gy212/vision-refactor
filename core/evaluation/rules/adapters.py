# -*- coding: utf-8 -*-
"""规则结论的对外结构：考试与 batch 逐视角报告的 ``RuleCheck``，以及练习 ``checks`` 字典里的单视角观测（下游不用改）。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from core.evaluation.rules.spec import (
    RULE_STATE_UNABLE,
    RULE_STATE_VIOLATION,
    RuleOutcome,
    RuleState,
    SkipReason,
)
from core.pose.validity import landmark_capabilities, landmark_names


@dataclass(frozen=True)
class RuleCheck:
    """一个检查项在一路视角上的结论（考试、batch 只报告，不折算分数）。

    ``state`` 为 ``violation``/``pass``/``unable``/``pending_rule``；``unable`` 的原因见 ``skip_reason`` 与 ``detail``。
    ``evidence_frames`` 是原视频帧号。
    """

    check_id: str
    code: str
    name: str
    segment: str
    phase: str
    view: str
    state: RuleState
    skip_reason: SkipReason | None
    violation_ratio: float
    valid_frames: int
    total_frames: int
    detail: str
    required_landmarks: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    missing_landmarks: tuple[str, ...] = ()
    evidence_frames: tuple[int, ...] = ()
    measurements: tuple[dict, ...] = ()


def to_rule_check(check: dict, view: str, outcome: RuleOutcome, frames, phase_reason: str = "") -> RuleCheck:
    """检查项（``build_checks`` 的一项）与它在本视角的结论 → ``RuleCheck``；``frames`` 把序列下标换成原视频帧号。"""
    detail = outcome.detail
    if outcome.state == RULE_STATE_UNABLE:
        detail = {"missing_landmarks": "后端缺少所需关键点",
                  "insufficient_valid_frames": f"该项测量有效帧不足（{outcome.valid_frames}/{outcome.total_frames}帧）",
                  "phase_not_found": "未找到该阶段；" + (phase_reason or "阶段不足")}[outcome.skip_reason]
    return RuleCheck(
        check_id=check["id"], code=check["code"], name=check["name"], segment=check["segment"], phase=check["phase"],
        view=view, state=outcome.state, skip_reason=outcome.skip_reason, violation_ratio=float(outcome.violation_ratio),
        valid_frames=int(outcome.valid_frames), total_frames=int(outcome.total_frames), detail=detail,
        required_landmarks=landmark_names(outcome.spec.required),
        required_capabilities=landmark_capabilities(outcome.spec.required),
        missing_landmarks=outcome.missing_landmarks,
        evidence_frames=tuple(int(frames[i]) for i in outcome.evidence_frames),
        measurements=outcome.measurements,
    )


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
