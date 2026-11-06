# -*- coding: utf-8 -*-
"""规则结论的对外结构：考试与 batch 逐视角报告的 ``RuleCheck``，以及练习 ``checks`` 字典里的单视角观测（下游不用改）。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from core.evaluation.rules.spec import (
    RULE_STATE_UNABLE,
    RULE_STATE_VIOLATION,
    RuleEvidence,
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
    evidence_details: tuple[dict, ...] = ()


def _evidence_detail(
    check: dict,
    view: str,
    outcome: RuleOutcome,
    ref: str,
    weight: float,
    instance: RuleEvidence,
) -> dict:
    """将规则自己的依据声明保存到检查项，不让展示端重新推断。"""
    spec = outcome.spec
    return {
        "ref": ref,
        "checkId": check["id"],
        "segment": check["segment"],
        "phase": check["phase"],
        "phaseLabel": check.get("phaseLabel", check["phase"]),
        "role": check.get("role", ""),
        "view": view,
        "highlightJoints": list(spec.evidence_joints_by_view.get(view, spec.highlight_joints)),
        "purpose": instance.purpose,
        "repetition": instance.repetition,
        "weight": weight,
        "strategy": spec.strategy,
        "requiredViews": list(spec.required_views),
    }


def _outcome_evidence(outcome: RuleOutcome) -> tuple[RuleEvidence, ...]:
    """兼容未提供实例的新旧规则调用；已提供的来源不再按时间反推。"""
    if outcome.evidence_instances:
        return outcome.evidence_instances
    purpose = "violation" if outcome.state == RULE_STATE_VIOLATION else "observation"
    return tuple(RuleEvidence(frame, purpose=purpose) for frame in outcome.evidence_frames)


def _register_frame(view: str, series: Mapping, index: int, evidence: dict) -> str:
    """画面按机位与原视频帧号共享，检查含义留在各项自己的引用中。"""
    ref = f"{view}:{series['frames'][index]}"
    evidence[ref] = {
        "id": ref,
        "view": view,
        "frame": series["frames"][index],
        "timeSeconds": series["times"][index],
        "landmarks": np.round(series["landmarks"][index], 5).tolist(),
        "validMask": series["valid_mask"][index].tolist(),
    }
    return ref


def _event_reference(
    check: dict, time: float, events: Sequence[dict], phase_instances: Sequence[dict] | None
) -> int | None:
    """只在明确的同动作事件范围内关联次数，不能由断帧猜测动作身份。"""
    if phase_instances is not None:
        matches = {
            phase["repetition"]
            for phase in phase_instances
            if phase["segment"] == check["segment"]
            and phase["phase"] == check["phase"]
            and phase["start"] <= time <= phase["end"]
        }
        return next(iter(matches)) if len(matches) == 1 else None
    selected = [event for event in events if event["segment"] == check["segment"]]
    for number, event in enumerate(selected, 1):
        if event["start"] <= time <= event["end"]:
            return number
    return None


def to_rule_check(
    check: dict, view: str, outcome: RuleOutcome, frames, phase_reason: str = ""
) -> RuleCheck:
    """检查项（``build_checks`` 的一项）与它在本视角的结论 → ``RuleCheck``；``frames`` 把序列下标换成原视频帧号。"""
    detail = outcome.detail
    if outcome.state == RULE_STATE_UNABLE:
        detail = {
            "missing_landmarks": "后端缺少所需关键点",
            "insufficient_valid_frames": f"该项测量有效帧不足（{outcome.valid_frames}/{outcome.total_frames}帧）",
            "phase_not_found": "未找到该阶段；" + (phase_reason or "阶段不足"),
        }[outcome.skip_reason]
    availability = check.get("availability", {})
    trial = availability.get("state") == "trial"
    if trial:
        detail = "试用检查。" + availability.get("reason", "") + " " + detail
    return RuleCheck(
        check_id=check["id"],
        code=check["code"],
        name=check["name"] + ("（试用）" if trial else ""),
        segment=check["segment"],
        phase=check["phase"],
        view=view,
        state=outcome.state,
        skip_reason=outcome.skip_reason,
        violation_ratio=float(outcome.violation_ratio),
        valid_frames=int(outcome.valid_frames),
        total_frames=int(outcome.total_frames),
        detail=detail,
        required_landmarks=landmark_names(outcome.spec.required),
        required_capabilities=landmark_capabilities(outcome.spec.required),
        missing_landmarks=outcome.missing_landmarks,
        evidence_frames=tuple(int(frames[i]) for i in outcome.evidence_frames),
        measurements=outcome.measurements,
        evidence_details=tuple(
            _evidence_detail(check, view, outcome, f"{view}:{int(frames[item.frame])}", 1.0, item)
            for item in _outcome_evidence(outcome)
        ),
    )


def to_practice_check(
    check: dict,
    view: str,
    outcome: RuleOutcome,
    base_weight: float,
    series: Mapping,
    evidence: dict,
    *,
    event_details: Sequence[dict] = (),
    phase_instances: Sequence[dict] | None = None,
    timing_source: str = "own_view",
) -> tuple[bool, float] | None:
    """把一路视角的练习结论写进现有 ``checks`` 字典（测量记录、视角问题、证据引用），返回融合输入。

    返回 ``(是否有问题, 基础权重 × 有效帧比例)``；有效帧不足时返回 None。证据帧连同原始骨架登记进
    ``evidence``（以 ``"视角:帧号"`` 去重），供教师直接核对，不保存视频图像。
    """
    count, total = outcome.valid_frames, outcome.total_frames
    quality = count / total
    measurement = {
        "view": view,
        "validFrames": count,
        "totalFrames": total,
        "validRatio": quality,
        "violationRatio": outcome.violation_ratio if count else None,
        "baseWeight": base_weight,
        "values": list(outcome.measurements),
    }
    check["measurements"].append(measurement)
    if outcome.state == RULE_STATE_UNABLE:
        check["viewProblems"][view] = f"该项测量有效帧不足（{count}/{total}帧），见关键点有效比例"
        return None
    issue = outcome.state == RULE_STATE_VIOLATION
    measurement["issueObserved"] = issue
    for instance in _outcome_evidence(outcome):
        index = instance.frame
        ref = _register_frame(view, series, index, evidence)
        if ref not in check["evidence"]:
            check["evidence"].append(ref)
        detail = _evidence_detail(check, view, outcome, ref, base_weight * quality, instance)
        if not outcome.evidence_instances:
            detail["repetition"] = _event_reference(
                check, series["times"][index], event_details, phase_instances
            )
        detail["timingSource"] = timing_source
        check.setdefault("evidenceDetails", []).append(detail)
    return issue, base_weight * quality


def register_paired_evidence(check: dict, views: dict, evidence: dict) -> None:
    """为双面原始数据测量保留同采集时刻的参照面，不伪造该面的独立判断。"""
    required = check.get("requiredViews", [])
    if len(required) < 2:
        return
    originals = tuple(check.get("evidenceDetails", []))
    saved = {(detail["ref"], detail.get("repetition")) for detail in originals}
    for detail in originals:
        source = evidence[detail["ref"]]
        for view in required:
            if view == detail["view"] or view not in views:
                continue
            series = views[view]
            indices = np.flatnonzero(np.asarray(series["times"]) == source["timeSeconds"])
            if not len(indices):
                continue
            index = int(indices[0])
            joints = check["evidenceJointsByView"].get(view, check["highlightJoints"])
            if not series["valid_mask"][index, joints].all():
                continue
            ref = f"{view}:{series['frames'][index]}"
            identity = (ref, detail.get("repetition"))
            if identity in saved:
                continue
            _register_frame(view, series, index, evidence)
            if ref not in check["evidence"]:
                check["evidence"].append(ref)
            check["evidenceDetails"].append(
                {
                    **detail,
                    "ref": ref,
                    "view": view,
                    "purpose": "reference",
                    "highlightJoints": list(joints),
                }
            )
            saved.add(identity)
