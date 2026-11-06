"""从规则结果构建各入口共用的证据记录与同步测量参照。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from core.evaluation.rules.spec import RULE_STATE_VIOLATION, RuleEvidence, RuleOutcome


@dataclass(frozen=True)
class EvidenceBundle:
    """检查含义与原画面分开保存，同一帧可支撑不同检查或动作次数。"""

    details: tuple[dict, ...]
    frames: dict[str, dict]


def _frame(view: str, series: Mapping, index: int) -> dict:
    frame = int(series["frames"][index])
    return {
        "id": f"{view}:{frame}",
        "view": view,
        "frame": frame,
        "timeSeconds": float(series["times"][index]),
        "landmarks": np.round(series["landmarks"][index], 5).tolist(),
        "validMask": series["valid_mask"][index].tolist(),
    }


def _detail(
    check: dict,
    outcome: RuleOutcome,
    instance: RuleEvidence,
    frame: dict,
    weight: float,
    timing_source: str,
) -> dict:
    spec = outcome.spec
    view = frame["view"]
    return {
        "ref": frame["id"],
        "frame": frame["frame"],
        "timeSeconds": frame["timeSeconds"],
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
        "timingSource": timing_source,
    }


def _paired_references(
    detail: dict, outcome: RuleOutcome, views: Mapping[str, dict], diagnostics: Mapping[str, dict]
) -> list[tuple[dict, dict]]:
    """按公式使用的同采集时刻补充参照；无对应画面或所需点时不补造。"""
    result = []
    spec = outcome.spec
    for view in spec.required_views:
        if view == detail["view"] or view not in views:
            continue
        series = views[view]
        indices = np.flatnonzero(np.asarray(series["times"]) == detail["timeSeconds"])
        if not len(indices):
            continue
        index = int(indices[0])
        joints = spec.evidence_joints_by_view.get(view, spec.highlight_joints)
        if not series["valid_mask"][index, list(joints)].all():
            continue
        frame = _frame(view, series, index)
        reference = {
            **detail,
            "ref": frame["id"],
            "frame": frame["frame"],
            "timeSeconds": frame["timeSeconds"],
            "view": view,
            "purpose": "reference",
            "referenceFor": detail["ref"],
            "highlightJoints": list(joints),
            "timingSource": diagnostics.get(view, {}).get("timingSource", "own_view"),
        }
        result.append((reference, frame))
    return result


def build_rule_evidence(
    check: dict,
    view: str,
    outcome: RuleOutcome,
    views: Mapping[str, dict],
    diagnostics: Mapping[str, dict],
    *,
    weight: float = 1.0,
) -> EvidenceBundle:
    """练习、考试及批量共用的完整证据，来源只取本次判定与原始序列。"""
    instances = outcome.evidence_instances
    if not instances:
        purpose = "violation" if outcome.state == RULE_STATE_VIOLATION else "observation"
        instances = tuple(RuleEvidence(i, purpose=purpose) for i in outcome.evidence_frames)
    details, frames = [], {}
    for instance in instances:
        frame = _frame(view, views[view], instance.frame)
        detail = _detail(
            check,
            outcome,
            instance,
            frame,
            weight,
            diagnostics.get(view, {}).get("timingSource", "own_view"),
        )
        details.append(detail)
        frames[frame["id"]] = frame
        for reference, paired_frame in _paired_references(detail, outcome, views, diagnostics):
            details.append(reference)
            frames[paired_frame["id"]] = paired_frame
    return EvidenceBundle(tuple(details), frames)
