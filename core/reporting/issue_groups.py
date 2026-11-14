"""合并同一动作里反复出现的同类护手提醒，同时保留原始检查项。"""

from __future__ import annotations

from dataclasses import replace

from core.reporting.practice_model import NormalizedCheck


_GROUPED_STRAIGHT_CODES = frozenset({"guard_back", "guard_shoulder", "guard_elbow"})
_GROUPED_KICK_CODES = frozenset(
    {
        "guard_back",
        "guard_shoulder",
        "guard_elbow",
        "guard_jaw",
        "arm_swing",
        "head_down",
        "body_lean",
        "foot_flexed",
        "hip_turn",
    }
)


def group_action_issues(
    checks: list[NormalizedCheck], action: str
) -> list[NormalizedCheck]:
    """按基础直拳或低鞭腿动作段合并过程与发力端同类提醒，不改变逐项判断和计数。"""
    del action
    groups: dict[tuple[str, ...], list[NormalizedCheck]] = {}
    for check in checks:
        key = (check.id,)
        is_candidate = (
            check.status == "candidate"
            and check.review != "revoked"
            and check.phase in {"motion", "finish"}
        )
        if is_candidate and (
            (
                check.segment in {"front_straight", "rear_straight"}
                and check.code in _GROUPED_STRAIGHT_CODES
            )
            or (
                check.segment in {"front_low_kick", "rear_low_kick"}
                and check.code in _GROUPED_KICK_CODES
            )
        ):
            key = (check.segment, check.body_part, check.code)
        groups.setdefault(key, []).append(check)
    result = []
    for group in groups.values():
        if len(group) == 1:
            result.append(group[0])
            continue
        confirmed = all(c.review == "confirmed" for c in group)
        result.append(
            replace(
                group[0],
                phase_label="、".join(c.phase_label for c in group),
                reason="\n".join(f"{c.phase_label}：{c.reason}" for c in group),
                evidence_refs=list(
                    dict.fromkeys(ref for c in group for ref in c.evidence_refs)
                ),
                measurements_summary="\n\n".join(
                    f"{c.phase_label}\n{c.measurements_summary}" for c in group
                ),
                audit_logs=[entry for c in group for entry in c.audit_logs],
                evidence_details=[entry for c in group for entry in c.evidence_details],
                members=tuple(group),
                review="confirmed" if confirmed else "pending",
                review_display="教师已确认" if confirmed else "",
            )
        )
    return result
