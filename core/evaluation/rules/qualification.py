"""检查绑定的资格选择，保留验证范围并识别未登记的新绑定。"""

from __future__ import annotations

from core.evaluation.rules.availability import RuleAvailability
from core.evaluation.rules.straight_policy import apply_straight_availability

LEGACY_CATALOG_ACTIONS = frozenset(
    {
        "stance",
        "straight_combo",
        "hook_combo",
        "front_straight",
        "rear_straight",
        "front_hook",
        "rear_hook",
        "front_low_kick",
        "rear_low_kick",
    }
)


def apply_rule_qualification(
    checks: list[dict], action: str, standard: str, qualification: str, pose_variant: str
) -> None:
    """资格按已登记的动作与标准匹配，不由测量标准自动继承。"""
    if (
        qualification == "legacy_catalog"
        and standard == "catalog"
        and action in LEGACY_CATALOG_ACTIONS
    ):
        # 旧目录没有此份验证登记，保留既有执行行为，不追加已验证标签。
        return
    if (
        qualification == "step_straight_validation"
        and action == "step_straight_combo"
        and standard == "teacher_straight"
    ):
        apply_straight_availability(checks, pose_variant)
        return
    reason = "当前动作与检查标准的绑定尚无适用的验证或试用记录"
    source = f"动作：{action}；检查标准：{standard}；资格来源：{qualification}（无适用登记）"
    for check in checks:
        availability = RuleAvailability(
            "unsupported" if check["blockedReason"] else "validation_pending",
            check["blockedReason"] or reason,
            source,
        )
        check["availability"] = availability.to_dict()
        check["blockedReason"] = availability.reason
