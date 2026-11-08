"""按基础动作段复用对应测量的验证记录，不按整套动作继承资格。"""

from __future__ import annotations

from core.evaluation.rules.straight_policy import apply_straight_availability


def apply_rule_qualification(checks: list[dict], pose_variant: str) -> None:
    """同一实战式或直拳段使用相同登记；摆拳腿法保留各自的规则状态。"""
    shared = [
        check
        for check in checks
        if check["segment"] == "stance" or check["segment"].endswith("_straight")
    ]
    apply_straight_availability(shared, pose_variant)
