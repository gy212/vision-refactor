"""按基础动作段复用对应测量的验证记录，不按整套动作继承资格。"""

from __future__ import annotations

from core.evaluation.rules.kick_measures import KICK_SEGMENTS
from core.evaluation.rules.kick_policy import apply_kick_availability
from core.evaluation.rules.straight_policy import apply_straight_availability


def apply_rule_qualification(checks: list[dict], pose_variant: str) -> None:
    """同一实战式、直拳或低鞭腿动作段复用各自验证记录；摆拳保留既有规则状态。"""
    shared = [
        check
        for check in checks
        if check["segment"] == "stance" or check["segment"].endswith("_straight")
    ]
    apply_straight_availability(shared, pose_variant)
    kick_checks = [check for check in checks if check["segment"] in KICK_SEGMENTS]
    apply_kick_availability(kick_checks, pose_variant)
