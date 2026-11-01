"""第二批上步直拳的逐项启用依据，不把未验证项目冒充标定失败。"""
from __future__ import annotations

from core.evaluation.rules.availability import RuleAvailability

REPORT = "docs/上步直拳组合标定报告.md"
TEACHER_SOURCE = "教师确认 #33 issuecomment-6235417498；卷三第7题更正：后手必须收回下颚"
PARAMS = {"elbow_raise": -.18, "shoulder_height": .17}

STRAIGHT_RULES = {
    "punch_predrop": (
        "出拳前先下沉",
        "hand",
        "出拳前不先把拳头向下放；允许收拳时从下方绕回",
        "",
    ),
    "punch_prepull": (
        "出拳前先向后拉手",
        "hand",
        "从护位直接出拳，不先向后预摆",
        "",
    ),
    "punch_preelbow": (
        "出拳前先抬肘",
        "arm",
        "后手直拳起动前不先将肘抬至肩高附近",
        "",
    ),
    "punch_recover": (
        "后手收回后未护住下颚",
        "hand",
        "后手收回后回到下颚，不能停在胸口；不限制收拳路径",
        "",
    ),
}

# 阶段是标定身份的一部分：护手公式不能直接套在正在出拳的手上。
ENABLED = {
    ("rear_straight", "motion", "front", "guard_shoulder"): "front",
    ("rear_straight", "finish", "front", "guard_shoulder"): "front",
    ("rear_straight", "motion", "front", "guard_elbow"): "front",
    ("rear_straight", "finish", "front", "guard_elbow"): "front",
    ("rear_straight", "finish", "rear", "shoulder_level"): "side",
    ("rear_straight", "prepare", "rear", "punch_prepull"): "side",
    ("rear_straight", "prepare", "rear", "punch_preelbow"): "side",
    ("rear_straight", "recovered", "rear", "punch_recover"): "front",
}

FAILED = {
    "guard_back": "现有护手后拉测量在参考样本中误报19%，C32两段均未检出。",
    "punch_extension": "现有肘角测量受直拳投影和遮挡影响，C51错误样本与正确样本重叠。",
    "guard_elbow": "前手出拳时的后手抬肘测量与教师确认正确的D11、D13、D14重叠。",
    "guard_shoulder": "前手出拳时，heavy模型曾错认教师示范的后手腕，产生误报。",
    "stance_knee": "现有双膝二维角度公式在11段正确对照上全部误报，尚不能用于上步直拳。",
}


def apply_straight_availability(checks: list[dict]) -> None:
    """给每个检查项登记真实启用依据；已有能力限制优先保留。"""
    for check in checks:
        identity = (
            check["segment"], check["phase"], check["role"], check["code"]
        )
        code = check["code"]
        if identity in ENABLED:
            availability = RuleAvailability("enabled", evidence=REPORT)
            check["primaryView"] = ENABLED[identity]
        elif check["blockedReason"]:
            availability = RuleAvailability("unsupported", check["blockedReason"], REPORT)
        elif code in FAILED:
            availability = RuleAvailability("calibration_failed", FAILED[code], REPORT)
        elif code == "punch_predrop":
            availability = RuleAvailability(
                "validation_pending",
                "教师已确认禁止出拳前下沉；测量已实现，但现有C35示范的是后拉，尚不能验证单独下沉的检出率。",
                REPORT,
            )
        else:
            availability = RuleAvailability(
                "validation_pending",
                f"教师标准已记录；“{check['name']}”尚无满足上线要求的逐项验证结果。",
                REPORT,
            )
        check["availability"] = availability.to_dict()
        check["blockedReason"] = availability.reason
        if code in STRAIGHT_RULES:
            check["source"] = TEACHER_SOURCE
