"""第二批上步直拳的逐项启用依据，不把未验证项目冒充标定失败。"""

from __future__ import annotations

from core.evaluation.rules.availability import RuleAvailability

REPORT = "docs/上步直拳组合标定报告.md"
TEACHER_SOURCE = "教师确认 #33 issuecomment-6235417498；卷三第7题更正：后手必须收回下颚"
PARAMS = {"elbow_raise": -0.18, "shoulder_height": 0.17}

STRAIGHT_RULES = {
    "guard_jaw": (
        "后手未护住下颚",
        "hand",
        "实战式后手靠近下颚；此项不判断夹臂和贴肋",
        "",
    ),
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
    ("stance", "start", "", "stance_torso"): "front",
    ("stance", "end", "", "stance_torso"): "front",
    ("stance", "start", "rear", "guard_jaw"): "front",
    ("stance", "end", "rear", "guard_jaw"): "front",
    ("rear_straight", "motion", "front", "guard_shoulder"): "front",
    ("rear_straight", "finish", "front", "guard_shoulder"): "front",
    ("rear_straight", "motion", "front", "guard_elbow"): "front",
    ("rear_straight", "finish", "front", "guard_elbow"): "front",
    ("rear_straight", "finish", "rear", "shoulder_level"): "side",
    ("rear_straight", "prepare", "rear", "punch_prepull"): "side",
    ("rear_straight", "prepare", "rear", "punch_preelbow"): "side",
    ("rear_straight", "recovered", "rear", "punch_recover"): "front",
}

ENABLED_MODELS = {"stance_torso": frozenset({"full"})}
FAILED = {
    "guard_back": "完整前手事件窗口复核后，full与heavy过程均误报2/10；C32虽各2/2检出，仍不满足10%误报上限。",
    "punch_extension": "现有肘角测量受直拳投影和遮挡影响，C51错误样本与正确样本重叠。",
    "guard_elbow": "前手出拳时的后手抬肘测量与教师确认正确的D11、D13、D14重叠。",
    "guard_shoulder": "前手出拳时的后手掉肩仅检出C31的2/3（66.7%），未达到70%上线标准；heavy结束窗另有误报与缺失。",
    "stance_knee": "现有双膝二维角度公式在11段正确对照上全部误报，尚不能用于上步直拳。",
    "stance_elbow": "已核对真实护位；90°～135°二维肘角在正确样本上大量误报，不能直接作为实际肘角。",
    "stance_height": "手腕不是拳峰；手端代理虽减少误报，full与heavy正确样本误报仍超标。",
    "stance_width": "踝距/投影肩宽受朝向压缩影响；正确样本误报，且宽站目标会被反向判成过窄。",
    "stance_toes": "二维脚尖方向受透视和短线段抖动影响；正确样本误报与内八目标漏检并存。",
    "stance_rear_guard": "完整夹臂、贴肋和护下颚公式对C5检出不足；下颚距离子项不能证明整个复合要领。",
    "stance_torso": "heavy双视角投影比未通过C11目标验证；仅full路径启用，不沿用其他模型的结论。",
    "shoulder_level": "前手高度的阶段与投影测量仍有正确误报；首次伸展平台候选也漏检低拳，未达到启用标准。",
}


def apply_straight_availability(checks: list[dict], pose_variant: str = "full") -> None:
    """给每个检查项登记真实启用依据；已有能力限制优先保留。"""
    for check in checks:
        identity = (check["segment"], check["phase"], check["role"], check["code"])
        code = check["code"]
        models = ENABLED_MODELS.get(code)
        allowed = models is None or pose_variant in models
        if identity in ENABLED and allowed:
            availability = RuleAvailability("enabled", evidence=REPORT)
            check["primaryView"] = ENABLED[identity]
            if code == "stance_torso":
                check["standard"] = (
                    "同步双视角归一肩宽比超过√3；正交机位近似检查，不表示真实三维角度"
                )
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
