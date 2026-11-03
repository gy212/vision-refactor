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
    "punch_extension": "现有错误示范中，每只出拳手各只有一位执行者展示未伸直，尚不足以验证。",
    "guard_elbow": "侧面抬肘候选仍有短窗口波动，须先通过稳定窗口验收和稳健性复核。",
    "guard_shoulder": "现有示范尚未确认有两位执行者在前手出拳时掉下后手，需教师按手确认。",
    "stance_knee": "现有画面中的膝部投影不能可靠区分正确微屈和腿伸直。",
    "stance_elbow": "前手位于侧面远侧，遮挡和投影使当前肘角不能可靠代表实际角度。",
    "stance_height": "不同机位造成的拳肩头部高度偏差与错误幅度相近，缺同机位多人正确样本。",
    "stance_width": "站距目标与清单方向存在冲突，当前测量也不能可靠区分宽窄。",
    "stance_toes": "鞋尖点位和透视不能可靠区分内八外八，现有验证仍有误报。",
    "stance_rear_guard": "现有示范同时改变手腕和肘的位置，尚不能独立验证夹臂贴肋；下颚另有检查。",
    "stance_torso": "heavy双视角投影比未通过C11目标验证；仅full路径启用，不沿用其他模型的结论。",
    "shoulder_level": "前手高拳与正确示范仍有重叠，低拳子项须先完成窗口验收与稳健性复核。",
}

FAILED_INSTANCES = {
    (
        "front_straight",
        "motion",
        "rear",
        "guard_back",
    ): "护手移到下颚侧是否属于后拉尚待教师确认，短窗口判定也容易波动。",
    (
        "front_straight",
        "finish",
        "rear",
        "guard_back",
    ): "结束时护手后撤的测量与教师示范重叠，暂不能可靠判断。",
    (
        "rear_straight",
        "motion",
        "front",
        "guard_back",
    ): "侧面前手位于远侧、常被遮挡，正面又看不到前后深度变化。",
    (
        "rear_straight",
        "finish",
        "front",
        "guard_back",
    ): "侧面前手位于远侧、常被遮挡，正面又看不到前后深度变化。",
    (
        "front_straight",
        "motion",
        "front",
        "shoulder_level",
    ): "出拳过程高度会自然变化，现有资料尚未明确独立于打到瞬间的高度判定标准。",
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
        elif identity in FAILED_INSTANCES or code in FAILED:
            reason = FAILED_INSTANCES.get(identity, FAILED.get(code, ""))
            availability = RuleAvailability("calibration_failed", reason, REPORT)
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
