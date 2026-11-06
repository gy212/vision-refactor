"""第二批上步直拳的逐项启用依据，不把未验证项目冒充标定失败。"""

from __future__ import annotations

from core.evaluation.rules.availability import RuleAvailability

REPORT = "docs/直拳逐项改进_20261105.md"
TEACHER_SOURCE = "教师确认 #33 issuecomment-6235417498；卷三第7题更正：后手必须收回下颚"
PARAMS = {
    "elbow_raise": -0.18,
    "shoulder_height": 0.17,
    "punch_elbow": 177.0,
    "stance_elbow_min": 80.0,
    "stance_knee_max": 175.0,
    "punch_level_deviation": 15.0,
}

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
    ("stance", "end", "front", "stance_height"): "front",
    ("front_straight", "motion", "rear", "guard_back"): "side",
    ("front_straight", "finish", "rear", "guard_back"): "side",
    ("front_straight", "motion", "rear", "guard_elbow"): "side",
    ("front_straight", "finish", "rear", "guard_elbow"): "side",
    ("rear_straight", "motion", "front", "guard_shoulder"): "front",
    ("rear_straight", "finish", "front", "guard_shoulder"): "front",
    ("rear_straight", "motion", "front", "guard_elbow"): "front",
    ("rear_straight", "finish", "front", "guard_elbow"): "front",
    ("rear_straight", "finish", "rear", "shoulder_level"): "side",
    ("rear_straight", "prepare", "rear", "punch_prepull"): "side",
    ("rear_straight", "prepare", "rear", "punch_preelbow"): "side",
    ("rear_straight", "recovered", "rear", "punch_recover"): "front",
}

# 每项先选能看清其运动方向的一面；不再给未启用项保留两面混算的默认值。
PREFERRED_VIEW = {
    "stance_height": "front",
    "stance_elbow": "side",
    "stance_toes": "front",
    "stance_width": "front",
    "stance_knee": "side",
    "stance_rear_guard": "front",
    "guard_back": "side",
    "guard_shoulder": "side",
    "guard_elbow": "side",
    "punch_extension": "side",
    "shoulder_level": "side",
    "punch_predrop": "side",
}
REVISED_STANDARDS = {
    "stance_elbow": (
        f"前手肘角不在{PARAMS['stance_elbow_min']:g}°～135°",
        f"前手大小臂夹角{PARAMS['stance_elbow_min']:g}°～135°；已给较小角度增加10°余量",
    ),
    "stance_knee": (
        "单腿或双腿伸得过直",
        f"两膝微屈；任一膝角超过{PARAMS['stance_knee_max']:g}°才提醒",
    ),
    "stance_height": ("前手手腕低于肩", "前手手腕不低于同侧肩膀"),
    "stance_rear_guard": ("后手肘离身体过远", "看肘与身体的距离；护下巴另查"),
    "punch_extension": ("直拳出拳时肘未伸直", "侧面看出拳肘角；177°及以上不提示弯肘"),
    "guard_back": (
        "护手肘明显退到身体后方",
        "侧面看护手肘到肩与腿根连线的后方距离；允许从身前收回到身旁",
    ),
}

ENABLED_MODELS = {"stance_torso": frozenset({"full"})}
INSTANCE_MODELS = {
    ("front_straight", "finish", "rear", "guard_back"): frozenset({"heavy"}),
}
TRIAL_CODES = frozenset(PREFERRED_VIEW) | {"stance_torso"}
TRIAL_EVIDENCE = "docs/直拳逐项启用_20261106.md；用户2026-11-06明确同意开启试用、逐项改进"
FAILED = {
    "guard_back": "侧面常看不清护手肘。正面能找到肘，但试过的正面判法还没分清后拉。",
    "punch_extension": "已只看侧面。按3°余量检查，仍会把正确出拳报错，先不开。",
    "guard_shoulder": "已改看侧面。三段错误录像只查出两段，且查出的两段来自同一个人，还不够。",
    "stance_knee": "已按用户确认增加容错：超过175°才提醒。接近伸直时仍可能看错，继续试用。",
    "stance_elbow": "已按用户确认把下限降到80°。肩肘位置仍可能找偏，继续试用。",
    "stance_height": "已改看手腕有没有低过肩。开始站姿仍有误报；旧的低拳录像也不都低于肩，先不开。",
    "stance_width": "已量较窄的脚尖或脚跟间距，并参照两面中较完整的肩宽。正确录像仍有误报。",
    "stance_toes": "已比较脚尖和脚跟间距。正确站姿仍有误报；两脚前后错开会影响画面距离。",
    "stance_rear_guard": "已把手肘贴身与护下巴分开。只看肘离身体的距离，仍没查出两段错误录像。",
    "stance_torso": "这套识别方式还没通过身体朝向的核对，先不开。",
    "shoulder_level": "已按用户确认把齐肩余量扩大到15°。接近边界的错误可能漏查，继续试用。",
}


def apply_straight_availability(checks: list[dict], pose_variant: str = "full") -> None:
    """给每个检查项登记真实启用依据；已有能力限制优先保留。"""
    checks[:] = [c for c in checks if c["code"] not in {"stance_head", "vertical_fist"}]
    for check in checks:
        identity = (check["segment"], check["phase"], check["role"], check["code"])
        code = check["code"]
        if code in PREFERRED_VIEW:
            check["primaryView"] = PREFERRED_VIEW[code]
        if code == "guard_back" and check["role"] == "front":
            check["primaryView"] = "front"
        if code in REVISED_STANDARDS:
            check["name"], check["standard"] = REVISED_STANDARDS[code]
            check["source"] = "用户2026-11-05逐项意见及核对回复"
        if code == "guard_back" and check["role"] == "front":
            check["standard"] = "正面看清护手肘，试算肘有没有明显退到身体后方；允许正常收回"
        if code == "shoulder_level" and check["role"] == "front":
            tolerance = PARAMS["punch_level_deviation"]
            check["standard"] = (
                f"侧面看出拳手臂与身体的夹角；{90 - tolerance:g}°到{90 + tolerance:g}°不提示高低偏差"
            )
        if code in {"stance_elbow", "stance_knee"} or (
            code == "shoulder_level" and check["role"] == "front"
        ):
            check["source"] = (
                "用户2026-11-06确认报告动作正确并要求增加容错；docs/直拳容错调整_20261106.md"
            )
        if code == "stance_toes":
            check["standard"] = "正面比较两脚尖与两脚跟的间距；保留原35°对应的容许范围"
        if code == "stance_width":
            check["standard"] = "较窄的脚尖或脚跟间距，与两面中较完整肩宽相比，应在0.8到1.2倍"
        models = INSTANCE_MODELS.get(identity, ENABLED_MODELS.get(code))
        allowed = models is None or pose_variant in models
        if identity in ENABLED and allowed:
            evidence = "docs/直拳结束低手修正_20261106.md" if code == "stance_height" else REPORT
            if code == "guard_back":
                evidence = "docs/直拳逐项启用_20261106.md"
            availability = RuleAvailability("enabled", evidence=evidence)
            check["primaryView"] = ENABLED[identity]
            if code == "stance_torso":
                check["standard"] = (
                    "同步双视角归一肩宽比超过√3；正交机位近似检查，不表示真实三维角度"
                )
        elif check["blockedReason"]:
            availability = RuleAvailability("unsupported", check["blockedReason"], REPORT)
        elif code in FAILED:
            reason = FAILED[code]
            if code == "guard_back" and check["role"] == "rear":
                reason = "已改看肘有没有退到身体后方。这套识别方式在前手拳打到头时，只查出两段错误中的一段。"
            availability = RuleAvailability("calibration_failed", reason, REPORT)
        elif code == "punch_predrop":
            availability = RuleAvailability(
                "validation_pending",
                "教师已确认出拳前不能先下放手。现在会比较手腕前后高低。已有错误录像主要演示后拉，单独下放还没查准。",
                REPORT,
            )
        else:
            availability = RuleAvailability(
                "validation_pending",
                f"教师标准已记录；“{check['name']}”尚无满足上线要求的逐项验证结果。",
                REPORT,
            )
        if (
            availability.state in {"calibration_failed", "validation_pending"}
            and code in TRIAL_CODES
            and pose_variant in {"full", "heavy"}
        ):
            note = availability.reason.replace("先不开。", "可能报错或漏查。")
            availability = RuleAvailability("trial", note, TRIAL_EVIDENCE)
        check["availability"] = availability.to_dict()
        check["blockedReason"] = "" if availability.state == "trial" else availability.reason
        if code in STRAIGHT_RULES:
            check["source"] = TEACHER_SOURCE
