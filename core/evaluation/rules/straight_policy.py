"""基础实战式与直拳检查的验证资格，按阶段、测量与模型登记。"""

from __future__ import annotations

from core.evaluation.rules.availability import RuleAvailability
from core.evaluation.rules.rule_descriptions import PARAMS

REPORT = "docs/直拳逐项改进_20261105.md（原标定素材：上步直拳组合；登记粒度：基础动作段与阶段）"

ENABLED = frozenset(
    (
        ("stance", "start", "", "stance_torso"),
        ("stance", "end", "", "stance_torso"),
        ("stance", "start", "rear", "guard_jaw"),
        ("stance", "end", "rear", "guard_jaw"),
        ("stance", "end", "front", "stance_height"),
        ("front_straight", "motion", "rear", "guard_back"),
        ("front_straight", "finish", "rear", "guard_back"),
        ("front_straight", "motion", "rear", "guard_elbow"),
        ("front_straight", "finish", "rear", "guard_elbow"),
        ("rear_straight", "motion", "front", "guard_shoulder"),
        ("rear_straight", "finish", "front", "guard_shoulder"),
        ("rear_straight", "motion", "front", "guard_elbow"),
        ("rear_straight", "finish", "front", "guard_elbow"),
        ("rear_straight", "finish", "rear", "shoulder_level"),
        ("rear_straight", "prepare", "rear", "punch_prepull"),
        ("rear_straight", "prepare", "rear", "punch_preelbow"),
        ("rear_straight", "recovered", "rear", "punch_recover"),
    )
)

ENABLED_MODELS = {"stance_torso": frozenset({"full"})}

INSTANCE_MODELS = {
    ("front_straight", "finish", "rear", "guard_back"): frozenset({"heavy"}),
}

TRIAL_EVIDENCE = "docs/直拳逐项启用_20261106.md；用户2026-11-06明确同意开启试用、逐项改进"

FAILED = {
    "guard_back": "侧面常看不清护手肘。正面能找到肘，但试过的正面判法还没分清后拉。",
    "punch_extension": (
        f"已按反馈增加容错：每次出拳最伸展时达到{PARAMS['punch_elbow']:g}°不提示弯肘。"
        "关键点偏差仍可能造成误报，接近边界的弯肘可能漏查，继续试用。"
    ),
    "guard_shoulder": "已改看侧面。三段错误录像只查出两段，且查出的两段来自同一个人，还不够。",
    "stance_knee": "已按用户确认增加容错：超过175°才提醒。接近伸直时仍可能看错，继续试用。",
    "stance_elbow": "已按用户确认把下限降到80°。肩肘位置仍可能找偏，继续试用。",
    "stance_height": "已改看手腕有没有低过肩。开始站姿仍有误报；旧的低拳录像也不都低于肩，先不开。",
    "stance_width": (
        f"为减少误报，按正确对照样本试调为{PARAMS['stance_width_min']:.1f}～{PARAMS['stance_width_max']:g}倍。"
        "关键点偏差仍可能造成误报或漏查，继续试用。"
    ),
    "stance_toes": "已比较脚尖和脚跟间距。正确站姿仍有误报；两脚前后错开会影响画面距离。",
    "stance_rear_guard": "已把手肘贴身与护下巴分开。只看肘离身体的距离，仍没查出两段错误录像。",
    "stance_torso": "这套识别方式还没通过身体朝向的核对，先不开。",
    "shoulder_level": "已按用户确认把齐肩余量扩大到15°。接近边界的错误可能漏查，继续试用。",
}

TRIAL_CODES = frozenset(
    (
        "stance_height",
        "stance_elbow",
        "stance_toes",
        "stance_width",
        "stance_knee",
        "stance_rear_guard",
        "guard_back",
        "guard_shoulder",
        "guard_elbow",
        "punch_extension",
        "shoulder_level",
        "punch_predrop",
        "stance_torso",
    )
)


# 登记对应具体测量方法，替换公式或新增阶段后不能冒领旧资格。
REGISTERED_STRATEGIES = {
    "stance_height": "wrist_shoulder_floor",
    "stance_elbow": "elbow_angle_range",
    "stance_width": "feet_paired_shoulder_ratio",
    "stance_toes": "toe_heel_spread",
    "stance_knee": "knee_angle_limit",
    "stance_rear_guard": "elbow_body_distance",
    "stance_torso": "paired_shoulder_projection_ratio",
    "guard_jaw": "guard_wrist_jaw_distance",
    "guard_back": "elbow_behind_body",
    "guard_shoulder": "wrist_shoulder_drop",
    "guard_elbow": "elbow_shoulder_height",
    "punch_extension": "elbow_extension_at_peak",
    "punch_predrop": "prepunch_wrist_drop",
    "punch_prepull": "prepunch_wrist_pull",
    "punch_preelbow": "prepunch_elbow_height",
    "punch_recover": "returned_wrist_jaw_distance",
}


def _registered_measurement(check: dict) -> bool:
    code, role, segment, phase = (check[k] for k in ("code", "role", "segment", "phase"))
    expected = REGISTERED_STRATEGIES.get(code)
    if code == "shoulder_level":
        expected = "arm_torso_angle" if role == "front" else "wrist_shoulder_height_gap"
    if check["strategy"] != expected:
        return False
    if segment == "stance":
        return phase in {"start", "end"}
    if code.startswith("punch_pre"):
        return segment == "rear_straight" and phase == "prepare"
    if code == "punch_recover":
        return segment == "rear_straight" and phase == "recovered"
    if code == "punch_extension":
        return phase == "finish"
    if code == "shoulder_level":
        return phase == "finish" or (segment == "front_straight" and phase == "motion")
    return phase in {"motion", "finish"}


def apply_straight_availability(checks: list[dict], pose_variant: str = "full") -> None:
    """登记验证状态，不修改检查清单、定义、参数或观察方向。"""
    for check in checks:
        identity = (check["segment"], check["phase"], check["role"], check["code"])
        code = check["code"]
        models = INSTANCE_MODELS.get(
            identity, ENABLED_MODELS.get(code, frozenset({"full", "heavy"}))
        )
        allowed = pose_variant in models
        registered = _registered_measurement(check)
        if not registered:
            availability = RuleAvailability(
                "validation_pending", "当前阶段与测量方法尚无适用的验证记录", REPORT
            )
        elif identity in ENABLED and allowed:
            evidence = "docs/直拳结束低手修正_20261106.md" if code == "stance_height" else REPORT
            if code == "guard_back":
                evidence = "docs/直拳逐项启用_20261106.md"
            availability = RuleAvailability("enabled", evidence=evidence)
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
            registered
            and availability.state in {"calibration_failed", "validation_pending"}
            and code in TRIAL_CODES
            and pose_variant in {"full", "heavy"}
        ):
            note = availability.reason.replace("先不开。", "可能报错或漏查。")
            availability = RuleAvailability("trial", note, TRIAL_EVIDENCE)
        check["availability"] = availability.to_dict()
        check["blockedReason"] = "" if availability.state == "trial" else availability.reason
