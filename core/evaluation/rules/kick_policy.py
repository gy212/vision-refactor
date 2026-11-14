"""低鞭腿与垫步检查项的分模型实证标定资格登记表。"""

from __future__ import annotations

from core.evaluation.rules.availability import RuleAvailability

KICK_REPORT = "docs/低鞭腿规则标定报告_20261114.md"
_VERIFIED_MODELS = frozenset({"full", "heavy"})

_REGISTERED_STRATEGIES = {
    "pad_step_missing": "pad_step_displacement",
    "pad_step_overtake": "pad_step_ankle_overtake",
    "pad_step_pause": "pad_step_settle_pause",
    "pivot_heel": "support_foot_turn_projection",
    "kick_height": "kick_peak_height_band",
    "kick_retract": "retract_ankle_reach",
    "foot_flexed": "kick_ankle_extension_angle",
    "hip_turn": "frontal_hip_width_ratio",
    "head_down": "nose_shoulder_drop",
    "body_lean": "shoulder_line_tilt_angle",
    "arm_swing": "arm_downward_extension",
    "guard_shoulder": "kick_wrist_shoulder_drop",
    "guard_back": "kick_elbow_behind_body",
    "guard_elbow": "upper_arm_torso_flare",
    "guard_jaw": "guard_wrist_jaw_distance",
}

# full 与 heavy 双模型均通过严格红线（VAL FP<=10% un=0、总 TGT TP>=70%、>=2人）
_ENABLED = frozenset(
    {
        ("pad_step", "prepare", "rear", "pad_step_missing"),
        ("pad_step", "prepare", "rear", "pad_step_overtake"),
        ("pad_step", "prepare", "rear", "pad_step_pause"),
        ("front_low_kick", "retract", "front", "kick_retract"),
        ("front_low_kick", "motion", "front", "arm_swing"),
        ("front_low_kick", "finish", "front", "arm_swing"),
        ("rear_low_kick", "motion", "rear", "arm_swing"),
        ("rear_low_kick", "finish", "rear", "arm_swing"),
        ("front_low_kick", "motion", "rear", "guard_shoulder"),
        ("front_low_kick", "finish", "rear", "guard_shoulder"),
        ("front_low_kick", "motion", "rear", "guard_back"),
        ("front_low_kick", "finish", "rear", "guard_back"),
        ("rear_low_kick", "motion", "front", "guard_back"),
        ("front_low_kick", "motion", "", "head_down"),
    }
)

# 仅在指定模型上独立达标的实例
_INSTANCE_MODELS = {
    ("front_low_kick", "motion", "", "body_lean"): frozenset({"heavy"}),
    ("front_low_kick", "finish", "", "body_lean"): frozenset({"heavy"}),
}

# VAL 误报达标但缺少目标错例或不足 2 名演示者的待补样本阻断项
_VALIDATION_PENDING_REASONS = {
    ("front_low_kick", "finish", "front", "kick_height"): (
        "合格对照无误报（0/11），但当前缺少踢高或踢低正例错误样本，暂不启用。"
    ),
    ("rear_low_kick", "finish", "rear", "kick_height"): (
        "合格对照无误报（0/3），但当前缺少踢高或踢低正例错误样本，暂不启用。"
    ),
    ("rear_low_kick", "motion", "front", "guard_shoulder"): (
        "合格对照无误报（0/3），但目标错误样本仅覆盖 1 名受试者（未达 2 人红线），暂不启用。"
    ),
    ("rear_low_kick", "finish", "front", "guard_shoulder"): (
        "合格对照无误报（0/3），但目标错误样本仅覆盖 1 名受试者（未达 2 人红线），暂不启用。"
    ),
    ("rear_low_kick", "motion", "", "body_lean"): (
        "合格对照无误报（0/3），但当前缺少原地后低鞭腿侧倾目标错例，暂不启用。"
    ),
    ("rear_low_kick", "finish", "", "body_lean"): (
        "合格对照无误报（0/3），但当前缺少原地后低鞭腿侧倾目标错例，暂不启用。"
    ),
}

# 单实例实证标定未通过项（检出不足 70%、unable 拉低总检出或误报过高）
_INSTANCE_FAILED_REASONS = {
    ("front_low_kick", "finish", "", "head_down"): (
        "前低鞭腿瞬间窗低头在双模型下目标检出均仅 1/2（50%），未达 70% 红线，暂不启用。"
    ),
    ("front_low_kick", "motion", "", "body_lean"): (
        "MediaPipe full 下前低鞭腿过程窗正面侧倾目标检出仅 1/2（50%），未达 70% 红线，仅 heavy 启用。"
    ),
    ("front_low_kick", "finish", "", "body_lean"): (
        "MediaPipe full 下前低鞭腿瞬间窗正面侧倾目标检出仅 1/2（50%），未达 70% 红线，仅 heavy 启用。"
    ),
    ("rear_low_kick", "finish", "front", "guard_back"): (
        "原地后低鞭腿瞬间窗护肘后拉受 C105_P1 遮挡影响总检出仅 1/2（50%），未达 70% 红线，暂不启用。"
    ),
    ("rear_low_kick", "motion", "", "head_down"): (
        "原地后低鞭腿过程窗低头检查在侧视转体下误报率过高（66.7%～100%），暂不启用。"
    ),
    ("rear_low_kick", "finish", "", "head_down"): (
        "原地后低鞭腿瞬间窗低头检查在侧视转体下误报率过高（66.7%～100%），暂不启用。"
    ),
}

_FAILED_CODES = {
    "pivot_heel": "实证标定显示支撑脚尖脚跟投影在正误样本间混叠且 heavy 遮挡率高，暂不启用。",
    "foot_flexed": "锁定 165° 阈值下合格样本误报率过高且远侧遮挡易误判，暂不启用。",
    "hip_turn": "正面二维两髋水平宽比值在正误样本间严重混叠，暂不启用。",
    "guard_elbow": "锁定 30° 阈值下正面大臂与躯干夹角投影导致正常护手普遍误报，暂不启用。",
    "guard_jaw": "原地后低鞭腿转体时前手自然离开正面嘴部投影且误报过高，暂不启用。",
}


def _registered_phase_ok(check: dict) -> bool:
    code, segment, phase = check["code"], check["segment"], check["phase"]
    if check["strategy"] != _REGISTERED_STRATEGIES.get(code):
        return False
    if segment == "pad_step":
        return code.startswith("pad_step_") and phase == "prepare"
    if code == "kick_retract":
        return segment == "front_low_kick" and phase == "retract"
    if code in {"pivot_heel", "kick_height"}:
        return segment in {"front_low_kick", "rear_low_kick"} and phase == "finish"
    if code == "guard_jaw":
        return segment == "rear_low_kick" and phase in {"motion", "finish"}
    return segment in {"front_low_kick", "rear_low_kick"} and phase in {
        "motion",
        "finish",
    }


def _resolve_kick_availability(check: dict, pose_variant: str) -> RuleAvailability:
    if not _registered_phase_ok(check):
        return RuleAvailability(
            "validation_pending", "当前阶段与测量策略尚无低鞭腿标定记录", KICK_REPORT
        )
    if pose_variant not in _VERIFIED_MODELS:
        return RuleAvailability(
            "validation_pending",
            "仅 MediaPipe full 与 heavy 已完成低鞭腿标定验证，当前模型暂不启用",
            KICK_REPORT,
        )
    identity = (check["segment"], check["phase"], check["role"], check["code"])
    if identity in _ENABLED or pose_variant in _INSTANCE_MODELS.get(identity, ()):
        return RuleAvailability("enabled", evidence=KICK_REPORT)
    if identity in _VALIDATION_PENDING_REASONS:
        return RuleAvailability(
            "validation_pending", _VALIDATION_PENDING_REASONS[identity], KICK_REPORT
        )
    if identity in _INSTANCE_FAILED_REASONS:
        return RuleAvailability(
            "calibration_failed", _INSTANCE_FAILED_REASONS[identity], KICK_REPORT
        )
    code = check["code"]
    if code in _FAILED_CODES:
        return RuleAvailability("calibration_failed", _FAILED_CODES[code], KICK_REPORT)
    return RuleAvailability(
        "validation_pending",
        f"“{check['name']}”尚未完成当前动作段的独立样本验证。",
        KICK_REPORT,
    )


def apply_kick_availability(checks: list[dict], pose_variant: str = "full") -> None:
    """按分模型标定结论设置低鞭腿与垫步检查项资格状态，不修改定义与参数。"""
    for check in checks:
        availability = _resolve_kick_availability(check, pose_variant)
        check["availability"] = availability.to_dict()
        check["blockedReason"] = availability.reason
