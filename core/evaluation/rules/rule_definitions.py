"""检查策略的统一定义，连接测量、身体部位、标准与证据需求。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping

from core.evaluation.paired_torso import paired_torso_check
from core.evaluation.rules.catalog_measurements import (
    FEET,
    measure_catalog,
    required_landmarks,
)
from core.evaluation.rules.feedback_config import FeedbackConfig
from core.evaluation.rules.kick_measures import (
    KICK_SEGMENTS,
    kick_landmarks,
    measure_kick,
)
from core.evaluation.rules.rule_descriptions import (
    KICK_PARAMS,
    KICK_STANDARDS,
    PARAMS,
    PREFERRED_VIEW,
    PRIMARY_VIEW,
    REVISED_STANDARDS,
    STRAIGHT_RULES,
    TEACHER_SOURCE,
)
from core.evaluation.rules.spec import PhaseCheck
from core.evaluation.rules.straight_feet import measure_toe_spread, stance_width_check
from core.evaluation.rules.straight_measures import measure_straight
from core.evaluation.rules.straight_motion import (
    guard_back_landmarks,
    measure_guard_back,
)
from core.evaluation.rules.straight_posture import (
    measure_extension_at_peak,
    measure_posture,
    posture_landmarks,
)


RULE_STRATEGIES = {
    "stance_elbow": "elbow_angle_range",
    "stance_height": "wrist_shoulder_floor",
    "stance_rear_guard": "elbow_body_distance",
    "stance_knee": "knee_angle_limit",
    "stance_width": "feet_paired_shoulder_ratio",
    "stance_toes": "toe_heel_spread",
    "stance_torso": "paired_shoulder_projection_ratio",
    "stance_head": "unsupported_head_posture",
    "guard_low": "wrist_mouth_height",
    "guard_back": "wrist_guard_displacement",
    "guard_elbow": "elbow_shoulder_height",
    "guard_shoulder": "wrist_shoulder_drop",
    "vertical_fist": "unsupported_fist_orientation",
    "shoulder_level": "wrist_shoulder_height_gap",
    "punch_extension": "elbow_extension_at_peak",
    "guard_recover": "wrist_mouth_distance",
    "foot_flexed": "knee_ankle_toe_angle",
    "hip_turn": "hip_axis_rotation",
    "arm_swing": "wrist_downward_displacement",
    "guard_jaw": "guard_wrist_jaw_distance",
    "punch_predrop": "prepunch_wrist_drop",
    "punch_prepull": "prepunch_wrist_pull",
    "punch_preelbow": "prepunch_elbow_height",
    "punch_recover": "returned_wrist_jaw_distance",
    "pad_step_missing": "pad_step_displacement",
    "pad_step_overtake": "pad_step_ankle_overtake",
    "pad_step_pause": "pad_step_settle_pause",
    "pivot_heel": "support_foot_turn_projection",
    "kick_height": "kick_peak_height_band",
    "kick_retract": "retract_ankle_reach",
    "head_down": "nose_shoulder_drop",
    "body_lean": "shoulder_line_tilt_angle",
}

_KICK_STRATEGIES = {
    **RULE_STRATEGIES,
    "foot_flexed": "kick_ankle_extension_angle",
    "hip_turn": "frontal_hip_width_ratio",
    "arm_swing": "arm_downward_extension",
    "guard_shoulder": "kick_wrist_shoulder_drop",
    "guard_back": "kick_elbow_behind_body",
    "guard_elbow": "upper_arm_torso_flare",
    "guard_jaw": "guard_wrist_jaw_distance",
}

_KICK_SIDE_CODES = frozenset(
    {
        "pad_step_missing",
        "pad_step_overtake",
        "pad_step_pause",
        "pivot_heel",
        "kick_height",
        "kick_retract",
        "foot_flexed",
        "head_down",
    }
)


@dataclass(frozen=True)
class RuleDefinition:
    """一条已绑定角色和标准的检查；不携带验证资格。"""

    strategy: str
    check: PhaseCheck
    required: tuple[int, ...]
    view: str
    name: str
    standard: str
    source: str
    parameter_overrides: Mapping[str, float]
    required_views: tuple[str, ...]
    evidence_joints_by_view: Mapping[str, tuple[int, ...]]

    @property
    def highlight_joints(self) -> tuple[int, ...]:
        return self.required


def standard_config(cfg: FeedbackConfig, segment: str) -> FeedbackConfig:
    """基础实战式与直拳采用同一批准参数，不随动作编排改变。"""
    return (
        replace(cfg, **PARAMS)
        if segment == "stance" or segment.endswith("_straight")
        else cfg
    )


def _posture_description(check: dict) -> tuple[str, str, str]:
    code, role = check["code"], check["role"]
    name, standard, source = check["name"], check["standard"], check["source"]
    if code in REVISED_STANDARDS:
        name, standard = REVISED_STANDARDS[code]
        source = "用户2026-11-05逐项意见及核对回复"
    if code == "guard_back" and role == "front":
        standard = "正面看清护手肘，试算肘有没有明显退到身体后方；允许正常收回"
    if code == "shoulder_level" and role == "front":
        tolerance = PARAMS["punch_level_deviation"]
        standard = f"侧面看出拳手臂与身体的夹角；{90 - tolerance:g}°到{90 + tolerance:g}°不提示高低偏差"
    if code in {"stance_elbow", "stance_knee"} or (
        code == "shoulder_level" and role == "front"
    ):
        source = "用户2026-11-06确认报告动作正确并要求增加容错；docs/直拳容错调整_20261106.md"
    if code == "stance_toes":
        standard = "正面比较两脚尖与两脚跟的间距；保留原35°对应的容许范围"
    if code == "stance_width":
        source = "用户2026-11-08授权按样本调整站距、优先减少误报；docs/实战式站距调整_20261108.md"
    if code == "stance_torso":
        standard = "同步双视角归一肩宽比超过√3；正交机位近似检查，不表示真实三维角度"
    if code in STRAIGHT_RULES:
        source = TEACHER_SOURCE
    return name, standard, source


def _posture_view(code: str, role: str) -> str:
    view = PREFERRED_VIEW.get(code, PRIMARY_VIEW.get(code, "any"))
    if code in {"guard_jaw", "punch_recover"}:
        return "front"
    if code in {"punch_prepull", "punch_preelbow"}:
        return "side"
    if code in {"guard_back", "guard_shoulder", "guard_elbow"} and role == "front":
        return "front"
    return view


def _kick_view(code: str, segment: str) -> str:
    if code in _KICK_SIDE_CODES:
        return "side"
    if code in {"guard_shoulder", "guard_back"} and segment == "front_low_kick":
        return "side"
    return "front"


def _kick_description(check: dict) -> tuple[str, str, str]:
    code, segment = check["code"], check["segment"]
    name, standard = KICK_STANDARDS[code]
    if code == "guard_shoulder":
        if segment == "front_low_kick":
            standard = (
                f"侧面看非踢腿侧护手腕相对同侧肩向下掉落不超过"
                f"{KICK_PARAMS['kick_guard_drop_side']:.2f}倍躯干长"
            )
        else:
            standard = (
                f"正面看非踢腿侧护手腕相对同侧肩向下掉落不超过"
                f"{KICK_PARAMS['kick_guard_drop_front']:.2f}倍躯干长"
            )
    elif code == "guard_back":
        if segment == "front_low_kick":
            standard = (
                f"侧面看非踢腿侧护手肘退到身体后方不超过"
                f"{KICK_PARAMS['kick_guard_back_side']:.2f}倍躯干长"
            )
        else:
            standard = (
                f"正面看非踢腿侧护手肘退到身体后方不超过"
                f"{KICK_PARAMS['kick_guard_back_front']:.2f}倍躯干长"
            )
    return name, standard, check["source"]


def _strategy_points(
    strategy: str, code: str, role: str, stance: str, view: str, segment: str = ""
) -> tuple[int, ...]:
    if segment in KICK_SEGMENTS:
        return kick_landmarks(strategy, role, stance, view)
    if strategy == "elbow_behind_body":
        return guard_back_landmarks(role, stance, view)
    if strategy == "feet_paired_shoulder_ratio":
        return (*FEET, 11, 12)
    if strategy in {
        "wrist_shoulder_floor",
        "elbow_body_distance",
        "arm_torso_angle",
        "elbow_extension_at_peak",
    }:
        return posture_landmarks(code, role, stance)
    return required_landmarks(code, role, stance)


def _phase_measurement(
    strategy: str, code: str, role: str, scope: str, segment: str
) -> PhaseCheck:
    if strategy == "paired_shoulder_projection_ratio":
        return paired_torso_check(scope)

    def check(lm, mask, baseline, baseline_mask, ctx):
        ctx = replace(ctx, params=standard_config(ctx.params, segment))
        if segment in KICK_SEGMENTS:
            return measure_kick(
                strategy, role, lm, mask, baseline, baseline_mask, ctx, scope
            )
        if strategy == "feet_paired_shoulder_ratio":
            return stance_width_check(scope)(lm, mask, baseline, baseline_mask, ctx)
        if strategy == "elbow_extension_at_peak":
            return measure_extension_at_peak(role, lm, mask, ctx)
        if strategy == "toe_heel_spread":
            return measure_toe_spread(lm, mask)
        if strategy == "elbow_behind_body":
            return measure_guard_back(role, lm, mask, baseline, baseline_mask, ctx)
        if strategy in {
            "wrist_shoulder_floor",
            "elbow_body_distance",
            "arm_torso_angle",
        }:
            return measure_posture(code, role, lm, mask, ctx)
        if code in STRAIGHT_RULES:
            return measure_straight(code, role, lm, mask, baseline, baseline_mask, ctx)
        details = {}
        bad, valid = measure_catalog(
            code,
            role,
            lm,
            mask,
            baseline,
            baseline_mask,
            ctx.stance,
            ctx.params,
            details,
            ctx.measurement_issues,
        )
        return bad, valid, details

    return check


def resolve_rule(check: dict, stance: str) -> RuleDefinition:
    """将检查绑定为可执行定义，绑定过程中不读取动作名或资格状态。"""
    code, role = check["code"], check["role"]
    strategy = RULE_STRATEGIES[code]
    view = PRIMARY_VIEW.get(code, "any")
    name, description, source = check["name"], check["standard"], check["source"]
    overrides = {}
    segment = check["segment"]
    if segment == "stance" or segment.endswith("_straight"):
        if code == "guard_back":
            strategy = "elbow_behind_body"
        if code == "shoulder_level" and role == "front":
            strategy = "arm_torso_angle"
        view = _posture_view(code, role)
        name, description, source = _posture_description(check)
        overrides = dict(PARAMS)
    elif segment in KICK_SEGMENTS:
        strategy = _KICK_STRATEGIES[code]
        view = _kick_view(code, segment)
        name, description, source = _kick_description(check)
        overrides = dict(KICK_PARAMS)
    required = _strategy_points(strategy, code, role, stance, view, segment)
    paired = strategy in {
        "paired_shoulder_projection_ratio",
        "feet_paired_shoulder_ratio",
    }
    evidence_points = {"front": required, "side": (11, 12)} if paired else {}
    required_views = ("front", "side") if paired else (() if view == "any" else (view,))
    scope = f"{check['segment']}:{check['phase']}"
    return RuleDefinition(
        strategy,
        _phase_measurement(strategy, code, role, scope, segment),
        required,
        view,
        name,
        description,
        source,
        overrides,
        required_views,
        evidence_points,
    )


def describe_rule(check: dict, definition: RuleDefinition) -> None:
    """把实际定义写入检查结果，供保存记录与报告直接使用。"""
    check.update(
        name=definition.name,
        standard=definition.standard,
        source=definition.source,
        strategy=definition.strategy,
        parameterOverrides=dict(definition.parameter_overrides),
        requiredLandmarks=list(definition.required),
        highlightJoints=list(definition.highlight_joints),
        requiredViews=list(definition.required_views),
        evidenceJointsByView={
            view: list(joints)
            for view, joints in definition.evidence_joints_by_view.items()
        },
    )
    if definition.view != "any":
        check["primaryView"] = definition.view
