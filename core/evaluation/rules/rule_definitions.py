"""声明式规则策略注册表与规则绑定解析（无动作家族 if/elif 分叉）。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

from core.evaluation.rules.body_posture_rules import (
    body_posture_landmarks,
    build_body_posture_check,
)
from core.evaluation.rules.feedback_config import FeedbackConfig
from core.evaluation.rules.guard_rules import (
    build_guard_check,
    guard_landmarks,
    guard_view,
)
from core.evaluation.rules.hook_rules import build_hook_check, hook_landmarks
from core.evaluation.rules.kick_rules import build_kick_check, kick_landmarks
from core.evaluation.rules.punch_rules import build_punch_check, punch_landmarks
from core.evaluation.rules.rule_descriptions import RULES
from core.evaluation.rules.segment_blueprints import (
    SEGMENT_CHECK_BLUEPRINTS,
    STANCE_END_BLUEPRINTS,
    STANCE_START_BLUEPRINTS,
    CheckBlueprint,
)
from core.evaluation.rules.spec import PhaseCheck, PhaseVerdict
from core.evaluation.rules.stance_rules import build_stance_check, stance_landmarks
from core.evaluation.rules.step_rules import build_step_check, step_landmarks
from core.evaluation.rules.straight_rules import (
    build_straight_check,
    straight_landmarks,
)
from core.pose.landmarks import L_SHOULDER, R_SHOULDER

__all__ = [
    "PUBLIC_DOMAINS",
    "PRIVATE_DOMAINS",
    "RuleDomain",
    "StrategySpec",
    "STRATEGY_REGISTRY",
    "CheckBlueprint",
    "STANCE_START_BLUEPRINTS",
    "STANCE_END_BLUEPRINTS",
    "SEGMENT_CHECK_BLUEPRINTS",
    "RuleDefinition",
    "required_landmarks",
    "resolve_blueprint",
    "resolve_rule",
    "describe_rule",
]

RuleDomain = Literal["stance", "guard", "punch", "body_posture", "step", "straight", "hook", "kick"]
PUBLIC_DOMAINS: frozenset[str] = frozenset({"stance", "guard", "punch", "body_posture", "step"})
PRIVATE_DOMAINS: frozenset[str] = frozenset({"straight", "hook", "kick"})


@dataclass(frozen=True)
class StrategySpec:
    """一条可执行测量策略的单一事实来源：归属域、点位契约、构造器、视角与聚合覆盖。"""

    name: str
    domain: RuleDomain
    landmarks: Callable[[str, str, str, str], tuple[int, ...]]
    build_check: Callable[..., PhaseCheck]
    default_view: str | Callable[[str], str]
    paired: bool = False
    absence: bool = False
    per_instance: bool = False
    valid_ratio_override: float | None = None
    trigger_ratio_override: float | None = None


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
    absence: bool = False
    per_instance: bool = False
    valid_ratio_override: float | None = None
    trigger_ratio_override: float | None = None

    @property
    def highlight_joints(self) -> tuple[int, ...]:
        return self.required

    def build_verdict(self, cfg: FeedbackConfig) -> PhaseVerdict:
        """按策略与蓝图声明的覆盖值构造引擎聚合参数。"""
        valid_ratio = (
            cfg.valid_ratio if self.valid_ratio_override is None else self.valid_ratio_override
        )
        trigger_ratio = (
            cfg.trigger_ratio
            if self.trigger_ratio_override is None
            else self.trigger_ratio_override
        )
        return PhaseVerdict(
            cfg.min_valid,
            valid_ratio,
            trigger_ratio,
            absence=self.absence,
            per_instance=self.per_instance,
        )


def _register(*specs: StrategySpec) -> dict[str, StrategySpec]:
    return {spec.name: spec for spec in specs}


STRATEGY_REGISTRY: dict[str, StrategySpec] = _register(
    # --- 公共域：实战式（stance_rules） ---
    StrategySpec("elbow_angle_range", "stance", stance_landmarks, build_stance_check, "side"),
    StrategySpec("wrist_shoulder_floor", "stance", stance_landmarks, build_stance_check, "front"),
    StrategySpec("elbow_body_distance", "stance", stance_landmarks, build_stance_check, "front"),
    StrategySpec("knee_angle_limit", "stance", stance_landmarks, build_stance_check, "side"),
    StrategySpec(
        "feet_paired_shoulder_ratio",
        "stance",
        stance_landmarks,
        build_stance_check,
        "front",
        paired=True,
    ),
    StrategySpec(
        "rear_foot_toe_projection", "stance", stance_landmarks, build_stance_check, "side"
    ),
    StrategySpec(
        "paired_shoulder_projection_ratio",
        "stance",
        stance_landmarks,
        build_stance_check,
        "front",
        paired=True,
    ),
    # --- 公共域：护手（guard_rules） ---
    StrategySpec("wrist_shoulder_drop", "guard", guard_landmarks, build_guard_check, guard_view),
    StrategySpec("elbow_behind_body", "guard", guard_landmarks, build_guard_check, guard_view),
    StrategySpec("elbow_shoulder_height", "guard", guard_landmarks, build_guard_check, guard_view),
    StrategySpec("upper_arm_torso_flare", "guard", guard_landmarks, build_guard_check, "front"),
    StrategySpec("guard_wrist_jaw_distance", "guard", guard_landmarks, build_guard_check, "front"),
    # --- 公共域：拳法（punch_rules） ---
    StrategySpec("prepunch_wrist_drop", "punch", punch_landmarks, build_punch_check, "side"),
    StrategySpec("prepunch_wrist_pull", "punch", punch_landmarks, build_punch_check, "side"),
    StrategySpec(
        "returned_wrist_jaw_distance",
        "punch",
        punch_landmarks,
        build_punch_check,
        "front",
    ),
    StrategySpec("wrist_shoulder_height_gap", "punch", punch_landmarks, build_punch_check, "side"),
    StrategySpec(
        "punch_post_peak_recovery",
        "punch",
        punch_landmarks,
        build_punch_check,
        "side",
        per_instance=True,
        trigger_ratio_override=0.01,
    ),
    StrategySpec(
        "combo_punch_pause_gap",
        "punch",
        punch_landmarks,
        build_punch_check,
        "side",
        per_instance=True,
        trigger_ratio_override=0.01,
    ),
    # --- 公共域：通用人体姿态与发力转体（body_posture_rules） ---
    StrategySpec(
        "nose_shoulder_drop",
        "body_posture",
        body_posture_landmarks,
        build_body_posture_check,
        "side",
        trigger_ratio_override=0.01,
    ),
    StrategySpec(
        "shoulder_line_tilt_angle",
        "body_posture",
        body_posture_landmarks,
        build_body_posture_check,
        "front",
        trigger_ratio_override=0.01,
    ),
    StrategySpec(
        "trunk_sagittal_tilt_angle",
        "body_posture",
        body_posture_landmarks,
        build_body_posture_check,
        "side",
    ),
    StrategySpec(
        "frontal_hip_width_ratio",
        "body_posture",
        body_posture_landmarks,
        build_body_posture_check,
        "front",
        absence=True,
    ),
    StrategySpec(
        "support_foot_turn_projection",
        "body_posture",
        body_posture_landmarks,
        build_body_posture_check,
        "side",
        absence=True,
    ),
    StrategySpec(
        "punch_heel_lift_pivot",
        "body_posture",
        body_posture_landmarks,
        build_body_posture_check,
        "side",
        absence=True,
    ),
    # --- 公共域：步法（step_rules） ---
    StrategySpec(
        "foot_forward_displacement",
        "step",
        step_landmarks,
        build_step_check,
        "side",
        per_instance=True,
        trigger_ratio_override=0.01,
    ),
    StrategySpec(
        "rear_foot_overtake",
        "step",
        step_landmarks,
        build_step_check,
        "side",
        per_instance=True,
        valid_ratio_override=0.5,
        trigger_ratio_override=0.01,
    ),
    StrategySpec(
        "step_feet_together_paired",
        "step",
        step_landmarks,
        build_step_check,
        "side",
        paired=True,
        valid_ratio_override=0.25,
    ),
    StrategySpec(
        "step_landing_sync",
        "step",
        step_landmarks,
        build_step_check,
        "side",
        per_instance=True,
        trigger_ratio_override=0.01,
    ),
    # --- 动作独有私有域：直拳（straight_rules） ---
    StrategySpec(
        "elbow_extension_at_peak",
        "straight",
        straight_landmarks,
        build_straight_check,
        "side",
    ),
    StrategySpec("arm_torso_angle", "straight", straight_landmarks, build_straight_check, "side"),
    StrategySpec(
        "prepunch_elbow_height",
        "straight",
        straight_landmarks,
        build_straight_check,
        "side",
    ),
    # --- 动作独有私有域：摆拳（hook_rules） ---
    StrategySpec("hook_forearm_upper_ratio", "hook", hook_landmarks, build_hook_check, "side"),
    StrategySpec(
        "hook_forearm_horizontal_angle", "hook", hook_landmarks, build_hook_check, "front"
    ),
    StrategySpec("hook_wrist_over_center", "hook", hook_landmarks, build_hook_check, "front"),
    # --- 动作独有私有域：低鞭腿与垫步（kick_rules） ---
    StrategySpec(
        "pad_step_settle_pause",
        "kick",
        kick_landmarks,
        build_kick_check,
        "side",
        per_instance=True,
        trigger_ratio_override=0.01,
    ),
    StrategySpec(
        "kick_peak_height_band",
        "kick",
        kick_landmarks,
        build_kick_check,
        "side",
        valid_ratio_override=0.5,
        trigger_ratio_override=0.5,
    ),
    StrategySpec(
        "retract_ankle_reach",
        "kick",
        kick_landmarks,
        build_kick_check,
        "side",
        per_instance=True,
        trigger_ratio_override=0.01,
    ),
    StrategySpec(
        "kick_ankle_extension_angle",
        "kick",
        kick_landmarks,
        build_kick_check,
        "side",
        valid_ratio_override=0.5,
        trigger_ratio_override=0.5,
    ),
    StrategySpec(
        "arm_downward_extension",
        "kick",
        kick_landmarks,
        build_kick_check,
        "front",
        absence=True,
    ),
)

_DEFAULT_STRATEGY_BY_CODE: dict[str, str] = {
    "stance_elbow": "elbow_angle_range",
    "stance_height": "wrist_shoulder_floor",
    "stance_rear_guard": "elbow_body_distance",
    "stance_knee": "knee_angle_limit",
    "stance_width": "feet_paired_shoulder_ratio",
    "stance_toes": "rear_foot_toe_projection",
    "stance_torso": "paired_shoulder_projection_ratio",
    "guard_shoulder": "wrist_shoulder_drop",
    "guard_back": "elbow_behind_body",
    "guard_elbow": "elbow_shoulder_height",
    "guard_jaw": "guard_wrist_jaw_distance",
    "punch_predrop": "prepunch_wrist_drop",
    "punch_prepull": "prepunch_wrist_pull",
    "punch_recover": "returned_wrist_jaw_distance",
    "shoulder_level": "wrist_shoulder_height_gap",
    "punch_unrecovered": "punch_post_peak_recovery",
    "combo_punch_pause": "combo_punch_pause_gap",
    "punch_extension": "elbow_extension_at_peak",
    "punch_preelbow": "prepunch_elbow_height",
    "hook_arm_bend": "hook_forearm_upper_ratio",
    "hook_forearm_level": "hook_forearm_horizontal_angle",
    "hook_over_center": "hook_wrist_over_center",
    "head_down": "nose_shoulder_drop",
    "body_lean": "shoulder_line_tilt_angle",
    "trunk_lean_sagittal": "trunk_sagittal_tilt_angle",
    "hip_turn": "frontal_hip_width_ratio",
    "pivot_heel": "support_foot_turn_projection",
    "step_feet_together": "step_feet_together_paired",
    "step_late_landing": "step_landing_sync",
    "step_follow_missing": "foot_forward_displacement",
    "step_overtake": "rear_foot_overtake",
    "pad_step_missing": "foot_forward_displacement",
    "pad_step_overtake": "rear_foot_overtake",
    "pad_step_pause": "pad_step_settle_pause",
    "kick_height": "kick_peak_height_band",
    "kick_retract": "retract_ankle_reach",
    "foot_flexed": "kick_ankle_extension_angle",
    "arm_swing": "arm_downward_extension",
}

_BLUEPRINT_INDEX: dict[tuple[str, str, str, str], CheckBlueprint] = {
    (segment, bp.phase, bp.role, bp.code): bp
    for segment, blueprints in SEGMENT_CHECK_BLUEPRINTS.items()
    for bp in blueprints
}


def required_landmarks(
    code_or_strategy: str, role: str, stance: str, view: str = "side"
) -> tuple[int, ...]:
    """统一代理查询策略或检查码所需的关键点下标。"""
    strategy = (
        code_or_strategy
        if code_or_strategy in STRATEGY_REGISTRY
        else _DEFAULT_STRATEGY_BY_CODE.get(code_or_strategy, "")
    )
    if strategy not in STRATEGY_REGISTRY:
        return ()
    return STRATEGY_REGISTRY[strategy].landmarks(strategy, role, stance, view)


def _resolve_view(spec: StrategySpec, bp: CheckBlueprint) -> str:
    if bp.view is not None:
        return bp.view
    if callable(spec.default_view):
        return spec.default_view(bp.role)
    return spec.default_view


def _required_views(view: str, paired: bool) -> tuple[str, ...]:
    if paired:
        return ("front", "side")
    if view == "any":
        return ()
    return (view,)


def _wrap_overrides(base_check: PhaseCheck, overrides: Mapping[str, float]) -> PhaseCheck:
    if not overrides:
        return base_check
    frozen = dict(overrides)

    def wrapped(lm, mask, baseline, baseline_mask, ctx):
        effective = (
            replace(ctx, params=replace(ctx.params, **frozen)) if ctx.params is not None else ctx
        )
        return base_check(lm, mask, baseline, baseline_mask, effective)

    return wrapped


def resolve_blueprint(segment: str, bp: CheckBlueprint, stance: str) -> RuleDefinition:
    """将一条声明式蓝图绑定为可执行定义，不分支判断动作家族。"""
    spec = STRATEGY_REGISTRY[bp.strategy]
    view = _resolve_view(spec, bp)
    required = spec.landmarks(bp.strategy, bp.role, stance, view)
    scope = f"{segment}:{bp.phase}"
    if bp.mask_filter is not None:
        base_check = spec.build_check(bp.strategy, bp.role, scope, mask_filter=bp.mask_filter)
    else:
        base_check = spec.build_check(bp.strategy, bp.role, scope)
    bound_check = _wrap_overrides(base_check, bp.param_overrides)
    default_name, _, default_standard = RULES[bp.code]
    if spec.paired:
        other_view = "side" if view == "front" else "front"
        evidence_points = {view: required, other_view: (L_SHOULDER, R_SHOULDER)}
    else:
        evidence_points = {}
    valid_override = (
        bp.valid_ratio_override
        if bp.valid_ratio_override is not None
        else spec.valid_ratio_override
    )
    trigger_override = (
        bp.trigger_ratio_override
        if bp.trigger_ratio_override is not None
        else spec.trigger_ratio_override
    )
    return RuleDefinition(
        strategy=bp.strategy,
        check=bound_check,
        required=required,
        view=view,
        name=bp.name or default_name,
        standard=bp.standard or default_standard,
        source=bp.source,
        parameter_overrides=dict(bp.param_overrides),
        required_views=_required_views(view, spec.paired),
        evidence_joints_by_view=evidence_points,
        absence=spec.absence,
        per_instance=spec.per_instance,
        valid_ratio_override=valid_override,
        trigger_ratio_override=trigger_override,
    )


def resolve_rule(check: dict, stance: str) -> RuleDefinition:
    """按检查字典主键查表并绑定为可执行定义，绑定过程不读取动作名或资格状态。"""
    key = (check["segment"], check["phase"], check["role"], check["code"])
    bp = _BLUEPRINT_INDEX[key]
    return resolve_blueprint(check["segment"], bp, stance)


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
            view: list(joints) for view, joints in definition.evidence_joints_by_view.items()
        },
    )
    if definition.view != "any":
        check["primaryView"] = definition.view
