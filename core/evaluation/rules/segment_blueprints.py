"""各动作段声明式检查蓝图：按公共项与动作独有项组合，并显式绑定标定参数与文案。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from core.evaluation.rules.kick_rules import support_foot_anti_swap
from core.evaluation.rules.rule_descriptions import (
    DOC,
    HOOK_GUARD_PARAMS,
    HOOK_SHOULDER_PARAMS,
    KICK_PARAMS,
    KICK_SOURCE,
    PARAMS,
    SUMMARY,
    TEACHER_SOURCE,
    USER_20261105_SOURCE,
    USER_20261106_SOURCE,
    USER_20261108_SOURCE,
)


@dataclass(frozen=True)
class CheckBlueprint:
    """动作段内一条检查项的声明式蓝图：引用策略并显式声明标定参数覆盖与标准文案。"""

    code: str
    phase: str
    role: str
    strategy: str
    source: str
    param_overrides: Mapping[str, float] = field(default_factory=dict)
    name: str | None = None
    standard: str | None = None
    view: str | None = None
    valid_ratio_override: float | None = None
    trigger_ratio_override: float | None = None
    mask_filter: Callable[..., np.ndarray] | None = None


def _stance_phase_blueprints(phase: str, doc_rows: str) -> tuple[CheckBlueprint, ...]:
    doc_source = f"{DOC} {doc_rows}"
    return (
        CheckBlueprint(
            "stance_elbow",
            phase,
            "front",
            "elbow_angle_range",
            USER_20261106_SOURCE,
            PARAMS,
        ),
        CheckBlueprint(
            "stance_height",
            phase,
            "front",
            "wrist_shoulder_floor",
            USER_20261105_SOURCE,
            PARAMS,
        ),
        CheckBlueprint(
            "stance_rear_guard",
            phase,
            "rear",
            "elbow_body_distance",
            USER_20261105_SOURCE,
            PARAMS,
        ),
        CheckBlueprint(
            "guard_jaw",
            phase,
            "rear",
            "guard_wrist_jaw_distance",
            TEACHER_SOURCE,
            PARAMS,
        ),
        CheckBlueprint(
            "stance_knee",
            phase,
            "",
            "knee_angle_limit",
            USER_20261106_SOURCE,
            PARAMS,
        ),
        CheckBlueprint(
            "stance_width",
            phase,
            "",
            "feet_paired_shoulder_ratio",
            USER_20261108_SOURCE,
            PARAMS,
        ),
        CheckBlueprint("stance_toes", phase, "", "toe_heel_spread", doc_source, PARAMS),
        CheckBlueprint(
            "stance_torso",
            phase,
            "",
            "paired_shoulder_projection_ratio",
            doc_source,
            PARAMS,
        ),
    )


def _punch_guard_blueprints(
    guard_role: str,
    *,
    source: str,
    param_overrides: Mapping[str, float],
    include_jaw: bool = False,
) -> tuple[CheckBlueprint, ...]:
    back_standard = (
        "正面看清护手肘，试算肘有没有明显退到身体后方；允许正常收回"
        if guard_role == "front"
        else None
    )
    items: list[CheckBlueprint] = []
    for phase in ("motion", "finish"):
        items.append(
            CheckBlueprint(
                "guard_shoulder",
                phase,
                guard_role,
                "wrist_shoulder_drop",
                source,
                param_overrides,
            )
        )
        items.append(
            CheckBlueprint(
                "guard_back",
                phase,
                guard_role,
                "elbow_behind_body",
                USER_20261105_SOURCE,
                param_overrides,
                standard=back_standard,
            )
        )
        items.append(
            CheckBlueprint(
                "guard_elbow",
                phase,
                guard_role,
                "elbow_shoulder_height",
                source,
                param_overrides,
            )
        )
        if include_jaw:
            items.append(
                CheckBlueprint(
                    "guard_jaw",
                    phase,
                    guard_role,
                    "guard_wrist_jaw_distance",
                    source,
                    param_overrides,
                    name="前手未护住下颚",
                    standard="后手摆拳出拳过程中前手保持护住下颚（手腕距口部不超过0.34倍躯干长）",
                )
            )
    return tuple(items)


def _rear_punch_common_blueprints(
    param_overrides: Mapping[str, float],
) -> tuple[CheckBlueprint, ...]:
    return (
        CheckBlueprint(
            "punch_predrop",
            "prepare",
            "rear",
            "prepunch_wrist_drop",
            TEACHER_SOURCE,
            param_overrides,
        ),
        CheckBlueprint(
            "punch_prepull",
            "prepare",
            "rear",
            "prepunch_wrist_pull",
            TEACHER_SOURCE,
            param_overrides,
        ),
        CheckBlueprint(
            "punch_recover",
            "recovered",
            "rear",
            "returned_wrist_jaw_distance",
            TEACHER_SOURCE,
            param_overrides,
        ),
    )


def _kick_guard_blueprints(
    phase: str,
    guard_role: str,
    *,
    guard_drop_key: str,
    guard_back_key: str,
    include_jaw: bool,
) -> list[CheckBlueprint]:
    side_word = "侧面" if guard_role == "rear" else "正面"
    drop_params = {**KICK_PARAMS, "guard_shoulder_drop": KICK_PARAMS[guard_drop_key]}
    back_params = {**KICK_PARAMS, "guard_back": KICK_PARAMS[guard_back_key]}
    jaw_params = {**KICK_PARAMS, "punch_recovery": KICK_PARAMS["kick_guard_jaw_dist"]}
    items = [
        CheckBlueprint(
            "guard_shoulder",
            phase,
            guard_role,
            "wrist_shoulder_drop",
            KICK_SOURCE,
            drop_params,
            standard=(
                f"{side_word}看非踢腿侧护手腕相对同侧肩向下掉落不超过"
                f"{KICK_PARAMS[guard_drop_key]:.2f}倍躯干长"
            ),
        ),
        CheckBlueprint(
            "guard_back",
            phase,
            guard_role,
            "elbow_behind_body",
            KICK_SOURCE,
            back_params,
            standard=(
                f"{side_word}看非踢腿侧护手肘退到身体后方不超过"
                f"{KICK_PARAMS[guard_back_key]:.2f}倍躯干长"
            ),
            trigger_ratio_override=0.01,
        ),
        CheckBlueprint(
            "guard_elbow",
            phase,
            guard_role,
            "upper_arm_torso_flare",
            KICK_SOURCE,
            KICK_PARAMS,
            name="护手开肘亮肋",
            standard=(
                f"正面非踢腿侧大臂与躯干夹角不超过{KICK_PARAMS['kick_guard_elbow_flare']:g}°"
            ),
        ),
    ]
    if include_jaw:
        items.append(
            CheckBlueprint(
                "guard_jaw",
                phase,
                "front",
                "guard_wrist_jaw_distance",
                KICK_SOURCE,
                jaw_params,
                name="前手未护住下颚",
                standard=(
                    f"原地后低鞭腿起腿过程中前手腕距口部不超过"
                    f"{KICK_PARAMS['kick_guard_jaw_dist']:.2f}倍躯干长"
                ),
            )
        )
    return items


def _low_kick_blueprints(
    role: str,
    *,
    guard_drop_key: str,
    guard_back_key: str,
    include_jaw: bool,
    include_retract: bool,
) -> tuple[CheckBlueprint, ...]:
    guard_role = "rear" if role == "front" else "front"
    items: list[CheckBlueprint] = []
    for phase in ("motion", "finish"):
        items.extend(
            _kick_guard_blueprints(
                phase,
                guard_role,
                guard_drop_key=guard_drop_key,
                guard_back_key=guard_back_key,
                include_jaw=include_jaw,
            )
        )
        items.extend(
            (
                CheckBlueprint(
                    "foot_flexed",
                    phase,
                    role,
                    "kick_ankle_extension_angle",
                    KICK_SOURCE,
                    KICK_PARAMS,
                ),
                CheckBlueprint(
                    "hip_turn",
                    phase,
                    role,
                    "frontal_hip_width_ratio",
                    KICK_SOURCE,
                    KICK_PARAMS,
                ),
                CheckBlueprint(
                    "arm_swing",
                    phase,
                    role,
                    "arm_downward_extension",
                    KICK_SOURCE,
                    KICK_PARAMS,
                ),
                CheckBlueprint(
                    "head_down", phase, "", "nose_shoulder_drop", KICK_SOURCE, KICK_PARAMS
                ),
                CheckBlueprint(
                    "body_lean",
                    phase,
                    "",
                    "shoulder_line_tilt_angle",
                    KICK_SOURCE,
                    KICK_PARAMS,
                ),
            )
        )
    items.append(
        CheckBlueprint(
            "pivot_heel",
            "finish",
            guard_role,
            "support_foot_turn_projection",
            KICK_SOURCE,
            KICK_PARAMS,
            mask_filter=support_foot_anti_swap,
        )
    )
    items.append(
        CheckBlueprint(
            "kick_height",
            "finish",
            role,
            "kick_peak_height_band",
            KICK_SOURCE,
            KICK_PARAMS,
        )
    )
    if include_retract:
        items.append(
            CheckBlueprint(
                "kick_retract",
                "retract",
                role,
                "retract_ankle_reach",
                KICK_SOURCE,
                KICK_PARAMS,
            )
        )
    return tuple(items)


_FRONT_STRAIGHT_LEVEL_STD = (
    f"侧面看出拳手臂与身体的夹角；{90 - PARAMS['punch_level_deviation']:g}°"
    f"到{90 + PARAMS['punch_level_deviation']:g}°不提示高低偏差"
)
_HOOK_LEVEL_STD = "出拳时拳与肘均应大致与肩齐平"

STANCE_START_BLUEPRINTS: tuple[CheckBlueprint, ...] = _stance_phase_blueprints("start", "第3～9行")
STANCE_END_BLUEPRINTS: tuple[CheckBlueprint, ...] = _stance_phase_blueprints("end", "第12～18行")

SEGMENT_CHECK_BLUEPRINTS: dict[str, tuple[CheckBlueprint, ...]] = {
    "stance": (*STANCE_START_BLUEPRINTS, *STANCE_END_BLUEPRINTS),
    "front_straight": (
        *_punch_guard_blueprints("rear", source=f"{DOC} 第10行", param_overrides=PARAMS),
        CheckBlueprint(
            "shoulder_level",
            "motion",
            "front",
            "arm_torso_angle",
            USER_20261106_SOURCE,
            PARAMS,
            standard=_FRONT_STRAIGHT_LEVEL_STD,
        ),
        CheckBlueprint(
            "shoulder_level",
            "finish",
            "front",
            "arm_torso_angle",
            USER_20261106_SOURCE,
            PARAMS,
            standard=_FRONT_STRAIGHT_LEVEL_STD,
        ),
        CheckBlueprint(
            "punch_extension",
            "finish",
            "front",
            "elbow_extension_at_peak",
            USER_20261105_SOURCE,
            PARAMS,
        ),
    ),
    "rear_straight": (
        *_rear_punch_common_blueprints(PARAMS),
        CheckBlueprint(
            "punch_preelbow",
            "prepare",
            "rear",
            "prepunch_elbow_height",
            TEACHER_SOURCE,
            PARAMS,
        ),
        *_punch_guard_blueprints("front", source=f"{DOC} 第11行", param_overrides=PARAMS),
        CheckBlueprint(
            "punch_extension",
            "finish",
            "rear",
            "elbow_extension_at_peak",
            USER_20261105_SOURCE,
            PARAMS,
        ),
        CheckBlueprint(
            "shoulder_level",
            "finish",
            "rear",
            "wrist_shoulder_height_gap",
            SUMMARY,
            PARAMS,
        ),
    ),
    "front_hook": (
        *_punch_guard_blueprints("rear", source=SUMMARY, param_overrides=HOOK_GUARD_PARAMS),
        CheckBlueprint(
            "shoulder_level",
            "motion",
            "front",
            "hook_arm_shoulder_level",
            SUMMARY,
            HOOK_SHOULDER_PARAMS,
            standard=_HOOK_LEVEL_STD,
        ),
        CheckBlueprint(
            "shoulder_level",
            "finish",
            "front",
            "hook_arm_shoulder_level",
            SUMMARY,
            HOOK_SHOULDER_PARAMS,
            standard=_HOOK_LEVEL_STD,
        ),
    ),
    "rear_hook": (
        *_rear_punch_common_blueprints(HOOK_GUARD_PARAMS),
        *_punch_guard_blueprints(
            "front", source=SUMMARY, param_overrides=HOOK_GUARD_PARAMS, include_jaw=True
        ),
        CheckBlueprint(
            "shoulder_level",
            "finish",
            "rear",
            "hook_arm_shoulder_level",
            SUMMARY,
            HOOK_SHOULDER_PARAMS,
            standard=_HOOK_LEVEL_STD,
        ),
    ),
    "pad_step": (
        CheckBlueprint(
            "pad_step_missing",
            "prepare",
            "rear",
            "pad_step_displacement",
            KICK_SOURCE,
            KICK_PARAMS,
        ),
        CheckBlueprint(
            "pad_step_overtake",
            "prepare",
            "rear",
            "pad_step_ankle_overtake",
            KICK_SOURCE,
            KICK_PARAMS,
        ),
        CheckBlueprint(
            "pad_step_pause",
            "prepare",
            "rear",
            "pad_step_settle_pause",
            KICK_SOURCE,
            KICK_PARAMS,
        ),
    ),
    "front_low_kick": _low_kick_blueprints(
        "front",
        guard_drop_key="kick_guard_drop_side",
        guard_back_key="kick_guard_back_side",
        include_jaw=False,
        include_retract=True,
    ),
    "rear_low_kick": _low_kick_blueprints(
        "rear",
        guard_drop_key="kick_guard_drop_front",
        guard_back_key="kick_guard_back_front",
        include_jaw=True,
        include_retract=False,
    ),
}
