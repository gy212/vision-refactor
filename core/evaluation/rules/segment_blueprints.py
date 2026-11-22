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
    HOOK_STEP_SOURCE,
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
            "stance_elbow", phase, "front", "elbow_angle_range", USER_20261106_SOURCE, PARAMS
        ),
        CheckBlueprint(
            "stance_height", phase, "front", "wrist_shoulder_floor", USER_20261105_SOURCE, PARAMS
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
            "guard_jaw", phase, "rear", "guard_wrist_jaw_distance", TEACHER_SOURCE, PARAMS
        ),
        CheckBlueprint("stance_knee", phase, "", "knee_angle_limit", USER_20261106_SOURCE, PARAMS),
        CheckBlueprint(
            "stance_width",
            phase,
            "",
            "feet_paired_shoulder_ratio",
            USER_20261108_SOURCE,
            PARAMS,
        ),
        CheckBlueprint("stance_toes", phase, "", "rear_foot_toe_projection", doc_source, PARAMS),
        CheckBlueprint(
            "stance_torso", phase, "", "paired_shoulder_projection_ratio", doc_source, PARAMS
        ),
    )


_FRONT_GUARD_JAW_STD = (
    "发力时前手要及时收回来护住下巴；手腕离嘴边超过约三分之一上身长"
    f"（{KICK_PARAMS['kick_guard_jaw_dist']:.2f} 倍）算没护住"
)
_KICK_GUARD_DROP_STD = (
    "踢腿时护脸的手不能明显往下掉；前低鞭腿手腕掉到肩下超过约三分之一上身长"
    f"（{KICK_PARAMS['kick_guard_drop_side']:.2f} 倍）、"
    f"后低鞭腿超过四分之一上身长（{KICK_PARAMS['kick_guard_drop_front']:.2f} 倍）算掉手"
)
_KICK_GUARD_BACK_STD = (
    "踢腿时护手肘不能明显往背后拉；前低鞭腿退到背后超过约三分之一上身长"
    f"（{KICK_PARAMS['kick_guard_back_side']:.2f} 倍）、"
    f"后低鞭腿超过五分之一上身长（{KICK_PARAMS['kick_guard_back_front']:.2f} 倍）算后拉"
)


def _punch_guard_blueprints(
    guard_role: str,
    *,
    source: str,
    param_overrides: Mapping[str, float],
    include_jaw: bool = False,
) -> tuple[CheckBlueprint, ...]:
    items: list[CheckBlueprint] = []
    for phase in ("motion", "finish"):
        items.append(
            CheckBlueprint(
                "guard_shoulder", phase, guard_role, "wrist_shoulder_drop", source, param_overrides
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
            )
        )
        items.append(
            CheckBlueprint(
                "guard_elbow", phase, guard_role, "elbow_shoulder_height", source, param_overrides
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
                    standard=_FRONT_GUARD_JAW_STD,
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


def _punch_posture_blueprints(att_role: str) -> tuple[CheckBlueprint, ...]:
    """全部 4 个拳法段共用的 10 项公共姿态与发力规则包。"""
    knee_params = {"stance_knee_max": PARAMS["stance_knee_max"]}
    hip_view = "front" if att_role == "front" else "side"
    hip_std = (
        None
        if att_role == "front"
        else (
            "打后手拳要充分拧腰送胯；侧面看两胯宽度没比站好时收窄到 "
            f"{KICK_PARAMS['hip_turn_width_ratio'] * 100:g}% 以内，算没充分转腰"
        )
    )
    items: list[CheckBlueprint] = [
        CheckBlueprint(
            "punch_unrecovered",
            "motion",
            att_role,
            "punch_post_peak_recovery",
            HOOK_STEP_SOURCE,
        )
    ]
    for phase in ("motion", "finish"):
        items.extend(
            (
                CheckBlueprint(
                    "trunk_lean_sagittal",
                    phase,
                    "",
                    "trunk_sagittal_tilt_angle",
                    HOOK_STEP_SOURCE,
                ),
                CheckBlueprint(
                    "head_down",
                    phase,
                    "",
                    "nose_shoulder_drop",
                    HOOK_STEP_SOURCE,
                    name="出拳时低头",
                ),
                CheckBlueprint(
                    "stance_knee",
                    phase,
                    "",
                    "knee_angle_limit",
                    HOOK_STEP_SOURCE,
                    knee_params,
                    name="出拳时单腿或双腿伸得过直",
                ),
                CheckBlueprint(
                    "hip_turn",
                    phase,
                    att_role,
                    "frontal_hip_width_ratio",
                    HOOK_STEP_SOURCE,
                    name="出拳未充分转胯",
                    standard=hip_std,
                    view=hip_view,
                ),
            )
        )
    items.append(
        CheckBlueprint(
            "pivot_heel",
            "finish",
            att_role,
            "punch_heel_lift_pivot",
            HOOK_STEP_SOURCE,
            name="出拳发力脚未蹬转提踵",
            standard="出拳发力时同侧脚要蹬地提跟拧转；侧面看发力脚脚跟没比站好时明显抬起（提跟不足 0.05 倍上身长）算没蹬转",
        )
    )
    return tuple(items)


def _hook_private_blueprints(att_role: str) -> tuple[CheckBlueprint, ...]:
    """摆拳打到瞬间（finish）独有 3 项私有检查。"""
    return (
        CheckBlueprint(
            "hook_arm_bend", "finish", att_role, "hook_forearm_upper_ratio", HOOK_STEP_SOURCE
        ),
        CheckBlueprint(
            "hook_forearm_level",
            "finish",
            att_role,
            "hook_forearm_horizontal_angle",
            HOOK_STEP_SOURCE,
        ),
        CheckBlueprint(
            "hook_over_center", "finish", att_role, "hook_wrist_over_center", HOOK_STEP_SOURCE
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
            standard=_KICK_GUARD_DROP_STD,
        ),
        CheckBlueprint(
            "guard_back",
            phase,
            guard_role,
            "elbow_behind_body",
            KICK_SOURCE,
            back_params,
            standard=_KICK_GUARD_BACK_STD,
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
                "踢腿时护手大臂贴近上身护住肋部；正面看大臂和身体夹角张开超过 "
                f"{KICK_PARAMS['kick_guard_elbow_flare']:g}° 算开肘亮肋"
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
                standard=_FRONT_GUARD_JAW_STD,
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
                    "hip_turn", phase, role, "frontal_hip_width_ratio", KICK_SOURCE, KICK_PARAMS
                ),
                CheckBlueprint(
                    "head_down", phase, "", "nose_shoulder_drop", KICK_SOURCE, KICK_PARAMS
                ),
                CheckBlueprint(
                    "body_lean", phase, "", "shoulder_line_tilt_angle", KICK_SOURCE, KICK_PARAMS
                ),
            )
        )
    items.append(
        CheckBlueprint(
            "arm_swing", "finish", role, "arm_downward_extension", KICK_SOURCE, KICK_PARAMS
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
            "kick_height", "finish", role, "kick_peak_height_band", KICK_SOURCE, KICK_PARAMS
        )
    )
    if include_retract:
        items.append(
            CheckBlueprint(
                "kick_retract", "retract", role, "retract_ankle_reach", KICK_SOURCE, KICK_PARAMS
            )
        )
    return tuple(items)


_FRONT_STRAIGHT_LEVEL_STD = (
    "前手直拳平直往前打，手臂大约与身体成直角；侧面看手臂和身体夹角低于 "
    f"{90 - PARAMS['punch_level_deviation']:g}°（打太低）或高于 "
    f"{90 + PARAMS['punch_level_deviation']:g}°（打太高）算偏离肩平"
)
_HOOK_LEVEL_STD = (
    "摆拳打到瞬间拳头要和肩膀大致齐平；手腕高于或低于肩膀超过约五分之一上身长"
    f"（{HOOK_SHOULDER_PARAMS['shoulder_height']:.2f} 倍）算偏高或偏低"
)

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
        *_punch_posture_blueprints("front"),
    ),
    "rear_straight": (
        *_rear_punch_common_blueprints(PARAMS),
        CheckBlueprint(
            "punch_preelbow", "prepare", "rear", "prepunch_elbow_height", TEACHER_SOURCE, PARAMS
        ),
        *_punch_guard_blueprints(
            "front", source=f"{DOC} 第11行", param_overrides=PARAMS, include_jaw=True
        ),
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
            view="side",
        ),
        *_punch_posture_blueprints("rear"),
    ),
    "front_hook": (
        *_punch_guard_blueprints("rear", source=SUMMARY, param_overrides=HOOK_GUARD_PARAMS),
        CheckBlueprint(
            "shoulder_level",
            "finish",
            "front",
            "wrist_shoulder_height_gap",
            HOOK_STEP_SOURCE,
            HOOK_SHOULDER_PARAMS,
            standard=_HOOK_LEVEL_STD,
            view="any",
        ),
        *_hook_private_blueprints("front"),
        *_punch_posture_blueprints("front"),
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
            "wrist_shoulder_height_gap",
            HOOK_STEP_SOURCE,
            HOOK_SHOULDER_PARAMS,
            standard=_HOOK_LEVEL_STD,
            view="any",
        ),
        *_hook_private_blueprints("rear"),
        *_punch_posture_blueprints("rear"),
    ),
    "combo_transition": (
        CheckBlueprint(
            "combo_punch_pause", "motion", "", "combo_punch_pause_gap", HOOK_STEP_SOURCE
        ),
    ),
    "step_footwork": (
        CheckBlueprint(
            "step_feet_together", "motion", "", "step_feet_together_paired", HOOK_STEP_SOURCE
        ),
        CheckBlueprint("step_late_landing", "motion", "", "step_landing_sync", HOOK_STEP_SOURCE),
        CheckBlueprint(
            "step_follow_missing", "motion", "rear", "foot_forward_displacement", HOOK_STEP_SOURCE
        ),
        CheckBlueprint("step_overtake", "motion", "rear", "rear_foot_overtake", HOOK_STEP_SOURCE),
    ),
    "pad_step": (
        CheckBlueprint(
            "pad_step_missing",
            "prepare",
            "rear",
            "foot_forward_displacement",
            KICK_SOURCE,
            KICK_PARAMS,
        ),
        CheckBlueprint(
            "pad_step_overtake", "prepare", "rear", "rear_foot_overtake", KICK_SOURCE, KICK_PARAMS
        ),
        CheckBlueprint(
            "pad_step_pause", "prepare", "rear", "pad_step_settle_pause", KICK_SOURCE, KICK_PARAMS
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
