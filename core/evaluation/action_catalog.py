"""动作编排目录：分别声明动作成员、阶段识别和考试取样要求。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ActionDefinition:
    label: str
    segments: tuple[str, ...]
    timing: Literal["shared_events", "side_combo"] = "shared_events"
    timing_views: tuple[str, ...] = ("side", "front")
    scoring: Literal["whole_clip", "per_repetition"] = "whole_clip"
    rule_input: Literal["matched_window", "whole_clip"] = "matched_window"
    rule_standard: Literal["catalog", "teacher_straight"] = "catalog"
    qualification: Literal["unregistered", "legacy_catalog", "step_straight_validation"] = (
        "unregistered"
    )


ACTION_DEFINITIONS = {
    "stance": ActionDefinition(
        "实战式", (), timing_views=("front", "side"), qualification="legacy_catalog"
    ),
    "step_straight_combo": ActionDefinition(
        "上步前后手直拳组合",
        ("front_straight", "rear_straight"),
        timing="side_combo",
        timing_views=("side",),
        scoring="per_repetition",
        rule_input="whole_clip",
        rule_standard="teacher_straight",
        qualification="step_straight_validation",
    ),
    "straight_combo": ActionDefinition(
        "前后手直拳组合", ("front_straight", "rear_straight"), qualification="legacy_catalog"
    ),
    "hook_combo": ActionDefinition(
        "前后手摆拳组合",
        ("front_hook", "rear_hook"),
        timing_views=("front", "side"),
        qualification="legacy_catalog",
    ),
    "front_straight": ActionDefinition(
        "前手直拳", ("front_straight",), qualification="legacy_catalog"
    ),
    "rear_straight": ActionDefinition(
        "后手直拳", ("rear_straight",), qualification="legacy_catalog"
    ),
    "front_hook": ActionDefinition(
        "前手摆拳", ("front_hook",), timing_views=("front", "side"), qualification="legacy_catalog"
    ),
    "rear_hook": ActionDefinition(
        "后手摆拳", ("rear_hook",), timing_views=("front", "side"), qualification="legacy_catalog"
    ),
    "front_low_kick": ActionDefinition(
        "前低鞭腿", ("front_low_kick",), qualification="legacy_catalog"
    ),
    "rear_low_kick": ActionDefinition(
        "后低鞭腿", ("rear_low_kick",), qualification="legacy_catalog"
    ),
}
ACTIONS = {key: definition.label for key, definition in ACTION_DEFINITIONS.items()}
ACTION_SEGMENTS = {
    key: definition.segments
    for key, definition in ACTION_DEFINITIONS.items()
    if len(definition.segments) > 1
}
STANCES = {"left": "左式（左手、左脚在前）", "right": "右式（右手、右脚在前）"}
PHASES = {
    "start": "开始实战式",
    "prepare": "出拳前",
    "motion": "动作过程",
    "finish": "动作结束瞬间",
    "recovered": "收回后护位",
    "end": "结束实战式",
}


def action_definition(action: str) -> ActionDefinition:
    """返回已登记动作的编排要求；未知动作在外部选择边界拒绝。"""
    if not isinstance(action, str) or action not in ACTION_DEFINITIONS:
        raise ValueError("请选择支持的动作")
    return ACTION_DEFINITIONS[action]


def action_segments(action: str) -> tuple[str, ...]:
    """动作成员只描述编排，不决定成员使用哪种测量或评分。"""
    return action_definition(action).segments
