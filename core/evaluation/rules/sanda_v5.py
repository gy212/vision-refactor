"""按动作编排绑定公共检查定义，提供统一规则目录。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.evaluation.action_catalog import (
    ACTIONS,
    ACTION_SEGMENTS,
    PHASES,
    STANCES,
    action_definition,
    action_segments,
)
from core.evaluation.body_sides import side_label
from core.evaluation.rules.feedback_config import (
    FeedbackConfig,
    load_feedback_config,
    default_front_weights,
)
from core.evaluation.rules.rule_descriptions import (
    RULES,
    STANCE_RULES,
    PARTS,
    PRIMARY_VIEW,
    DOC,
    SUMMARY,
    FORMAL,
)
from core.evaluation.rules.rule_definitions import describe_rule, resolve_rule, standard_config
from core.evaluation.rules.catalog_measurements import required_landmarks
from core.evaluation.rules.spec import PhaseVerdict, RuleProfile, RuleSpec
from core.evaluation.rules.qualification import apply_rule_qualification
from core.evaluation.view_weights import practice_view_weights

__all__ = [
    "ACTIONS",
    "ACTION_SEGMENTS",
    "PHASES",
    "STANCES",
    "RULE_VERSION",
    "PROFILE_NAME",
    "DEFAULT_STANCE",
    "STATE_LABELS",
    "FeedbackConfig",
    "load_feedback_config",
    "default_front_weights",
    "required_landmarks",
    "action_segments",
    "action_config",
    "build_checks",
    "check_view_weights",
    "rule_profile",
    "SIDE_TIMED_ACTIONS",
]

RULE_VERSION = "sanda-rules-mediapipe-2026-11-06-v5.10"

PROFILE_NAME = "sanda_v5"

DEFAULT_STANCE = "left"  # 考试与 batch 的左右式：左手、左脚在前，右手护脸（D-3，配置值，以后可改）

STATE_LABELS = {
    "candidate": "待教师确认",
    "not_observed": "已检查帧中未发现",
    "unable": "证据不足",
    "pending_rule": "未自动检查",
}

ABSENCE_RULES = frozenset({"hip_turn", "arm_swing"})

# 消费者按分步迁移计划接入动作目录后移除；集合只派生自公共编排定义。
SIDE_TIMED_ACTIONS = frozenset(
    action for action in ACTIONS if action_definition(action).timing == "side_combo"
)


def action_config(cfg: FeedbackConfig, action: str) -> FeedbackConfig:
    """返回动作明确选用的标准参数，独立于分段和评分方式。"""
    return standard_config(cfg, action_definition(action).rule_standard)


def _append_segment_checks(add: Callable[..., None], segment: str, step_straight: bool) -> None:
    """按动作成员和明确标准组装过程与结束检查。"""
    if segment == "stance":
        return
    front = segment.startswith("front_")
    role = "front" if front else "rear"
    guard = "rear" if front else "front"
    hook = "hook" in segment
    kick = "kick" in segment
    straight = "straight" in segment
    source = f"{DOC} 第{10 if front else 11}行" if straight else SUMMARY
    for phase in ("motion", "finish"):
        if not kick or front:
            guard_codes = (
                ("guard_shoulder", "guard_back", "guard_elbow")
                if step_straight
                else ("guard_low", "guard_back", "guard_elbow")
            )
            for code in guard_codes:
                # 后手摆拳前手回收未包含向后拉手，不扩展问卷清单。
                if segment == "rear_hook" and code == "guard_back":
                    continue
                blocked = "原文摆拳阶段误写为直拳，需教师核对" if segment == "front_hook" else ""
                if hook and code == "guard_elbow":
                    blocked = "摆拳要求抬肘与错误抬肘的幅度/阶段需区分"
                add(code, segment, phase, guard, blocked, source)
        if segment in {"rear_straight", "rear_hook"} and not step_straight:
            for code in ("guard_low", "guard_back", "guard_elbow"):
                add(
                    code,
                    segment,
                    phase,
                    "rear",
                    "教师已确认阶段标准；该动作仍需实现并验证出拳侧的分阶段测量",
                    SUMMARY,
                )
        if segment == "rear_hook" or segment == "rear_low_kick":
            add("guard_recover", segment, phase, "front")
        if kick:
            for code in ("foot_flexed", "hip_turn", "arm_swing"):
                add(code, segment, phase, role)
        elif phase == "finish" or front:
            add(
                "shoulder_level",
                segment,
                phase,
                role,
                "原文摆拳阶段误写为直拳，需教师核对" if hook else "",
            )
        if straight and phase == "finish":
            add("punch_extension", segment, phase, role, source=FORMAL)
        if segment == "front_straight" and phase == "finish":
            add("vertical_fist", segment, phase, "front", source=source)


def build_checks(action: str, stance: str, *, pose_variant: str = "full") -> list[dict[str, Any]]:
    """所选动作与左右式的检查清单（开始实战式 → 各动作段 → 结束实战式）；多次动作合并在同一检查项里判定。"""
    if (
        not isinstance(action, str)
        or not isinstance(stance, str)
        or action not in ACTIONS
        or stance not in STANCES
    ):
        raise ValueError("请选择支持的动作和左式/右式")
    checks: list[dict[str, Any]] = []
    action_spec = action_definition(action)
    rule_standard = action_spec.rule_standard
    step_straight = rule_standard == "teacher_straight"

    def add(
        code: str,
        segment: str,
        phase: str,
        role: str = "",
        blocked: str = "",
        source: str = SUMMARY,
    ):
        name, part, standard, reason = RULES[code]
        if part.startswith(("front_", "rear_")):
            role, part = part.split("_", 1)
        side = side_label(role, stance)
        prefix = "前侧" if role == "front" else "后侧" if role == "rear" else ""
        check = {
            "id": f"{segment}:{phase}:{role}:{code}",
            "code": code,
            "name": name,
            "bodyPart": f"{prefix}{PARTS[part]}" + (f"（{side}）" if side else ""),
            "segment": segment,
            "segmentLabel": ACTIONS[segment],
            "phase": phase,
            "phaseLabel": PHASES[phase],
            "standard": standard,
            "source": source,
            "blockedReason": blocked or reason,
            "role": role,
        }
        if step_straight and code in {"stance_head", "vertical_fist"}:
            return
        describe_rule(check, resolve_rule(check, rule_standard, stance))
        checks.append(check)

    for code in STANCE_RULES:
        add(code, "stance", "start", source=f"{DOC} 第3～9行")
    segments = list(action_segments(action))
    for segment in segments:
        _append_segment_checks(add, segment, step_straight)
    if step_straight:
        for phase in ("start", "end"):
            add("guard_jaw", "stance", phase, "rear")
    if step_straight and "rear_straight" in segments:
        for code in ("punch_predrop", "punch_prepull", "punch_preelbow"):
            add(code, "rear_straight", "prepare", "rear")
        add("punch_recover", "rear_straight", "recovered", "rear")
    for code in STANCE_RULES:
        add(code, "stance", "end", source=f"{DOC} 第12～18行")
    apply_rule_qualification(checks, action, rule_standard, action_spec.qualification, pose_variant)
    # 保持开始→前手→后手→结束，再按各阶段身体部位排列；不统计重复次数。
    order = {name: i + 1 for i, name in enumerate(segments)}
    checks.sort(
        key=lambda c: (
            0 if c["phase"] == "start" else 99 if c["phase"] == "end" else order[c["segment"]],
            list(PHASES).index(c["phase"]),
            c["bodyPart"],
            c["id"],
        )
    )
    return checks


def check_view_weights(check: dict, cfg: FeedbackConfig) -> dict[str, float]:
    """一个检查项的正侧基础权重（单视角公式的项与标定为只用一个视角的项为 1/0）。"""
    primary = (
        {**PRIMARY_VIEW, check["code"]: check["primaryView"]}
        if "primaryView" in check
        else PRIMARY_VIEW
    )
    return practice_view_weights(check["code"], check["segment"], cfg.front_weights, primary)


def rule_profile(action: str, stance: str, cfg: FeedbackConfig) -> RuleProfile:
    """动作引用公共检查定义；验证资格只决定是否执行。"""
    standard = action_definition(action).rule_standard
    rules = []
    for check in build_checks(action, stance, pose_variant=cfg.pose_variant):
        definition = resolve_rule(check, standard, stance)
        rules.append(
            RuleSpec(
                code=check["id"],
                title=definition.name,
                check=definition.check,
                required=definition.required,
                view=definition.view,
                scope=f"{check['segment']}:{check['phase']}",
                aggregation=PhaseVerdict(
                    cfg.min_valid,
                    cfg.valid_ratio,
                    cfg.trigger_ratio,
                    absence=check["code"] in ABSENCE_RULES,
                ),
                authorization="blocked" if check["blockedReason"] else "candidate",
                blocked_reason=check["blockedReason"],
                source=definition.source,
                strategy=definition.strategy,
                highlight_joints=definition.highlight_joints,
                required_views=definition.required_views,
                evidence_joints_by_view=definition.evidence_joints_by_view,
            )
        )
    return RuleProfile(name=PROFILE_NAME, rules=tuple(rules))
