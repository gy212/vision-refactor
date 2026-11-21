"""按动作编排绑定声明式蓝图，提供全系统唯一规则目录 sanda_v5。"""

from __future__ import annotations

from typing import Any

from core.evaluation.action_catalog import (
    ACTIONS,
    ACTION_SEGMENTS,
    PHASES,
    SEGMENT_LABELS,
    STANCES,
    action_segments,
)
from core.evaluation.body_sides import side_label
from core.evaluation.rules.feedback_config import (
    FeedbackConfig,
    default_front_weights,
    load_feedback_config,
)
from core.evaluation.rules.rule_definitions import (
    SEGMENT_CHECK_BLUEPRINTS,
    STANCE_END_BLUEPRINTS,
    STANCE_START_BLUEPRINTS,
    CheckBlueprint,
    describe_rule,
    required_landmarks,
    resolve_blueprint,
    resolve_rule,
)
from core.evaluation.rules.rule_descriptions import PARTS, PRIMARY_VIEW, RULES
from core.evaluation.rules.spec import RuleProfile, RuleSpec
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
    "build_checks",
    "catalog_checks",
    "check_view_weights",
    "rule_profile",
]

RULE_VERSION = "sanda-rules-mediapipe-2026-11-21-v5.19"

PROFILE_NAME = "sanda_v5"

DEFAULT_STANCE = "left"  # 考试与 batch 的左右式：左手、左脚在前，右手护脸（D-3，配置值，以后可改）

STATE_LABELS = {
    "candidate": "待教师确认",
    "not_observed": "已检查帧中未发现",
    "unable": "证据不足",
}

_ROLE_PREFIX = {"front": "前侧", "rear": "后侧"}
_PHASE_ORDER = tuple(PHASES)


def _body_part_label(part: str, role: str, stance: str) -> str:
    prefix = _ROLE_PREFIX.get(role, "")
    side = side_label(role, stance)
    suffix = f"（{side}）" if side else ""
    return f"{prefix}{PARTS[part]}{suffix}"


def _build_check_dict(segment: str, bp: CheckBlueprint, stance: str) -> dict[str, Any]:
    default_name, part, default_standard = RULES[bp.code]
    check: dict[str, Any] = {
        "id": f"{segment}:{bp.phase}:{bp.role}:{bp.code}",
        "code": bp.code,
        "name": bp.name or default_name,
        "bodyPart": _body_part_label(part, bp.role, stance),
        "segment": segment,
        "segmentLabel": SEGMENT_LABELS[segment],
        "phase": bp.phase,
        "phaseLabel": PHASES[bp.phase],
        "standard": bp.standard or default_standard,
        "source": bp.source,
        "role": bp.role,
    }
    describe_rule(check, resolve_blueprint(segment, bp, stance))
    return check


def _segment_rank(check: dict[str, Any], order: dict[str, int]) -> int:
    phase = check["phase"]
    if phase == "start":
        return 0
    if phase == "end":
        return 99
    return order[check["segment"]]


def _catalog_rank(check: dict[str, Any], segment_order: dict[str, int], end_rank: int) -> int:
    if check["segment"] == "stance" and check["phase"] == "start":
        return 0
    if check["segment"] == "stance" and check["phase"] == "end":
        return end_rank
    return segment_order[check["segment"]]


def build_checks(action: str, stance: str, *, pose_variant: str = "full") -> list[dict[str, Any]]:
    """所选动作与左右式的检查清单（开始实战式 → 各动作段 → 结束实战式）；多次动作合并在同一检查项里判定。"""
    if (
        not isinstance(action, str)
        or not isinstance(stance, str)
        or action not in ACTIONS
        or stance not in STANCES
    ):
        raise ValueError("请选择支持的动作和左式/右式")
    _ = pose_variant
    segments = list(action_segments(action))
    checks = [_build_check_dict("stance", bp, stance) for bp in STANCE_START_BLUEPRINTS]
    for segment in segments:
        for bp in SEGMENT_CHECK_BLUEPRINTS[segment]:
            checks.append(_build_check_dict(segment, bp, stance))
    checks.extend(_build_check_dict("stance", bp, stance) for bp in STANCE_END_BLUEPRINTS)
    order = {name: i + 1 for i, name in enumerate(segments)}
    checks.sort(
        key=lambda c: (
            _segment_rank(c, order),
            _PHASE_ORDER.index(c["phase"]),
            c["bodyPart"],
            c["id"],
        )
    )
    return checks


def catalog_checks(stance: str = DEFAULT_STANCE) -> list[dict[str, Any]]:
    """全系统全部唯一检查项目录：开始实战式 → 各动作段 → 结束实战式（按 id 去重，不复制规则定义）。"""
    if not isinstance(stance, str) or stance not in STANCES:
        raise ValueError("请选择支持的左式/右式")
    seen: dict[str, dict[str, Any]] = {}
    segment_order: dict[str, int] = {}
    for action in ACTIONS:
        for check in build_checks(action, stance):
            cid = check["id"]
            if cid in seen:
                continue
            seen[cid] = check
            seg = check["segment"]
            if seg != "stance" and seg not in segment_order:
                segment_order[seg] = len(segment_order) + 1
    end_rank = len(segment_order) + 1
    return sorted(seen.values(), key=lambda c: _catalog_rank(c, segment_order, end_rank))


def check_view_weights(check: dict, cfg: FeedbackConfig) -> dict[str, float]:
    """一个检查项的正侧基础权重（单视角公式的项与标定为只用一个视角的项为 1/0）。"""
    primary = (
        {**PRIMARY_VIEW, check["code"]: check["primaryView"]}
        if "primaryView" in check
        else PRIMARY_VIEW
    )
    return practice_view_weights(check["code"], check["segment"], cfg.front_weights, primary)


def rule_profile(action: str, stance: str, cfg: FeedbackConfig) -> RuleProfile:
    """动作引用公共检查定义，所有已登记检查项在全模型下直接正式启用。"""
    rules = []
    for check in build_checks(action, stance, pose_variant=cfg.pose_variant):
        definition = resolve_rule(check, stance)
        rules.append(
            RuleSpec(
                code=check["id"],
                title=definition.name,
                check=definition.check,
                required=definition.required,
                view=definition.view,
                scope=f"{check['segment']}:{check['phase']}",
                aggregation=definition.build_verdict(cfg),
                authorization="candidate",
                source=definition.source,
                strategy=definition.strategy,
                highlight_joints=definition.highlight_joints,
                required_views=definition.required_views,
                evidence_joints_by_view=definition.evidence_joints_by_view,
            )
        )
    return RuleProfile(name=PROFILE_NAME, rules=tuple(rules))
