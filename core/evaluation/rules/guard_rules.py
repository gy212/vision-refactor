"""公共护手规则的统一视角规律、点位契约与 5 项公共护手测量策略。"""

from __future__ import annotations

from typing import Literal

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.guard import (
    measure_guard_elbow_behind_front,
    measure_guard_elbow_behind_side,
    measure_guard_elbow_flare,
    measure_guard_elbow_height,
    measure_guard_jaw_distance,
    measure_guard_shoulder_drop,
)
from core.evaluation.measures.torso import chain_torso
from core.evaluation.rules.spec import PhaseCheck, RuleContext
from core.pose.landmarks import (
    L_HIP,
    L_SHOULDER,
    MOUTH_L,
    MOUTH_R,
    NOSE,
    R_HIP,
    R_SHOULDER,
)


def guard_view(guard_role: str) -> Literal["side", "front"]:
    """公共护手唯一视角规律：后侧护手看侧面，前侧护手看正面。"""
    return "side" if guard_role == "rear" else "front"


def guard_back_landmarks(role: str, stance: str, view: str = "side") -> tuple[int, ...]:
    """肩到腿根确定身体方向，起始鼻和腿根确定面朝哪边。"""
    shoulder, elbow, _, hip, *_ = sides_for_stance(role, stance)
    if view == "front":
        return NOSE, L_SHOULDER, R_SHOULDER, elbow, L_HIP, R_HIP
    return shoulder, elbow, NOSE, hip


def guard_landmarks(strategy: str, role: str, stance: str, view: str = "side") -> tuple[int, ...]:
    """返回公共护手策略所需的关键点下标。"""
    if strategy == "elbow_behind_body":
        return guard_back_landmarks(role, stance, view)
    sh, el, wr, hip, *_ = sides_for_stance(role, stance)
    mapping = {
        "wrist_shoulder_drop": (wr, sh),
        "elbow_shoulder_height": (sh, el),
        "upper_arm_torso_flare": (el, sh, hip),
        "guard_wrist_jaw_distance": (wr, MOUTH_L, MOUTH_R),
    }
    if strategy not in mapping:
        raise KeyError(f"未知的公共护手策略: {strategy!r}")
    return mapping[strategy]


def measure_guard_shoulder(
    role: str, lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """公共护手掉肩判定：手腕低于同侧肩超过配置阈值判掉落。"""
    sh, _, wr, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    return measure_guard_shoulder_drop(
        lm, mask, torso, wr, sh, max_drop=ctx.params.guard_shoulder_drop
    )


def measure_guard_back(
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """公共护手后拉判定：护手从身前收回到身旁不判错，肘明显退到身体后方才提示。"""
    shoulder, elbow, _, hip, *_ = sides_for_stance(role, ctx.stance)
    if ctx.view == "front":
        required = guard_back_landmarks(role, ctx.stance, "front")
        return measure_guard_elbow_behind_front(
            lm,
            mask,
            baseline,
            baseline_mask,
            shoulder,
            elbow,
            required,
            max_back=ctx.params.guard_back,
            min_valid=ctx.params.min_valid,
            issues=ctx.measurement_issues,
        )
    torso = chain_torso(lm, mask)
    base_torso = chain_torso(baseline, baseline_mask)
    return measure_guard_elbow_behind_side(
        lm,
        mask,
        torso,
        baseline,
        baseline_mask,
        base_torso,
        shoulder,
        elbow,
        hip,
        max_back=ctx.params.guard_back,
        min_valid=ctx.params.min_valid,
        issues=ctx.measurement_issues,
    )


def measure_guard_elbow(
    role: str, lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """公共护手抬肘高度判定：肘相对肩高度超过配置阈值判抬肘。"""
    sh, el, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    return measure_guard_elbow_height(lm, mask, torso, sh, el, max_raise=ctx.params.elbow_raise)


def measure_guard_flare(
    role: str, lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """公共护手正面开肘亮肋判定：大臂与躯干夹角大于上限判开肘。"""
    sh, el, _, hip, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    return measure_guard_elbow_flare(
        lm, mask, torso, el, sh, hip, max_flare=ctx.params.kick_guard_elbow_flare
    )


def measure_guard_jaw(
    role: str, lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """公共护手护颚判定：护手腕距口部超过上限判未护住下颚。"""
    _, _, wr, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    return measure_guard_jaw_distance(lm, mask, torso, wr, max_dist=ctx.params.punch_recovery)


def build_guard_check(strategy: str, role: str, scope: str) -> PhaseCheck:
    """按公共护手策略名构造对应的相位判定函数。"""
    del scope
    if strategy == "wrist_shoulder_drop":
        return lambda lm, mask, _b, _bm, ctx: measure_guard_shoulder(role, lm, mask, ctx)
    if strategy == "elbow_behind_body":
        return lambda lm, mask, base, base_mask, ctx: measure_guard_back(
            role, lm, mask, base, base_mask, ctx
        )
    if strategy == "elbow_shoulder_height":
        return lambda lm, mask, _b, _bm, ctx: measure_guard_elbow(role, lm, mask, ctx)
    if strategy == "upper_arm_torso_flare":
        return lambda lm, mask, _b, _bm, ctx: measure_guard_flare(role, lm, mask, ctx)
    if strategy == "guard_wrist_jaw_distance":
        return lambda lm, mask, _b, _bm, ctx: measure_guard_jaw(role, lm, mask, ctx)
    raise KeyError(f"未知的公共护手策略: {strategy!r}")
