"""摆拳独有私有规则：打到瞬间大小臂弯曲度、正面小臂水平角与中线越界判定。"""

from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.hooks import (
    measure_hook_arm_bend,
    measure_hook_forearm_level,
    measure_hook_over_center,
)
from core.evaluation.measures.torso import chain_torso
from core.evaluation.rules.spec import PhaseCheck, RuleContext
from core.evaluation.rules.window_context import empty_result, resolve_window_facing


def hook_landmarks(strategy: str, role: str, stance: str, view: str = "front") -> tuple[int, ...]:
    """返回摆拳独有策略所需的关键点下标。"""
    del view
    sh, el, wr, *_ = sides_for_stance(role, stance)
    if strategy == "hook_forearm_upper_ratio":
        return sh, el, wr
    if strategy == "hook_forearm_horizontal_angle":
        return el, wr
    if strategy == "hook_wrist_over_center":
        opp_role = "rear" if role == "front" else "front"
        opp_sh = sides_for_stance(opp_role, stance)[0]
        return wr, sh, opp_sh
    raise KeyError(f"未知的摆拳独有策略: {strategy!r}")


def measure_hook_arm_bend_rule(
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """摆拳打到瞬间大小臂弯曲度判定（防弯太紧与直臂甩大臂）。"""
    facing = resolve_window_facing(lm, mask, baseline, baseline_mask)
    if facing is None:
        ctx.measurement_issues.append("起始护位与摆拳窗口关键点不足以确定面朝方向")
        return empty_result(len(lm))
    sh, el, wr, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    return measure_hook_arm_bend(
        lm,
        mask,
        torso,
        sh,
        el,
        wr,
        facing,
        tight_max=ctx.params.hook_bend_tight_ratio,
        straight_max=ctx.params.hook_bend_straight_ratio,
    )


def measure_hook_forearm_level_rule(
    role: str, lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """摆拳正面小臂水平度判定。"""
    _, el, wr, *_ = sides_for_stance(role, ctx.stance)
    return measure_hook_forearm_level(
        lm, mask, el, wr, max_angle=ctx.params.hook_forearm_level_angle
    )


def measure_hook_over_center_rule(
    role: str, lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """摆拳正面单侧防摆过中线判定。"""
    att_sh, _, wr, *_ = sides_for_stance(role, ctx.stance)
    opp_role = "rear" if role == "front" else "front"
    opp_sh = sides_for_stance(opp_role, ctx.stance)[0]
    return measure_hook_over_center(
        lm, mask, wr, att_sh, opp_sh, max_ratio=ctx.params.hook_over_center_ratio
    )


def build_hook_check(strategy: str, role: str, scope: str) -> PhaseCheck:
    """按摆拳独有策略名构造对应的相位判定函数。"""
    del scope
    if strategy == "hook_forearm_upper_ratio":
        return lambda lm, mask, base, base_mask, ctx: measure_hook_arm_bend_rule(
            role, lm, mask, base, base_mask, ctx
        )
    if strategy == "hook_forearm_horizontal_angle":
        return lambda lm, mask, _b, _bm, ctx: measure_hook_forearm_level_rule(role, lm, mask, ctx)
    if strategy == "hook_wrist_over_center":
        return lambda lm, mask, _b, _bm, ctx: measure_hook_over_center_rule(role, lm, mask, ctx)
    raise KeyError(f"未知的摆拳独有策略: {strategy!r}")
