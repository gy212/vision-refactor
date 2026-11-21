"""直拳独有私有规则：发力端伸肘、起动前贴肋禁抬肘与直拳专用齐肩策略。"""

from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.posture import joint_angle, relative_height
from core.evaluation.measures.torso import chain_torso
from core.evaluation.rules.spec import PhaseCheck, RuleContext


def straight_landmarks(
    strategy: str, role: str, stance: str, view: str = "side"
) -> tuple[int, ...]:
    """返回直拳独有策略所需的关键点下标。"""
    del view
    sh, el, wr, hip, *_ = sides_for_stance(role, stance)
    mapping = {
        "elbow_extension_at_peak": (sh, el, wr),
        "arm_torso_angle": (sh, wr, hip),
        "wrist_shoulder_height_gap": (sh, wr),
        "prepunch_elbow_height": (sh, el),
    }
    if strategy not in mapping:
        raise KeyError(f"未知的直拳独有策略: {strategy!r}")
    return mapping[strategy]


def measure_punch_extension(
    role: str, lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """直拳逐帧肘角测量（供峰值伸展判定调用）。"""
    sh, el, wr, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    values, valid = joint_angle(lm, mask, (sh, el, wr), torso)
    return values < ctx.params.punch_elbow, valid, {"出拳手肘的夹角": (values, "度")}


def measure_extension_at_peak(
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    ctx: RuleContext,
    scope: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """每次直拳打到头附近只需实际伸直过，前后弯曲画面不否定已达到的伸直。

    每次拳单独检查。一次伸直不能替另一次通过；缺失画面仍为无效。
    """
    bad, valid, details = measure_punch_extension(role, lm, mask, ctx)
    target_scope = scope or f"{role}_straight:finish"
    indices = ctx.phase_windows[target_scope]
    boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
    for run in np.split(np.arange(len(indices)), boundaries):
        observed = run[valid[run]]
        if len(observed):
            bad[run] = bad[observed].all()
    return bad, valid, details


def measure_arm_torso_angle(
    role: str, lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """前手直拳专用齐肩判定：侧面看出拳手臂与躯干夹角 90° ± 容差。"""
    sh, _, wr, hip, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    values, valid = joint_angle(lm, mask, (hip, sh, wr), torso)
    return (
        np.abs(values - 90) > ctx.params.punch_level_deviation,
        valid,
        {"出拳手臂与身体的夹角": (values, "度")},
    )


def measure_wrist_shoulder_height_gap(
    role: str, lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """后手直拳专用齐肩判定：出拳手腕与同侧肩纵向绝对差不超过上限。"""
    sh, _, wr, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    height, valid = relative_height(lm, mask, wr, sh, torso)
    gap = np.abs(height)
    return gap > ctx.params.shoulder_height, valid, {"拳肩高度差": (gap, "躯干长度")}


def measure_punch_preelbow(
    role: str, lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """后手直拳独有起动前禁抬肘判定（贴肋直出，不先抬肘）。"""
    sh, el, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    values, valid = relative_height(lm, mask, sh, el, torso)
    return (
        values > ctx.params.punch_elbow_raise,
        valid,
        {"起动前肘高于肩": (values, "躯干长度")},
    )


def build_straight_check(strategy: str, role: str, scope: str) -> PhaseCheck:
    """按直拳独有策略名构造对应的相位判定函数。"""
    if strategy == "elbow_extension_at_peak":
        return lambda lm, mask, _b, _bm, ctx: measure_extension_at_peak(
            role, lm, mask, ctx, scope=scope
        )
    if strategy == "arm_torso_angle":
        return lambda lm, mask, _b, _bm, ctx: measure_arm_torso_angle(role, lm, mask, ctx)
    if strategy == "wrist_shoulder_height_gap":
        return lambda lm, mask, _b, _bm, ctx: measure_wrist_shoulder_height_gap(role, lm, mask, ctx)
    if strategy == "prepunch_elbow_height":
        return lambda lm, mask, _b, _bm, ctx: measure_punch_preelbow(role, lm, mask, ctx)
    raise KeyError(f"未知的直拳独有策略: {strategy!r}")
