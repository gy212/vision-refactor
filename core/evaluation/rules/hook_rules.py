"""摆拳独有私有规则：横摆发力时拳与肘同时齐肩判定。"""

from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.posture import relative_height
from core.evaluation.measures.torso import chain_torso
from core.evaluation.rules.spec import PhaseCheck, RuleContext


def hook_landmarks(strategy: str, role: str, stance: str, view: str = "any") -> tuple[int, ...]:
    """返回摆拳独有策略所需的关键点下标。"""
    del view
    if strategy != "hook_arm_shoulder_level":
        raise KeyError(f"未知的摆拳独有策略: {strategy!r}")
    sh, el, wr, *_ = sides_for_stance(role, stance)
    return sh, el, wr


def measure_hook_shoulder_level(
    role: str, lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """摆拳齐肩判定：打到瞬间拳与肘均应大致与肩齐平，缺任一点均记为无效帧。"""
    sh, el, wr, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    wrist_gap, _ = relative_height(lm, mask, wr, sh, torso)
    elbow_gap, _ = relative_height(lm, mask, el, sh, torso)
    wrist_abs = np.abs(wrist_gap)
    elbow_abs = np.abs(elbow_gap)
    details = {
        "拳肩高度差": (wrist_abs, "躯干长度"),
        "肘肩高度差": (elbow_abs, "躯干长度"),
    }
    bad = (wrist_abs > ctx.params.shoulder_height) | (elbow_abs > ctx.params.shoulder_height)
    valid = mask[:, (sh, el, wr)].all(axis=1) & (torso > 1e-4)
    return bad, valid, details


def build_hook_check(strategy: str, role: str, scope: str) -> PhaseCheck:
    """按摆拳独有策略名构造对应的相位判定函数。"""
    del scope
    if strategy == "hook_arm_shoulder_level":
        return lambda lm, mask, _b, _bm, ctx: measure_hook_shoulder_level(role, lm, mask, ctx)
    raise KeyError(f"未知的摆拳独有策略: {strategy!r}")
