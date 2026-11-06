"""上步直拳中按单面画面检查手腕高度、贴身距离与出拳角度。"""

from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.posture import elbow_body_distance, joint_angle, relative_height
from core.evaluation.measures.torso import chain_torso
from core.evaluation.rules.spec import RuleContext


def posture_landmarks(code: str, role: str, stance: str) -> tuple[int, ...]:
    """声明本项实际使用的身体部位。"""
    shoulder, elbow, wrist, hip, *_ = sides_for_stance(role, stance)
    if code == "stance_height":
        return shoulder, wrist
    if code == "stance_rear_guard":
        return shoulder, elbow, hip
    if code == "shoulder_level":
        return shoulder, wrist, hip
    return shoulder, elbow, wrist


def measure_posture(
    code: str,
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    ctx: RuleContext,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """逐张测量；手臂是否伸直与出拳是否齐肩各自判断。"""
    shoulder, elbow, wrist, hip, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    required = posture_landmarks(code, role, ctx.stance)
    valid = mask[:, required].all(axis=1) & (torso > 1e-4)
    if code == "stance_height":
        values, valid = relative_height(lm, mask, wrist, shoulder, torso)
        return values > 0, valid, {"手腕低于肩": (values, "躯干长度")}
    if code == "stance_rear_guard":
        values, valid = elbow_body_distance(lm, mask, shoulder, elbow, hip, torso)
        return values > 0.5, valid, {"肘离身体中部的距离": (values, "躯干长度")}
    if code == "shoulder_level":
        values, valid = joint_angle(lm, mask, (hip, shoulder, wrist), torso)
        return (
            np.abs(values - 90) > ctx.params.punch_level_deviation,
            valid,
            {"出拳手臂与身体的夹角": (values, "度")},
        )
    values, valid = joint_angle(lm, mask, (shoulder, elbow, wrist), torso)
    return values < ctx.params.punch_elbow, valid, {"出拳手肘的夹角": (values, "度")}


def measure_extension_at_peak(
    role: str, lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """每次打到头附近只需实际伸直过，前后弯曲画面不否定已达到的伸直。

    每次拳单独检查。一次伸直不能替另一次通过；缺失画面仍为无效。
    """
    bad, valid, details = measure_posture("punch_extension", role, lm, mask, ctx)
    indices = ctx.phase_windows[f"{role}_straight:finish"]
    boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
    for run in np.split(np.arange(len(indices)), boundaries):
        observed = run[valid[run]]
        if len(observed):
            bad[run] = bad[observed].all()
    return bad, valid, details
