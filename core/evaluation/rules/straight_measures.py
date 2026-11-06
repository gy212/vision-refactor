"""直拳起动前预摆与收回护位的逐帧测量，收拳路径不作为掉手错误。"""

from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.arms import distance_to, series_mouth
from core.evaluation.measures.motion import relative_position
from core.evaluation.measures.posture import relative_height
from core.evaluation.measures.torso import chain_torso
from core.evaluation.rules.spec import RuleContext


def straight_landmarks(code: str, role: str, stance: str) -> tuple[int, ...]:
    """新检查项的点位契约；与测量实际取点一致。"""
    shoulder, elbow, wrist, hip, *_ = sides_for_stance(role, stance)
    if code == "punch_preelbow":
        return (shoulder, elbow)
    if code in {"punch_recover", "guard_jaw"}:
        return (wrist, 9, 10)
    if code == "punch_prepull":
        return (shoulder, wrist, hip, 0)
    return (shoulder, wrist)


def measure_straight(
    code: str,
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """在专用阶段上返回问题帧、有效帧和可复核数值。"""
    shoulder, elbow, wrist, hip, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    required = straight_landmarks(code, role, ctx.stance)
    valid = mask[:, required].all(axis=1) & (torso > 1e-4)
    scale = np.maximum(torso, 1e-4)
    cfg = ctx.params

    if code == "punch_preelbow":
        values, _ = relative_height(lm, mask, shoulder, elbow, torso)
        label, threshold = "起动前肘高于肩", cfg.punch_elbow_raise
    elif code in {"punch_recover", "guard_jaw"}:
        values = distance_to(lm, wrist, series_mouth(lm)) / scale
        label = "实战式后手手腕距口部" if code == "guard_jaw" else "收回后手腕距口部"
        threshold = cfg.punch_recovery
    else:
        base_torso = chain_torso(baseline, baseline_mask)
        base_valid = baseline_mask[:, required].all(axis=1) & (base_torso > 1e-4)
        if base_valid.sum() < cfg.min_valid:
            return np.zeros(len(lm), bool), np.zeros(len(lm), bool), {}
        base = baseline[base_valid]
        position, _ = relative_position(lm, mask, torso, shoulder, wrist)
        base_position, _ = relative_position(
            base, baseline_mask[base_valid], base_torso[base_valid], shoulder, wrist
        )
        if code == "punch_prepull":
            facing = np.sign(np.median(base[:, 0, 0] - base[:, hip, 0]))
            initial = np.median(base_position[:, 0])
            values = (initial - position[:, 0]) * facing
            valid &= facing != 0
            label, threshold = "起动前向后预摆距离", cfg.punch_pull
        else:
            initial = np.median(base_position[:, 1])
            values = position[:, 1] - initial
            label, threshold = "起动前向下预摆距离", cfg.punch_drop

    return values > threshold, valid, {label: (values, "躯干长度")}
