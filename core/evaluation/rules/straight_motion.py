"""直拳动作阶段护手相对口部后撤的逐帧测量。"""
from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.arms import series_mouth
from core.evaluation.measures.torso import chain_torso
from core.evaluation.rules.spec import RuleContext


def guard_back_landmarks(role: str, stance: str) -> tuple[int, ...]:
    """护手后撤所需的腕、口部和基线朝向点。"""
    _, _, wrist, hip, *_ = sides_for_stance(role, stance)
    return wrist, 9, 10, 0, hip


def measure_guard_back(
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """侧面量护手相对口部的有符号后移，起始有效鼻–髋用于定朝向。

    当前帧须有腕、两个嘴角和可见肩髋链，基线须有鼻、同侧髋和可见肩髋链。
    后手护脸相对起始腕–口位置；前手允许从前伸护位收回口部，不减其起始腕位置。
    """
    _, _, wrist, hip, *_ = sides_for_stance(role, ctx.stance)
    base_torso = chain_torso(baseline, baseline_mask)
    base_valid = baseline_mask[:, [0, hip]].all(axis=1) & (base_torso > 1e-4)
    if base_valid.sum() < ctx.params.min_valid:
        return np.zeros(len(lm), bool), np.zeros(len(lm), bool), {}
    direction = np.median(
        (baseline[base_valid, 0, 0] - baseline[base_valid, hip, 0]) / base_torso[base_valid]
    )
    if abs(direction) < .1:
        return np.zeros(len(lm), bool), np.zeros(len(lm), bool), {}

    initial = 0.0
    if role == "rear":
        guard_valid = base_valid & baseline_mask[:, [wrist, 9, 10]].all(axis=1)
        if guard_valid.sum() < ctx.params.min_valid:
            return np.zeros(len(lm), bool), np.zeros(len(lm), bool), {}
        initial = np.median(
            (series_mouth(baseline)[guard_valid, 0] - baseline[guard_valid, wrist, 0])
            * np.sign(direction) / base_torso[guard_valid]
        )
    torso = chain_torso(lm, mask)
    valid = mask[:, [wrist, 9, 10]].all(axis=1) & (torso > 1e-4)
    values = (series_mouth(lm)[:, 0] - lm[:, wrist, 0]) * np.sign(direction) / np.maximum(torso, 1e-4) - initial
    label = "护手相对起始口部位置后移" if role == "rear" else "护手后于口部"
    return values > ctx.params.guard_back, valid, {label: (values, "躯干长度")}
