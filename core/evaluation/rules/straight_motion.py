"""直拳护手肘退到身体后方的测量。"""

from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.arms import segment_lengths
from core.evaluation.measures.torso import body_forward, chain_torso
from core.evaluation.rules.spec import RuleContext


def guard_back_landmarks(role: str, stance: str, view: str = "side") -> tuple[int, ...]:
    """肩到腿根确定身体方向，起始鼻和腿根确定面朝哪边。"""
    shoulder, elbow, _, hip, *_ = sides_for_stance(role, stance)
    if view == "front":
        return 0, 11, 12, elbow, 23, 24
    return shoulder, elbow, 0, hip


def _front_guard_back(role, lm, mask, baseline, baseline_mask, ctx):
    """正面看清远侧护手肘后，试算肘是否退到身体后方。"""
    shoulder, elbow, _, _, *_ = sides_for_stance(role, ctx.stance)
    required = guard_back_landmarks(role, ctx.stance, "front")
    base_direction, base_length, _ = body_forward(baseline)
    base_valid = baseline_mask[:, [0, 11, 12, 23, 24]].all(axis=1) & (base_length > 1e-4)
    if base_valid.sum() < ctx.params.min_valid:
        return np.zeros(len(lm), bool), np.zeros(len(lm), bool), {}
    hips = (baseline[:, 23, :3] + baseline[:, 24, :3]) / 2
    facing = np.sign(
        np.median(np.sum((baseline[:, 0, :3] - hips) * base_direction, axis=1)[base_valid])
    )
    direction, length, torso = body_forward(lm)
    values = np.sum((lm[:, shoulder, :3] - lm[:, elbow, :3]) * direction, axis=1)
    values *= facing / np.maximum(torso, 1e-4)
    valid = mask[:, required].all(axis=1) & (length > 1e-4) & (torso > 1e-4) & (facing != 0)
    return values > ctx.params.guard_back, valid, {"估计护手肘退到身体后方": (values, "身体长度")}


def measure_guard_back(
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """侧面量肘到肩髋线后方的距离，随身体倾斜修正前后方向。

    护手从身前收回到身旁本身不是后拉。只有肘明显退到身体后方才提示，
    不把上步、身体倾斜或从身前正常收回的路程当成后拉。看不到肘时保持无法判断。
    """
    if ctx.view == "front":
        return _front_guard_back(role, lm, mask, baseline, baseline_mask, ctx)
    shoulder, elbow, _, hip, *_ = sides_for_stance(role, ctx.stance)
    base_torso = chain_torso(baseline, baseline_mask)
    base_valid = baseline_mask[:, [0, hip]].all(axis=1) & (base_torso > 1e-4)
    if base_valid.sum() < ctx.params.min_valid:
        return np.zeros(len(lm), bool), np.zeros(len(lm), bool), {}
    direction = np.median(
        (baseline[base_valid, 0, 0] - baseline[base_valid, hip, 0]) / base_torso[base_valid]
    )
    if abs(direction) < 0.1:
        return np.zeros(len(lm), bool), np.zeros(len(lm), bool), {}
    torso = chain_torso(lm, mask)
    body = lm[:, hip, :2] - lm[:, shoulder, :2]
    body_length = segment_lengths(lm, shoulder, hip)
    forward = np.stack((body[:, 1], -body[:, 0]), axis=1) * np.sign(direction)
    forward /= np.maximum(body_length[:, None], 1e-4)
    values = np.sum((lm[:, shoulder, :2] - lm[:, elbow, :2]) * forward, axis=1)
    values /= np.maximum(torso, 1e-4)
    valid = mask[:, [shoulder, elbow, hip]].all(axis=1) & (torso > 1e-4) & (body_length > 1e-4)
    return values > ctx.params.guard_back, valid, {"护手肘退到身体后方": (values, "躯干长度")}
