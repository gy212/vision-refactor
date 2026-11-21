"""公共拳法规则：起动前预摆（下沉、后拉）与收回护颚判定（直拳、摆拳共用）。"""

from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.guard import measure_guard_jaw_distance
from core.evaluation.measures.motion import relative_position
from core.evaluation.measures.torso import chain_torso
from core.evaluation.rules.spec import PhaseCheck, RuleContext
from core.evaluation.rules.window_context import empty_result
from core.pose.landmarks import MOUTH_L, MOUTH_R, NOSE


def punch_landmarks(strategy: str, role: str, stance: str, view: str = "side") -> tuple[int, ...]:
    """返回公共拳法策略所需的关键点下标。"""
    del view
    sh, _, wr, hip, *_ = sides_for_stance(role, stance)
    mapping = {
        "prepunch_wrist_drop": (sh, wr),
        "prepunch_wrist_pull": (sh, wr, hip, NOSE),
        "returned_wrist_jaw_distance": (wr, MOUTH_L, MOUTH_R),
    }
    if strategy not in mapping:
        raise KeyError(f"未知的公共拳法策略: {strategy!r}")
    return mapping[strategy]


def _baseline_relative_wrist(
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    required: tuple[int, ...],
    shoulder: int,
    wrist: int,
    ctx: RuleContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """提取相对起始护位的归一手腕轨迹与有效基线，基线不足时记录原因并返回 None。"""
    torso = chain_torso(lm, mask)
    base_torso = chain_torso(baseline, baseline_mask)
    base_valid = baseline_mask[:, required].all(axis=1) & (base_torso > 1e-4)
    if base_valid.sum() < ctx.params.min_valid:
        ctx.measurement_issues.append(
            f"起始护位参照可用帧不足（{base_valid.sum()}/{len(baseline)}帧，至少{ctx.params.min_valid}帧）"
        )
        return None
    base = baseline[base_valid]
    position, _ = relative_position(lm, mask, torso, shoulder, wrist)
    base_position, _ = relative_position(
        base, baseline_mask[base_valid], base_torso[base_valid], shoulder, wrist
    )
    valid = mask[:, required].all(axis=1) & (torso > 1e-4)
    return position, base_position, base, valid


def measure_punch_predrop(
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """公共拳法起动前下沉预摆判定。"""
    sh, _, wr, *_ = sides_for_stance(role, ctx.stance)
    rel = _baseline_relative_wrist(lm, mask, baseline, baseline_mask, (sh, wr), sh, wr, ctx)
    if rel is None:
        return empty_result(len(lm))
    position, base_position, _, valid = rel
    values = position[:, 1] - np.median(base_position[:, 1])
    return values > ctx.params.punch_drop, valid, {"起动前向下预摆距离": (values, "躯干长度")}


def measure_punch_prepull(
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """公共拳法起动前后拉预摆判定。"""
    sh, _, wr, hip, *_ = sides_for_stance(role, ctx.stance)
    rel = _baseline_relative_wrist(
        lm, mask, baseline, baseline_mask, (sh, wr, hip, NOSE), sh, wr, ctx
    )
    if rel is None:
        return empty_result(len(lm))
    position, base_position, base, valid = rel
    facing = np.sign(np.median(base[:, NOSE, 0] - base[:, hip, 0]))
    values = (np.median(base_position[:, 0]) - position[:, 0]) * facing
    valid &= facing != 0
    if facing == 0:
        ctx.measurement_issues.append("起始护位的鼻与髋部位置不足以确定面朝方向")
    return values > ctx.params.punch_pull, valid, {"起动前向后预摆距离": (values, "躯干长度")}


def measure_punch_recover(
    role: str, lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """公共拳法收回护颚判定，直接复用公共护手护颚测量。"""
    _, _, wr, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    return measure_guard_jaw_distance(
        lm,
        mask,
        torso,
        wr,
        max_dist=ctx.params.punch_recovery,
        label="收回后手腕距口部",
    )


def build_punch_check(strategy: str, role: str, scope: str) -> PhaseCheck:
    """按公共拳法策略名构造对应的相位判定函数。"""
    del scope
    if strategy == "prepunch_wrist_drop":
        return lambda lm, mask, base, base_mask, ctx: measure_punch_predrop(
            role, lm, mask, base, base_mask, ctx
        )
    if strategy == "prepunch_wrist_pull":
        return lambda lm, mask, base, base_mask, ctx: measure_punch_prepull(
            role, lm, mask, base, base_mask, ctx
        )
    if strategy == "returned_wrist_jaw_distance":
        return lambda lm, mask, _b, _bm, ctx: measure_punch_recover(role, lm, mask, ctx)
    raise KeyError(f"未知的公共拳法策略: {strategy!r}")
