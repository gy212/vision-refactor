"""通用人体姿态、转体与脚跟蹬转规则策略（低头、侧倾、前后倾、转胯、碾脚内旋与出拳提踵）。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.arms import segment_lengths
from core.evaluation.measures.feet import measure_pivot_heel, measure_punch_heel_pivot
from core.evaluation.measures.posture import (
    measure_body_lean,
    measure_head_down,
    measure_hip_turn_width,
    measure_trunk_lean_sagittal,
)
from core.evaluation.measures.torso import chain_torso
from core.evaluation.rules.spec import PhaseCheck, RuleContext
from core.evaluation.rules.window_context import empty_result, resolve_window_facing
from core.pose.landmarks import L_HIP, L_SHOULDER, NOSE, R_HIP, R_SHOULDER

MaskFilter = Callable[[str, np.ndarray, np.ndarray, np.ndarray, RuleContext, str], np.ndarray]


def body_posture_landmarks(
    strategy: str, role: str, stance: str, view: str = "side"
) -> tuple[int, ...]:
    """返回通用姿态与步法转体策略所需的关键点下标。"""
    del view
    if strategy == "nose_shoulder_drop":
        return NOSE, L_SHOULDER, R_SHOULDER
    if strategy == "shoulder_line_tilt_angle":
        return L_SHOULDER, R_SHOULDER
    if strategy == "trunk_sagittal_tilt_angle":
        return L_SHOULDER, R_SHOULDER, L_HIP, R_HIP
    if strategy == "frontal_hip_width_ratio":
        return L_HIP, R_HIP
    if strategy in {"support_foot_turn_projection", "punch_heel_lift_pivot"}:
        _, _, _, _, _, _, foot_hl, foot_toe = sides_for_stance(role, stance)
        return foot_hl, foot_toe
    raise KeyError(f"未知的通用人体姿态策略: {strategy!r}")


def measure_head_down_rule(
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """低头规则判定：相对起始护位鼻肩垂向位置下压超过阈值判低头。"""
    if "stance:start" not in ctx.phase_windows:
        ctx.measurement_issues.append("缺少起始护位窗口，无法确定正面翻胯或低头基线")
        return empty_result(len(lm))
    torso = chain_torso(lm, mask)
    base_torso = chain_torso(baseline, baseline_mask)
    base_ok = baseline_mask[:, [NOSE, L_SHOULDER, R_SHOULDER]].all(axis=1) & (base_torso > 1e-4)
    if int(base_ok.sum()) < ctx.params.min_valid:
        ctx.measurement_issues.append(
            f"起始护位头肩参照可用帧不足（{int(base_ok.sum())}/{len(baseline)}帧，至少{ctx.params.min_valid}帧）"
        )
        return empty_result(len(lm))
    return measure_head_down(
        lm,
        mask,
        torso,
        baseline,
        baseline_mask,
        base_torso,
        max_drop=ctx.params.head_down_drop,
    )


def measure_body_lean_rule(
    lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """上体侧倾规则判定：正面双肩连线倾斜角超过阈值判侧倾。"""
    return measure_body_lean(lm, mask, max_angle=ctx.params.body_lean_angle)


def measure_trunk_lean_sagittal_rule(
    lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """侧面躯干前后倾规则判定：肩中点到髋中点连线相对竖直线倾角超过阈值判前倾或后仰。"""
    return measure_trunk_lean_sagittal(lm, mask, max_angle=ctx.params.trunk_sagittal_lean_angle)


def measure_hip_turn_rule(
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """转胯/翻胯规则判定：在正交变窄的指定视角相对起始护位检查双髋水平宽收窄。"""
    if "stance:start" not in ctx.phase_windows:
        ctx.measurement_issues.append("缺少起始护位窗口，无法确定正面翻胯或低头基线")
        return empty_result(len(lm))
    view_word = "正面" if ctx.view == "front" else "侧面"
    torso = chain_torso(lm, mask)
    base_torso = chain_torso(baseline, baseline_mask)
    base_ok = baseline_mask[:, L_HIP] & baseline_mask[:, R_HIP] & (base_torso > 1e-4)
    if int(base_ok.sum()) < ctx.params.min_valid:
        ctx.measurement_issues.append(
            f"起始护位{view_word}两髋参照可用帧不足（{int(base_ok.sum())}/{len(baseline)}帧，至少{ctx.params.min_valid}帧）"
        )
        return empty_result(len(lm))
    return measure_hip_turn_width(
        lm,
        mask,
        torso,
        baseline,
        baseline_mask,
        base_torso,
        max_ratio=ctx.params.hip_turn_width_ratio,
        label=f"{view_word}两髋水平宽比值",
    )


def measure_pivot_heel_rule(
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
    scope: str,
    *,
    mask_filter: MaskFilter | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """支撑脚碾脚内旋规则判定；支持由特定动作挂载可选掩码过滤钩子。"""
    torso = chain_torso(lm, mask)
    safe_mask = mask if mask_filter is None else mask_filter(role, lm, mask, torso, ctx, scope)
    facing = resolve_window_facing(lm, mask, baseline, baseline_mask)
    if facing is None:
        ctx.measurement_issues.append("起始护位与支撑脚窗口关键点不足以确定面朝方向")
        return empty_result(len(lm))
    _, _, _, _, _, _, sup_hl, sup_toe = sides_for_stance(role, ctx.stance)
    return measure_pivot_heel(lm, safe_mask, torso, sup_hl, sup_toe, facing)


def measure_punch_heel_pivot_rule(
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """拳法发力脚蹬转提踵判定：相对起始护位基线检查侧面提踵增量。"""
    if "stance:start" not in ctx.phase_windows:
        ctx.measurement_issues.append("缺少起始护位窗口，无法确定发力脚提踵基线")
        return empty_result(len(lm))
    _, _, _, _, _, _, foot_hl, foot_toe = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    base_torso = chain_torso(baseline, baseline_mask)
    base_ok = (
        baseline_mask[:, [foot_hl, foot_toe]].all(axis=1)
        & (base_torso > 1e-4)
        & (segment_lengths(baseline, foot_hl, foot_toe) > 1e-4)
    )
    if int(base_ok.sum()) < ctx.params.min_valid:
        ctx.measurement_issues.append(
            f"起始护位发力脚参照可用帧不足（{int(base_ok.sum())}/{len(baseline)}帧，至少{ctx.params.min_valid}帧）"
        )
        return empty_result(len(lm))
    return measure_punch_heel_pivot(
        lm,
        mask,
        torso,
        baseline,
        baseline_mask,
        base_torso,
        foot_hl,
        foot_toe,
        min_lift=ctx.params.punch_heel_lift_min,
    )


def build_body_posture_check(
    strategy: str,
    role: str,
    scope: str,
    *,
    mask_filter: MaskFilter | None = None,
) -> PhaseCheck:
    """按通用人体姿态与转体策略名构造对应的相位判定函数。"""
    if strategy == "nose_shoulder_drop":
        return lambda lm, mask, base, base_mask, ctx: measure_head_down_rule(
            lm, mask, base, base_mask, ctx
        )
    if strategy == "shoulder_line_tilt_angle":
        return lambda lm, mask, _b, _bm, ctx: measure_body_lean_rule(lm, mask, ctx)
    if strategy == "trunk_sagittal_tilt_angle":
        return lambda lm, mask, _b, _bm, ctx: measure_trunk_lean_sagittal_rule(lm, mask, ctx)
    if strategy == "frontal_hip_width_ratio":
        return lambda lm, mask, base, base_mask, ctx: measure_hip_turn_rule(
            lm, mask, base, base_mask, ctx
        )
    if strategy == "support_foot_turn_projection":
        return lambda lm, mask, base, base_mask, ctx: measure_pivot_heel_rule(
            role,
            lm,
            mask,
            base,
            base_mask,
            ctx,
            scope,
            mask_filter=mask_filter,
        )
    if strategy == "punch_heel_lift_pivot":
        return lambda lm, mask, base, base_mask, ctx: measure_punch_heel_pivot_rule(
            role, lm, mask, base, base_mask, ctx
        )
    raise KeyError(f"未知的通用人体姿态策略: {strategy!r}")
