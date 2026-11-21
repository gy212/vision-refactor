"""通用人体姿态、转体与支撑脚碾转规则策略（低头、上体侧倾、正面翻胯、碾脚外旋）。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.feet import measure_pivot_heel
from core.evaluation.measures.posture import (
    measure_body_lean,
    measure_head_down,
    measure_hip_turn_width,
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
    if strategy == "frontal_hip_width_ratio":
        return L_HIP, R_HIP
    if strategy == "support_foot_turn_projection":
        _, _, _, _, _, _, sup_hl, sup_toe = sides_for_stance(role, stance)
        return sup_hl, sup_toe
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


def measure_hip_turn_rule(
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """正面翻胯规则判定：相对起始护位双髋水平宽未充分收窄判未翻胯。"""
    if "stance:start" not in ctx.phase_windows:
        ctx.measurement_issues.append("缺少起始护位窗口，无法确定正面翻胯或低头基线")
        return empty_result(len(lm))
    torso = chain_torso(lm, mask)
    base_torso = chain_torso(baseline, baseline_mask)
    base_ok = baseline_mask[:, L_HIP] & baseline_mask[:, R_HIP] & (base_torso > 1e-4)
    if int(base_ok.sum()) < ctx.params.min_valid:
        ctx.measurement_issues.append(
            f"起始护位正面两髋参照可用帧不足（{int(base_ok.sum())}/{len(baseline)}帧，至少{ctx.params.min_valid}帧）"
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
    """支撑脚碾脚外旋规则判定；支持由特定动作挂载可选掩码过滤钩子。"""
    torso = chain_torso(lm, mask)
    safe_mask = mask if mask_filter is None else mask_filter(role, lm, mask, torso, ctx, scope)
    facing = resolve_window_facing(lm, mask, baseline, baseline_mask)
    if facing is None:
        ctx.measurement_issues.append("起始护位与支撑脚窗口关键点不足以确定面朝方向")
        return empty_result(len(lm))
    _, _, _, _, _, _, sup_hl, sup_toe = sides_for_stance(role, ctx.stance)
    return measure_pivot_heel(lm, safe_mask, torso, sup_hl, sup_toe, facing)


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
    raise KeyError(f"未知的通用人体姿态策略: {strategy!r}")
