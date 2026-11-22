# -*- coding: utf-8 -*-
"""公共步法规则：跟步/垫步前推位移、后脚越前、双摄侧身并脚与落脚同步判定。"""

from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.feet import (
    measure_foot_forward_displacement,
    measure_rear_foot_overtake,
    measure_step_feet_together,
    measure_step_landing_frames,
)
from core.evaluation.rules.spec import PhaseCheck, RuleContext
from core.evaluation.rules.window_context import (
    empty_result,
    iter_instance_slices,
    iter_labeled_instance_slices,
    resolve_window_facing,
    write_segment_details,
)
from core.pose.landmarks import L_HEEL, L_SHOULDER, R_HEEL, R_SHOULDER


def step_landmarks(strategy: str, role: str, stance: str, view: str = "side") -> tuple[int, ...]:
    """返回公共步法策略所需的关键点下标。"""
    del view
    if strategy == "foot_forward_displacement":
        return (sides_for_stance(role or "rear", stance)[5],)
    if strategy == "rear_foot_overtake":
        return sides_for_stance("rear", stance)[5], sides_for_stance("front", stance)[5]
    if strategy == "step_feet_together_paired":
        return L_HEEL, R_HEEL, L_SHOULDER, R_SHOULDER
    if strategy == "step_landing_sync":
        return sides_for_stance("front", stance)[5], sides_for_stance("rear", stance)[5]
    raise KeyError(f"未知的公共步法策略: {strategy!r}")


def measure_foot_displacement_rule(
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """公共后脚向前推进判定（pad_step 垫步与 step_footwork 上步跟步共享底层测量）。"""
    n_frames = len(lm)
    facing = resolve_window_facing(lm, mask, baseline, baseline_mask)
    if facing is None:
        ctx.measurement_issues.append("起始护位与步法窗口关键点不足以确定面朝方向")
        return empty_result(n_frames)
    is_pad = scope.startswith("pad_step:")
    min_disp = ctx.params.pad_step_min_disp if is_pad else ctx.params.step_follow_min_disp
    label = "垫步后脚前推距离" if is_pad else "后脚跟步前推距离"
    ankle = sides_for_stance(role or "rear", ctx.stance)[5]
    bad = np.zeros(n_frames, dtype=bool)
    valid = np.zeros(n_frames, dtype=bool)
    detail_arrays: dict[str, tuple[np.ndarray, str]] = {}
    for sub, sub_times in iter_instance_slices(n_frames, ctx, scope):
        bad_seg, valid_seg, seg_details = measure_foot_forward_displacement(
            lm[sub],
            mask[sub],
            sub_times,
            ankle,
            min_disp=min_disp,
            facing=facing,
            label=label,
        )
        valid[sub] = valid_seg
        bad[sub] = valid_seg if bad_seg else False
        write_segment_details(detail_arrays, sub, n_frames, seg_details)
    return bad, valid, detail_arrays


def measure_rear_foot_overtake_rule(
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """公共后脚超越前脚判定（pad_step 用窗尾中位，step_footwork 用窗内峰值中位）。"""
    n_frames = len(lm)
    facing = resolve_window_facing(lm, mask, baseline, baseline_mask)
    if facing is None:
        ctx.measurement_issues.append("起始护位与步法窗口关键点不足以确定面朝方向")
        return empty_result(n_frames)
    is_pad = scope.startswith("pad_step:")
    overtake_max = ctx.params.pad_step_overtake_max if is_pad else ctx.params.step_overtake_max
    mode = "tail_median" if is_pad else "peak_median"
    label = "垫步末端后脚越前距离" if is_pad else "跟步后脚越前距离"
    rear_ankle = sides_for_stance("rear", ctx.stance)[5]
    front_ankle = sides_for_stance("front", ctx.stance)[5]
    bad = np.zeros(n_frames, dtype=bool)
    valid = np.zeros(n_frames, dtype=bool)
    detail_arrays: dict[str, tuple[np.ndarray, str]] = {}
    for sub, _ in iter_instance_slices(n_frames, ctx, scope):
        bad_seg, valid_seg, seg_details = measure_rear_foot_overtake(
            lm[sub],
            mask[sub],
            rear_ankle,
            front_ankle,
            overtake_max=overtake_max,
            facing=facing,
            mode=mode,
            label=label,
        )
        valid[sub] = valid_seg
        bad[sub] = valid_seg if bad_seg else False
        write_segment_details(detail_arrays, sub, n_frames, seg_details)
    return bad, valid, detail_arrays


def measure_step_feet_together_rule(
    lm: np.ndarray, mask: np.ndarray, ctx: RuleContext, scope: str
) -> tuple[np.ndarray, np.ndarray, dict]:
    """双摄协同上步并脚判定：仅对双摄确认的侧身帧测两脚跟水平距，非侧身帧不判违规。"""
    if ctx.paired_torso is None:
        ctx.measurement_issues.append("缺少同步正面肩宽参照，无法区分侧身帧")
        return empty_result(len(lm))
    indices = ctx.phase_windows[scope]
    bad, valid, details = measure_step_feet_together(
        lm,
        mask,
        ctx.paired_torso,
        indices,
        min_ratio=ctx.params.step_feet_together_ratio,
    )
    if int(valid.sum()) < ctx.params.min_valid:
        ctx.measurement_issues.append(
            f"双摄确认的侧身有效帧不足（{int(valid.sum())}/{len(lm)}帧），非侧身帧不参与并脚判定"
        )
    return bad, valid, details


def measure_step_landing_sync_rule(
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """上步落脚同步判定：逐次核对对应脚步落稳时间相对出拳峰值的正向滞后。"""
    n_frames = len(lm)
    facing = resolve_window_facing(lm, mask, baseline, baseline_mask)
    if facing is None:
        ctx.measurement_issues.append("起始护位与步法窗口关键点不足以确定面朝方向")
        return empty_result(n_frames)
    if not ctx.repetitions:
        ctx.measurement_issues.append("缺少上步组合逐次出拳峰值时间，无法核对落脚同步")
        return empty_result(n_frames)
    rep_by_index = {int(rep["index"]): rep for rep in ctx.repetitions if "index" in rep}
    front_ankle = sides_for_stance("front", ctx.stance)[5]
    rear_ankle = sides_for_stance("rear", ctx.stance)[5]
    bad = np.zeros(n_frames, dtype=bool)
    valid = np.zeros(n_frames, dtype=bool)
    detail_arrays: dict[str, tuple[np.ndarray, str]] = {}
    for rep_id, sub, sub_times in iter_labeled_instance_slices(n_frames, ctx, scope):
        rep = rep_by_index.get(rep_id) if rep_id is not None else None
        if rep is None and rep_id is None and len(ctx.repetitions) == 1:
            rep = ctx.repetitions[0]
        if rep is None:
            continue
        bad_seg, valid_seg, seg_details = measure_step_landing_frames(
            lm[sub],
            mask[sub],
            sub_times,
            front_ankle,
            rear_ankle,
            front_peak=rep.get("frontPeak"),
            rear_peak=rep.get("rearPeak"),
            max_lag=ctx.params.step_late_landing_max,
            facing=facing,
            min_valid_ratio=ctx.params.valid_ratio,
        )
        valid[sub] = valid_seg
        bad[sub] = bad_seg
        write_segment_details(detail_arrays, sub, n_frames, seg_details)
    return bad, valid, detail_arrays


def build_step_check(strategy: str, role: str, scope: str) -> PhaseCheck:
    """按公共步法策略名构造对应的相位判定函数。"""
    if strategy == "foot_forward_displacement":
        return lambda lm, mask, base, base_mask, ctx: measure_foot_displacement_rule(
            role, lm, mask, base, base_mask, ctx, scope
        )
    if strategy == "rear_foot_overtake":
        return lambda lm, mask, base, base_mask, ctx: measure_rear_foot_overtake_rule(
            lm, mask, base, base_mask, ctx, scope
        )
    if strategy == "step_feet_together_paired":
        return lambda lm, mask, _b, _bm, ctx: measure_step_feet_together_rule(lm, mask, ctx, scope)
    if strategy == "step_landing_sync":
        return lambda lm, mask, base, base_mask, ctx: measure_step_landing_sync_rule(
            lm, mask, base, base_mask, ctx, scope
        )
    raise KeyError(f"未知的公共步法策略: {strategy!r}")
