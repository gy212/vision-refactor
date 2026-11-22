"""公共拳法规则：起动前预摆、收回护颚、拳肩齐平、峰值后未收回与组合衔接停顿（直拳、摆拳共用）。"""

from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.guard import measure_guard_jaw_distance
from core.evaluation.measures.motion import measure_punch_post_peak_hold, relative_position
from core.evaluation.measures.posture import relative_height
from core.evaluation.measures.torso import chain_torso
from core.evaluation.rules.spec import PhaseCheck, RuleContext
from core.evaluation.rules.window_context import (
    empty_result,
    iter_labeled_instance_slices,
    resolve_window_facing,
    write_segment_details,
)
from core.pose.landmarks import (
    L_SHOULDER,
    L_WRIST,
    MOUTH_L,
    MOUTH_R,
    NOSE,
    R_SHOULDER,
    R_WRIST,
)

_STRAIGHT_RETURN_REACH = 0.12
_HOOK_RETURN_REACH = 0.10


def punch_landmarks(strategy: str, role: str, stance: str, view: str = "side") -> tuple[int, ...]:
    """返回公共拳法策略所需的关键点下标。"""
    del view
    if strategy == "combo_punch_pause_gap":
        return L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST
    sh, _, wr, hip, *_ = sides_for_stance(role, stance)
    mapping = {
        "prepunch_wrist_drop": (sh, wr),
        "prepunch_wrist_pull": (sh, wr, hip, NOSE),
        "returned_wrist_jaw_distance": (wr, MOUTH_L, MOUTH_R),
        "wrist_shoulder_height_gap": (sh, wr),
        "punch_post_peak_recovery": (sh, wr),
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


def measure_wrist_shoulder_height_gap(
    role: str, lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """公共拳法齐肩判定（后手直拳与前后手摆拳共用）：出拳手腕与同侧肩纵向绝对差不超过上限。"""
    sh, _, wr, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    height, valid = relative_height(lm, mask, wr, sh, torso)
    gap = np.abs(height)
    return gap > ctx.params.shoulder_height, valid, {"拳肩高度差": (gap, "躯干长度")}


def measure_punch_unrecovered(
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """公共拳法峰值后未收回判定：按每次出拳核对是否保持未收或滞留超过上限。"""
    n_frames = len(lm)
    if "stance:start" not in ctx.phase_windows:
        ctx.measurement_issues.append("缺少起始护位窗口，无法确定出拳回收基线")
        return empty_result(n_frames)
    sh, _, wr, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    base_torso = chain_torso(baseline, baseline_mask)
    base_ok = baseline_mask[:, [sh, wr]].all(axis=1) & (base_torso > 1e-4)
    if int(base_ok.sum()) < ctx.params.min_valid:
        ctx.measurement_issues.append(
            f"起始护位肩腕参照可用帧不足（{int(base_ok.sum())}/{len(baseline)}帧，至少{ctx.params.min_valid}帧）"
        )
        return empty_result(n_frames)
    facing = resolve_window_facing(lm, mask, baseline, baseline_mask)
    if facing is None:
        ctx.measurement_issues.append("起始护位与出拳窗口关键点不足以确定面朝方向")
        return empty_result(n_frames)
    segment = scope.split(":", 1)[0]
    return_reach = _HOOK_RETURN_REACH if segment.endswith("_hook") else _STRAIGHT_RETURN_REACH
    event_by_rep = {
        int(ev["repetition"]): ev
        for ev in ctx.event_details
        if ev.get("segment") == segment and "repetition" in ev
    }
    bad = np.zeros(n_frames, dtype=bool)
    valid = np.zeros(n_frames, dtype=bool)
    detail_arrays: dict[str, tuple[np.ndarray, str]] = {}
    for rep_id, sub, sub_times in iter_labeled_instance_slices(n_frames, ctx, scope):
        ev = event_by_rep.get(rep_id) if rep_id is not None else None
        if ev is None and rep_id is None and len(event_by_rep) == 1:
            ev = next(iter(event_by_rep.values()))
        held_ending = bool(ev is not None and ev.get("ending") == "held")
        bad_seg, valid_seg, seg_details = measure_punch_post_peak_hold(
            lm[sub],
            mask[sub],
            torso[sub],
            sub_times,
            baseline,
            baseline_mask,
            base_torso,
            sh,
            wr,
            facing,
            hold_max=ctx.params.punch_unrecovered_hold_seconds,
            return_reach=return_reach,
            held_ending=held_ending,
        )
        valid[sub] = valid_seg
        bad[sub] = valid_seg if bad_seg else False
        write_segment_details(detail_arrays, sub, n_frames, seg_details)
    return bad, valid, detail_arrays


def measure_combo_punch_pause(
    lm: np.ndarray, mask: np.ndarray, ctx: RuleContext, scope: str
) -> tuple[np.ndarray, np.ndarray, dict]:
    """组合拳前后手衔接停顿判定：第一拳收完到第二拳起动空档超过上限判停顿，边收边出直接通过。"""
    n_frames = len(lm)
    if not ctx.combo_pairs:
        ctx.measurement_issues.append("未检测到成对的前后手组合衔接，无法测量两拳停顿空档")
        return empty_result(n_frames)
    pair_by_rep = {int(p["repetition"]): p for p in ctx.combo_pairs if "repetition" in p}
    f_sh, _, f_wr, *_ = sides_for_stance("front", ctx.stance)
    r_sh, _, r_wr, *_ = sides_for_stance("rear", ctx.stance)
    torso = chain_torso(lm, mask)
    bad = np.zeros(n_frames, dtype=bool)
    valid = np.zeros(n_frames, dtype=bool)
    detail_arrays: dict[str, tuple[np.ndarray, str]] = {}
    for rep_id, sub, _ in iter_labeled_instance_slices(n_frames, ctx, scope):
        pair = pair_by_rep.get(rep_id) if rep_id is not None else None
        if pair is None and rep_id is None and len(ctx.combo_pairs) == 1:
            pair = ctx.combo_pairs[0]
        if pair is None:
            continue
        pause_s = max(0.0, float(pair["rearStart"]) - float(pair["frontEnd"]))
        sub_mask = mask[sub]
        arm_ok = (sub_mask[:, f_sh] & sub_mask[:, f_wr]) | (sub_mask[:, r_sh] & sub_mask[:, r_wr])
        valid_seg = (torso[sub] > 1e-4) & arm_ok
        bad_seg = bool(pause_s > ctx.params.combo_punch_pause_max)
        valid[sub] = valid_seg
        bad[sub] = valid_seg if bad_seg else False
        write_segment_details(detail_arrays, sub, n_frames, {"前后手衔接停顿": (pause_s, "秒")})
    return bad, valid, detail_arrays


def build_punch_check(strategy: str, role: str, scope: str) -> PhaseCheck:
    """按公共拳法策略名构造对应的相位判定函数。"""
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
    if strategy == "wrist_shoulder_height_gap":
        return lambda lm, mask, _b, _bm, ctx: measure_wrist_shoulder_height_gap(role, lm, mask, ctx)
    if strategy == "punch_post_peak_recovery":
        return lambda lm, mask, base, base_mask, ctx: measure_punch_unrecovered(
            role, lm, mask, base, base_mask, ctx, scope
        )
    if strategy == "combo_punch_pause_gap":
        return lambda lm, mask, _b, _bm, ctx: measure_combo_punch_pause(lm, mask, ctx, scope)
    raise KeyError(f"未知的公共拳法策略: {strategy!r}")
