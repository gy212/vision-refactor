"""按重复次数切片执行垫步与收腿序列规则，并处理侧视结束段互换屏蔽。"""

from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import facing_direction, sides_for_stance
from core.evaluation.measures.kicks import (
    kicking_leg_side_safe,
    measure_kick_retract_segment,
    measure_pad_step_missing,
    measure_pad_step_overtake,
    measure_pad_step_pause,
)
from core.evaluation.measures.torso import chain_torso
from core.evaluation.rules.spec import RuleContext
from core.foundation.signal import contiguous_runs

_PAD_STEP_DISPATCH = {
    "pad_step_displacement": measure_pad_step_missing,
    "pad_step_ankle_overtake": measure_pad_step_overtake,
    "pad_step_settle_pause": measure_pad_step_pause,
}


def resolve_window_facing(
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
) -> float | None:
    """优先依据起始护位确定面朝方向，缺失时回退到当前窗逐帧投票。"""
    base_dirs = [
        d
        for row, m_row in zip(baseline, baseline_mask)
        if (d := facing_direction(row, m_row)) is not None
    ]
    if len(base_dirs) >= 3:
        return 1.0 if sum(base_dirs) >= 0.0 else -1.0
    win_dirs = [
        d
        for row, m_row in zip(lm, mask)
        if (d := facing_direction(row, m_row)) is not None
    ]
    if not win_dirs:
        return None
    return 1.0 if sum(win_dirs) >= 0.0 else -1.0


def iter_instance_slices(
    n_frames: int,
    ctx: RuleContext,
    scope: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """根据 phase_windows 与 phase_instances 提取每次重复动作的局部切片及时间戳。"""
    if n_frames == 0:
        return []
    win_orig = ctx.phase_windows.get(scope)
    if win_orig is None or len(win_orig) != n_frames:
        win_orig = np.arange(n_frames, dtype=int)
    else:
        win_orig = np.asarray(win_orig, dtype=int)
    fps = (
        float(ctx.params.sample_fps)
        if ctx.params and ctx.params.sample_fps > 0
        else 30.0
    )
    if (
        ctx.times is not None
        and win_orig.size
        and int(np.max(win_orig)) < len(ctx.times)
    ):
        win_times = np.asarray(ctx.times[win_orig], dtype=float)
    else:
        win_times = np.asarray(win_orig, dtype=float) / fps
    instances = ctx.phase_instances.get(scope, ())
    if not instances:
        return [(np.arange(n_frames, dtype=int), win_times)]
    slices: list[tuple[np.ndarray, np.ndarray]] = []
    covered = np.zeros(n_frames, dtype=bool)
    for _, frames in instances:
        sub = np.flatnonzero(np.isin(win_orig, frames))
        if sub.size:
            covered[sub] = True
            slices.append((sub, win_times[sub]))
    for run in contiguous_runs(win_orig[~covered]):
        sub = np.flatnonzero(~covered & (win_orig >= run[0]) & (win_orig <= run[-1]))
        if sub.size:
            slices.append((sub, win_times[sub]))
    return slices or [(np.arange(n_frames, dtype=int), win_times)]


def apply_finish_anti_swap(
    lm: np.ndarray,
    mask: np.ndarray,
    kick_ankle: int,
    sup_ankle: int,
    torso: np.ndarray,
    ctx: RuleContext,
    scope: str,
) -> np.ndarray:
    """过滤侧视左右腿互换帧；finish 阶段按次若超半数互换则整次置无效。"""
    if not scope.endswith(":finish"):
        return kicking_leg_side_safe(lm, mask, kick_ankle, sup_ankle, torso)
    safe_mask = mask.copy()
    for sub, _ in iter_instance_slices(len(lm), ctx, scope):
        safe_mask[sub] = kicking_leg_side_safe(
            lm[sub],
            mask[sub],
            kick_ankle,
            sup_ankle,
            torso[sub],
            majority_threshold=0.5,
        )
    return safe_mask


def _write_segment_details(
    detail_arrays: dict[str, tuple[np.ndarray, str]],
    sub: np.ndarray,
    n_frames: int,
    seg_details: dict[str, tuple[float, str]],
) -> None:
    """将单次动作的标量测量值回写到对应子切片序列以适配引擎摘要统计。"""
    for name, (val, unit) in seg_details.items():
        if name not in detail_arrays:
            detail_arrays[name] = (np.full(n_frames, float("nan"), dtype=float), unit)
        detail_arrays[name][0][sub] = float(val)


def measure_pad_step_rule(
    strategy: str,
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """逐次执行后脚垫步规则并将结果回写为引擎可用的逐帧布尔与数值序列。"""
    n_frames = len(lm)
    bad = np.zeros(n_frames, dtype=bool)
    valid = np.zeros(n_frames, dtype=bool)
    detail_arrays: dict[str, tuple[np.ndarray, str]] = {}
    facing = resolve_window_facing(lm, mask, baseline, baseline_mask)
    if facing is None:
        ctx.measurement_issues.append("起始护位与垫步窗口关键点不足以确定面朝方向")
        return bad, valid, {}
    fn = _PAD_STEP_DISPATCH[strategy]
    cfg = ctx.params
    for sub, sub_times in iter_instance_slices(n_frames, ctx, scope):
        bad_seg, valid_seg, seg_details = fn(
            lm[sub],
            mask[sub],
            sub_times,
            ctx.stance,
            cfg.pad_step_min_disp,
            cfg.pad_step_overtake_max,
            cfg.pad_step_pause_max,
            facing=facing,
        )
        bad[sub] = bad_seg
        valid[sub] = valid_seg
        _write_segment_details(detail_arrays, sub, n_frames, seg_details)
    return bad, valid, detail_arrays


def measure_kick_retract_rule(
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """逐次执行前鞭腿收腿回屈规则，按次判落地前未收脚后回写序列结果。"""
    n_frames = len(lm)
    bad = np.zeros(n_frames, dtype=bool)
    valid = np.zeros(n_frames, dtype=bool)
    detail_arrays: dict[str, tuple[np.ndarray, str]] = {}
    facing = resolve_window_facing(lm, mask, baseline, baseline_mask)
    if facing is None:
        ctx.measurement_issues.append("起始护位与收腿窗口关键点不足以确定面朝方向")
        return bad, valid, {}
    _, _, _, kick_hip, _, kick_ankle, _, _ = sides_for_stance(role, ctx.stance)
    sup_role = "rear" if role == "front" else "front"
    _, _, _, _, _, sup_ankle, _, _ = sides_for_stance(sup_role, ctx.stance)
    torso = chain_torso(lm, mask)
    safe_mask = kicking_leg_side_safe(lm, mask, kick_ankle, sup_ankle, torso)
    max_x = ctx.params.kick_retract_max_x
    for sub, _ in iter_instance_slices(n_frames, ctx, scope):
        bad_seg, valid_seg, seg_details = measure_kick_retract_segment(
            lm[sub],
            safe_mask[sub],
            torso[sub],
            kick_ankle,
            kick_hip,
            facing,
            retract_max_x=max_x,
        )
        bad[sub] = bad_seg
        valid[sub] = valid_seg
        _write_segment_details(detail_arrays, sub, n_frames, seg_details)
    return bad, valid, detail_arrays
