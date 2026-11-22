"""规则窗口的公共切片、面朝方向推断与逐次标量结果回写原语。"""

from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import facing_direction
from core.evaluation.rules.spec import RuleContext
from core.foundation.signal import contiguous_runs


def empty_result(n_frames: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """构造无有效帧的空判定结果。"""
    empty = np.zeros(n_frames, dtype=bool)
    return empty, empty.copy(), {}


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
        d for row, m_row in zip(lm, mask) if (d := facing_direction(row, m_row)) is not None
    ]
    if not win_dirs:
        return None
    return 1.0 if sum(win_dirs) >= 0.0 else -1.0


def iter_labeled_instance_slices(
    n_frames: int,
    ctx: RuleContext,
    scope: str,
) -> list[tuple[int | None, np.ndarray, np.ndarray]]:
    """按 phase_instances 提取带 repetition 身份的每次动作切片及时间戳。"""
    if n_frames == 0:
        return []
    win_orig = ctx.phase_windows.get(scope)
    if win_orig is None or len(win_orig) != n_frames:
        win_orig = np.arange(n_frames, dtype=int)
    else:
        win_orig = np.asarray(win_orig, dtype=int)
    fps = float(ctx.params.sample_fps) if ctx.params and ctx.params.sample_fps > 0 else 30.0
    if ctx.times is not None and win_orig.size and int(np.max(win_orig)) < len(ctx.times):
        win_times = np.asarray(ctx.times[win_orig], dtype=float)
    else:
        win_times = np.asarray(win_orig, dtype=float) / fps
    instances = ctx.phase_instances.get(scope, ())
    if not instances:
        return [(None, np.arange(n_frames, dtype=int), win_times)]
    slices: list[tuple[int | None, np.ndarray, np.ndarray]] = []
    covered = np.zeros(n_frames, dtype=bool)
    for rep_id, frames in instances:
        sub = np.flatnonzero(np.isin(win_orig, frames))
        if sub.size:
            covered[sub] = True
            slices.append((rep_id, sub, win_times[sub]))
    for run in contiguous_runs(win_orig[~covered]):
        sub = np.flatnonzero(~covered & (win_orig >= run[0]) & (win_orig <= run[-1]))
        if sub.size:
            slices.append((None, sub, win_times[sub]))
    return slices or [(None, np.arange(n_frames, dtype=int), win_times)]


def iter_instance_slices(
    n_frames: int,
    ctx: RuleContext,
    scope: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """根据 phase_windows 与 phase_instances 提取每次重复动作的局部切片及时间戳。"""
    return [
        (sub, sub_times) for _, sub, sub_times in iter_labeled_instance_slices(n_frames, ctx, scope)
    ]


def write_segment_details(
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
