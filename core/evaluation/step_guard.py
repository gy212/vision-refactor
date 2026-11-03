"""选择上步直拳起止实战式的同步稳定窗口。"""

from __future__ import annotations

import math

import numpy as np

from core.evaluation.phases import stable_guard_windows
from core.evaluation.rules.sanda_v5 import FeedbackConfig
from core.foundation.geometry import torso_length
from core.foundation.signal import contiguous_runs


def guard_windows(
    series: dict, cfg: FeedbackConfig, front: dict | None = None
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """保留侧面原滑窗，按帧覆盖并集求双路稳定段；不按姿势合格与否筛选。"""
    times = np.asarray(series["times"])
    edge = max(cfg.min_valid, int(math.ceil(cfg.stance_window_seconds / np.median(np.diff(times)))))
    stable = stable_guard_windows(
        series["landmarks"],
        series["valid_mask"],
        torso_length(series["landmarks"], series["valid_mask"], "visible_side_chain"),
        edge,
        cfg,
    )
    covered = np.zeros(len(times), dtype=bool)
    for window in stable:
        covered[window] = True
    if front is not None:
        ft = np.asarray(front["times"])
        if len(ft) < 2:
            return stable, []
        front_edge = max(
            cfg.min_valid, int(math.ceil(cfg.stance_window_seconds / np.median(np.diff(ft))))
        )
        front_stable = stable_guard_windows(
            front["landmarks"],
            front["valid_mask"],
            torso_length(front["landmarks"], front["valid_mask"], "visible_side_chain"),
            front_edge,
            cfg,
        )
        front_covered = np.zeros(len(ft), dtype=bool)
        for window in front_stable:
            front_covered[window] = True
        shared = np.zeros(len(times), dtype=bool)
        for run in contiguous_runs(np.flatnonzero(front_covered)):
            shared |= (times >= ft[run[0]] - 1e-6) & (times <= ft[run[-1]] + 1e-6)
        covered &= shared
    return stable, contiguous_runs(np.flatnonzero(covered))


def start_guard(
    runs: list[np.ndarray], times: np.ndarray, before: int, fallback: np.ndarray
) -> tuple[np.ndarray, str]:
    """优先第一个半秒稳定段的后半段，其次最长七帧段，最后原始滑窗。"""
    dt = float(np.median(np.diff(times)))
    candidates = [r for r in runs if r[-1] < before and len(r) >= 7]
    for run in candidates:
        if len(run) * dt >= 0.5 - 1e-6:
            count = min(int(round(1 / dt)), len(run) // 2)
            return run[-count:], "stable_run"
    if candidates:
        return max(candidates, key=len), "short_run"
    return fallback, "fallback"


def end_guard(
    runs: list[np.ndarray], times: np.ndarray, after: int, fallback: np.ndarray
) -> tuple[np.ndarray, str]:
    """末次结束后首个稳定段跳过0.1秒，取最多一秒；不足七帧则回退。"""
    for run in runs:
        tail = run[run >= after]
        if not len(tail):
            continue
        tail = tail[
            (times[tail] >= times[tail[0]] + 0.1 - 1e-6)
            & (times[tail] < times[tail[0]] + 1.1 - 1e-6)
        ]
        if len(tail) >= 7:
            return tail, "stable_run"
    return fallback, "fallback"
