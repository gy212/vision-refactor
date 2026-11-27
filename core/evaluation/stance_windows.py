"""从已找到的站姿时段中选取稳定中段，供站姿与护位参照检查使用。"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from core.evaluation.rules.feedback_config import FeedbackConfig
from core.foundation.geometry import torso_length
from core.foundation.signal import contiguous_runs

_GUARD_ARMS = ((11, 15, 23), (12, 16, 24))


def stance_window_edge(times: np.ndarray, cfg: FeedbackConfig) -> int | None:
    """按采样间隔计算稳定站姿滑窗长度；序列过短或时间戳异常时返回 ``None``。"""
    if len(times) < max(2, cfg.min_valid * 2 + 1):
        return None
    with np.errstate(all="ignore"):
        dt = float(np.median(np.diff(times)))
    if not math.isfinite(dt) or dt <= 0:
        return None
    span = cfg.stance_window_seconds / dt
    if not math.isfinite(span) or span > len(times):
        return None
    return max(cfg.min_valid, int(math.ceil(span)))


def stable_guard_windows(
    lm: np.ndarray,
    mask: np.ndarray,
    torso: np.ndarray,
    edge: int,
    cfg: FeedbackConfig,
) -> tuple[np.ndarray, ...]:
    """长度为 ``edge`` 的滑窗中，护位链保持稳定且手已抬到护位的只读窗口（按起点升序）。"""
    n = len(lm)
    if edge <= 0 or n < edge:
        return ()
    with np.errstate(all="ignore"):
        wrists = (lm[:, [15, 16], :2] - lm[:, [11, 12], :2]) / np.maximum(
            torso[:, None, None], 1e-4
        )
        torso_w = sliding_window_view(torso, edge, axis=0)
        all_arms_ok = (torso_w > 1e-4).all(axis=1)
        if not all_arms_ok.any():
            return ()
        # 每窗至少一条完整可见的护位链。远侧遮挡不否定近侧，但可见的移动手仍否定稳定。
        any_visible = np.zeros(n - edge + 1, dtype=bool)
        any_raised = np.zeros(n - edge + 1, dtype=bool)
        for arm, (sh, wr, hip) in enumerate(_GUARD_ARMS):
            arm_mask_w = sliding_window_view(mask[:, [sh, wr, hip]], edge, axis=0)
            vis = all_arms_ok & arm_mask_w.all(axis=(1, 2))
            if not vis.any():
                continue
            any_visible |= vis
            rel_w = sliding_window_view(wrists[:, arm, :], edge, axis=0)[vis]
            hip_w = sliding_window_view(lm[:, hip, :2], edge, axis=0)[vis]
            wr_y = sliding_window_view(lm[:, wr, 1], edge, axis=0)[vis]
            sh_y = sliding_window_view(lm[:, sh, 1], edge, axis=0)[vis]
            hip_y = sliding_window_view(lm[:, hip, 1], edge, axis=0)[vis]
            t_w = torso_w[vis]
            w_dev = np.max(
                np.linalg.norm(rel_w - np.median(rel_w, axis=2, keepdims=True), axis=1),
                axis=1,
            )
            h_dev = np.max(
                np.linalg.norm(hip_w - np.median(hip_w, axis=2, keepdims=True), axis=1),
                axis=1,
            )
            t_med = np.median(t_w, axis=1)
            # 严格按德摩根律保留窗口，避免 NaN 比较翻转语义。
            arm_ok = ~(
                (w_dev > cfg.return_distance / 2)
                | (h_dev > 0.15 * t_med)
                | ~(wr_y < hip_y + 0.1 * t_w).all(axis=1)
            )
            all_arms_ok[vis] &= arm_ok
            any_raised[vis] |= arm_ok & (wr_y <= sh_y + cfg.guard_wrist_height * t_w).all(axis=1)
        # 至少一只手抬到护位：直立双手下垂不算实战式（D-3，考生先直立再摆实战式）；单手掉低仍算，交给检查项判定。
        keep = all_arms_ok & any_visible & any_raised
    stable = []
    for start in np.flatnonzero(keep):
        ix = np.arange(int(start), int(start) + edge)
        ix.setflags(write=False)
        stable.append(ix)
    return tuple(stable)


def settled_stance_windows(
    series: dict,
    windows: dict,
    repetitions: list[dict],
    cfg: FeedbackConfig,
    stable: Sequence[np.ndarray] | None = None,
) -> dict:
    """按本路画面的稳定程度选站姿，不读取任何检查项的对错。

    动作事件已有自己的起止时间。站姿检查取首段稳定站姿的中间，避免把刚抬手、
    刚收手的边缘当成已站好。后面的预摆停顿不能替代最早的站姿。
    """
    times = np.asarray(series["times"], dtype=float)
    if not windows or len(times) < cfg.min_valid * 2 + 1:
        return windows
    lm, mask = series["landmarks"], series["valid_mask"]
    if stable is None:
        edge = stance_window_edge(times, cfg)
        if edge is None:
            stable = ()
        else:
            stable = stable_guard_windows(
                lm, mask, torso_length(lm, mask, "visible_side_chain"), edge, cfg
            )
    revised = dict(windows)
    for phase in ("start", "end"):
        key = ("stance", phase)
        if key not in windows:
            continue
        if not stable:
            revised.pop(key)
            continue
        floor = int(windows[key][0])
        ceiling = (
            int(np.searchsorted(times, repetitions[0]["start"]))
            if phase == "start" and repetitions
            else len(times)
        )
        unpaired_end = phase == "end" and not repetitions
        candidates = (
            stable if unpaired_end else [ix for ix in stable if ix[0] >= floor and ix[-1] < ceiling]
        )
        if not candidates:
            revised.pop(key)
            continue
        runs = contiguous_runs(np.unique(np.concatenate(candidates)))
        # 没认出完整组合时，仍检查本路最后一次站姿。不能让另一面较晚的
        # 站姿边缘挤掉本路已看清的低手画面，也不据此补出动作次数。
        run = runs[-1] if unpaired_end else runs[0]
        middle = float(np.median(run))
        inside = [ix for ix in candidates if ix[0] >= run[0] and ix[-1] <= run[-1]]
        # 先看手腕、嘴部是否可见，再靠近稳定段中间；不按姿势对错挑帧。
        revised[key] = min(
            inside,
            key=lambda ix: (
                -int(mask[ix][:, [9, 10, 15, 16]].sum()),
                abs(float(np.median(ix)) - middle),
            ),
        )
    return revised
