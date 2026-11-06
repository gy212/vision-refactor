"""从已找到的站姿时段中选取稳定中段，供站姿与护位参照检查使用。"""

from __future__ import annotations

import math

import numpy as np

from core.evaluation.rules.sanda_v5 import FeedbackConfig
from core.foundation.geometry import torso_length
from core.foundation.signal import contiguous_runs


def stable_guard_windows(
    lm: np.ndarray, mask: np.ndarray, torso: np.ndarray, edge: int, cfg
) -> list[np.ndarray]:
    """长度为 ``edge`` 的滑窗中，护位链保持稳定且手已抬到护位的窗口（按起点升序）。"""
    n = len(lm)
    wrists = (lm[:, [15, 16], :2] - lm[:, [11, 12], :2]) / np.maximum(torso[:, None, None], 1e-4)
    stable = []
    for start in range(n - edge + 1):
        ix = np.arange(start, start + edge)
        if not (torso[ix] > 1e-4).all():
            continue
        # 每窗至少一条完整可见的护位链。远侧遮挡不否定近侧，但可见的移动手仍否定稳定。
        visible = 0
        raised = False
        for arm, (sh, wr, hip) in enumerate(((11, 15, 23), (12, 16, 24))):
            if not mask[ix][:, [sh, wr, hip]].all():
                continue
            visible += 1
            relative = wrists[ix, arm]
            if (
                np.max(np.linalg.norm(relative - np.median(relative, axis=0), axis=1))
                > cfg.return_distance / 2
                or np.max(
                    np.linalg.norm(lm[ix, hip, :2] - np.median(lm[ix, hip, :2], axis=0), axis=1)
                )
                > 0.15 * np.median(torso[ix])
                or not (lm[ix, wr, 1] < lm[ix, hip, 1] + 0.1 * torso[ix]).all()
            ):
                visible = 0
                break
            raised |= bool(
                (lm[ix, wr, 1] <= lm[ix, sh, 1] + cfg.guard_wrist_height * torso[ix]).all()
            )
        # 至少一只手抬到护位：直立双手下垂不算实战式（D-3，考生先直立再摆实战式）；单手掉低仍算，交给检查项判定。
        if visible and raised:
            stable.append(ix)
    return stable


def settled_stance_windows(
    series: dict, windows: dict, repetitions: list[dict], cfg: FeedbackConfig
) -> dict:
    """按本路画面的稳定程度选站姿，不读取任何检查项的对错。

    动作事件已有自己的起止时间。站姿检查取首段稳定站姿的中间，避免把刚抬手、
    刚收手的边缘当成已站好。后面的预摆停顿不能替代最早的站姿。
    """
    times = np.asarray(series["times"])
    if not windows or len(times) < cfg.min_valid * 2 + 1:
        return windows
    dt = float(np.median(np.diff(times)))
    edge = max(cfg.min_valid, int(math.ceil(cfg.stance_window_seconds / dt)))
    lm, mask = series["landmarks"], series["valid_mask"]
    stable = stable_guard_windows(lm, mask, torso_length(lm, mask, "visible_side_chain"), edge, cfg)
    revised = dict(windows)
    for phase in ("start", "end"):
        key = ("stance", phase)
        if key not in windows:
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
