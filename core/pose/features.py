# -*- coding: utf-8 -*-
"""正式路径的归一化特征序列：``FORMAL`` 提取 + 逐帧归一化特征 + 视角分数（模板生成与比对共用，原 ``action_compare._extract_pose_features``）。"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event

import numpy as np

from core.foundation.tasks import Cancelled, from_stopped
from core.pose.extract import FORMAL, extract_pose_sequence, formal_engine
from core.pose.layouts import POSE33_V3, FeatureLayoutSpec
from core.pose.normalize import normalize_pose_xy
from core.pose.sequence import GapPolicy

ProgressCb = Callable[[str, int, int], None]  # (stage, done, total)


def extract_features(
    video_path: Path,
    *,
    pose_variant: str,
    workers: int = 1,
    normalizer: Callable = normalize_pose_xy,
    compute_view: bool = False,
    layout: FeatureLayoutSpec = POSE33_V3,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """整段视频的归一化特征，返回 ``(features[T,J,2], fps, view_scores[T] 或 None)``，J 由 ``layout`` 决定。

    归一化返回 None 的帧沿用上一帧、首帧前按 layout 形状补零；视角分数按 ``HOLD_LAST``。
    ``workers > 1`` 走分段并行（每段前 15 帧跟踪预热）。取消抛 ``InterruptedError``。
    """
    def progress(done: int, total: int) -> None:
        if progress_cb is not None:
            progress_cb("提取骨架特征", done, total)

    cancel = from_stopped(stop_evt.is_set) if stop_evt is not None else None
    try:
        seq = extract_pose_sequence(video_path, formal_engine(pose_variant), FORMAL, workers=workers,
                                    cancel=cancel, progress=progress)
        if cancel is not None:
            cancel.raise_if_cancelled()  # 最后一帧推理期间收到的取消也生效
    except Cancelled:
        raise InterruptedError("姿态特征提取已取消") from None
    if seq.decoded_count == 0:
        raise RuntimeError("视频为空或无法读取帧")
    zero_shape = tuple(int(x) for x in layout.shape)
    feats = seq.features(normalizer, zero_shape).copy()
    views = seq.view_scores(GapPolicy.HOLD_LAST).copy() if compute_view else None
    return feats, seq.meta.fps, views
