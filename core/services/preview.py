# -*- coding: utf-8 -*-
"""匹配区间的骨架预览视频导出：逐帧推理并画骨架写出（模板生成、CompareWindow 与 CLI 共用）。"""
from __future__ import annotations

from pathlib import Path
from threading import Event

from core.foundation.tasks import from_stopped
from core.media.video_source import DecodeOptions
from core.pose.annotate import annotate_video
from core.pose.extract import formal_engine
from core.pose.features import ProgressCb


def write_pose_preview(
    video_path: Path,
    *,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    pose_variant: str,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> tuple[Path, str]:
    """把 ``[start_frame, end_frame]`` 逐帧推理并画骨架写成预览视频；返回 ``(实际路径, 编码)``。取消时保留已写部分。"""
    cancel = from_stopped(stop_evt.is_set) if stop_evt is not None else None
    progress = None
    if progress_cb is not None:
        progress = lambda done, total: progress_cb("导出匹配预览", done, total)  # noqa: E731
    path, codec, _frames = annotate_video(
        video_path, out_path, formal_engine(pose_variant),
        decode=DecodeOptions(start=int(start_frame), end=int(end_frame)),
        cancel=cancel, progress=progress, keep_partial=True,
    )
    return path, codec


def export_match_preview(
    video_path: Path,
    *,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    pose_variant: str,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> Path:
    path, _codec = write_pose_preview(
        video_path, out_path=out_path, start_frame=start_frame, end_frame=end_frame,
        pose_variant=pose_variant, progress_cb=progress_cb, stop_evt=stop_evt,
    )
    return path
