# -*- coding: utf-8 -*-
"""匹配区间的骨架预览视频导出：逐帧推理并画骨架写出（模板生成、CompareWindow 与 CLI 共用）。"""

from __future__ import annotations

from pathlib import Path
from threading import Event

from core.foundation.tasks import Cancelled, from_stopped
from core.media.video_source import DecodeOptions
from core.media.writer import transcode_to_h264
from core.pose.annotate import annotate_video
from core.pose.extract import formal_engine
from core.pose.features import ProgressCb


def _cleanup_partial_preview(out_path: Path, *extra: Path) -> None:
    candidates = dict.fromkeys((out_path, out_path.with_suffix(".avi"), *extra))
    for path in candidates:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def write_pose_preview(
    video_path: Path,
    *,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    pose_variant: str,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
    keep_partial: bool = False,
) -> tuple[Path, str]:
    """把 ``[start_frame, end_frame]`` 逐帧推理并画骨架写成预览视频；非 MP4 回退输出自动尝试转码为 H.264 MP4。"""
    if stop_evt is not None and stop_evt.is_set():
        raise InterruptedError("预览导出已取消")
    cancel = from_stopped(stop_evt.is_set) if stop_evt is not None else None
    progress = None
    if progress_cb is not None:
        progress = lambda done, total: progress_cb("导出匹配预览", done, total)  # noqa: E731
    out_file = Path(out_path)
    try:
        path, codec, _frames = annotate_video(
            video_path,
            out_file,
            formal_engine(pose_variant),
            decode=DecodeOptions(start=int(start_frame), end=int(end_frame)),
            cancel=cancel,
            progress=progress,
            keep_partial=keep_partial,
        )
    except Cancelled:
        if not keep_partial:
            _cleanup_partial_preview(out_file)
        raise InterruptedError("预览导出已取消") from None

    if not keep_partial and stop_evt is not None and stop_evt.is_set():
        _cleanup_partial_preview(out_file, path)
        raise InterruptedError("预览导出已取消")

    should_transcode = path.suffix.lower() != ".mp4" and not (
        keep_partial and stop_evt is not None and stop_evt.is_set()
    )
    if should_transcode:
        try:
            transcoded = transcode_to_h264(path, stop_evt=stop_evt)
        except InterruptedError:
            if not keep_partial:
                _cleanup_partial_preview(out_file, path)
            raise InterruptedError("预览导出已取消") from None
        if transcoded is not None and transcoded != path:
            path = transcoded
            if path.suffix.lower() == ".mp4":
                codec = "H264"
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
    keep_partial: bool = False,
) -> Path:
    """导出匹配/模板片段骨架预览视频并返回实际落盘路径（AVI 回退会自动转码为 H.264 MP4）。"""
    path, _codec = write_pose_preview(
        video_path,
        out_path=out_path,
        start_frame=start_frame,
        end_frame=end_frame,
        pose_variant=pose_variant,
        progress_cb=progress_cb,
        stop_evt=stop_evt,
        keep_partial=keep_partial,
    )
    return path
