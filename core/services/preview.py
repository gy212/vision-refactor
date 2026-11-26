# -*- coding: utf-8 -*-
"""匹配区间的骨架预览视频导出：逐帧推理并画骨架写出（模板生成、CompareWindow 与 CLI 共用）。"""

from __future__ import annotations

from pathlib import Path
from threading import Event
import uuid

from core.foundation.atomic_io import replace_file
from core.foundation.tasks import Cancelled, from_stopped
from core.media.video_source import DecodeOptions
from core.media.writer import transcode_to_h264
from core.pose.annotate import annotate_video
from core.pose.extract import formal_engine
from core.pose.features import ProgressCb


def _cleanup_partial_preview(*paths: Path) -> None:
    expanded: list[Path] = []
    for p in paths:
        expanded.append(p)
        expanded.append(p.with_suffix(".avi"))
    for path in dict.fromkeys(expanded):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _preview_targets(out_file: Path, actual_suffix: str) -> tuple[Path, Path | None]:
    ext = ".avi" if actual_suffix.lower() == ".avi" else (out_file.suffix or ".mp4")
    final_target = out_file.with_suffix(ext)
    stale_ext = ".mp4" if final_target.suffix.lower() == ".avi" else ".avi"
    stale = out_file.with_suffix(stale_ext)
    return final_target, (stale if stale != final_target else None)


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
    out_file.parent.mkdir(parents=True, exist_ok=True)
    work_path = (
        out_file
        if keep_partial
        else out_file.with_name(f"_tmp_{uuid.uuid4().hex}{out_file.suffix or '.mp4'}")
    )
    actual_path: Path | None = None
    final_target: Path | None = None
    try:
        try:
            actual_path, codec, _frames = annotate_video(
                video_path,
                work_path,
                formal_engine(pose_variant),
                decode=DecodeOptions(start=int(start_frame), end=int(end_frame)),
                cancel=cancel,
                progress=progress,
                keep_partial=keep_partial,
            )
        except Cancelled:
            raise InterruptedError("预览导出已取消") from None

        if not keep_partial and stop_evt is not None and stop_evt.is_set():
            raise InterruptedError("预览导出已取消")

        should_transcode = actual_path.suffix.lower() != ".mp4" and not (
            keep_partial and stop_evt is not None and stop_evt.is_set()
        )
        if should_transcode:
            transcoded = transcode_to_h264(actual_path, stop_evt=stop_evt)
            if transcoded is not None and transcoded != actual_path:
                actual_path = transcoded
                if actual_path.suffix.lower() == ".mp4":
                    codec = "H264"

        if keep_partial:
            final_target = actual_path
            return actual_path, codec

        if stop_evt is not None and stop_evt.is_set():
            raise InterruptedError("预览导出已取消")

        final_target, stale = _preview_targets(out_file, actual_path.suffix)
        if actual_path != final_target:
            replace_file(actual_path, final_target)
        if stale is not None:
            stale.unlink(missing_ok=True)
        return final_target, codec
    finally:
        if not keep_partial:
            leftovers = [work_path]
            if actual_path is not None and actual_path != final_target:
                leftovers.append(actual_path)
            _cleanup_partial_preview(*leftovers)


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
