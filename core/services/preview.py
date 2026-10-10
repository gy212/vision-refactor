# -*- coding: utf-8 -*-
"""匹配区间的骨架预览视频导出：逐帧推理并画骨架写出（模板生成、CompareWindow 与 CLI 共用）。"""
from __future__ import annotations

from pathlib import Path
from threading import Event

from core.foundation.tasks import Cancelled, from_stopped
from core.media.video_source import DecodeOptions, open_video
from core.media.writer import open_video_writer
from core.pose.engine import create_engine
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
    """把 ``[start_frame, end_frame]`` 逐帧推理并画骨架写成预览视频；返回 ``(实际路径, 编码)``。"""
    cancel = from_stopped(stop_evt.is_set) if stop_evt is not None else None
    total = max(0, int(end_frame - start_frame + 1))
    with open_video(video_path, DecodeOptions(start=int(start_frame), end=int(end_frame)), cancel) as source:
        meta = source.meta
        vw, actual_path, codec = open_video_writer(out_path, fps=meta.fps, size=(meta.width, meta.height))
        pipe = create_engine(formal_engine(pose_variant))
        done = 0
        try:
            for frame in source:
                annotated, _actions = pipe.annotate(frame.bgr, timestamp_ms=frame.t_ms)
                vw.write(annotated)
                done += 1
                if progress_cb is not None and (done % 5 == 0 or done == total):
                    progress_cb("导出匹配预览", done, total)
        except Cancelled:
            pass  # 取消时保留已写出的部分（与旧实现一致）
        finally:
            vw.release()
            pipe.close()
    return actual_path, codec


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
