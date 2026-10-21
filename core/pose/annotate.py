# -*- coding: utf-8 -*-
"""骨架视频导出 ``annotate_video``：逐帧推理、画骨架与手部，写成新视频；匹配预览与录后骨架成片（D-26）共用。

每帧用 ``PoseEngine.annotate``（VIDEO 模式，时间戳取解码帧的 ``t_ms``），与实时预览的 ``render_pose`` 绘制一致；
不画 HUD。writer 由 ``open_writer`` 打开（默认 ``open_video_writer``，H.264 不可用时回退 AVI），帧率与尺寸取源视频。
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.foundation.tasks import Cancelled, CancelToken
from core.media.video_source import DecodeOptions, open_video
from core.media.writer import open_video_writer
from core.pose.engine import EngineSpec, create_engine

WriterOpener = Callable[..., tuple[Any, Path, str]]   # (path, *, fps, size) -> (writer, 实际路径, 编码)
Progress = Callable[[int, int], None]                 # (已写帧数, 预计总帧数；未知为 0)


def annotate_video(
    video: Path | str,
    out_path: Path | str,
    engine: EngineSpec,
    *,
    decode: DecodeOptions = DecodeOptions(),
    cancel: CancelToken | None = None,
    progress: Progress | None = None,
    open_writer: WriterOpener = open_video_writer,
    keep_partial: bool = False,
) -> tuple[Path, str, int]:
    """把 ``video`` 画上骨架写到 ``out_path``；返回 ``(实际路径, 编码, 帧数)``。

    取消时抛 ``Cancelled``（writer 已释放，部分输出留给调用方处理）；``keep_partial`` 时改为停止并返回已写部分。
    视频打不开抛 ``VideoOpenError``；推理或写出出错原样抛出。
    """
    with open_video(video, decode, cancel) as source:
        meta = source.meta
        total = meta.frame_count if decode.end is None else max(0, int(decode.end) - int(decode.start) + 1)
        writer, actual_path, codec = open_writer(Path(out_path), fps=meta.fps, size=(meta.width, meta.height))
        pipe = None
        done = 0
        try:
            pipe = create_engine(engine)
            for frame in source:
                annotated, _actions = pipe.annotate(frame.bgr, timestamp_ms=frame.t_ms)
                writer.write(annotated)
                done += 1
                if progress is not None and (done % 5 == 0 or done == total):
                    progress(done, total)
        except Cancelled:
            if not keep_partial:
                raise
        finally:
            writer.release()
            if pipe is not None:
                pipe.close()
    return actual_path, codec, done


__all__ = ["annotate_video"]
