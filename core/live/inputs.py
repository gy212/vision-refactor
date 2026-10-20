# -*- coding: utf-8 -*-
"""会话的帧来源：``CameraInput`` 包装 ``CameraStream``，``VideoFileInput`` 包装 ``VideoSource``，对 runner 提供同一组步骤。

步骤顺序：``wait_ready``（摄像头等各路有效首帧；视频已打开）→ ``snapshot``（模型加载期的裸帧）→
``start``（摄像头接管租约，各路 reader 开始持续读帧）→ ``actual_sizes`` → ``read``（返回 ``InputFrames``；
视频读完返回 None；摄像头断流抛 ``CameraReadError``）→ ``close``。阻塞调用只在 runner 线程里执行。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from core.foundation.tasks import CancelToken
from core.media import (
    DEFAULT_FPS,
    PRIMARY,
    CameraStream,
    DecodeOptions,
    FrameSet,
    VideoSource,
)

FALLBACK_SIZE = (1920, 1080)  # 视频元数据缺尺寸时的回退，与采集格式一致（D-17）


@dataclass(frozen=True)
class InputFrames:
    """一次 ``read`` 的结果：各路帧，以及主路的采集时刻（``capture_clock_ns``；视频文件没有采集时刻，为 None）。"""

    frames: FrameSet
    captured_at_ns: int | None = None


class FrameInput(Protocol):
    is_file: bool
    roles: tuple[str, ...]

    def wait_ready(self, cancel: CancelToken) -> FrameSet: ...
    def first_cursor(self) -> dict[str, int]: ...
    def snapshot(self, after: dict[str, int]) -> tuple[FrameSet, dict[str, int]] | None: ...
    def start(self, cancel: CancelToken) -> None: ...
    def actual_sizes(self) -> dict[str, tuple[int, int]]: ...
    def fps(self) -> float: ...
    def total_frames(self) -> int: ...
    def read(self, cancel: CancelToken) -> InputFrames | None: ...
    def close(self, *, keep_warm: bool) -> None: ...


class CameraInput:
    """摄像头会话：等池里各路首帧 → 加载期取最新预热帧 → 接管租约 → 取各路最新帧 → 归还或释放（V-1）。"""

    is_file = False

    def __init__(self, stream: CameraStream) -> None:
        self._stream = stream
        self.roles = tuple(stream.specs)
        self._nominal_fps = 30.0
        self._cursor: dict[str, int] = {}

    @property
    def stream(self) -> CameraStream:
        return self._stream

    def wait_ready(self, cancel: CancelToken) -> FrameSet:
        ready = self._stream.wait_ready(cancel)
        self._cursor = {role: frame.seq for role, frame in ready.items()}
        return {role: frame.bgr for role, frame in ready.items()}

    def first_cursor(self) -> dict[str, int]:
        return dict(self._cursor)

    def snapshot(self, after: dict[str, int]) -> tuple[FrameSet, dict[str, int]] | None:
        latest = self._stream.snapshot(after=after)
        if latest is None:
            return None
        return (
            {role: frame.bgr for role, frame in latest.items()},
            {role: frame.seq for role, frame in latest.items()},
        )

    def start(self, cancel: CancelToken) -> None:
        self._stream.claim(cancel)

    def actual_sizes(self) -> dict[str, tuple[int, int]]:
        return self._stream.actual_sizes()

    def fps(self) -> float:
        return self._nominal_fps

    def total_frames(self) -> int:
        return 0

    def read(self, cancel: CancelToken) -> InputFrames:
        captured = self._stream.read(cancel)
        return InputFrames(
            {role: frame.bgr for role, frame in captured.items()}, captured[PRIMARY].captured_at_ns,
        )

    def close(self, *, keep_warm: bool) -> None:
        self._stream.close(keep_warm=keep_warm)


class VideoFileInput:
    """视频文件会话：逐帧顺序解码（``VideoSource`` 默认选项，读取失败即结束）；不旋转、不预热。"""

    is_file = True
    roles = (PRIMARY,)

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._source: VideoSource | None = None
        self._frames = None

    def wait_ready(self, cancel: CancelToken) -> FrameSet:
        self._source = VideoSource(self._path, DecodeOptions(), cancel)
        self._frames = iter(self._source)
        return {}

    def first_cursor(self) -> dict[str, int]:
        return {}

    def snapshot(self, after: dict[str, int]) -> tuple[FrameSet, dict[str, int]] | None:
        return None

    def start(self, cancel: CancelToken) -> None:
        return None

    def actual_sizes(self) -> dict[str, tuple[int, int]]:
        meta = self._source.meta if self._source is not None else None
        width = (meta.width if meta else 0) or FALLBACK_SIZE[0]
        height = (meta.height if meta else 0) or FALLBACK_SIZE[1]
        return {PRIMARY: (width, height)}

    def fps(self) -> float:
        reported = self._source.meta.fps_reported if self._source is not None else 0.0
        return reported if reported > 1e-3 else DEFAULT_FPS

    def total_frames(self) -> int:
        return self._source.meta.frame_count if self._source is not None else 0

    def read(self, cancel: CancelToken) -> InputFrames | None:
        if self._frames is None:
            return None
        frame = next(self._frames, None)
        return None if frame is None else InputFrames({PRIMARY: frame.bgr})

    def close(self, *, keep_warm: bool) -> None:
        if self._source is not None:
            self._source.close()


__all__ = ["CameraInput", "FrameInput", "InputFrames", "VideoFileInput"]
