# -*- coding: utf-8 -*-
"""N 路同边界录制 ``SegmentRecorder``：每路一个 ``RecordingController``，一把锁覆盖开始、写帧、暂停、结束与关闭。

视角名取自 ``begin_session`` 的 ``sizes`` 键（单路 ``("front",)``，双路 ``("front", "side")``）。
一路是单路片段（``SegmentLayout.single_path``），多路是片段目录下的 ``<view>.mp4``。
recorder 只管录制边界，不提交后处理；结束后返回 ``FinishedSegment`` 交给 ``RecordingService``。
各路共用一个 ``FramePacer``：按帧的采集时刻决定写几次，使录像按真实时间回放、各路帧数相同（D-18）。
"""
from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.media.pacing import FramePacer
from core.media.segment_layout import SegmentLayout, new_stamp, segment_id_for_stamp
from core.recording.controller import (
    RecordingController,
    RecordingSnapshot,
    RecordingState,
    WriterFactory,
)

_ACTIVE = frozenset({"recording", "paused"})


@dataclass(frozen=True)
class SegmentMeta:
    """开始片段时由调用方给出：录制根目录，以及构建处理计划时要用的上下文（recorder 不解读）。"""

    base_dir: Path
    context: Any = None


@dataclass(frozen=True)
class SegmentRef:
    """一个已开始的片段：id、时间戳、视角与路径形状。"""

    stamp: str
    views: tuple[str, ...]
    meta: SegmentMeta

    @property
    def segment_id(self) -> str:
        return segment_id_for_stamp(self.stamp)

    @property
    def layout(self) -> SegmentLayout:
        return SegmentLayout(Path(self.meta.base_dir))

    @property
    def multi(self) -> bool:
        return len(self.views) > 1

    @property
    def directory(self) -> Path | None:
        """多路片段目录；单路片段没有自己的目录。"""
        return self.layout.segment_dir(self.stamp) if self.multi else None

    def target(self, view: str) -> Path:
        """某一路的计划写出路径（编码回退时实际后缀可能不同）。"""
        if self.multi:
            return self.layout.view_path(self.stamp, view)
        return self.layout.single_path(self.stamp)


@dataclass(frozen=True)
class BeginError:
    """开始片段失败的原因：``session_inactive`` 或 ``not_idle``。"""

    reason: str

    def __bool__(self) -> bool:
        return False


@dataclass(frozen=True)
class ViewRecording:
    path: Path | None
    frames: int
    error: str | None


@dataclass(frozen=True)
class FinishedSegment:
    """已结束的片段：每路的实际路径、帧数与写错误。"""

    ref: SegmentRef
    views: Mapping[str, ViewRecording]
    started: bool
    discarded: bool = False

    @property
    def segment_id(self) -> str:
        return self.ref.segment_id

    @property
    def directory(self) -> Path | None:
        return self.ref.directory

    @property
    def multi(self) -> bool:
        return self.ref.multi

    def source(self, view: str) -> Path:
        """该路的录像文件；没有实际写出时取计划路径。"""
        return self.views[view].path or self.ref.target(view)

    @property
    def errors(self) -> dict[str, str]:
        return {view: rec.error for view, rec in self.views.items() if rec.error}

    @property
    def has_activity(self) -> bool:
        return bool(self.started or any(r.path or r.frames > 0 or r.error for r in self.views.values()))


@dataclass(frozen=True)
class RecorderSnapshot:
    state: RecordingState
    session_active: bool
    segment: SegmentRef | None
    views: Mapping[str, RecordingSnapshot] = field(default_factory=dict)

    @property
    def errors(self) -> dict[str, str]:
        return {view: snap.last_error for view, snap in self.views.items() if snap.last_error}


def fit_frame_to_size(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """将帧缩放到 writer 固定尺寸（运行中旋转会改变帧尺寸）；尺寸一致时原样返回。"""
    tw, th = int(size[0]), int(size[1])
    if tw <= 0 or th <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == tw and h == th:
        return frame
    return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)


class SegmentRecorder:
    """多路录制器（实现 ``live.FrameSink``）。各路片段边界由同一把锁保证一致。"""

    def __init__(self, writer_factory: WriterFactory | None = None) -> None:
        self._writer_factory = writer_factory
        self._lock = threading.Lock()
        self._controllers: dict[str, RecordingController] = {}
        self._segment: SegmentRef | None = None
        self._pacer: FramePacer | None = None

    # ------------------------------------------------------------------ 会话
    def begin_session(self, sizes: Mapping[str, tuple[int, int]], fps: float) -> None:
        """登记本次会话的各路尺寸与名义 fps；不创建 writer。"""
        if not sizes:
            raise ValueError("录制会话至少需要一路")
        with self._lock:
            self._close_controllers()
            self._segment = None
            self._controllers = {}
            self._pacer = FramePacer(fps) if fps > 0 else None   # 名义帧率未知时逐帧写一次
            for view, size in sizes.items():
                kwargs = {} if self._writer_factory is None else {"writer_factory": self._writer_factory}
                controller = RecordingController(path_provider=self._path_provider(view), **kwargs)
                controller.begin_session(fps=fps, size=tuple(size))
                self._controllers[view] = controller

    def update_sizes(self, sizes: Mapping[str, tuple[int, int]]) -> None:
        """writer 创建前用首帧真实尺寸修正各路会话尺寸；已建 writer 的路不变。"""
        with self._lock:
            for view, size in sizes.items():
                controller = self._controllers.get(view)
                if controller is not None:
                    controller.update_session_size(size=tuple(size))

    def close_session(self) -> FinishedSegment | None:
        """会话结束：关闭各路 writer；返回仍在进行的片段（没有则 None）。"""
        with self._lock:
            finished = self._finish(lambda c: c.close_session(), discard=False)
            self._controllers = {}
            return finished

    # ------------------------------------------------------------------ 片段
    def start_segment(self, meta: SegmentMeta) -> SegmentRef | BeginError:
        """各路同时 idle→recording；任一路失败则全部复位，不留分叉状态。"""
        with self._lock:
            controllers = list(self._controllers.values())
            if not controllers:
                return BeginError("session_inactive")
            if any(c.state != "idle" for c in controllers):
                for controller in controllers:
                    controller.stop_recording()
                if any(c.state != "idle" for c in controllers):
                    return BeginError("not_idle")
            self._segment = SegmentRef(new_stamp(), tuple(self._controllers), meta)
            states = [c.request_toggle() for c in controllers]
            if all(state == "recording" for state in states):
                if self._pacer is not None:
                    self._pacer.reset()
                return self._segment
            for controller, state in zip(controllers, states):
                if state in _ACTIVE:
                    controller.stop_recording()
            self._segment = None
            return BeginError("session_inactive")

    def toggle_pause(self) -> RecordingState | None:
        """recording⇄paused。各路切换后状态不一致时返回 None（调用方应结束该片段）。"""
        with self._lock:
            states = {c.request_toggle() for c in self._controllers.values()}
            if len(states) > 1:
                return None
            state = next(iter(states), "idle")
            if state == "recording" and self._pacer is not None:
                self._pacer.resync()
            return state

    def broken(self) -> bool:
        """多路片段已分叉：有一路写失败或各路状态不一致。"""
        with self._lock:
            if len(self._controllers) < 2:
                return False
            snaps = [c.snapshot() for c in self._controllers.values()]
            return any(s.last_error for s in snaps) or len({s.state for s in snaps}) > 1

    def write(self, frames: Mapping[str, np.ndarray], captured_at: float | None = None) -> None:
        """写入同一轮的各路帧；尺寸与会话不符时先缩放。

        ``captured_at`` 是这一轮的采集时刻（``time.monotonic()`` 秒），各路按同一节拍重复或跳过；
        None 表示逐帧写一次（视频文件输入）。
        """
        with self._lock:
            repeats = 1
            if self._pacer is not None and any(c.state == "recording" for c in self._controllers.values()):
                repeats = self._pacer.repeats(captured_at)
            for view, controller in self._controllers.items():
                frame = frames.get(view)
                if frame is None:
                    continue
                try:
                    frame = fit_frame_to_size(frame, controller.session_size)
                except Exception:
                    pass
                controller.write_frame(frame, repeats)

    def end_segment(self, *, discard: bool = False) -> FinishedSegment | None:
        """结束当前片段，会话保持；返回结束的片段（没有进行中的片段则 None）。"""
        with self._lock:
            return self._finish(lambda c: c.stop_recording(), discard=discard)

    # ------------------------------------------------------------------ 查询
    @property
    def state(self) -> RecordingState:
        with self._lock:
            first = next(iter(self._controllers.values()), None)
            return "idle" if first is None else first.state

    @property
    def views(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._controllers)

    @property
    def session_sizes(self) -> dict[str, tuple[int, int]]:
        """各路当前会话尺寸；没有会话时为空。"""
        with self._lock:
            return {view: c.session_size for view, c in self._controllers.items()}

    def snapshot(self) -> RecorderSnapshot:
        with self._lock:
            views = {view: c.snapshot() for view, c in self._controllers.items()}
            first = next(iter(views.values()), None)
            return RecorderSnapshot(
                state="idle" if first is None else first.state,
                session_active=bool(self._controllers),
                segment=self._segment,
                views=views,
            )

    # ------------------------------------------------------------------ 内部
    def _path_provider(self, view: str):
        def provide() -> Path:
            segment = self._segment
            if segment is None:  # 只有片段内才会进入 recording 状态并创建 writer
                raise RuntimeError("没有进行中的录制片段")
            return segment.target(view)

        return provide

    def _finish(self, stop, *, discard: bool) -> FinishedSegment | None:
        segment = self._segment
        snaps = {view: c.snapshot() for view, c in self._controllers.items()}
        paths = {view: stop(c) for view, c in self._controllers.items()}
        self._segment = None
        if segment is None:
            return None
        return FinishedSegment(
            ref=segment,
            views={
                view: ViewRecording(paths[view], int(snap.frames_written), snap.last_error)
                for view, snap in snaps.items()
            },
            started=any(snap.state in _ACTIVE for snap in snaps.values()),
            discarded=discard,
        )

    def _close_controllers(self) -> None:
        for controller in self._controllers.values():
            controller.close_session()
