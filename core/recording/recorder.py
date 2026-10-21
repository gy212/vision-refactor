# -*- coding: utf-8 -*-
"""N 路同边界录制 ``SegmentRecorder``：一把锁覆盖开始、暂停、结束与关闭，各路共用同一组边界。

两种会话：
- 摄像头会话（``begin_session`` 给出 ``feed``）：每路一个 ``ViewWorker`` 录像线程，从该路的帧队列取帧，
  按共同边界（``capture_clock_ns``）换成恒定帧率写盘（D-18，设计文档 §8.8）；结束时写出 ``capture.json``。
  旋转在开录时取一次，段内不变（D-29）。
- 视频文件会话（没有 ``feed``）：每路一个 ``RecordingController``，由会话线程逐帧 ``write``，保序、用源帧率。

视角名取自 ``begin_session`` 的 ``sizes`` 键（单路 ``("front",)``，双路 ``("front", "side")``）。
一路是单路片段（``SegmentLayout.single_path``），多路是片段目录下的 ``<view>.mp4``。
recorder 只管录制边界，不提交后处理；结束后返回 ``FinishedSegment`` 交给 ``RecordingService``。
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from core.foundation.tasks import TaskGroup
from core.media import capture_clock_ns, capture_document, role_for_view, write_capture_record
from core.media.segment_layout import SegmentLayout, new_stamp, segment_id_for_stamp
from core.recording.controller import (
    RecordingController,
    RecordingSnapshot,
    RecordingState,
    WriterFactory,
)
from core.recording.worker import PRIME_WAIT_S, FrameFeed, ViewResult, ViewWorker

RotationOf = Callable[[str], int]

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

    @property
    def capture_path(self) -> Path:
        """片段的采集记录 ``capture.json``。"""
        return self.layout.capture_path(self.stamp, multi=self.multi)


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


class SegmentRecorder:
    """多路录制器（实现 ``live.FrameSink``）。各路片段边界由同一把锁保证一致。"""

    def __init__(self, writer_factory: WriterFactory | None = None) -> None:
        self._writer_factory = writer_factory
        self._lock = threading.Lock()
        self._controllers: dict[str, RecordingController] = {}   # 视频文件会话
        self._workers: dict[str, ViewWorker] = {}                # 摄像头会话
        self._rotation_of: RotationOf = lambda _role: 0
        self._state: RecordingState = "idle"
        self._sizes: dict[str, tuple[int, int]] = {}
        self._segment: SegmentRef | None = None
        self._tasks = TaskGroup("recording")

    # ------------------------------------------------------------------ 会话
    def begin_session(self, sizes: Mapping[str, tuple[int, int]], fps: float, *,
                      feed: FrameFeed | None = None, rotation_of: RotationOf | None = None) -> None:
        """登记本次会话的各路与输出帧率。摄像头会话给出 ``feed``：各路录像线程立即订阅并开始取帧，
        返回前等每路拿到首帧（最多 ``PRIME_WAIT_S``），之后开录就有边界之前的帧可用；
        视频文件会话只建控制器，不创建 writer。"""
        if not sizes:
            raise ValueError("录制会话至少需要一路")
        with self._lock:
            self._close_views()
            self._segment = None
            self._state = "idle"
            self._sizes = {view: (int(size[0]), int(size[1])) for view, size in sizes.items()}
            if feed is not None:
                self._rotation_of = rotation_of or (lambda _role: 0)
                self._workers = {
                    view: ViewWorker(view, role_for_view(view), feed, fps=fps, tasks=self._tasks,
                                     writer_factory=self._writer_factory)
                    for view in sizes
                }
                deadline = time.monotonic() + PRIME_WAIT_S
                for worker in self._workers.values():
                    worker.wait_primed(max(0.0, deadline - time.monotonic()))
                return
            for view, size in sizes.items():
                kwargs = {} if self._writer_factory is None else {"writer_factory": self._writer_factory}
                controller = RecordingController(path_provider=self._path_provider(view), **kwargs)
                controller.begin_session(fps=fps, size=tuple(size))
                self._controllers[view] = controller

    def close_session(self) -> FinishedSegment | None:
        """会话结束：结束仍在进行的片段并关闭各路；返回该片段（没有则 None）。"""
        with self._lock:
            finished = self._finish(discard=False)
            self._close_views()
            return finished

    # ------------------------------------------------------------------ 片段
    def start_segment(self, meta: SegmentMeta) -> SegmentRef | BeginError:
        """各路同时 idle→recording；任一路失败则全部复位，不留分叉状态。"""
        with self._lock:
            if self._workers:
                return self._start_workers(meta)
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
                return self._segment
            for controller, state in zip(controllers, states):
                if state in _ACTIVE:
                    controller.stop_recording()
            self._segment = None
            return BeginError("session_inactive")

    def _start_workers(self, meta: SegmentMeta) -> SegmentRef | BeginError:
        """摄像头会话开录：各路收到同一个开录边界，旋转在此刻取定（D-29）。"""
        if self._state != "idle":
            return BeginError("not_idle")
        segment = SegmentRef(new_stamp(), tuple(self._workers), meta)
        at = capture_clock_ns()
        for view, worker in self._workers.items():
            worker.start(at, segment.target(view), self._rotation_of(worker.role))
        self._segment, self._state = segment, "recording"
        return segment

    def toggle_pause(self) -> RecordingState | None:
        """recording⇄paused。各路切换后状态不一致时返回 None（调用方应结束该片段）。"""
        with self._lock:
            if self._workers:
                if self._state == "idle":
                    return "idle"
                at = capture_clock_ns()
                pausing = self._state == "recording"
                for worker in self._workers.values():
                    if pausing:
                        worker.pause(at)
                    else:
                        worker.resume(at)
                self._state = "paused" if pausing else "recording"
                return self._state
            states = {c.request_toggle() for c in self._controllers.values()}
            if len(states) > 1:
                return None
            return next(iter(states), "idle")

    def broken(self) -> bool:
        """多路片段已分叉：有一路失败（写盘、队列溢出、断流）或各路状态不一致。"""
        with self._lock:
            views = self._views()
            if len(views) < 2:
                return False
            snaps = [item.snapshot() for item in views.values()]
            if any(s.last_error for s in snaps):
                return True
            return not self._workers and len({s.state for s in snaps}) > 1

    def write(self, frames: Mapping[str, np.ndarray]) -> None:
        """视频文件会话：写入同一轮的各路帧（writer 尺寸取首帧）。摄像头会话由录像线程取帧，这里不做事。"""
        with self._lock:
            for view, controller in self._controllers.items():
                frame = frames.get(view)
                if frame is None:
                    continue
                controller.update_session_size(size=(int(frame.shape[1]), int(frame.shape[0])))
                controller.write_frame(frame)

    def end_segment(self, *, discard: bool = False) -> FinishedSegment | None:
        """结束当前片段，会话保持；返回结束的片段（没有进行中的片段则 None）。"""
        with self._lock:
            return self._finish(discard=discard)

    # ------------------------------------------------------------------ 查询
    @property
    def state(self) -> RecordingState:
        with self._lock:
            if self._workers:
                return self._state
            first = next(iter(self._controllers.values()), None)
            return "idle" if first is None else first.state

    @property
    def views(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._views())

    @property
    def session_sizes(self) -> dict[str, tuple[int, int]]:
        """各路登记的会话尺寸；没有会话时为空。"""
        with self._lock:
            if self._workers:
                return dict(self._sizes)
            return {view: c.session_size for view, c in self._controllers.items()}

    def snapshot(self) -> RecorderSnapshot:
        with self._lock:
            views = {view: item.snapshot() for view, item in self._views().items()}
            if self._workers:
                views = {view: replace(snap, state=self._state) for view, snap in views.items()}
            first = next(iter(views.values()), None)
            return RecorderSnapshot(
                state="idle" if first is None else first.state,
                session_active=bool(views),
                segment=self._segment,
                views=views,
            )

    # ------------------------------------------------------------------ 内部
    def _views(self) -> Mapping[str, Any]:
        return self._workers or self._controllers

    def _path_provider(self, view: str):
        def provide() -> Path:
            segment = self._segment
            if segment is None:  # 只有片段内才会进入 recording 状态并创建 writer
                raise RuntimeError("没有进行中的录制片段")
            return segment.target(view)

        return provide

    def _finish(self, *, discard: bool) -> FinishedSegment | None:
        if self._workers:
            return self._finish_workers(discard=discard)
        segment = self._segment
        snaps = {view: c.snapshot() for view, c in self._controllers.items()}
        paths = {view: c.stop_recording() for view, c in self._controllers.items()}
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

    def _finish_workers(self, *, discard: bool) -> FinishedSegment | None:
        """各路收到同一个结束边界，等每路写完边界之前的输出帧，再写 ``capture.json``。"""
        segment = self._segment
        if segment is None:
            return None
        at = capture_clock_ns()
        results = {view: worker.stop(at) for view, worker in self._workers.items()}
        self._segment, self._state = None, "idle"
        _write_capture(segment, results)
        return FinishedSegment(
            ref=segment,
            views={view: ViewRecording(r.path, r.frames, r.error) for view, r in results.items()},
            started=True,
            discarded=discard,
        )

    def _close_views(self) -> None:
        for controller in self._controllers.values():
            controller.close_session()
        for worker in self._workers.values():
            worker.close()
        self._controllers, self._workers = {}, {}


def _write_capture(segment: SegmentRef, results: Mapping[str, ViewResult]) -> None:
    """写出片段的 ``capture.json``（开录失败、没有时间轴的路不写）；写不出不影响录像本身。"""
    timelines = {view: r.timeline for view, r in results.items() if r.timeline is not None}
    if not timelines:
        return
    extra = {view: {**results[view].extra, "error": results[view].error} for view in timelines}
    try:
        segment.capture_path.parent.mkdir(parents=True, exist_ok=True)
        write_capture_record(segment.capture_path, capture_document(timelines, extra))
    except (OSError, ValueError):
        pass
