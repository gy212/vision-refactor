# -*- coding: utf-8 -*-
"""摄像头会话每路一个录像线程 ``ViewWorker``：按序取该路的源帧 → 去抖 → 换成恒定帧率 → 按段首方向转正 → 写盘（D-18，设计文档 §8.8）。

- 会话开始时订阅该路 reader 的有界帧队列（``QUEUE_CAPACITY`` 帧，reader 非阻塞投递），整个会话都在取帧：
  没有在录时也把帧交给 ``FrameClock``，开录时时钟已经稳定，并留着开录边界之前最近的一帧作首帧。
- 开录、暂停、继续、结束是带共同边界（``capture_clock_ns``）的命令。线程按帧的去抖时刻排序命令与帧：
  第一帧采集于边界之后的帧放出时才执行该边界，因此 ``Timeline`` 的调用约定（边界之前的帧先推完、
  边界之后的帧不提前推）由这里保证。边界按 ``FrameClock.segment_shift`` 在段首取的推迟量换算，段内不变。
  边界之后迟迟没有帧（卡顿、断流）时，等 ``COMMAND_GRACE_S`` 后按已到的帧执行。
- 方向与尺寸在开录时固定（D-29）：本段每帧按开录时的旋转转正，writer 的尺寸取首个输出帧。
- 失败（D-27）：录制期间队列溢出、该路断流或写盘出错，这一段就失败，之后不再写，结束时返回错误；
  已写的文件保留。
- 写盘经 ``RecordingController``（懒创建 writer、写错误复位），只在本线程调用 ``write``/``release``。
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from core.foundation.tasks import CancelToken, TaskGroup
from core.media import (
    BoundaryShift,
    CapturedFrame,
    FrameClock,
    FrameClockStats,
    FrameQueue,
    Timeline,
    apply_rotation,
    capture_clock_ns,
)
from core.recording.controller import RecordingController, RecordingSnapshot, WriterFactory

QUEUE_CAPACITY = 8          # 每路录像队列容量（D-27；甲方实测峰值 2 帧）
COMMAND_GRACE_S = 0.6       # 边界之后这么久还没有新帧，就按已到的帧执行边界
STOP_WAIT_S = 3.0           # 结束一段时等录像线程确认的上限
PRIME_WAIT_S = 1.0          # 会话开始时等每路录像线程拿到首帧的上限（开录要用边界之前的帧）
IDLE_POLL_S = 0.05
_NS = 1_000_000_000


class FrameFeed(Protocol):
    """录像线程要的帧来源（``CameraStream`` 满足）：按角色订阅之后的每一帧、退订、查询断流原因。"""

    def subscribe(self, role: str, capacity: int, *, watermark: int | None = None) -> FrameQueue: ...
    def unsubscribe(self, role: str, tap: FrameQueue) -> None: ...
    def failure(self, role: str) -> BaseException | None: ...


@dataclass(frozen=True)
class ViewResult:
    """一段录制在这一路的结果：实际路径、写出帧数、错误，以及 ``capture.json`` 用的时间轴与补充字段。"""

    path: Path | None
    frames: int
    error: str | None
    timeline: Timeline | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False)
class _Command:
    kind: str                       # start / pause / resume / stop
    at_ns: int
    target: Path | None = None
    rotation: int = 0
    done: threading.Event = field(default_factory=threading.Event)
    result: ViewResult | None = None


class ViewWorker:
    """一路的录像线程；``start``/``pause``/``resume`` 不等待，``stop`` 等这一段写完（最多 ``STOP_WAIT_S``）。"""

    def __init__(self, view: str, role: str, feed: FrameFeed, *, fps: float, tasks: TaskGroup,
                 writer_factory: WriterFactory | None = None, capacity: int = QUEUE_CAPACITY) -> None:
        self.view = view
        self.role = role
        self._feed = feed
        self._fps = float(fps)
        self._capacity = int(capacity)
        kwargs = {} if writer_factory is None else {"writer_factory": writer_factory}
        self._controller = RecordingController(path_provider=self._target_path, **kwargs)
        self._controller.begin_session(fps=self._fps, size=(0, 0))
        self._clock = FrameClock(self._fps)
        self._lock = threading.Condition()
        self._commands: deque[_Command] = deque()
        self._error: str | None = None
        self._open = False              # 有一段已开录（命令已发出）且尚未结束
        self._overflow_base = 0
        self._dead = False
        self._closing = False
        self._primed = threading.Event()   # 已有去抖后的帧，可以开录
        # 以下只在录像线程里读写
        self._last: CapturedFrame | None = None
        self._timeline: Timeline | None = None
        self._shift: BoundaryShift | None = None
        self._shift_for: _Command | None = None
        self._target: Path | None = None
        self._rotation = 0
        self._rotated: tuple[tuple[int, int], np.ndarray] | None = None
        self._queue = feed.subscribe(role, self._capacity)
        self._handle = tasks.spawn(f"record-{view}", self._run, owner="recording", close_budget_s=STOP_WAIT_S)

    # ------------------------------------------------------------------ 命令（调用方线程）
    def start(self, at_ns: int, target: Path, rotation: int) -> None:
        with self._lock:
            self._error = None
            self._open = True
            self._overflow_base = self._queue.overflows
            self._enqueue(_Command("start", int(at_ns), Path(target), int(rotation)))

    def pause(self, at_ns: int) -> None:
        with self._lock:
            self._enqueue(_Command("pause", int(at_ns)))

    def resume(self, at_ns: int) -> None:
        with self._lock:
            self._enqueue(_Command("resume", int(at_ns)))

    def stop(self, at_ns: int, timeout: float = STOP_WAIT_S) -> ViewResult:
        """结束本段并等录像线程写完边界之前的帧；超时或线程已退出时返回错误结果。"""
        command = _Command("stop", int(at_ns))
        with self._lock:
            if self._dead:
                self._open = False
                return ViewResult(self._target, self._controller.snapshot().frames_written,
                                  self._error or "录像线程已退出")
            self._enqueue(command)
        if command.done.wait(timeout) and command.result is not None:
            return command.result
        with self._lock:
            self._open = False
        snap = self._controller.snapshot()
        return ViewResult(snap.result_path, snap.frames_written, self._error or "录像线程未按时结束")

    def close(self, timeout: float = STOP_WAIT_S) -> bool:
        """退订并停止线程（会话结束时，在 ``stop`` 之后调用）；按时退出返回 True。"""
        with self._lock:
            self._closing = True
            self._lock.notify_all()
        self._feed.unsubscribe(self.role, self._queue)
        self._handle.token.cancel("session_end")
        stopped = self._handle.join(timeout)
        self._controller.close_session()
        return stopped

    def wait_primed(self, timeout: float) -> bool:
        """等到已有可作开录首帧的帧；返回是否已就绪。"""
        return self._primed.wait(timeout)

    def _enqueue(self, command: _Command) -> None:
        self._commands.append(command)
        self._lock.notify_all()

    # ------------------------------------------------------------------ 查询
    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def snapshot(self) -> RecordingSnapshot:
        snap = self._controller.snapshot()
        error = self.error
        return replace(snap, last_error=error or snap.last_error)

    # ------------------------------------------------------------------ 录像线程
    def _run(self, token: CancelToken) -> None:
        try:
            while not token.cancelled:
                frame = self._queue.get(IDLE_POLL_S)
                if frame is not None:
                    self._check_overflow()
                    for released in self._clock.push(frame):
                        self._route(released)
                    self._apply_overdue()
                elif self._queue.closed:
                    self._drain_closed()
                    with self._lock:
                        self._lock.wait_for(lambda: self._commands or self._closing or token.cancelled, IDLE_POLL_S)
                else:
                    self._apply_overdue()
        except BaseException as exc:  # noqa: BLE001 - 线程出错也要让等待中的 stop 返回
            self._fail(f"录像线程出错：{exc}")
            raise
        finally:
            self._finish_dead()

    def _route(self, frame: CapturedFrame) -> None:
        """先执行边界早于这一帧的命令，再把帧交给时间轴。"""
        while True:
            command = self._head()
            if command is None or frame.captured_at_ns <= self._boundary(command):
                break
            self._apply(command)
        if self._timeline is not None and self.error is None and not self._timeline.stopped:
            self._write(self._timeline.push(frame))
        self._last = frame
        self._primed.set()

    def _apply_overdue(self) -> None:
        command = self._head()
        if command is None or capture_clock_ns() - command.at_ns < COMMAND_GRACE_S * _NS:
            return
        for released in self._clock.flush():
            self._route(released)
        while (command := self._head()) is not None and capture_clock_ns() - command.at_ns >= COMMAND_GRACE_S * _NS:
            self._apply(command)

    def _drain_closed(self) -> None:
        """reader 已停止：放出剩下的帧；录制中断流即本段失败，之后的命令直接执行。"""
        for released in self._clock.flush():
            self._route(released)
        with self._lock:
            pending = bool(self._commands)
            open_segment = self._open and not self._closing
        if open_segment and self.error is None and (pending or self._timeline is not None):
            failure = self._feed.failure(self.role)
            self._fail(str(failure) if failure is not None else "摄像头已停止出帧")
        while (command := self._head()) is not None:
            self._apply(command)

    def _head(self) -> _Command | None:
        with self._lock:
            return self._commands[0] if self._commands else None

    def _boundary(self, command: _Command) -> int:
        if command.kind == "start" and self._shift_for is not command:
            self._shift, self._shift_for = self._clock.segment_shift(command.at_ns), command
        return self._shift.apply(command.at_ns) if self._shift is not None else command.at_ns

    def _apply(self, command: _Command) -> None:
        boundary = self._boundary(command)
        with self._lock:
            self._commands.popleft()
        if command.kind == "start":
            self._begin(command, boundary)
        elif self._timeline is None or self.error is not None:
            pass
        elif command.kind == "pause" and self._timeline.recording:
            self._write(self._timeline.pause(boundary))
        elif command.kind == "resume" and not self._timeline.recording and not self._timeline.stopped:
            self._write(self._timeline.resume(boundary))
        if command.kind == "stop":
            command.result = self._end(boundary)
            command.done.set()

    def _begin(self, command: _Command, boundary: int) -> None:
        self._target, self._rotation, self._rotated = command.target, command.rotation, None
        self._clock.stats = FrameClockStats()
        self._queue.reset_peak()
        self._timeline = Timeline(self._fps)
        seed = self._last
        if seed is None or seed.captured_at_ns > boundary:
            self._timeline = None
            self._fail("开录时还没有这一路的画面")
            return
        self._timeline.start(boundary, seed)
        self._controller.request_toggle()

    def _end(self, boundary: int) -> ViewResult:
        timeline = self._timeline
        if timeline is not None and self.error is None:
            self._write(timeline.stop(boundary))
        snap = self._controller.snapshot()
        path = self._controller.stop_recording() or snap.result_path
        with self._lock:
            error = self._error or snap.last_error
            self._open = False
            overflows = self._queue.overflows - self._overflow_base
        extra = {
            "role": self.role,
            "rotation": self._rotation,
            "queue": {"capacity": self._capacity, "peak": self._queue.peak, "overflows": overflows},
            "frame_clock": self._clock.stats.as_dict(),
        }
        self._timeline, self._shift, self._shift_for = None, None, None
        return ViewResult(path, int(snap.frames_written), error, timeline, extra)

    def _write(self, frames: list[CapturedFrame]) -> None:
        if not frames or self.error is not None:
            return
        for frame in frames:
            image = self._oriented(frame)
            self._controller.update_session_size(size=(int(image.shape[1]), int(image.shape[0])))
            self._controller.write_frame(image)
        error = self._controller.snapshot().last_error
        if error:
            self._fail(f"写盘失败：{error}")

    def _oriented(self, frame: CapturedFrame) -> np.ndarray:
        key = (frame.generation, frame.seq)
        if self._rotated is None or self._rotated[0] != key:
            self._rotated = (key, apply_rotation(frame.bgr, self._rotation))
        return self._rotated[1]

    def _check_overflow(self) -> None:
        with self._lock:
            overflowed = self._open and self._queue.overflows > self._overflow_base
        if overflowed and self.error is None:
            self._fail(f"录像队列已满（{self._capacity} 帧），写盘跟不上采集")

    def _fail(self, message: str) -> None:
        with self._lock:
            if self._error is None:
                self._error = message

    def _finish_dead(self) -> None:
        with self._lock:
            self._dead = True
            pending, self._commands = list(self._commands), deque()
        snap = self._controller.snapshot()
        for command in pending:
            if command.kind == "stop":
                command.result = ViewResult(snap.result_path, snap.frames_written, self.error or "录像线程已退出")
            command.done.set()

    def _target_path(self) -> Path:
        if self._target is None:
            raise RuntimeError("没有进行中的录制片段")
        return self._target


__all__ = ["COMMAND_GRACE_S", "FrameFeed", "PRIME_WAIT_S", "QUEUE_CAPACITY", "STOP_WAIT_S", "ViewResult", "ViewWorker"]
