# -*- coding: utf-8 -*-
"""摄像头会话每路一个录像线程 ``ViewWorker``：按序取该路的源帧 → 去抖 → 换成恒定帧率 → 写盘（D-18，设计文档 §8.8）。

- 会话开始时订阅该路 reader 的有界帧队列（``QUEUE_CAPACITY`` 帧，reader 非阻塞投递），整个会话都在取帧：
  没有在录时也把帧交给 ``FrameClock``，开录时时钟已经稳定，并留着开录边界之前最近的一帧作首帧。
- 开录、暂停、继续、结束是带共同边界（``capture_clock_ns``）的命令。线程按帧的去抖时刻排序命令与帧：
  第一帧采集于边界之后的帧放出时才执行该边界，因此 ``Timeline`` 的调用约定（边界之前的帧先推完、
  边界之后的帧不提前推）由这里保证。边界按 ``FrameClock.segment_shift`` 在段首取的推迟量换算，段内不变。
  边界之后迟迟没有帧（卡顿、断流）时，等 ``COMMAND_GRACE_S`` 后按已到的帧执行。
- 方向与尺寸在开录时固定（D-29）：writer 按开录时的旋转打开（转正在 writer 里做，D-31），尺寸取开录首帧转正后的尺寸；
  默认 writer 是 ``open_recording_writer``（libjpeg-turbo 编码、ffmpeg 封装成 MJPG AVI）。
- writer 在开录时由任务线程打开（``WriterOpener``；打开要启动子进程或初始化编码器，在本线程打开会让队列排满）：
  打开期间本线程照常取帧，时间轴的输出帧按序暂存；打开后每轮先取一帧、再写一帧积压，追平后与之前一样直接写。
  打开超过 ``WRITER_OPEN_TIMEOUT_S``、打开失败或积压超过这段时间的帧数，这一段就失败并写明原因。
  结束时还在打开就等到上限，再写完积压。
- 失败（D-27）：录制期间队列溢出、该路断流或写盘出错，这一段就失败，之后不再写，结束时返回错误；
  已写的文件保留。
- 开录确认（D-27）：开录命令带回调，这一路写出本段首帧后回调 ``None``，在此之前失败则回调失败原因。
- 结束分两步：``request_stop`` 立即发出结束边界，``wait_stop`` 等这一路写完边界之前的帧（调用方可在别的线程等）。
- 写盘经 ``RecordingController``（写错误复位）：打开好的 writer 交给它之后只在本线程 ``write``/``release``；
  交出之前放弃（结束、失败、关闭）的 writer 由 ``WriterOpener`` 释放，从未写过。段末释放 writer 出错（封装进程
  异常退出、没按时写完文件）时这一段失败。``capture.json`` 的 ``writer`` 记下编码、封装方式与每帧写出耗时。
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol


from core.foundation.tasks import CancelToken, TaskGroup
from core.foundation.timing import ms_summary
from core.media import (
    BoundaryShift,
    CapturedFrame,
    FrameClock,
    FrameClockStats,
    FrameQueue,
    Timeline,
    WriterOpener,
    capture_clock_ns,
    open_recording_writer,
    rotated_size,
)
from core.recording.controller import RecordingController, RecordingSnapshot, VideoWriterLike

QUEUE_CAPACITY = 8          # 每路录像队列容量（D-27；甲方实测峰值 2 帧）
COMMAND_GRACE_S = 0.6       # 边界之后这么久还没有新帧，就按已到的帧执行边界
STOP_WAIT_S = 3.0           # 结束一段时等录像线程确认的上限
WRITER_OPEN_TIMEOUT_S = 1.5  # 开录时后台打开 writer 的上限（本机 1080p avc1 两路同时打开 0.2～0.6 s）；
                             # 加上去抖前瞻仍小于 START_CONFIRM_S，失败时报的是打开的原因而不是"没有写出首帧"
PRIME_WAIT_S = 1.0          # 会话开始时等每路录像线程拿到首帧的上限（开录要用边界之前的帧）
IDLE_POLL_S = 0.05
WRITE_SAMPLES = 9000        # 每帧写出耗时最多留这么多个样本（30 fps 下 5 分钟）
_NS = 1_000_000_000

StartCallback = Callable[[str | None], None]   # 写出首帧时 None，首帧之前失败时为原因
# 录像 writer 工厂：(目标路径, 帧率, 转正后的 (宽, 高), 旋转度数) → (writer, 实际落盘路径, codec 标签)；
# writer 的 ``write`` 收未转正的源帧，按旋转转正后写出。默认 ``open_recording_writer``。
ViewWriterFactory = Callable[[Path, float, tuple[int, int], int], tuple[VideoWriterLike, Path, str]]


def _default_view_writer(path: Path, fps: float, size: tuple[int, int], rotation: int):
    return open_recording_writer(path, fps=fps, size=size, rotation=rotation)


def _no_lazy_writer(_path: Path, _fps: float, _size: tuple[int, int]):
    raise RuntimeError("录像线程的 writer 只经后台打开交给控制器")


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
    on_started: StartCallback | None = None
    done: threading.Event = field(default_factory=threading.Event)
    result: ViewResult | None = None


class ViewWorker:
    """一路的录像线程；``start``/``pause``/``resume``/``request_stop`` 不等待，``wait_stop``/``stop`` 等这一段写完（最多 ``STOP_WAIT_S``）。"""

    def __init__(self, view: str, role: str, feed: FrameFeed, *, fps: float, tasks: TaskGroup,
                 writer_factory: ViewWriterFactory | None = None, capacity: int = QUEUE_CAPACITY) -> None:
        self.view = view
        self.role = role
        self._feed = feed
        self._fps = float(fps)
        self._capacity = int(capacity)
        self._writer_factory = writer_factory or _default_view_writer
        self._controller = RecordingController(writer_factory=_no_lazy_writer, path_provider=self._target_path)
        self._controller.begin_session(fps=self._fps, size=(0, 0))
        self._tasks = tasks
        self._backlog_limit = math.ceil(WRITER_OPEN_TIMEOUT_S * self._fps)
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
        self._on_started: StartCallback | None = None
        self._opening: WriterOpener | None = None      # 本段的 writer 还在后台打开
        self._backlog: deque[CapturedFrame] = deque()  # writer 接上并追平之前的输出帧
        self._writer_info: dict[str, Any] = {}
        self._attached: Any = None                     # 本段接上的 writer（取它的说明写进 capture.json）
        self._write_ms: deque[float] = deque(maxlen=WRITE_SAMPLES)
        self._queue = feed.subscribe(role, self._capacity)
        self._handle = tasks.spawn(f"record-{view}", self._run, owner="recording", close_budget_s=STOP_WAIT_S)

    # ------------------------------------------------------------------ 命令（调用方线程）
    def start(self, at_ns: int, target: Path, rotation: int, on_started: StartCallback | None = None) -> None:
        """开录；``on_started`` 在录像线程里回调一次：写出本段首帧时为 None，之前失败时为原因。"""
        with self._lock:
            self._error = None
            self._open = True
            self._overflow_base = self._queue.overflows
            self._enqueue(_Command("start", int(at_ns), Path(target), int(rotation), on_started))

    def pause(self, at_ns: int) -> None:
        with self._lock:
            self._enqueue(_Command("pause", int(at_ns)))

    def resume(self, at_ns: int) -> None:
        with self._lock:
            self._enqueue(_Command("resume", int(at_ns)))

    def stop(self, at_ns: int, timeout: float = STOP_WAIT_S) -> ViewResult:
        """结束本段并等录像线程写完边界之前的帧；超时或线程已退出时返回错误结果。"""
        return self.wait_stop(self.request_stop(at_ns), timeout)

    def request_stop(self, at_ns: int) -> _Command:
        """发出结束边界，不等待；返回的命令交给 ``wait_stop``。"""
        command = _Command("stop", int(at_ns))
        with self._lock:
            if self._dead:
                self._open = False
                command.result = ViewResult(self._target, self._controller.snapshot().frames_written,
                                            self._error or "录像线程已退出")
                command.done.set()
            else:
                self._enqueue(command)
        return command

    def wait_stop(self, command: _Command, timeout: float = STOP_WAIT_S) -> ViewResult:
        """等 ``request_stop`` 的边界写完；超时返回错误结果。"""
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
                catching_up = self._opening is None and bool(self._backlog)
                frame = self._queue.get(0.0 if catching_up else IDLE_POLL_S)
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
                self._advance_writer()
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
        self._drop_writer()
        self._target, self._rotation = command.target, command.rotation
        self._on_started = command.on_started
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
        self._open_writer(seed)

    def _end(self, boundary: int) -> ViewResult:
        with self._lock:   # 先记下：结束时写积压期间溢出的是边界之后的帧，不算本段
            overflows = self._queue.overflows - self._overflow_base
        timeline = self._timeline
        if timeline is not None and self.error is None:
            self._write(timeline.stop(boundary))
        self._flush_writer()
        release_error = self._controller.release_writer()
        if release_error is not None:
            self._fail(f"写盘失败：{release_error}")
        self._started(self.error or "写出首帧之前录制就结束了")
        snap = self._controller.snapshot()
        path = self._controller.stop_recording() or snap.result_path
        with self._lock:
            error = self._error or snap.last_error
            self._open = False
        extra = {
            "role": self.role,
            "rotation": self._rotation,
            "queue": {"capacity": self._capacity, "peak": self._queue.peak, "overflows": overflows},
            "frame_clock": self._clock.stats.as_dict(),
            "writer": self._writer_summary(),
        }
        self._timeline, self._shift, self._shift_for, self._attached = None, None, None, None
        return ViewResult(path, int(snap.frames_written), error, timeline, extra)

    def _write(self, frames: list[CapturedFrame]) -> None:
        """写出时间轴确认的输出帧；writer 还没接上或还有积压时排到积压之后。"""
        if not frames or self.error is not None:
            return
        if self._opening is None and not self._backlog:
            self._write_now(frames)
            return
        self._backlog.extend(frames)
        self._writer_info["backlog_peak"] = max(self._writer_info.get("backlog_peak", 0), len(self._backlog))
        if len(self._backlog) > self._backlog_limit:
            self._fail(self._open_too_slow() if self._opening is not None
                       else f"开录积压超过 {self._backlog_limit} 帧，写盘跟不上采集")

    def _write_now(self, frames: list[CapturedFrame]) -> None:
        for frame in frames:      # 时间轴补帧时同一帧重复出现，writer 复用它的编码
            started = time.perf_counter()
            self._controller.write_frame(frame.bgr)
            self._write_ms.append((time.perf_counter() - started) * 1000.0)
        snap = self._controller.snapshot()
        if snap.last_error:
            self._fail(f"写盘失败：{snap.last_error}")
        elif snap.frames_written > 0:
            self._started(None)

    # ------------------------------------------------------------------ writer 的后台打开与积压
    def _open_writer(self, seed: CapturedFrame) -> None:
        """按开录时的旋转与首帧转正后的尺寸，在任务线程里打开本段的 writer。"""
        height, width = seed.bgr.shape[:2]
        size = rotated_size(int(width), int(height), self._rotation)
        target, fps, rotation, factory = self._target, self._fps, self._rotation, self._writer_factory
        self._writer_info = {"backlog_peak": 0}
        self._write_ms.clear()
        self._opening = WriterOpener(lambda: factory(target, fps, size, rotation), self._tasks,
                                     name=f"open-writer-{self.view}", owner="recording",
                                     close_budget_s=WRITER_OPEN_TIMEOUT_S)

    def _attach_opened(self, timeout: float) -> bool:
        """取后台打开的结果：成功则交给控制器并返回 True；失败、超时使本段失败；仍在打开返回 False。"""
        opening = self._opening
        try:
            opened = opening.take(timeout)
        except Exception as exc:  # noqa: BLE001 - 打开失败即本段失败
            self._fail(f"打开录像文件失败：{exc}")
            return False
        if opened is None:
            if opening.elapsed_s() >= WRITER_OPEN_TIMEOUT_S:
                self._fail(self._open_too_slow())
            return False
        self._opening = None
        self._writer_info.update(codec=opened.codec, open_ms=opened.open_ms)
        if self._controller.attach_writer(opened.writer, opened.path):
            self._attached = opened.writer
            return True
        try:
            opened.writer.release()
        except Exception:  # noqa: BLE001 - 释放失败不影响报错
            pass
        self._fail("录像文件打开后录制状态已变化，未能接上")
        return False

    def _advance_writer(self) -> None:
        """每轮一次：writer 打开完就接上；接上后写一帧积压（先取帧再追积压，追赶期间队列不排满）。失败时丢掉积压。"""
        if self.error is not None:
            self._drop_writer()
            return
        if self._opening is not None and not self._attach_opened(0.0):
            return
        if self._backlog:
            self._write_now([self._backlog.popleft()])

    def _flush_writer(self) -> None:
        """结束时：writer 还在打开就等到打开上限，再按序写完全部积压；之后不再留着打开中的 writer 与积压。"""
        if self._opening is not None and self.error is None:
            self._attach_opened(max(0.0, WRITER_OPEN_TIMEOUT_S - self._opening.elapsed_s()))
            if self._opening is not None and self.error is None:
                self._fail(self._open_too_slow())
        if self.error is None and self._opening is None and self._backlog:
            frames = list(self._backlog)
            self._backlog.clear()
            self._write_now(frames)
        self._drop_writer()

    def _drop_writer(self) -> None:
        """放弃还在打开的 writer（由 ``WriterOpener`` 释放）并丢掉积压。"""
        opening, self._opening = self._opening, None
        if opening is not None:
            opening.abandon()
        self._backlog.clear()

    @staticmethod
    def _open_too_slow() -> str:
        return f"打开录像文件超过 {WRITER_OPEN_TIMEOUT_S:g} 秒仍未完成（视频编码器初始化太慢）"

    def _writer_summary(self) -> dict[str, Any]:
        """``capture.json`` 的 ``writer``：编码、打开耗时、开录积压峰值、writer 自己的说明（封装方式等）与每帧写出耗时。"""
        info = getattr(self._attached, "info", None)
        summary = {**self._writer_info, **(dict(info) if isinstance(info, dict) else {})}
        summary["write_ms"] = ms_summary(self._write_ms)
        return summary

    def _check_overflow(self) -> None:
        with self._lock:
            overflowed = self._open and self._queue.overflows > self._overflow_base
        if overflowed and self.error is None:
            self._fail(f"录像队列已满（{self._capacity} 帧），写盘跟不上采集")

    def _fail(self, message: str) -> None:
        with self._lock:
            if self._error is None:
                self._error = message
        self._started(message)

    def _started(self, error: str | None) -> None:
        """回调本段的开录结果（只回调一次）；只在录像线程里调用。"""
        callback, self._on_started = self._on_started, None
        if callback is not None:
            try:
                callback(error)
            except Exception:
                pass

    def _finish_dead(self) -> None:
        with self._lock:
            self._dead = True
            pending, self._commands = list(self._commands), deque()
        self._drop_writer()
        snap = self._controller.snapshot()
        self._started(self.error or "录像线程已退出")
        for command in pending:
            if command.kind == "start" and command.on_started is not None:
                command.on_started(self.error or "录像线程已退出")
            if command.kind == "stop":
                command.result = ViewResult(snap.result_path, snap.frames_written, self.error or "录像线程已退出")
            command.done.set()

    def _target_path(self) -> Path:
        if self._target is None:
            raise RuntimeError("没有进行中的录制片段")
        return self._target


__all__ = ["COMMAND_GRACE_S", "FrameFeed", "PRIME_WAIT_S", "QUEUE_CAPACITY", "STOP_WAIT_S", "ViewResult", "ViewWorker",
           "ViewWriterFactory", "WRITER_OPEN_TIMEOUT_S"]
