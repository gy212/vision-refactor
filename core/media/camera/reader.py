# -*- coding: utf-8 -*-
"""会话期间每路摄像头唯一的持续读帧线程 ``CaptureReader``，以及它发布帧所用的最新帧槽 ``LatestSlot`` 与有界队列 ``FrameQueue``（设计文档 §8.8）。

- 接管租约后，``CameraStream`` 为每路启动一个 ``CaptureReader``：持续读 → 校验 → 打采集时间戳与序号 →
  发布到该路的 ``LatestSlot``。两路各读各的，一路慢或卡住不拖住另一路。
- 消费方（预览、推理）按自己的序号游标从槽里取更新；处理不过来时只拿到最新帧，被覆盖的帧不排队。
- 要按序拿到每一帧的消费方（录像）订阅一个 ``FrameQueue``：reader 非阻塞投递，队列满时这一帧不进队列、
  记一次溢出，由消费方判定失败；reader 从不等待消费方。
- ``read_stamped`` 是读一帧并取采集时刻的唯一实现，预热 reader 与会话 reader 共用。
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any, Generic, TypeVar

import numpy as np

from core.foundation.tasks import CancelToken, TaskGroup
from core.media.camera.spec import ROLES
from core.media.frames import CapturedFrame, capture_clock_ns, is_valid_bgr

T = TypeVar("T")

READER_STOP_BUDGET_S = 2.5  # 覆盖 ProcessCamera 一次读帧的 2 秒超时


class CameraReadError(RuntimeError):
    """会话读帧失败（设备断开或采集进程已关闭）。"""

    def __init__(self, role: str, *, numbered: bool) -> None:
        prefix = f"第{ROLES.index(role) + 1}路" if numbered else ""
        super().__init__(f"{prefix}摄像头读帧失败，请检查连接后重新开始")
        self.role = role


def read_stamped(capture: Any) -> tuple[bool, np.ndarray | None, int]:
    """读一帧并取采集时刻（``capture_clock_ns``）。采集子进程给出时刻时用它（``ProcessCamera.read_with_time``），
    这样不含管道往返与像素拷贝的耗时；其它设备在读完后立即取时刻。"""
    read_with_time = getattr(capture, "read_with_time", None)
    if callable(read_with_time):
        return read_with_time()
    ok, frame = capture.read()
    return ok, frame, capture_clock_ns()


class LatestSlot(Generic[T]):
    """只保留最新一项的槽。发布方覆盖旧项并递增序号；读取方带着自己看到的序号 ``after`` 取更新的项。

    槽关闭后 ``take`` 不再等待：有比 ``after`` 新的项照常返回，否则返回 None；关闭时带的错误由 ``error`` 给出。
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._seq = 0
        self._item: T | None = None
        self._closed = False
        self._error: BaseException | None = None

    def publish(self, item: T) -> int:
        with self._condition:
            if self._closed:
                return self._seq
            self._seq += 1
            self._item = item
            self._condition.notify_all()
            return self._seq

    def close(self, error: BaseException | None = None) -> None:
        with self._condition:
            if not self._closed:
                self._closed = True
                self._error = error
            self._condition.notify_all()

    @property
    def seq(self) -> int:
        with self._condition:
            return self._seq

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    @property
    def error(self) -> BaseException | None:
        with self._condition:
            return self._error

    def take(self, after: int = 0, timeout: float | None = None) -> tuple[int, T] | None:
        """返回 ``(序号, 项)``，序号大于 ``after``；在 ``timeout`` 内没有更新、或槽已关闭且没有更新时返回 None。"""
        with self._condition:
            self._condition.wait_for(lambda: self._seq > after or self._closed, timeout)
            if self._seq > after and self._item is not None:
                return self._seq, self._item
            return None


class FrameQueue:
    """有界 FIFO。``put_nowait`` 不阻塞：队列满时丢掉这一帧并计入 ``overflows``；``watermark`` 统计到达时
    已有至少这么多帧排队的次数（用于"按更小容量运行会不会溢出"的测量）。``peak`` 是排队帧数的峰值。"""

    def __init__(self, capacity: int, *, watermark: int | None = None) -> None:
        if capacity < 1:
            raise ValueError("队列容量至少为 1")
        self.capacity = int(capacity)
        self.watermark = watermark
        self._items: deque[CapturedFrame] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self.peak = 0
        self.overflows = 0
        self.over_watermark = 0
        self.accepted = 0

    def put_nowait(self, frame: CapturedFrame) -> bool:
        with self._condition:
            if self._closed:
                return False
            depth = len(self._items)
            if self.watermark is not None and depth >= self.watermark:
                self.over_watermark += 1
            if depth >= self.capacity:
                self.overflows += 1
                return False
            self._items.append(frame)
            self.accepted += 1
            self.peak = max(self.peak, depth + 1)
            self._condition.notify_all()
            return True

    def get(self, timeout: float | None = None) -> CapturedFrame | None:
        """按序取一帧；``timeout`` 内没有帧、或已关闭且取空时返回 None。"""
        with self._condition:
            self._condition.wait_for(lambda: self._items or self._closed, timeout)
            return self._items.popleft() if self._items else None

    def reset_peak(self) -> None:
        """峰值从此刻的排队帧数重新统计（录像每段开始时调用）。"""
        with self._condition:
            self.peak = len(self._items)

    def close(self) -> None:
        """不再接收新帧；已排队的帧仍可取完。"""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    @property
    def depth(self) -> int:
        with self._condition:
            return len(self._items)


class CaptureReader:
    """一路设备一个读帧线程，产出该路的 ``CapturedFrame``。读失败、无效帧或驱动异常时停止并关闭槽，
    错误是 ``CameraReadError``（驱动异常作为其 ``__cause__``）。"""

    def __init__(
        self,
        role: str,
        capture: Any,
        tasks: TaskGroup,
        *,
        generation: int = 0,
        numbered: bool = False,
    ) -> None:
        self.role = role
        self.latest: LatestSlot[CapturedFrame] = LatestSlot()
        self._capture = capture
        self._generation = generation
        self._numbered = numbered
        self._taps: list[FrameQueue] = []
        self._taps_lock = threading.Lock()
        self._handle = tasks.spawn(
            f"reader-{role}", self._run, owner="camera-reader", close_budget_s=READER_STOP_BUDGET_S,
        )

    def _run(self, token: CancelToken) -> None:
        seq = 0
        error: BaseException | None = None
        try:
            while not token.cancelled:
                ok, frame, captured_at_ns = read_stamped(self._capture)
                if token.cancelled:
                    break
                if not ok or not is_valid_bgr(frame):
                    error = CameraReadError(self.role, numbered=self._numbered)
                    break
                seq += 1
                captured = CapturedFrame(self.role, seq, captured_at_ns, frame, self._generation)
                self.latest.publish(captured)
                with self._taps_lock:
                    taps = tuple(self._taps)
                for tap in taps:
                    tap.put_nowait(captured)
        except BaseException as exc:
            if not token.cancelled:
                error = CameraReadError(self.role, numbered=self._numbered)
                error.__cause__ = exc
        finally:
            self.latest.close(error)
            with self._taps_lock:
                taps, self._taps = tuple(self._taps), []
            for tap in taps:
                tap.close()

    def subscribe(self, capacity: int, *, watermark: int | None = None) -> FrameQueue:
        """订阅之后采到的每一帧（按采集顺序）；reader 停止时队列被关闭。"""
        tap = FrameQueue(capacity, watermark=watermark)
        with self._taps_lock:
            if self.latest.closed:
                tap.close()
            else:
                self._taps.append(tap)
        return tap

    def unsubscribe(self, tap: FrameQueue) -> None:
        with self._taps_lock:
            if tap in self._taps:
                self._taps.remove(tap)
        tap.close()

    def request_stop(self) -> None:
        """请求停止，不等待；多路同时停止时先逐路请求，再逐路 ``stop``。"""
        self._handle.token.cancel("stream_close")

    def stop(self, timeout: float | None = None) -> bool:
        """请求停止并等线程退出（默认等 ``READER_STOP_BUDGET_S``）；按时退出返回 True，此后不会再读这台设备。"""
        self.request_stop()
        return self._handle.join(READER_STOP_BUDGET_S if timeout is None else timeout)


__all__ = ["CameraReadError", "CaptureReader", "FrameQueue", "LatestSlot", "READER_STOP_BUDGET_S", "read_stamped"]
