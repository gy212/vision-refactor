# -*- coding: utf-8 -*-
"""会话 reader（设计文档 §8.8，D-18 第 2 层）：每路持续读帧、最新帧槽、采集时间戳与停止顺序。"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from core.foundation import CancelToken, TaskGroup
from core.media import (
    PRIMARY,
    SECONDARY,
    CameraPool,
    CameraReadError,
    CameraService,
    CaptureReader,
    CaptureSpec,
    LatestSlot,
    ProcessCamera,
    capture_clock_ns,
)
from core.media.camera import reader as reader_module


def _eventually(predicate, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


class PacedCapture:
    """按 ``interval`` 出帧的假驱动，帧值是读帧计数；``opened`` 记录各设备号的实例。"""

    opened: dict[int, "PacedCapture"] = {}
    intervals: dict[int, float] = {}

    def __init__(self, index, *, width=1280, height=720, **_kwargs) -> None:
        self.index = index
        self.interval = PacedCapture.intervals.get(index, 0.005)
        self.reads = 0
        self.released = False
        self.read_after_release = 0
        self.lock = threading.Lock()
        PacedCapture.opened[index] = self

    def isOpened(self) -> bool:
        return not self.released

    def get(self, prop):
        return {3: 64, 4: 48, 5: 30.0}.get(prop, 0.0)

    def read(self):
        time.sleep(self.interval)
        with self.lock:
            if self.released:
                self.read_after_release += 1
                return False, None
            self.reads += 1
            return True, np.full((48, 64, 3), self.reads % 251, np.uint8)

    def release(self) -> None:
        with self.lock:
            self.released = True


@pytest.fixture
def paced():
    PacedCapture.opened, PacedCapture.intervals = {}, {}
    svc = CameraService(pool=CameraPool(PacedCapture, join_timeout=0.5))
    try:
        yield svc
    finally:
        svc.close(2.0)


def _claimed(svc, specs):
    stream = svc.open_stream(specs)
    stream.wait_ready(CancelToken())
    stream.claim(CancelToken())
    return stream


# --------------------------------------------------------------------------- #
# LatestSlot
# --------------------------------------------------------------------------- #


def test_latest_slot_keeps_only_newest_and_reports_skipped_by_sequence():
    slot: LatestSlot[str] = LatestSlot()
    assert slot.take(0, timeout=0.01) is None
    for item in ("a", "b", "c"):
        slot.publish(item)
    assert slot.take(0, timeout=0.01) == (3, "c")       # 被覆盖的 a、b 不排队
    assert slot.take(3, timeout=0.01) is None          # 没有更新
    slot.publish("d")
    assert slot.take(3, timeout=0.01) == (4, "d")


def test_latest_slot_wakes_waiting_reader_and_stops_waiting_once_closed():
    slot: LatestSlot[int] = LatestSlot()
    threading.Timer(0.05, slot.publish, args=(7,)).start()
    assert slot.take(0, timeout=2.0) == (1, 7)
    error = RuntimeError("gone")
    slot.close(error)
    started = time.monotonic()
    assert slot.take(1, timeout=2.0) is None
    assert time.monotonic() - started < 0.5
    assert slot.closed and slot.error is error
    assert slot.publish(8) == 1                         # 关闭后不再接收


# --------------------------------------------------------------------------- #
# CaptureReader
# --------------------------------------------------------------------------- #


class StampedCapture:
    """带 ``read_with_time`` 的假设备：时间戳由设备给出（模拟采集子进程）。"""

    def __init__(self) -> None:
        self.stamp = 1_000

    def read_with_time(self):
        time.sleep(0.002)
        self.stamp += 33_000_000
        return True, np.zeros((4, 4, 3), np.uint8), self.stamp

    def read(self):  # pragma: no cover - 有 read_with_time 时不应调用
        raise AssertionError("read() must not be used when read_with_time() exists")


def test_reader_uses_device_timestamps_and_numbers_frames():
    tasks = TaskGroup("test")
    reader = CaptureReader(PRIMARY, StampedCapture(), tasks, generation=5)
    try:
        _eventually(lambda: reader.latest.seq >= 3)
        seq, frame = reader.latest.take(0, timeout=1.0)
        assert frame.role == PRIMARY and frame.generation == 5
        assert frame.seq == seq
        assert frame.captured_at_ns == 1_000 + 33_000_000 * frame.seq
    finally:
        assert reader.stop()


def test_reader_without_device_timestamp_stamps_right_after_read():
    tasks = TaskGroup("test")
    before = capture_clock_ns()
    capture = PacedCapture(9)
    reader = CaptureReader(PRIMARY, capture, tasks)
    try:
        _, first = reader.latest.take(0, timeout=1.0)
        _eventually(lambda: reader.latest.seq > first.seq)
        _, later = reader.latest.take(first.seq, timeout=1.0)
        assert before < first.captured_at_ns < later.captured_at_ns <= capture_clock_ns()
    finally:
        assert reader.stop()


def test_reader_failure_closes_slot_with_numbered_read_error():
    tasks = TaskGroup("test")
    capture = PacedCapture(3)
    reader = CaptureReader(SECONDARY, capture, tasks, numbered=True)
    _eventually(lambda: reader.latest.seq >= 1)
    capture.release()
    _eventually(lambda: reader.latest.closed)
    assert isinstance(reader.latest.error, CameraReadError)
    assert "第2路" in str(reader.latest.error)
    assert reader.stop(0.5)


def test_capture_clock_resolves_well_below_one_output_frame():
    """§8.8：采集时钟必须远细于 33 ms 的输出帧周期（``time.monotonic`` 在 Windows 上是 15.6 ms）。"""
    assert time.get_clock_info("perf_counter").resolution < 1e-4
    a = capture_clock_ns()
    while (b := capture_clock_ns()) == a:
        pass
    assert 0 < b - a < 1_000_000


# --------------------------------------------------------------------------- #
# CameraStream：持续读帧与取帧
# --------------------------------------------------------------------------- #


def test_readers_keep_capturing_while_the_consumer_is_slow(paced):
    """消费方慢（相当于推理 5 fps）时，各路 reader 照常读帧；取到的总是最新帧。"""
    stream = _claimed(paced, {PRIMARY: CaptureSpec(0), SECONDARY: CaptureSpec(1)})
    try:
        first = stream.read()
        reads_before = {i: PacedCapture.opened[i].reads for i in (0, 1)}
        time.sleep(0.2)
        second = stream.read()
        for index, role in ((0, PRIMARY), (1, SECONDARY)):
            assert PacedCapture.opened[index].reads - reads_before[index] >= 10
            assert second[role].seq - first[role].seq >= 10
    finally:
        stream.close()


def test_slow_second_camera_does_not_slow_the_first(paced):
    """两路各读各的：第二路 100 ms 一帧时，第一路仍按自己的速度采集。"""
    PacedCapture.intervals = {0: 0.005, 1: 0.1}
    stream = _claimed(paced, {PRIMARY: CaptureSpec(0), SECONDARY: CaptureSpec(1)})
    try:
        start = {i: PacedCapture.opened[i].reads for i in (0, 1)}
        time.sleep(0.5)
        fast = PacedCapture.opened[0].reads - start[0]
        slow = PacedCapture.opened[1].reads - start[1]
        assert slow <= 7
        assert fast >= 5 * slow
    finally:
        stream.close()


def test_read_never_returns_the_same_frame_twice(paced):
    stream = _claimed(paced, {PRIMARY: CaptureSpec(0), SECONDARY: CaptureSpec(1)})
    try:
        seen = [stream.read() for _ in range(5)]
        for role in (PRIMARY, SECONDARY):
            seqs = [frames[role].seq for frames in seen]
            assert seqs == sorted(set(seqs))
            stamps = [frames[role].captured_at_ns for frames in seen]
            assert stamps == sorted(set(stamps))
    finally:
        stream.close()


def test_read_honours_cancel_while_waiting_for_a_new_frame(paced):
    PacedCapture.intervals = {0: 0.5}
    stream = _claimed(paced, {PRIMARY: CaptureSpec(0)})
    try:
        stream.read()
        token = CancelToken()
        threading.Timer(0.05, token.cancel, args=("stop",)).start()
        started = time.monotonic()
        with pytest.raises(Exception) as info:
            stream.read(token)
        assert type(info.value).__name__ == "Cancelled"
        assert time.monotonic() - started < 0.4
    finally:
        stream.close()


def test_close_stops_readers_before_returning_the_device(paced):
    stream = _claimed(paced, {PRIMARY: CaptureSpec(0)})
    stream.read()
    capture = PacedCapture.opened[0]
    stream.close(keep_warm=False)
    assert capture.released
    assert capture.read_after_release == 0          # reader 已停，释放后没人再读这台设备


def test_reader_that_misses_its_stop_budget_is_released_not_returned(paced, monkeypatch):
    """reader 卡在一次读帧里超过预算时，这一路不交回预热，直接释放。"""
    PacedCapture.intervals = {0: 0.005}
    stream = _claimed(paced, {PRIMARY: CaptureSpec(0)})
    stream.read()
    capture = PacedCapture.opened[0]
    capture.interval = 0.6
    time.sleep(0.02)                                # 让 reader 进入一次慢读
    monkeypatch.setattr(reader_module, "READER_STOP_BUDGET_S", 0.05)
    stream.close(keep_warm=True)
    assert capture.released


# --------------------------------------------------------------------------- #
# ProcessCamera：采集子进程打时间戳（真实 spawn）
# --------------------------------------------------------------------------- #


class ChildDriver:
    def __init__(self, index, *, width, height):
        self.size = (width, height)

    def get(self, prop):
        return 30

    def read(self):
        time.sleep(0.01)
        return True, np.zeros((self.size[1], self.size[0], 3), np.uint8)

    def release(self):
        pass


def test_process_camera_timestamps_come_from_the_child_on_the_shared_clock():
    camera = ProcessCamera(0, width=64, height=48, _factory=ChildDriver)
    try:
        ok, _, opened_stamp = camera.read_with_time()   # 打开时读到的首帧，时刻是打开时的
        assert ok and opened_stamp < capture_clock_ns()
        stamps = [opened_stamp]
        for _ in range(3):
            before = capture_clock_ns()
            ok, frame, stamp = camera.read_with_time()
            after = capture_clock_ns()
            assert ok and frame.shape == (48, 64, 3)
            assert before < stamp < after             # 子进程的时刻落在主进程这次调用之内：同一时钟
            stamps.append(stamp)
        assert stamps == sorted(set(stamps))
        ok, frame = camera.read()
        assert ok and frame.shape == (48, 64, 3)
    finally:
        camera.release()
