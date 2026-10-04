# -*- coding: utf-8 -*-
"""Deterministic tests for the Tk-independent camera warmup pool."""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.camera_warmup import (  # noqa: E402
    CameraWarmupError,
    CameraWarmupPool,
    CameraWarmupStopped,
    CameraWarmupTimeout,
    CaptureSpec,
    PRIMARY,
    SECONDARY,
)


def _frame(value: int) -> np.ndarray:
    return np.full((2, 3, 3), value, dtype=np.uint8)


def _value(frame: np.ndarray) -> int:
    return int(frame[0, 0, 0])


def _eventually(predicate, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


class ControlledCapture:
    def __init__(self, *initial_values: int) -> None:
        self._condition = threading.Condition()
        self._frames = deque(_frame(value) for value in initial_values)
        self._error: BaseException | None = None
        self._released = False
        self._interrupted = False
        self.release_calls = 0
        self.interrupt_calls = 0
        self.read_calls = 0

    def isOpened(self) -> bool:
        with self._condition:
            return not self._released

    def read(self):
        with self._condition:
            self.read_calls += 1
            while (
                not self._frames
                and self._error is None
                and not self._released
                and not self._interrupted
            ):
                self._condition.wait()
            if self._error is not None:
                raise self._error
            if self._released:
                return False, None
            if self._interrupted:
                return False, None
            return True, self._frames.popleft()

    def interrupt(self) -> None:
        with self._condition:
            self.interrupt_calls += 1
            self._interrupted = True
            self._condition.notify_all()

    def push(self, value: int) -> None:
        with self._condition:
            self._frames.append(_frame(value))
            self._condition.notify_all()

    def fail(self, exc: BaseException) -> None:
        with self._condition:
            self._error = exc
            self._condition.notify_all()

    def release(self) -> None:
        with self._condition:
            self.release_calls += 1
            self._released = True
            self._condition.notify_all()


def test_wait_pair_opens_both_roles_concurrently() -> None:
    barrier = threading.Barrier(2)
    captures = {0: ControlledCapture(10), 1: ControlledCapture(20)}
    calls: list[int] = []
    calls_lock = threading.Lock()

    def factory(index: int):
        with calls_lock:
            calls.append(index)
        barrier.wait(timeout=1.0)
        return captures[index]

    pool = CameraWarmupPool(factory)
    try:
        primary, secondary = pool.wait_pair(
            0, 1, timeout=1.0, stop_event=threading.Event()
        )
        assert (_value(primary.frame), _value(secondary.frame)) == (10, 20)
        assert sorted(calls) == [0, 1]
    finally:
        pool.close()


def test_latest_only_snapshot_and_after_cursor() -> None:
    captures = {0: ControlledCapture(1), 1: ControlledCapture(10)}
    pool = CameraWarmupPool(captures.__getitem__)
    try:
        first = pool.wait_pair(0, 1, timeout=1.0, stop_event=threading.Event())
        cursor = (first[0].sequence, first[1].sequence)
        assert pool.snapshot_pair(0, 1, after=cursor) is None

        captures[0].push(2)
        captures[0].push(3)
        captures[1].push(11)

        latest = None

        def has_latest() -> bool:
            nonlocal latest
            latest = pool.snapshot_pair(0, 1, after=cursor)
            return latest is not None and _value(latest[0].frame) == 3

        _eventually(has_latest)
        assert latest is not None
        assert (_value(latest[0].frame), _value(latest[1].frame)) == (3, 11)
        new_cursor = (latest[0].sequence, latest[1].sequence)
        assert pool.snapshot_pair(0, 1, after=new_cursor) is None
    finally:
        pool.close()


def test_warm_same_role_and_index_is_idempotent() -> None:
    captures = {0: ControlledCapture(1), 1: ControlledCapture(2)}
    calls: list[int] = []

    def factory(index: int):
        calls.append(index)
        return captures[index]

    pool = CameraWarmupPool(factory)
    try:
        pool.warm(PRIMARY, 0)
        pool.warm(PRIMARY, 0)
        pool.wait_pair(0, 1, timeout=1.0, stop_event=threading.Event())
        pool.warm(PRIMARY, 0)
        assert calls.count(0) == 1
        assert calls.count(1) == 1
    finally:
        pool.close()


def test_role_index_change_invalidates_late_old_generation() -> None:
    old_entered = threading.Event()
    allow_old_open = threading.Event()
    old = ControlledCapture(90)
    captures = {1: ControlledCapture(10), 2: ControlledCapture(20)}

    def factory(index: int):
        if index == 0:
            old_entered.set()
            assert allow_old_open.wait(1.0)
            return old
        return captures[index]

    pool = CameraWarmupPool(factory)
    try:
        pool.warm(PRIMARY, 0)
        assert old_entered.wait(1.0)
        pool.warm(PRIMARY, 2)
        pool.warm(SECONDARY, 1)
        allow_old_open.set()

        pair = pool.wait_pair(2, 1, timeout=1.0, stop_event=threading.Event())
        assert (_value(pair[0].frame), _value(pair[1].frame)) == (20, 10)
        _eventually(lambda: old.release_calls == 1)
        assert pair[0].generation > 1
    finally:
        allow_old_open.set()
        pool.close()


def test_wait_pair_timeout_cancels_and_releases_both_roles() -> None:
    captures = {0: ControlledCapture(), 1: ControlledCapture()}
    pool = CameraWarmupPool(captures.__getitem__, join_timeout=0.2)

    with pytest.raises(CameraWarmupTimeout):
        pool.wait_pair(0, 1, timeout=0.05, stop_event=threading.Event())

    _eventually(lambda: all(cap.release_calls == 1 for cap in captures.values()))
    assert pool.snapshot_pair(0, 1) is None
    pool.close()


def test_wait_pair_stop_event_cancels_and_releases_both_roles() -> None:
    captures = {0: ControlledCapture(), 1: ControlledCapture()}
    stop_event = threading.Event()
    pool = CameraWarmupPool(captures.__getitem__, join_timeout=0.2)

    stopper = threading.Thread(
        target=lambda: (time.sleep(0.03), stop_event.set()), daemon=True
    )
    stopper.start()
    with pytest.raises(CameraWarmupStopped):
        pool.wait_pair(0, 1, timeout=1.0, stop_event=stop_event)

    _eventually(lambda: all(cap.release_calls == 1 for cap in captures.values()))
    assert pool.snapshot_pair(0, 1) is None
    pool.close()


def test_timeout_then_close_interrupts_retired_blocked_readers_once() -> None:
    captures = {0: ControlledCapture(), 1: ControlledCapture()}
    pool = CameraWarmupPool(captures.__getitem__, join_timeout=0.02)

    with pytest.raises(CameraWarmupTimeout):
        pool.wait_pair(0, 1, timeout=0.02, stop_event=threading.Event())

    pool.close()
    _eventually(lambda: all(cap.release_calls == 1 for cap in captures.values()))
    _eventually(
        lambda: not any(
            thread.name.startswith("camera-warmup-") and thread.is_alive()
            for thread in threading.enumerate()
        )
    )
    assert all(cap.release_calls == 1 for cap in captures.values())


def test_release_never_runs_concurrently_with_active_read() -> None:
    read_entered = threading.Event()
    allow_read_exit = threading.Event()

    class StrictOwnerCapture(ControlledCapture):
        def __init__(self) -> None:
            super().__init__()
            self.in_read = False
            self.release_during_read = False

        def read(self):
            self.in_read = True
            read_entered.set()
            allow_read_exit.wait(1.0)
            self.in_read = False
            return False, None

        def interrupt(self) -> None:
            super().interrupt()
            allow_read_exit.set()

        def release(self) -> None:
            self.release_during_read = self.in_read
            super().release()

    captures = {0: StrictOwnerCapture(), 1: StrictOwnerCapture()}
    pool = CameraWarmupPool(captures.__getitem__, join_timeout=0.01)
    try:
        with pytest.raises(CameraWarmupTimeout):
            pool.wait_pair(0, 1, timeout=0.01, stop_event=threading.Event())
        assert read_entered.wait(1.0)
        _eventually(lambda: all(cap.release_calls == 1 for cap in captures.values()))
        assert not any(cap.release_during_read for cap in captures.values())
    finally:
        allow_read_exit.set()
        pool.close()


def test_blocked_open_is_quarantined_and_retry_does_not_duplicate_open() -> None:
    allow_open = threading.Event()
    calls: list[int] = []
    captures = {0: ControlledCapture(1), 1: ControlledCapture(2)}

    def factory(index: int):
        calls.append(index)
        allow_open.wait(1.0)
        return captures[index]

    pool = CameraWarmupPool(factory, join_timeout=0.01)
    try:
        with pytest.raises(CameraWarmupTimeout):
            pool.wait_pair(0, 1, timeout=0.01, stop_event=threading.Event())
        assert sorted(calls) == [0, 1]
        with pytest.raises(CameraWarmupTimeout, match="上次采集仍未释放"):
            pool.wait_pair(0, 1, timeout=0.01, stop_event=threading.Event())
        assert sorted(calls) == [0, 1]
    finally:
        allow_open.set()
        _eventually(lambda: all(cap.release_calls == 1 for cap in captures.values()))
        pool.close()


@pytest.mark.parametrize("prewarm", [False, True])
@pytest.mark.parametrize("finish", ["ready", "stop", "timeout"])
def test_restart_waits_for_cancelled_capture_release(prewarm, finish) -> None:
    release_entered = threading.Event()
    allow_release = threading.Event()
    waiting = threading.Event()
    stop = threading.Event()
    result = {}
    calls = []

    class SlowReleaseCapture(ControlledCapture):
        def release(self):
            release_entered.set()
            assert allow_release.wait(2.0)
            super().release()

    previous = SlowReleaseCapture(10)
    current = {0: ControlledCapture(20), 1: ControlledCapture(30)}

    def factory(index):
        calls.append(index)
        return previous if len(calls) == 1 else current[index]

    pool = CameraWarmupPool(factory, join_timeout=0.01)

    def restart():
        waiting.set()
        try:
            if prewarm:
                pool.warm(PRIMARY, 0)
            result["pair"] = pool.wait_pair(
                0, 1, timeout=0.2 if finish == "timeout" else 1.0, stop_event=stop
            )
        except BaseException as exc:
            result["error"] = exc

    worker = threading.Thread(target=restart)
    try:
        pool.warm(PRIMARY, 0)
        _eventually(lambda: previous.read_calls >= 2)
        pool.cancel(PRIMARY)
        assert release_entered.wait(1.0)
        worker.start()
        assert waiting.wait(1.0)
        # The new reader may be scheduled, but the physical device stays exclusive.
        worker.join(0.05)
        assert "error" not in result, result.get("error")
        assert worker.is_alive()
        assert calls.count(0) == 1
        if finish != "ready":
            if finish == "stop":
                stop.set()
            worker.join(1.0)
            assert not worker.is_alive()
            expected = CameraWarmupStopped if finish == "stop" else CameraWarmupTimeout
            assert isinstance(result.get("error"), expected)
            allow_release.set()
            _eventually(lambda: previous.release_calls == 1)
            assert pool.snapshot_pair(0, 1) is None
            assert calls.count(0) == 1  # Cancelled retries must never open late.
            return
        allow_release.set()
        worker.join(1.0)
        assert not worker.is_alive()
        assert "error" not in result, result.get("error")
        assert [_value(frame.frame) for frame in result["pair"]] == [20, 30]
        assert calls.count(0) == 2
        assert previous.release_calls == 1
    finally:
        allow_release.set()
        if worker.ident is not None:
            worker.join(2.0)
        pool.close()


def test_release_failure_is_retained_and_retried() -> None:
    class RetryReleaseCapture(ControlledCapture):
        def __init__(self) -> None:
            super().__init__()
            self.release_attempts = 0

        def release(self) -> None:
            self.release_attempts += 1
            if self.release_attempts <= 2:
                raise RuntimeError("driver release failed")
            super().release()

    capture = RetryReleaseCapture()
    pool = CameraWarmupPool(lambda _index: capture, join_timeout=0.01)
    try:
        pool.warm(PRIMARY, 0)
        _eventually(lambda: capture.read_calls >= 1)
        pool.cancel(PRIMARY)
        _eventually(lambda: capture.release_attempts >= 2)
        assert capture.release_calls == 0
        pool.close()
        _eventually(lambda: capture.release_attempts >= 3)
        assert capture.release_calls == 1
    finally:
        pool.close()


def test_positional_only_stop_event_factory_uses_one_argument_contract() -> None:
    captures = {0: ControlledCapture(1), 1: ControlledCapture(2)}
    calls: list[int] = []

    def factory(index: int, stop_event=None, /):
        calls.append(index)
        return captures[index]

    pool = CameraWarmupPool(factory)
    try:
        pair = pool.wait_pair(0, 1, timeout=1.0, stop_event=threading.Event())
        assert pair[0].generation > 0
        assert sorted(calls) == [0, 1]
    finally:
        pool.close()


def test_claim_pair_stops_readers_and_transfers_both_captures() -> None:
    captures = {0: ControlledCapture(1), 1: ControlledCapture(2)}
    pool = CameraWarmupPool(captures.__getitem__)
    ready = pool.wait_pair(0, 1, timeout=1.0, stop_event=threading.Event())

    pump_done = threading.Event()

    def keep_cameras_streaming() -> None:
        value = 3
        while not pump_done.wait(0.005):
            for cap in captures.values():
                cap.push(value)
            value += 1

    pump = threading.Thread(target=keep_cameras_streaming, daemon=True)
    pump.start()
    try:
        claimed = pool.claim_pair(
            0,
            1,
            timeout=1.0,
            expected_generations=(ready[0].generation, ready[1].generation),
        )
    finally:
        pump_done.set()
        pump.join(1.0)

    assert tuple(lease.capture for lease in claimed) == (captures[0], captures[1])
    assert captures[0].release_calls == 0
    assert captures[1].release_calls == 0
    assert pool.snapshot_pair(0, 1) is None
    # lease 期间设备仍记为占用；close 不代替会话释放。
    assert pool.device_busy(0) and pool.device_busy(1)
    pool.close()
    assert captures[0].release_calls == 0
    assert captures[1].release_calls == 0

    for cap in claimed:
        cap.release()
    assert captures[0].release_calls == 1
    assert not pool.device_busy(0) and not pool.device_busy(1)


def test_claim_pair_never_opens_a_missing_pair() -> None:
    open_calls: list[int] = []
    pool = CameraWarmupPool(
        lambda index: (open_calls.append(index), ControlledCapture(1))[1]
    )
    try:
        with pytest.raises(CameraWarmupError, match="not ready"):
            pool.claim_pair(0, 1, timeout=0.1)
        assert open_calls == []
    finally:
        pool.close()


def test_claim_pair_rejects_changed_generations_without_replacing_pair() -> None:
    captures = {0: ControlledCapture(1), 1: ControlledCapture(2)}
    pool = CameraWarmupPool(captures.__getitem__)
    try:
        ready = pool.wait_pair(0, 1, timeout=1.0, stop_event=threading.Event())
        with pytest.raises(CameraWarmupError, match="generation changed"):
            pool.claim_pair(
                0,
                1,
                timeout=0.1,
                expected_generations=(ready[0].generation + 1, ready[1].generation),
            )
        assert pool.snapshot_pair(0, 1) is not None
    finally:
        pool.close()


def test_claim_pair_stop_event_aborts_blocked_reader_shutdown() -> None:
    captures = {0: ControlledCapture(1), 1: ControlledCapture(2)}
    stop_event = threading.Event()
    pool = CameraWarmupPool(
        captures.__getitem__, stop_poll_interval=0.005, join_timeout=0.02
    )
    ready = pool.wait_pair(0, 1, timeout=1.0, stop_event=threading.Event())
    result: dict[str, object] = {}

    def claim() -> None:
        try:
            result["caps"] = pool.claim_pair(
                0,
                1,
                timeout=1.0,
                expected_generations=(ready[0].generation, ready[1].generation),
                stop_event=stop_event,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            result["error"] = exc

    worker = threading.Thread(target=claim, daemon=True)
    worker.start()
    _eventually(
        lambda: pool._slots[PRIMARY] is not None  # noqa: SLF001
        and pool._slots[PRIMARY].claimed  # noqa: SLF001
    )
    stop_event.set()
    worker.join(0.3)

    assert not worker.is_alive()
    assert isinstance(result.get("error"), CameraWarmupStopped)
    assert "caps" not in result
    _eventually(lambda: all(cap.release_calls == 1 for cap in captures.values()))
    pool.close()


def test_wait_timeout_reaps_blocked_readers_without_close() -> None:
    captures = {0: ControlledCapture(), 1: ControlledCapture()}
    pool = CameraWarmupPool(captures.__getitem__, join_timeout=0.02)
    try:
        with pytest.raises(CameraWarmupTimeout):
            pool.wait_pair(0, 1, timeout=0.01, stop_event=threading.Event())
        _eventually(lambda: all(cap.release_calls == 1 for cap in captures.values()))
    finally:
        pool.close()


def test_warm_reader_start_failure_reaps_previous_slot(monkeypatch) -> None:
    previous = ControlledCapture(1)
    pool = CameraWarmupPool(lambda _index: previous, join_timeout=0.02)
    pool.warm(PRIMARY, 0)
    _eventually(lambda: previous.read_calls >= 2)

    real_start = threading.Thread.start

    def start(thread: threading.Thread) -> None:
        if thread.name == "camera-warmup-primary-2":
            raise RuntimeError("thread start failed")
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    try:
        with pytest.raises(CameraWarmupError, match="failed to start"):
            pool.warm(PRIMARY, 2)
        _eventually(lambda: previous.release_calls == 1)
        assert pool._slots[PRIMARY] is None  # noqa: SLF001
    finally:
        pool.close()


def test_pair_reader_start_failure_rolls_back_reused_role(monkeypatch) -> None:
    primary = ControlledCapture(1)
    calls: list[int] = []

    def factory(index: int):
        calls.append(index)
        return primary

    pool = CameraWarmupPool(factory, join_timeout=0.02)
    pool.warm(PRIMARY, 0)
    _eventually(lambda: primary.read_calls >= 2)
    real_start = threading.Thread.start

    def start(thread: threading.Thread) -> None:
        if thread.name == "camera-warmup-secondary-1":
            raise RuntimeError("thread start failed")
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    try:
        with pytest.raises(CameraWarmupError, match="failed to start"):
            pool.wait_pair(0, 1, timeout=0.2, stop_event=threading.Event())
        _eventually(lambda: primary.release_calls == 1)
        assert calls == [0]
        assert pool._slots[PRIMARY] is None  # noqa: SLF001
        assert pool._slots[SECONDARY] is None  # noqa: SLF001
    finally:
        pool.close()


def test_reaper_start_failure_uses_bounded_release_fallback(monkeypatch) -> None:
    capture = ControlledCapture(1)
    pool = CameraWarmupPool(lambda _index: capture, join_timeout=0.02)
    pool.warm(PRIMARY, 0)
    _eventually(lambda: capture.read_calls >= 2)
    real_start = threading.Thread.start

    def start(thread: threading.Thread) -> None:
        if thread.name.startswith("camera-warmup-reaper-"):
            raise RuntimeError("reaper start failed")
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    try:
        started = time.monotonic()
        pool.cancel(PRIMARY)
        assert time.monotonic() - started < 0.2
        _eventually(lambda: capture.release_calls == 1)
    finally:
        pool.close()


def test_wait_during_claim_is_rejected_without_reopening_or_breaking_claim() -> None:
    captures = {0: ControlledCapture(1), 1: ControlledCapture(2)}
    open_calls: list[int] = []

    def factory(index: int):
        open_calls.append(index)
        return captures[index]

    pool = CameraWarmupPool(factory)
    pool.wait_pair(0, 1, timeout=1.0, stop_event=threading.Event())
    result: dict[str, object] = {}

    def claim() -> None:
        try:
            result["caps"] = pool.claim_pair(0, 1, timeout=1.0)
        except BaseException as exc:  # pragma: no cover - asserted below
            result["error"] = exc

    worker = threading.Thread(target=claim, daemon=True)
    worker.start()
    _eventually(
        lambda: pool._slots[PRIMARY] is not None  # noqa: SLF001
        and pool._slots[PRIMARY].claimed  # noqa: SLF001
    )

    with pytest.raises(CameraWarmupError, match="being claimed"):
        pool.wait_pair(0, 1, timeout=0.2, stop_event=threading.Event())
    assert sorted(open_calls) == [0, 1]

    captures[0].push(3)
    captures[1].push(4)
    worker.join(1.0)
    assert not worker.is_alive()
    assert "error" not in result
    assert tuple(lease.capture for lease in result["caps"]) == (captures[0], captures[1])
    assert sorted(open_calls) == [0, 1]
    for cap in result["caps"]:
        cap.release()
    pool.close()


def test_claim_pair_failure_releases_other_ready_capture() -> None:
    captures = {0: ControlledCapture(1), 1: ControlledCapture(2)}
    pool = CameraWarmupPool(captures.__getitem__, join_timeout=0.2)
    pool.wait_pair(0, 1, timeout=1.0, stop_event=threading.Event())
    captures[1].fail(RuntimeError("secondary driver failed"))
    _eventually(lambda: captures[1].release_calls == 1)

    with pytest.raises(CameraWarmupError):
        pool.claim_pair(0, 1, timeout=0.5)

    captures[0].push(3)
    _eventually(lambda: captures[0].release_calls == 1)
    assert pool.snapshot_pair(0, 1) is None
    pool.close()


def test_cancel_and_close_release_resources_and_close_is_final() -> None:
    captures = {0: ControlledCapture(1), 1: ControlledCapture(2)}
    pool = CameraWarmupPool(captures.__getitem__)
    pool.wait_pair(0, 1, timeout=1.0, stop_event=threading.Event())

    pool.cancel(PRIMARY)
    captures[0].push(3)
    _eventually(lambda: captures[0].release_calls == 1)
    assert pool.snapshot_pair(0, 1) is None

    pool.close()
    pool.close()
    captures[1].push(4)
    _eventually(lambda: captures[1].release_calls == 1)
    with pytest.raises(CameraWarmupError, match="closed"):
        pool.warm(PRIMARY, 0)


def test_cancel_returns_before_blocking_driver_release_finishes() -> None:
    release_entered = threading.Event()
    allow_release = threading.Event()

    class BlockingReleaseCapture(ControlledCapture):
        def release(self) -> None:
            release_entered.set()
            assert allow_release.wait(1.0)
            super().release()

    capture = BlockingReleaseCapture(1)
    pool = CameraWarmupPool(lambda _index: capture, join_timeout=0.01)
    try:
        pool.warm(PRIMARY, 0)
        _eventually(lambda: capture.read_calls >= 2)

        started = time.monotonic()
        pool.cancel(PRIMARY)
        elapsed = time.monotonic() - started

        assert elapsed < 0.05
        assert release_entered.wait(1.0)
        assert capture.release_calls == 0
        allow_release.set()
        _eventually(lambda: capture.release_calls == 1)
    finally:
        allow_release.set()
        pool.close()


def test_spec_change_renegotiates_after_old_format_releases() -> None:
    """模式切换只改请求尺寸：旧格式释放完成后才按新尺寸打开，同尺寸则复用。"""
    release_entered = threading.Event()
    allow_release = threading.Event()
    calls: list[tuple[int, int, int]] = []

    class SlowReleaseCapture(ControlledCapture):
        def release(self) -> None:
            release_entered.set()
            assert allow_release.wait(2.0)
            super().release()

    old = SlowReleaseCapture(1)
    new = ControlledCapture(2)

    def factory(index: int, *, width: int, height: int):
        calls.append((index, width, height))
        return old if len(calls) == 1 else new

    preview = CaptureSpec(0, 1280, 720)
    student = CaptureSpec(0, 1920, 1080)
    pool = CameraWarmupPool(factory, join_timeout=0.01)
    try:
        pool.sync(preview, None)
        _eventually(lambda: old.read_calls >= 2)
        pool.sync(preview, None)
        assert calls == [(0, 1280, 720)]

        result: dict[str, object] = {}
        worker = threading.Thread(
            target=lambda: result.setdefault(
                "frame",
                pool.wait_one(student, timeout=2.0, stop_event=threading.Event()),
            )
        )
        worker.start()
        assert release_entered.wait(1.0)
        worker.join(0.05)
        assert worker.is_alive()
        assert calls == [(0, 1280, 720)]  # 旧进程未释放前不重叠打开同一设备
        allow_release.set()
        worker.join(1.0)
        assert calls == [(0, 1280, 720), (0, 1920, 1080)]
        assert _value(result["frame"].frame) == 2
    finally:
        allow_release.set()
        pool.close()


def test_leased_device_blocks_reopen_until_session_releases() -> None:
    opened: list[int] = []
    captures = [ControlledCapture(1), ControlledCapture(2)]

    def factory(index: int):
        opened.append(index)
        return captures[len(opened) - 1]

    pool = CameraWarmupPool(factory, join_timeout=0.01)
    try:
        ready = pool.wait_one(0, timeout=1.0, stop_event=threading.Event())
        captures[0].push(3)  # 真实采集 read 有界返回；模拟驱动需补一帧让 reader 退出
        lease = pool.claim_one(0, timeout=1.0, expected_generation=ready.generation)
        pool.warm(PRIMARY, 0)
        time.sleep(0.05)
        assert opened == [0]  # 会话持有 lease 时，下一次预热只能等待
        lease.release()
        _eventually(lambda: opened == [0, 0])
        assert captures[0].release_calls == 1
    finally:
        pool.close()


def test_failed_lease_release_is_quarantined_then_retried_before_reopen() -> None:
    class FailOnceCapture(ControlledCapture):
        def __init__(self, *values: int) -> None:
            super().__init__(*values)
            self.release_attempts = 0

        def release(self) -> None:
            self.release_attempts += 1
            if self.release_attempts == 1:
                raise RuntimeError("driver stuck")
            super().release()

    stuck = FailOnceCapture(1)
    fresh = ControlledCapture(2)
    opened: list[int] = []

    def factory(index: int):
        opened.append(index)
        return stuck if len(opened) == 1 else fresh

    pool = CameraWarmupPool(factory, join_timeout=0.01, release_retry_interval=0.0)
    try:
        ready = pool.wait_one(0, timeout=1.0, stop_event=threading.Event())
        stuck.push(3)
        lease = pool.claim_one(0, timeout=1.0, expected_generation=ready.generation)
        with pytest.raises(CameraWarmupError, match="隔离"):
            lease.release()
        assert pool.device_busy(0)
        frame = pool.wait_one(0, timeout=1.0, stop_event=threading.Event())
        assert _value(frame.frame) == 2
        assert stuck.release_attempts == 2 and stuck.release_calls == 1
        assert opened == [0, 0]
    finally:
        pool.close()


def test_generation_scoped_cancel_keeps_newer_warmup() -> None:
    captures = {0: ControlledCapture(1)}
    pool = CameraWarmupPool(captures.__getitem__, join_timeout=0.01)
    try:
        first = pool.wait_one(0, timeout=1.0, stop_event=threading.Event())
        pool.cancel(PRIMARY, generation=first.generation + 1)
        assert pool.device_busy(0)
        pool.cancel(PRIMARY, generation=first.generation)
        _eventually(lambda: captures[0].release_calls == 1)
    finally:
        pool.close()


def test_sync_swaps_roles_without_rejecting_same_index() -> None:
    captures = {0: ControlledCapture(1, 3), 1: ControlledCapture(2, 4)}
    reopened = {0: ControlledCapture(5), 1: ControlledCapture(6)}
    calls: list[int] = []

    def factory(index: int):
        calls.append(index)
        return captures[index] if calls.count(index) == 1 else reopened[index]

    pool = CameraWarmupPool(factory, join_timeout=0.01)
    try:
        pool.wait_pair(0, 1, timeout=1.0, stop_event=threading.Event())
        swapped = pool.wait_pair(1, 0, timeout=1.0, stop_event=threading.Event())
        assert (_value(swapped[0].frame), _value(swapped[1].frame)) == (6, 5)
        assert all(cap.release_calls == 1 for cap in captures.values())
    finally:
        pool.close()


@pytest.mark.parametrize(
    ("role", "index"),
    [("other", 0), (PRIMARY, -1), (PRIMARY, True)],
)
def test_warm_rejects_invalid_role_or_index(role, index) -> None:
    pool = CameraWarmupPool(lambda unused: ControlledCapture(1))
    try:
        with pytest.raises(ValueError):
            pool.warm(role, index)
    finally:
        pool.close()


def test_pair_rejects_same_camera_index() -> None:
    pool = CameraWarmupPool(lambda unused: ControlledCapture(1))
    try:
        with pytest.raises(ValueError, match="different"):
            pool.wait_pair(0, 0, timeout=1.0, stop_event=threading.Event())
    finally:
        pool.close()


def test_pre_set_stop_event_does_not_start_or_open_cameras() -> None:
    open_calls: list[int] = []

    def factory(index: int):
        open_calls.append(index)
        return ControlledCapture(1)

    stop_event = threading.Event()
    stop_event.set()
    pool = CameraWarmupPool(factory)
    try:
        with pytest.raises(CameraWarmupStopped):
            pool.wait_pair(0, 1, timeout=1.0, stop_event=stop_event)
        assert open_calls == []
    finally:
        pool.close()


def test_warm_rejects_same_index_for_other_role_without_replacing_first() -> None:
    cap = ControlledCapture(1)
    calls: list[int] = []

    def factory(index: int):
        calls.append(index)
        return cap

    pool = CameraWarmupPool(factory)
    try:
        pool.warm(PRIMARY, 0)
        with pytest.raises(ValueError, match="both roles"):
            pool.warm(SECONDARY, 0)
        _eventually(lambda: calls == [0])
        time.sleep(0.02)
        assert calls == [0]
    finally:
        pool.cancel(PRIMARY)
        cap.push(2)
        pool.close()
