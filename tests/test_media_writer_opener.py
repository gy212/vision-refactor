# -*- coding: utf-8 -*-
"""``WriterOpener``（``core.media.writer_opener``）：在任务线程里打开 writer，交出一次，放弃时无论早晚都释放、不泄漏。"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from core.foundation import TaskGroup
from core.media import WriterOpener


class Writer:
    def __init__(self) -> None:
        self.released = threading.Event()

    def release(self) -> None:
        self.released.set()


def _opener(open_writer) -> WriterOpener:
    return WriterOpener(open_writer, TaskGroup("test"), name="open-writer", owner="test", close_budget_s=1.0)


def _slow(writer: Writer, delay: float, gate: threading.Event | None = None):
    def open_writer():
        if gate is not None:
            gate.wait(2.0)
        time.sleep(delay)
        return writer, Path("a.avi"), "MJPG"

    return open_writer


def test_take_hands_over_the_opened_writer_once_and_runs_off_the_caller_thread():
    writer = Writer()
    threads: list[str] = []

    def open_writer():
        threads.append(threading.current_thread().name)
        time.sleep(0.1)
        return writer, "a.avi", "MJPG"

    opener = _opener(open_writer)
    assert opener.take(0.0) is None                       # 还在打开：不阻塞调用方
    opened = opener.take(2.0)
    assert opened is not None and opened.writer is writer and opened.path == Path("a.avi")
    assert opened.codec == "MJPG" and opened.open_ms >= 90 and opener.elapsed_s() >= 0.1
    assert threads == ["test/test/open-writer"]
    opener.abandon()                                      # 已交出：放弃不释放
    assert not writer.released.is_set()
    with pytest.raises(RuntimeError):
        opener.take(0.0)


def test_open_failure_is_raised_by_take():
    def broken():
        raise RuntimeError("VideoWriter open failed")

    opener = _opener(broken)
    with pytest.raises(RuntimeError, match="VideoWriter open failed"):
        opener.take(2.0)


def test_abandon_after_open_releases_immediately():
    writer = Writer()
    opener = _opener(_slow(writer, 0.0))
    assert opener._done.wait(2.0)                         # 已打开、还没交出
    opener.abandon()
    assert writer.released.is_set()


def test_abandon_while_opening_releases_the_late_writer():
    writer = Writer()
    gate = threading.Event()
    opener = _opener(_slow(writer, 0.05, gate))
    opener.abandon()
    gate.set()
    assert writer.released.wait(2.0)
    with pytest.raises(RuntimeError):
        opener.take(2.0)
