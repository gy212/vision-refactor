# -*- coding: utf-8 -*-
"""在任务线程里打开一路视频 writer 的 ``WriterOpener``：写出方不等打开完成，边取帧边取结果（D-18，设计文档 §8.8）。

1080p 的 ``avc1`` 经 Media Foundation 打开要 0.2～0.6 s，在写出线程里打开会让它这段时间不取帧，
有界帧队列随即排满。``WriterOpener`` 把打开放进 ``TaskGroup`` 的任务：

- ``take`` 取结果：打开成功时把 writer 交给调用方（之后由调用方写与释放），失败时抛出打开的异常，还没打开完返回 None；
- ``abandon`` 放弃：已打开、还没交出的 writer 立即释放；还在打开的，打开完由打开任务释放（这个 writer 从未写过）。
  交出之后 ``abandon`` 不再释放，所以无论在哪一步放弃都不会漏掉 writer。
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.foundation.tasks import CancelToken, TaskGroup

OpenWriter = Callable[[], tuple[Any, Path, str]]   # 打开一路 writer：返回 (writer, 实际路径, 编码)


@dataclass(frozen=True)
class OpenedWriter:
    """打开好的 writer 与实际路径、编码，以及打开用的毫秒数。"""

    writer: Any
    path: Path
    codec: str
    open_ms: float


def _release(writer: Any) -> None:
    try:
        writer.release()
    except Exception:  # noqa: BLE001 - 释放失败不影响放弃
        pass


class WriterOpener:
    """一次后台打开；创建即开始。``take``/``abandon``/``elapsed_s`` 可在任意线程调用。"""

    def __init__(self, open_writer: OpenWriter, tasks: TaskGroup, *, name: str, owner: str,
                 close_budget_s: float) -> None:
        self._open_writer = open_writer
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._opened: OpenedWriter | None = None
        self._error: Exception | None = None
        self._abandoned = False
        self._started = time.perf_counter()
        tasks.spawn(name, self._run, owner=owner, close_budget_s=close_budget_s)

    def _run(self, _token: CancelToken) -> None:
        opened: OpenedWriter | None = None
        error: Exception | None = None
        try:
            writer, path, codec = self._open_writer()
            opened = OpenedWriter(writer, Path(path), str(codec),
                                  round((time.perf_counter() - self._started) * 1000.0, 1))
        except Exception as exc:  # noqa: BLE001 - 打开失败交给 take 的调用方
            error = exc
        finally:
            if opened is None and error is None:
                error = RuntimeError("打开 writer 的任务异常退出")
            with self._lock:   # 先记下结果再置位，take 看到"已完成"时结果一定在
                keep = not self._abandoned
                self._error = error
                if keep:
                    self._opened = opened
            self._done.set()
        if opened is not None and not keep:
            _release(opened.writer)

    def elapsed_s(self) -> float:
        """开始打开至今的秒数。"""
        return time.perf_counter() - self._started

    def take(self, timeout: float = 0.0) -> OpenedWriter | None:
        """最多等 ``timeout`` 秒：打开成功返回结果并交出 writer（只交出一次），失败抛出打开的异常，仍在打开返回 None。"""
        if not self._done.wait(max(0.0, timeout)):
            return None
        with self._lock:
            if self._error is not None:
                raise self._error
            opened, self._opened = self._opened, None
        if opened is None:
            raise RuntimeError("writer 已交出或已放弃")
        return opened

    def abandon(self) -> None:
        """不再要这个 writer：已打开未交出的立即释放，仍在打开的由打开任务在打开后释放。"""
        with self._lock:
            self._abandoned = True
            opened, self._opened = self._opened, None
        if opened is not None:
            _release(opened.writer)


__all__ = ["OpenWriter", "OpenedWriter", "WriterOpener"]
