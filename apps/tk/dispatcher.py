# -*- coding: utf-8 -*-
"""后台线程回到 Tk 主线程的唯一通道 ``UiDispatcher``：线程安全地投递回调，主线程每 16 ms 排空一次。

- 后台线程只调用 ``post(callback, token=None)``；Tk 不保证跨线程调用安全，后台线程不得直接调用 ``after``。
- ``token`` 是"结果仍有效"的判定（见 ``epoch_token``），排空时判定为假的回调被丢弃。
- 进入关闭状态（``close()``）后只执行以 ``during_close=True`` 投递的回调（关闭协调器自己的回调）；
  窗口销毁（``destroy()``）后拒绝一切回调。
- 每次排空之后依次调用 ``every_tick`` 登记的逐帧工作（渲染最新预览帧等），它们同样在 Tk 主线程运行。
"""
from __future__ import annotations

import threading
import traceback
from collections import deque
from collections.abc import Callable
from tkinter import TclError
from typing import Any

from core.foundation import Epoch

Token = Callable[[], bool]

TICK_MS = 16


def epoch_token(epoch: Epoch) -> Token:
    """以 ``epoch`` 的当前代次作 token：所有者推进代次后，此前投递的回调一律过期。"""
    stamp = epoch.current
    return lambda: epoch.is_current(stamp)


class UiDispatcher:
    """Tk 主线程的回调泵。``start()`` 之后由 ``root.after`` 每 ``interval_ms`` 调用一次 ``pump()``。"""

    def __init__(self, root: Any, *, interval_ms: int = TICK_MS) -> None:
        self._root = root
        self._interval_ms = int(interval_ms)
        self._lock = threading.Lock()
        self._queue: deque[tuple[Callable[[], Any], Token | None, bool]] = deque()
        self._ticks: list[Callable[[], Any]] = []
        self._closing = False
        self._destroyed = False
        self._started = False

    @property
    def closing(self) -> bool:
        return self._closing

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    def post(self, callback: Callable[[], Any], token: Token | None = None, *, during_close: bool = False) -> bool:
        """任意线程可调用：把回调排进队列，下次排空时在 Tk 主线程执行。已拒收时返回 False。"""
        with self._lock:
            if self._destroyed or (self._closing and not during_close):
                return False
            self._queue.append((callback, token, during_close))
            return True

    def every_tick(self, work: Callable[[], Any]) -> None:
        """登记每次排空之后执行的逐帧工作（只在 Tk 主线程调用）。"""
        self._ticks.append(work)

    def start(self) -> None:
        """开始周期性排空（只在 Tk 主线程调用，重复调用无效）。"""
        if self._started or self._destroyed:
            return
        self._started = True
        self._schedule()

    def pump(self) -> None:
        """排空一次队列并执行逐帧工作；回调抛出的异常只打印，不中断后续回调。"""
        if self._destroyed:
            return
        with self._lock:
            batch = list(self._queue)
            self._queue.clear()
        for callback, token, during_close in batch:
            if self._destroyed or (self._closing and not during_close):
                continue
            _run(lambda: None if token is not None and not token() else callback())
        for work in list(self._ticks):
            if self._destroyed:
                return
            _run(work)

    def close(self) -> None:
        """进入关闭状态：此后只执行关闭协调器以 ``during_close=True`` 投递的回调。"""
        with self._lock:
            self._closing = True
            self._queue = deque(item for item in self._queue if item[2])

    def destroy(self) -> None:
        """窗口即将销毁：丢弃队列，拒绝一切回调，停止排空。"""
        with self._lock:
            self._closing = True
            self._destroyed = True
            self._queue.clear()
        self._ticks.clear()

    def _schedule(self) -> None:
        try:
            self._root.after(self._interval_ms, self._loop)
        except (TclError, RuntimeError):  # 窗口已销毁
            pass

    def _loop(self) -> None:
        self.pump()
        if not self._destroyed:
            self._schedule()


def _run(callback: Callable[[], Any]) -> None:
    try:
        callback()
    except Exception:
        traceback.print_exc()
