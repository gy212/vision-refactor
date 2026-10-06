# -*- coding: utf-8 -*-
"""后台任务与取消的唯一原语：``Cancelled``、``CancelToken``、``TaskGroup``，以及迁移期的旧取消信号适配器。

- ``Cancelled`` 是唯一的取消异常；``CancelToken`` 可级联（``child()``），父 token 取消时子 token 一并取消。
- ``TaskGroup`` 记录每个任务的名称、所有者和关窗预算 ``close_budget_s``；关窗时问它还有哪些任务在跑，
  按 ``close_deadline()`` 等待，不再逐类手写 join。
- 适配器 ``as_stopped`` / ``translate_interrupted`` 把旧接口的 ``stopped()`` 回调与 ``InterruptedError``
  映射到新原语，只在迁移期使用（设计文档 §4.3），旧接口迁完后删除。
"""
from __future__ import annotations

import threading
import time
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


class Cancelled(Exception):
    """任务被取消。``reason`` 取 §4.5 的取消原因，如 ``user_stop``、``app_close``、``superseded``。"""

    def __init__(self, reason: str = "") -> None:
        super().__init__(reason or "cancelled")
        self.reason = reason


class CancelToken:
    """线程安全的取消标记。取消不可撤销，第一次的原因生效。

    父 token 只弱引用子 token，子 token 强引用父 token：长寿的父 token（如 ``TaskGroup`` 的组 token）
    不会攒下已结束任务的子 token，而仍被持有的孙 token 能经由中间层收到级联取消。
    """

    def __init__(self, parent: "CancelToken | None" = None) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = ""
        self._parent = parent
        self._children: weakref.WeakSet[CancelToken] = weakref.WeakSet()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def cancel(self, reason: str = "") -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._reason = reason
            self._event.set()
            children = list(self._children)
            self._children = weakref.WeakSet()
        for child in children:
            child.cancel(reason)

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise Cancelled(self._reason)

    def wait(self, timeout: float | None = None) -> bool:
        """阻塞到被取消或超时；返回是否已取消。可代替 ``time.sleep`` 做可中断的等待。"""
        return self._event.wait(timeout)

    def child(self) -> "CancelToken":
        """派生子 token：父取消时级联取消，子取消不影响父。父已取消时子立即取消。"""
        token = CancelToken(parent=self)
        with self._lock:
            if not self._event.is_set():
                self._children.add(token)
                return token
            reason = self._reason
        token.cancel(reason)
        return token


class TaskHandle:
    """``TaskGroup.spawn`` 返回的句柄。任务函数抛出的非取消异常记在 ``error`` 并照常交给线程的 excepthook。"""

    def __init__(self, name: str, owner: str, close_budget_s: float, token: CancelToken) -> None:
        self.name = name
        self.owner = owner
        self.close_budget_s = close_budget_s
        self.token = token
        self.result: Any = None
        self.error: BaseException | None = None
        self.was_cancelled = False
        self._finished = threading.Event()

    @property
    def done(self) -> bool:
        return self._finished.is_set()

    def join(self, timeout: float | None = None) -> bool:
        """等待任务结束；返回是否已结束。"""
        return self._finished.wait(timeout)

    def __repr__(self) -> str:
        return f"TaskHandle({self.owner}:{self.name}, done={self.done})"


class TaskGroup:
    """一组后台线程任务。每个任务拿到组 token 的子 token；``cancel`` 可按所有者筛选。"""

    def __init__(self, name: str = "tasks") -> None:
        self.name = name
        self._token = CancelToken()
        self._lock = threading.Lock()
        self._handles: list[TaskHandle] = []

    def spawn(
        self,
        name: str,
        fn: Callable[[CancelToken], Any],
        *,
        owner: str,
        close_budget_s: float = 2.0,
    ) -> TaskHandle:
        """启动守护线程运行 ``fn(token)``。``close_budget_s`` 是关窗时愿意为它等待的秒数。"""
        handle = TaskHandle(name, owner, float(close_budget_s), self._token.child())

        def run() -> None:
            try:
                handle.result = fn(handle.token)
            except Cancelled:
                handle.was_cancelled = True
            except BaseException as exc:
                handle.error = exc
                raise
            finally:
                handle._finished.set()

        thread = threading.Thread(target=run, name=f"{self.name}/{owner}/{name}", daemon=True)
        with self._lock:
            self._handles = [h for h in self._handles if not h.done]
            self._handles.append(handle)
        thread.start()
        return handle

    def cancel(self, owner: str | None = None, reason: str = "") -> None:
        """取消某个所有者的全部任务；``owner=None`` 取消整组（之后 spawn 的任务也立即处于取消状态）。"""
        if owner is None:
            self._token.cancel(reason)
            return
        for handle in self.pending(owner):
            handle.token.cancel(reason)

    def pending(self, owner: str | None = None) -> list[TaskHandle]:
        """仍在运行的任务，按启动顺序。"""
        with self._lock:
            self._handles = [h for h in self._handles if not h.done]
            alive = list(self._handles)
        return [h for h in alive if owner is None or h.owner == owner]

    def close_deadline(self) -> float:
        """``time.monotonic()`` 时间轴上的关窗截止时刻：现在加上存活任务里最大的 ``close_budget_s``。"""
        budgets = [h.close_budget_s for h in self.pending()]
        return time.monotonic() + (max(budgets) if budgets else 0.0)

    def join_until(self, deadline: float) -> list[TaskHandle]:
        """等到全部任务结束或到达截止时刻；返回仍未结束的任务。"""
        for handle in self.pending():
            handle.join(max(0.0, deadline - time.monotonic()))
        return self.pending()


# --------------------------------------------------------------------------- #
# 迁移期适配器：旧接口迁到 CancelToken/Cancelled 后删除
# --------------------------------------------------------------------------- #


def as_stopped(token: CancelToken) -> Callable[[], bool]:
    """把 token 适配成旧接口的 ``stopped()`` / ``should_stop()`` 回调。"""
    return lambda: token.cancelled


class _PolledToken(CancelToken):
    _POLL_INTERVAL_S = 0.05

    def __init__(self, stopped: Callable[[], bool]) -> None:
        super().__init__()
        self._stopped = stopped

    @property
    def cancelled(self) -> bool:
        if not self._event.is_set() and self._stopped():
            self.cancel("user_stop")
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise Cancelled(self._reason)

    def wait(self, timeout: float | None = None) -> bool:
        """按小间隔轮询回调直到取消或超时；父类只等内部 Event，看不到回调置位（问题 4）。"""
        if self.cancelled:
            return True
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = self._POLL_INTERVAL_S if deadline is None else min(self._POLL_INTERVAL_S, deadline - time.monotonic())
            if remaining <= 0:
                return self.cancelled
            if self._event.wait(remaining):
                return True
            if self.cancelled:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False

    def child(self) -> "CancelToken":
        # 子 token 也要轮询父回调，否则父的取消只有在被查询时才会级联。
        return _PolledToken(lambda: self.cancelled)


def from_stopped(stopped: Callable[[], bool] | None) -> CancelToken | None:
    """把旧接口的 ``stopped()`` 回调（或 ``Event.is_set``）适配成 token：每次查询时轮询回调。"""
    return None if stopped is None else _PolledToken(stopped)


@contextmanager
def translate_interrupted(token: CancelToken | None = None, reason: str = "") -> Iterator[None]:
    """把块内旧代码抛出的 ``InterruptedError`` 转成 ``Cancelled``；原因优先取 token 的取消原因。"""
    try:
        yield
    except InterruptedError as exc:
        actual = token.reason if token is not None and token.cancelled else reason
        raise Cancelled(actual) from exc
