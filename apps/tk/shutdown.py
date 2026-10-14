# -*- coding: utf-8 -*-
"""关窗的唯一流程 ``ShutdownCoordinator``：按设计文档 §9.1.3 的顺序收尾，有界等待后台工作后销毁窗口。

各步骤由组合根以 ``ShutdownSteps`` 注入；协调器只负责顺序、截止时刻和 Tk 侧的分片轮询，自身不含业务。
设备释放等可能阻塞的步骤在后台任务里执行，Tk 主线程每 ``POLL_MS`` 毫秒查看一次，不卡住界面。
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import TclError
from typing import Any

from core.foundation import TaskGroup, TaskHandle

from apps.tk.dispatcher import UiDispatcher

JOIN_TIMEOUT_S = 3.0    # 基础等待预算；录后处理与台账回填的额外预算由 ``extra_budget_s`` 给出
JOIN_SLICE_S = 0.05     # 每轮在 Tk 线程上最多阻塞的时长
POLL_MS = 50


@dataclass(frozen=True)
class ShutdownSteps:
    """关窗各步骤。除 ``release`` 在后台任务里运行外，都在 Tk 主线程调用。"""

    prepare_workflows: Callable[[], None]       # 1：考试中止并移交台账，练习取消等待与分析
    stop_session: Callable[[], None]            # 2：进入关闭态、取消会话、禁用录制控件
    extra_budget_s: Callable[[], float]         # 录后处理与台账回填在基础预算之外还要等的秒数
    release: Callable[[], None]                 # 3–5：收尾录制片段、finalizer cancel_all、释放摄像头
    wait_session_idle: Callable[[float], bool]  # 等会话线程结束，返回是否已结束
    finalizer_idle: Callable[[], bool]          # 录后处理是否空闲
    flush: Callable[[], None]                   # 6：最后一次应用后台更新，台账落盘关闭


class ShutdownCoordinator:
    """``request()`` 只生效一次；之后由 ``poll()`` 自行调度，直到 ``root.destroy()``。"""

    def __init__(self, root: Any, dispatcher: UiDispatcher, steps: ShutdownSteps) -> None:
        self._root = root
        self._dispatcher = dispatcher
        self._steps = steps
        self._tasks = TaskGroup("shutdown")
        self._release: TaskHandle | None = None
        self.deadline: float | None = None

    @property
    def started(self) -> bool:
        return self.deadline is not None

    @property
    def releasing(self) -> bool:
        """后台释放任务是否仍在运行。"""
        return self._release is not None and not self._release.done

    def join_release(self, timeout: float) -> bool:
        """等待后台释放任务结束；返回是否已结束。"""
        return self._release is None or self._release.join(timeout)

    def request(self) -> None:
        if self.started:
            return
        _quietly(self._steps.prepare_workflows)
        self._dispatcher.close()
        self._steps.stop_session()
        self.deadline = time.monotonic() + JOIN_TIMEOUT_S + float(self._steps.extra_budget_s())
        self._release = self._tasks.spawn(
            "release", lambda _token: self._steps.release(), owner="app-close", close_budget_s=JOIN_TIMEOUT_S,
        )
        self.poll()

    def poll(self) -> None:
        """分片等待释放任务、会话线程与录后处理；全部结束或到达截止时刻后销毁窗口。"""
        deadline = self.deadline if self.deadline is not None else time.monotonic()
        remaining = deadline - time.monotonic()
        if self.releasing and remaining > 0.0:
            self.join_release(min(JOIN_SLICE_S, remaining))
        remaining = deadline - time.monotonic()
        if remaining > 0.0:
            self._steps.wait_session_idle(min(JOIN_SLICE_S, remaining))
        # 会话线程收尾时会提交最后的片段；决定销毁前重新查看 finalizer 是否空闲。
        busy = self.releasing or not self._steps.wait_session_idle(0) or not self._steps.finalizer_idle()
        if busy and time.monotonic() < deadline:
            try:
                self._root.after(POLL_MS, self.poll)
            except (TclError, RuntimeError):
                pass
            return
        self._steps.flush()
        self._dispatcher.destroy()
        try:
            self._root.destroy()
        except (TclError, RuntimeError):
            pass


def _quietly(step: Callable[[], None]) -> None:
    try:
        step()
    except Exception:
        pass
