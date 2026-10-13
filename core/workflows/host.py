# -*- coding: utf-8 -*-
"""工作流事件路由：把站位样本、会话就绪/停止/结束、片段更新与定时 tick 交给当前激活的工作流。

``WorkflowHost`` 是纯 Python（不依赖 Tk），由组合根接到 L4 的事件源上（设计文档 §8.6）。同一时刻至多
一个工作流激活（考试或学生练习）；会话事件带会话 token，不属于当前会话的过期事件直接丢弃。考试片段的
录后结果先按 ``exam_run_id`` 交给 ``ScorebookRegistry`` 回填（面板关闭后仍能回填），再交给激活的工作流。
"""

from __future__ import annotations

from typing import Any

from core.workflows.exam.scorebook import ScorebookRegistry
from core.workflows.executor import CommandExecutor, Workflow


class WorkflowHost:
    def __init__(self, scorebooks: ScorebookRegistry) -> None:
        self.scorebooks = scorebooks
        self._workflow: Workflow | None = None
        self._executor: CommandExecutor | None = None
        self._token: Any = None

    # ------------------------------------------------------------------ 激活
    @property
    def active(self) -> Workflow | None:
        return self._workflow

    def activate(self, workflow: Workflow, executor: CommandExecutor) -> None:
        self._workflow = workflow
        self._executor = executor

    def deactivate(self, workflow: Workflow) -> None:
        """只停用仍处于激活的那个工作流；已被替换的旧工作流不影响新的。"""
        if self._workflow is workflow:
            self._workflow = None
            self._executor = None

    def dispatch(self, event: str, **kw: Any) -> bool:
        """把事件交给激活的工作流；没有激活的工作流时返回 False。"""
        workflow, executor = self._workflow, self._executor
        if workflow is None or executor is None:
            return False
        executor.dispatch(workflow, event, **kw)
        return True

    # ------------------------------------------------------------------ 会话事件
    def session_started(self, token: Any) -> None:
        """登记当前会话；之后只接受这个 token 的会话事件。"""
        self._token = token

    def _current(self, token: Any) -> bool:
        return token is not None and token == self._token

    def presence(self, token: Any, present: bool, now: float) -> None:
        if self._current(token):
            self.dispatch("presence", present=bool(present), now=float(now))

    def session_ready(self, token: Any, now: float) -> None:
        if self._current(token):
            self.dispatch("session_ready", now=float(now))

    def session_stopped(self, now: float) -> None:
        """用户停止会话（会话线程还在收尾）。"""
        self.dispatch("session_stopped", now=float(now))

    def session_ended(self, token: Any, now: float, status: str = "") -> None:
        if self._current(token):
            self._token = None
            self.dispatch("session_ended", now=float(now), status=status)

    def tick(self, now: float) -> None:
        self.dispatch("tick", now=float(now))

    # ------------------------------------------------------------------ 片段更新
    def segment_update(self, update: Any, *, route: bool = True) -> None:
        """考试结果按 run_id 回填台账；``route`` 时再交给激活的工作流（关窗收尾时只回填）。"""
        applied = self.scorebooks.apply(update)
        executor = self._executor
        if applied is not None and executor is not None and executor.ports.scorebook is applied[0]:
            executor.ports.ui.refresh_rows()
        if route:
            self.dispatch("segment_update", update=update)
