# -*- coding: utf-8 -*-
"""``LiveSessionManager``：实时会话的启动、停止与状态（idle/starting/running/stopping），持有会话 ``Epoch``（设计文档 §8.3）。

- 同一时刻最多一个会话；``start`` 只在 idle 时接受，返回会话 token。
- runner 发出的事件经这里过滤：token 不是当前会话、或会话已结束后的迟到事件一律丢弃，再交给消费方。
- ``ended`` 先把状态置回 idle 再转发，消费方在回调里就能开始下一个会话。
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from core.foundation.epoch import Epoch
from core.foundation.tasks import CancelToken, TaskGroup, TaskHandle
from core.live.config import LiveControls, PresenceProbe, SessionConfig
from core.live.inputs import FrameInput
from core.live.metrics import StartupMetrics
from core.live.preview import PreviewStore, TargetSize
from core.live.runner import FrameSink, SessionEvent, SessionRunner
from core.live.stages import build_stages
from core.media import PRIMARY, SECONDARY

IDLE, STARTING, RUNNING, STOPPING = "idle", "starting", "running", "stopping"
SESSION_CLOSE_BUDGET_S = 3.0

InputFactory = Callable[[SessionConfig], FrameInput]


class LiveSessionManager:
    """组装并运行会话；输入、录制端口、站位判定与预览目标尺寸由入口层注入。"""

    def __init__(
        self,
        *,
        make_input: InputFactory,
        on_event: Callable[[SessionEvent], None],
        controls: LiveControls | None = None,
        sink: FrameSink | None = None,
        probe: PresenceProbe | None = None,
        preview_target: TargetSize | None = None,
    ) -> None:
        self.controls = controls or LiveControls()
        self._make_input = make_input
        self._on_event = on_event
        self._sink = sink
        self._probe = probe
        self._preview_target = preview_target
        self._epoch = Epoch()
        self._lock = threading.Lock()
        self._state = IDLE
        self._cancel: CancelToken | None = None
        self._preview: PreviewStore | None = None
        self._config: SessionConfig | None = None
        self._tasks = TaskGroup("live")
        self._handle: TaskHandle | None = None

    # ---- 查询 ----

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def token(self) -> int:
        return self._epoch.current

    @property
    def config(self) -> SessionConfig | None:
        with self._lock:
            return self._config

    @property
    def preview(self) -> PreviewStore | None:
        """当前会话的预览槽；没有会话时为 None。"""
        with self._lock:
            return self._preview

    def is_current(self, token: int) -> bool:
        return self._epoch.is_current(token)

    def active(self) -> bool:
        return self.state != IDLE

    # ---- 控制 ----

    def start(self, config: SessionConfig, *, start_click: float | None = None) -> int | None:
        """开始一个会话；已有会话（含仍在收尾的）时返回 None。不阻塞。"""
        with self._lock:
            if self._state != IDLE:
                return None
            token = self._epoch.advance()
            self._state = STARTING
            self._config = config
            self._cancel = CancelToken()
            cancel = self._cancel
            metrics = None
            if config.cameras is not None:
                metrics = StartupMetrics(
                    token,
                    primary_index=config.cameras[PRIMARY].index,
                    secondary_index=config.cameras[SECONDARY].index if config.dual else None,
                    inference=config.inference,
                    start_click=time.monotonic() if start_click is None else start_click,
                )
            self._preview = (
                PreviewStore(token, self._preview_target, metrics=metrics) if self._preview_target else None
            )
            preview = self._preview
        try:
            source = self._make_input(config)
        except BaseException:
            with self._lock:
                self._state = IDLE
                self._preview = None
            raise
        pose, stages = build_stages(config, probe=self._probe)
        runner = SessionRunner(
            config, source, pose, stages, self._sink, preview,
            lambda event: self._dispatch(event), token, cancel,
            controls=self.controls, metrics=metrics,
        )
        self._handle = self._tasks.spawn(
            f"session-{token}", lambda _t: runner.run(), owner="session", close_budget_s=SESSION_CLOSE_BUDGET_S,
        )
        return token

    def stop(self, reason: str = "user_stop") -> bool:
        """请求停止当前会话（不等待）；没有会话时返回 False。"""
        with self._lock:
            if self._state in (IDLE, STOPPING) or self._cancel is None:
                return False
            self._state = STOPPING
            cancel = self._cancel
        cancel.cancel(reason)
        return True

    def wait_idle(self, timeout: float) -> bool:
        """等当前会话线程结束（关窗、测试用）；返回是否已结束。"""
        handle = self._handle
        return handle is None or handle.join(timeout)

    # ---- 事件 ----

    def _dispatch(self, event: SessionEvent) -> None:
        with self._lock:
            if not self._epoch.is_current(event.token) or self._state == IDLE:
                return
            if event.kind == "running" and self._state == STARTING:
                self._state = RUNNING
            elif event.kind == "ended":
                self._state = IDLE
                self._cancel = None
                self._preview = None
        self._on_event(event)


__all__ = ["IDLE", "LiveSessionManager", "RUNNING", "STARTING", "STOPPING"]
