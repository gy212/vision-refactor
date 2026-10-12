# -*- coding: utf-8 -*-
"""会话启动耗时：从点击"开始"到各路首帧就绪、首帧入槽与显示、模型就绪、首个标注帧入槽与显示（原 ``_DualStartupMetrics``）。

只输出一次 ``session.metrics`` 诊断事件：会话进入运行且目标帧（推理开时为首个标注帧，否则为首帧）已显示时，
或会话结束时。字段与旧的 ``[dual-startup]`` 行相同，耗时单位 ms。
"""
from __future__ import annotations

import threading
import time

from core.foundation import diag

MARKS = (
    "pair_ready",
    "first_pair_enqueued",
    "first_pair_rendered",
    "pipeline_ready",
    "first_annotated_enqueued",
    "first_annotated_rendered",
)


class StartupMetrics:
    """一次会话的启动时间点；``mark`` 只记第一次，线程安全。"""

    def __init__(
        self,
        session: int,
        *,
        primary_index: int | None,
        secondary_index: int | None,
        inference: bool,
        start_click: float,
    ) -> None:
        self.session = session
        self.primary_index = primary_index
        self.secondary_index = secondary_index
        self.inference = inference
        self.start_click = float(start_click)
        self.outcome = "starting"
        self.error_type: str | None = None
        self._marks: dict[str, float] = {}
        self._emitted = False
        self._lock = threading.Lock()

    def mark(self, name: str, when: float | None = None) -> None:
        if name not in MARKS:
            raise ValueError(f"unknown startup mark: {name}")
        with self._lock:
            self._marks.setdefault(name, time.monotonic() if when is None else float(when))

    def elapsed_ms(self, when: float | None = None) -> float:
        return ((time.monotonic() if when is None else when) - self.start_click) * 1000.0

    def set_outcome(self, outcome: str, error_type: str | None = None) -> None:
        with self._lock:
            self.outcome = outcome
            self.error_type = error_type

    def maybe_emit(self) -> bool:
        """会话已在运行且目标帧已显示时输出；返回本次是否输出。"""
        with self._lock:
            target = "first_annotated_rendered" if self.inference else "first_pair_rendered"
            if self._emitted or self.outcome == "starting" or target not in self._marks:
                return False
            return self._emit_locked()

    def finish(self, outcome: str, error_type: str | None = None) -> bool:
        """会话结束：记下终态，未输出过就输出。"""
        with self._lock:
            self.outcome = outcome
            self.error_type = error_type
            if self._emitted:
                return False
            return self._emit_locked()

    def payload(self) -> dict[str, object]:
        with self._lock:
            return self._payload_locked()

    def _payload_locked(self) -> dict[str, object]:
        fields: dict[str, object] = {
            "session_generation": self.session,
            "primary_index": self.primary_index,
            "secondary_index": self.secondary_index,
            "record_skeleton": self.inference,
            "outcome": self.outcome,
            "error_type": self.error_type,
        }
        for name in MARKS:
            value = self._marks.get(name)
            fields[f"start_to_{name}_ms"] = None if value is None else round((value - self.start_click) * 1000.0, 3)
        return fields

    def _emit_locked(self) -> bool:
        self._emitted = True
        diag.event("session.metrics", **self._payload_locked())
        return True


__all__ = ["MARKS", "StartupMetrics"]
