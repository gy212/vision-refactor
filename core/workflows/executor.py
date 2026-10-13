# -*- coding: utf-8 -*-
"""命令执行：把工作流产出的 ``Command`` 映射到 ``WorkflowPorts``，并把执行结果作为事件回送给工作流。

考试与学生练习共用（设计文档 §8.6）。开录、结束片段的结果以 ``record_started`` / ``begin_failed`` /
``record_stopped`` 事件回送，工作流据此推进；单条命令失败只在界面提示，不影响后续命令。
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from core.workflows.commands import Command


class Workflow(Protocol):
    """可由 ``CommandExecutor`` 驱动的流程状态机。"""

    def handle(self, event: str, **kw: Any) -> list[Command]: ...

    def row(self, row_id: str) -> Any: ...


class SegmentControl(Protocol):
    def start_segment(self, payload: dict[str, Any]) -> Any:
        """开录；成功返回带 ``segment_id``、``directory`` 的片段引用，失败返回 None。"""

    def finish(self, *, discard: bool) -> None: ...


class OccupancyControl(Protocol):
    def arm(self, armed: bool) -> None: ...

    def set_roi(self, roi: tuple[float, float, float, float]) -> None: ...


class Announcer(Protocol):
    def announce(self, text: str) -> None: ...


class WorkflowUi(Protocol):
    def lock_manual_record(self, locked: bool) -> None: ...

    def status(self, text: str) -> None: ...

    def alert(self, title: str, text: str, *, error: bool = False) -> None: ...

    def show_result(self, record: dict[str, Any]) -> None: ...

    def refresh_rows(self) -> None: ...

    def refresh(self) -> None:
        """每次分派事件后调用：同步阶段、按钮等显示。"""


class Scorebook(Protocol):
    path: Any
    write_error: str | None

    def update_row(self, row: Any, **fields: Any) -> None: ...

    def bind_segment(self, row: Any, *, segment_id: str, segment_dir: str | None) -> None: ...

    def flush(self) -> None: ...


class AnalysisControl(Protocol):
    def start_practice_analysis(self, request: dict[str, Any]) -> bool:
        """启动后台分析；同步失败（配置错误、正忙）时返回 False。"""
        ...


@dataclass
class WorkflowPorts:
    """工作流对外的全部能力。``scorebook`` 随考试场次替换，练习没有台账，``analysis`` 只有练习用。"""

    recorder: SegmentControl
    occupancy: OccupancyControl
    announcer: Announcer
    ui: WorkflowUi
    scorebook: Scorebook | None = None
    analysis: AnalysisControl | None = None


FollowUp = tuple[str, dict[str, Any]]


class CommandExecutor:
    def __init__(self, ports: WorkflowPorts, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.ports = ports
        self._clock = clock

    def dispatch(self, workflow: Workflow, event: str, **kw: Any) -> None:
        """把事件交给工作流，执行它产出的命令，最后刷新界面。"""
        self.run(workflow, workflow.handle(event, **kw))
        self.ports.ui.refresh()

    def run(self, workflow: Workflow, commands: list[Command]) -> None:
        for command in commands:
            try:
                follow_up = self._execute(workflow, command)
            except Exception as exc:
                self.ports.ui.status(f"命令失败 {command.kind}: {exc}")
                traceback.print_exc()
                continue
            if follow_up is not None:
                event, kw = follow_up
                self.run(workflow, workflow.handle(event, **kw))

    def _execute(self, workflow: Workflow, command: Command) -> FollowUp | None:
        kind, payload = command.kind, command.payload
        ports = self.ports
        if kind == "ANNOUNCE":
            ports.announcer.announce(str(payload.get("text") or ""))
        elif kind in {"ARM_OCCUPANCY", "DISARM_OCCUPANCY"}:
            ports.occupancy.arm(kind == "ARM_OCCUPANCY")
        elif kind == "BEGIN_SEGMENT":
            segment = ports.recorder.start_segment(payload)
            if not segment:
                return "begin_failed", {"now": self._clock()}
            return "record_started", {"now": self._clock(), "segment": segment}
        elif kind in {"END_SEGMENT", "DISCARD_SEGMENT"}:
            discard = kind == "DISCARD_SEGMENT"
            ports.recorder.finish(discard=discard)
            return "record_stopped", {"now": self._clock(), "discarded": discard}
        elif kind in {"UPDATE_ROW", "BIND_SEGMENT_PENDING"}:
            self._update_row(workflow, kind, payload)
        elif kind == "EXPORT":
            self._export()
        elif kind in {"UI_LOCK_MANUAL_RECORD", "UI_UNLOCK_MANUAL_RECORD"}:
            ports.ui.lock_manual_record(kind == "UI_LOCK_MANUAL_RECORD")
        elif kind == "STATUS":
            if payload.get("text"):
                ports.ui.status(str(payload["text"]))
            if payload.get("alert"):
                ports.ui.alert(str(payload.get("title") or ""), str(payload["alert"]), error=bool(payload.get("error")))
        elif kind == "START_ANALYSIS":
            started = False
            try:
                if ports.analysis is not None:
                    started = bool(ports.analysis.start_practice_analysis(dict(payload)))
            except Exception:
                traceback.print_exc()
            if not started:
                return "analysis_start_failed", {"token": payload.get("token")}
        elif kind == "SHOW_RESULT":
            ports.ui.show_result(payload["record"])
        # SCHEDULE_CALL_GUARD / SCHEDULE_INTER_GAP / ADVANCE_POINTER：只是记录，定时由 tick 驱动
        return None

    def _update_row(self, workflow: Workflow, kind: str, payload: dict[str, Any]) -> None:
        row = workflow.row(str(payload.get("row_id") or ""))
        book = self.ports.scorebook
        if row is not None and book is not None:
            if kind == "BIND_SEGMENT_PENDING":
                book.bind_segment(row, segment_id=payload["segment_id"], segment_dir=payload.get("segment_dir"))
            else:
                book.update_row(row, **{k: v for k, v in payload.items() if k != "row_id"})
        self.ports.ui.refresh_rows()

    def _export(self) -> None:
        book = self.ports.scorebook
        if book is None:
            return
        book.flush()
        if book.write_error:
            self.ports.ui.status(book.write_error)
        else:
            self.ports.ui.status(f"成绩已写入：{book.path}" if book.path else "成绩已更新")
