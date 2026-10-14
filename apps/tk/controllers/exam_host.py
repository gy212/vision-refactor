# -*- coding: utf-8 -*-
"""考试窗口的宿主（``ExamHost``）：向考试控制器提供会话、录制与模式能力，并负责考试窗口的打开、复用与关闭登记。

考试窗口不访问主窗口的私有成员；开考进入考试模式，主会话停止、换名单或关闭窗口时退出。
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox
from typing import Any

from core.live import LiveControls
from core.media import PRIMARY
from core.workflows import EXAM, ModeManager, ScorebookRegistry, WorkflowHost, WorkflowPorts

from apps.tk.controllers.flags import LiveFlags
from apps.tk.controllers.session import OccupancyPort


class ExamHostAdapter:
    def __init__(self, root: Any, *, workflows: WorkflowHost, scorebooks: ScorebookRegistry, modes: ModeManager,
                 controls: LiveControls, flags: LiveFlags, post: Callable[..., Any],
                 session_active: Callable[[], bool], open_window: Callable[[Any, Any], Any]) -> None:
        self.root = root
        self.workflows = workflows
        self.exam_scorebooks = scorebooks
        self.modes = modes
        self.controls = controls
        self.flags = flags
        self._post = post
        self._session_active = session_active
        self._open_window = open_window
        self.recording: Any = None   # RecordingController，由组合根接上
        self.exam_panel: Any = None  # 打开中的 ExamController

    def open(self) -> None:
        if self.modes.is_active("practice"):
            messagebox.showinfo("学生练习", "请先退出学生练习模式，再打开考试模式。")
            return
        if self.exam_panel is not None:
            try:
                self.exam_panel.view.lift()
                return
            except Exception:
                pass
        self.exam_panel = self._open_window(self.root, self).controller

    def prepare_for_app_close(self) -> None:
        """关窗第 1 步：中止考试并移交面板台账；没有面板但仍在考试模式时丢弃进行中的片段。

        ``root.destroy`` 不会触发面板的 WM_DELETE_WINDOW，所以必须显式收尾。
        """
        if self.exam_panel is not None:
            try:
                self.exam_panel.prepare_for_app_close()
            except Exception:
                pass
        elif self.modes.is_active("exam"):
            try:
                self.recording.end(discard=True)
            except Exception:
                pass

    # ------------------------------------------------------------------ ExamHost
    def post(self, callback: Callable[[], Any]) -> bool:
        """后台线程回到 Tk 线程（考试播报文字等）。"""
        return self._post(callback)

    def exam_primary_rotation(self) -> int:
        """考试 ROI 按主路旋转后的画面判定。"""
        return int(self.controls.rotation_of(PRIMARY))

    def exam_record_dir(self) -> Path:
        return self.recording.record_dir()

    def exam_closed(self, controller: Any) -> None:
        if self.exam_panel is controller:
            self.exam_panel = None

    def exam_ports(self, *, ui: Any, announcer: Any) -> WorkflowPorts:
        """录制与站位检测接到主窗口的会话，界面与播报由考试窗口提供。"""
        return WorkflowPorts(recorder=self.recording, occupancy=OccupancyPort(self.controls, "exam"),
                             announcer=announcer, ui=ui)

    def exam_preflight(self) -> list[str]:
        """开考预检（考试模式声明的条件）；返回不满足的条件，空表示可以开考。"""
        return EXAM.requirements.preflight(self.recording.preflight_context())

    def set_exam_mode(self, active: bool) -> None:
        if active:
            self.modes.enter(EXAM)
            self.flags.auto_compare = True
            return
        self.modes.leave("exam")
        self.controls.arm_occupancy("exam", False)

    def lock_manual_record(self, locked: bool) -> None:
        """考试进行中锁定手动录制按钮与骨架、自动比对开关（考试强制比对、骨架关）。"""
        self.recording.lock_manual(locked, session_active=self._session_active())
