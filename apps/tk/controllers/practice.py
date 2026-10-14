# -*- coding: utf-8 -*-
"""学生练习模式的界面编排：进入/退出练习、开始/结束/动作评判、分析结果页的显示与导出、教师在结果页的复核。

练习流程本身在 ``core.workflows.PracticeSession``，经 ``WorkflowHost`` 驱动；本控制器同时是练习流程的界面端口
（``WorkflowUi``）与分析端口（``AnalysisControl``）。晚到或作废的分析结果由练习流程按任务代次与学号拦截。
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

from core.evaluation import ACTIONS
from core.foundation import EXAM_ROI, PreferenceStore
from core.live import LiveControls
from core.workflows import PRACTICE, Announcer, CommandExecutor, ModeManager, PracticeSession, WorkflowHost, WorkflowPorts

from apps.tk.controllers.flags import LiveFlags
from apps.tk.controllers.session import OccupancyPort
from apps.tk.views.practice_block import READY_TEXT

INTRO_TEXT = (
    "动作问题说明由MediaPipe关键点与几何规则计算，候选问题待教师确认，不打分。\n"
    "流程：填学号、选择动作与左/右式 → 开始 → 进入黄框站稳，播报开始并录制 → 离开2秒，播报结束 → 动作评判。\n"
    "黄框沿用考试模式保存的站位区域；可在考试模式中调整。也可手动点「结束」。\n"
    "也可导入仍保留的旧视频；个人历史支持文字导出和教师确认/撤销。"
)
PRACTICE_STATUS = "学生练习模式"


class PracticeController:
    """``window`` 是主窗口（练习块、结果页、状态区）；``session``、``cameras``、``recording`` 是其余控制器。"""

    def __init__(self, window: Any, *, session: Any, cameras: Any, recording: Any, modes: ModeManager,
                 workflows: WorkflowHost, controls: LiveControls, flags: LiveFlags,
                 exam: Callable[[], Any], announcer_factory: Callable[[], Any] = Announcer) -> None:
        self.window = window
        self.block = window.practice
        self.session = session
        self.cameras = cameras
        self.recording = recording
        self.modes = modes
        self.workflows = workflows
        self.controls = controls
        self.flags = flags
        self._exam = exam
        self._announcer_factory = announcer_factory
        self.practice: PracticeSession | None = None
        self.announcer: Any = None

    @property
    def analysis_controls(self) -> Any:
        return self.block.controls

    def _main_status(self, text: str) -> None:
        self.window.status.status_var.set(text)

    # ------------------------------------------------------------------ 进入与退出
    def enter(self) -> None:
        """切换到练习界面；骨架、手部、自动比对、在线识别由模式要求在会话配置里锁定为关，不改写用户勾选（V-5）。"""
        if self.modes.is_active("practice"):
            return
        if self.modes.is_active("exam") or self.flags.exam_locked:
            messagebox.showinfo("学生练习", "考试进行中，无法进入学生练习。")
            return
        if self.session.active():
            messagebox.showinfo("学生练习", "请先停止当前识别会话，再进入学生练习。")
            return
        self.modes.enter(PRACTICE)
        self.practice = PracticeSession(self.analysis_controls.identity)
        self.announcer = self._announcer_factory()
        ports = WorkflowPorts(recorder=self.recording, occupancy=OccupancyPort(self.controls, "practice"),
                              announcer=self.announcer, ui=self, analysis=self)
        self.workflows.activate(self.practice, CommandExecutor(ports))
        self.window.result_panel.clear()
        self.show_preview_view()
        self.window.show_practice(True)
        self.block.set_result_text(INTRO_TEXT)
        self.block.status_var.set(READY_TEXT)
        self.refresh()
        self._main_status(PRACTICE_STATUS)
        self.cameras.sync_warmup()

    def exit(self) -> None:
        """退出练习并恢复教师控件；考试窗口仍开着时把站位与 tick 交还考试流程。"""
        practice = self.practice
        if practice is None:
            return
        if practice.busy:
            messagebox.showinfo("学生练习", "动作评判进行中，请稍候再退出。")
            return
        self.workflows.dispatch("exit")   # 若仍在录制则先结束片段，再停会话
        self._close_announcer()
        if self.session.active():
            self.session.stop()
        self.workflows.deactivate(practice)
        self.practice = None
        self.modes.leave("practice")
        exam = self._exam()
        if exam is not None:
            self.workflows.activate(exam.session, exam.executor)
        self.window.result_panel.clear()
        self.show_preview_view()
        self.window.show_practice(False)
        if self.session.active():
            # 会话尚在收尾：保持开始禁用，避免点击被静默丢弃；会话结束事件负责恢复
            self._main_status("正在停止…")
            for button in (self.window.capture.start_btn, self.window.capture.stop_btn):
                button.configure(state="disabled")
            return
        self.session.set_running_controls(False)
        self.cameras.sync_warmup()

    def _close_announcer(self) -> None:
        if self.announcer is not None:
            self.announcer.close()
            self.announcer = None

    def close(self) -> None:
        """关窗第 1 步：取消分析任务与等待到位，关闭播报。"""
        self.analysis_controls.close()
        if self.practice is not None:
            self.dispatch("analysis_cancelled")
            self.dispatch("session_stopped", now=time.monotonic())
        self._close_announcer()

    # ------------------------------------------------------------------ 流程事件
    def dispatch(self, event: str, **kw: Any) -> None:
        """把界面事件交给练习流程（只在练习流程处于激活时）。"""
        if self.practice is not None and self.workflows.active is self.practice:
            self.workflows.dispatch(event, **kw)

    def can_analyze(self) -> bool:
        practice = self.practice
        return (practice is not None and not (practice.busy or practice.waiting)
                and self.recording.state not in {"recording", "paused"})

    def start(self) -> None:
        """「开始」：预检通过后开启双摄（未开时），进入站位区域站稳后由练习流程开录。"""
        practice = self.practice
        if practice is None or practice.busy or practice.waiting or practice.recording:
            return
        try:
            self.analysis_controls.identity()
            identity_error = None
        except ValueError as exc:
            identity_error = str(exc)
        problems = PRACTICE.requirements.preflight(self.recording.preflight_context(
            practice=True, identity_error=identity_error, cameras_error=self.cameras.dual_selection_problem()))
        if problems:
            messagebox.showwarning("学生练习", problems[0])
            return
        self.controls.set_roi(PreferenceStore().get(EXAM_ROI))
        if self.session.active():
            self.dispatch("start", now=time.monotonic(), session_ready=bool(self.flags.recording_ready))
            return
        # 会话冻结 1080p 请求；池发现预热格式不一致时会在旧进程释放后重开
        self.dispatch("start", now=time.monotonic(), session_ready=False, launching=True)
        try:
            self.session.start()
        except Exception as exc:
            self.dispatch("session_failed")
            messagebox.showerror("学生练习", f"启动失败：{exc}")
            return
        if not self.session.active():
            self.dispatch("session_failed", message="双摄未启动，请检查摄像头后重试")
        elif self.flags.recording_ready:
            self.dispatch("session_ready", now=time.monotonic())

    def end(self) -> None:
        """「结束」：停录落盘（或取消等待到位），不自动评判。"""
        self.dispatch("end")

    def judge(self) -> None:
        """「动作评判」：按开录时绑定的学生与动作后台分析。"""
        self.dispatch("judge")

    def analysis_busy(self, busy: bool) -> None:
        self.dispatch("analysis_busy", busy=bool(busy))

    def identity_changed(self) -> None:
        """学号改了：作废当前任务代次，清空旧记录。"""
        self.dispatch("identity_changed")

    def cancelled(self) -> None:
        self.dispatch("analysis_cancelled")

    def practice_again(self) -> None:
        """结果页「再练一次」：切回预览并重置状态，但不自动开始录制。"""
        if self.practice is None or self.practice.busy:
            return
        self.show_preview_view()
        self.dispatch("practice_again")

    def analysis_saved(self, token: int, student_id: str, record: dict) -> None:
        """分析保存后的 Tk 线程回调：交给练习流程按任务代次与当前学号拦截晚到或作废的结果。"""
        if self.flags.closing:
            return
        self.dispatch("analysis_saved", token=token, student_id=student_id, record=record,
                      current_student_id=self.analysis_controls.student_id.get())

    # ------------------------------------------------------------------ WorkflowUi / AnalysisControl
    def status(self, text: str) -> None:
        self.block.status_var.set(text)

    def alert(self, title: str, text: str, *, error: bool = False) -> None:
        (messagebox.showerror if error else messagebox.showinfo)(title, text)

    def lock_manual_record(self, locked: bool) -> None:
        pass

    def show_result(self, record: dict) -> None:
        self.show_result_view(record)

    def refresh_rows(self) -> None:
        pass

    def refresh(self) -> None:
        """按练习状态联动按钮：评判只在后处理终态就绪后允许（避免与转码抢文件）。"""
        practice = self.practice
        if practice is None:
            return
        judging, active = practice.busy, practice.recording or practice.waiting
        self.analysis_controls.set_recording(active)
        self.block.set_buttons(start=not (judging or active), end=active and not judging,
                               judge=practice.can_judge, exit=not judging)

    def start_practice_analysis(self, request: dict) -> bool:
        token, student_id = request["token"], request["student_id"]

        def on_saved(record: dict) -> None:
            self.dispatch("record_saved", record_id=record["id"])
            self.analysis_saved(token=token, student_id=student_id, record=record)

        return self.analysis_controls.analyze_recording(
            request["identity"], request["front"], request["side"],
            record_id=request["record_id"], on_saved=on_saved, token=token,
        )

    # ------------------------------------------------------------------ 结果页
    def show_result_view(self, record: dict) -> None:
        self.window.show_result_page(record)
        self._main_status("学生练习分析结果")

    def show_preview_view(self) -> None:
        self.window.show_preview_page()
        self._main_status(PRACTICE_STATUS)

    def export_report(self, record: dict) -> None:
        """从结果页导出 HTML 图文报告或 TXT 文字报告。"""
        from core.reporting import render_html_report

        action = record.get("action", "action")
        default_name = f"{record.get('studentId', 'student')}_{ACTIONS.get(action, action)}_练习分析报告"
        root = self.window.root
        dest = filedialog.asksaveasfilename(
            parent=root, title="导出动作问题说明报告", initialfile=default_name, defaultextension=".html",
            filetypes=[("图文报告网页 (*.html)", "*.html"), ("详细文字报告 (*.txt)", "*.txt")],
        )
        if not dest:
            return
        path = Path(dest)
        history = self.analysis_controls.history()
        try:
            if path.suffix.lower() == ".txt":
                history.export(record["id"], path)
            else:
                path.write_text(render_html_report(record, history_root=history.root), encoding="utf-8")
            self.block.status_var.set(f"报告已成功导出至：{path.name}")
            messagebox.showinfo("导出成功", f"报告已导出至：\n{path}", parent=root)
        except Exception as exc:
            messagebox.showerror("导出失败", f"无法写入报告文件：{exc}", parent=root)

    def view_history(self) -> None:
        self.analysis_controls.open_history()

    def review(self, record_id: str, check_id: str, decision: str, teacher: str, reason: str) -> None:
        """教师在结果页切换复核状态并即时刷新结果页。"""
        try:
            updated = self.analysis_controls.history().review(record_id, check_id, decision, teacher, reason)
            if self.practice is not None:
                self.practice.current_record = updated
            self.window.result_panel.set_record(updated)
        except Exception as exc:
            messagebox.showerror("复核失败", str(exc), parent=self.window.root)
