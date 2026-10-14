# -*- coding: utf-8 -*-
"""Tk 桌面的组合根：创建服务、dispatcher、控制器与主窗口并把它们连起来，自身不含业务逻辑；``main()`` 是桌面入口。

逐帧节奏由 ``UiDispatcher`` 调度（后台回调 → 工作流 tick → 录制状态 → 预览）；关窗由 ``ShutdownCoordinator`` 按步骤收尾。
"""
from __future__ import annotations

import time
from tkinter import Tk, messagebox
from typing import Any

from core.foundation import load_record_dir, load_template_pools
from core.live import APP_CLOSE, LiveControls
from core.paths import models_dir
from core.workflows import ModeManager, ScorebookRegistry, WorkflowHost

from apps.tk.controllers.cameras import CameraController, make_camera_service
from apps.tk.controllers.exam_host import ExamHostAdapter
from apps.tk.controllers.flags import LiveFlags
from apps.tk.controllers.practice import PracticeController
from apps.tk.controllers.recording import RecordingController
from apps.tk.controllers.session import SessionController
from apps.tk.dispatcher import UiDispatcher
from apps.tk.shutdown import ShutdownCoordinator, ShutdownSteps
from apps.tk.views.compare_dialog import CompareDialog
from apps.tk.views.exam_window import ExamWindow
from apps.tk.views.main_window import MainWindow
from apps.tk.views.settings_dialog import SettingsDialog


def close_extra_budget_s(finalizer_budget_s: float, scorebooks: ScorebookRegistry) -> float:
    """考试裁剪、已移交台账（含刚移交的面板台账）的回填与落盘需要比基础预算更长的时间。"""
    extra = finalizer_budget_s
    if scorebooks.has_detached:
        extra = max(extra, 8.0)
    if scorebooks.has_processing:
        extra = max(extra, 12.0)
    return extra


class DesktopApp:
    def __init__(self, root: Tk, *, camera_service: Any = None) -> None:
        self.root = root
        self.ui = UiDispatcher(root)
        self.flags = LiveFlags()
        self.modes = ModeManager()
        self.scorebooks = ScorebookRegistry()   # 各场考试的成绩台账：考试窗口关闭后仍可回填
        self.workflows = WorkflowHost(self.scorebooks)
        self.controls = LiveControls()          # 会话运行中可调的参数：主线程写，会话线程每帧读
        record_dir = load_record_dir()
        window = self.window = MainWindow(root, post=self.ui.post, record_dir=record_dir,
                                          templates=load_template_pools())
        self.exam = ExamHostAdapter(
            root, workflows=self.workflows, scorebooks=self.scorebooks, modes=self.modes, controls=self.controls,
            flags=self.flags, post=self.ui.post, session_active=lambda: self.session.active(),
            open_window=ExamWindow,
        )
        self.cameras = CameraController(
            window.capture, camera_service or make_camera_service(), modes=self.modes, flags=self.flags,
            post=self.ui.post, session_active=lambda: self.session.active(), set_status=window.status.status_var.set,
        )
        self.recording = RecordingController(
            window, flags=self.flags, modes=self.modes, workflows=self.workflows, post=self.ui.post,
            session_live=lambda: self.session.live_now(), exam_open=lambda: self.exam.exam_panel is not None,
            record_dir=record_dir,
        )
        self.recording.build_service()
        self.exam.recording = self.recording
        self.session = SessionController(
            window, controls=self.controls, cameras=self.cameras, recording=self.recording, modes=self.modes,
            workflows=self.workflows, flags=self.flags, post=self.ui.post, set_exam_mode=self.exam.set_exam_mode,
        )
        self.session.build_live(sink=self.recording)
        self.practice = PracticeController(
            window, session=self.session, cameras=self.cameras, recording=self.recording, modes=self.modes,
            workflows=self.workflows, controls=self.controls, flags=self.flags, exam=lambda: self.exam.exam_panel,
        )
        self._dialogs: dict[str, Any] = {}
        self.shutdown: ShutdownCoordinator | None = None
        self._connect()
        root.protocol("WM_DELETE_WINDOW", self.close)
        for work in (lambda: self.workflows.tick(time.monotonic()), self.recording.refresh_status,
                     self.session.render_preview):
            self.ui.every_tick(work)
        self.ui.start()
        self.cameras.start_enumeration()

    def _connect(self) -> None:
        window, session, recording, practice = self.window, self.session, self.recording, self.practice
        window.preview.on_click = session.rotate_clockwise
        window.capture.connect(settings=self.open_settings, refresh=self.cameras.refresh, start=session.start,
                               stop=session.stop, camera_selected=self.cameras.primary_selected,
                               camera_2_selected=self.cameras.secondary_selected)
        window.recording.connect(toggle=recording.toggle, stop=recording.stop_clicked,
                                 add_templates=recording.add_templates, remove_templates=recording.remove_templates,
                                 auto_scale=recording.auto_scale_moved, auto_changed=recording.auto_changed,
                                 choose_dir=recording.choose_record_dir)
        recording.sync_template_status()
        recording.sync_hint()
        window.connect(compare=self.open_compare, exam=self.exam.open, practice=practice.enter)
        window.practice.connect(exit=practice.exit, start=practice.start, end=practice.end, judge=practice.judge,
                                set_busy=practice.analysis_busy, can_analyze=practice.can_analyze,
                                identity_changed=practice.identity_changed, cancelled=practice.cancelled)
        panel = window.result_panel
        panel.on_practice_again = practice.practice_again
        panel.on_export_report = practice.export_report
        panel.on_view_history = practice.view_history
        panel.on_review_toggle = practice.review
        panel.history_root = window.practice.controls.history().root

    # ------------------------------------------------------------------ 对话框
    def _open_dialog(self, name: str, title: str, factory: Any) -> None:
        existing = self._dialogs.get(name)
        if existing is not None and existing.is_open():
            existing.focus()
            return
        # 创建失败时弹框并保持主窗口状态不变
        try:
            self._dialogs[name] = factory()
        except Exception as exc:
            messagebox.showerror("打开失败", f"{title}创建失败：{exc}")

    def open_compare(self) -> None:
        self._open_dialog("compare", "动作比对窗口", lambda: CompareDialog(self.root, self.ui.post))

    def open_settings(self) -> None:
        capture = self.window.capture
        self._open_dialog("settings", "设置窗口", lambda: SettingsDialog(
            self.root, self.ui.post, current_pose=capture.pose_var.get(),
            hands_enabled=bool(capture.enable_hands_var.get())))

    # ------------------------------------------------------------------ 关窗
    def close(self) -> None:
        if self.shutdown is None:
            self.shutdown = ShutdownCoordinator(self.root, self.ui, self.shutdown_steps())
        self.shutdown.request()

    def shutdown_steps(self) -> ShutdownSteps:
        recording, live = self.recording, self.session.live
        return ShutdownSteps(
            prepare_workflows=self._close_workflows,
            stop_session=self._enter_closing,
            extra_budget_s=lambda: close_extra_budget_s(recording.service.finalizer.close_budget_s(),
                                                        self.scorebooks),
            release=self._release_for_close,
            wait_session_idle=live.wait_idle,
            finalizer_idle=lambda: recording.service.finalizer.idle,
            flush=self._flush_for_close,
        )

    def _close_workflows(self) -> None:
        """第 1 步（尚未进入关闭态，discard 仍可走）：考试中止并移交台账，练习取消等待与分析。"""
        self.exam.prepare_for_app_close()
        self.practice.close()

    def _enter_closing(self) -> None:
        """第 2 步：进入关闭态，取消会话，释放预热并禁用录制控件。"""
        self.flags.closing = True
        self.session.live.stop(APP_CLOSE)
        self.flags.recording_ready = False
        self.cameras.release_warmups()
        self.recording.disable_for_close()

    def _release_for_close(self) -> None:
        """第 3–5 步（后台任务）：收尾仍在录的双摄片段、取消后处理队列、释放摄像头。"""
        try:
            self.recording.release_for_close()
        finally:
            self.cameras.close()

    def _flush_for_close(self) -> None:
        """第 6 步：最后一次应用后台投递的比对更新，再把所有考试台账落盘关闭。"""
        self.ui.pump()
        self.scorebooks.close_all()


def main() -> None:
    models_dir()   # 离线包首次启动释放内置模型到用户数据目录，源码运行保持原路径
    root = Tk()
    DesktopApp(root)
    root.mainloop()
