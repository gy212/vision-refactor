# -*- coding: utf-8 -*-
"""Tk 桌面的组合根：创建服务、dispatcher、控制器与主窗口并把它们连起来，自身不含业务逻辑；``main()`` 是桌面入口。

逐帧节奏由 ``UiDispatcher`` 调度（后台回调 → 工作流 tick → 录制状态 → 预览）；关窗由 ``ShutdownCoordinator`` 按步骤收尾。
"""

from __future__ import annotations

import time
from tkinter import Tk, messagebox
from typing import Any

from core.foundation import load_record_dir, load_template_pools, models_dir
from core.live import APP_CLOSE, LiveControls
from core.workflows import ModeManager, ScorebookRegistry, WorkflowHost

from apps.tk.controllers.benchmark import BenchmarkContext
from apps.tk.controllers.cameras import CameraController, make_camera_service
from apps.tk.controllers.exam_host import ExamHostAdapter
from apps.tk.controllers.flags import LiveFlags
from apps.tk.controllers.practice import PracticeController
from apps.tk.controllers.recording import RecordingController
from apps.tk.controllers.session import SessionController, session_config
from apps.tk.dispatcher import UiDispatcher
from apps.tk.shutdown import ShutdownCoordinator, ShutdownSteps
from apps.tk.views.benchmark_dialog import BenchmarkDialog
from apps.tk.views.compare_dialog import CompareDialog
from apps.tk.views.exam_window import ExamWindow
from apps.tk.views.main_window import MainWindow
from apps.tk.views.settings_dialog import SettingsDialog
from apps.tk.views.template_builder_dialog import TemplateBuilderDialog


def close_extra_budget_s(
    finalizer_budget_s: float, scorebooks: ScorebookRegistry
) -> float:
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
        self.scorebooks = ScorebookRegistry()
        self.workflows = WorkflowHost(self.scorebooks)
        self.controls = LiveControls()
        record_dir = load_record_dir()
        win = self.window = MainWindow(
            root,
            post=self.ui.post,
            record_dir=record_dir,
            templates=load_template_pools(),
        )
        shared = dict(modes=self.modes, flags=self.flags, post=self.ui.post)
        self.exam = ExamHostAdapter(
            root,
            workflows=self.workflows,
            scorebooks=self.scorebooks,
            controls=self.controls,
            session_active=lambda: self.session.active(),
            open_window=ExamWindow,
            **shared,
        )
        self.cameras = CameraController(
            win.capture,
            camera_service or make_camera_service(),
            session_active=lambda: self.session.active(),
            set_status=win.status.status_var.set,
            **shared,
        )
        self.recording = RecordingController(
            win,
            workflows=self.workflows,
            session_live=lambda: self.session.live_now(),
            exam_open=lambda: self.exam.exam_panel is not None,
            record_dir=record_dir,
            **shared,
        )
        self.recording.build_service()
        self.exam.recording = self.recording
        self.session = SessionController(
            win,
            controls=self.controls,
            cameras=self.cameras,
            recording=self.recording,
            workflows=self.workflows,
            set_exam_mode=self.exam.set_exam_mode,
            **shared,
        )
        self.session.build_live(sink=self.recording)
        self.practice = PracticeController(
            win,
            session=self.session,
            cameras=self.cameras,
            recording=self.recording,
            workflows=self.workflows,
            controls=self.controls,
            modes=self.modes,
            flags=self.flags,
            exam=lambda: self.exam.exam_panel,
        )
        self._dialogs: dict[str, Any] = {}
        self.shutdown: ShutdownCoordinator | None = None
        self._connect()
        root.protocol("WM_DELETE_WINDOW", self.close)
        for work in (
            lambda: self.workflows.tick(time.monotonic()),
            self.recording.refresh_status,
            self.session.render_preview,
        ):
            self.ui.every_tick(work)
        self.ui.start()
        self.cameras.start_enumeration()

    def _connect(self) -> None:
        w, s, r, p = self.window, self.session, self.recording, self.practice
        w.preview.on_click = s.rotate_clockwise
        w.capture.connect(
            settings=self.open_settings,
            refresh=self.cameras.refresh,
            start=s.start,
            stop=s.stop,
            camera_selected=self.cameras.primary_selected,
            camera_2_selected=self.cameras.secondary_selected,
        )
        w.recording.connect(
            toggle=r.toggle,
            stop=r.stop_clicked,
            add_templates=r.pools.add,
            remove_templates=r.pools.remove,
            auto_scale=r.auto_scale_moved,
            auto_changed=r.auto_changed,
            choose_dir=r.choose_record_dir,
            build_templates=self.open_template_builder,
        )
        r.pools.sync_status()
        r.sync_hint()
        w.connect(
            compare=self.open_compare,
            exam=self.exam.open,
            practice=p.enter,
            back_to_preview=p.show_preview_view,
        )
        for key in ("<Control-Shift-KeyPress-B>", "<Control-Shift-KeyPress-b>"):
            self.root.bind_all(key, lambda _event: self.open_benchmark())
        w.practice.connect(
            exit=p.exit,
            start=p.start,
            end=p.end,
            judge=p.judge,
            set_busy=p.analysis_busy,
            can_analyze=p.can_analyze,
            identity_changed=p.identity_changed,
            cancelled=p.cancelled,
        )
        panel = w.result_panel
        panel.on_practice_again = p.practice_again
        panel.on_export_report = p.export_report
        panel.on_view_history = p.view_history
        panel.on_review_toggle = p.review
        panel.history_root = w.practice.controls.history().root

    # ------------------------------------------------------------------ 对话框
    def _open_dialog(self, name: str, title: str, factory: Any) -> None:
        existing = self._dialogs.get(name)
        if existing is not None and existing.is_open():
            existing.focus()
            return
        try:
            self._dialogs[name] = factory()
        except Exception as exc:
            messagebox.showerror("打开失败", f"{title}创建失败：{exc}")

    def open_compare(self) -> None:
        cb = lambda fn: self.open_template_builder(on_created=fn)
        self._open_dialog(
            "compare",
            "动作比对窗口",
            lambda: CompareDialog(self.root, self.ui.post, open_builder=cb),
        )

    def open_template_builder(self, *, on_created: Any = None) -> None:
        cb = self.recording.pools.add_created if on_created is None else on_created
        dlg = self._dialogs.get("template_builder")
        if dlg is not None and dlg.is_open():
            dlg.set_on_created(cb)
            dlg.focus()
            return
        make = lambda: TemplateBuilderDialog(
            self.root, self.ui.post, record_dir=self.recording.record_dir, on_created=cb
        )
        self._open_dialog("template_builder", "制作模板窗口", make)

    def open_settings(self) -> None:
        c = self.window.capture
        make = lambda: SettingsDialog(
            self.root,
            self.ui.post,
            current_pose=c.pose_var.get(),
            hands_enabled=bool(c.enable_hands_var.get()),
        )
        self._open_dialog("settings", "设置窗口", make)

    def open_benchmark(self) -> None:
        c = self.window.capture
        cfg = lambda: session_config(
            self.modes.session_settings(self.session.collect_state())
        )
        ctx = BenchmarkContext(
            cameras=self.cameras,
            session_active=self.session.active,
            mode_name=lambda: self.modes.name,
            rotation_of=self.controls.rotation_of,
            pose_variant=lambda: c.pose_var.get(),
            enable_hands=lambda: bool(c.enable_hands_var.get()),
            record_dir=self.recording.record_dir,
            post=self.ui.post,
            session_config=cfg,
            preview=self.window.preview,
        )
        self._open_dialog(
            "benchmark",
            "录制性能测试窗口",
            lambda: BenchmarkDialog(self.root, self.ui.post, ctx),
        )

    # ------------------------------------------------------------------ 关窗
    def close(self) -> None:
        if self.shutdown is None:
            self.shutdown = ShutdownCoordinator(
                self.root, self.ui, self.shutdown_steps()
            )
        self.shutdown.request()

    def shutdown_steps(self) -> ShutdownSteps:
        rec, live = self.recording, self.session.live
        budget = lambda: close_extra_budget_s(
            rec.service.finalizer.close_budget_s(), self.scorebooks
        )
        return ShutdownSteps(
            prepare_workflows=self._close_workflows,
            stop_session=self._enter_closing,
            extra_budget_s=budget,
            release=self._release_for_close,
            wait_session_idle=live.wait_idle,
            finalizer_idle=lambda: rec.service.finalizer.idle,
            flush=self._flush_for_close,
        )

    def _close_workflows(self) -> None:
        """第 1 步（尚未进入关闭态，discard 仍可走）：考试中止并移交台账，练习取消等待与分析。"""
        self.exam.prepare_for_app_close()
        self.practice.close()
        for name in ("benchmark", "compare", "template_builder"):
            dialog = self._dialogs.get(name)
            if dialog is not None and dialog.is_open():
                dialog.close()

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
    models_dir()
    root = Tk()
    DesktopApp(root)
    root.mainloop()
