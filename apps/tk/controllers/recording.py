# -*- coding: utf-8 -*-
"""录制编排：录制按钮三态、片段开录与结束、按模式与用户设置构建录后处理计划、比对状态回显、模板池与保存目录。

本控制器同时是会话的 ``FrameSink``（转给 ``RecordingService``）与工作流的录制端口（``SegmentControl``）。
录后处理的状态更新从 finalizer 线程经 dispatcher 回到 Tk 线程；考试台账按片段回填，关窗收尾时也要回填。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from tkinter import messagebox
from typing import Any

from core.foundation import save_record_dir, save_template_pools
from core.foundation import outputs_dir
from core.pose import MEDIAPIPE_MODELS, is_installed
from core.recording import (
    BeginError,
    FinishedSegment,
    ProcessingPlan,
    RecordingService,
    RecordingState,
    SegmentFinalizer,
    SegmentMeta,
    SegmentUpdate,
    try_load_heavy_template,
)
from core.workflows import ModeManager, PlanSettings, PreflightContext, SegmentIntent, WorkflowHost, build_segment_plan

from apps.tk.controllers.flags import LiveFlags
from apps.tk.views.status_panel import EMPTY_SCORES

STATUS_LABELS = {
    "queued": "排队中", "transcoding": "正在转码", "validating": "正在校验", "comparing": "正在比对",
    "completed": "已完成", "failed": "失败", "skipped": "已跳过", "cancelled": "已取消",
}


def model_installed(key: str) -> bool:
    """按模型键（如 ``pose_lite``）查询 MediaPipe 模型是否已安装。"""
    spec = next((m for m in MEDIAPIPE_MODELS if m.key == key), None)
    return spec is not None and is_installed(spec)


def percent(score: float | None) -> str:
    return "-" if score is None else f"{float(score) * 100.0:.1f}%"


def template_status_text(front: int, side: int) -> str:
    if front and side:
        return f"模板池：正面 {front} 个 · 侧面 {side} 个（正对正、侧对侧，逐动作出分）"
    if front:
        return f"模板池：正面 {front} 个 · 侧面 0 个（仅比对正面；两侧都空才报错）"
    if side:
        return f"模板池：正面 0 个 · 侧面 {side} 个（仅比对侧面；两侧都空才报错）"
    return "模板池：正/侧均为空 — 开启自动比对将失败"


class RecordingController:
    """``view`` 提供 ``recording``（录制面板）与 ``status``（状态区）；``service`` 由 ``build_service`` 或调用方接上。"""

    def __init__(self, view: Any, *, flags: LiveFlags, modes: ModeManager, workflows: WorkflowHost,
                 post: Callable[..., Any], session_live: Callable[[], bool], exam_open: Callable[[], bool],
                 record_dir: Path) -> None:
        self.view = view
        self.flags = flags
        self.modes = modes
        self.workflows = workflows
        self._post = post
        self._session_live = session_live
        self._exam_open = exam_open
        # worker 的 path_provider 不读 Tk 变量：主线程在初始化与开录时把目录缓存为普通 Path
        self.record_base_dir = Path(record_dir)
        self.service: RecordingService | None = None
        self.latest_segment_id: str | None = None
        self.pending_errors: list[str] = []   # 会话关闭前转存、尚未提示的单摄写错误
        self._error_shown = False             # 同一录制错误只提示一次
        self._templates = (tuple(view.recording.pool_paths("front")), tuple(view.recording.pool_paths("side")))

    def build_service(self) -> RecordingService:
        """结束片段并提交处理计划在 ``RecordingService`` 的同一把锁内完成；计划由 ``plan`` 构建。"""
        self.service = RecordingService(SegmentFinalizer(on_update=self.post_update), plan_for=self.plan)
        return self.service

    @property
    def state(self) -> RecordingState:
        return self.service.state

    # ------------------------------------------------------------------ FrameSink
    def begin_session(self, sizes, fps: float) -> None:
        self.service.begin_session(sizes, fps)

    def update_sizes(self, sizes) -> None:
        self.service.update_sizes(sizes)

    def write(self, frames) -> None:
        self.service.write(frames)

    def close_session(self) -> None:
        """会话结束时关闭录制：仍在进行的片段照常提交；单摄写错误留给下一次 Tk tick 提示。"""
        outcome = self.service.close_session()
        self.flags.dual_active = False
        segment = outcome.segment
        if segment is not None and not segment.multi:
            self.pending_errors.extend(f"第一路：{error}" for error in segment.errors.values())

    # ------------------------------------------------------------------ SegmentControl（工作流）
    def start_segment(self, payload: dict) -> Any:
        return self.begin(SegmentIntent(exam=payload.get("exam")))

    def finish(self, *, discard: bool) -> None:
        self.end(discard=discard)

    # ------------------------------------------------------------------ 录制按钮
    def _accepting(self) -> bool:
        return not self.flags.closing and self._session_live() and self.flags.recording_ready

    def toggle(self) -> None:
        """idle→开录、recording→暂停、paused→继续；双摄片段已分叉时先结束并提交，不在分叉状态上开新片段。"""
        if self.flags.exam_locked or not self._accepting():
            return
        if self.service.finish_if_broken() is not None:
            new_state: RecordingState = "idle"
        elif self.service.state == "idle":
            self.begin()
            return
        else:
            new_state = self.service.toggle_pause()
        self.view.recording.set_record_text(new_state)
        self.sync_stop_enabled(new_state)

    def stop_clicked(self) -> None:
        """「结束录制」：结束当前片段并落盘，但不结束识别会话。"""
        if self.flags.exam_locked or self.flags.closing or not self._session_live():
            return
        self.end(discard=False)

    def begin(self, intent: SegmentIntent | None = None) -> Any:
        """从 idle 开录一段（工作流与手动共用）；成功返回片段，失败返回 None。各路同时开录，任一路失败全部复位。"""
        if not self._accepting():
            return None
        typed = self.view.recording.record_dir_var.get().strip()
        # 空串多为 Tk 读取时序异常而非"要默认目录"：保持上次缓存的目录，不把持久化路径覆盖成默认
        if typed:
            self.record_base_dir = Path(typed)
        save_record_dir(self.record_base_dir)
        intent = intent or SegmentIntent()
        if intent.auto_compare is None:
            intent = replace(intent, auto_compare=self.modes.segment_auto_compare)
        segment = self.service.start(SegmentMeta(self.record_base_dir, context=intent))
        if isinstance(segment, BeginError):
            return None
        if segment.multi:
            self._show_new_dual_segment(segment.segment_id, intent)
        if not self.flags.exam_locked:
            self.view.recording.set_record_text("recording")
        self.sync_stop_enabled("recording")
        return segment

    def _show_new_dual_segment(self, segment_id: str, intent: SegmentIntent) -> None:
        auto_compare = intent.auto_compare
        if auto_compare is None:
            auto_compare = self.current_auto_compare(prefer_ui=True)
        self.flags.auto_compare = bool(auto_compare)
        self.view.status.show_compare(segment_id, "自动比对：录制中" if auto_compare else "仅录制：录制中")

    def end(self, *, discard: bool = False) -> None:
        """结束当前片段；``discard`` 只落盘不比对。discard 在关窗/中止路径也必须执行，否则已中止的片段会被送去评分。"""
        if not discard and (self.flags.closing or not self._session_live()):
            return
        self.service.finish(discard=discard)
        if not self.flags.exam_locked:
            self.view.recording.set_record_text("idle")
        self.sync_stop_enabled("idle")

    def sync_stop_enabled(self, state: RecordingState) -> None:
        enabled = state in ("recording", "paused")
        self.view.recording.record_stop_btn.configure(state="normal" if enabled else "disabled")

    def session_running(self, running: bool) -> None:
        """会话开始/结束时的录制控件：骨架开关在会话期间锁定；录制按钮等录制就绪；考试锁定期间不可调自动比对。"""
        panel = self.view.recording
        panel.record_skeleton_check.configure(state="disabled" if running else "normal")
        panel.auto_compare_scale.configure(state="disabled" if self.flags.exam_locked else "normal")
        if running and self.flags.recording_ready:
            panel.record_btn.configure(state="normal")
            panel.set_record_text("idle")
        else:
            panel.record_btn.configure(state="disabled")
        self.sync_stop_enabled("idle")

    def lock_manual(self, locked: bool, *, session_active: bool) -> None:
        """考试进行中锁定手动录制、骨架与自动比对（考试强制比对、骨架关）；解锁时按会话态恢复。"""
        self.flags.exam_locked = bool(locked)
        panel = self.view.recording
        if locked:
            for widget in (panel.record_btn, panel.record_stop_btn, panel.record_skeleton_check,
                           panel.auto_compare_scale):
                widget.configure(state="disabled")
            panel.record_skeleton_var.set(False)
            self.set_auto_compare(True)
            return
        if not session_active:
            panel.record_skeleton_check.configure(state="normal")
        panel.auto_compare_scale.configure(state="normal")
        if self.flags.recording_ready or session_active:
            panel.record_btn.configure(state="normal")

    def disable_for_close(self) -> None:
        self.view.recording.set_controls_state("disabled")

    def release_for_close(self) -> None:
        """关窗后台步骤：收尾仍在录的双摄片段，再取消后处理队列（考试中止已在进入关闭态前 discard）。"""
        if self.flags.dual_active:
            self.service.finish()
        self.service.cancel_all()

    # ------------------------------------------------------------------ 自动比对开关
    def current_auto_compare(self, *, prefer_ui: bool = False) -> bool:
        """片段级自动比对：模式锁定时按模式（考试强制开、练习强制关）；``prefer_ui`` 时（仅主线程）从界面刷新缓存。"""
        forced = self.modes.segment_auto_compare
        if forced is not None:
            return forced
        if prefer_ui:
            self.flags.auto_compare = bool(self.view.recording.auto_compare_var.get())
        return self.flags.auto_compare

    def set_auto_compare(self, on: bool) -> None:
        panel = self.view.recording
        panel.auto_compare_scale_var.set(1.0 if on else 0.0)
        if bool(panel.auto_compare_var.get()) != on:
            panel.auto_compare_var.set(on)
        self.flags.auto_compare = bool(on)
        self.sync_hint()

    def auto_scale_moved(self, value: str) -> None:
        """滑块吸附到两端。"""
        try:
            on = float(value) >= 0.5
        except (TypeError, ValueError):
            on = True
        self.set_auto_compare(on)

    def auto_changed(self) -> None:
        self.set_auto_compare(bool(self.view.recording.auto_compare_var.get()))

    def sync_hint(self) -> None:
        on = bool(self.view.recording.auto_compare_var.get())
        self.view.recording.auto_compare_hint_var.set("录制后自动比对（检测一条龙）" if on else "仅录制落盘，不自动比对")

    # ------------------------------------------------------------------ 模板池与保存目录
    def template_lists(self) -> tuple[list[Path], list[Path]]:
        front, side = self._templates
        return list(front), list(side)

    def add_templates(self, view: str) -> None:
        paths = self.view.recording.ask_templates(view)
        if not paths:
            return
        existing = {str(p) for p in self.view.recording.pool_paths(view)}
        rejected: list[str] = []
        for path in paths:
            usable = try_load_heavy_template(path)
            if usable is None:
                rejected.append(Path(path).name)
            elif str(usable) not in existing:
                self.view.recording.add_pool_path(view, usable)
                existing.add(str(usable))
        self.templates_changed()
        if rejected:
            messagebox.showerror(
                "部分模板无效",
                "以下文件不是可用的 heavy + pose33_v3 模板，已跳过：\n" + "\n".join(rejected)
                + "\n请用 make_template.py --pose heavy 生成后再添加。",
                parent=self.view.recording.root,
            )

    def remove_templates(self, view: str) -> None:
        self.view.recording.remove_selected(view)
        self.templates_changed()

    def templates_changed(self) -> None:
        """池变化后缓存为普通路径（录后计划可能在会话线程构建）、持久化并刷新提示。"""
        panel = self.view.recording
        self._templates = (tuple(panel.pool_paths("front")), tuple(panel.pool_paths("side")))
        try:
            save_template_pools(front=list(self._templates[0]), side=list(self._templates[1]))
        except Exception:
            pass
        self.sync_template_status()

    def sync_template_status(self) -> None:
        self.view.recording.template_status_var.set(template_status_text(*map(len, self._templates)))

    def choose_record_dir(self) -> None:
        current = self.view.recording.record_dir_var.get().strip()
        chosen = self.view.recording.ask_record_dir(current or str(outputs_dir()))
        if chosen:
            self.view.recording.record_dir_var.set(chosen)
            save_record_dir(chosen)

    def record_dir(self) -> Path:
        """考试场次目录的父目录：录制保存目录，未填时回落 outputs。"""
        return Path(self.view.recording.record_dir_var.get().strip() or str(outputs_dir()))

    # ------------------------------------------------------------------ 录后处理
    def plan(self, segment: FinishedSegment) -> ProcessingPlan | None:
        """按开录时登记的意图与结束时的设置构建处理计划；在服务锁内、提交前先把本片段登记为最新。"""
        if not segment.multi:
            return build_segment_plan(segment, PlanSettings(record_skeleton=False, auto_compare=False))
        front, side = self._templates
        plan = build_segment_plan(segment, PlanSettings(
            record_skeleton=self.flags.record_skeleton,
            auto_compare=bool(self.current_auto_compare()),
            front_templates=tuple(front), side_templates=tuple(side),
        ))
        self.latest_segment_id = segment.segment_id
        return plan

    def post_update(self, update: SegmentUpdate) -> None:
        """finalizer 线程：只投递不可变状态。关窗后只为考试台账（面板或已移交的台账）投递。"""
        if self.flags.closing and not self._exam_open() and not self.workflows.scorebooks.has_detached:
            return
        self._post(lambda: self.apply_update(update), during_close=True)

    def apply_update(self, update: SegmentUpdate) -> None:
        """Tk 线程：比对状态栏只显示最新片段；考试台账按 segment_id 回填每一位，关窗收尾时只回填台账。"""
        closing = self.flags.closing
        if update.segment_id == self.latest_segment_id and not closing:
            label = f"自动比对：{STATUS_LABELS.get(update.status, update.status)}"
            if update.status == "completed":
                scores = (f"正面：{percent(update.front_score)}　侧面：{percent(update.side_score)}　"
                          f"综合：{int(update.combined_percent or 0)}%")
                self.view.status.show_compare(update.segment_id, label, scores)
            elif update.status in {"failed", "skipped", "cancelled"}:
                detail = f"{update.error_code}：{update.message}" if update.error_code else update.message
                self.view.status.show_compare(update.segment_id, label, EMPTY_SCORES, detail)
            else:
                self.view.status.show_compare(update.segment_id, label, None)
        self.workflows.segment_update(update, route=not closing)

    def refresh_status(self) -> None:
        """每 tick：显示录制状态与保存路径；写错误时结束各路当前片段并提交，单摄弹框提示（只提示一次）。"""
        if self.flags.closing:
            return
        recorder = self.service.snapshot()
        multi = len(recorder.views) > 1
        errors: list[str] = []
        while self.pending_errors:   # 逐条取出，会话线程可能同时追加
            errors.append(self.pending_errors.pop(0))
        for view, error in recorder.errors.items():
            text = f"{'第二路' if view == 'side' else '第一路'}：{error}"
            if text not in errors:
                errors.append(text)
        status_var = self.view.status.recording_status_var
        if errors:
            if not self._error_shown:
                self._error_shown = True
                self.service.finish()   # 双摄必须保持成对：任一路失败即结束各路并提交
                self.view.recording.set_record_text("idle")
                self.sync_stop_enabled("idle")
                status_var.set("")
                if not multi:
                    messagebox.showerror("录制失败", "录制发生错误：\n" + "\n".join(errors))
            return
        self._error_shown = False
        if recorder.state in ("recording", "paused"):
            texts = [str(snap.result_path) if snap.result_path is not None else "（准备中）"
                     for snap in recorder.views.values()]
            path_text = f"正面 {texts[0]}；侧面 {texts[1]}" if multi else texts[0]
            status_var.set(f"{'录制中' if recorder.state == 'recording' else '已暂停'}：{path_text}")
        else:
            status_var.set("")
        self.sync_stop_enabled(recorder.state)

    # ------------------------------------------------------------------ 预检
    def preflight_context(self, *, practice: bool = False, identity_error: str | None = None,
                          cameras_error: str | None = None) -> PreflightContext:
        """采集模式预检要用的事实；练习另查身份与双摄选择，考试另查模板池。"""
        skeleton_on = self.flags.record_skeleton or bool(self.view.recording.record_skeleton_var.get())
        front, side = self._templates if (not practice and self.flags.dual_active) else ((), ())
        return PreflightContext(
            dual_active=self.flags.dual_active, recording_ready=self.flags.recording_ready,
            skeleton_on=skeleton_on, front_templates=tuple(front), side_templates=tuple(side),
            model_installed=model_installed, identity_error=identity_error, cameras_error=cameras_error,
        )
