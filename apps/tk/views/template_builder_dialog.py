# -*- coding: utf-8 -*-
"""独立「制作模板」弹窗：双摄/单摄视频来源选择、同步双路起止滑块与画面预览、命名防覆盖与质量体检反馈。

媒体句柄与后台生成逻辑在 ``TemplateBuilderController``，界面只做输入收集、防抖刷新与结果显示。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from tkinter import (
    BooleanVar,
    DoubleVar,
    Scrollbar,
    StringVar,
    TclError,
    Text,
    Tk,
    Toplevel,
    filedialog,
    messagebox,
    ttk,
)
from typing import Any

from core.evaluation import ACTIONS
from core.services import TemplateCreateResult, sanitize_action_name

from apps.tk.controllers.template_builder import (
    TemplateBuilderController,
    format_range_summary,
    format_sources_info,
    min_duration_seconds,
    min_step_seconds,
    nudge_time,
)
from apps.tk.views.widgets import FramePreviewSlot, ProgressPanel

logger = logging.getLogger(__name__)

_VIDEO_TYPES = [("视频文件", "*.mp4;*.avi;*.mov;*.mkv;*.m4v"), ("所有文件", "*.*")]
_SINGLE_VIEW_OPTIONS = ("正面", "侧面", "不加视角后缀")
_SINGLE_VIEW_MAP = {"正面": "front", "侧面": "side", "不加视角后缀": "single"}
_PREVIEW_DEBOUNCE_MS = 30
_NUDGE_DELTAS = (-5, -1, 1, 5)


class TemplateBuilderDialog:
    """独立可复用「从视频制作模板」窗口；可由主界面或动作分析窗口打开并自动回填模板池。"""

    def __init__(
        self,
        parent: Tk,
        post: Callable[[Callable[[], Any]], Any],
        *,
        record_dir: Path | Callable[[], Path] | None = None,
        on_created: Callable[[list[TemplateCreateResult]], None] | None = None,
        controller_factory: Callable[..., TemplateBuilderController] | None = None,
    ) -> None:
        self._win = Toplevel(parent)
        self._win.title("从视频制作模板")
        self._win.geometry("820x790")
        self._win.minsize(720, 680)
        self._win.transient(parent)

        self._record_dir = record_dir
        self._on_created = on_created
        self._max_s = 0.0
        self._suppress_scale = False
        self._preview_after_id: str | None = None
        self._lockable_widgets: list[Any] = []

        self.single_view_label_var = StringVar(value="正面")
        self.source_info_var = StringVar(value=format_sources_info({}))
        self.anchor_var = StringVar(value="start")
        self.start_var = DoubleVar(value=0.0)
        self.end_var = DoubleVar(value=0.0)
        self.start_time_var = StringVar(value="0.00s")
        self.end_time_var = StringVar(value="0.00s")
        self.range_info_var = StringVar(value="已选区间：—")
        self.action_name_var = StringVar(value="")
        self.output_hint_var = StringVar(value="预计保存：请先选择视频与动作名称")
        self.preview_var = BooleanVar(value=False)
        self.auto_add_var = BooleanVar(value=True)
        self.status_var = StringVar(value="请选择双摄录制目录或单个视频文件")
        self.progress_text_var = StringVar(value="")

        guarded = lambda cb: post(lambda: self._win.winfo_exists() and cb())
        self.controller = (controller_factory or TemplateBuilderController)(
            self, guarded
        )

        outer = ttk.Frame(self._win, padding=12)
        outer.pack(fill="both", expand=True)
        self._build_source(outer)
        self._build_preview_and_range(outer)
        self._build_options(outer)
        self._build_actions(outer)
        self._sync_auto_add_visibility()
        self._set_range_controls_enabled(False)
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

    def is_open(self) -> bool:
        return bool(self._win.winfo_exists())

    def focus(self) -> None:
        try:
            self._win.deiconify()
            self._win.lift()
            self._win.focus_force()
        except TclError as exc:
            logger.debug("聚焦制作模板窗口失败：%s", exc)

    def set_on_created(
        self, on_created: Callable[[list[TemplateCreateResult]], None] | None
    ) -> None:
        """复用已打开弹窗时更新回填回调与勾选框可见性。"""
        self._on_created = on_created
        self._sync_auto_add_visibility()

    def close(self) -> None:
        """外部（主窗关闭）或窗口关闭协议触发：停任务、消定时器、关视频句柄并销毁窗体。"""
        self._cancel_preview_debounce()
        self.controller.close()
        if self._win.winfo_exists():
            self._win.destroy()

    # ------------------------------------------------------------------ 布局
    def _build_source(self, outer: ttk.Frame) -> None:
        box = ttk.Labelframe(outer, text="① 选择视频来源", padding=8)
        box.pack(fill="x")
        row = ttk.Frame(box)
        row.pack(fill="x")
        self.pick_dir_btn = ttk.Button(
            row, text="选择双摄录制目录…", command=self._choose_dir
        )
        self.pick_dir_btn.pack(side="left")
        self.pick_file_btn = ttk.Button(
            row, text="选择单视频…", command=self._choose_file
        )
        self.pick_file_btn.pack(side="left", padx=(8, 0))
        ttk.Label(row, text="单视频视角：").pack(side="left", padx=(16, 4))
        self.single_view_combo = ttk.Combobox(
            row,
            textvariable=self.single_view_label_var,
            values=list(_SINGLE_VIEW_OPTIONS),
            state="readonly",
            width=14,
        )
        self.single_view_combo.pack(side="left")
        self.single_view_combo.bind(
            "<<ComboboxSelected>>", self._on_single_view_changed
        )
        ttk.Label(
            box, textvariable=self.source_info_var, foreground="#444444", wraplength=760
        ).pack(anchor="w", pady=(6, 0))
        self._lockable_widgets.extend(
            [self.pick_dir_btn, self.pick_file_btn, self.single_view_combo]
        )

    def _build_preview_and_range(self, outer: ttk.Frame) -> None:
        box = ttk.Labelframe(outer, text="② 同步区间与画面预览", padding=8)
        box.pack(fill="x", pady=(10, 0))

        preview_row = ttk.Frame(box)
        preview_row.pack(fill="x")
        preview_row.columnconfigure(0, weight=1)
        preview_row.columnconfigure(1, weight=1)
        self._slots = {
            "front": FramePreviewSlot(preview_row, "正面画面", 0),
            "side": FramePreviewSlot(preview_row, "侧面画面", 1),
        }

        anchor_row = ttk.Frame(box)
        anchor_row.pack(fill="x", pady=(8, 0))
        ttk.Label(anchor_row, text="预览跟随：").pack(side="left")
        self._range_widgets: list[Any] = []
        for val, text in (("start", "起始帧"), ("end", "结束帧")):
            rb = ttk.Radiobutton(
                anchor_row,
                text=text,
                value=val,
                variable=self.anchor_var,
                command=self._render_preview,
            )
            rb.pack(side="left", padx=(4, 12) if val == "start" else (0, 0))
            self._range_widgets.append(rb)

        self.start_scale = self._build_range_row(
            box, "起始位置：", self.start_var, self.start_time_var, "start"
        )
        self.end_scale = self._build_range_row(
            box, "结束位置：", self.end_var, self.end_time_var, "end"
        )
        ttk.Label(box, textvariable=self.range_info_var, foreground="#222222").pack(
            anchor="w", pady=(6, 0)
        )
        self._lockable_widgets.extend(self._range_widgets)

    def _build_range_row(
        self,
        parent: ttk.Frame,
        label: str,
        var: DoubleVar,
        time_var: StringVar,
        which: str,
    ) -> ttk.Scale:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text=label, width=10).pack(side="left")
        scale = ttk.Scale(
            row,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=var,
            command=lambda val, w=which: self._on_slide(w, val),
        )
        scale.pack(side="left", fill="x", expand=True, padx=(4, 8))
        ttk.Label(row, textvariable=time_var, width=8, anchor="e").pack(
            side="left", padx=(0, 6)
        )
        self._range_widgets.append(scale)
        for delta in _NUDGE_DELTAS:
            btn = ttk.Button(
                row,
                text=f"{delta:+d}帧",
                width=5,
                command=lambda w=which, d=delta: self._nudge(w, d),
            )
            btn.pack(side="left", padx=(2, 0))
            self._range_widgets.append(btn)
        return scale

    def _build_options(self, outer: ttk.Frame) -> None:
        box = ttk.Labelframe(outer, text="③ 命名与生成选项", padding=8)
        box.pack(fill="x", pady=(10, 0))
        row1 = ttk.Frame(box)
        row1.pack(fill="x")
        ttk.Label(row1, text="动作名称：").pack(side="left")
        self.action_combo = ttk.Combobox(
            row1,
            textvariable=self.action_name_var,
            values=list(ACTIONS.values()),
            state="normal",
            width=28,
        )
        self.action_combo.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.action_name_var.trace_add(
            "write", lambda *_args: self._refresh_output_hint()
        )

        self.output_hint_label = ttk.Label(
            box, textvariable=self.output_hint_var, foreground="#555555", wraplength=760
        )
        self.output_hint_label.pack(anchor="w", pady=(6, 0))

        opt_row = ttk.Frame(box)
        opt_row.pack(fill="x", pady=(6, 0))
        self.preview_check = ttk.Checkbutton(
            opt_row,
            text="同时导出骨架预览视频（.preview.mp4）",
            variable=self.preview_var,
        )
        self.preview_check.pack(side="left")
        self.auto_add_check = ttk.Checkbutton(
            opt_row, text="生成后自动加入模板池", variable=self.auto_add_var
        )
        self.auto_add_check.pack(side="left", padx=(16, 0))
        self._lockable_widgets.extend(
            [self.action_combo, self.preview_check, self.auto_add_check]
        )

    def _build_actions(self, outer: ttk.Frame) -> None:
        box = ttk.Labelframe(outer, text="④ 生成与体检结果", padding=8)
        box.pack(fill="both", expand=True, pady=(10, 0))

        btn_row = ttk.Frame(box)
        btn_row.pack(fill="x")
        self.generate_btn = ttk.Button(
            btn_row, text="开始生成", command=self.start_generate
        )
        self.generate_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(
            btn_row, text="停止", command=self.controller.stop, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=(8, 0))
        self.close_btn = ttk.Button(btn_row, text="关闭", command=self._on_close)
        self.close_btn.pack(side="left", padx=(8, 0))

        self._progress = ProgressPanel(box, self.progress_text_var)
        self.progress_bar = self._progress.bar

        ttk.Label(box, textvariable=self.status_var).pack(anchor="w", pady=(6, 0))
        txt_box = ttk.Frame(box)
        txt_box.pack(fill="both", expand=True, pady=(4, 0))
        self.result_text = Text(txt_box, height=5, wrap="word")
        self.result_text.pack(side="left", fill="both", expand=True)
        sb = Scrollbar(txt_box, command=self.result_text.yview)
        sb.pack(side="right", fill="y")
        self.result_text.configure(yscrollcommand=sb.set)

    # ------------------------------------------------------------------ 交互
    def _initial_dir(self) -> str | None:
        base = self._record_dir() if callable(self._record_dir) else self._record_dir
        return str(base) if base is not None and Path(base).exists() else None

    def _selected_single_view(self) -> str:
        return _SINGLE_VIEW_MAP.get(self.single_view_label_var.get(), "front")

    def _load_with(
        self, picker: Callable[[], str], loader: Callable[[str], Any]
    ) -> None:
        if self.controller.busy:
            return
        path = picker()
        self.focus()
        if not path:
            return
        try:
            loader(path)
        except Exception as exc:
            messagebox.showerror("加载失败", str(exc), parent=self._win)
            return
        self._on_sources_loaded()

    def _choose_dir(self) -> None:
        self._load_with(
            lambda: filedialog.askdirectory(
                title="选择双摄录制片段目录",
                initialdir=self._initial_dir(),
                parent=self._win,
            ),
            self.controller.load_directory,
        )

    def _choose_file(self) -> None:
        self._load_with(
            lambda: filedialog.askopenfilename(
                title="选择视频文件",
                filetypes=_VIDEO_TYPES,
                initialdir=self._initial_dir(),
                parent=self._win,
            ),
            lambda p: self.controller.load_single_file(p, self._selected_single_view()),
        )

    def _on_single_view_changed(self, _event: Any = None) -> None:
        if not self.controller.single_mode:
            return
        self.controller.set_single_view(self._selected_single_view())
        self._sync_source_ui()
        self._render_preview()

    def _on_sources_loaded(self) -> None:
        metas = self.controller.metas
        self._max_s = max(0.0, min_duration_seconds(metas))
        self._suppress_scale = True
        try:
            upper = max(self._max_s, 1e-3)
            self.start_scale.configure(from_=0.0, to=upper)
            self.end_scale.configure(from_=0.0, to=upper)
            self.start_var.set(0.0)
            self.end_var.set(self._max_s)
        finally:
            self._suppress_scale = False
        self._set_range_controls_enabled(bool(metas))
        self._sync_source_ui()
        self._render_preview()
        self.set_status("视频已加载，请拖动滑块确认起止区间并填写动作名称")

    def _sync_source_ui(self) -> None:
        metas = self.controller.metas
        self.source_info_var.set(format_sources_info(metas))
        self._sync_slot_layout(metas)
        self._sync_range_labels()
        self._refresh_output_hint()

    def _sync_slot_layout(self, metas: dict[str, Any]) -> None:
        if not metas:
            for slot in self._slots.values():
                slot.frame.grid()
                slot.clear("未加载")
            return
        if "front" in metas and "side" in metas:
            self._slots["front"].frame.configure(text="正面画面")
            self._slots["front"].frame.grid()
            self._slots["side"].frame.configure(text="侧面画面")
            self._slots["side"].frame.grid()
            return
        active = next(iter(metas))
        hide_key, show_key = (
            ("front", "side") if active == "side" else ("side", "front")
        )
        title = {"side": "侧面画面", "front": "正面画面"}.get(active, "视频画面")
        self._slots[hide_key].frame.grid_remove()
        self._slots[show_key].frame.configure(text=title)
        self._slots[show_key].frame.grid()

    def _on_slide(self, which: str, value: str) -> None:
        if self._suppress_scale or not self.controller.metas:
            return
        step = min_step_seconds(self.controller.metas)
        raw = float(value)
        self._suppress_scale = True
        try:
            if which == "start":
                val = min(max(0.0, raw), max(0.0, self.end_var.get() - step))
                self.start_var.set(val)
            else:
                val = max(
                    min(self._max_s, raw), min(self._max_s, self.start_var.get() + step)
                )
                self.end_var.set(val)
        finally:
            self._suppress_scale = False
        self.anchor_var.set(which)
        self._sync_range_labels()
        self._schedule_preview()

    def _nudge(self, which: str, delta_frames: int) -> None:
        metas = self.controller.metas
        if not metas:
            return
        fps = next(iter(metas.values())).fps
        step = min_step_seconds(metas)
        self._suppress_scale = True
        try:
            if which == "start":
                upper = max(0.0, self.end_var.get() - step)
                self.start_var.set(
                    nudge_time(self.start_var.get(), delta_frames, fps, 0.0, upper)
                )
            else:
                lower = min(self._max_s, self.start_var.get() + step)
                self.end_var.set(
                    nudge_time(
                        self.end_var.get(), delta_frames, fps, lower, self._max_s
                    )
                )
            self.anchor_var.set(which)
        finally:
            self._suppress_scale = False
        self._sync_range_labels()
        self._render_preview()

    def _sync_range_labels(self) -> None:
        s, e = self.start_var.get(), self.end_var.get()
        self.start_time_var.set(f"{s:.2f}s")
        self.end_time_var.set(f"{e:.2f}s")
        self.range_info_var.set(format_range_summary(self.controller.metas, s, e))

    def _schedule_preview(self) -> None:
        self._cancel_preview_debounce()
        self._preview_after_id = self._win.after(
            _PREVIEW_DEBOUNCE_MS, self._render_preview
        )

    def _cancel_preview_debounce(self) -> None:
        if self._preview_after_id is not None:
            try:
                self._win.after_cancel(self._preview_after_id)
            except TclError as exc:
                logger.debug("取消预览防抖定时器失败：%s", exc)
            self._preview_after_id = None

    def _render_preview(self) -> None:
        self._preview_after_id = None
        metas = self.controller.metas
        if not metas:
            return
        t_s = (
            self.start_var.get()
            if self.anchor_var.get() == "start"
            else self.end_var.get()
        )
        for view, (idx, bgr) in self.controller.read_preview_frames(t_s).items():
            slot_key = "side" if view == "side" else "front"
            fps = float(metas[view].fps) or 30.0
            self._slots[slot_key].show_frame(
                f"帧 {idx}（{idx / fps:.2f}s）", bgr, master=self._win
            )

    def _refresh_output_hint(self) -> None:
        if not self.controller.metas:
            self._set_hint("预计保存：请先选择视频与动作名称", "#555555")
            return
        raw_name = self.action_name_var.get()
        if not raw_name.strip():
            self._set_hint("预计保存：请输入动作名称", "#555555")
            return
        try:
            paths = self.controller.plan_paths(raw_name)
        except ValueError as exc:
            self._set_hint(f"动作名称无效：{exc}", "#c62828")
            return
        names = "、".join(p.name for p in paths.values())
        existing = [p.name for p in paths.values() if p.exists()]
        if existing:
            self._set_hint(
                f"预计保存：{names}（提示：已存在同名模板 {'、'.join(existing)}，生成前将询问是否覆盖）",
                "#b35900",
            )
        else:
            self._set_hint(f"预计保存：{names}", "#555555")

    def _set_hint(self, text: str, color: str) -> None:
        self.output_hint_var.set(text)
        self.output_hint_label.configure(foreground=color)

    def _sync_auto_add_visibility(self) -> None:
        if self._on_created is not None:
            self.auto_add_check.pack(side="left", padx=(16, 0))
        else:
            self.auto_add_check.pack_forget()

    def _set_range_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in self._range_widgets:
            widget.configure(state=state)

    def start_generate(self) -> None:
        if self.controller.busy:
            return
        if not self.controller.metas:
            messagebox.showerror(
                "配置错误", "请先选择双摄录制目录或单个视频文件。", parent=self._win
            )
            return
        action_name = self.action_name_var.get()
        try:
            sanitize_action_name(action_name)
            existing = self.controller.existing_targets(action_name)
        except ValueError as exc:
            messagebox.showerror("配置错误", str(exc), parent=self._win)
            return
        if existing:
            names = "\n".join(p.name for p in existing)
            if not messagebox.askyesno(
                "覆盖确认", f"以下同名模板已存在，是否覆盖？\n{names}", parent=self._win
            ):
                return
        self._set_result("")
        self.controller.start_generate(
            action_name=action_name,
            start_s=self.start_var.get(),
            end_s=self.end_var.get(),
            preview=bool(self.preview_var.get()),
            overwrite=True,
        )

    def _on_close(self) -> None:
        self.close()

    def _set_result(self, text: str) -> None:
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)

    # ------------------------------------------------------------------ TemplateBuilderView
    def set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.generate_btn.configure(state=state)
        self.stop_btn.configure(state="normal" if running else "disabled")
        for widget in self._lockable_widgets:
            if widget is self.single_view_combo and not running:
                widget.configure(state="readonly")
            elif widget in self._range_widgets and not running:
                widget.configure(
                    state="normal" if self.controller.metas else "disabled"
                )
            else:
                widget.configure(state=state)
        if not running:
            self._progress.hide()

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def show_progress(self, stage: str, done: int, total: int) -> None:
        self._progress.show(stage, done, total)

    def on_generated(self, results: list[TemplateCreateResult], summary: str) -> None:
        self.status_var.set("模板生成完成")
        self._set_result(summary)
        self._refresh_output_hint()
        if self._on_created is not None and self.auto_add_var.get():
            self._on_created(results)

    def on_cancelled(self) -> None:
        self.status_var.set("已取消")
        self._set_result("已取消模板生成。")

    def on_failed(self, message: str) -> None:
        self.status_var.set("生成失败")
        self._set_result(message)
        messagebox.showerror("模板生成失败", message, parent=self._win)
