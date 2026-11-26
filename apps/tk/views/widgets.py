# -*- coding: utf-8 -*-
"""跨视图复用的小控件：可折叠分区 ``CollapsibleSection``、进度条组 ``ProgressPanel`` 与视频帧缩略图槽 ``FramePreviewSlot``。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from tkinter import BooleanVar, DoubleVar, Scrollbar, StringVar, Text, ttk
from typing import Any

import cv2
from PIL import Image, ImageTk

from core.media import VIDEO_SUFFIXES

from apps.tk.controllers.template_builder import (
    format_range_summary,
    min_duration_seconds,
    min_step_seconds,
    nudge_time,
)


def video_filetypes() -> list[tuple[str, str]]:
    """返回符合 Tk 规范的空格分隔视频文件过滤器，与 core.media.VIDEO_SUFFIXES 同步。"""
    patterns = " ".join(f"*{suffix}" for suffix in sorted(VIDEO_SUFFIXES))
    return [("视频文件", patterns), ("所有文件", "*.*")]


def _button_width(title: str) -> int:
    """按全角 2 栏、半角 1 栏估算 ttk.Button 宽度，避免中文标题被截断。"""
    display = sum(2 if ord(ch) > 127 else 1 for ch in title)
    return max(12, display + 4)


class CollapsibleSection:
    """带切换按钮的折叠分区：``header`` 放切换按钮，``content`` 放分区内容。"""

    def __init__(self, parent: ttk.Frame, title: str, expanded: bool = False) -> None:
        self.parent = parent
        self._title = title
        self._expanded = BooleanVar(value=expanded)

        self.header = ttk.Frame(parent)
        self._toggle_btn = ttk.Button(
            self.header,
            text=self._toggle_text(),
            command=self._toggle,
            width=_button_width(title),
        )
        self._toggle_btn.pack(side="left", anchor="w")

        self.content = ttk.Frame(parent)
        if expanded:
            self.content.pack(fill="x", pady=(4, 0))

    def _toggle_text(self) -> str:
        arrow = "▾" if self._expanded.get() else "▸"
        return f"{arrow} {self._title}"

    def _toggle(self) -> None:
        self._expanded.set(not self._expanded.get())
        self._toggle_btn.configure(text=self._toggle_text())

        if self._expanded.get():
            self.content.pack(fill="x", pady=(4, 0))
        else:
            self.content.pack_forget()

    def toggle(self) -> None:
        """展开或收起。"""
        self._toggle()

    def set_title(self, title: str) -> None:
        self._title = title
        self._toggle_btn.configure(text=self._toggle_text(), width=_button_width(title))

    def pack_header(self, **kwargs) -> None:
        self.header.pack(**kwargs)

    def is_expanded(self) -> bool:
        return self._expanded.get()


class ProgressPanel:
    """可隐藏的 determinate 进度条 + 状态文案，供比对与制模弹窗共用。"""

    def __init__(self, parent: ttk.Frame, text_var: StringVar) -> None:
        self.frame = ttk.Frame(parent)
        self.text_var = text_var
        self.bar = ttk.Progressbar(
            self.frame, orient="horizontal", mode="determinate", maximum=100.0
        )
        self.bar.pack(fill="x")
        ttk.Label(self.frame, textvariable=self.text_var).pack(anchor="w", pady=(4, 0))
        self.hide()

    def show(
        self, stage: str, done: int, total: int, *, pady: tuple[int, int] = (8, 0)
    ) -> None:
        self.frame.pack(fill="x", pady=pady)
        self.bar.configure(
            mode="determinate", maximum=float(total) if total > 0 else 100.0
        )
        if total > 0:
            self.bar["value"] = float(done)
            self.text_var.set(
                f"{stage}：{done}/{total}（{(done / total) * 100.0:.1f}%）"
            )
        else:
            self.bar["value"] = 0.0
            self.text_var.set(f"{stage}…")

    def hide(self) -> None:
        self.frame.pack_forget()


class FramePreviewSlot:
    """单路视频帧缩略图框（保持 PhotoImage 引用防 GC）。"""

    def __init__(
        self,
        parent: ttk.Frame,
        title: str,
        column: int,
        *,
        preview_size: tuple[int, int] = (340, 200),
    ) -> None:
        self._size = preview_size
        self.frame = ttk.Labelframe(parent, text=title, padding=6)
        self.frame.grid(
            row=0, column=column, sticky="nsew", padx=(0, 6) if column == 0 else (6, 0)
        )
        self.title_var = StringVar(value="未加载")
        ttk.Label(self.frame, textvariable=self.title_var, foreground="#555555").pack(
            anchor="w"
        )
        self.image_label = ttk.Label(self.frame, text="暂无画面", anchor="center")
        self.image_label.pack(fill="both", expand=True, pady=(4, 0))
        self.photo: ImageTk.PhotoImage | None = None

    def show_frame(self, title: str, bgr: Any, *, master: Any) -> None:
        self.title_var.set(title)
        if bgr is None:
            self.photo = None
            self.image_label.configure(image="", text="无法解码当前帧")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail(self._size)
        photo = ImageTk.PhotoImage(pil_img, master=master)
        self.photo = photo
        self.image_label.configure(image=photo, text="")

    def clear(self, title: str = "未加载") -> None:
        self.title_var.set(title)
        self.photo = None
        self.image_label.configure(image="", text="暂无画面")


class DualPreviewPanel:
    """双摄/单摄并排帧预览区：按已加载视角自动显示双框或单框并渲染缩略图。"""

    def __init__(self, parent: ttk.Frame) -> None:
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="x")
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)
        self.slots = {
            "front": FramePreviewSlot(self.frame, "正面画面", 0),
            "side": FramePreviewSlot(self.frame, "侧面画面", 1),
        }

    def sync_layout(self, metas: dict[str, Any]) -> None:
        if not metas:
            for slot in self.slots.values():
                slot.frame.grid()
                slot.clear("未加载")
            return
        if "front" in metas and "side" in metas:
            self.slots["front"].frame.configure(text="正面画面")
            self.slots["front"].frame.grid()
            self.slots["side"].frame.configure(text="侧面画面")
            self.slots["side"].frame.grid()
            return
        active = next(iter(metas))
        hide_key, show_key = (
            ("front", "side") if active == "side" else ("side", "front")
        )
        title = {"side": "侧面画面", "front": "正面画面"}.get(active, "视频画面")
        self.slots[hide_key].frame.grid_remove()
        self.slots[show_key].frame.configure(text=title)
        self.slots[show_key].frame.grid()

    def show_frames(
        self, frames: dict[str, tuple[int, Any]], metas: dict[str, Any], *, master: Any
    ) -> None:
        for view, (idx, bgr) in frames.items():
            slot_key = "side" if view == "side" else "front"
            fps = float(metas[view].fps) or 30.0
            self.slots[slot_key].show_frame(
                f"帧 {idx}（{idx / fps:.2f}s）", bgr, master=master
            )


def build_range_row(
    parent: ttk.Frame,
    label: str,
    var: DoubleVar,
    time_var: StringVar,
    *,
    deltas: Sequence[int],
    on_slide: Callable[[str], None],
    on_nudge: Callable[[int], None],
) -> tuple[ttk.Scale, dict[int, ttk.Button], list[Any]]:
    """构造带时间读数与微调帧按钮的一行区间滑块，返回 (scale, nudge_buttons, widgets)。"""
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=(6, 0))
    ttk.Label(row, text=label, width=10).pack(side="left")
    scale = ttk.Scale(
        row, from_=0.0, to=1.0, orient="horizontal", variable=var, command=on_slide
    )
    scale.pack(side="left", fill="x", expand=True, padx=(4, 8))
    ttk.Label(row, textvariable=time_var, width=8, anchor="e").pack(
        side="left", padx=(0, 6)
    )
    widgets: list[Any] = [scale]
    buttons: dict[int, ttk.Button] = {}
    for delta in deltas:
        btn = ttk.Button(
            row, text=f"{delta:+d}帧", width=5, command=lambda d=delta: on_nudge(d)
        )
        btn.pack(side="left", padx=(2, 0))
        buttons[delta] = btn
        widgets.append(btn)
    return scale, buttons, widgets


def build_scrolled_text(
    parent: ttk.Frame, *, height: int = 5, wrap: str = "word"
) -> Text:
    """带垂直滚动条的只写/结果文本框。"""
    box = ttk.Frame(parent)
    box.pack(fill="both", expand=True, pady=(4, 0))
    widget = Text(box, height=height, wrap=wrap)
    widget.pack(side="left", fill="both", expand=True)
    sb = Scrollbar(box, command=widget.yview)
    sb.pack(side="right", fill="y")
    widget.configure(yscrollcommand=sb.set)
    return widget


class SyncRangePanel:
    """同步起止时间滑块、预览跟随单选与逐帧微调控件组。"""

    def __init__(
        self,
        parent: ttk.Frame,
        *,
        deltas: Sequence[int],
        get_metas: Callable[[], dict[str, Any]],
        on_preview: Callable[[bool], None],
    ) -> None:
        self._get_metas = get_metas
        self._on_preview = on_preview
        self._max_s = 0.0
        self._suppress = False
        self.anchor_var = StringVar(value="start")
        self.start_var, self.end_var = DoubleVar(value=0.0), DoubleVar(value=0.0)
        self.start_time_var, self.end_time_var = (
            StringVar(value="0.00s"),
            StringVar(value="0.00s"),
        )
        self.range_info_var = StringVar(value="已选区间：—")
        self.nudge_buttons: dict[tuple[str, int], ttk.Button] = {}
        self.widgets: list[Any] = []

        anchor_row = ttk.Frame(parent)
        anchor_row.pack(fill="x", pady=(8, 0))
        ttk.Label(anchor_row, text="预览跟随：").pack(side="left")
        for val, text in (("start", "起始帧"), ("end", "结束帧")):
            rb = ttk.Radiobutton(
                anchor_row,
                text=text,
                value=val,
                variable=self.anchor_var,
                command=lambda: self._on_preview(False),
            )
            rb.pack(side="left", padx=(4, 12) if val == "start" else (0, 0))
            self.widgets.append(rb)

        self.start_scale = self._add_row(
            parent, "起始位置：", self.start_var, self.start_time_var, "start", deltas
        )
        self.end_scale = self._add_row(
            parent, "结束位置：", self.end_var, self.end_time_var, "end", deltas
        )
        ttk.Label(parent, textvariable=self.range_info_var, foreground="#222222").pack(
            anchor="w", pady=(6, 0)
        )

    def _add_row(
        self,
        parent: ttk.Frame,
        label: str,
        var: DoubleVar,
        time_var: StringVar,
        which: str,
        deltas: Sequence[int],
    ) -> ttk.Scale:
        scale, btns, widgets = build_range_row(
            parent,
            label,
            var,
            time_var,
            deltas=deltas,
            on_slide=lambda val, w=which: self._on_slide(w, val),
            on_nudge=lambda d, w=which: self._nudge(w, d),
        )
        for delta, btn in btns.items():
            self.nudge_buttons[(which, delta)] = btn
        self.widgets.extend(widgets)
        return scale

    def reset_for_metas(self, metas: dict[str, Any]) -> None:
        self._max_s = max(0.0, min_duration_seconds(metas))
        self._suppress = True
        try:
            upper = max(self._max_s, 1e-3)
            self.start_scale.configure(from_=0.0, to=upper)
            self.end_scale.configure(from_=0.0, to=upper)
            self.start_var.set(0.0)
            self.end_var.set(self._max_s)
        finally:
            self._suppress = False
        self.set_enabled(bool(metas))
        self.sync_labels()

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in self.widgets:
            widget.configure(state=state)

    def sync_labels(self) -> None:
        s, e = self.start_var.get(), self.end_var.get()
        self.start_time_var.set(f"{s:.2f}s")
        self.end_time_var.set(f"{e:.2f}s")
        self.range_info_var.set(format_range_summary(self._get_metas(), s, e))

    def active_time(self) -> float:
        return (
            self.start_var.get()
            if self.anchor_var.get() == "start"
            else self.end_var.get()
        )

    def _on_slide(self, which: str, value: str) -> None:
        metas = self._get_metas()
        if self._suppress or not metas:
            return
        step, raw = min_step_seconds(metas), float(value)
        self._suppress = True
        try:
            if which == "start":
                self.start_var.set(
                    min(max(0.0, raw), max(0.0, self.end_var.get() - step))
                )
            else:
                self.end_var.set(
                    max(
                        min(self._max_s, raw),
                        min(self._max_s, self.start_var.get() + step),
                    )
                )
        finally:
            self._suppress = False
        self.anchor_var.set(which)
        self.sync_labels()
        self._on_preview(True)

    def _nudge(self, which: str, delta_frames: int) -> None:
        metas = self._get_metas()
        if not metas:
            return
        fps, step = next(iter(metas.values())).fps, min_step_seconds(metas)
        self._suppress = True
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
            self._suppress = False
        self.sync_labels()
        self._on_preview(False)
