# -*- coding: utf-8 -*-
"""主窗口布局：左侧可滚动控制区（主要操作 → 分隔线 → 次要选项 → 状态）与右侧预览，以及跨两列的练习结果页。

主要操作区自上而下是摄像头、预览模型、开始/停止、录制、动作分析五项核心控件；学生练习模式隐藏教师块、
次要选项与状态区的教师附加区，换上练习块（``show_practice``）。练习结果页显示时整体替换控制区与预览。
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from tkinter import Canvas, ttk
from typing import Any

from apps.tk.views.capture_panel import CapturePanel
from apps.tk.views.practice_block import PracticeBlock
from apps.tk.views.practice_result import FeedbackResultPanel
from apps.tk.views.preview_surface import PreviewSurface
from apps.tk.views.recording_panel import RecordingPanel
from apps.tk.views.status_panel import StatusPanel

TEACHER_TITLE = "MediaPipe 动作识别（人体姿态 + 手部）"
PRACTICE_TITLE = "学生练习（动作问题说明）"


class MainWindow:
    def __init__(self, root: Any, *, post: Callable[[Callable[[], Any]], Any], record_dir: Path,
                 templates: tuple[Iterable[Path], Iterable[Path]], history_root: Path | None = None) -> None:
        self.root = root
        root.title(TEACHER_TITLE)
        root.geometry("1100x720")
        root.minsize(800, 600)   # 控制区可滚动，限制最小尺寸保证滚动条与预览区可用
        try:
            ttk.Style().theme_use("clam")
        except Exception:
            pass
        outer = self.outer = ttk.Frame(root, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=0)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)
        left = self._build_scroll_area(outer)

        primary = self.primary = ttk.Labelframe(left, text="主要操作", padding=10)
        primary.pack(fill="x", pady=(0, 10))
        self.capture = CapturePanel(root)
        self.capture.build_cameras(primary)
        teacher = self.teacher_block = ttk.Frame(primary)
        teacher.pack(fill="x")
        self.capture.build_session(teacher)
        front, side = templates
        self.recording = RecordingPanel(teacher, root=root, record_dir=record_dir, front=front, side=side)
        self.compare_btn = ttk.Button(teacher, text="动作分析…")
        self.compare_btn.pack(fill="x", pady=(10, 0))
        self.exam_btn = ttk.Button(teacher, text="考试模式…")
        self.exam_btn.pack(fill="x", pady=(6, 0))
        self.student_practice_btn = ttk.Button(teacher, text="学生练习…")
        self.student_practice_btn.pack(fill="x", pady=(6, 0))
        self.practice = PracticeBlock(primary, root=root, post=post)   # 默认不显示

        # 主要操作与次要选项之间的显式分隔线
        self.primary_secondary_separator = ttk.Separator(left, orient="horizontal")
        self.primary_secondary_separator.pack(fill="x", pady=8)
        self.secondary_block = ttk.Labelframe(left, text="次要选项", padding=10)
        self.secondary_block.pack(fill="x", pady=(0, 10))
        self.capture.build_secondary(self.secondary_block)
        self.status = StatusPanel(left, root=root)

        self.preview = PreviewSurface(outer)
        self.preview.frame.grid(row=0, column=1, sticky="nsew")
        # 「返回录制预览」只切视图、不重置片段；处理函数在 connect 时注入
        self._back_to_preview: Callable[[], None] = self.show_preview_page
        self.result_panel = FeedbackResultPanel(outer, history_root=history_root,
                                                on_switch_to_preview=lambda: self._back_to_preview())
        self.result_panel.grid_remove()

    def _build_scroll_area(self, outer: Any) -> Any:
        """左列：Canvas + 垂直滚动条 + 内嵌 Frame；指针在控制区内时鼠标滚轮滚动。"""
        container = self.left_container = ttk.Frame(outer)
        container.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        container.rowconfigure(0, weight=1)
        canvas = self.left_canvas = Canvas(container, width=360, highlightthickness=0, borderwidth=0)
        canvas.grid(row=0, column=0, sticky="ns")
        scroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scroll.set)
        inner = self.controls_inner = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))

        def wheel(event: Any) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        for widget in (canvas, inner):
            widget.bind("<Enter>", lambda _event: canvas.bind_all("<MouseWheel>", wheel))
            widget.bind("<Leave>", lambda _event: canvas.unbind_all("<MouseWheel>"))
        return inner

    def connect(self, *, compare: Callable[[], None], exam: Callable[[], None],
                practice: Callable[[], None], back_to_preview: Callable[[], None]) -> None:
        self._back_to_preview = back_to_preview
        self.compare_btn.configure(command=compare)
        self.exam_btn.configure(command=exam)
        self.student_practice_btn.configure(command=practice)

    def show_practice(self, active: bool) -> None:
        """学生练习模式：隐藏教师块、分隔线、次要选项与状态区教师附加区，换上练习块；退出时按原顺序恢复。"""
        if active:
            for block in (self.teacher_block, self.primary_secondary_separator, self.secondary_block):
                block.pack_forget()
            self.status.show_teacher_extra(False)
            self.capture.settings_btn.configure(state="disabled")
            self.practice.frame.pack(fill="both", expand=True, pady=(10, 0))
            self.root.title(PRACTICE_TITLE)
            return
        self.practice.frame.pack_forget()
        self.teacher_block.pack(fill="x")
        self.primary_secondary_separator.pack(fill="x", pady=8, before=self.status.frame)
        self.secondary_block.pack(fill="x", pady=(0, 10), before=self.status.frame)
        self.status.show_teacher_extra(True)
        self.capture.settings_btn.configure(state="normal")
        self.root.title(TEACHER_TITLE)

    def show_result_page(self, record: dict) -> None:
        """练习结果页：隐藏控制区和预览，结果页跨两列显示。"""
        self.left_container.grid_remove()
        self.preview.frame.grid_remove()
        self.result_panel.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.result_panel.set_record(record)

    def show_preview_page(self) -> None:
        self.result_panel.grid_remove()
        self.left_container.grid()
        self.preview.frame.grid()
