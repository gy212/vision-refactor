# -*- coding: utf-8 -*-
"""结果页的动作阶段导航栏：按报告的相位时间线动态生成"全部"与各阶段按钮，有疑似问题的阶段带星标。"""
from __future__ import annotations

from collections.abc import Callable
import tkinter as tk

from core.reporting import PhaseTimelineItem

from apps.tk.views.practice_result.style import FONT, STYLE_A, label, toggle_colors


class PhaseBar:
    """点击阶段按钮时调用 ``on_select(phase)``；``None`` 表示全部阶段。"""

    def __init__(self, parent: tk.Misc, on_select: Callable[[str | None], None]) -> None:
        self._on_select = on_select
        self.frame = tk.Frame(parent, bg=STYLE_A["bg_page"])
        self.frame.pack(fill="x", pady=(0, 8))
        label(self.frame, bold=True, bg=STYLE_A["bg_page"], fg=STYLE_A["text_secondary"],
              text="动作阶段：").pack(side="left")
        self.buttons_frame = tk.Frame(self.frame, bg=STYLE_A["bg_page"])
        self.buttons_frame.pack(side="left", fill="x", expand=True)

    def _button(self, text: str, active: bool, phase: str | None) -> None:
        tk.Button(
            self.buttons_frame,
            text=text,
            font=(FONT, 8),
            relief="solid",
            bd=1,
            padx=6,
            pady=1,
            cursor="hand2",
            command=lambda: self._on_select(phase),
            **toggle_colors(active),
        ).pack(side="left", padx=2)

    def render(self, phases: list[PhaseTimelineItem], total_checks: int, selected: str | None) -> None:
        """重建按钮：当前选中的阶段高亮。"""
        self.clear()
        self._button(f"全部 ({total_checks})", selected is None, None)
        for phase in phases:
            star = " ★" if phase.candidate_count > 0 else ""
            self._button(f"{phase.title}{star}", selected == phase.phase, phase.phase)

    def clear(self) -> None:
        for child in self.buttons_frame.winfo_children():
            child.destroy()
