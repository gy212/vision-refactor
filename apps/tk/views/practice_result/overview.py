# -*- coding: utf-8 -*-
"""结果页的四态概览卡片与"教师已撤销"计数标签；计数严格守恒，不展示任何分数或进度。"""
from __future__ import annotations

import tkinter as tk
from tkinter import StringVar

from PIL import ImageTk

from core.reporting import NormalizedReportModel

from apps.tk.views.practice_result.style import STYLE_A, card, label

_CARDS = [
    ("candidate", "发现疑似问题"),
    ("not_observed", "已检查未发现"),
    ("unable", "暂无法判断"),
    ("pending_rule", "需教师判断"),
]


class OverviewCards:
    """四个状态卡片（图标、标题、数量）与撤销项标签。"""

    def __init__(self, parent: tk.Misc, icons: dict[str, ImageTk.PhotoImage]) -> None:
        self.frame = tk.Frame(parent, bg=STYLE_A["bg_page"])
        self.frame.pack(fill="x", pady=(0, 8))
        self.count_labels: dict[str, tk.Label] = {}
        for i in range(len(_CARDS)):
            self.frame.columnconfigure(i, weight=1, uniform="card")
        for col, (status, title) in enumerate(_CARDS):
            self._build_card(col, status, title, icons)
        self.revoked_var = StringVar(value="教师已撤销 0 项")
        label(self.frame, size=8, bg=STYLE_A["bg_page"], fg=STYLE_A["text_secondary"],
              textvariable=self.revoked_var).grid(row=1, column=0, columnspan=4, sticky="e", pady=(4, 0))

    def _build_card(self, col: int, status: str, title: str, icons: dict[str, ImageTk.PhotoImage]) -> None:
        token = STYLE_A[status]
        box = card(self.frame, bg=token["bg"], border=token["border"], padx=8, pady=6)
        box.grid(row=0, column=col, sticky="ew", padx=3)
        top_row = tk.Frame(box, bg=token["bg"])
        top_row.pack(fill="x")
        if status in icons:
            tk.Label(top_row, image=icons[status], bg=token["bg"]).pack(side="left")
        label(top_row, bg=token["bg"], fg=token["fg"], text=title).pack(side="left", padx=4)
        count = label(box, size=13, bold=True, bg=token["bg"], fg=token["fg"], text="0 项")
        count.pack(anchor="w", pady=(2, 0))
        self.count_labels[status] = count

    def show(self, model: NormalizedReportModel) -> None:
        counts = model.counts
        self.count_labels["candidate"].configure(text=f"{counts.candidate_count} 项")
        self.count_labels["not_observed"].configure(text=f"{counts.not_observed_count} 项")
        self.count_labels["unable"].configure(text=f"{counts.unable_count} 项")
        self.count_labels["pending_rule"].configure(text=f"{counts.pending_rule_count} 项")
        self.revoked_var.set(f"教师已撤销 {counts.revoked_count} 项（不计入未发现）")

    def clear(self) -> None:
        for count in self.count_labels.values():
            count.configure(text="0 项")
        self.revoked_var.set("教师已撤销 0 项")
