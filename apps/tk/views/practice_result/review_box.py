# -*- coding: utf-8 -*-
"""结果页的教师复核操作区：当前复核状态、教师姓名与说明输入，以及"确认问题""撤销问题"按钮。"""
from __future__ import annotations

from collections.abc import Callable
import tkinter as tk
from tkinter import StringVar, ttk

from core.reporting import NormalizedCheck

from apps.tk.views.practice_result.style import FONT, STYLE_A, card, label

_REVIEW_TEXT = {"confirmed": "教师已确认", "revoked": "教师已撤销"}


class ReviewBox:
    """点击按钮时调用 ``on_decide(decision)``，``decision`` 为 ``confirmed`` 或 ``revoked``。"""

    def __init__(self, parent: tk.Misc, on_decide: Callable[[str], None]) -> None:
        bg = STYLE_A["accent_blue_bg"]
        self.frame = card(parent, bg=bg, border=STYLE_A["pending_rule"]["border"], padx=8, pady=6)
        self.frame.pack(fill="x", pady=(0, 4))
        top_row = tk.Frame(self.frame, bg=bg)
        top_row.pack(fill="x")
        self.status_var = StringVar(value="教师复核：未复核")
        label(top_row, bold=True, bg=bg, textvariable=self.status_var).pack(side="left")
        row = tk.Frame(self.frame, bg=bg)
        row.pack(fill="x", pady=(4, 0))
        label(row, size=8, bg=bg, text="教师姓名:").pack(side="left")
        self.teacher_entry = ttk.Entry(row, width=8)
        self.teacher_entry.pack(side="left", padx=4)
        label(row, size=8, bg=bg, text="说明:").pack(side="left")
        self.reason_entry = ttk.Entry(row, width=14)
        self.reason_entry.pack(side="left", padx=4, fill="x", expand=True)
        self.confirm_btn = tk.Button(row, text="确认问题", font=(FONT, 8), bg="#2563EB", fg="#FFFFFF",
                                     relief="flat", padx=6, pady=1, cursor="hand2",
                                     command=lambda: on_decide("confirmed"))
        self.confirm_btn.pack(side="left", padx=2)
        self.revoke_btn = tk.Button(row, text="撤销问题", font=(FONT, 8), bg="#E5E7EB", fg=STYLE_A["text_primary"],
                                    relief="flat", padx=6, pady=1, cursor="hand2",
                                    command=lambda: on_decide("revoked"))
        self.revoke_btn.pack(side="left", padx=2)

    def show(self, check: NormalizedCheck) -> None:
        self.status_var.set(f"教师复核：{_REVIEW_TEXT.get(check.review, '未复核')}")

    def clear(self) -> None:
        self.status_var.set("教师复核：未复核")

    def inputs(self) -> tuple[str, str]:
        """教师姓名（留空为"任课教师"）与说明。"""
        return self.teacher_entry.get().strip() or "任课教师", self.reason_entry.get().strip()
