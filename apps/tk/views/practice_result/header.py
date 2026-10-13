# -*- coding: utf-8 -*-
"""结果页头部：学员与记录信息卡片，以及按七级结论着色的核心结论横幅。"""
from __future__ import annotations

import tkinter as tk
from tkinter import StringVar

from core.reporting import NormalizedReportModel

from apps.tk.views.practice_result.style import STYLE_A, card, label

_TITLE_COLORS = {
    "error": "#B91C1C",
    "warning": "#92400E",
    "info": "#1D4ED8",
    "success": "#166534",
}


class HeaderCard:
    """学号、姓名、动作、实战式、分析时间、记录编号与规则版本。"""

    def __init__(self, parent: tk.Misc) -> None:
        self.frame = card(parent)
        self.frame.pack(fill="x", pady=(0, 8))
        self.meta_var = StringVar(value="尚未加载记录")
        label(self.frame, size=10, textvariable=self.meta_var, anchor="w", justify="left").pack(fill="x")
        self.sub_var = StringVar(value="")
        label(self.frame, size=8, fg=STYLE_A["text_secondary"], textvariable=self.sub_var,
              anchor="w", justify="left").pack(fill="x", pady=(2, 0))

    def show(self, model: NormalizedReportModel) -> None:
        header = model.header
        self.meta_var.set(
            f"学号：{header.student_id} · 姓名：{header.student_name or '未登记'} · "
            f"动作：{header.action_label}（{header.stance_label}） · "
            f"时间：{header.created_at_local} · 编号：{header.record_id[:8]}…"
        )
        self.sub_var.set(
            f"算法规则：{header.rule_version} · 结果版本：rev{header.revision} · {header.backend_description}"
        )

    def clear(self) -> None:
        self.meta_var.set("尚未加载记录")
        self.sub_var.set("")


class HeadlineBanner:
    """核心结论横幅：主标题按结论级别着色，副标题给出引导说明。"""

    def __init__(self, parent: tk.Misc) -> None:
        self.frame = card(parent)
        self.frame.pack(fill="x", pady=(0, 8))
        self.title_var = StringVar(value="等待分析记录…")
        self.title_label = label(self.frame, size=12, bold=True, textvariable=self.title_var,
                                 anchor="w", justify="left")
        self.title_label.pack(fill="x")
        self.sub_var = StringVar(value="")
        label(self.frame, fg=STYLE_A["text_secondary"], textvariable=self.sub_var,
              anchor="w", justify="left", wraplength=850).pack(fill="x", pady=(3, 0))

    def show(self, model: NormalizedReportModel) -> None:
        self.title_var.set(model.headline.title)
        self.sub_var.set(model.headline.subtitle)
        self.title_label.configure(fg=_TITLE_COLORS.get(model.headline.level, STYLE_A["text_primary"]))

    def show_error(self, exc: Exception) -> None:
        self.title_var.set("记录解析错误")
        self.sub_var.set(f"无法正确解析分析记录: {exc}")

    def clear(self) -> None:
        self.title_var.set("等待分析记录…")
        self.title_label.configure(fg=STYLE_A["text_primary"])
        self.sub_var.set("")
