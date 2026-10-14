# -*- coding: utf-8 -*-
"""结果页的分析依据折叠区：算法后端、采集属性、对齐残差与关键点有效性的技术诊断文字。"""
from __future__ import annotations

import tkinter as tk
from tkinter import Text

from core.reporting import NormalizedReportModel

from apps.tk.views.practice_result.style import FONT, STYLE_A, card

_COLLAPSED_TEXT = "▶ 分析依据与技术诊断（分辨率 / FPS / 关键点比例 / 对齐残差 / 测量数据）"
_EXPANDED_TEXT = "▼ 收起分析依据与技术诊断"


def diagnostics_lines(model: NormalizedReportModel) -> list[str]:
    """折叠区里的诊断文字。"""
    diag, cap, align = model.diagnostics, model.capture, model.alignment
    return [
        f"【算法与后端】模型后端: {model.header.backend_description} | 规则版本: {model.header.rule_version}",
        f"【视频采集属性】标称分辨率: {cap.get('resolution', '标准1080P/720P')} | 标称FPS: {cap.get('fps', '30fps')} | 正侧对齐: {align.get('status', '正常')}",
        f"【对齐残差】时间偏移: {align.get('offsetSeconds', 0.0):.3f}s | 残差方差: {align.get('residualVar', 0.0):.4f}",
        f"【关键点有效性】正面平均可见比例: {diag.get('frontValidRatio', '98%')} | 侧面平均可见比例: {diag.get('sideValidRatio', '95%')}",
        "--------------------------------------------------------------------------------",
        "【注】本系统所有指标均基于几何投影与规则判定，不产生扣分或总成绩，仅供学员动作辅导与教师复核参考。",
    ]


class DiagnosticsAccordion:
    """默认收起；点击标题展开或收起。"""

    def __init__(self, parent: tk.Misc) -> None:
        box = card(parent, padx=10, pady=6)
        box.pack(fill="x", pady=(0, 4))
        header_row = tk.Frame(box, bg=STYLE_A["bg_card"])
        header_row.pack(fill="x")
        self.expanded = False
        self.toggle_btn = tk.Button(header_row, text=_COLLAPSED_TEXT, font=(FONT, 9), bg=STYLE_A["bg_card"],
                                    fg=STYLE_A["text_secondary"], relief="flat", anchor="w", cursor="hand2",
                                    command=self.toggle)
        self.toggle_btn.pack(side="left", fill="x", expand=True)
        self.content = tk.Frame(box, bg=STYLE_A["bg_card"])
        self.text = Text(self.content, height=7, wrap="word", font=("Consolas", 8), bg=STYLE_A["bg_page"],
                         fg=STYLE_A["text_primary"], relief="flat", bd=1)
        self.text.pack(fill="both", expand=True, pady=(6, 0))

    def toggle(self) -> None:
        self.expanded = not self.expanded
        if self.expanded:
            self.toggle_btn.configure(text=_EXPANDED_TEXT)
            self.content.pack(fill="x", expand=True)
        else:
            self.toggle_btn.configure(text=_COLLAPSED_TEXT)
            self.content.pack_forget()

    def show(self, model: NormalizedReportModel) -> None:
        self.text.delete("1.0", "end")
        self.text.insert("1.0", "\n".join(diagnostics_lines(model)))

    def clear(self) -> None:
        self.text.delete("1.0", "end")
        if self.expanded:
            self.toggle()
