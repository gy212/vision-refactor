# -*- coding: utf-8 -*-
"""结果页底部操作栏：状态提示，以及"个人历史""导出图文报告""再练一次"三个按钮。"""
from __future__ import annotations

from collections.abc import Callable
import tkinter as tk
from tkinter import StringVar

from apps.tk.views.practice_result.style import FONT, STYLE_A, card, label


class ActionBar:
    """三个按钮分别调用 ``on_history``、``on_export``、``on_practice_again``。"""

    def __init__(self, parent: tk.Misc, *, on_history: Callable[[], None], on_export: Callable[[], None],
                 on_practice_again: Callable[[], None]) -> None:
        self.frame = card(parent)
        self.status_var = StringVar(value="")
        label(self.frame, size=8, fg=STYLE_A["text_secondary"], textvariable=self.status_var).pack(side="left")
        btn_box = tk.Frame(self.frame, bg=STYLE_A["bg_card"])
        btn_box.pack(side="right")
        self.history_btn = tk.Button(btn_box, text="个人历史", font=(FONT, 9), bg=STYLE_A["bg_card"],
                                     fg=STYLE_A["text_primary"], relief="solid", bd=1, padx=10, pady=3,
                                     cursor="hand2", command=on_history)
        self.history_btn.pack(side="left", padx=4)
        self.export_btn = tk.Button(btn_box, text="导出图文报告", font=(FONT, 9), bg=STYLE_A["bg_card"],
                                    fg=STYLE_A["accent_blue"], relief="solid", bd=1, padx=10, pady=3,
                                    cursor="hand2", command=on_export)
        self.export_btn.pack(side="left", padx=4)
        self.practice_again_btn = tk.Button(btn_box, text="再练一次", font=(FONT, 9, "bold"), bg=STYLE_A["accent_blue"],
                                            fg="#FFFFFF", activebackground="#1D4ED8", activeforeground="#FFFFFF",
                                            relief="flat", padx=14, pady=3, cursor="hand2", command=on_practice_again)
        self.practice_again_btn.pack(side="left", padx=(4, 0))
