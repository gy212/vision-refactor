# -*- coding: utf-8 -*-
"""学生练习块（进入练习模式后替换教师操作块）：退出按钮、身份与分析控件、开始/结束/动作评判、状态与点评文字。

练习流程在 ``controllers/practice.py``；本视图只布局，命令与分析回调经 ``connect`` 接上。
"""
from __future__ import annotations

from collections.abc import Callable
from tkinter import StringVar, Text, ttk
from typing import Any

from apps.tk.views.practice_controls import PracticeControls

READY_TEXT = "请选择正/侧双摄后点「开始」"
_INTRO = ("仅指出动作问题，不打分；MediaPipe Lite 本机CPU运行。骨架、手部、自动比对、在线识别由学生练习模式"
          "锁定为关闭，不改你的设置")


class _Hooks:
    """控制器建好之前的占位回调；``connect`` 时替换。"""

    set_busy: Callable[[bool], None] = staticmethod(lambda _busy: None)
    can_analyze: Callable[[], bool] = staticmethod(lambda: False)


class PracticeBlock:
    def __init__(self, parent: Any, *, root: Any, post: Callable[[Callable[[], Any]], Any]) -> None:
        self._hooks = _Hooks()
        block = self.frame = ttk.Labelframe(parent, text="学生练习（动作问题说明）", padding=8)
        self.exit_btn = ttk.Button(block, text="退出学生练习")
        self.exit_btn.pack(fill="x")
        ttk.Label(block, text=_INTRO, foreground="#555555", wraplength=320).pack(anchor="w", pady=(6, 0))
        self.status_var = StringVar(root, value=READY_TEXT)
        self.controls = PracticeControls(
            block, root=root, post=post,
            set_busy=lambda busy: self._hooks.set_busy(busy),
            show_result=self.set_result_text,
            set_status=self.status_var.set,
            can_analyze=lambda: self._hooks.can_analyze(),
        )
        row = ttk.Frame(block)
        row.pack(fill="x", pady=(10, 0))
        self.start_btn = ttk.Button(row, text="开始")
        self.start_btn.pack(side="left", fill="x", expand=True)
        self.end_btn = ttk.Button(row, text="结束", state="disabled")
        self.end_btn.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.judge_btn = ttk.Button(block, text="动作评判", state="disabled")
        self.judge_btn.pack(fill="x", pady=(8, 0))
        ttk.Label(block, textvariable=self.status_var, wraplength=320).pack(anchor="w", pady=(8, 0))
        ttk.Label(block, text="点评结果：").pack(anchor="w", pady=(8, 0))
        frame = ttk.Frame(block)
        frame.pack(fill="both", expand=True, pady=(4, 0))
        self.result_text = Text(frame, height=9, width=36, wrap="word", state="disabled",
                                font=("Microsoft YaHei UI", 9))
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scroll.set)
        self.result_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def connect(self, *, exit: Callable[[], None], start: Callable[[], None], end: Callable[[], None],
                judge: Callable[[], None], set_busy: Callable[[bool], None], can_analyze: Callable[[], bool],
                identity_changed: Callable[[], None], cancelled: Callable[[], None]) -> None:
        self.exit_btn.configure(command=exit)
        self.start_btn.configure(command=start)
        self.end_btn.configure(command=end)
        self.judge_btn.configure(command=judge)
        self._hooks.set_busy = set_busy
        self._hooks.can_analyze = can_analyze
        self.controls.on_identity_changed = identity_changed
        self.controls.on_cancelled = cancelled

    def set_result_text(self, text: str) -> None:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

    def set_buttons(self, *, start: bool, end: bool, judge: bool, exit: bool) -> None:
        for button, enabled in ((self.start_btn, start), (self.end_btn, end), (self.judge_btn, judge),
                                (self.exit_btn, exit)):
            button.configure(state="normal" if enabled else "disabled")
