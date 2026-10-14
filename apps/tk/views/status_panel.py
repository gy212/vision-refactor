# -*- coding: utf-8 -*-
"""状态区（控制区底部）：会话状态、录制状态与路径、双摄自动比对结果、识别结果、实时动作识别开关与进度条。

教师附加区（比对、识别、进度）在学生练习模式里整体隐藏，由 ``show_teacher_extra`` 切换。
"""
from __future__ import annotations

from tkinter import BooleanVar, StringVar, ttk
from typing import Any

EMPTY_SCORES = "正面：-　侧面：-　综合：-"


class StatusPanel:
    def __init__(self, parent: Any, *, root: Any) -> None:
        self.status_var = StringVar(root, value="就绪")
        self.recording_status_var = StringVar(root, value="")
        self.compare_segment_var = StringVar(root, value="片段：-")
        self.compare_status_var = StringVar(root, value="自动比对：待机")
        self.compare_score_var = StringVar(root, value=EMPTY_SCORES)
        self.compare_error_var = StringVar(root, value="")
        self.actions_var = StringVar(root, value="-")
        self.match_var = StringVar(root, value="识别：待机")
        self.online_match_var = BooleanVar(root, value=True)
        self.progress_text_var = StringVar(root, value="")
        block = self.frame = ttk.Labelframe(parent, text="状态", padding=10)
        block.pack(fill="x")
        ttk.Label(block, textvariable=self.status_var, wraplength=320).pack(anchor="w")
        ttk.Label(block, textvariable=self.recording_status_var, wraplength=320).pack(anchor="w", pady=(4, 0))
        extra = self.teacher_extra = ttk.Frame(block)
        extra.pack(fill="x")
        ttk.Separator(extra, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(extra, text="双摄自动比对：").pack(anchor="w")
        for var in (self.compare_segment_var, self.compare_status_var, self.compare_score_var,
                    self.compare_error_var):
            ttk.Label(extra, textvariable=var, wraplength=320).pack(anchor="w", pady=(4, 0))
        ttk.Label(extra, text="识别结果：").pack(anchor="w", pady=(8, 0))
        ttk.Label(extra, textvariable=self.actions_var, wraplength=320).pack(anchor="w")
        match_row = ttk.Frame(extra)
        match_row.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(match_row, text="实时动作识别", variable=self.online_match_var).pack(side="left")
        ttk.Label(match_row, textvariable=self.match_var, wraplength=210).pack(side="left", padx=(8, 0))
        ttk.Label(extra, textvariable=self.progress_text_var, wraplength=320).pack(anchor="w", pady=(8, 0))
        self.progress_bar = ttk.Progressbar(extra, orient="horizontal", mode="determinate", maximum=100.0)
        self.progress_bar.pack(fill="x", pady=(6, 0))
        ttk.Label(extra, text="提示：点击“停止”结束识别。").pack(anchor="w", pady=(8, 0))

    def show_teacher_extra(self, visible: bool) -> None:
        if visible:
            self.teacher_extra.pack(fill="x")
        else:
            self.teacher_extra.pack_forget()

    def set_progress(self, done: int, total: int) -> None:
        """视频文件显示进度；摄像头没有总帧数，不显示进度。"""
        if total > 0:
            self.progress_bar.configure(mode="determinate", maximum=float(total))
            self.progress_bar["value"] = float(done)
            self.progress_text_var.set(f"进度：{done}/{total}（{(done / total) * 100.0:.1f}%）")
        else:
            self.progress_bar.configure(mode="determinate", maximum=100.0)
            self.progress_bar["value"] = 0.0
            self.progress_text_var.set("")

    def reset_for_start(self) -> None:
        self.actions_var.set("-")
        self.match_var.set("识别：待机")
        self.progress_text_var.set("")
        self.progress_bar.configure(mode="determinate", maximum=100.0, value=0.0)

    def show_compare(self, segment: str, status: str, scores: str | None = EMPTY_SCORES, error: str = "") -> None:
        """刷新比对状态栏；``scores`` 为 None 时保留原分数。"""
        self.compare_segment_var.set(f"片段：{segment}")
        self.compare_status_var.set(status)
        if scores is not None:
            self.compare_score_var.set(scores)
        self.compare_error_var.set(error)
