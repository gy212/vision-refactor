# -*- coding: utf-8 -*-
"""采集与会话控件：设置按钮、正/侧摄像头下拉与刷新、预览姿态模型、开始/停止，以及次要选项（输入源提示、线程数、手部检测）。

控件分属主窗口的三个区域（摄像头行、教师操作块、次要选项），由主窗口按顺序调用三个 ``build_*``；
命令在控制器建好后经 ``connect`` 接上。
"""
from __future__ import annotations

from collections.abc import Callable
from tkinter import BooleanVar, IntVar, StringVar, ttk
from typing import Any

from apps.tk.controllers.session import default_workers

NO_SECOND_CAMERA = "无（单摄像头）"   # 第二摄像头下拉的「不选」项（双摄双面视图，issue #57）
POSE_VARIANTS = ("lite", "full", "heavy")


class CapturePanel:
    def __init__(self, root: Any) -> None:
        self.camera_choice = StringVar(root, value="")
        self.camera_choice_2 = StringVar(root, value=NO_SECOND_CAMERA)
        self.source_hint_var = StringVar(root, value="当前输入源：未选择")
        self.pose_var = StringVar(root, value="full")
        self.workers_var = IntVar(root, value=default_workers())
        self.enable_hands_var = BooleanVar(root, value=True)

    def build_cameras(self, parent: Any) -> None:
        """右上角设置按钮与两路摄像头选择（主要操作区顶部）。"""
        top_row = ttk.Frame(parent)
        top_row.pack(fill="x")
        self.settings_btn = ttk.Button(top_row, text="⚙ 设置", width=8)
        self.settings_btn.pack(side="right")
        ttk.Label(parent, text="第一摄像头（正面）：").pack(anchor="w")
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(6, 0))
        self.camera_combo = ttk.Combobox(row, textvariable=self.camera_choice, state="readonly")
        self.camera_combo.pack(side="left", fill="x", expand=True)
        self.refresh_btn = ttk.Button(row, text="刷新")
        self.refresh_btn.pack(side="left", padx=(8, 0))
        # 画面旋转不用下拉：运行中点击对应预览窗顺时针 +90°（两路隔离）。
        ttk.Label(parent, text="第二摄像头（侧面，可选）：").pack(anchor="w", pady=(10, 0))
        row_2 = ttk.Frame(parent)
        row_2.pack(fill="x", pady=(6, 0))
        self.camera_combo_2 = ttk.Combobox(row_2, textvariable=self.camera_choice_2, values=[NO_SECOND_CAMERA],
                                           state="readonly")
        self.camera_combo_2.pack(side="left", fill="x", expand=True)

    def build_session(self, parent: Any) -> None:
        """预览姿态模型与开始/停止（教师操作块）。黑盒评分固定 heavy，模型只影响实时预览。"""
        ttk.Label(parent, text="预览姿态模型：").pack(anchor="w", pady=(10, 0))
        self.model_combo = ttk.Combobox(parent, textvariable=self.pose_var, values=list(POSE_VARIANTS),
                                        state="readonly")
        self.model_combo.pack(fill="x", pady=(6, 0))
        ttk.Label(parent, text="仅影响实时预览；录后黑盒评分固定 heavy", foreground="#555555").pack(
            anchor="w", pady=(2, 0))
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(10, 0))
        self.start_btn = ttk.Button(row, text="开始")
        self.start_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(row, text="停止", state="disabled")
        self.stop_btn.pack(side="left", fill="x", expand=True, padx=(8, 0))

    def build_secondary(self, parent: Any) -> None:
        """次要选项：输入源提示、线程数、手部检测。"""
        ttk.Label(parent, textvariable=self.source_hint_var, wraplength=320).pack(anchor="w")
        ttk.Label(parent, text="线程数（>1：多核并行，关闭时序平滑）：").pack(anchor="w", pady=(10, 0))
        self.workers_spin = ttk.Spinbox(parent, from_=1, to=16, textvariable=self.workers_var, width=6)
        self.workers_spin.pack(anchor="w", pady=(6, 0))
        ttk.Checkbutton(parent, text="启用手部检测（V 手势 / 手部骨架）", variable=self.enable_hands_var).pack(
            anchor="w", pady=(10, 0))

    def connect(self, *, settings: Callable[[], None], refresh: Callable[[], None], start: Callable[[], None],
                stop: Callable[[], None], camera_selected: Callable[[], None],
                camera_2_selected: Callable[[], None]) -> None:
        self.settings_btn.configure(command=settings)
        self.refresh_btn.configure(command=refresh)
        self.start_btn.configure(command=start)
        self.stop_btn.configure(command=stop)
        self.camera_combo.bind("<<ComboboxSelected>>", lambda _event: camera_selected())
        self.camera_combo_2.bind("<<ComboboxSelected>>", lambda _event: camera_2_selected())
