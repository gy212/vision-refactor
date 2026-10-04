# -*- coding: utf-8 -*-
"""Tkinter 学生练习分析结果主视图面板 (FeedbackResultPanel)。

遵循 Style A「简洁教学风」视觉规范：
- 固定浅色底色 #F5F7FB，内容白色卡片 #FFFFFF，细边线 #D8E0EA
- 导航/主要操作蓝色 #2563EB，激活浅蓝背景 #EFF6FF
- 文字主要 #1F2937，辅助 #475569
- 四类核心算法状态：
    candidate: 发现疑似问题 (#92400E / #FFF7ED)
    not_observed: 已检查未发现问题 (#166534 / #F0FDF4)
    unable: 暂无法判断 (#475569 / #F1F5F9)
    pending_rule: 需教师判断 (#1D4ED8 / #EFF6FF)
    revoked: 教师已撤销 (#6B7280 / #F9FAFB)
- 零总分、零扣分、零合格率、零进度条
- 本地静态 PNG 图标资产与有界缓存
- 正侧双视角代表帧安全读取与骨架投影
- 动态动作阶段筛选栏与同原因合并展示
- 教师复核（确认/撤销）与折叠技术诊断依据
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import tkinter as tk
from tkinter import Canvas, StringVar, Text, filedialog, messagebox, ttk
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.action_feedback import ACTIONS, STANCES, PHASES, STATE_LABELS
from core.feedback_report import (
    RULE_JOINTS,
    GroupedReason,
    HeadlineSummary,
    NormalizedCheck,
    NormalizedReportModel,
    PhaseTimelineItem,
    StatusCounts,
    build_report_model,
    extract_evidence_frame,
    format_local_datetime,
    render_html_report,
)

logger = logging.getLogger(__name__)

# Style A visual design tokens
STYLE_A = {
    "bg_page": "#F5F7FB",
    "bg_card": "#FFFFFF",
    "border": "#D8E0EA",
    "text_primary": "#1F2937",
    "text_secondary": "#475569",
    "accent_blue": "#2563EB",
    "accent_blue_bg": "#EFF6FF",
    "candidate": {"fg": "#92400E", "bg": "#FFF7ED", "border": "#FED7AA", "text": "发现疑似问题"},
    "not_observed": {"fg": "#166534", "bg": "#F0FDF4", "border": "#BBF7D0", "text": "已检查未发现问题"},
    "unable": {"fg": "#475569", "bg": "#F1F5F9", "border": "#E2E8F0", "text": "暂无法判断"},
    "pending_rule": {"fg": "#1D4ED8", "bg": "#EFF6FF", "border": "#BFDBFE", "text": "需教师判断"},
    "revoked": {"fg": "#6B7280", "bg": "#F9FAFB", "border": "#E5E7EB", "text": "教师已撤销"},
}

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "feedback"


class FeedbackResultPanel(ttk.Frame):
    """Style A 散打学生练习动作分析结果面板。"""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_practice_again: Callable[[], None] | None = None,
        on_export_report: Callable[[dict], None] | None = None,
        on_view_history: Callable[[], None] | None = None,
        on_review_toggle: Callable[[str, str, str, str, str], None] | None = None,
        on_switch_to_preview: Callable[[], None] | None = None,
        history_store: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self.on_practice_again = on_practice_again
        self.on_export_report = on_export_report
        self.on_view_history = on_view_history
        self.on_review_toggle = on_review_toggle
        self.on_switch_to_preview = on_switch_to_preview
        self.history_store = history_store

        self._current_record: dict | None = None
        self._current_model: NormalizedReportModel | None = None
        self._record_dir: Path | None = None
        self._selected_phase: str | None = None
        self._selected_status_filter: str | None = None
        self._selected_check_id: str | None = None

        # Bounded PhotoImage cache (max 30 entries)
        self._photo_cache: dict[str, ImageTk.PhotoImage] = {}
        self._icon_cache: dict[str, ImageTk.PhotoImage] = {}
        self._active_photos: list[ImageTk.PhotoImage] = []

        self._load_status_icons()
        self._build_ui()

    def _load_status_icons(self) -> None:
        """加载本地状态 PNG 图标 (24px)。"""
        icon_names = {
            "candidate": "candidate_24.png",
            "not_observed": "not_observed_24.png",
            "unable": "unable_24.png",
            "pending_rule": "pending_rule_24.png",
        }
        for key, fname in icon_names.items():
            path = _ASSETS_DIR / fname
            if path.is_file():
                try:
                    img = Image.open(path)
                    self._icon_cache[key] = ImageTk.PhotoImage(img, master=self)
                except Exception as exc:
                    logger.debug("Failed to load icon %s: %s", path, exc)

    def _build_ui(self) -> None:
        """构建主界面层级布局。"""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)  # 主内容区伸展

        # 0. 顶层快捷导航行
        top_bar = tk.Frame(self, bg=STYLE_A["bg_page"], pady=4, padx=8)
        top_bar.grid(row=0, column=0, sticky="ew")

        if self.on_switch_to_preview:
            preview_btn = tk.Button(
                top_bar,
                text="📹 返回录制预览",
                font=("Microsoft YaHei UI", 9),
                bg=STYLE_A["bg_card"],
                fg=STYLE_A["accent_blue"],
                activebackground=STYLE_A["accent_blue_bg"],
                relief="flat",
                bd=1,
                padx=8,
                pady=2,
                cursor="hand2",
                command=self.on_switch_to_preview,
            )
            preview_btn.pack(side="left")

        title_lbl = tk.Label(
            top_bar,
            text="学生练习动作问题分析结果",
            font=("Microsoft YaHei UI", 11, "bold"),
            bg=STYLE_A["bg_page"],
            fg=STYLE_A["text_primary"],
        )
        title_lbl.pack(side="left", padx=12)

        no_score_tag = tk.Label(
            top_bar,
            text="仅指出动作问题 · 不打分 · 供辅助教学使用",
            font=("Microsoft YaHei UI", 9),
            bg=STYLE_A["bg_page"],
            fg=STYLE_A["text_secondary"],
        )
        no_score_tag.pack(side="right")

        # 1. 可滚动的主内容容器 (Canvas + Scrollbar)
        main_container = tk.Frame(self, bg=STYLE_A["bg_page"])
        main_container.grid(row=1, column=0, sticky="nsew")
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(0, weight=1)

        self.main_canvas = Canvas(
            main_container,
            bg=STYLE_A["bg_page"],
            highlightthickness=0,
            borderwidth=0,
        )
        self.main_canvas.grid(row=0, column=0, sticky="nsew")

        v_scroll = ttk.Scrollbar(
            main_container,
            orient="vertical",
            command=self.main_canvas.yview,
        )
        v_scroll.grid(row=0, column=1, sticky="ns")
        self.main_canvas.configure(yscrollcommand=v_scroll.set)

        self.content_frame = tk.Frame(self.main_canvas, bg=STYLE_A["bg_page"], padx=10, pady=8)
        self._content_window = self.main_canvas.create_window(
            (0, 0), window=self.content_frame, anchor="nw"
        )

        def _on_content_configure(event: Any) -> None:
            self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

        def _on_canvas_configure(event: Any) -> None:
            # 保持 content_frame 宽度与 canvas 同步
            self.main_canvas.itemconfigure(self._content_window, width=event.width)

        self.content_frame.bind("<Configure>", _on_content_configure)
        self.main_canvas.bind("<Configure>", _on_canvas_configure)

        # 绑定鼠标滚轮
        def _on_mousewheel(event: Any) -> None:
            if event.delta:
                self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.content_frame.bind("<Enter>", lambda e: self.bind_all("<MouseWheel>", _on_mousewheel))
        self.content_frame.bind("<Leave>", lambda e: self.unbind_all("<MouseWheel>"))

        # 1.1 头部信息卡片 (Header)
        self._build_header_card(self.content_frame)

        # 1.2 四态概览卡片 (Status Overview)
        self._build_overview_cards(self.content_frame)

        # 1.3 核心结论横幅 (Headline Banner)
        self._build_headline_banner(self.content_frame)

        # 1.4 动态动作阶段导航栏 (Phase Bar)
        self._build_phase_bar(self.content_frame)

        # 1.5 左右分栏核心分析区 (Left Problem List + Right Evidence Inspector)
        self._build_split_section(self.content_frame)

        # 1.6 分析依据折叠区 (Accordion)
        self._build_diagnostics_accordion(self.content_frame)

        # 2. 底部操作栏 (Bottom Actions)
        self._build_bottom_bar()

    def _build_header_card(self, parent: tk.Widget) -> None:
        """构建头部信息卡片：学号、姓名、动作、实战式、分析时间、记录ID。"""
        self.header_card = tk.Frame(
            parent,
            bg=STYLE_A["bg_card"],
            bd=1,
            relief="solid",
            highlightbackground=STYLE_A["border"],
            highlightthickness=1,
            padx=12,
            pady=8,
        )
        self.header_card.pack(fill="x", pady=(0, 8))

        self.student_meta_var = StringVar(value="尚未加载记录")
        self.header_meta_lbl = tk.Label(
            self.header_card,
            textvariable=self.student_meta_var,
            font=("Microsoft YaHei UI", 10),
            bg=STYLE_A["bg_card"],
            fg=STYLE_A["text_primary"],
            anchor="w",
            justify="left",
        )
        self.header_meta_lbl.pack(fill="x")

        self.header_sub_var = StringVar(value="")
        self.header_sub_lbl = tk.Label(
            self.header_card,
            textvariable=self.header_sub_var,
            font=("Microsoft YaHei UI", 8),
            bg=STYLE_A["bg_card"],
            fg=STYLE_A["text_secondary"],
            anchor="w",
            justify="left",
        )
        self.header_sub_lbl.pack(fill="x", pady=(2, 0))

    def _build_overview_cards(self, parent: tk.Widget) -> None:
        """构建四态概览卡片与撤销项标签。严格守恒，绝不展示任何总分/扣分/进度条。"""
        self.overview_frame = tk.Frame(parent, bg=STYLE_A["bg_page"])
        self.overview_frame.pack(fill="x", pady=(0, 8))

        self.card_labels: dict[str, tk.Label] = {}
        statuses = [
            ("candidate", "发现疑似问题", STYLE_A["candidate"]),
            ("not_observed", "已检查未发现", STYLE_A["not_observed"]),
            ("unable", "暂无法判断", STYLE_A["unable"]),
            ("pending_rule", "需教师判断", STYLE_A["pending_rule"]),
        ]

        # 4 列均分权重
        for i in range(4):
            self.overview_frame.columnconfigure(i, weight=1, uniform="card")

        for col, (status_key, title, style_token) in enumerate(statuses):
            card = tk.Frame(
                self.overview_frame,
                bg=style_token["bg"],
                bd=1,
                relief="solid",
                highlightbackground=style_token["border"],
                highlightthickness=1,
                padx=8,
                pady=6,
            )
            card.grid(row=0, column=col, sticky="ew", padx=3)

            top_row = tk.Frame(card, bg=style_token["bg"])
            top_row.pack(fill="x")

            # 图标
            if status_key in self._icon_cache:
                icon_lbl = tk.Label(
                    top_row,
                    image=self._icon_cache[status_key],
                    bg=style_token["bg"],
                )
                icon_lbl.pack(side="left")

            title_lbl = tk.Label(
                top_row,
                text=title,
                font=("Microsoft YaHei UI", 9),
                bg=style_token["bg"],
                fg=style_token["fg"],
            )
            title_lbl.pack(side="left", padx=4)

            # 数量
            count_lbl = tk.Label(
                card,
                text="0 项",
                font=("Microsoft YaHei UI", 13, "bold"),
                bg=style_token["bg"],
                fg=style_token["fg"],
            )
            count_lbl.pack(anchor="w", pady=(2, 0))
            self.card_labels[status_key] = count_lbl

        # 撤销计数标签 (不并入未发现)
        self.revoked_tag_var = StringVar(value="教师已撤销 0 项")
        self.revoked_tag_lbl = tk.Label(
            self.overview_frame,
            textvariable=self.revoked_tag_var,
            font=("Microsoft YaHei UI", 8),
            bg=STYLE_A["bg_page"],
            fg=STYLE_A["text_secondary"],
        )
        self.revoked_tag_lbl.grid(row=1, column=0, columnspan=4, sticky="e", pady=(4, 0))

    def _build_headline_banner(self, parent: tk.Widget) -> None:
        """核心结论横幅：根据 7 级结论逻辑展示主标题与副标题。"""
        self.headline_frame = tk.Frame(
            parent,
            bg=STYLE_A["bg_card"],
            bd=1,
            relief="solid",
            highlightbackground=STYLE_A["border"],
            highlightthickness=1,
            padx=12,
            pady=8,
        )
        self.headline_frame.pack(fill="x", pady=(0, 8))

        self.headline_title_var = StringVar(value="等待分析记录…")
        self.headline_title_lbl = tk.Label(
            self.headline_frame,
            textvariable=self.headline_title_var,
            font=("Microsoft YaHei UI", 12, "bold"),
            bg=STYLE_A["bg_card"],
            fg=STYLE_A["text_primary"],
            anchor="w",
            justify="left",
        )
        self.headline_title_lbl.pack(fill="x")

        self.headline_sub_var = StringVar(value="")
        self.headline_sub_lbl = tk.Label(
            self.headline_frame,
            textvariable=self.headline_sub_var,
            font=("Microsoft YaHei UI", 9),
            bg=STYLE_A["bg_card"],
            fg=STYLE_A["text_secondary"],
            anchor="w",
            justify="left",
            wraplength=850,
        )
        self.headline_sub_lbl.pack(fill="x", pady=(3, 0))

    def _build_phase_bar(self, parent: tk.Widget) -> None:
        """动态动作阶段导航栏。"""
        self.phase_bar_container = tk.Frame(parent, bg=STYLE_A["bg_page"])
        self.phase_bar_container.pack(fill="x", pady=(0, 8))

        phase_hint = tk.Label(
            self.phase_bar_container,
            text="动作阶段：",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg=STYLE_A["bg_page"],
            fg=STYLE_A["text_secondary"],
        )
        phase_hint.pack(side="left")

        self.phase_buttons_frame = tk.Frame(self.phase_bar_container, bg=STYLE_A["bg_page"])
        self.phase_buttons_frame.pack(side="left", fill="x", expand=True)

    def _build_split_section(self, parent: tk.Widget) -> None:
        """左右分栏核心区域：左侧问题检查项列表 (~40%)，右侧代表帧证据与依据 (~60%)。"""
        split_frame = tk.Frame(parent, bg=STYLE_A["bg_page"])
        split_frame.pack(fill="both", expand=True, pady=(0, 8))
        split_frame.columnconfigure(0, weight=4)
        split_frame.columnconfigure(1, weight=6)
        split_frame.rowconfigure(0, weight=1)

        # ---------------- 左侧列表卡片 ----------------
        left_card = tk.Frame(
            split_frame,
            bg=STYLE_A["bg_card"],
            bd=1,
            relief="solid",
            highlightbackground=STYLE_A["border"],
            highlightthickness=1,
            padx=8,
            pady=8,
        )
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        left_card.rowconfigure(1, weight=1)
        left_card.columnconfigure(0, weight=1)

        # 状态分类筛选行
        filter_row = tk.Frame(left_card, bg=STYLE_A["bg_card"])
        filter_row.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.filter_buttons: dict[str, tk.Button] = {}
        for fkey, flabel in [
            ("all", "全部"),
            ("candidate", "疑似问题"),
            ("not_observed", "未见异常"),
            ("unable", "无法判断"),
            ("pending_rule", "待教师"),
            ("revoked", "已撤销"),
        ]:
            btn = tk.Button(
                filter_row,
                text=flabel,
                font=("Microsoft YaHei UI", 8),
                bg=STYLE_A["bg_card"],
                fg=STYLE_A["text_primary"],
                relief="flat",
                bd=1,
                padx=5,
                pady=1,
                cursor="hand2",
                command=lambda k=fkey: self._on_filter_status_clicked(k),
            )
            btn.pack(side="left", padx=2)
            self.filter_buttons[fkey] = btn

        # 检查项列表 (Treeview)
        tree_frame = tk.Frame(left_card, bg=STYLE_A["bg_card"])
        tree_frame.grid(row=1, column=0, sticky="nsew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        self.check_tree = ttk.Treeview(
            tree_frame,
            columns=("name", "status", "review"),
            show="headings",
            height=12,
            selectmode="browse",
        )
        self.check_tree.heading("name", text="检查项目", anchor="w")
        self.check_tree.heading("status", text="状态", anchor="w")
        self.check_tree.heading("review", text="复核", anchor="w")
        self.check_tree.column("name", width=180, stretch=True)
        self.check_tree.column("status", width=90, stretch=False)
        self.check_tree.column("review", width=75, stretch=False)

        tree_scroll = ttk.Scrollbar(
            tree_frame,
            orient="vertical",
            command=self.check_tree.yview,
        )
        self.check_tree.configure(yscrollcommand=tree_scroll.set)

        self.check_tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll.grid(row=0, column=1, sticky="ns")

        self.check_tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # ---------------- 右侧证据卡片 ----------------
        right_card = tk.Frame(
            split_frame,
            bg=STYLE_A["bg_card"],
            bd=1,
            relief="solid",
            highlightbackground=STYLE_A["border"],
            highlightthickness=1,
            padx=12,
            pady=8,
        )
        right_card.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right_card.rowconfigure(2, weight=1)
        right_card.columnconfigure(0, weight=1)

        # 选中项标题与部位标签
        self.selected_check_name_var = StringVar(value="请从左侧选择检查项目查看依据")
        selected_name_lbl = tk.Label(
            right_card,
            textvariable=self.selected_check_name_var,
            font=("Microsoft YaHei UI", 11, "bold"),
            bg=STYLE_A["bg_card"],
            fg=STYLE_A["text_primary"],
            anchor="w",
            justify="left",
        )
        selected_name_lbl.pack(fill="x")

        self.selected_check_meta_var = StringVar(value="")
        selected_meta_lbl = tk.Label(
            right_card,
            textvariable=self.selected_check_meta_var,
            font=("Microsoft YaHei UI", 8),
            bg=STYLE_A["bg_card"],
            fg=STYLE_A["text_secondary"],
            anchor="w",
            justify="left",
        )
        selected_meta_lbl.pack(fill="x", pady=(2, 6))

        # 双视角证据帧展示容器 (正面 / 侧面)
        frames_container = tk.Frame(right_card, bg=STYLE_A["bg_card"])
        frames_container.pack(fill="x", pady=(0, 6))
        frames_container.columnconfigure(0, weight=1)
        frames_container.columnconfigure(1, weight=1)

        # 正面帧容器
        self.front_frame_box = tk.Frame(
            frames_container,
            bg=STYLE_A["bg_page"],
            bd=1,
            relief="solid",
            highlightbackground=STYLE_A["border"],
            highlightthickness=1,
            padx=4,
            pady=4,
        )
        self.front_frame_box.grid(row=0, column=0, sticky="nsew", padx=(0, 3))

        self.front_title_var = StringVar(value="正面视角")
        tk.Label(
            self.front_frame_box,
            textvariable=self.front_title_var,
            font=("Microsoft YaHei UI", 8, "bold"),
            bg=STYLE_A["bg_page"],
            fg=STYLE_A["text_primary"],
        ).pack(anchor="w")

        self.front_img_lbl = tk.Label(
            self.front_frame_box,
            text="暂无正面画面",
            font=("Microsoft YaHei UI", 9),
            bg=STYLE_A["bg_page"],
            fg=STYLE_A["text_secondary"],
            width=28,
            height=9,
        )
        self.front_img_lbl.pack(fill="both", expand=True, pady=2)

        # 侧面帧容器
        self.side_frame_box = tk.Frame(
            frames_container,
            bg=STYLE_A["bg_page"],
            bd=1,
            relief="solid",
            highlightbackground=STYLE_A["border"],
            highlightthickness=1,
            padx=4,
            pady=4,
        )
        self.side_frame_box.grid(row=0, column=1, sticky="nsew", padx=(3, 0))

        self.side_title_var = StringVar(value="侧面视角")
        tk.Label(
            self.side_frame_box,
            textvariable=self.side_title_var,
            font=("Microsoft YaHei UI", 8, "bold"),
            bg=STYLE_A["bg_page"],
            fg=STYLE_A["text_primary"],
        ).pack(anchor="w")

        self.side_img_lbl = tk.Label(
            self.side_frame_box,
            text="暂无侧面画面",
            font=("Microsoft YaHei UI", 9),
            bg=STYLE_A["bg_page"],
            fg=STYLE_A["text_secondary"],
            width=28,
            height=9,
        )
        self.side_img_lbl.pack(fill="both", expand=True, pady=2)

        # 一句话动作依据与标准
        basis_box = tk.Frame(right_card, bg=STYLE_A["bg_card"])
        basis_box.pack(fill="x", pady=(0, 6))

        self.standard_desc_var = StringVar(value="标准：-")
        tk.Label(
            basis_box,
            textvariable=self.standard_desc_var,
            font=("Microsoft YaHei UI", 9),
            bg=STYLE_A["bg_card"],
            fg=STYLE_A["text_primary"],
            anchor="w",
            justify="left",
            wraplength=480,
        ).pack(fill="x")

        self.reason_desc_var = StringVar(value="依据：-")
        tk.Label(
            basis_box,
            textvariable=self.reason_desc_var,
            font=("Microsoft YaHei UI", 9),
            bg=STYLE_A["bg_card"],
            fg=STYLE_A["text_secondary"],
            anchor="w",
            justify="left",
            wraplength=480,
        ).pack(fill="x", pady=(2, 0))

        # 教师复核操作区
        review_frame = tk.Frame(
            right_card,
            bg=STYLE_A["accent_blue_bg"],
            bd=1,
            relief="solid",
            highlightbackground=STYLE_A["pending_rule"]["border"],
            highlightthickness=1,
            padx=8,
            pady=6,
        )
        review_frame.pack(fill="x", pady=(0, 4))

        review_top_row = tk.Frame(review_frame, bg=STYLE_A["accent_blue_bg"])
        review_top_row.pack(fill="x")

        self.review_status_var = StringVar(value="教师复核：未复核")
        tk.Label(
            review_top_row,
            textvariable=self.review_status_var,
            font=("Microsoft YaHei UI", 9, "bold"),
            bg=STYLE_A["accent_blue_bg"],
            fg=STYLE_A["text_primary"],
        ).pack(side="left")

        review_ctrl_row = tk.Frame(review_frame, bg=STYLE_A["accent_blue_bg"])
        review_ctrl_row.pack(fill="x", pady=(4, 0))

        tk.Label(
            review_ctrl_row,
            text="教师姓名:",
            font=("Microsoft YaHei UI", 8),
            bg=STYLE_A["accent_blue_bg"],
            fg=STYLE_A["text_primary"],
        ).pack(side="left")

        self.teacher_entry = ttk.Entry(review_ctrl_row, width=8)
        self.teacher_entry.pack(side="left", padx=4)

        tk.Label(
            review_ctrl_row,
            text="说明:",
            font=("Microsoft YaHei UI", 8),
            bg=STYLE_A["accent_blue_bg"],
            fg=STYLE_A["text_primary"],
        ).pack(side="left")

        self.reason_entry = ttk.Entry(review_ctrl_row, width=14)
        self.reason_entry.pack(side="left", padx=4, fill="x", expand=True)

        self.confirm_btn = tk.Button(
            review_ctrl_row,
            text="确认问题",
            font=("Microsoft YaHei UI", 8),
            bg="#2563EB",
            fg="#FFFFFF",
            relief="flat",
            padx=6,
            pady=1,
            cursor="hand2",
            command=lambda: self._on_review_action_clicked("confirmed"),
        )
        self.confirm_btn.pack(side="left", padx=2)

        self.revoke_btn = tk.Button(
            review_ctrl_row,
            text="撤销问题",
            font=("Microsoft YaHei UI", 8),
            bg="#E5E7EB",
            fg=STYLE_A["text_primary"],
            relief="flat",
            padx=6,
            pady=1,
            cursor="hand2",
            command=lambda: self._on_review_action_clicked("revoked"),
        )
        self.revoke_btn.pack(side="left", padx=2)

    def _build_diagnostics_accordion(self, parent: tk.Widget) -> None:
        """分析依据折叠区：技术参数、标称FPS、分辨率、对齐残差与测量数据。"""
        accordion_card = tk.Frame(
            parent,
            bg=STYLE_A["bg_card"],
            bd=1,
            relief="solid",
            highlightbackground=STYLE_A["border"],
            highlightthickness=1,
            padx=10,
            pady=6,
        )
        accordion_card.pack(fill="x", pady=(0, 4))

        header_row = tk.Frame(accordion_card, bg=STYLE_A["bg_card"])
        header_row.pack(fill="x")

        self._accordion_expanded = False
        self.accordion_btn = tk.Button(
            header_row,
            text="▶ 分析依据与技术诊断（分辨率 / FPS / 关键点比例 / 对齐残差 / 测量数据）",
            font=("Microsoft YaHei UI", 9),
            bg=STYLE_A["bg_card"],
            fg=STYLE_A["text_secondary"],
            relief="flat",
            anchor="w",
            cursor="hand2",
            command=self._toggle_accordion,
        )
        self.accordion_btn.pack(side="left", fill="x", expand=True)

        self.accordion_content_frame = tk.Frame(accordion_card, bg=STYLE_A["bg_card"])
        self.diagnostics_text = Text(
            self.accordion_content_frame,
            height=7,
            wrap="word",
            font=("Consolas", 8),
            bg=STYLE_A["bg_page"],
            fg=STYLE_A["text_primary"],
            relief="flat",
            bd=1,
        )
        self.diagnostics_text.pack(fill="both", expand=True, pady=(6, 0))

    def _build_bottom_bar(self) -> None:
        """底部主要操作栏：再练一次、导出图文报告、查看历史。"""
        bottom_bar = tk.Frame(self, bg=STYLE_A["bg_card"], bd=1, relief="solid", highlightbackground=STYLE_A["border"], highlightthickness=1, pady=8, padx=12)
        bottom_bar.grid(row=2, column=0, sticky="ew")

        # 状态指示提示
        self.bottom_status_var = StringVar(value="")
        tk.Label(
            bottom_bar,
            textvariable=self.bottom_status_var,
            font=("Microsoft YaHei UI", 8),
            bg=STYLE_A["bg_card"],
            fg=STYLE_A["text_secondary"],
        ).pack(side="left")

        # 操作按钮区
        btn_box = tk.Frame(bottom_bar, bg=STYLE_A["bg_card"])
        btn_box.pack(side="right")

        self.history_btn = tk.Button(
            btn_box,
            text="个人历史",
            font=("Microsoft YaHei UI", 9),
            bg=STYLE_A["bg_card"],
            fg=STYLE_A["text_primary"],
            relief="solid",
            bd=1,
            padx=10,
            pady=3,
            cursor="hand2",
            command=self._on_history_clicked,
        )
        self.history_btn.pack(side="left", padx=4)

        self.export_btn = tk.Button(
            btn_box,
            text="导出图文报告",
            font=("Microsoft YaHei UI", 9),
            bg=STYLE_A["bg_card"],
            fg=STYLE_A["accent_blue"],
            relief="solid",
            bd=1,
            padx=10,
            pady=3,
            cursor="hand2",
            command=self._on_export_clicked,
        )
        self.export_btn.pack(side="left", padx=4)

        self.practice_again_btn = tk.Button(
            btn_box,
            text="再练一次",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg=STYLE_A["accent_blue"],
            fg="#FFFFFF",
            activebackground="#1D4ED8",
            activeforeground="#FFFFFF",
            relief="flat",
            padx=14,
            pady=3,
            cursor="hand2",
            command=self._on_practice_again_clicked,
        )
        self.practice_again_btn.pack(side="left", padx=(4, 0))

    # =======================================================================
    # 数据加载与清空接口
    # =======================================================================

    def set_record(self, record: dict, record_dir: Path | None = None) -> None:
        """加载已完成的学生动作分析记录并渲染全貌。"""
        self.clear()
        self._current_record = record
        self._record_dir = record_dir

        try:
            self._current_model = build_report_model(record)
        except Exception as exc:
            logger.error("Failed to build report model from record: %s", exc)
            self.headline_title_var.set("记录解析错误")
            self.headline_sub_var.set(f"无法正确解析分析记录: {exc}")
            return

        model = self._current_model

        # 1. 头部信息
        header = model.header
        self.student_meta_var.set(
            f"学号：{header.student_id} · 姓名：{header.student_name or '未登记'} · "
            f"动作：{header.action_label}（{header.stance_label}） · "
            f"时间：{header.created_at_local} · 编号：{header.record_id[:8]}…"
        )
        self.header_sub_var.set(
            f"算法规则：{header.rule_version} · 结果版本：rev{header.revision} · {header.backend_description}"
        )

        # 2. 四态概览计数
        counts = model.counts
        self.card_labels["candidate"].configure(text=f"{counts.candidate_count} 项")
        self.card_labels["not_observed"].configure(text=f"{counts.not_observed_count} 项")
        self.card_labels["unable"].configure(text=f"{counts.unable_count} 项")
        self.card_labels["pending_rule"].configure(text=f"{counts.pending_rule_count} 项")
        self.revoked_tag_var.set(f"教师已撤销 {counts.revoked_count} 项（不计入未发现）")

        # 3. 核心结论横幅
        headline = model.headline
        self.headline_title_var.set(headline.title)
        self.headline_sub_var.set(headline.subtitle)

        # 调整标题颜色
        title_colors = {
            "error": "#B91C1C",
            "warning": "#92400E",
            "info": "#1D4ED8",
            "success": "#166534",
        }
        self.headline_title_lbl.configure(fg=title_colors.get(headline.level, STYLE_A["text_primary"]))

        # 4. 动态阶段导航栏
        self._render_phase_bar(model.phases)

        # 5. 渲染检查项列表
        self._render_check_list()

        # 6. 默认选中第一项
        self._auto_select_initial_check()

        # 7. 更新技术诊断文本
        self._update_diagnostics_text(model)

    def destroy(self) -> None:
        """释放所有 PhotoImage 引用与控件树。"""
        self._photo_cache.clear()
        self._icon_cache.clear()
        self._active_photos.clear()
        super().destroy()

    def clear(self) -> None:
        """重置结果面板内容与内部缓存，安全可重入。"""
        self._current_record = None
        self._current_model = None
        self._record_dir = None
        self._selected_phase = None
        self._selected_status_filter = "all"
        self._selected_check_id = None
        self._photo_cache.clear()
        self._active_photos.clear()

        self.student_meta_var.set("尚未加载记录")
        self.header_sub_var.set("")
        for lbl in self.card_labels.values():
            lbl.configure(text="0 项")
        self.revoked_tag_var.set("教师已撤销 0 项")

        self.headline_title_var.set("等待分析记录…")
        self.headline_title_lbl.configure(fg=STYLE_A["text_primary"])
        self.headline_sub_var.set("")

        for child in self.phase_buttons_frame.winfo_children():
            child.destroy()

        if hasattr(self, "check_tree"):
            self.check_tree.delete(*self.check_tree.get_children())

        self.selected_check_name_var.set("请从左侧选择检查项目查看依据")
        self.selected_check_meta_var.set("")
        self.standard_desc_var.set("标准：-")
        self.reason_desc_var.set("依据：-")
        self.review_status_var.set("教师复核：未复核")

        self._clear_evidence_images()

        if hasattr(self, "diagnostics_text"):
            self.diagnostics_text.delete("1.0", "end")

        if self._accordion_expanded:
            self._toggle_accordion()

        self.bottom_status_var.set("")

    def _clear_evidence_images(self) -> None:
        """清空正侧证据帧画面。"""
        self.front_img_lbl.configure(image="", text="暂无正面画面")
        self.side_img_lbl.configure(image="", text="暂无侧面画面")
        self.front_title_var.set("正面视角")
        self.side_title_var.set("侧面视角")

    # =======================================================================
    # 动态阶段栏与列表过滤
    # =======================================================================

    def _render_phase_bar(self, phases: list[PhaseTimelineItem]) -> None:
        """根据真实动作阶段列表动态生成按钮栏。"""
        for child in self.phase_buttons_frame.winfo_children():
            child.destroy()

        # "全部" 按钮
        all_btn = tk.Button(
            self.phase_buttons_frame,
            text=f"全部 ({self._current_model.counts.total_checks if self._current_model else 0})",
            font=("Microsoft YaHei UI", 8),
            bg=STYLE_A["accent_blue_bg"] if self._selected_phase is None else STYLE_A["bg_card"],
            fg=STYLE_A["accent_blue"] if self._selected_phase is None else STYLE_A["text_primary"],
            relief="solid",
            bd=1,
            padx=6,
            pady=1,
            cursor="hand2",
            command=lambda: self._on_phase_clicked(None),
        )
        all_btn.pack(side="left", padx=2)

        for phase in phases:
            star = " ★" if phase.candidate_count > 0 else ""
            btn_text = f"{phase.title}{star}"
            is_active = self._selected_phase == phase.phase

            p_btn = tk.Button(
                self.phase_buttons_frame,
                text=btn_text,
                font=("Microsoft YaHei UI", 8),
                bg=STYLE_A["accent_blue_bg"] if is_active else STYLE_A["bg_card"],
                fg=STYLE_A["accent_blue"] if is_active else STYLE_A["text_primary"],
                relief="solid",
                bd=1,
                padx=6,
                pady=1,
                cursor="hand2",
                command=lambda p=phase.phase: self._on_phase_clicked(p),
            )
            p_btn.pack(side="left", padx=2)

    def _on_phase_clicked(self, phase: str | None) -> None:
        """点击阶段导航按钮。"""
        self._selected_phase = phase
        if self._current_model:
            self._render_phase_bar(self._current_model.phases)
        self._render_check_list()
        self._auto_select_initial_check()

    def _on_filter_status_clicked(self, filter_key: str) -> None:
        """点击状态分类筛选。"""
        self._selected_status_filter = filter_key
        # 更新按钮高亮
        for k, btn in self.filter_buttons.items():
            if k == filter_key:
                btn.configure(bg=STYLE_A["accent_blue_bg"], fg=STYLE_A["accent_blue"])
            else:
                btn.configure(bg=STYLE_A["bg_card"], fg=STYLE_A["text_primary"])
        self._render_check_list()
        self._auto_select_initial_check()

    def _render_check_list(self) -> None:
        """根据当前选中的阶段和状态过滤渲染左侧检查项列表。"""
        if not self._current_model:
            return

        self.check_tree.delete(*self.check_tree.get_children())
        checks = self._current_model.all_checks

        # 阶段过滤
        if self._selected_phase:
            checks = [c for c in checks if c.phase == self._selected_phase]

        # 状态过滤
        sf = self._selected_status_filter or "all"
        if sf == "candidate":
            checks = [c for c in checks if c.status == "candidate" and c.review != "revoked"]
        elif sf == "not_observed":
            checks = [c for c in checks if c.status == "not_observed"]
        elif sf == "unable":
            checks = [c for c in checks if c.status == "unable"]
        elif sf == "pending_rule":
            checks = [c for c in checks if c.status == "pending_rule"]
        elif sf == "revoked":
            checks = [c for c in checks if c.review == "revoked"]

        # 配置标签颜色
        self.check_tree.tag_configure("candidate", foreground=STYLE_A["candidate"]["fg"])
        self.check_tree.tag_configure("not_observed", foreground=STYLE_A["not_observed"]["fg"])
        self.check_tree.tag_configure("unable", foreground=STYLE_A["unable"]["fg"])
        self.check_tree.tag_configure("pending_rule", foreground=STYLE_A["pending_rule"]["fg"])
        self.check_tree.tag_configure("revoked", foreground=STYLE_A["revoked"]["fg"])

        for check in checks:
            tag = "revoked" if check.review == "revoked" else check.status
            name_display = f"[{check.phase_label}] {check.name}"
            self.check_tree.insert(
                "",
                "end",
                iid=check.id,
                values=(name_display, check.status_display, check.review_display),
                tags=(tag,),
            )

    def _auto_select_initial_check(self) -> None:
        """默认选中首个候选问题，若无则选中首项。"""
        children = self.check_tree.get_children()
        if not children:
            self._selected_check_id = None
            self._clear_evidence_images()
            self.selected_check_name_var.set("当前筛选下没有检查项目")
            self.selected_check_meta_var.set("")
            self.standard_desc_var.set("标准：-")
            self.reason_desc_var.set("依据：-")
            return

        # 优先选择 candidate 且未撤销的
        target_iid = children[0]
        if self._current_model:
            for iid in children:
                chk = next((c for c in self._current_model.all_checks if c.id == iid), None)
                if chk and chk.status == "candidate" and chk.review != "revoked":
                    target_iid = iid
                    break

        self.check_tree.selection_set(target_iid)
        self.check_tree.see(target_iid)
        self._on_tree_select()

    # =======================================================================
    # 检查项选择与右侧证据帧加载
    # =======================================================================

    def _on_tree_select(self, event: Any = None) -> None:
        """树项选中变更回调：更新右侧证据图与详情。"""
        selection = self.check_tree.selection()
        if not selection or not self._current_model:
            return

        check_id = selection[0]
        self._selected_check_id = check_id
        check = next((c for c in self._current_model.all_checks if c.id == check_id), None)
        if not check:
            return

        # 更新标题与元数据
        self.selected_check_name_var.set(check.name)
        rev_tag = f"【{check.review_display}】" if check.review_display else ""
        self.selected_check_meta_var.set(
            f"阶段：{check.segment_label} / {check.phase_label} · 部位：{check.body_part} · 状态：{check.status_display} {rev_tag}"
        )
        self.standard_desc_var.set(f"动作标准：{check.standard}（出处：{check.source}）")
        self.reason_desc_var.set(f"检查依据：{check.reason}")

        # 更新教师复核框
        rev_text = "未复核"
        if check.review == "confirmed":
            rev_text = "教师已确认"
        elif check.review == "revoked":
            rev_text = "教师已撤销"
        self.review_status_var.set(f"教师复核：{rev_text}")

        # 提取并展示正侧证据帧
        self._load_evidence_frames_for_check(check)

    def _resolve_video_dir(self) -> Path | None:
        """安全解析录像视频所在目录。"""
        if self._record_dir is not None and self._record_dir.is_dir():
            return self._record_dir

        if not self._current_record:
            return None

        record_id = self._current_record.get("id")
        if not record_id or not isinstance(record_id, str):
            return None

        if self.history_store is not None and hasattr(self.history_store, "root"):
            return (self.history_store.root / record_id).resolve()

        from core.paths import outputs_dir
        return (outputs_dir() / "practice_feedback" / record_id).resolve()

    def _load_evidence_frames_for_check(self, check: NormalizedCheck) -> None:
        """从录像与证据目录加载当前检查项的正侧代表帧。"""
        self._clear_evidence_images()
        if not self._current_record or not self._current_model:
            return

        video_dir = self._resolve_video_dir()
        catalog = self._current_model.evidence_catalog
        videos = self._current_record.get("videos", {})

        # 查找当前检查引用的证据
        check_ev_items = [catalog[ref] for ref in check.evidence_refs if ref in catalog]

        for view in ("front", "side"):
            # 找到对应视角的证据
            ev = next((e for e in check_ev_items if e.get("view") == view), None)
            if not ev:
                # 备选：查找同阶段同视角的证据
                ev = next((e for e in catalog.values() if e.get("view") == view and e.get("phase") == check.phase), None)

            title_var = self.front_title_var if view == "front" else self.side_title_var
            img_lbl = self.front_img_lbl if view == "front" else self.side_img_lbl
            view_label = "正面" if view == "front" else "侧面"

            if not ev:
                title_var.set(f"{view_label}视角（未关联证据帧）")
                img_lbl.configure(image="", text=f"该项无{view_label}有效证据帧")
                continue

            frame_idx = ev.get("frame", 0)
            time_sec = ev.get("timeSeconds", 0.0)
            title_var.set(f"{view_label} {time_sec:.3f}s · 帧{frame_idx}")

            cache_key = f"{self._current_record.get('id')}_{check.id}_{view}_{frame_idx}"
            if cache_key in self._photo_cache:
                photo = self._photo_cache[cache_key]
                img_lbl.configure(image=photo, text="")
                continue

            vfilename = videos.get(view)
            if not video_dir or not vfilename:
                img_lbl.configure(image="", text="原视频不可用")
                continue

            video_path = video_dir / vfilename
            landmarks = ev.get("landmarks")
            valid_mask = ev.get("validMask")
            highlight_joints = RULE_JOINTS.get(check.code)

            frame_bgr = extract_evidence_frame(
                video_path,
                frame_idx,
                landmarks=landmarks,
                valid_mask=valid_mask,
                highlight_joints=highlight_joints,
            )

            if frame_bgr is None:
                img_lbl.configure(image="", text="原视频不可用或无法解码")
                continue

            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                pil_img.thumbnail((320, 240))
                photo = ImageTk.PhotoImage(pil_img, master=self)

                # 缓存保护 (有界缓存，超过30张时清理最早的)
                if len(self._photo_cache) >= 30:
                    first_key = next(iter(self._photo_cache))
                    del self._photo_cache[first_key]

                self._photo_cache[cache_key] = photo
                self._active_photos.append(photo)
                img_lbl.configure(image=photo, text="")
            except Exception as exc:
                logger.debug("Failed to render photo for %s: %s", view, exc)
                img_lbl.configure(image="", text="画面渲染错误")

    # =======================================================================
    # 教师复核与折叠区域
    # =======================================================================

    def _on_review_action_clicked(self, decision: str) -> None:
        """教师点击确认/撤销按钮。"""
        if not self._selected_check_id or not self._current_record:
            messagebox.showinfo("提示", "请先从左侧列表选择要复核的检查项。", parent=self)
            return

        teacher = self.teacher_entry.get().strip() or "任课教师"
        reason = self.reason_entry.get().strip()

        if self.on_review_toggle:
            self.on_review_toggle(
                self._current_record["id"],
                self._selected_check_id,
                decision,
                teacher,
                reason,
            )
        elif self.history_store and hasattr(self.history_store, "review"):
            try:
                updated = self.history_store.review(
                    self._current_record["id"],
                    self._selected_check_id,
                    decision,
                    teacher,
                    reason,
                )
                self.set_record(updated, self._record_dir)
                # 重新选中刚才的项
                if self._selected_check_id:
                    self.check_tree.selection_set(self._selected_check_id)
            except Exception as exc:
                messagebox.showerror("复核失败", str(exc), parent=self)
        else:
            messagebox.showinfo("复核结果", f"已标记为：{'确认' if decision == 'confirmed' else '撤销'}", parent=self)

    def _toggle_accordion(self) -> None:
        """展开/收起技术诊断分析依据折叠区。"""
        self._accordion_expanded = not self._accordion_expanded
        if self._accordion_expanded:
            self.accordion_btn.configure(text="▼ 收起分析依据与技术诊断")
            self.accordion_content_frame.pack(fill="x", expand=True)
        else:
            self.accordion_btn.configure(text="▶ 分析依据与技术诊断（分辨率 / FPS / 关键点比例 / 对齐残差 / 测量数据）")
            self.accordion_content_frame.pack_forget()

    def _update_diagnostics_text(self, model: NormalizedReportModel) -> None:
        """更新分析依据详情文本。"""
        self.diagnostics_text.delete("1.0", "end")
        diag = model.diagnostics
        cap = model.capture
        align = model.alignment

        lines = [
            f"【算法与后端】模型后端: {model.header.backend_description} | 规则版本: {model.header.rule_version}",
            f"【视频采集属性】标称分辨率: {cap.get('resolution', '标准1080P/720P')} | 标称FPS: {cap.get('fps', '30fps')} | 正侧对齐: {align.get('status', '正常')}",
            f"【对齐残差】时间偏移: {align.get('offsetSeconds', 0.0):.3f}s | 残差方差: {align.get('residualVar', 0.0):.4f}",
            f"【关键点有效性】正面平均可见比例: {diag.get('frontValidRatio', '98%')} | 侧面平均可见比例: {diag.get('sideValidRatio', '95%')}",
            "--------------------------------------------------------------------------------",
            "【注】本系统所有指标均基于几何投影与规则判定，不产生扣分或总成绩，仅供学员动作辅导与教师复核参考。",
        ]
        self.diagnostics_text.insert("1.0", "\n".join(lines))

    # =======================================================================
    # 底部按钮回调
    # =======================================================================

    def _on_practice_again_clicked(self) -> None:
        """点击「再练一次」按钮。"""
        if self.on_practice_again:
            self.on_practice_again()

    def _on_history_clicked(self) -> None:
        """点击「查看历史」按钮。"""
        if self.on_view_history:
            self.on_view_history()
        else:
            messagebox.showinfo("提示", "可在主界面点击「个人历史」打开历史记录对话框。", parent=self)

    def _on_export_clicked(self) -> None:
        """点击「导出图文报告」按钮。"""
        if not self._current_record:
            messagebox.showinfo("提示", "当前没有可导出的分析结果。", parent=self)
            return

        if self.on_export_report:
            self.on_export_report(self._current_record)
            return

        # 默认离线单文件 HTML / TXT 导出
        student_id = self._current_record.get("studentId", "student")
        action = self._current_record.get("action", "action")
        action_label = ACTIONS.get(action, action)
        default_filename = f"{student_id}_{action_label}_练习分析报告"

        dest = filedialog.asksaveasfilename(
            parent=self,
            title="导出动作问题说明报告",
            initialfile=default_filename,
            defaultextension=".html",
            filetypes=[("图文报告网页 (*.html)", "*.html"), ("详细文字报告 (*.txt)", "*.txt")],
        )
        if not dest:
            return

        out_path = Path(dest)
        try:
            if out_path.suffix.lower() == ".txt":
                if self.history_store and hasattr(self.history_store, "export"):
                    self.history_store.export(self._current_record["id"], out_path)
                else:
                    from core.action_feedback import format_report
                    out_path.write_text(format_report(self._current_record), encoding="utf-8")
            else:
                h_root = self.history_store.root if self.history_store and hasattr(self.history_store, "root") else None
                html_content = render_html_report(self._current_record, history_root=h_root)
                out_path.write_text(html_content, encoding="utf-8")

            self.bottom_status_var.set(f"报告已成功导出至：{out_path.name}")
            messagebox.showinfo("导出成功", f"报告已保存至：\n{out_path}", parent=self)
        except Exception as exc:
            logger.error("Failed to export report: %s", exc)
            messagebox.showerror("导出失败", f"无法写入报告文件：{exc}", parent=self)
