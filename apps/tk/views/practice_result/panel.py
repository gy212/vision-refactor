# -*- coding: utf-8 -*-
"""学生练习结果页的容器：可滚动主区按顺序摆放头部、概览、结论、阶段栏、检查列表与证据、诊断和底部操作栏。

只消费 ``core.reporting`` 的报告模型；复核、导出、历史、再练一次都交给构造时注入的回调（练习控制器）。
"""
from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
import tkinter as tk
from tkinter import Canvas, messagebox, ttk
from typing import Any

from core.foundation import outputs_dir
from core.reporting import NormalizedReportModel, build_report_model

from apps.tk.views.practice_result.action_bar import ActionBar
from apps.tk.views.practice_result.check_list import CheckList, filter_checks, initial_check_id
from apps.tk.views.practice_result.diagnostics import DiagnosticsAccordion
from apps.tk.views.practice_result.evidence_view import EvidenceView
from apps.tk.views.practice_result.header import HeaderCard, HeadlineBanner
from apps.tk.views.practice_result.overview import OverviewCards
from apps.tk.views.practice_result.phase_bar import PhaseBar
from apps.tk.views.practice_result.review_box import ReviewBox
from apps.tk.views.practice_result.style import FONT, STYLE_A, label, load_status_icons

logger = logging.getLogger(__name__)

ReviewCallback = Callable[[str, str, str, str, str], None]  # (记录 id, 检查项 id, 结论, 教师, 说明)


class FeedbackResultPanel(ttk.Frame):
    """Style A 散打学生练习动作分析结果面板。"""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_practice_again: Callable[[], None] | None = None,
        on_export_report: Callable[[dict], None] | None = None,
        on_view_history: Callable[[], None] | None = None,
        on_review_toggle: ReviewCallback | None = None,
        on_switch_to_preview: Callable[[], None] | None = None,
        history_root: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self.on_practice_again = on_practice_again
        self.on_export_report = on_export_report
        self.on_view_history = on_view_history
        self.on_review_toggle = on_review_toggle
        self.history_root = history_root
        self._record: dict | None = None
        self._model: NormalizedReportModel | None = None
        self._record_dir: Path | None = None
        self._phase: str | None = None
        self._status_filter: str = "all"
        self._check_id: str | None = None
        self._icons = load_status_icons(self)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._build_top_bar(on_switch_to_preview)
        content = self._build_scroll_area()
        self.header = HeaderCard(content)
        self.overview = OverviewCards(content, self._icons)
        self.headline = HeadlineBanner(content)
        self.phase_bar = PhaseBar(content, self.select_phase)
        self._build_split(content)
        self.diagnostics = DiagnosticsAccordion(content)
        self.actions = ActionBar(self, on_history=self.view_history, on_export=self.export,
                                 on_practice_again=self.practice_again)
        self.actions.frame.grid(row=2, column=0, sticky="ew")

    def _build_top_bar(self, on_switch_to_preview: Callable[[], None] | None) -> None:
        top_bar = tk.Frame(self, bg=STYLE_A["bg_page"], pady=4, padx=8)
        top_bar.grid(row=0, column=0, sticky="ew")
        if on_switch_to_preview:
            tk.Button(top_bar, text="📹 返回录制预览", font=(FONT, 9), bg=STYLE_A["bg_card"], fg=STYLE_A["accent_blue"],
                      activebackground=STYLE_A["accent_blue_bg"], relief="flat", bd=1, padx=8, pady=2,
                      cursor="hand2", command=on_switch_to_preview).pack(side="left")
        label(top_bar, size=11, bold=True, bg=STYLE_A["bg_page"], text="学生练习动作问题分析结果").pack(side="left", padx=12)
        label(top_bar, bg=STYLE_A["bg_page"], fg=STYLE_A["text_secondary"],
              text="仅指出动作问题 · 不打分 · 供辅助教学使用").pack(side="right")

    def _build_scroll_area(self) -> tk.Frame:
        """Canvas + 竖向滚动条的主内容区；鼠标在内容上时滚轮滚动。"""
        container = tk.Frame(self, bg=STYLE_A["bg_page"])
        container.grid(row=1, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        canvas = Canvas(container, bg=STYLE_A["bg_page"], highlightthickness=0, borderwidth=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scroll.set)
        content = tk.Frame(canvas, bg=STYLE_A["bg_page"], padx=10, pady=8)
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))

        def _on_mousewheel(event: Any) -> None:
            if event.delta:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        content.bind("<Enter>", lambda _e: self.bind_all("<MouseWheel>", _on_mousewheel))
        content.bind("<Leave>", lambda _e: self.unbind_all("<MouseWheel>"))
        return content

    def _build_split(self, parent: tk.Misc) -> None:
        """左栏检查项列表（约 40%），右栏证据与复核（约 60%）。"""
        split = tk.Frame(parent, bg=STYLE_A["bg_page"])
        split.pack(fill="both", expand=True, pady=(0, 8))
        split.columnconfigure(0, weight=4)
        split.columnconfigure(1, weight=6)
        split.rowconfigure(0, weight=1)
        self.check_list = CheckList(split, self.select_status, self._on_tree_select)
        self.check_list.frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.evidence = EvidenceView(split)
        self.evidence.frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.review_box = ReviewBox(self.evidence.frame, self.review)

    # ------------------------------------------------------------------ 对外接口
    @property
    def record(self) -> dict | None:
        return self._record

    @property
    def check_tree(self) -> ttk.Treeview:
        return self.check_list.tree

    @property
    def teacher_entry(self) -> ttk.Entry:
        return self.review_box.teacher_entry

    @property
    def reason_entry(self) -> ttk.Entry:
        return self.review_box.reason_entry

    @property
    def photo_count(self) -> int:
        return self.evidence.photo_count

    def set_record(self, record: dict, record_dir: Path | None = None) -> None:
        """加载已完成的学生动作分析记录并渲染全貌。"""
        self.clear()
        self._record = record
        self._record_dir = record_dir
        try:
            self._model = build_report_model(record)
        except Exception as exc:
            logger.error("Failed to build report model from record: %s", exc)
            self.headline.show_error(exc)
            return
        model = self._model
        self.header.show(model)
        self.overview.show(model)
        self.headline.show(model)
        self.phase_bar.render(model.phases, model.counts.total_checks, self._phase)
        self._refresh_checks()
        self.diagnostics.show(model)

    def clear(self) -> None:
        """重置结果面板内容与内部缓存，安全可重入。"""
        self._record = None
        self._model = None
        self._record_dir = None
        self._phase = None
        self._status_filter = "all"
        self._check_id = None
        self.header.clear()
        self.overview.clear()
        self.headline.clear()
        self.phase_bar.clear()
        self.check_list.clear()
        self.evidence.clear()
        self.review_box.clear()
        self.diagnostics.clear()
        self.actions.status_var.set("")

    def destroy(self) -> None:
        """释放所有 PhotoImage 引用与控件树。"""
        self.evidence.clear()
        self._icons.clear()
        super().destroy()

    def select_phase(self, phase: str | None) -> None:
        """切换阶段筛选；``None`` 为全部阶段。"""
        self._phase = phase
        if self._model:
            self.phase_bar.render(self._model.phases, self._model.counts.total_checks, phase)
        self._refresh_checks()

    def select_status(self, status: str) -> None:
        """切换状态分类筛选。"""
        self._status_filter = status
        self.check_list.highlight_filter(status)
        self._refresh_checks()

    def select_check(self, check_id: str) -> None:
        """选中检查项并显示其证据与依据。"""
        self.check_list.select(check_id)
        self._on_tree_select()

    def review(self, decision: str) -> None:
        """教师复核当前选中项：``confirmed`` 或 ``revoked``。"""
        if not self._check_id or not self._record:
            messagebox.showinfo("提示", "请先从左侧列表选择要复核的检查项。", parent=self)
            return
        teacher, reason = self.review_box.inputs()
        if self.on_review_toggle:
            self.on_review_toggle(self._record["id"], self._check_id, decision, teacher, reason)

    def practice_again(self) -> None:
        if self.on_practice_again:
            self.on_practice_again()

    def view_history(self) -> None:
        if self.on_view_history:
            self.on_view_history()
        else:
            messagebox.showinfo("提示", "可在主界面点击「个人历史」打开历史记录对话框。", parent=self)

    def export(self) -> None:
        if not self._record:
            messagebox.showinfo("提示", "当前没有可导出的分析结果。", parent=self)
            return
        if self.on_export_report:
            self.on_export_report(self._record)

    # ------------------------------------------------------------------ 内部
    def _refresh_checks(self) -> None:
        """按当前阶段与状态重建检查项列表，并默认选中首个疑似问题。"""
        if not self._model:
            return
        self.check_list.render(filter_checks(self._model.all_checks, self._phase, self._status_filter))
        target = initial_check_id(self._model.all_checks, self.check_list.shown_ids())
        if target is None:
            self._check_id = None
            self.evidence.show_empty("当前筛选下没有检查项目")
            return
        self.select_check(target)

    def _on_tree_select(self) -> None:
        check_id = self.check_list.selected_id()
        if not check_id or not self._model:
            return
        self._check_id = check_id
        check = next((c for c in self._model.all_checks if c.id == check_id), None)
        if not check:
            return
        self.evidence.show_check(check)
        self.review_box.show(check)
        self.evidence.load_frames(self._record or {}, check, self._model.evidence_catalog, self._video_dir())

    def _video_dir(self) -> Path | None:
        """录像所在的练习记录目录：优先显式目录，其次按练习历史根目录与记录 id 定位。"""
        if self._record_dir is not None and self._record_dir.is_dir():
            return self._record_dir
        record_id = (self._record or {}).get("id")
        if not record_id or not isinstance(record_id, str):
            return None
        root = self.history_root if self.history_root is not None else outputs_dir() / "practice_feedback"
        return (root / record_id).resolve()
