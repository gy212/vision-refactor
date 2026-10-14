# -*- coding: utf-8 -*-
"""结果页左栏的检查项列表：状态分类筛选按钮 + 检查项表格，按阶段与状态过滤报告模型的检查项。"""
from __future__ import annotations

from collections.abc import Callable
import tkinter as tk
from tkinter import ttk

from core.reporting import NormalizedCheck

from apps.tk.views.practice_result.style import FONT, STYLE_A, card, toggle_colors

STATUS_FILTERS = [
    ("all", "全部"),
    ("candidate", "疑似问题"),
    ("not_observed", "未见异常"),
    ("unable", "无法判断"),
    ("pending_rule", "待教师"),
    ("revoked", "已撤销"),
]


def filter_checks(checks: list[NormalizedCheck], phase: str | None, status: str | None) -> list[NormalizedCheck]:
    """按阶段与状态筛选；"疑似问题"不含已撤销项，"已撤销"只看复核结论。"""
    if phase:
        checks = [c for c in checks if c.phase == phase]
    status = status or "all"
    if status == "candidate":
        return [c for c in checks if c.status == "candidate" and c.review != "revoked"]
    if status == "revoked":
        return [c for c in checks if c.review == "revoked"]
    if status in {"not_observed", "unable", "pending_rule"}:
        return [c for c in checks if c.status == status]
    return checks


def initial_check_id(checks: list[NormalizedCheck], shown_ids: tuple[str, ...]) -> str | None:
    """默认选中项：列表里首个未撤销的疑似问题，没有则选首项。"""
    if not shown_ids:
        return None
    by_id = {c.id: c for c in checks}
    for iid in shown_ids:
        check = by_id.get(iid)
        if check and check.status == "candidate" and check.review != "revoked":
            return iid
    return shown_ids[0]


class CheckList:
    """``on_filter(status)`` 在点击筛选按钮时调用；``on_select()`` 在表格选中项变化时调用。"""

    def __init__(self, parent: tk.Misc, on_filter: Callable[[str], None], on_select: Callable[[], None]) -> None:
        self.frame = card(parent, padx=8, pady=8)
        self.frame.rowconfigure(1, weight=1)
        self.frame.columnconfigure(0, weight=1)
        filter_row = tk.Frame(self.frame, bg=STYLE_A["bg_card"])
        filter_row.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.filter_buttons: dict[str, tk.Button] = {}
        for key, text in STATUS_FILTERS:
            btn = tk.Button(filter_row, text=text, font=(FONT, 8), bg=STYLE_A["bg_card"], fg=STYLE_A["text_primary"],
                            relief="flat", bd=1, padx=5, pady=1, cursor="hand2",
                            command=lambda k=key: on_filter(k))
            btn.pack(side="left", padx=2)
            self.filter_buttons[key] = btn
        self.tree = self._build_tree()
        self.tree.bind("<<TreeviewSelect>>", lambda _event: on_select())

    def _build_tree(self) -> ttk.Treeview:
        tree_frame = tk.Frame(self.frame, bg=STYLE_A["bg_card"])
        tree_frame.grid(row=1, column=0, sticky="nsew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        tree = ttk.Treeview(tree_frame, columns=("name", "status", "review"), show="headings",
                            height=12, selectmode="browse")
        tree.heading("name", text="检查项目", anchor="w")
        tree.heading("status", text="状态", anchor="w")
        tree.heading("review", text="复核", anchor="w")
        tree.column("name", width=180, stretch=True)
        tree.column("status", width=90, stretch=False)
        tree.column("review", width=75, stretch=False)
        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        for tag in ("candidate", "not_observed", "unable", "pending_rule", "revoked"):
            tree.tag_configure(tag, foreground=STYLE_A[tag]["fg"])
        return tree

    def highlight_filter(self, selected: str) -> None:
        for key, btn in self.filter_buttons.items():
            btn.configure(**toggle_colors(key == selected))

    def render(self, checks: list[NormalizedCheck]) -> None:
        self.clear()
        for check in checks:
            tag = "revoked" if check.review == "revoked" else check.status
            self.tree.insert("", "end", iid=check.id,
                             values=(f"[{check.phase_label}] {check.name}", check.status_display, check.review_display),
                             tags=(tag,))

    def shown_ids(self) -> tuple[str, ...]:
        return tuple(self.tree.get_children())

    def select(self, check_id: str) -> None:
        self.tree.selection_set(check_id)
        self.tree.see(check_id)

    def selected_id(self) -> str | None:
        selection = self.tree.selection()
        return selection[0] if selection else None

    def clear(self) -> None:
        self.tree.delete(*self.tree.get_children())
