# -*- coding: utf-8 -*-
"""学生练习检查项全局设置弹窗：按基础动作段分组展示全部检查项并支持批量勾选与保存。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from tkinter import BooleanVar, Canvas, StringVar, Toplevel, messagebox, ttk
from typing import Any

from core.evaluation import DEFAULT_STANCE
from core.foundation import load_practice_disabled_checks

from apps.tk.controllers.practice_checks import (
    CheckGroup,
    build_check_groups,
    check_item_label,
    checks_summary_text,
    validate_and_save_checks,
    zero_enabled_actions,
)


class PracticeChecksDialog:
    """学生练习全部检查项的统一开关弹窗。"""

    def __init__(
        self,
        parent: Any,
        *,
        stance: str = DEFAULT_STANCE,
        on_saved: Callable[[], None] | None = None,
    ) -> None:
        self._win = Toplevel(parent)
        self._win.title("检查项设置")
        self._win.geometry("640x680")
        self._win.minsize(540, 480)
        self._on_saved = on_saved
        self._bulk_updating = False
        self.groups: list[CheckGroup] = build_check_groups(stance)
        self._all_ids: list[str] = [
            item["id"] for group in self.groups for item in group.items
        ]
        self._group_ids: dict[str, tuple[str, ...]] = {
            group.key: tuple(item["id"] for item in group.items)
            for group in self.groups
        }
        self.vars: dict[str, BooleanVar] = {}
        self.group_frames: list[ttk.Labelframe] = []
        self.summary_var = StringVar(self._win, value="")

        outer = ttk.Frame(self._win, padding=12)
        outer.pack(fill="both", expand=True)
        self._build_header(outer)
        content = self._build_scroll_area(outer)
        self._build_groups(content, set(load_practice_disabled_checks()))
        self._build_footer(outer)
        self._update_summary()
        self._win.protocol("WM_DELETE_WINDOW", self.close)

    def _build_header(self, outer: ttk.Frame) -> None:
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, textvariable=self.summary_var, wraplength=420).pack(
            side="left", fill="x", expand=True
        )
        self.enable_all_btn = ttk.Button(
            header, text="全部开启（恢复默认）", command=self.enable_all
        )
        self.enable_all_btn.pack(side="right", padx=(8, 0))

    def _build_scroll_area(self, outer: ttk.Frame) -> ttk.Frame:
        container = ttk.Frame(outer)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        canvas = Canvas(container, highlightthickness=0, borderwidth=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scroll.set)
        content = ttk.Frame(canvas, padding=(0, 0, 6, 0))
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind(
            "<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind(
            "<Configure>", lambda e: canvas.itemconfigure(window, width=e.width)
        )
        self._win.bind(
            "<MouseWheel>",
            lambda e: self._scroll_canvas(canvas, -1 if e.delta > 0 else 1)
            if e.delta
            else None,
        )
        self._win.bind("<Button-4>", lambda _e: self._scroll_canvas(canvas, -1))
        self._win.bind("<Button-5>", lambda _e: self._scroll_canvas(canvas, 1))
        return content

    @staticmethod
    def _scroll_canvas(canvas: Canvas, units: int) -> None:
        if canvas.winfo_exists():
            canvas.yview_scroll(units, "units")

    def _build_groups(self, content: ttk.Frame, disabled_set: set[str]) -> None:
        for group in self.groups:
            box = ttk.Labelframe(content, text=group.title, padding=8)
            box.pack(fill="x", pady=(0, 8))
            self.group_frames.append(box)
            actions = ttk.Frame(box)
            actions.pack(fill="x", pady=(0, 4))
            ttk.Button(
                actions,
                text="全选本组",
                command=lambda k=group.key: self.set_group(k, True),
            ).pack(side="left")
            ttk.Button(
                actions,
                text="取消本组",
                command=lambda k=group.key: self.set_group(k, False),
            ).pack(side="left", padx=(6, 0))
            for item in group.items:
                cid = item["id"]
                var = BooleanVar(self._win, value=cid not in disabled_set)
                var.trace_add("write", lambda *_args: self._update_summary())
                self.vars[cid] = var
                ttk.Checkbutton(box, text=check_item_label(item), variable=var).pack(
                    anchor="w", fill="x", pady=1
                )

    def _build_footer(self, outer: ttk.Frame) -> None:
        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(10, 0))
        self.cancel_btn = ttk.Button(footer, text="取消", command=self.close)
        self.cancel_btn.pack(side="right")
        self.save_btn = ttk.Button(footer, text="保存", command=self.save)
        self.save_btn.pack(side="right", padx=(0, 8))

    def _update_summary(self) -> None:
        if self._bulk_updating or not self.is_open():
            return
        enabled = sum(1 for cid in self._all_ids if self.vars[cid].get())
        self.summary_var.set(checks_summary_text(enabled, len(self._all_ids)))

    def is_open(self) -> bool:
        try:
            return bool(self._win.winfo_exists())
        except Exception:
            return False

    def focus(self) -> None:
        try:
            self._win.deiconify()
            self._win.lift()
            self._win.focus_force()
        except Exception:
            pass

    def _set_vars(self, cids: Iterable[str], enabled: bool) -> None:
        self._bulk_updating = True
        try:
            for cid in cids:
                self.vars[cid].set(enabled)
        finally:
            self._bulk_updating = False
        self._update_summary()

    def set_group(self, group_key: str, enabled: bool) -> None:
        """批量勾选或取消指定分组内的全部检查项。"""
        self._set_vars(self._group_ids.get(group_key, ()), enabled)

    def enable_all(self) -> None:
        """恢复全部检查项为开启状态。"""
        self._set_vars(self._all_ids, True)

    def save(self) -> bool:
        """校验并保存当前勾选结果；全部关闭或写盘失败时弹窗提示并保持窗口打开。"""
        enabled_ids = [cid for cid in self._all_ids if self.vars[cid].get()]
        try:
            saved = validate_and_save_checks(self._all_ids, enabled_ids)
        except ValueError as exc:
            messagebox.showwarning("检查项设置", str(exc), parent=self._win)
            return False
        except OSError as exc:
            messagebox.showerror(
                "检查项设置",
                f"保存设置失败，请检查偏好文件读写权限：{exc}",
                parent=self._win,
            )
            return False
        zero_actions = zero_enabled_actions(saved)
        if zero_actions:
            messagebox.showinfo(
                "检查项设置",
                "设置已保存。\n注意：以下动作当前未开启任何检查项，使用前请至少开启 1 项："
                + "、".join(zero_actions),
                parent=self._win,
            )
        if self._on_saved is not None:
            self._on_saved()
        self.close()
        return True

    def close(self) -> None:
        if self.is_open():
            self._win.destroy()
