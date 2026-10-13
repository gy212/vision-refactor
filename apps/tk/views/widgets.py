# -*- coding: utf-8 -*-
"""跨视图复用的小控件：可折叠分区 ``CollapsibleSection``。"""
from __future__ import annotations

from tkinter import BooleanVar, ttk


class CollapsibleSection:
    """带切换按钮的折叠分区：``header`` 放切换按钮，``content`` 放分区内容。"""

    def __init__(self, parent: ttk.Frame, title: str, expanded: bool = False) -> None:
        self.parent = parent
        self._expanded = BooleanVar(value=expanded)

        self.header = ttk.Frame(parent)
        self._toggle_btn = ttk.Button(
            self.header,
            text=self._get_toggle_text(),
            command=self._toggle,
            width=len(title) + 4,
        )
        self._toggle_btn.pack(side="left", anchor="w")

        self.content = ttk.Frame(parent)
        if expanded:
            self.content.pack(fill="x", pady=(4, 0))

    def _get_toggle_text(self) -> str:
        arrow = "▾" if self._expanded.get() else "▸"
        return f"{arrow} {self._toggle_btn.cget('text').lstrip('▸▾ ') if hasattr(self, '_toggle_btn') else ''}"

    def _toggle(self) -> None:
        self._expanded.set(not self._expanded.get())
        current_text = self._toggle_btn.cget("text")
        base_text = current_text.lstrip("▸▾ ")
        arrow = "▾" if self._expanded.get() else "▸"
        self._toggle_btn.configure(text=f"{arrow} {base_text}")

        if self._expanded.get():
            self.content.pack(fill="x", pady=(4, 0))
        else:
            self.content.pack_forget()

    def toggle(self) -> None:
        """展开或收起。"""
        self._toggle()

    def set_title(self, title: str) -> None:
        arrow = "▾" if self._expanded.get() else "▸"
        self._toggle_btn.configure(text=f"{arrow} {title}", width=len(title) + 4)

    def pack_header(self, **kwargs) -> None:
        self.header.pack(**kwargs)

    def is_expanded(self) -> bool:
        return self._expanded.get()
