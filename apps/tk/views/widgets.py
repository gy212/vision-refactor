# -*- coding: utf-8 -*-
"""跨视图复用的小控件：可折叠分区 ``CollapsibleSection``、进度条组 ``ProgressPanel`` 与视频帧缩略图槽 ``FramePreviewSlot``。"""

from __future__ import annotations

from tkinter import BooleanVar, StringVar, ttk
from typing import Any

import cv2
from PIL import Image, ImageTk


def _button_width(title: str) -> int:
    """按全角 2 栏、半角 1 栏估算 ttk.Button 宽度，避免中文标题被截断。"""
    display = sum(2 if ord(ch) > 127 else 1 for ch in title)
    return max(12, display + 4)


class CollapsibleSection:
    """带切换按钮的折叠分区：``header`` 放切换按钮，``content`` 放分区内容。"""

    def __init__(self, parent: ttk.Frame, title: str, expanded: bool = False) -> None:
        self.parent = parent
        self._title = title
        self._expanded = BooleanVar(value=expanded)

        self.header = ttk.Frame(parent)
        self._toggle_btn = ttk.Button(
            self.header,
            text=self._toggle_text(),
            command=self._toggle,
            width=_button_width(title),
        )
        self._toggle_btn.pack(side="left", anchor="w")

        self.content = ttk.Frame(parent)
        if expanded:
            self.content.pack(fill="x", pady=(4, 0))

    def _toggle_text(self) -> str:
        arrow = "▾" if self._expanded.get() else "▸"
        return f"{arrow} {self._title}"

    def _toggle(self) -> None:
        self._expanded.set(not self._expanded.get())
        self._toggle_btn.configure(text=self._toggle_text())

        if self._expanded.get():
            self.content.pack(fill="x", pady=(4, 0))
        else:
            self.content.pack_forget()

    def toggle(self) -> None:
        """展开或收起。"""
        self._toggle()

    def set_title(self, title: str) -> None:
        self._title = title
        self._toggle_btn.configure(text=self._toggle_text(), width=_button_width(title))

    def pack_header(self, **kwargs) -> None:
        self.header.pack(**kwargs)

    def is_expanded(self) -> bool:
        return self._expanded.get()


class ProgressPanel:
    """可隐藏的 determinate 进度条 + 状态文案，供比对与制模弹窗共用。"""

    def __init__(self, parent: ttk.Frame, text_var: StringVar) -> None:
        self.frame = ttk.Frame(parent)
        self.text_var = text_var
        self.bar = ttk.Progressbar(
            self.frame, orient="horizontal", mode="determinate", maximum=100.0
        )
        self.bar.pack(fill="x")
        ttk.Label(self.frame, textvariable=self.text_var).pack(anchor="w", pady=(4, 0))
        self.hide()

    def show(
        self, stage: str, done: int, total: int, *, pady: tuple[int, int] = (8, 0)
    ) -> None:
        self.frame.pack(fill="x", pady=pady)
        self.bar.configure(
            mode="determinate", maximum=float(total) if total > 0 else 100.0
        )
        if total > 0:
            self.bar["value"] = float(done)
            self.text_var.set(
                f"{stage}：{done}/{total}（{(done / total) * 100.0:.1f}%）"
            )
        else:
            self.bar["value"] = 0.0
            self.text_var.set(f"{stage}…")

    def hide(self) -> None:
        self.frame.pack_forget()


class FramePreviewSlot:
    """单路视频帧缩略图框（保持 PhotoImage 引用防 GC）。"""

    def __init__(
        self,
        parent: ttk.Frame,
        title: str,
        column: int,
        *,
        preview_size: tuple[int, int] = (340, 200),
    ) -> None:
        self._size = preview_size
        self.frame = ttk.Labelframe(parent, text=title, padding=6)
        self.frame.grid(
            row=0, column=column, sticky="nsew", padx=(0, 6) if column == 0 else (6, 0)
        )
        self.title_var = StringVar(value="未加载")
        ttk.Label(self.frame, textvariable=self.title_var, foreground="#555555").pack(
            anchor="w"
        )
        self.image_label = ttk.Label(self.frame, text="暂无画面", anchor="center")
        self.image_label.pack(fill="both", expand=True, pady=(4, 0))
        self.photo: ImageTk.PhotoImage | None = None

    def show_frame(self, title: str, bgr: Any, *, master: Any) -> None:
        self.title_var.set(title)
        if bgr is None:
            self.photo = None
            self.image_label.configure(image="", text="无法解码当前帧")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail(self._size)
        photo = ImageTk.PhotoImage(pil_img, master=master)
        self.photo = photo
        self.image_label.configure(image=photo, text="")

    def clear(self, title: str = "未加载") -> None:
        self.title_var.set(title)
        self.photo = None
        self.image_label.configure(image="", text="暂无画面")
