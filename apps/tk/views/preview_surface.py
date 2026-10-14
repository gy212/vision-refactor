# -*- coding: utf-8 -*-
"""实时预览面：主/侧两路预览控件的显示、双摄布局（竖画面左右并排、其余上下堆叠）与尺寸缓存。

点击某一路预览交给注入的 ``on_click(role)``（旋转该路）；预览尺寸由 Tk 主线程的 Configure 回调写入，
会话线程按 ``size_of(role)`` 缩放预览帧（tuple 读写原子）。
"""
from __future__ import annotations

from collections.abc import Callable
from tkinter import ttk
from typing import Any

from PIL import Image, ImageTk

from core.media import PRIMARY, SECONDARY

LAYOUTS = ("stacked", "side_by_side")


def choose_dual_preview_layout(size: tuple[int, int], size2: tuple[int, int]) -> str:
    """两路都为竖画面时左右并排；横向、方形或混合方向时上下堆叠。"""
    w, h = size
    w2, h2 = size2
    return "side_by_side" if h > w and h2 > w2 else "stacked"


class PreviewSurface:
    """``frame`` 由主窗口放进网格；第二路默认隐藏，只在双摄会话显示。"""

    def __init__(self, parent: Any, *, on_click: Callable[[str], None] | None = None) -> None:
        self.on_click = on_click
        frame = self.frame = ttk.Labelframe(parent, text="预览", padding=10)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        # 第二行/列默认权重 0；单摄隐藏第二路时必须同时清空第二条 grid 轨道的权重，避免主预览被空白挤压。
        frame.rowconfigure(1, weight=0)
        frame.columnconfigure(1, weight=0)
        self._sizes: dict[str, tuple[int, int]] = {PRIMARY: (0, 0), SECONDARY: (0, 0)}
        self._photos: dict[str, Any] = {}
        self.labels = {PRIMARY: ttk.Label(frame, cursor="hand2"), SECONDARY: ttk.Label(frame, cursor="hand2")}
        for row, role in enumerate((PRIMARY, SECONDARY)):
            label = self.labels[role]
            label.grid(row=row, column=0, sticky="nsew")
            label.bind("<Configure>", lambda event, r=role: self._resized(r, event))
            label.bind("<Button-1>", lambda _event, r=role: self._clicked(r))
        self.dual_visible = False
        self.layout = "stacked"
        self.set_dual_visible(False)

    def size_of(self, role: str) -> tuple[int, int]:
        return self._sizes[role]

    def _resized(self, role: str, event: Any) -> None:
        self._sizes[role] = (event.width, event.height)

    def _clicked(self, role: str) -> None:
        if role == SECONDARY and not self.dual_visible:
            return
        if self.on_click is not None:
            self.on_click(role)

    def set_dual_visible(self, visible: bool) -> None:
        """显示/隐藏第二预览，并同步清理不再使用的 grid 轨道。"""
        self.dual_visible = bool(visible)
        if visible:
            self.labels[SECONDARY].grid()
            self.set_layout(self.layout)
            return
        # 仅 grid_remove() 不会释放旧行/列权重，空轨道仍会挤占主预览；坐标和四个权重一起复位。
        self.layout = "stacked"
        self.labels[PRIMARY].grid_configure(row=0, column=0, sticky="nsew")
        self.labels[SECONDARY].grid_configure(row=1, column=0, sticky="nsew")
        self.labels[SECONDARY].grid_remove()
        self._weights(rows=(1, 0), columns=(1, 0))

    def set_layout(self, layout: str) -> None:
        """切换双摄布局：竖画面左右、其他情况上下。"""
        if layout not in LAYOUTS:
            raise ValueError(f"未知双摄预览布局：{layout}")
        self.layout = layout
        self.labels[PRIMARY].grid_configure(row=0, column=0, sticky="nsew")
        if not self.dual_visible:
            self._weights(rows=(1, 0), columns=(1, 0))
        elif layout == "side_by_side":
            self.labels[SECONDARY].grid_configure(row=0, column=1, sticky="nsew")
            self._weights(rows=(1, 0), columns=(1, 1))
        else:
            self.labels[SECONDARY].grid_configure(row=1, column=0, sticky="nsew")
            self._weights(rows=(1, 1), columns=(1, 0))

    def _weights(self, *, rows: tuple[int, int], columns: tuple[int, int]) -> None:
        for index, weight in enumerate(rows):
            self.frame.rowconfigure(index, weight=weight)
        for index, weight in enumerate(columns):
            self.frame.columnconfigure(index, weight=weight)

    def show(self, images: dict[str, Any], sizes: dict[str, tuple[int, int]]) -> None:
        """显示一帧（各路 RGB 数组）；双摄按两路画面方向选择布局。"""
        for role, image in images.items():
            photo = self._photos[role] = ImageTk.PhotoImage(Image.fromarray(image))
            self.labels[role].configure(image=photo)
        if SECONDARY in images:
            layout = choose_dual_preview_layout(sizes[PRIMARY], sizes[SECONDARY])
            if layout != self.layout:
                self.set_layout(layout)
