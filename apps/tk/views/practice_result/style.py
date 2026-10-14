# -*- coding: utf-8 -*-
"""结果页的 Style A 视觉规范：配色令牌、字体、卡片与按钮的统一构造，以及本地状态图标的加载。

固定浅色底 #F5F7FB、白色卡片、细边线 #D8E0EA；四类核心状态与"教师已撤销"各有前景/底色/边线。
不展示总分、扣分、合格率或进度条。
"""
from __future__ import annotations

import logging
from pathlib import Path
import tkinter as tk
from typing import Any

from PIL import Image, ImageTk

logger = logging.getLogger(__name__)

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

FONT = "Microsoft YaHei UI"
_ICON_FILES = {
    "candidate": "candidate_24.png",
    "not_observed": "not_observed_24.png",
    "unable": "unable_24.png",
    "pending_rule": "pending_rule_24.png",
}


def assets_dir() -> Path:
    """结果页状态图标目录：源码模式在仓库根、冻结模式在 PyInstaller 解包目录下的 ``assets/feedback``。"""
    return Path(__file__).resolve().parents[4] / "assets" / "feedback"


def load_status_icons(master: tk.Misc) -> dict[str, ImageTk.PhotoImage]:
    """加载本地状态 PNG 图标（24px）；缺失或损坏的图标跳过。"""
    icons: dict[str, ImageTk.PhotoImage] = {}
    for key, fname in _ICON_FILES.items():
        path = assets_dir() / fname
        if path.is_file():
            try:
                icons[key] = ImageTk.PhotoImage(Image.open(path), master=master)
            except Exception as exc:
                logger.debug("Failed to load icon %s: %s", path, exc)
    return icons


def card(parent: tk.Misc, *, bg: str | None = None, border: str | None = None, padx: int = 12, pady: int = 8) -> tk.Frame:
    """带细边线的卡片容器。"""
    return tk.Frame(
        parent,
        bg=bg or STYLE_A["bg_card"],
        bd=1,
        relief="solid",
        highlightbackground=border or STYLE_A["border"],
        highlightthickness=1,
        padx=padx,
        pady=pady,
    )


def label(parent: tk.Misc, *, size: int = 9, bold: bool = False, bg: str | None = None,
          fg: str | None = None, **kwargs: Any) -> tk.Label:
    """统一字体与配色的标签；``kwargs`` 透传 ``text``/``textvariable``/``anchor`` 等。"""
    font = (FONT, size, "bold") if bold else (FONT, size)
    return tk.Label(parent, font=font, bg=bg or STYLE_A["bg_card"], fg=fg or STYLE_A["text_primary"], **kwargs)


def toggle_colors(active: bool) -> dict[str, str]:
    """筛选/阶段按钮的选中与未选中配色。"""
    if active:
        return {"bg": STYLE_A["accent_blue_bg"], "fg": STYLE_A["accent_blue"]}
    return {"bg": STYLE_A["bg_card"], "fg": STYLE_A["text_primary"]}
