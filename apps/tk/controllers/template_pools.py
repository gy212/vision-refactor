# -*- coding: utf-8 -*-
"""录后自动比对的模板池：正/侧模板的添加、移除、校验、持久化与提示文字；录后计划与预检读取这里缓存的模板。

模板在 Tk 线程里缓存为普通路径（``lists``），录后计划可能在录制的任务线程里构建，不读 Tk 变量。
"""
from __future__ import annotations

from pathlib import Path
from tkinter import messagebox
from typing import Any

from core.foundation import save_template_pools
from core.recording import try_load_heavy_template


def template_status_text(front: int, side: int) -> str:
    if front and side:
        return f"模板池：正面 {front} 个 · 侧面 {side} 个（正对正、侧对侧，逐动作出分）"
    if front:
        return f"模板池：正面 {front} 个 · 侧面 0 个（仅比对正面；两侧都空才报错）"
    if side:
        return f"模板池：正面 0 个 · 侧面 {side} 个（仅比对侧面；两侧都空才报错）"
    return "模板池：正/侧均为空 — 开启自动比对将失败"


class TemplatePools:
    """``panel`` 是录制面板（模板列表、提示文字与文件选择）。"""

    def __init__(self, panel: Any) -> None:
        self.panel = panel
        self._templates = (tuple(panel.pool_paths("front")), tuple(panel.pool_paths("side")))

    def lists(self) -> tuple[list[Path], list[Path]]:
        front, side = self._templates
        return list(front), list(side)

    def add(self, view: str) -> None:
        paths = self.panel.ask_templates(view)
        if not paths:
            return
        existing = {str(p) for p in self.panel.pool_paths(view)}
        rejected: list[str] = []
        for path in paths:
            usable = try_load_heavy_template(path)
            if usable is None:
                rejected.append(Path(path).name)
            elif str(usable) not in existing:
                self.panel.add_pool_path(view, usable)
                existing.add(str(usable))
        self.changed()
        if rejected:
            messagebox.showerror(
                "部分模板无效",
                "以下文件不是可用的 heavy + pose33_v3 模板，已跳过：\n" + "\n".join(rejected)
                + "\n请用 make_template.py --pose heavy 生成后再添加。",
                parent=self.panel.root,
            )

    def remove(self, view: str) -> None:
        self.panel.remove_selected(view)
        self.changed()

    def changed(self) -> None:
        """池变化后缓存为普通路径、持久化并刷新提示。"""
        self._templates = (tuple(self.panel.pool_paths("front")), tuple(self.panel.pool_paths("side")))
        try:
            save_template_pools(front=list(self._templates[0]), side=list(self._templates[1]))
        except Exception:
            pass
        self.sync_status()

    def sync_status(self) -> None:
        self.panel.template_status_var.set(template_status_text(*map(len, self._templates)))
