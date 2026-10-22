# -*- coding: utf-8 -*-
"""录制性能测试窗口的布局：说明与报告保存位置、开始/取消按钮、进度、结果摘要和报告路径（可复制、可打开所在文件夹）。

窗口是模态的：测试期间主界面不能操作，避免开预览或录制去抢摄像头。流程与保存在 ``BenchmarkController``。
"""
from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from tkinter import DoubleVar, StringVar, Text, Tk, Toplevel, messagebox, ttk
from typing import Any

from apps.tk.controllers.benchmark import BenchmarkContext, BenchmarkController, report_directory, total_seconds

INTRO = (
    "这个测试检查本机能否同时稳定地采集并录制两路 1080p 画面，结果用来决定录制方案。\n\n"
    "开始前：在主界面选好正面和侧面摄像头（就是平时录制用的两个），不用点「开始」。\n"
    "测试分四段，共约 {minutes} 分钟：只采集 → 采集并录像 → 再加人体检测 → 再加骨架识别。\n"
    "测试期间请不要拔摄像头，也不要运行其它占用电脑的程序；摄像头前有没有人都可以。\n"
    "测试录下的视频会自动删除，不会留在录制目录里。"
)


class BenchmarkDialog:
    def __init__(self, parent: Tk, post: Callable[[Callable[[], Any]], Any], ctx: BenchmarkContext) -> None:
        self._win = Toplevel(parent)
        self._win.title("录制性能测试")
        self._win.geometry("620x620")
        self._win.minsize(560, 560)
        self._win.transient(parent)

        self.progress_var = StringVar(value="尚未开始")
        self.progress_value = DoubleVar(value=0.0)
        self.path_var = StringVar(value="")
        self._report_path: Path | None = None

        guarded = lambda cb: post(lambda: self._win.winfo_exists() and cb())
        self.controller = BenchmarkController(replace(ctx, post=guarded), self)
        outer = ttk.Frame(self._win, padding=12)
        outer.pack(fill="both", expand=True)
        self._build_intro(outer)
        self._build_progress(outer)
        self._build_result(outer)
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)
        self._win.grab_set()

    def is_open(self) -> bool:
        return bool(self._win.winfo_exists())

    def focus(self) -> None:
        try:
            self._win.deiconify()
            self._win.lift()
            self._win.focus_force()
        except Exception:
            pass

    # ------------------------------------------------------------------ 布局
    def _build_intro(self, outer: ttk.Frame) -> None:
        minutes = max(1, round((total_seconds() + 30) / 60))
        ttk.Label(outer, text=INTRO.format(minutes=minutes), wraplength=580, justify="left").pack(anchor="w")
        where = ttk.Labelframe(outer, text="报告保存位置", padding=8)
        where.pack(fill="x", pady=(10, 0))
        ttk.Label(where, text=f"{report_directory()}\n文件名：录制性能测试_日期_时间.json（测试结束后显示完整路径）",
                  wraplength=560, justify="left").pack(anchor="w")
        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(10, 0))
        self.start_btn = ttk.Button(buttons, text="开始测试", command=self.controller.start)
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(buttons, text="取消测试", command=self.controller.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=(8, 0))
        self.close_btn = ttk.Button(buttons, text="关闭", command=self._on_close)
        self.close_btn.pack(side="right")

    def _build_progress(self, outer: ttk.Frame) -> None:
        frame = ttk.Frame(outer)
        frame.pack(fill="x", pady=(10, 0))
        ttk.Label(frame, textvariable=self.progress_var, wraplength=580).pack(anchor="w")
        ttk.Progressbar(frame, variable=self.progress_value, maximum=100.0).pack(fill="x", pady=(4, 0))

    def _build_result(self, outer: ttk.Frame) -> None:
        frame = ttk.Labelframe(outer, text="测试结果", padding=8)
        frame.pack(fill="both", expand=True, pady=(10, 0))
        self.summary = Text(frame, height=10, wrap="word", state="disabled")
        self.summary.pack(fill="both", expand=True)
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text="报告文件：").pack(side="left")
        self.path_entry = ttk.Entry(row, textvariable=self.path_var, state="readonly")
        self.path_entry.pack(side="left", fill="x", expand=True)
        actions = ttk.Frame(frame)
        actions.pack(fill="x", pady=(6, 0))
        self.open_btn = ttk.Button(actions, text="打开所在文件夹", command=self.open_folder, state="disabled")
        self.open_btn.pack(side="left")
        self.copy_btn = ttk.Button(actions, text="复制路径", command=self.copy_path, state="disabled")
        self.copy_btn.pack(side="left", padx=(8, 0))

    # ------------------------------------------------------------------ 控制器回调
    def set_running(self, running: bool) -> None:
        self.start_btn.configure(state="disabled" if running else "normal",
                                 text="开始测试" if running or self._report_path is None else "再测一次")
        self.cancel_btn.configure(state="normal" if running else "disabled")
        if running:
            self._report_path = None
            self.path_var.set("")
            self._set_summary("测试进行中……")
            for button in (self.open_btn, self.copy_btn):
                button.configure(state="disabled")

    def show_progress(self, text: str, done: float) -> None:
        self.progress_var.set(text)
        self.progress_value.set(max(0.0, min(1.0, float(done))) * 100.0)

    def show_problem(self, text: str) -> None:
        self.progress_var.set(text)
        messagebox.showwarning("暂时不能测试", text, parent=self._win)

    def show_result(self, summary: list[str], ok: bool, path: Path | None, error: str | None) -> None:
        self._report_path = path
        lines = list(summary)
        if path is not None:
            lines += ["", "报告已保存到：", str(path), "请把这个文件发给技术人员。"]
            self.path_var.set(str(path))
            for button in (self.open_btn, self.copy_btn):
                button.configure(state="normal")
        if error:
            lines += ["", error]
        self._set_summary("\n".join(lines))
        self.progress_var.set("测试完成，报告已保存。" if path is not None else "测试结束，但报告没有保存成功。")
        self.start_btn.configure(text="再测一次")

    def _set_summary(self, text: str) -> None:
        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("1.0", text)
        self.summary.configure(state="disabled")

    # ------------------------------------------------------------------ 报告文件
    def open_folder(self) -> None:
        path = self._report_path
        if path is None:
            return
        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", f"/select,{path}"])
            else:
                subprocess.Popen(["xdg-open", str(path.parent)])
        except OSError as exc:
            messagebox.showerror("打开失败", f"无法打开文件夹：{exc}\n报告路径：{path}", parent=self._win)

    def copy_path(self) -> None:
        if self._report_path is None:
            return
        self._win.clipboard_clear()
        self._win.clipboard_append(str(self._report_path))
        self.progress_var.set("报告路径已复制，可以粘贴到微信或文件选择框里。")

    def _on_close(self) -> None:
        if self.controller.running:
            if not messagebox.askyesno("测试进行中", "测试还没结束，确定要取消并关闭吗？", parent=self._win):
                return
            self.controller.close()
        try:
            self._win.grab_release()
        except Exception:
            pass
        self._win.destroy()

    def close(self) -> None:
        """主程序退出时调用：取消测试并关窗，不再询问。"""
        self.controller.close()
        if self._win.winfo_exists():
            self._win.destroy()
