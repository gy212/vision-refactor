# -*- coding: utf-8 -*-
"""设置窗口的布局：当前模型汇总、MediaPipe 模型逐行状态与下载按钮、下载进度与状态区。

本期只提供 MediaPipe 模型下载；状态文字与下载任务在 ``SettingsController``。
"""
from __future__ import annotations

from collections.abc import Callable
from tkinter import StringVar, Tk, Toplevel, messagebox, ttk
from typing import Any

from core.foundation import models_dir
from core.pose import MEDIAPIPE_MODELS, ModelSpec, is_installed

from apps.tk.controllers.settings import SettingsController, active_model_keys, current_summary, model_state


class SettingsDialog:
    def __init__(self, parent: Tk, post: Callable[[Callable[[], Any]], Any], *,
                 current_pose: str = "full", hands_enabled: bool = True) -> None:
        self._win = Toplevel(parent)
        self._win.title("设置")
        self._win.geometry("560x560")
        self._win.minsize(520, 500)

        # 主界面当前选择：用于在模型列表中标注「当前使用」并在顶部汇总展示。
        self._current_pose = (current_pose or "full").strip().lower()
        self._hands_enabled = bool(hands_enabled)
        self._active_keys = active_model_keys(self._current_pose, self._hands_enabled)

        self.status_var = StringVar(value="就绪")
        self.progress_text_var = StringVar(value="")
        self.current_model_var = StringVar(value="")
        # 每个模型一行的控件引用：key -> dict(spec, state_var, btn)
        self._rows: dict[str, dict] = {}

        # 后台线程只经 UiDispatcher 回到 Tk 线程；窗口已关时丢弃
        self.controller = SettingsController(self, lambda cb: post(lambda: self._win.winfo_exists() and cb()))
        outer = ttk.Frame(self._win, padding=12)
        outer.pack(fill="both", expand=True)
        self._build_models(outer)
        self._build_status(outer)
        self.refresh_states()
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

    def is_open(self) -> bool:
        return bool(self._win.winfo_exists())

    def focus(self) -> None:
        try:
            self._win.deiconify()
            self._win.lift()
            self._win.focus_force()
        except Exception:
            pass

    def _build_models(self, outer: ttk.Frame) -> None:
        current_frame = ttk.Labelframe(outer, text="当前模型", padding=10)
        current_frame.pack(fill="x")
        ttk.Label(current_frame, textvariable=self.current_model_var, wraplength=510).pack(anchor="w")

        mp_frame = ttk.Labelframe(outer, text="MediaPipe 模型", padding=10)
        mp_frame.pack(fill="x", pady=(12, 0))
        ttk.Label(mp_frame, text="模型目录：" + str(models_dir()), wraplength=500).pack(anchor="w", pady=(0, 8))
        for spec in MEDIAPIPE_MODELS:
            row = ttk.Frame(mp_frame)
            row.pack(fill="x", pady=(4, 0))
            ttk.Label(row, text=spec.label, width=34, anchor="w").grid(row=0, column=0, sticky="w")
            state_var = StringVar(value="检查中…")
            ttk.Label(row, textvariable=state_var, width=14, anchor="w").grid(row=0, column=1, sticky="w", padx=(8, 0))
            btn = ttk.Button(row, text="下载", width=8, command=lambda s=spec: self.controller.download([s]))
            btn.grid(row=0, column=2, sticky="e", padx=(8, 0))
            row.columnconfigure(0, weight=1)
            self._rows[spec.key] = {"spec": spec, "state_var": state_var, "btn": btn}

        bulk_row = ttk.Frame(mp_frame)
        bulk_row.pack(fill="x", pady=(12, 0))
        self.download_all_btn = ttk.Button(bulk_row, text="下载全部缺失模型", command=self.download_missing)
        self.download_all_btn.pack(side="left")
        self.refresh_btn = ttk.Button(bulk_row, text="刷新状态", command=self.refresh_states)
        self.refresh_btn.pack(side="left", padx=(8, 0))

    def _build_status(self, outer: ttk.Frame) -> None:
        status_frame = ttk.Labelframe(outer, text="状态", padding=10)
        status_frame.pack(fill="both", expand=True, pady=(12, 0))
        ttk.Label(status_frame, textvariable=self.status_var, wraplength=510).pack(anchor="w")
        self.progress_bar = ttk.Progressbar(status_frame, orient="horizontal", mode="determinate", maximum=100.0)
        self.progress_bar.pack(fill="x", pady=(8, 0))
        ttk.Label(status_frame, textvariable=self.progress_text_var, wraplength=510).pack(anchor="w", pady=(4, 0))
        ttk.Label(
            status_frame,
            text="说明：模型从 Google 官方源下载。若长时间无进度或失败，通常是网络无法访问"
            " storage.googleapis.com，请自行配置代理后重试。",
            wraplength=510,
            foreground="#666",
        ).pack(anchor="w", pady=(10, 0))

    def download_missing(self) -> None:
        missing: list[ModelSpec] = [info["spec"] for info in self._rows.values() if not is_installed(info["spec"])]
        if not missing:
            messagebox.showinfo("模型管理", "所有模型均已安装。", parent=self._win)
            return
        self.controller.download(missing)

    def _on_close(self) -> None:
        self.controller.close()
        self._win.destroy()

    # ------------------------------------------------------------------ SettingsView
    def refresh_states(self) -> None:
        """刷新每个模型的「已安装 / 未安装」状态显示，并标注当前使用的模型。"""
        for key, info in self._rows.items():
            state, button = model_state(info["spec"], key in self._active_keys)
            info["state_var"].set(state)
            info["btn"].configure(text=button)
        specs = [info["spec"] for info in self._rows.values()]
        self.current_model_var.set(current_summary(self._current_pose, self._hands_enabled, self._active_keys, specs))

    def set_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for info in self._rows.values():
            info["btn"].configure(state=state)
        self.download_all_btn.configure(state=state)
        self.refresh_btn.configure(state=state)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def set_progress(self, downloaded: int, total: int | None) -> None:
        self.progress_bar.configure(mode="determinate", maximum=100.0)
        if total and total > 0:
            pct = (downloaded / total) * 100.0
            self.progress_bar["value"] = pct
            self.progress_text_var.set(f"{downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB（{pct:.0f}%）")
        else:
            self.progress_bar["value"] = 0.0
            self.progress_text_var.set(f"{downloaded / 1024 / 1024:.1f} MB…")

    def show_download_errors(self, detail: str) -> None:
        messagebox.showerror(
            "下载失败",
            "以下模型下载失败（多为网络无法访问 Google 源，请配置代理后重试）：\n\n" + detail,
            parent=self._win,
        )
