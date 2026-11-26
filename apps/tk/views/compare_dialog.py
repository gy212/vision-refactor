# -*- coding: utf-8 -*-
"""动作分析（模板比对）窗口的布局：模板池、唤起制作模板弹窗、目标视频、执行区与结果区。

按钮只读输入并调用 ``CompareController``；后台结果经 dispatcher 回到视图的 ``CompareView`` 方法。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from tkinter import (
    IntVar,
    Listbox,
    Scrollbar,
    StringVar,
    TclError,
    Text,
    Tk,
    Toplevel,
    filedialog,
    messagebox,
    ttk,
)

from apps.tk.controllers.compare import CompareController, usable_template
from apps.tk.views.widgets import CollapsibleSection, ProgressPanel

logger = logging.getLogger(__name__)

_VIDEO_TYPES = [("视频文件", "*.mp4;*.avi;*.mov;*.mkv"), ("所有文件", "*.*")]


def score_color(score: float) -> str:
    """分数 → 红黄绿。≥0.8 绿 / ≥0.5 黄 / 否则红。"""
    if score >= 0.8:
        return "#2e7d32"
    if score >= 0.5:
        return "#f9a825"
    return "#c62828"


class CompareDialog:
    def __init__(
        self,
        parent: Tk,
        post: Callable[[Callable[[], Any]], Any],
        *,
        open_builder: Callable[[Callable[[list[Any]], None]], None] | None = None,
    ) -> None:
        self._win = Toplevel(parent)
        self._win.title("动作分析（模板比对）")
        self._win.geometry("720x820")
        self._win.minsize(640, 680)

        self._open_builder = open_builder
        self.target_video_var = StringVar(value="")
        self.workers_var = IntVar(value=1)
        self.status_var = StringVar(value="就绪")
        self.result_var = StringVar(value="")
        self.progress_text_var = StringVar(value="")
        self._action_rows: list[tuple[ttk.Label, ttk.Label]] = []
        self.last_score: float | None = None

        # 后台线程只经 UiDispatcher 回到 Tk 线程；窗口已关时丢弃
        self.controller = CompareController(
            self, lambda cb: post(lambda: self._win.winfo_exists() and cb())
        )
        outer = ttk.Frame(self._win, padding=12)
        outer.pack(fill="both", expand=True)
        self._build_pool(outer)
        self._build_target(outer)
        self._build_run(outer)
        self._build_results(outer)
        self.reset_scores()
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

    def is_open(self) -> bool:
        return bool(self._win.winfo_exists())

    def focus(self) -> None:
        try:
            self._win.deiconify()
            self._win.lift()
            self._win.focus_force()
        except TclError as exc:
            logger.debug("聚焦动作分析窗口失败：%s", exc)

    def close(self) -> None:
        self.controller.close()
        if self._win.winfo_exists():
            self._win.destroy()

    # ------------------------------------------------------------------ 布局
    def _build_pool(self, outer: ttk.Frame) -> None:
        """① 模板池（池内每个 .npz 各出一个动作分，动作名 = 文件 stem）与制作模板入口。"""
        step1 = ttk.Labelframe(
            outer, text="① 模板池（每个模板对应一个动作，各出一个成绩）", padding=10
        )
        step1.pack(fill="x")
        body = ttk.Frame(step1)
        body.pack(fill="x")

        pool_row = ttk.Frame(body)
        pool_row.pack(fill="x")
        lb_box = ttk.Frame(pool_row)
        lb_box.pack(side="left", fill="both", expand=True)
        self.template_listbox = Listbox(
            lb_box, height=5, selectmode="extended", exportselection=False
        )
        self.template_listbox.pack(side="left", fill="both", expand=True)
        pool_sb = Scrollbar(lb_box, command=self.template_listbox.yview)
        pool_sb.pack(side="right", fill="y")
        self.template_listbox.configure(yscrollcommand=pool_sb.set)

        pool_btns = ttk.Frame(pool_row)
        pool_btns.pack(side="left", fill="y", padx=(8, 0))
        self.add_tpl_btn = ttk.Button(
            pool_btns, text="添加模板…", command=self._add_templates
        )
        self.add_tpl_btn.pack(fill="x")
        self.remove_tpl_btn = ttk.Button(
            pool_btns, text="移除选中", command=self._remove_selected_templates
        )
        self.remove_tpl_btn.pack(fill="x", pady=(6, 0))
        self.clear_tpl_btn = ttk.Button(
            pool_btns, text="清空", command=self._clear_templates
        )
        self.clear_tpl_btn.pack(fill="x", pady=(6, 0))

        self.build_tpl_btn = ttk.Button(
            body,
            text="从视频制作模板…",
            command=self._open_template_builder,
            state="normal" if self._open_builder is not None else "disabled",
        )
        self.build_tpl_btn.pack(fill="x", pady=(10, 0))

    def _build_target(self, outer: ttk.Frame) -> None:
        step2 = ttk.Labelframe(outer, text="② 选择目标视频", padding=10)
        step2.pack(fill="x", pady=(12, 0))
        tgt_row = ttk.Frame(step2)
        tgt_row.pack(fill="x")
        ttk.Label(tgt_row, text="目标视频：").grid(row=0, column=0, sticky="w")
        ttk.Entry(tgt_row, textvariable=self.target_video_var).grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )
        self.browse_target_btn = ttk.Button(
            tgt_row, text="选择…", command=self._browse_target
        )
        self.browse_target_btn.grid(row=0, column=2, padx=(8, 0))
        tgt_row.columnconfigure(1, weight=1)
        ttk.Label(tgt_row, text="线程数：").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.workers_spin = ttk.Spinbox(
            tgt_row, from_=1, to=16, textvariable=self.workers_var, width=6
        )
        self.workers_spin.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

    def _build_run(self, outer: ttk.Frame) -> None:
        step3 = ttk.Labelframe(outer, text="③ 执行分析", padding=10)
        step3.pack(fill="x", pady=(12, 0))
        btn_row = ttk.Frame(step3)
        btn_row.pack(fill="x")
        self.start_btn = ttk.Button(
            btn_row, text="═══ 开始分析 ═══", command=self.start_compare
        )
        self.start_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(
            btn_row, text="停止", command=self.controller.stop, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=(8, 0))

        self._progress = ProgressPanel(step3, self.progress_text_var)
        self.progress_bar = self._progress.bar

    def _build_results(self, outer: ttk.Frame) -> None:
        result_frame = ttk.Labelframe(outer, text="结果", padding=10)
        result_frame.pack(fill="both", expand=True, pady=(12, 0))
        ttk.Label(result_frame, textvariable=self.status_var).pack(anchor="w")

        score_frame = ttk.Frame(result_frame)
        score_frame.pack(fill="x", pady=(8, 0))
        self._score_label = ttk.Label(
            score_frame, text="—", font=("Helvetica", 36, "bold"), anchor="center"
        )
        self._score_label.pack()
        ttk.Label(score_frame, text="综合分（各动作平均）", anchor="center").pack()

        # 逐动作成绩列表：每个模板一行「动作名 …… 百分比」，颜色随分数。
        self._action_frame = ttk.Labelframe(result_frame, text="各动作成绩", padding=8)
        self._action_frame.pack(fill="x", pady=(8, 0))
        ttk.Label(result_frame, textvariable=self.result_var, wraplength=600).pack(
            anchor="w", pady=(8, 0)
        )

        json_section = CollapsibleSection(result_frame, "查看详细数据", expanded=False)
        json_section.set_title("查看详细数据")
        json_section.pack_header(fill="x", pady=(10, 0))
        raw_box = ttk.Frame(json_section.content)
        raw_box.pack(fill="both", expand=True, pady=(4, 0))
        self.raw_text = Text(raw_box, height=10, wrap="none")
        self.raw_text.pack(side="left", fill="both", expand=True)
        sb = Scrollbar(raw_box, command=self.raw_text.yview)
        sb.pack(side="right", fill="y")
        self.raw_text.configure(yscrollcommand=sb.set)
        ttk.Button(json_section.content, text="复制数据", command=self._copy_raw).pack(
            anchor="e", pady=(6, 0)
        )

    # ------------------------------------------------------------------ 按钮
    def _ask_video(self, title: str, var: StringVar) -> None:
        path = filedialog.askopenfilename(
            title=title, filetypes=_VIDEO_TYPES, parent=self._win
        )
        self._win.lift()
        self._win.focus_force()
        if path:
            var.set(path)

    def _browse_target(self) -> None:
        self._ask_video("选择目标视频", self.target_video_var)

    def _open_template_builder(self) -> None:
        if self.controller.busy or self._open_builder is None:
            return
        self._open_builder(self._on_templates_built)

    def _on_templates_built(self, results: list[Any]) -> None:
        if not self.is_open():
            return
        for res in results:
            self.add_template(str(res.path))

    def pool_paths(self) -> list[str]:
        """当前模板池中的所有路径（去空）。"""
        return [s for s in self.template_listbox.get(0, "end") if str(s).strip()]

    def _add_templates(self) -> None:
        paths = filedialog.askopenfilenames(
            title="添加模板（可多选）",
            filetypes=[("模板文件", "*.npz"), ("所有文件", "*.*")],
            parent=self._win,
        )
        self._win.lift()
        self._win.focus_force()
        for p in paths:
            self.add_template(p)

    def _remove_selected_templates(self) -> None:
        for i in reversed(self.template_listbox.curselection()):
            self.template_listbox.delete(i)

    def _clear_templates(self) -> None:
        self.template_listbox.delete(0, "end")

    def start_compare(self) -> None:
        if self.controller.busy:
            return
        templates = self.pool_paths()
        video = self.target_video_var.get().strip()
        if not video:
            messagebox.showerror("配置错误", "请先选择目标视频。", parent=self._win)
            return
        if not templates:
            messagebox.showerror(
                "配置错误",
                "模板池为空，请先添加模板（.npz），或从视频制作模板。",
                parent=self._win,
            )
            return
        try:
            workers = max(1, int(self.workers_var.get() or 1))
        except (TclError, TypeError, ValueError):
            messagebox.showerror(
                "配置错误", "线程数请输入 1 到 16 的整数。", parent=self._win
            )
            return
        self.controller.compare(templates, video, workers)

    def _copy_raw(self) -> None:
        try:
            s = self.raw_text.get("1.0", "end").strip()
            self._win.clipboard_clear()
            self._win.clipboard_append(s)
            self.status_var.set("已复制原始数据到剪贴板")
        except Exception as e:
            messagebox.showerror("复制失败", str(e), parent=self._win)

    def _on_close(self) -> None:
        self.close()

    # ------------------------------------------------------------------ CompareView
    def add_template(self, path: str) -> None:
        """校验后加入池（heavy + pose33_v3），去重；无效模板弹窗提示后跳过。"""
        usable, invalid = usable_template(path, self.pool_paths())
        if invalid:
            messagebox.showerror(
                "模板无效",
                f"{Path(path).name} 不是可用的 heavy + pose33_v3 模板，已跳过。\n"
                "请用 make_template.py --pose heavy 生成后再添加。",
                parent=self._win,
            )
        elif usable is not None:
            self.template_listbox.insert("end", usable)

    def show_progress(self, stage: str, done: int, total: int) -> None:
        self._progress.show(stage, done, total)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def set_result(self, text: str) -> None:
        self.result_var.set(text)

    def set_raw(self, text: str) -> None:
        self.raw_text.delete("1.0", "end")
        self.raw_text.insert("1.0", text)

    def set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.start_btn.configure(state=state)
        self.stop_btn.configure(state="normal" if running else "disabled")
        for widget in (
            self.add_tpl_btn,
            self.remove_tpl_btn,
            self.clear_tpl_btn,
            self.browse_target_btn,
            self.workers_spin,
        ):
            widget.configure(state=state)
        self.build_tpl_btn.configure(
            state="normal"
            if (not running and self._open_builder is not None)
            else "disabled"
        )
        if not running:
            self._progress.hide()

    def reset_scores(self) -> None:
        self.last_score = None
        self._score_label.configure(text="—", foreground="")
        self.result_var.set("")
        self._clear_action_rows()

    def show_scores(
        self, actions: list[tuple[str, float]], match_info: str = ""
    ) -> None:
        """刷新综合分 + 逐动作成绩列表。actions = [(动作名, 0..1 分), ...]；综合分 = 各动作分平均。"""
        self._clear_action_rows()
        if not actions:
            self._score_label.configure(text="—", foreground="")
            self.result_var.set(match_info)
            return
        combined = sum(s for _, s in actions) / len(actions)
        self.last_score = combined
        self._score_label.configure(
            text=f"{int(combined * 100)}%", foreground=score_color(combined)
        )
        for name, score in actions:
            row = ttk.Frame(self._action_frame)
            row.pack(fill="x", pady=1)
            name_lbl = ttk.Label(row, text=name, anchor="w")
            name_lbl.pack(side="left")
            score_lbl = ttk.Label(
                row,
                text=f"{int(score * 100)}%",
                anchor="e",
                foreground=score_color(score),
            )
            score_lbl.pack(side="right")
            self._action_rows.append((name_lbl, score_lbl))
        self.result_var.set(match_info)

    def _clear_action_rows(self) -> None:
        for name_lbl, score_lbl in self._action_rows:
            name_lbl.destroy()
            score_lbl.destroy()
        self._action_rows = []
