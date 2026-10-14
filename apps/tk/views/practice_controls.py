# -*- coding: utf-8 -*-
"""学生练习的身份与分析控件：学号、姓名、动作、实战式、分析模型，以及个人历史、导入旧视频、导出文字与取消分析按钮。

分析任务与历史存储在 ``PracticeAnalysis``；本视图负责输入校验、弹框与按钮状态，历史窗口见 ``practice_history``。
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from tkinter import StringVar, filedialog, messagebox, ttk
from typing import Any

from core.action_feedback import ACTIONS, STANCES
from core.services import HistoryStore, validate_identity

from apps.tk.controllers.practice_analysis import DEFAULT_MODEL, MODEL_CHOICES, PracticeAnalysis
from apps.tk.views.practice_history import HistoryWindow

_VIDEO_TYPES = [("视频", "*.mp4 *.avi *.mkv *.mov *.wmv *.m4v")]


class PracticeControls:
    def __init__(self, parent: Any, *, root: Any, post: Callable[[Callable[[], Any]], Any],
                 set_busy: Callable[[bool], None], show_result: Callable[[str], None],
                 set_status: Callable[[str], None], can_analyze: Callable[[], bool],
                 on_identity_changed: Callable[[], None] | None = None,
                 on_cancelled: Callable[[], None] | None = None) -> None:
        self.root = root
        self.set_busy, self.show_result, self.set_status = set_busy, show_result, set_status
        self.can_analyze = can_analyze
        self.on_identity_changed = on_identity_changed
        self.on_cancelled = on_cancelled
        self.analysis = PracticeAnalysis(post, on_busy=set_busy, on_result=show_result, on_status=set_status,
                                         on_finished=self._on_finished)
        self.history_window: HistoryWindow | None = None
        self.student_id = StringVar(root, value="")
        self.student_name = StringVar(root, value="")
        self.action = StringVar(root, value=ACTIONS["straight_combo"])
        self.stance = StringVar(root, value=STANCES["left"])
        self.analysis_model = StringVar(root, value=DEFAULT_MODEL)
        self.fields: list[tuple[Any, str]] = []
        self._build(parent)
        self.student_id.trace_add("write", self.identity_changed)

    def _build(self, parent: Any) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(6, 0))
        for label, variable, values in (("学号", self.student_id, None), ("姓名（选填）", self.student_name, None),
                                         ("动作", self.action, list(ACTIONS.values())), ("实战式", self.stance, list(STANCES.values())),
                                         ("分析模型", self.analysis_model, list(MODEL_CHOICES))):
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=12).pack(side="left")
            widget = ttk.Combobox(row, textvariable=variable, values=values, state="readonly") if values else ttk.Entry(row, textvariable=variable)
            widget.pack(side="left", fill="x", expand=True)
            self.fields.append((widget, "readonly" if values else "normal"))
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=5)
        self.history_button = ttk.Button(row, text="个人历史", command=self.open_history)
        self.history_button.pack(side="left")
        self.import_button = ttk.Button(row, text="导入旧视频", command=self.import_video)
        self.import_button.pack(side="left", padx=4)
        self.export_button = ttk.Button(row, text="导出文字", command=self.export, state="disabled")
        self.export_button.pack(side="left")
        self.cancel_button = ttk.Button(frame, text="取消分析", command=self.cancel, state="disabled")
        self.cancel_button.pack(fill="x")
        ttk.Label(frame, text="按学号保存14天；规则检出问题可在历史中由教师复核。", wraplength=310).pack(anchor="w", pady=4)

    # ------------------------------------------------------------------ 状态
    @property
    def busy(self) -> bool:
        return self.analysis.busy

    @property
    def current(self) -> dict | None:
        return self.analysis.current

    def history(self) -> HistoryStore:
        return self.analysis.history()

    def identity(self) -> dict:
        action = next((k for k, v in ACTIONS.items() if v == self.action.get()), "")
        stance = next((k for k, v in STANCES.items() if v == self.stance.get()), "")
        return validate_identity(self.student_id.get(), self.student_name.get(), action, stance)

    def identity_changed(self, *_args: object) -> None:
        self.analysis.invalidate()
        self.export_button.configure(state="disabled")
        self._close_history()
        if self.on_identity_changed:
            try:
                self.on_identity_changed()
            except Exception:
                pass

    def set_current(self, record: dict | None) -> None:
        self.analysis.current = record
        self.export_button.configure(state="normal" if record else "disabled")

    def set_recording(self, recording: bool) -> None:
        locked = recording or self.busy
        for widget, normal in self.fields:
            widget.configure(state="disabled" if locked else normal)
        self.import_button.configure(state="disabled" if locked else "normal")
        self.history_button.configure(state="disabled" if locked else "normal")
        self.export_button.configure(state="normal" if self.current and not self.busy else "disabled")

    # ------------------------------------------------------------------ 分析
    def analyze_recording(self, identity: dict, front: Path | None, side: Path | None, *, record_id: str | None = None,
                          on_saved: Callable[[dict], None] | None = None, token: Any = None) -> bool:
        """发起后台分析；配置错误或当前不可分析时返回 False，调用方据此恢复可重试状态。"""
        analyzer = self._analyzer()
        if analyzer is None or not self._ready():
            return False
        return self._begin(self.analysis.analyze(identity, front, side, analyzer=analyzer,
                                                 record_id=record_id, on_saved=on_saved, token=token))

    def reanalyze(self, record_id: str) -> None:
        analyzer = self._analyzer()
        if analyzer is not None and self._ready():
            self._begin(self.analysis.analyze({}, None, None, analyzer=analyzer, record_id=record_id))

    def _analyzer(self) -> Callable[..., Any] | None:
        """Tk 变量在主线程读取一次；配置损坏时展示原因并返回 None。"""
        try:
            return self.analysis.analyzer(self.analysis_model.get())
        except (ValueError, OSError) as exc:
            self.show_result(str(exc))
            return None

    def _ready(self) -> bool:
        if self.busy or not self.can_analyze():
            messagebox.showinfo("动作问题说明", "请先结束录制并等待当前任务完成。", parent=self.root)
            return False
        return True

    def _begin(self, started: bool) -> bool:
        if started:
            self.set_recording(False)
            self.cancel_button.configure(state="normal")
        return started

    def _on_finished(self, applied: bool) -> None:
        self.cancel_button.configure(state="disabled")
        self.set_recording(False)
        if applied and self.history_window is not None and self.history_window.exists():
            self.history_window.refresh()

    def cancel(self) -> None:
        self.analysis.cancel()
        self.cancel_button.configure(state="disabled")
        if self.on_cancelled:
            try:
                self.on_cancelled()
            except Exception:
                pass

    def close(self) -> None:
        self.analysis.close()

    # ------------------------------------------------------------------ 导入、导出与历史
    def import_video(self) -> None:
        try:
            identity = self.identity()
        except ValueError as exc:
            messagebox.showerror("导入旧视频", str(exc), parent=self.root)
            return
        front = filedialog.askopenfilename(parent=self.root, title="选择正面原视频（没有可取消，下一步选择侧面）", filetypes=_VIDEO_TYPES)
        side = filedialog.askopenfilename(parent=self.root, title="选择对应侧面视频（取消表示缺少侧面）", filetypes=_VIDEO_TYPES)
        if front or side:
            self.analyze_recording(identity, Path(front) if front else None, Path(side) if side else None)

    def export(self) -> None:
        current = self.current
        if not current:
            return
        destination = filedialog.asksaveasfilename(parent=self.root, title="导出动作问题说明", defaultextension=".txt",
            initialfile=f"{current['studentId']}_动作问题说明.txt", filetypes=[("文字报告", "*.txt")])
        if not destination:
            return
        try:
            self.analysis.export_text(current["id"], Path(destination))
            self.set_status(f"已导出：{destination}")
        except (OSError, ValueError) as exc:
            messagebox.showerror("导出失败", str(exc), parent=self.root)

    def open_history(self) -> None:
        try:
            self.identity()
        except ValueError as exc:
            messagebox.showerror("个人历史", str(exc), parent=self.root)
            return
        if self.history_window is not None and self.history_window.exists():
            self.history_window.lift()
            self.history_window.refresh()
            return
        self.history_window = HistoryWindow(self.root, self)

    def _close_history(self) -> None:
        if self.history_window is not None and self.history_window.exists():
            self.history_window.destroy()
        self.history_window = None
