# -*- coding: utf-8 -*-
"""考试窗口的布局：阶段/播报/占用/ROI 文字、名单与 ROI 操作行、开考控制按钮与成绩表。

按钮只读输入并调用 ``ExamController``；控制器经本视图的 ``ExamView`` 方法更新界面与弹框。
"""
from __future__ import annotations

from pathlib import Path
from tkinter import DoubleVar, StringVar, Toplevel, filedialog, messagebox, ttk
from collections.abc import Sequence
from typing import Any

from core.workflows import ExamResultRow, write_import_template

from apps.tk.controllers.exam import ExamController, ExamHost

_NOTE = (
    "说明：v1 输出各动作黑盒相似度子分 + 综合分（无规则扣分）。"
    "正对正池、侧对侧池，每边可填多个动作模板。"
    "占用判定用 primary 旋转后 ROI。须先双摄「开始」并至少备好一侧 heavy 模板。"
    "重考：先开考产生台账 → 选中已考完/失败/跳过的考生 → 点重考（追加到队尾）"
    "→ 若整场已结束或已中止，再点「开始考试」才会叫号。"
)
_COLUMNS = (("order", "序号", 50), ("id", "学号", 90), ("name", "姓名", 80), ("status", "状态", 90), ("score", "综合分", 60))


def row_values(row: ExamResultRow) -> tuple:
    score = "" if row.combined_percent is None else str(row.combined_percent)
    return (row.candidate.order, row.candidate.student_id, row.candidate.name, row.status, score)


class ExamWindow:
    """考试 Toplevel；``controller`` 在控件建好后创建（构造时恢复上次名单）。"""

    def __init__(self, parent: Any, host: ExamHost) -> None:
        self.win = Toplevel(parent)
        self.win.title("考试模式")
        self.win.geometry("520x640")
        self.phase_var = StringVar(value="阶段：idle")
        self.announce_var = StringVar(value="播报：-")
        self.present_var = StringVar(value="占用：-")
        self.status_var = StringVar(value="")
        self.roi_var = StringVar(value="")
        self.roi_vars = [DoubleVar(value=0.0) for _ in range(4)]
        self._build()
        self.controller = ExamController(host, self)
        self.win.protocol("WM_DELETE_WINDOW", self.controller.close)

    def _build(self) -> None:
        f = ttk.Frame(self.win, padding=10)
        f.pack(fill="both", expand=True)
        ttk.Label(f, textvariable=self.phase_var).pack(anchor="w")
        ttk.Label(f, textvariable=self.announce_var, wraplength=480).pack(anchor="w")
        ttk.Label(f, textvariable=self.present_var).pack(anchor="w")
        ttk.Label(f, textvariable=self.roi_var).pack(anchor="w")
        ttk.Label(f, textvariable=self.status_var, wraplength=480).pack(anchor="w", pady=(4, 0))

        row = ttk.Frame(f)
        row.pack(fill="x", pady=(8, 0))
        ttk.Button(row, text="下载导入模板…", command=self._download_template).pack(side="left")
        ttk.Button(row, text="导入名单…", command=self._import_roster).pack(side="left", padx=(8, 0))
        ttk.Button(row, text="保存 ROI", command=self._save_roi).pack(side="left", padx=(8, 0))

        roi_row = ttk.Frame(f)
        roi_row.pack(fill="x", pady=(6, 0))
        for var, lab in zip(self.roi_vars, ("x0", "y0", "x1", "y1")):
            ttk.Label(roi_row, text=lab).pack(side="left")
            ttk.Entry(roi_row, textvariable=var, width=6).pack(side="left", padx=(2, 6))

        btns = ttk.Frame(f)
        btns.pack(fill="x", pady=(10, 0))
        self.btn_start = ttk.Button(btns, text="开始考试", command=self._start_exam)
        self.btn_start.pack(side="left")
        for text, event in (("暂停", "pause"), ("继续", "resume"), ("跳过当前", "skip_current"),
                            ("强制结束", "force_finish_current")):
            ttk.Button(btns, text=text, command=lambda e=event: self.controller.event(e)).pack(side="left", padx=4)

        btns2 = ttk.Frame(f)
        btns2.pack(fill="x", pady=(6, 0))
        ttk.Button(btns2, text="重考（选中考生）", command=lambda: self.controller.retest(self._save_roi)).pack(side="left")
        ttk.Button(btns2, text="导出成绩…", command=self._export).pack(side="left", padx=4)
        ttk.Button(btns2, text="中止考试", command=lambda: self.controller.event("abort_exam")).pack(side="left", padx=4)

        self.tree = ttk.Treeview(f, columns=tuple(c for c, _t, _w in _COLUMNS), show="headings", height=16)
        for c, t, w in _COLUMNS:
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True, pady=(10, 0))
        ttk.Label(f, text=_NOTE, wraplength=480, foreground="#444").pack(anchor="w", pady=(8, 0))

    # ------------------------------------------------------------------ 按钮
    def _download_template(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.win, title="保存导入模板", defaultextension=".xlsx",
            initialfile="考试名单模板.xlsx", filetypes=[("Excel", "*.xlsx")],
        )
        if not path:
            return
        write_import_template(path)
        messagebox.showinfo("完成", f"已保存：{path}", parent=self.win)

    def _import_roster(self) -> None:
        if not self.controller.can_import():
            return
        path = filedialog.askopenfilename(parent=self.win, title="导入考试名单", filetypes=[("Excel", "*.xlsx")])
        if path:
            self.controller.import_roster(Path(path))

    def _save_roi(self) -> bool:
        return self.controller.save_roi(lambda: [v.get() for v in self.roi_vars])

    def _start_exam(self) -> None:
        self.controller.start_exam(self._save_roi)

    def _export(self) -> None:
        name = self.controller.export_default_name()
        if name is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self.win, title="导出成绩", defaultextension=".xlsx", initialfile=name,
            filetypes=[("Excel", "*.xlsx")],
        )
        if path:
            self.controller.export(path)

    def lift(self) -> None:
        self.win.lift()

    # ------------------------------------------------------------------ ExamView
    def info(self, title: str, text: str) -> None:
        messagebox.showinfo(title, text, parent=self.win)

    def error(self, title: str, text: str) -> None:
        messagebox.showerror(title, text, parent=self.win)

    def ask(self, title: str, text: str) -> bool:
        return bool(messagebox.askyesno(title, text, parent=self.win))

    def set_phase(self, text: str) -> None:
        _set_if_changed(self.phase_var, text)

    def set_status(self, text: str) -> None:
        _set_if_changed(self.status_var, text)

    def set_present(self, text: str) -> None:
        _set_if_changed(self.present_var, text)

    def set_announce(self, text: str) -> None:
        self.announce_var.set(text)

    def set_roi(self, roi: Sequence[float]) -> None:
        for var, value in zip(self.roi_vars, roi):
            var.set(value)
        self.roi_var.set(f"ROI(primary旋转后): ({roi[0]:.2f},{roi[1]:.2f})-({roi[2]:.2f},{roi[3]:.2f})")

    def show_rows(self, rows: Sequence[ExamResultRow]) -> None:
        """按台账顺序同步表格：已有行更新并移到末尾，新行插入，消失的行删除（保留选中）。"""
        existing = set(self.tree.get_children())
        active: set[str] = set()
        for r in rows:
            active.add(r.row_id)
            if r.row_id in existing:
                self.tree.item(r.row_id, values=row_values(r))
                self.tree.move(r.row_id, "", "end")
            else:
                self.tree.insert("", "end", iid=r.row_id, values=row_values(r))
        for stale in existing - active:
            self.tree.delete(stale)

    def select_row(self, row_id: str) -> None:
        try:
            self.tree.selection_set(row_id)
            self.tree.focus(row_id)
            self.tree.see(row_id)
        except Exception:
            pass

    def selected(self) -> tuple[str, str] | None:
        sel = self.tree.selection()
        if not sel:
            return None
        try:
            values = self.tree.item(sel[0], "values")
        except Exception:
            values = ()
        return str(sel[0]), (str(values[1]) if len(values) > 1 else "")

    def is_open(self) -> bool:
        try:
            return bool(self.win.winfo_exists())
        except Exception:
            return False

    def destroy(self) -> None:
        self.win.destroy()


def _set_if_changed(var: StringVar, value: str) -> None:
    if var.get() != value:
        var.set(value)
