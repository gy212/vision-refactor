# -*- coding: utf-8 -*-
"""个人练习历史窗口：按学号列出未到期记录、逐项检查与依据、教师复核、证据帧查看与补充分析。

记录读写经 ``PracticeAnalysis``（历史存储与分析任务）；当前选中的记录经 ``PracticeControls.set_current`` 同步。
"""
from __future__ import annotations

from pathlib import Path
from tkinter import StringVar, Text, Toplevel, messagebox, ttk
from typing import Any

from core.evaluation import ACTIONS, STATE_LABELS
from core.reporting import format_fusion, format_report

_RETENTION_NOTE = "已存副本与结果从首次导入/分析起保留14天；重分析不延长。外部原视频不由此模块删除。"


def check_state_text(check: dict) -> str:
    return {"confirmed": "教师已确认", "revoked": "教师已撤销"}.get(check["review"], STATE_LABELS[check["status"]])


def check_detail_text(record: dict, check: dict) -> str:
    """检查项的标准、出处、原因、融合说明与复核记录。"""
    audit = [entry for entry in record["result"]["reviewHistory"] if entry["checkId"] == check["id"]]
    return (f"{check['name']}\n标准：{check['standard']}\n出处：{check['source']}\n{check['reason']}\n{format_fusion(check)}\n" +
            "\n".join(f"{e['at']} {e['teacher']}：{'确认' if e['after'] == 'confirmed' else '撤销'} {e['reason']}" for e in audit))


class HistoryWindow:
    """``controls`` 提供 ``analysis``、``identity()``、``set_current``、``show_result``、``export``、``reanalyze``。"""

    def __init__(self, root: Any, controls: Any) -> None:
        self.controls = controls
        self.analysis = controls.analysis
        window = self.window = Toplevel(root)
        window.title("个人练习历史与教师复核")
        window.geometry("1040x760")
        window.minsize(760, 520)
        self.history_title = ttk.Label(window)
        self.history_title.pack(anchor="w", padx=12, pady=8)
        self.history_tree = ttk.Treeview(window, columns=("time", "action", "status"), show="headings", height=5)
        for col, label in (("time", "记录时间"), ("action", "动作"), ("status", "结果")):
            self.history_tree.heading(col, text=label)
            self.history_tree.column(col, width=220)
        self.history_tree.pack(fill="x", padx=12)
        self.history_tree.bind("<<TreeviewSelect>>", self.select_record)
        buttons = ttk.Frame(window)
        buttons.pack(fill="x", padx=12, pady=8)
        ttk.Button(buttons, text="刷新", command=self.refresh).pack(side="left")
        ttk.Button(buttons, text="补充分析所选记录", command=self.reanalyze_selected).pack(side="left", padx=8)
        ttk.Button(buttons, text="导出当前记录", command=controls.export).pack(side="left")
        ttk.Button(buttons, text="查看证据帧", command=self.show_evidence).pack(side="left", padx=5)
        ttk.Label(window, text=_RETENTION_NOTE, wraplength=990).pack(anchor="w", padx=12)
        self._build_checks(window)
        self._build_review(window)
        self.detail = Text(window, height=9, wrap="word", state="disabled")
        self.detail.pack(fill="both", padx=12, pady=(0, 12))
        self.refresh()

    def _build_checks(self, window: Toplevel) -> None:
        check_frame = ttk.Frame(window)
        check_frame.pack(fill="both", expand=True, padx=12, pady=8)
        self.check_tree = ttk.Treeview(check_frame, columns=("phase", "part", "name", "state"), show="headings", height=10)
        for col, label, width in (("phase", "动作 / 阶段", 210), ("part", "身体部位", 125), ("name", "检查问题", 300), ("state", "状态", 150)):
            self.check_tree.heading(col, text=label)
            self.check_tree.column(col, width=width)
        self.check_tree.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(check_frame, orient="vertical", command=self.check_tree.yview)
        scroll.pack(side="right", fill="y")
        self.check_tree.configure(yscrollcommand=scroll.set)
        self.check_tree.bind("<<TreeviewSelect>>", self.select_check)

    def _build_review(self, window: Toplevel) -> None:
        review = ttk.Labelframe(window, text="教师复核（请由任课教师操作）", padding=8)
        review.pack(fill="x", padx=12, pady=6)
        self.teacher = StringVar(window)
        self.review_reason = StringVar(window)
        ttk.Label(review, text="教师姓名").pack(side="left")
        ttk.Entry(review, textvariable=self.teacher, width=12).pack(side="left", padx=5)
        ttk.Label(review, text="原因（选填）").pack(side="left")
        ttk.Entry(review, textvariable=self.review_reason).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(review, text="确认问题", command=lambda: self.review("confirmed")).pack(side="left", padx=3)
        ttk.Button(review, text="撤销问题", command=lambda: self.review("revoked")).pack(side="left")

    def exists(self) -> bool:
        try:
            return bool(self.window.winfo_exists())
        except Exception:
            return False

    def lift(self) -> None:
        self.window.lift()

    def destroy(self) -> None:
        self.window.destroy()

    def refresh(self) -> None:
        if self.analysis.busy:
            return
        try:
            student_id = self.controls.identity()["studentId"]
            records = self.analysis.history().list(student_id)
            self.history_title.configure(text=f"学号：{student_id} · {len(records)} 条未到期记录")
            self.history_tree.delete(*self.history_tree.get_children())
            for record in records:
                self.history_tree.insert("", "end", iid=record["id"], values=(
                    record["createdAt"][:19].replace("T", " ") + " UTC", ACTIONS[record["action"]], record["result"]["summary"]))
            self.check_tree.delete(*self.check_tree.get_children())
            self._detail("")
            current = self.analysis.current
            if current and any(r["id"] == current["id"] for r in records):
                self.history_tree.selection_set(current["id"])
            else:
                self.controls.set_current(None)
        except (OSError, ValueError) as exc:
            messagebox.showerror("读取历史失败", str(exc), parent=self.window)

    def select_record(self, _event: object = None) -> None:
        if self.analysis.busy:
            return
        selection = self.history_tree.selection()
        if not selection:
            return
        try:
            record = self.analysis.history().get(selection[0])
            self.check_tree.delete(*self.check_tree.get_children())
            for check in record["result"]["checks"]:
                self.check_tree.insert("", "end", iid=check["id"], values=(
                    check["segmentLabel"] + " / " + check["phaseLabel"], check["bodyPart"], check["name"], check_state_text(check)))
            self._detail(format_report(record))
            self.controls.show_result(format_report(record))
            self.controls.set_current(record)
        except (OSError, ValueError) as exc:
            messagebox.showerror("读取记录失败", str(exc), parent=self.window)

    def _detail(self, text: str) -> None:
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", text)
        self.detail.configure(state="disabled")

    def _selected_check(self) -> dict | None:
        selection = self.check_tree.selection()
        current = self.analysis.current
        if not current or not selection:
            return None
        return next(c for c in current["result"]["checks"] if c["id"] == selection[0])

    def select_check(self, _event: object = None) -> None:
        check = self._selected_check()
        if check is not None:
            self._detail(check_detail_text(self.analysis.current, check))

    def review(self, decision: str) -> None:
        current = self.analysis.current
        if self.analysis.busy or not current or not self.check_tree.selection():
            return
        try:
            self.analysis.current = self.analysis.history().review(
                current["id"], self.check_tree.selection()[0], decision, self.teacher.get(), self.review_reason.get())
            self.select_record()
        except (OSError, ValueError) as exc:
            messagebox.showerror("复核失败", str(exc), parent=self.window)

    def reanalyze_selected(self) -> None:
        if self.history_tree.selection():
            self.controls.reanalyze(self.history_tree.selection()[0])

    def show_evidence(self) -> None:
        check = self._selected_check()
        if check is not None:
            EvidenceWindow(self.window.master, self.analysis.current, check,
                           (self.analysis.history().root / self.analysis.current["id"]).resolve())


class EvidenceWindow:
    """检查项的证据帧：正/侧各取前三帧，画上有效关键点。"""

    def __init__(self, root: Any, record: dict, check: dict, directory: Path) -> None:
        refs = set(check["evidence"])
        items = [e for e in record["result"]["evidence"] if e["id"] in refs]
        window = self.window = Toplevel(root)
        window.title("证据帧 · " + check["name"])
        window._photos = []
        if not items:
            ttk.Label(window, text="该项没有足够有效证据帧，不能据此判断动作对错。", padding=20).pack()
            return
        for row, view in enumerate(("front", "side")):
            for col, item in enumerate([e for e in items if e["view"] == view][:3]):
                self._frame_panel(window, record, view, item, directory, row, col)

    @staticmethod
    def _frame_panel(window: Toplevel, record: dict, view: str, item: dict, directory: Path, row: int, col: int) -> None:
        import cv2
        from PIL import Image, ImageTk

        from core.media import read_frame_at

        frame = None
        name = record["videos"].get(view)
        path = directory / name if name else None
        if path and path.resolve().parent == directory and path.is_file() and not path.is_symlink():
            frame = read_frame_at(path, item["frame"])
        title = ("正面" if view == "front" else "侧面") + f" {item['timeSeconds']:.3f}s · 帧{item['frame']}"
        panel = ttk.Frame(window, padding=5)
        panel.grid(row=row, column=col)
        ttk.Label(panel, text=title).pack()
        if frame is None:
            ttk.Label(panel, text="原视频已不可用").pack()
            return
        h = frame.shape[0]
        points, valid = item.get("landmarks", []), item.get("validMask", [])
        for index, point in enumerate(points):
            if index < len(valid) and valid[index]:
                cv2.circle(frame, (round(point[0] * h), round(point[1] * h)), max(2, h // 300), (0, 230, 100), -1)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((340, 260))
        photo = ImageTk.PhotoImage(image, master=window)
        window._photos.append(photo)
        ttk.Label(panel, image=photo).pack()
