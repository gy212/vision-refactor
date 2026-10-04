"""Tk 学生练习的问题说明、个人历史、导出与可选教师复核控件。"""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from queue import Empty, Queue
import threading
from tkinter import StringVar, Text, Toplevel, filedialog, messagebox, ttk

from core.action_feedback import ACTIONS, STANCES, STATE_LABELS, format_report, format_fusion, analyze_feedback
from core.feedback_geometry import load_feedback_config
from core.feedback_history import FeedbackHistory, validate_identity


class FeedbackControls:
    def __init__(self, parent, *, root, set_busy, show_result, set_status, can_analyze, on_identity_changed=None, on_cancelled=None):
        self.root = root
        self.set_busy, self.show_result, self.set_status = set_busy, show_result, set_status
        self.can_analyze = can_analyze
        self.on_identity_changed = on_identity_changed
        self.on_cancelled = on_cancelled
        self.store = None
        self.current = None
        self.busy = False
        self.closed = False
        self.active_token = None
        self.stop_event = threading.Event()
        self.queue = Queue()
        self.poll_id = None
        self.history_window = None
        self.student_id = StringVar(root, value="")
        self.student_name = StringVar(root, value="")
        self.action = StringVar(root, value=ACTIONS["straight_combo"])
        self.stance = StringVar(root, value=STANCES["left"])
        self.model_choices = {"Full（精细分析）": "full", "Lite（快速对比）": "lite", "Heavy（较慢）": "heavy"}
        self.analysis_model = StringVar(root, value="Full（精细分析）")
        self.fields = []
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(6, 0))
        for label, variable, values in (("学号", self.student_id, None), ("姓名（选填）", self.student_name, None),
                                         ("动作", self.action, list(ACTIONS.values())), ("实战式", self.stance, list(STANCES.values())),
                                         ("分析模型", self.analysis_model, list(self.model_choices))):
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
        self.student_id.trace_add("write", self.identity_changed)

    def identity_changed(self, *_args):
        self.active_token = None
        self.current = None
        self.export_button.configure(state="disabled")
        if self.history_window is not None and self.history_window.winfo_exists():
            self.history_window.destroy()
        self.history_window = None
        if hasattr(self, "on_identity_changed") and self.on_identity_changed:
            try:
                self.on_identity_changed()
            except Exception:
                pass

    def history(self):
        if self.store is None:
            self.store = FeedbackHistory()
        return self.store

    def identity(self):
        action = next((k for k, v in ACTIONS.items() if v == self.action.get()), "")
        stance = next((k for k, v in STANCES.items() if v == self.stance.get()), "")
        return validate_identity(self.student_id.get(), self.student_name.get(), action, stance)

    def set_recording(self, recording):
        for widget, normal in self.fields:
            widget.configure(state="disabled" if recording or self.busy else normal)
        self.import_button.configure(state="disabled" if recording or self.busy else "normal")
        self.history_button.configure(state="disabled" if recording or self.busy else "normal")
        self.export_button.configure(state="normal" if self.current and not self.busy else "disabled")

    def _start(self, work, on_saved=None, token=None):
        if self.busy or not self.can_analyze():
            messagebox.showinfo("动作问题说明", "请先结束录制并等待当前任务完成。", parent=self.root)
            return
        self.active_token = token
        self.busy = True
        self.stop_event.clear()
        self.set_busy(True)
        self.set_recording(False)
        self.cancel_button.configure(state="normal")
        self.set_status("正在用MediaPipe CPU提取关键点并检查规则，完成后保存到个人历史…")

        def run():
            try:
                self.queue.put((work(), None, token))
            except Exception as exc:
                if isinstance(exc, InterruptedError):
                    error = "分析已取消"
                elif isinstance(exc, (ValueError, FileNotFoundError, PermissionError)):
                    error = str(exc)
                else:
                    error = "分析或保存失败，请检查MediaPipe模型、视频和保存目录后重试。"
                self.queue.put((None, error, token))

        def poll():
            if self.closed:
                return
            try:
                item = self.queue.get_nowait()
                if len(item) == 3:
                    record, error, task_token = item
                else:
                    record, error = item
                    task_token = None
            except Empty:
                self.poll_id = self.root.after(100, poll)
                return
            self.poll_id = None
            self.busy = False
            self.set_busy(False)
            self.cancel_button.configure(state="disabled")

            if task_token is not None and task_token != self.active_token:
                self.set_recording(False)
                return
            if self.stop_event.is_set():
                self.set_recording(False)
                return

            if error:
                self.set_status(error)
                self.show_result(error)
            else:
                self.current = record
                self.show_result(format_report(record))
                self.set_status("问题说明已保存" + ("；部分证据不足，请重新录制" if record["result"]["needsRerecord"] else ""))
                if on_saved:
                    on_saved(record)
            self.set_recording(False)
            if self.history_window is not None and self.history_window.winfo_exists():
                self.refresh_history()

        threading.Thread(target=run, name="student-action-feedback", daemon=True).start()
        self.poll_id = self.root.after(100, poll)

    def analyze_recording(self, identity, front, side, *, record_id=None, on_saved=None, token=None):
        try:
            analyzer = self._analyzer()
        except (ValueError, OSError) as exc:
            self.show_result(str(exc))
            return
        self._start(
            lambda: self.history().reanalyze(record_id, stopped=self.stop_event.is_set, analyzer=analyzer) if record_id else
            self.history().add(identity, front, side, stopped=self.stop_event.is_set, analyzer=analyzer),
            on_saved=on_saved,
            token=token,
        )

    def _analyzer(self):
        # Tk变量在主线程读取一次，后台任务与本次选择绑定。
        config = replace(load_feedback_config(), pose_variant=self.model_choices[self.analysis_model.get()])
        return lambda *args, **kwargs: analyze_feedback(*args, config=config, **kwargs)

    def cancel(self):
        self.active_token = None
        self.stop_event.set()
        self.cancel_button.configure(state="disabled")
        self.set_status("正在取消；当前帧处理结束后释放任务…")
        if hasattr(self, "on_cancelled") and self.on_cancelled:
            try:
                self.on_cancelled()
            except Exception:
                pass

    def close(self):
        self.closed = True
        self.active_token = None
        self.stop_event.set()
        if self.poll_id is not None:
            self.root.after_cancel(self.poll_id)
            self.poll_id = None

    def import_video(self):
        try:
            identity = self.identity()
        except ValueError as exc:
            messagebox.showerror("导入旧视频", str(exc), parent=self.root)
            return
        types = [("视频", "*.mp4 *.avi *.mkv *.mov *.wmv *.m4v")]
        front = filedialog.askopenfilename(parent=self.root, title="选择正面原视频（没有可取消，下一步选择侧面）", filetypes=types)
        side = filedialog.askopenfilename(parent=self.root, title="选择对应侧面视频（取消表示缺少侧面）", filetypes=types)
        if front or side:
            self.analyze_recording(identity, Path(front) if front else None, Path(side) if side else None)

    def export(self):
        if not self.current:
            return
        destination = filedialog.asksaveasfilename(parent=self.root, title="导出动作问题说明", defaultextension=".txt",
            initialfile=f"{self.current['studentId']}_动作问题说明.txt", filetypes=[("文字报告", "*.txt")])
        if not destination:
            return
        try:
            self.history().export(self.current["id"], Path(destination))
            self.set_status(f"已导出：{destination}")
        except (OSError, ValueError) as exc:
            messagebox.showerror("导出失败", str(exc), parent=self.root)

    def open_history(self):
        try:
            self.identity()
        except ValueError as exc:
            messagebox.showerror("个人历史", str(exc), parent=self.root)
            return
        if self.history_window is not None and self.history_window.winfo_exists():
            self.history_window.lift()
            self.refresh_history()
            return
        window = self.history_window = Toplevel(self.root)
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
        ttk.Button(buttons, text="刷新", command=self.refresh_history).pack(side="left")
        ttk.Button(buttons, text="补充分析所选记录", command=self.reanalyze_selected).pack(side="left", padx=8)
        ttk.Button(buttons, text="导出当前记录", command=self.export).pack(side="left")
        ttk.Button(buttons, text="查看证据帧", command=self.show_evidence).pack(side="left", padx=5)
        ttk.Label(window, text="已存副本与结果从首次导入/分析起保留14天；重分析不延长。外部原视频不由此模块删除。", wraplength=990).pack(anchor="w", padx=12)
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
        self.detail = Text(window, height=9, wrap="word", state="disabled")
        self.detail.pack(fill="both", padx=12, pady=(0, 12))
        self.refresh_history()

    def refresh_history(self):
        if self.busy:
            return
        try:
            student_id = self.identity()["studentId"]
            records = self.history().list(student_id)
            self.history_title.configure(text=f"学号：{student_id} · {len(records)} 条未到期记录")
            self.history_tree.delete(*self.history_tree.get_children())
            for record in records:
                self.history_tree.insert("", "end", iid=record["id"], values=(record["createdAt"][:19].replace("T", " ") + " UTC", ACTIONS[record["action"]], record["result"]["summary"]))
            self.check_tree.delete(*self.check_tree.get_children())
            self._detail("")
            if self.current and any(r["id"] == self.current["id"] for r in records):
                self.history_tree.selection_set(self.current["id"])
            else:
                self.current = None
                self.export_button.configure(state="disabled")
        except (OSError, ValueError) as exc:
            messagebox.showerror("读取历史失败", str(exc), parent=self.history_window)

    def select_record(self, _event=None):
        if self.busy:
            return
        selection = self.history_tree.selection()
        if not selection:
            return
        try:
            self.current = self.history().get(selection[0])
            self.check_tree.delete(*self.check_tree.get_children())
            for check in self.current["result"]["checks"]:
                state = {"confirmed": "教师已确认", "revoked": "教师已撤销"}.get(check["review"], STATE_LABELS[check["status"]])
                self.check_tree.insert("", "end", iid=check["id"], values=(check["segmentLabel"] + " / " + check["phaseLabel"], check["bodyPart"], check["name"], state))
            self._detail(format_report(self.current))
            self.show_result(format_report(self.current))
            self.export_button.configure(state="normal")
        except (OSError, ValueError) as exc:
            messagebox.showerror("读取记录失败", str(exc), parent=self.history_window)

    def _detail(self, text):
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", text)
        self.detail.configure(state="disabled")

    def select_check(self, _event=None):
        selection = self.check_tree.selection()
        if not self.current or not selection:
            return
        check = next(c for c in self.current["result"]["checks"] if c["id"] == selection[0])
        audit = [entry for entry in self.current["result"]["reviewHistory"] if entry["checkId"] == check["id"]]
        self._detail(f"{check['name']}\n标准：{check['standard']}\n出处：{check['source']}\n{check['reason']}\n{format_fusion(check)}\n" +
                     "\n".join(f"{e['at']} {e['teacher']}：{'确认' if e['after'] == 'confirmed' else '撤销'} {e['reason']}" for e in audit))

    def review(self, decision):
        if self.busy or not self.current or not self.check_tree.selection():
            return
        try:
            self.current = self.history().review(self.current["id"], self.check_tree.selection()[0], decision, self.teacher.get(), self.review_reason.get())
            self.select_record()
        except (OSError, ValueError) as exc:
            messagebox.showerror("复核失败", str(exc), parent=self.history_window)

    def show_evidence(self):
        if not self.current or not self.check_tree.selection():
            return
        import cv2
        from PIL import Image, ImageTk
        check = next(c for c in self.current["result"]["checks"] if c["id"] == self.check_tree.selection()[0])
        refs = set(check["evidence"])
        items = [e for e in self.current["result"]["evidence"] if e["id"] in refs]
        window = Toplevel(self.root)
        window.title("证据帧 · " + check["name"])
        window._photos = []
        if not items:
            ttk.Label(window, text="该项没有足够有效证据帧，不能据此判断动作对错。", padding=20).pack()
            return
        directory = (self.history().root / self.current["id"]).resolve()
        for row, view in enumerate(("front", "side")):
            selected = [e for e in items if e["view"] == view][:3]
            for col, item in enumerate(selected):
                frame = None
                name = self.current["videos"].get(view)
                path = directory / name if name else None
                if path and path.resolve().parent == directory and path.is_file() and not path.is_symlink():
                    cap = cv2.VideoCapture(str(path))
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, item["frame"])
                        ok, frame = cap.read()
                        if not ok:
                            frame = None
                    finally:
                        cap.release()
                title = ("正面" if view == "front" else "侧面") + f" {item['timeSeconds']:.3f}s · 帧{item['frame']}"
                panel = ttk.Frame(window, padding=5)
                panel.grid(row=row, column=col)
                ttk.Label(panel, text=title).pack()
                if frame is None:
                    ttk.Label(panel, text="原视频已不可用").pack()
                    continue
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

    def reanalyze_selected(self):
        if not self.history_tree.selection():
            return
        record_id = self.history_tree.selection()[0]
        try:
            analyzer = self._analyzer()
        except (ValueError, OSError) as exc:
            self.show_result(str(exc))
            return
        self._start(lambda: self.history().reanalyze(record_id, stopped=self.stop_event.is_set, analyzer=analyzer))
