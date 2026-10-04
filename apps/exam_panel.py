# -*- coding: utf-8 -*-
"""Tk 考试面板：名单、ROI、开考控制；状态机在 core.exam_session。"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from tkinter import DoubleVar, StringVar, Toplevel, filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any, Callable

from core.exam_announcer import ExamAnnouncer
from core.exam_roster import (
    ExamCandidate,
    ExamResultRow,
    ExamScorebook,
    RosterError,
    import_roster_xlsx,
    local_timestamp,
    write_import_template,
)
from core.exam_session import ExamCommand, ExamSession, ExamSessionConfig
from core.paths import outputs_dir, repo_root
from core.presence_gate import PresenceGate, PresenceGateConfig

if TYPE_CHECKING:
    from apps.app_ui import App


_EXAM_ROSTER_PATH_PREF_KEY = "exam_roster_path"


def apply_postprocess_update_to_scorebook(
    scorebook: ExamScorebook, update: Any
) -> ExamResultRow | None:
    """把后处理更新写入台账（面板存活或关窗后 sink 共用）。"""
    row = scorebook.get_by_segment(str(update.segment_id))
    if row is None:
        return None
    if update.status == "completed":
        # 逐动作子分：{name: 0..1}（同名后者覆盖前者，正常不重名）
        action_scores = {
            str(d.get("name")): float(d.get("score") or 0.0)
            for d in getattr(update, "action_scores", ()) or ()
            if d.get("name")
        }
        scorebook.update_row(
            row,
            status="completed",
            front_score=float(update.front_score or 0.0),
            side_score=float(update.side_score or 0.0),
            combined_percent=int(update.combined_percent or 0),
            combined_score=float(update.combined_percent or 0) / 100.0,
            action_scores=action_scores,
            error_code=None,
            error_message=None,
        )
    elif update.status in {"failed", "skipped", "cancelled"}:
        scorebook.update_row(
            row,
            status="failed" if update.status == "failed" else update.status,
            error_code=update.error_code,
            error_message=update.message,
        )
    elif update.status in {"queued", "transcoding", "validating", "comparing"}:
        scorebook.update_row(row, status="processing")
    return row


def _prefs_path() -> Path:
    return repo_root() / "user_prefs.json"


def load_exam_prefs() -> dict[str, Any]:
    try:
        data = json.loads(_prefs_path().read_text(encoding="utf-8"))
        return dict(data) if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_exam_prefs(updates: dict[str, Any]) -> None:
    path = _prefs_path()
    data = load_exam_prefs()
    data.update(updates)
    try:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


class ExamPanel:
    def __init__(self, app: "App") -> None:
        self.app = app
        prefs = load_exam_prefs()
        roi = prefs.get("exam_roi_norm") or [0.2, 0.1, 0.8, 0.95]
        self.session = ExamSession(
            ExamSessionConfig(
                post_call_guard_s=float(prefs.get("exam_post_call_guard_s", 1.5)),
                inter_student_gap_s=float(prefs.get("exam_inter_student_gap_s", 2.0)),
                enter_stable_s=float(prefs.get("exam_enter_stable_s", 0.8)),
                empty_hold_s=float(prefs.get("exam_empty_hold_s", 2.0)),
                min_record_s=float(prefs.get("exam_min_record_s", 3.0)),
            )
        )
        self.gate = PresenceGate(
            PresenceGateConfig(
                enter_stable_s=self.session.config.enter_stable_s,
                empty_hold_s=self.session.config.empty_hold_s,
                min_record_s=self.session.config.min_record_s,
            )
        )
        self.roi = (
            float(roi[0]),
            float(roi[1]),
            float(roi[2]),
            float(roi[3]),
        )
        self.run_dir: Path | None = None
        self.scorebook: ExamScorebook | None = None
        self._announcer = ExamAnnouncer(on_text=self._on_announce_text)
        self._pending_discard = False
        self._row_by_id: dict[str, Any] = {}

        self.win = Toplevel(app.root)
        self.win.title("考试模式")
        self.win.geometry("520x640")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self.phase_var = StringVar(value="阶段：idle")
        self.announce_var = StringVar(value="播报：-")
        self.present_var = StringVar(value="占用：-")
        self.status_var = StringVar(value="")
        self.roi_var = StringVar(value=self._roi_text())

        self._build()
        self._restore_roster(prefs)
        self._tick_id = self.win.after(100, self._panel_tick)
        app._exam_panel = self  # type: ignore[attr-defined]

    def _roi_text(self) -> str:
        r = self.roi
        return f"ROI(primary旋转后): ({r[0]:.2f},{r[1]:.2f})-({r[2]:.2f},{r[3]:.2f})"

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
        ttk.Button(row, text="下载导入模板…", command=self._download_template).pack(
            side="left"
        )
        ttk.Button(row, text="导入名单…", command=self._import_roster).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(row, text="保存 ROI", command=self._save_roi).pack(
            side="left", padx=(8, 0)
        )

        roi_row = ttk.Frame(f)
        roi_row.pack(fill="x", pady=(6, 0))
        self.roi_vars = [DoubleVar(value=self.roi[i]) for i in range(4)]
        for i, lab in enumerate(("x0", "y0", "x1", "y1")):
            ttk.Label(roi_row, text=lab).pack(side="left")
            ttk.Entry(roi_row, textvariable=self.roi_vars[i], width=6).pack(
                side="left", padx=(2, 6)
            )

        btns = ttk.Frame(f)
        btns.pack(fill="x", pady=(10, 0))
        self.btn_start = ttk.Button(btns, text="开始考试", command=self._start_exam)
        self.btn_start.pack(side="left")
        ttk.Button(btns, text="暂停", command=lambda: self._event("pause")).pack(
            side="left", padx=4
        )
        ttk.Button(btns, text="继续", command=lambda: self._event("resume")).pack(
            side="left", padx=4
        )
        ttk.Button(btns, text="跳过当前", command=lambda: self._event("skip_current")).pack(
            side="left", padx=4
        )
        ttk.Button(
            btns, text="强制结束", command=lambda: self._event("force_finish_current")
        ).pack(side="left", padx=4)

        btns2 = ttk.Frame(f)
        btns2.pack(fill="x", pady=(6, 0))
        ttk.Button(btns2, text="重考（选中考生）", command=self._retest_current).pack(
            side="left"
        )
        ttk.Button(btns2, text="导出成绩…", command=self._export).pack(
            side="left", padx=4
        )
        ttk.Button(btns2, text="中止考试", command=lambda: self._event("abort_exam")).pack(
            side="left", padx=4
        )

        cols = ("order", "id", "name", "status", "score")
        self.tree = ttk.Treeview(f, columns=cols, show="headings", height=16)
        for c, t, w in (
            ("order", "序号", 50),
            ("id", "学号", 90),
            ("name", "姓名", 80),
            ("status", "状态", 90),
            ("score", "综合分", 60),
        ):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True, pady=(10, 0))

        note = (
            "说明：v1 输出各动作黑盒相似度子分 + 综合分（无规则扣分）。"
            "正对正池、侧对侧池，每边可填多个动作模板。"
            "占用判定用 primary 旋转后 ROI。须先双摄「开始」并至少备好一侧 heavy 模板。"
            "重考：先开考产生台账 → 选中已考完/失败/跳过的考生 → 点重考（追加到队尾）"
            "→ 若整场已结束或已中止，再点「开始考试」才会叫号。"
        )
        ttk.Label(f, text=note, wraplength=480, foreground="#444").pack(
            anchor="w", pady=(8, 0)
        )

    def _on_announce_text(self, text: str) -> None:
        try:
            self.win.after(0, lambda: self.announce_var.set(f"播报：{text}"))
        except Exception:
            pass

    def _download_template(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.win,
            title="保存导入模板",
            defaultextension=".xlsx",
            initialfile="考试名单模板.xlsx",
            filetypes=[("Excel", "*.xlsx")],
        )
        if not path:
            return
        write_import_template(path)
        messagebox.showinfo("完成", f"已保存：{path}", parent=self.win)

    def _import_roster(self) -> None:
        if self.session.phase not in {"idle", "ready", "completed", "aborted"}:
            messagebox.showerror(
                "无法导入",
                f"当前阶段 {self.session.phase}，请先中止当前考试再更换名单",
                parent=self.win,
            )
            return
        path = filedialog.askopenfilename(
            parent=self.win,
            title="导入考试名单",
            filetypes=[("Excel", "*.xlsx")],
        )
        if not path:
            return
        try:
            count = self._load_roster_path(Path(path), persist=True)
        except RosterError as exc:
            messagebox.showerror("导入失败", exc.message, parent=self.win)
            return
        self.status_var.set(f"已导入 {count} 人")

    def _replace_roster(self, candidates: list[ExamCandidate]) -> None:
        # 新名单 = 新场次：关闭旧台账，避免与旧成绩混用
        if self.scorebook is not None:
            old_scorebook = self.scorebook
            try:
                old_scorebook.flush()
            except Exception:
                pass
            has_processing = any(
                row.status == "processing" for row in old_scorebook.rows
            )
            if has_processing:
                try:
                    self.app._register_exam_scorebook_sink(old_scorebook)  # type: ignore[attr-defined]
                except Exception:
                    sinks = getattr(self.app, "_exam_scorebook_sinks", None)
                    if sinks is None:
                        self.app._exam_scorebook_sinks = [old_scorebook]  # type: ignore[attr-defined]
                    elif old_scorebook not in sinks:
                        sinks.append(old_scorebook)
            else:
                try:
                    old_scorebook.close()
                except Exception:
                    pass
            self.scorebook = None
        self.run_dir = None
        for method_name, args in (
            ("_exam_arm_occupancy", (False,)),
            ("_exam_set_active", (False,)),
            ("_exam_lock_manual_record", (False,)),
        ):
            try:
                getattr(self.app, method_name)(*args)
            except Exception:
                pass
        self.session.load_roster(candidates)
        self._row_by_id = {r.row_id: r for r in self.session.rows}
        self._refresh_tree()
        self.phase_var.set(f"阶段：{self.session.phase}")

    def _load_roster_path(self, path: Path, *, persist: bool) -> int:
        candidates = import_roster_xlsx(path)
        self._replace_roster(candidates)
        if persist:
            save_exam_prefs({_EXAM_ROSTER_PATH_PREF_KEY: str(path.resolve())})
        return len(candidates)

    def _restore_roster(self, prefs: dict[str, Any]) -> None:
        raw_path = str(prefs.get(_EXAM_ROSTER_PATH_PREF_KEY) or "").strip()
        if not raw_path:
            return
        try:
            count = self._load_roster_path(Path(raw_path), persist=False)
        except RosterError as exc:
            self.status_var.set(f"上次名单未能恢复：{exc.message}；请重新导入")
            return
        self.status_var.set(f"已恢复上次名单：{count} 人")

    def _save_roi(self) -> bool:
        try:
            r = tuple(float(v.get()) for v in self.roi_vars)
        except Exception:
            messagebox.showerror("ROI", "ROI 数值无效", parent=self.win)
            return False
        if not (0.0 <= r[0] < r[2] <= 1.0 and 0.0 <= r[1] < r[3] <= 1.0):
            messagebox.showerror("ROI", "须满足 0≤x0<x1≤1 且 0≤y0<y1≤1", parent=self.win)
            return False
        self.roi = r  # type: ignore[assignment]
        rotate = 0
        try:
            rotate = int(getattr(self.app, "_parse_rotate_for_exam", lambda: 0)())
        except Exception:
            rotate = 0
        # 从 app 读当前 primary 旋转
        try:
            rotate = self.app._exam_primary_rotate()  # type: ignore[attr-defined]
        except Exception:
            pass
        save_exam_prefs({"exam_roi_norm": list(self.roi), "exam_roi_rotate": rotate})
        self.roi_var.set(self._roi_text())
        self.app._exam_set_roi(self.roi)  # type: ignore[attr-defined]
        self.status_var.set("ROI 已保存")
        return True

    def _start_exam(self) -> None:
        if not self.session.rows:
            messagebox.showerror("无法开考", "请先导入名单", parent=self.win)
            return
        if self.session.phase != "ready":
            if self.session.phase == "completed":
                detail = "名单已全部完成，请先选择考生并点击重考"
            elif self.session.phase == "aborted":
                detail = "考试已中止，请先选择考生并点击重考"
            else:
                detail = f"当前阶段 {self.session.phase}，无法重复开始"
            messagebox.showerror("无法开考", detail, parent=self.win)
            return
        if not any(row.status == "pending" for row in self.session.rows):
            messagebox.showerror(
                "无法开考",
                "名单中没有待考考生，请先选择考生并点击重考",
                parent=self.win,
            )
            return
        ok, msg = self.app._exam_preflight()  # type: ignore[attr-defined]
        if not ok:
            messagebox.showerror("无法开考", msg, parent=self.win)
            return
        if not self._save_roi():
            return
        # 已有台账（例如整场完成后追加重考）时复用，禁止 load_candidates 清空成绩
        if self.scorebook is None:
            run_id = time.strftime("%Y%m%d_%H%M%S")
            base = Path(getattr(self.app, "_record_base_dir", outputs_dir()))
            try:
                base = Path(self.app.record_dir_var.get().strip() or str(outputs_dir()))
            except Exception:
                base = outputs_dir()
            self.run_dir = base / f"exam_{run_id}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            score_path = self.run_dir / "成绩汇总.xlsx"
            self.scorebook = ExamScorebook(path=score_path)
            self.scorebook.load_candidates([r.candidate for r in self.session.rows])
            # 同步 row 对象到 scorebook
            self.session.rows = self.scorebook.rows
            self._row_by_id = {r.row_id: r for r in self.session.rows}
        else:
            run_id = self.session.run_id or time.strftime("%Y%m%d_%H%M%S")
            # 保持 session 与 scorebook 同一批 row 对象
            self.session.rows = self.scorebook.rows
            self._row_by_id = {r.row_id: r for r in self.session.rows}
        self.app._exam_set_active(True, run_id=run_id, run_dir=self.run_dir)  # type: ignore[attr-defined]
        self._dispatch(self.session.handle("start_exam", now=time.monotonic(), run_id=run_id))
        self._refresh_tree()

    def _event(self, name: str) -> None:
        now = time.monotonic()
        if name == "skip_current" and self.session.phase == "recording":
            self._pending_discard = True
        self._dispatch(self.session.handle(name, now=now))
        self._refresh_tree()

    def _resolve_retest_row(self) -> ExamResultRow | None:
        """解析重考目标：优先列表选中行，否则回退到当前叫号行。"""
        sel = self.tree.selection()
        if sel:
            item = sel[0]
            selected_row = self._row_by_id.get(str(item))
            if selected_row is not None:
                return selected_row
            try:
                vals = self.tree.item(item, "values")
            except Exception:
                vals = ()
            sid = str(vals[1]) if len(vals) > 1 else ""
            if sid:
                return next(
                    (
                        row
                        for row in self.session.rows
                        if row.candidate.student_id == sid
                    ),
                    None,
                )
            return None
        # 未选中时：若正在考场流程中，允许对「当前考生」重考
        current = self.session.current
        if current is not None and self.session.phase not in {
            "idle",
            "ready",
            "completed",
            "aborted",
        }:
            return current
        return None

    def _retest_current(self) -> None:
        """队尾追加同名重考行（设计 §7.6）；不立即改叫号，除非场次已结束/中止后再次开考。"""
        selected_row = self._resolve_retest_row()
        if selected_row is None:
            messagebox.showinfo(
                "重考",
                "请先在列表中点选一名考生（已完成 / 失败 / 跳过），再点重考。\n"
                "考试进行中未选中时，默认对当前叫号考生操作。",
                parent=self.win,
            )
            return
        cand = selected_row.candidate
        if self.scorebook is None:
            messagebox.showerror(
                "重考",
                "尚无本场成绩台账，无法重考。\n\n"
                "正确顺序：\n"
                "1. 双摄主界面点「开始」\n"
                "2. 本面板点「开始考试」产生场次与成绩表\n"
                "3. 考生考完（或失败/跳过）后，选中该考生点「重考」\n"
                "4. 若整场已结束/中止，再点一次「开始考试」才会叫到重考行\n\n"
                "说明：仅恢复名单不会恢复旧成绩；关闭面板后重考记录不自动续接。",
                parent=self.win,
            )
            return

        # 设计 §7.6：failed / skipped / completed（不满意）才补考；纯 pending 无需重考
        retestable = frozenset(
            {"completed", "failed", "skipped", "superseded", "cancelled"}
        )
        sid_rows = [
            r
            for r in self.session.rows
            if r.candidate.student_id == cand.student_id
        ]
        if selected_row.status == "processing":
            messagebox.showinfo(
                "重考",
                f"{cand.name} 正在后台比对（processing），请待出分后再重考。",
                parent=self.win,
            )
            return
        if selected_row.status == "recording":
            messagebox.showinfo(
                "重考",
                f"{cand.name} 正在录制。请先「强制结束」或「跳过当前」，"
                "待本段结束后再重考；或等其正常离场完成后再补考。",
                parent=self.win,
            )
            return
        if not any(r.status in retestable for r in sid_rows):
            messagebox.showinfo(
                "重考",
                f"{cand.name} 尚未完成首次作答（仍为 pending），无需重考。\n"
                "请等待叫号上场，或用「跳过当前」后再安排补考。",
                parent=self.win,
            )
            return
        if any(
            r.status == "pending" and r.attempt_index > 1
            for r in sid_rows
        ):
            messagebox.showinfo(
                "重考",
                f"{cand.name} 已有待考的重考行在队尾，请先考完该次再追加。",
                parent=self.win,
            )
            return

        phase_before = self.session.phase
        row = self.scorebook.append_retest(cand)
        self.session.rows = self.scorebook.rows
        self.session.queue_retest_row(row)
        self._row_by_id[row.row_id] = row
        self._refresh_tree()
        try:
            self.tree.selection_set(row.row_id)
            self.tree.focus(row.row_id)
            self.tree.see(row.row_id)
        except Exception:
            pass
        summary = (
            f"已追加重考：{cand.name} 第{row.attempt_index}次（队尾，"
            f"学号 {cand.student_id}）"
        )
        self.status_var.set(summary)

        # 终态入队后 phase 会回到 ready，但不会自动叫号——这是现场最易踩坑处
        if self.session.phase == "ready" and phase_before in {"completed", "aborted"}:
            tip = (
                f"{summary}\n\n"
                f"场次已从「{phase_before}」恢复为 ready。\n"
                "重考不会自动叫号，必须再点一次「开始考试」。\n"
                "（中止后会先续考名单中仍 pending 的人，重考行在队尾。）\n\n"
                "是否现在开始考试？"
            )
            if messagebox.askyesno("重考已入队", tip, parent=self.win):
                self._start_exam()
            return

        if phase_before not in {"idle", "ready", "completed", "aborted"}:
            messagebox.showinfo(
                "重考已入队",
                f"{summary}\n\n"
                "考试仍在进行：新行已排在名单最末尾，"
                "不会打断当前考生；轮到队尾时才会叫号。",
                parent=self.win,
            )

    def _export(self) -> None:
        if self.scorebook is None:
            messagebox.showinfo("导出", "无成绩台账", parent=self.win)
            return
        self.scorebook.flush()
        default = str(self.run_dir / "成绩汇总.xlsx") if self.run_dir else "成绩汇总.xlsx"
        path = filedialog.asksaveasfilename(
            parent=self.win,
            title="导出成绩",
            defaultextension=".xlsx",
            initialfile=Path(default).name,
            filetypes=[("Excel", "*.xlsx")],
        )
        if not path:
            return
        try:
            from core.exam_roster import write_scorebook_xlsx

            write_scorebook_xlsx(path, self.scorebook.rows)
            messagebox.showinfo("导出", f"已导出：{path}", parent=self.win)
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc), parent=self.win)

    def _dispatch(self, cmds: list[ExamCommand]) -> None:
        for cmd in cmds:
            try:
                self._run_command(cmd)
            except Exception as exc:
                self.status_var.set(f"命令失败 {cmd.kind}: {exc}")
                traceback.print_exc()
        self.phase_var.set(f"阶段：{self.session.phase}")

    def _run_command(self, cmd: ExamCommand) -> None:
        k = cmd.kind
        p = cmd.payload
        if k == "ANNOUNCE":
            self._announcer.announce(str(p.get("text") or ""))
        elif k == "ARM_OCCUPANCY":
            mode = "wait_enter" if self.session.phase == "wait_enter" else "recording"
            if self.session.phase == "recording":
                self.gate.begin_recording(time.monotonic())
            else:
                self.gate.begin_wait_enter()
            self.app._exam_arm_occupancy(True)  # type: ignore[attr-defined]
        elif k == "DISARM_OCCUPANCY":
            self.app._exam_arm_occupancy(False)  # type: ignore[attr-defined]
        elif k == "BEGIN_SEGMENT":
            row = self._row_by_id.get(str(p.get("row_id") or ""))
            ok = self.app._begin_recording_segment(exam_row=row)  # type: ignore[attr-defined]
            if not ok:
                # begin_failed 内部会 UPDATE failed 并推进指针；不得再执行外层
                # 同批残留的 UPDATE_ROW(recording)（enter_stable 已不再附带该命令）。
                self._dispatch(
                    self.session.handle(
                        "begin_failed",
                        now=time.monotonic(),
                        message="开录失败（请确认双摄已运行且空闲）",
                    )
                )
            else:
                if row is not None and self.scorebook is not None:
                    # begin_failed 重试成功：清失败标记
                    self.scorebook.update_row(
                        row,
                        status="recording",
                        error_code=None,
                        error_message=None,
                    )
                self.gate.begin_recording(time.monotonic())
                self.app._exam_arm_occupancy(True)  # type: ignore[attr-defined]
                self.session.handle("record_started", now=time.monotonic())
                self._refresh_tree()
        elif k == "END_SEGMENT":
            self._pending_discard = False
            row = self._row_by_id.get(str(p.get("row_id") or ""))
            self.app._end_recording_segment(discard=False, exam_row=row)  # type: ignore[attr-defined]
            self._dispatch(
                self.session.handle(
                    "record_stopped", now=time.monotonic(), discarded=False
                )
            )
        elif k == "DISCARD_SEGMENT":
            self._pending_discard = True
            row = self._row_by_id.get(str(p.get("row_id") or ""))
            self.app._end_recording_segment(discard=True, exam_row=row)  # type: ignore[attr-defined]
            # 正常 skip 路径 phase=finishing，需 record_stopped 推进；
            # abort 路径 phase 已是 aborted，只停 writer，不再推进/覆盖行状态。
            if self.session.phase == "finishing":
                self._dispatch(
                    self.session.handle(
                        "record_stopped", now=time.monotonic(), discarded=True
                    )
                )
        elif k == "UPDATE_ROW":
            row = self._row_by_id.get(str(p.get("row_id") or ""))
            if row is not None and self.scorebook is not None:
                fields = {kk: vv for kk, vv in p.items() if kk != "row_id"}
                self.scorebook.update_row(row, **fields)
            self._refresh_tree()
        elif k == "EXPORT":
            if self.scorebook is not None:
                self.scorebook.flush()
                err = self.scorebook.write_error
                if err:
                    self.status_var.set(err)
                else:
                    self.status_var.set(
                        f"成绩已写入：{self.scorebook.path}" if self.scorebook.path else "成绩已更新"
                    )
        elif k == "UI_LOCK_MANUAL_RECORD":
            self.app._exam_lock_manual_record(True)  # type: ignore[attr-defined]
        elif k == "UI_UNLOCK_MANUAL_RECORD":
            self.app._exam_lock_manual_record(False)  # type: ignore[attr-defined]
        elif k in {"SCHEDULE_CALL_GUARD", "SCHEDULE_INTER_GAP", "ADVANCE_POINTER", "BIND_SEGMENT_PENDING"}:
            pass

    def on_occupancy(self, present: bool, now: float) -> None:
        """主线程：worker 上报占用。"""
        self.present_var.set("占用：有人" if present else "占用：无人")
        phase = self.session.phase
        if phase not in {"wait_enter", "recording"}:
            return
        mode = "wait_enter" if phase == "wait_enter" else "recording"
        events = self.gate.update(present, now, mode=mode)
        for ev in events:
            if ev.kind == "enter_stable":
                self._dispatch(self.session.handle("enter_stable", now=now))
            elif ev.kind == "empty_stable":
                self._dispatch(self.session.handle("empty_stable", now=now))
        self._refresh_tree()

    def on_postprocess_update(self, update: Any) -> ExamResultRow | None:
        """回填当前面板台账；未匹配返回 None（App 会继续尝试旧 sinks）。"""
        if self.scorebook is None:
            return None
        row = apply_postprocess_update_to_scorebook(self.scorebook, update)
        if row is None:
            return None
        try:
            if self.win.winfo_exists():
                self._refresh_tree()
        except Exception:
            pass
        return row

    def on_session_stop(self) -> None:
        """主窗口「停止」：同步中止考试状态机并解除占用/手动锁。"""
        try:
            if self.session.phase not in {"idle", "ready", "completed", "aborted"}:
                self._dispatch(
                    self.session.handle("abort_exam", now=time.monotonic())
                )
        except Exception:
            traceback.print_exc()
        try:
            self.app._exam_arm_occupancy(False)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            self.app._exam_set_active(False)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            self.app._exam_lock_manual_record(False)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            self.phase_var.set(f"阶段：{self.session.phase}")
            self.status_var.set("主会话已停止，考试已中止")
        except Exception:
            pass

    def prepare_for_app_close(self) -> None:
        """主窗口销毁前：中止考试、flush 成绩簿并移交 App sinks（保证落盘）。

        主窗 ``root.destroy`` 不会触发本面板 ``WM_DELETE_WINDOW``/``_on_close``，
        必须在此显式移交 scorebook，否则只关 sinks 会漏掉当前面板台账。
        """
        self.on_session_stop()
        try:
            self.win.after_cancel(self._tick_id)
        except Exception:
            pass
        # 与 _on_close 一致：停播报线程/语音，避免窗毁后仍 Speak
        try:
            self._announcer.close()
        except Exception:
            pass
        if self.scorebook is not None:
            try:
                self.scorebook.flush()
            except Exception:
                pass
            # 一律登记 sink：应用关闭末段统一再 flush/close；processing 行仍可回填
            try:
                self.app._register_exam_scorebook_sink(self.scorebook)  # type: ignore[attr-defined]
            except Exception:
                sinks = getattr(self.app, "_exam_scorebook_sinks", None)
                if sinks is None:
                    self.app._exam_scorebook_sinks = [self.scorebook]  # type: ignore[attr-defined]
                elif self.scorebook not in sinks:
                    sinks.append(self.scorebook)
            self.scorebook = None
        try:
            delattr(self.app, "_exam_panel")
        except Exception:
            pass
        try:
            self.win.destroy()
        except Exception:
            pass

    def bind_segment(self, row_id: str, segment_id: str, segment_dir: str) -> None:
        row = self._row_by_id.get(row_id)
        if row is None or self.scorebook is None:
            return
        self.scorebook.bind_segment(row, segment_id=segment_id, segment_dir=segment_dir)

    def _refresh_tree(self) -> None:
        existing = set(self.tree.get_children())
        active: set[str] = set()
        for r in self.session.rows:
            iid = r.row_id
            active.add(iid)
            score = "" if r.combined_percent is None else str(r.combined_percent)
            values = (
                r.candidate.order,
                r.candidate.student_id,
                r.candidate.name,
                r.status,
                score,
            )
            if iid in existing:
                self.tree.item(iid, values=values)
                self.tree.move(iid, "", "end")
            else:
                self.tree.insert("", "end", iid=iid, values=values)
            self._row_by_id[iid] = r
        for stale in existing - active:
            self.tree.delete(stale)
            self._row_by_id.pop(str(stale), None)
        self.phase_var.set(f"阶段：{self.session.phase}")

    def _panel_tick(self) -> None:
        try:
            if not self.win.winfo_exists():
                return
            now = time.monotonic()
            self._dispatch(self.session.handle("tick", now=now))
            if self.scorebook and self.scorebook.write_error:
                self.status_var.set(self.scorebook.write_error)
        except Exception:
            traceback.print_exc()
        try:
            self._tick_id = self.win.after(100, self._panel_tick)
        except Exception:
            pass

    def _on_close(self) -> None:
        try:
            self.win.after_cancel(self._tick_id)
        except Exception:
            pass
        # 关窗必须先中止：录制中/暂停自 recording 则 discard，避免后台继续写
        try:
            if self.session.phase not in {"idle", "ready", "completed", "aborted"}:
                self._dispatch(
                    self.session.handle("abort_exam", now=time.monotonic())
                )
        except Exception:
            traceback.print_exc()
            try:
                self.app._end_recording_segment(discard=True)  # type: ignore[attr-defined]
            except Exception:
                pass
        self.app._exam_arm_occupancy(False)  # type: ignore[attr-defined]
        self.app._exam_set_active(False)  # type: ignore[attr-defined]
        self.app._exam_lock_manual_record(False)  # type: ignore[attr-defined]
        self._announcer.close()
        if self.scorebook is not None:
            try:
                self.scorebook.flush()
            except Exception:
                pass
            # 仍有 processing 行时登记到 App sinks（可多场并存，禁止覆盖）；
            # 无未完成行才立即 close。
            has_processing = any(
                r.status == "processing" for r in self.scorebook.rows
            )
            if has_processing:
                try:
                    self.app._register_exam_scorebook_sink(self.scorebook)  # type: ignore[attr-defined]
                except Exception:
                    # 兼容旧桩
                    sinks = getattr(self.app, "_exam_scorebook_sinks", None)
                    if sinks is None:
                        self.app._exam_scorebook_sinks = [self.scorebook]  # type: ignore[attr-defined]
                    elif self.scorebook not in sinks:
                        sinks.append(self.scorebook)
            else:
                try:
                    self.scorebook.close()
                except Exception:
                    pass
            self.scorebook = None
        try:
            delattr(self.app, "_exam_panel")
        except Exception:
            pass
        self.win.destroy()
