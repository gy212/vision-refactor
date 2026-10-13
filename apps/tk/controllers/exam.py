# -*- coding: utf-8 -*-
"""考试窗口的控制器：名单导入与恢复、ROI、开考、重考、导出成绩与关窗收尾；叫号状态机与命令执行在 ``core.workflows``。

控制器经 ``ExamHost``（组合根提供：录制与站位端口、开考预检、考试模式、手动录制锁、工作流路由、台账登记表）
与主窗口协作，经 ``ExamView`` 更新界面与弹框。它自己实现 ``WorkflowUi``，供 ``CommandExecutor`` 回写状态。
"""
from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from core.foundation import (
    EXAM_EMPTY_HOLD_S,
    EXAM_ENTER_STABLE_S,
    EXAM_INTER_STUDENT_GAP_S,
    EXAM_MIN_RECORD_S,
    EXAM_POST_CALL_GUARD_S,
    EXAM_ROI,
    EXAM_ROI_ROTATE,
    EXAM_ROSTER_PATH,
    PreferenceStore,
)
from core.workflows import (
    Announcer,
    Command,
    CommandExecutor,
    ExamCandidate,
    ExamResultRow,
    ExamScorebook,
    ExamSession,
    ExamSessionConfig,
    RosterError,
    WorkflowPorts,
    import_roster_xlsx,
    write_scorebook_xlsx,
)

IDLE_PHASES = frozenset({"idle", "ready", "completed", "aborted"})
RETESTABLE = frozenset({"completed", "failed", "skipped", "superseded", "cancelled"})

NO_SCOREBOOK_TEXT = (
    "尚无本场成绩台账，无法重考。\n\n"
    "正确顺序：\n"
    "1. 双摄主界面点「开始」\n"
    "2. 本面板点「开始考试」产生场次与成绩表\n"
    "3. 考生考完（或失败/跳过）后，选中该考生点「重考」\n"
    "4. 若整场已结束/中止，再点一次「开始考试」才会叫到重考行\n\n"
    "说明：仅恢复名单不会恢复旧成绩；关闭面板后重考记录不自动续接。"
)
NO_SELECTION_TEXT = (
    "请先在列表中点选一名考生（已完成 / 失败 / 跳过），再点重考。\n"
    "考试进行中未选中时，默认对当前叫号考生操作。"
)


class ExamHost(Protocol):
    """主窗口为考试提供的公共接口。"""

    workflows: Any
    exam_scorebooks: Any

    def exam_ports(self, *, ui: Any, announcer: Any) -> WorkflowPorts: ...
    def exam_preflight(self) -> list[str]: ...
    def set_exam_mode(self, active: bool) -> None: ...
    def lock_manual_record(self, locked: bool) -> None: ...
    def exam_primary_rotation(self) -> int: ...
    def exam_record_dir(self) -> Path: ...
    def post(self, callback: Callable[[], Any]) -> Any: ...
    def exam_closed(self, controller: "ExamController") -> None: ...


class ExamView(Protocol):
    def info(self, title: str, text: str) -> None: ...
    def error(self, title: str, text: str) -> None: ...
    def ask(self, title: str, text: str) -> bool: ...
    def set_phase(self, text: str) -> None: ...
    def set_status(self, text: str) -> None: ...
    def set_present(self, text: str) -> None: ...
    def set_announce(self, text: str) -> None: ...
    def set_roi(self, roi: tuple[float, float, float, float]) -> None: ...
    def show_rows(self, rows: Sequence[ExamResultRow]) -> None: ...
    def select_row(self, row_id: str) -> None: ...
    def selected(self) -> tuple[str, str] | None: ...
    def is_open(self) -> bool: ...
    def destroy(self) -> None: ...


def session_config(prefs: dict) -> ExamSessionConfig:
    return ExamSessionConfig(
        post_call_guard_s=EXAM_POST_CALL_GUARD_S.read(prefs),
        inter_student_gap_s=EXAM_INTER_STUDENT_GAP_S.read(prefs),
        enter_stable_s=EXAM_ENTER_STABLE_S.read(prefs),
        empty_hold_s=EXAM_EMPTY_HOLD_S.read(prefs),
        min_record_s=EXAM_MIN_RECORD_S.read(prefs),
    )


def roi_problem(roi: Sequence[float]) -> str | None:
    """ROI 须满足 0≤x0<x1≤1 且 0≤y0<y1≤1；不满足时返回提示。"""
    if not (0.0 <= roi[0] < roi[2] <= 1.0 and 0.0 <= roi[1] < roi[3] <= 1.0):
        return "须满足 0≤x0<x1≤1 且 0≤y0<y1≤1"
    return None


def start_refusal(session: ExamSession) -> str | None:
    """开考前的阶段与名单检查；可以开考时返回 None。"""
    if not session.rows:
        return "请先导入名单"
    if session.phase != "ready":
        if session.phase == "completed":
            return "名单已全部完成，请先选择考生并点击重考"
        if session.phase == "aborted":
            return "考试已中止，请先选择考生并点击重考"
        return f"当前阶段 {session.phase}，无法重复开始"
    if not any(row.status == "pending" for row in session.rows):
        return "名单中没有待考考生，请先选择考生并点击重考"
    return None


def retest_refusal(row: ExamResultRow, same_student: Sequence[ExamResultRow]) -> str | None:
    """设计 §7.6：failed / skipped / completed（不满意）才补考；纯 pending 无需重考。可以重考时返回 None。"""
    name = row.candidate.name
    if row.status == "processing":
        return f"{name} 正在后台比对（processing），请待出分后再重考。"
    if row.status == "recording":
        return (f"{name} 正在录制。请先「强制结束」或「跳过当前」，"
                "待本段结束后再重考；或等其正常离场完成后再补考。")
    if not any(r.status in RETESTABLE for r in same_student):
        return (f"{name} 尚未完成首次作答（仍为 pending），无需重考。\n"
                "请等待叫号上场，或用「跳过当前」后再安排补考。")
    if any(r.status == "pending" and r.attempt_index > 1 for r in same_student):
        return f"{name} 已有待考的重考行在队尾，请先考完该次再追加。"
    return None


class ExamController:
    def __init__(self, host: ExamHost, view: ExamView, *, prefs: dict | None = None,
                 announcer: Any = None, restore: bool = True) -> None:
        self.host = host
        self.view = view
        prefs = PreferenceStore().raw() if prefs is None else prefs
        self.session = ExamSession(session_config(prefs))
        self.roi = EXAM_ROI.read(prefs)
        self.run_dir: Path | None = None
        self._announcer = announcer if announcer is not None else Announcer(on_text=self._on_announce_text)
        self.executor = CommandExecutor(host.exam_ports(ui=self, announcer=self._announcer))
        self.scorebook = None
        self._row_by_id: dict[str, ExamResultRow] = {}
        view.set_roi(self.roi)
        if restore:
            self.restore_roster(EXAM_ROSTER_PATH.read(prefs))
        # 站位样本、会话停止与定时 tick 由主窗口的 WorkflowHost 送来
        host.workflows.activate(self.session, self.executor)

    @property
    def scorebook(self) -> ExamScorebook | None:
        return self._scorebook

    @scorebook.setter
    def scorebook(self, book: ExamScorebook | None) -> None:
        """当前场次的台账；同步给命令执行端口，UPDATE_ROW / EXPORT 写进它。"""
        self._scorebook = book
        executor = self.__dict__.get("executor")
        if executor is not None:
            executor.ports.scorebook = book

    def _on_announce_text(self, text: str) -> None:
        """播报线程回调：经 dispatcher 回到 Tk 线程更新文字。"""
        self.host.post(lambda: self.view.is_open() and self.view.set_announce(f"播报：{text}"))

    # ------------------------------------------------------------------ 名单
    def can_import(self) -> bool:
        if self.session.phase not in IDLE_PHASES:
            self.view.error("无法导入", f"当前阶段 {self.session.phase}，请先中止当前考试再更换名单")
            return False
        return True

    def import_roster(self, path: Path) -> None:
        try:
            count = self.load_roster(path, persist=True)
        except RosterError as exc:
            self.view.error("导入失败", exc.message)
            return
        self.view.set_status(f"已导入 {count} 人")

    def load_roster(self, path: Path, *, persist: bool) -> int:
        candidates = import_roster_xlsx(path)
        self.replace_roster(candidates)
        if persist:
            PreferenceStore().set(EXAM_ROSTER_PATH, str(path.resolve()))
        return len(candidates)

    def restore_roster(self, raw_path: str) -> None:
        if not raw_path:
            return
        try:
            count = self.load_roster(Path(raw_path), persist=False)
        except RosterError as exc:
            self.view.set_status(f"上次名单未能恢复：{exc.message}；请重新导入")
            return
        self.view.set_status(f"已恢复上次名单：{count} 人")

    def replace_roster(self, candidates: list[ExamCandidate]) -> None:
        """新名单 = 新场次：关闭旧台账（仍有 processing 行时移交登记表），解除占用、考试模式与手动锁。"""
        if self.scorebook is not None:
            self._hand_over_scorebook()
        self.run_dir = None
        for release in (
            lambda: self.executor.ports.occupancy.arm(False),
            lambda: self.host.set_exam_mode(False),
            lambda: self.host.lock_manual_record(False),
        ):
            try:
                release()
            except Exception:
                pass
        self.session.load_roster(candidates)
        self._row_by_id = {r.row_id: r for r in self.session.rows}
        self.refresh_rows()

    def _hand_over_scorebook(self) -> None:
        book = self.scorebook
        try:
            book.flush()
        except Exception:
            pass
        if any(row.status == "processing" for row in book.rows):
            self.host.exam_scorebooks.detach(book)
        else:
            try:
                book.close()
            except Exception:
                pass
        self.scorebook = None

    # ------------------------------------------------------------------ ROI 与开考
    def save_roi(self, values: Callable[[], Sequence[float]]) -> bool:
        try:
            roi = tuple(float(v) for v in values())
        except Exception:
            self.view.error("ROI", "ROI 数值无效")
            return False
        problem = roi_problem(roi)
        if problem:
            self.view.error("ROI", problem)
            return False
        self.roi = roi  # type: ignore[assignment]
        # ROI 在 primary 旋转后的画面上判定，一并记下当前旋转
        try:
            rotate = int(self.host.exam_primary_rotation())
        except Exception:
            rotate = 0
        PreferenceStore().update({EXAM_ROI: self.roi, EXAM_ROI_ROTATE: rotate})
        self.view.set_roi(self.roi)
        self.executor.ports.occupancy.set_roi(self.roi)
        self.view.set_status("ROI 已保存")
        return True

    def start_exam(self, save_roi: Callable[[], bool]) -> None:
        refusal = start_refusal(self.session)
        if refusal:
            self.view.error("无法开考", refusal)
            return
        problems = self.host.exam_preflight()
        if problems:
            self.view.error("无法开考", problems[0])
            return
        if not save_roi():
            return
        # 已有台账（例如整场完成后追加重考）时复用，禁止 load_candidates 清空成绩
        if self.scorebook is None:
            run_id = time.strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path(self.host.exam_record_dir()) / f"exam_{run_id}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.scorebook = ExamScorebook(path=self.run_dir / "成绩汇总.xlsx")
            self.scorebook.load_candidates([r.candidate for r in self.session.rows])
        else:
            run_id = self.session.run_id or time.strftime("%Y%m%d_%H%M%S")
        # session 与 scorebook 用同一批 row 对象
        self.session.rows = self.scorebook.rows
        self._row_by_id = {r.row_id: r for r in self.session.rows}
        self.host.exam_scorebooks.register(run_id, self.scorebook)
        self.host.set_exam_mode(True)
        self.dispatch(self.session.handle("start_exam", now=time.monotonic(), run_id=run_id))
        self.refresh_rows()

    def event(self, name: str) -> None:
        self.dispatch(self.session.handle(name, now=time.monotonic()))
        self.refresh_rows()

    def dispatch(self, cmds: list[Command]) -> None:
        """执行面板按钮产生的命令（命令失败只提示，不中断后续命令）。"""
        self.executor.run(self.session, cmds)
        self.refresh()

    # ------------------------------------------------------------------ 重考
    def retest_target(self) -> ExamResultRow | None:
        """重考目标：优先列表选中行，否则回退到考试进行中的当前叫号行。"""
        selected = self.view.selected()
        if selected is not None:
            row_id, student_id = selected
            row = self._row_by_id.get(row_id)
            if row is not None or not student_id:
                return row
            return next((r for r in self.session.rows if r.candidate.student_id == student_id), None)
        current = self.session.current
        if current is not None and self.session.phase not in IDLE_PHASES:
            return current
        return None

    def retest(self, save_roi: Callable[[], bool]) -> None:
        """队尾追加同名重考行（设计 §7.6）；不立即改叫号，除非场次已结束/中止后再次开考。"""
        selected = self.retest_target()
        if selected is None:
            self.view.info("重考", NO_SELECTION_TEXT)
            return
        if self.scorebook is None:
            self.view.error("重考", NO_SCOREBOOK_TEXT)
            return
        cand = selected.candidate
        same = [r for r in self.session.rows if r.candidate.student_id == cand.student_id]
        refusal = retest_refusal(selected, same)
        if refusal:
            self.view.info("重考", refusal)
            return
        phase_before = self.session.phase
        row = self.scorebook.append_retest(cand)
        self.session.rows = self.scorebook.rows
        self.session.queue_retest_row(row)
        self._row_by_id[row.row_id] = row
        self.refresh_rows()
        self.view.select_row(row.row_id)
        summary = f"已追加重考：{cand.name} 第{row.attempt_index}次（队尾，学号 {cand.student_id}）"
        self.view.set_status(summary)
        # 终态入队后 phase 会回到 ready，但不会自动叫号——这是现场最易踩坑处
        if self.session.phase == "ready" and phase_before in {"completed", "aborted"}:
            tip = (
                f"{summary}\n\n"
                f"场次已从「{phase_before}」恢复为 ready。\n"
                "重考不会自动叫号，必须再点一次「开始考试」。\n"
                "（中止后会先续考名单中仍 pending 的人，重考行在队尾。）\n\n"
                "是否现在开始考试？"
            )
            if self.view.ask("重考已入队", tip):
                self.start_exam(save_roi)
            return
        if phase_before not in IDLE_PHASES:
            self.view.info(
                "重考已入队",
                f"{summary}\n\n考试仍在进行：新行已排在名单最末尾，不会打断当前考生；轮到队尾时才会叫号。",
            )

    # ------------------------------------------------------------------ 导出
    def export_default_name(self) -> str | None:
        """导出前 flush 台账并给出默认文件名；没有台账时提示并返回 None。"""
        if self.scorebook is None:
            self.view.info("导出", "无成绩台账")
            return None
        self.scorebook.flush()
        return Path(str(self.run_dir / "成绩汇总.xlsx") if self.run_dir else "成绩汇总.xlsx").name

    def export(self, path: str) -> None:
        try:
            write_scorebook_xlsx(path, self.scorebook.rows)
            self.view.info("导出", f"已导出：{path}")
        except Exception as exc:
            self.view.error("导出失败", str(exc))

    # ------------------------------------------------------------------ WorkflowUi
    def status(self, text: str) -> None:
        self.view.set_status(text)

    def alert(self, title: str, text: str, *, error: bool = False) -> None:
        (self.view.error if error else self.view.info)(title, text)

    def lock_manual_record(self, locked: bool) -> None:
        self.host.lock_manual_record(locked)

    def show_result(self, record: dict[str, Any]) -> None:
        pass

    def refresh_rows(self) -> None:
        """台账被命令或录后结果更新后刷新表格与阶段。"""
        if not self.view.is_open():
            return
        self.view.show_rows(self.session.rows)
        self._row_by_id = {r.row_id: r for r in self.session.rows}
        self.view.set_phase(f"阶段：{self.session.phase}")

    def refresh(self) -> None:
        """每次分派后同步阶段、占用与台账写入错误。"""
        self.view.set_phase(f"阶段：{self.session.phase}")
        if self.session.present is not None:
            self.view.set_present("占用：有人" if self.session.present else "占用：无人")
        if self.scorebook is not None and self.scorebook.write_error:
            self.view.set_status(self.scorebook.write_error)

    # ------------------------------------------------------------------ 关闭
    def prepare_for_app_close(self) -> None:
        """主窗口销毁前：中止考试、停播报、flush 台账并一律移交登记表（应用关闭末段统一 flush/close）。

        主窗 ``root.destroy`` 不会触发本窗口的 ``WM_DELETE_WINDOW``，必须在此显式移交台账。
        """
        try:
            self.executor.dispatch(self.session, "session_stopped", now=time.monotonic())
        except Exception:
            traceback.print_exc()
        try:
            self.host.set_exam_mode(False)
        except Exception:
            pass
        try:
            self._announcer.close()
        except Exception:
            pass
        if self.scorebook is not None:
            try:
                self.scorebook.flush()
            except Exception:
                pass
            self.host.exam_scorebooks.detach(self.scorebook)
            self.scorebook = None
        self._detach()

    def close(self) -> None:
        """用户关窗：先中止（录制中则 discard，避免后台继续写），再解除锁与模式、收台账。"""
        try:
            if self.session.phase not in IDLE_PHASES:
                self.dispatch(self.session.handle("abort_exam", now=time.monotonic()))
        except Exception:
            traceback.print_exc()
            try:
                self.executor.ports.recorder.finish(discard=True)
            except Exception:
                pass
        self.executor.ports.occupancy.arm(False)
        self.host.set_exam_mode(False)
        self.host.lock_manual_record(False)
        self._announcer.close()
        if self.scorebook is not None:
            # 仍有 processing 行时移交台账登记表（可多场并存，禁止覆盖）；无未完成行才立即 close。
            self._hand_over_scorebook()
        self._detach()

    def _detach(self) -> None:
        self.host.workflows.deactivate(self.session)
        self.host.exam_closed(self)
        try:
            self.view.destroy()
        except Exception:
            pass
