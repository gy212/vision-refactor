# -*- coding: utf-8 -*-
"""考试控制器：名单导入/恢复、换名单时的台账移交、表格刷新、重考与开考、关窗收尾（假视图 + 假主窗口）。"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from apps.tk.controllers import exam_rules as exam_mod
from apps.tk.controllers.exam import ExamController
from core.foundation import prefs as user_prefs
from core.workflows import (
    ExamCandidate,
    ExamScorebook,
    ExamSession,
    ScorebookRegistry,
    WorkflowPorts,
    write_import_template,
)


class _View:
    """``ExamView`` 替身：表格按台账顺序记录行，弹框按类型记录文字。"""

    def __init__(self) -> None:
        self.phase = ""
        self.status = ""
        self.present = ""
        self.rows: dict[str, tuple] = {}
        self.order: list[str] = []
        self.selection: list[str] = []
        self.infos: list[str] = []
        self.errors: list[str] = []
        self.answer = False
        self.destroyed = False

    def info(self, _title, text) -> None:
        self.infos.append(str(text))

    def error(self, _title, text) -> None:
        self.errors.append(str(text))

    def ask(self, _title, _text) -> bool:
        return self.answer

    def set_phase(self, text) -> None:
        self.phase = text

    def set_status(self, text) -> None:
        self.status = text

    def set_present(self, text) -> None:
        self.present = text

    def set_announce(self, _text) -> None:
        pass

    def set_roi(self, _roi) -> None:
        pass

    def show_rows(self, rows) -> None:
        from apps.tk.views.exam_window import row_values

        self.order = [r.row_id for r in rows]
        self.rows = {r.row_id: row_values(r) for r in rows}
        self.selection = [i for i in self.selection if i in self.rows]

    def select_row(self, row_id) -> None:
        self.selection = [row_id]

    def selected(self):
        if not self.selection:
            return None
        return self.selection[0], str(self.rows.get(self.selection[0], ("", ""))[1])

    def is_open(self) -> bool:
        return not self.destroyed

    def destroy(self) -> None:
        self.destroyed = True


def _controller(session: ExamSession | None = None, lifecycle: list | None = None, **host_kw) -> ExamController:
    """命令经真实 CommandExecutor 执行，端口与主窗口回调记录到 ``lifecycle``。"""
    events = lifecycle if lifecycle is not None else []
    registry = ScorebookRegistry()
    host = SimpleNamespace(
        workflows=SimpleNamespace(activate=lambda *_a: None, deactivate=lambda s: events.append(("deactivated", s))),
        exam_scorebooks=registry,
        exam_ports=lambda *, ui, announcer: WorkflowPorts(
            recorder=SimpleNamespace(),
            occupancy=SimpleNamespace(
                arm=lambda armed: events.append(("armed", armed)),
                set_roi=lambda roi: events.append(("roi", roi)),
            ),
            announcer=announcer,
            ui=ui,
        ),
        exam_preflight=lambda: [],
        set_exam_mode=lambda active: events.append(("active", active)),
        lock_manual_record=lambda locked: events.append(("locked", locked)),
        exam_primary_rotation=lambda: 0,
        exam_record_dir=lambda: Path("."),
        post=lambda cb: cb(),
        exam_closed=lambda c: events.append(("closed", c)),
    )
    for key, value in host_kw.items():
        setattr(host, key, value)
    announcer = SimpleNamespace(announce=lambda text: events.append(("announce", text)),
                                close=lambda: events.append(("announcer_closed",)))
    controller = ExamController(host, _View(), prefs={}, announcer=announcer, restore=False)
    if session is not None:
        controller.session = session
        controller.refresh_rows()
    return controller


def test_close_callback_detaches_and_destroys_window() -> None:
    """WM_DELETE_WINDOW 回调：释放工作流、通知主窗口后必须销毁窗口，不留下脱管的面板。"""
    events: list = []
    controller = _controller(lifecycle=events)

    controller.close()

    assert ("deactivated", controller.session) in events
    assert ("closed", controller) in events
    assert ("announcer_closed",) in events
    assert controller.view.destroyed


def test_imported_roster_path_is_saved_and_restored(tmp_path: Path) -> None:
    roster_path = write_import_template(tmp_path / "roster.xlsx")
    controller = _controller()

    count = controller.load_roster(roster_path, persist=True)

    assert count == 2
    assert controller.session.phase == "ready"
    assert len(controller.session.rows) == 2
    store = user_prefs.PreferenceStore()
    assert store.get(user_prefs.EXAM_ROSTER_PATH) == str(roster_path.resolve())

    restored = _controller()
    restored.restore_roster(store.get(user_prefs.EXAM_ROSTER_PATH))

    assert restored.session.phase == "ready"
    assert len(restored.session.rows) == 2
    assert "已恢复上次名单" in restored.view.status


def test_missing_saved_roster_stays_idle_with_actionable_status(tmp_path: Path) -> None:
    controller = _controller()

    controller.restore_roster(str(tmp_path / "missing.xlsx"))

    assert controller.session.phase == "idle"
    assert controller.session.rows == []
    assert "请重新导入" in controller.view.status


def test_import_roster_is_blocked_while_exam_is_active() -> None:
    session = ExamSession()
    session.load_roster([ExamCandidate(1, "S1", "甲")])
    session.phase = "recording"
    controller = _controller(session)

    assert controller.can_import() is False

    assert len(controller.view.errors) == 1
    assert "先中止当前考试" in controller.view.errors[0]


class _Book:
    def __init__(self, status: str) -> None:
        self.rows = [SimpleNamespace(status=status)]
        self.flush_calls = 0
        self.close_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def test_replacing_roster_hands_processing_scorebook_to_registry() -> None:
    book = _Book("processing")
    registered = []
    lifecycle: list = []
    controller = _controller(lifecycle=lifecycle, exam_scorebooks=SimpleNamespace(detach=registered.append))
    controller.scorebook = book
    controller.run_dir = Path("old-run")

    controller.replace_roster([ExamCandidate(1, "NEW", "新考生")])

    assert book.flush_calls == 1
    assert book.close_calls == 0
    assert registered == [book]
    assert controller.scorebook is None
    assert controller.run_dir is None
    assert controller.session.phase == "ready"
    assert lifecycle == [("armed", False), ("active", False), ("locked", False)]


def test_replacing_roster_closes_terminal_scorebook() -> None:
    book = _Book("completed")
    controller = _controller()
    controller.scorebook = book

    controller.replace_roster([ExamCandidate(1, "NEW", "新考生")])

    assert book.flush_calls == 1
    assert book.close_calls == 1
    assert controller.scorebook is None
    assert controller.session.phase == "ready"


def test_tree_refresh_preserves_selected_candidate() -> None:
    session = ExamSession()
    session.load_roster([ExamCandidate(1, "S1", "甲"), ExamCandidate(2, "S2", "乙")])
    controller = _controller(session)
    selected_id = session.rows[0].row_id
    controller.view.select_row(selected_id)

    session.rows[0].status = "processing"
    controller.refresh_rows()

    assert controller.view.selected()[0] == selected_id
    assert controller.view.rows[selected_id][3] == "processing"


def _book_session(candidates, phase: str, *, pointer: int = 0, run_id: str | None = None):
    book = ExamScorebook(path=None)
    book.load_candidates(candidates)
    session = ExamSession()
    session.rows = book.rows
    session.phase = phase
    session.pointer = pointer
    session.run_id = run_id
    return book, session


def test_aborted_retest_becomes_startable_without_skipping_remaining_roster(tmp_path: Path) -> None:
    book, session = _book_session(
        [ExamCandidate(1, "S1", "甲"), ExamCandidate(2, "S2", "乙")], "aborted", run_id="run-existing",
    )
    first = book.rows[0]
    book.update_row(first, status="skipped", error_code="exam_aborted")
    events: list = []
    controller = _controller(session, lifecycle=events)
    controller.scorebook = book
    controller.run_dir = tmp_path
    controller.view.select_row(first.row_id)
    commands = []
    controller.dispatch = commands.extend
    controller.view.answer = False  # 测「入队后手动再开考」；不自动点是

    controller.retest(lambda: True)

    assert controller.session.phase == "ready"
    assert controller.session.pointer == 0
    assert len(controller.session.rows) == 3
    assert controller.session.rows[-1].attempt_index == 2
    assert controller.view.selected()[0] == controller.session.rows[-1].row_id

    controller.start_exam(lambda: True)

    assert controller.session.phase == "calling"
    assert controller.session.pointer == 1
    assert controller.session.current is controller.session.rows[1]
    assert [e for e in events if e[0] == "active"] == [("active", True)]
    assert controller.session.run_id == "run-existing"
    assert controller.run_dir == tmp_path
    # 沿用的台账登记到新场次，录后结果按 run_id 回填到它
    registry = controller.host.exam_scorebooks
    assert registry.apply(SimpleNamespace(segment_id="none", exam_run_id="run-existing", status="queued")) is None
    assert registry._by_run["run-existing"] is book
    assert any(command.kind == "ANNOUNCE" for command in commands)
    book.close()


def test_completed_retest_can_auto_start_when_confirmed(tmp_path: Path) -> None:
    book, session = _book_session([ExamCandidate(1, "S1", "甲")], "completed", pointer=1, run_id="run-done")
    first = book.rows[0]
    book.update_row(first, status="completed", combined_percent=88, combined_score=0.88)
    controller = _controller(session)
    controller.scorebook = book
    controller.run_dir = tmp_path
    controller.view.select_row(first.row_id)
    commands: list = []
    controller.dispatch = commands.extend
    controller.view.answer = True

    controller.retest(lambda: True)

    assert controller.session.phase == "calling"
    assert controller.session.pointer == 1
    assert controller.session.current is not None
    assert controller.session.current.attempt_index == 2
    assert any(c.kind == "ANNOUNCE" for c in commands)
    book.close()


def test_retest_without_scorebook_explains_workflow() -> None:
    session = ExamSession()
    session.load_roster([ExamCandidate(1, "S1", "甲")])
    controller = _controller(session)
    controller.view.select_row(session.rows[0].row_id)

    controller.retest(lambda: True)

    assert len(controller.view.errors) == 1
    assert "成绩台账" in controller.view.errors[0]
    assert "开始考试" in controller.view.errors[0]


def test_retest_rejects_pure_pending_candidate() -> None:
    book, session = _book_session([ExamCandidate(1, "S1", "甲")], "ready")
    controller = _controller(session)
    controller.scorebook = book
    controller.view.select_row(session.rows[0].row_id)

    controller.retest(lambda: True)

    assert len(controller.session.rows) == 1
    assert any("无需重考" in msg for msg in controller.view.infos)
    book.close()


def test_retest_falls_back_to_session_current_when_no_selection() -> None:
    book, session = _book_session([ExamCandidate(1, "S1", "甲"), ExamCandidate(2, "S2", "乙")], "wait_enter")
    book.update_row(book.rows[0], status="failed", error_code="begin_failed")
    controller = _controller(session)
    controller.scorebook = book
    controller.view.selection = []  # 故意不选中

    controller.retest(lambda: True)

    assert len(controller.session.rows) == 3
    assert controller.session.rows[-1].attempt_index == 2
    assert controller.session.rows[-1].candidate.student_id == "S1"
    assert any("队尾" in msg for msg in controller.view.infos)
    book.close()


def test_nonempty_aborted_session_does_not_claim_roster_is_missing() -> None:
    session = ExamSession()
    session.load_roster([ExamCandidate(1, "S1", "甲")])
    session.phase = "aborted"

    def preflight():
        raise AssertionError("phase validation must run before preflight")

    controller = _controller(session, exam_preflight=preflight)

    controller.start_exam(lambda: True)

    assert len(controller.view.errors) == 1
    assert "中止" in controller.view.errors[0]
    assert "导入名单" not in controller.view.errors[0]


def test_retest_and_start_refusal_texts() -> None:
    row = SimpleNamespace(candidate=SimpleNamespace(name="甲"), status="processing", attempt_index=1)
    assert "processing" in exam_mod.retest_refusal(row, [row])
    row.status = "completed"
    assert exam_mod.retest_refusal(row, [row]) is None
    assert exam_mod.roi_problem((0.5, 0.0, 0.4, 1.0)) is not None
    assert exam_mod.roi_problem((0.1, 0.1, 0.9, 0.9)) is None
    assert exam_mod.start_refusal(ExamSession()) == "请先导入名单"


def test_exam_window_builds_on_real_tk() -> None:
    import tkinter as tk

    import pytest

    from apps.tk.views import ExamWindow

    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - 无显示环境
        pytest.skip(f"Tk 不可用：{exc}")
    root.withdraw()
    try:
        host = _controller().host
        window = ExamWindow(root, host)
        controller = window.controller
        controller.replace_roster([ExamCandidate(1, "S1", "甲"), ExamCandidate(2, "S2", "乙")])
        assert window.tree.get_children() == tuple(r.row_id for r in controller.session.rows)
        assert window.phase_var.get() == "阶段：ready"
        window.select_row(controller.session.rows[1].row_id)
        assert window.selected() == (controller.session.rows[1].row_id, "S2")
        controller.close()
        assert not window.is_open()
    finally:
        root.destroy()
