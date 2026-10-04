# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from apps import exam_panel
from core.exam_roster import ExamCandidate, ExamScorebook, write_import_template
from core.exam_session import ExamSession


class _Var:
    def __init__(self, value="") -> None:
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


class _Tree:
    def __init__(self) -> None:
        self.items: dict[str, tuple[object, ...]] = {}
        self.order: list[str] = []
        self.selected: list[str] = []
        self.focused = ""
        self.seen = ""

    def get_children(self):
        return tuple(self.order)

    def insert(self, _parent, _where, *, iid, values) -> None:
        key = str(iid)
        self.items[key] = tuple(values)
        self.order.append(key)

    def item(self, iid, option=None, **kwargs):
        key = str(iid)
        if "values" in kwargs:
            self.items[key] = tuple(kwargs["values"])
        if option == "values":
            return self.items[key]
        return {"values": self.items[key]}

    def move(self, iid, _parent, _where) -> None:
        key = str(iid)
        self.order.remove(key)
        self.order.append(key)

    def delete(self, iid) -> None:
        key = str(iid)
        self.items.pop(key, None)
        if key in self.order:
            self.order.remove(key)
        if key in self.selected:
            self.selected.remove(key)

    def selection(self):
        return tuple(self.selected)

    def selection_set(self, *iids) -> None:
        self.selected = [str(iid) for iid in iids]

    def focus(self, iid=None):
        if iid is not None:
            self.focused = str(iid)
        return self.focused

    def see(self, iid) -> None:
        self.seen = str(iid)


def _bare_panel(session: ExamSession | None = None) -> exam_panel.ExamPanel:
    panel = object.__new__(exam_panel.ExamPanel)
    panel.session = session or ExamSession()
    panel.scorebook = None
    panel.run_dir = None
    panel._row_by_id = {row.row_id: row for row in panel.session.rows}
    panel.tree = _Tree()
    panel.phase_var = _Var(f"阶段：{panel.session.phase}")
    panel.status_var = _Var("")
    panel.win = object()
    return panel


def test_imported_roster_path_is_saved_and_restored(
    monkeypatch, tmp_path: Path
) -> None:
    roster_path = write_import_template(tmp_path / "roster.xlsx")
    saved: list[dict[str, object]] = []
    monkeypatch.setattr(exam_panel, "save_exam_prefs", saved.append)
    panel = _bare_panel()

    count = panel._load_roster_path(roster_path, persist=True)

    assert count == 2
    assert panel.session.phase == "ready"
    assert len(panel.session.rows) == 2
    assert saved == [
        {exam_panel._EXAM_ROSTER_PATH_PREF_KEY: str(roster_path.resolve())}
    ]

    restored = _bare_panel()
    restored._restore_roster(
        {exam_panel._EXAM_ROSTER_PATH_PREF_KEY: str(roster_path)}
    )

    assert restored.session.phase == "ready"
    assert len(restored.session.rows) == 2
    assert "已恢复上次名单" in restored.status_var.get()


def test_missing_saved_roster_stays_idle_with_actionable_status(tmp_path: Path) -> None:
    panel = _bare_panel()

    panel._restore_roster(
        {exam_panel._EXAM_ROSTER_PATH_PREF_KEY: str(tmp_path / "missing.xlsx")}
    )

    assert panel.session.phase == "idle"
    assert panel.session.rows == []
    assert "请重新导入" in panel.status_var.get()


def test_import_roster_is_blocked_while_exam_is_active(monkeypatch) -> None:
    session = ExamSession()
    session.load_roster([ExamCandidate(1, "S1", "甲")])
    session.phase = "recording"
    panel = _bare_panel(session)
    dialog_calls: list[str] = []
    errors: list[str] = []
    monkeypatch.setattr(
        exam_panel.filedialog,
        "askopenfilename",
        lambda **_kwargs: dialog_calls.append("open") or "roster.xlsx",
    )
    monkeypatch.setattr(
        exam_panel.messagebox,
        "showerror",
        lambda _title, message, **_kwargs: errors.append(str(message)),
    )

    panel._import_roster()

    assert dialog_calls == []
    assert len(errors) == 1
    assert "先中止当前考试" in errors[0]


def test_replacing_roster_hands_processing_scorebook_to_app_sink() -> None:
    class _Book:
        def __init__(self) -> None:
            self.rows = [SimpleNamespace(status="processing")]
            self.flush_calls = 0
            self.close_calls = 0

        def flush(self) -> None:
            self.flush_calls += 1

        def close(self) -> None:
            self.close_calls += 1

    book = _Book()
    registered = []
    lifecycle: list[tuple[str, bool]] = []
    panel = _bare_panel()
    panel.scorebook = book
    panel.run_dir = Path("old-run")
    panel.app = SimpleNamespace(
        _register_exam_scorebook_sink=registered.append,
        _exam_arm_occupancy=lambda active: lifecycle.append(("armed", active)),
        _exam_set_active=lambda active: lifecycle.append(("active", active)),
        _exam_lock_manual_record=lambda active: lifecycle.append(("locked", active)),
    )

    panel._replace_roster([ExamCandidate(1, "NEW", "新考生")])

    assert book.flush_calls == 1
    assert book.close_calls == 0
    assert registered == [book]
    assert panel.scorebook is None
    assert panel.run_dir is None
    assert panel.session.phase == "ready"
    assert lifecycle == [
        ("armed", False),
        ("active", False),
        ("locked", False),
    ]


def test_replacing_roster_closes_terminal_scorebook() -> None:
    class _Book:
        rows = [SimpleNamespace(status="completed")]

        def __init__(self) -> None:
            self.flush_calls = 0
            self.close_calls = 0

        def flush(self) -> None:
            self.flush_calls += 1

        def close(self) -> None:
            self.close_calls += 1

    book = _Book()
    panel = _bare_panel()
    panel.scorebook = book

    panel._replace_roster([ExamCandidate(1, "NEW", "新考生")])

    assert book.flush_calls == 1
    assert book.close_calls == 1
    assert panel.scorebook is None
    assert panel.session.phase == "ready"


def test_tree_refresh_preserves_selected_candidate() -> None:
    session = ExamSession()
    session.load_roster(
        [
            ExamCandidate(1, "S1", "甲"),
            ExamCandidate(2, "S2", "乙"),
        ]
    )
    panel = _bare_panel(session)
    panel._refresh_tree()
    selected_id = session.rows[0].row_id
    panel.tree.selection_set(selected_id)

    session.rows[0].status = "processing"
    panel._refresh_tree()

    assert panel.tree.selection() == (selected_id,)
    assert panel.tree.item(selected_id, "values")[3] == "processing"


def test_aborted_retest_becomes_startable_without_skipping_remaining_roster(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidates = [
        ExamCandidate(1, "S1", "甲"),
        ExamCandidate(2, "S2", "乙"),
    ]
    book = ExamScorebook(path=None)
    book.load_candidates(candidates)
    first = book.rows[0]
    book.update_row(first, status="skipped", error_code="exam_aborted")

    session = ExamSession()
    session.rows = book.rows
    session.phase = "aborted"
    session.pointer = 0
    session.run_id = "run-existing"
    panel = _bare_panel(session)
    panel.scorebook = book
    panel.run_dir = tmp_path
    panel._row_by_id = {row.row_id: row for row in session.rows}
    panel._refresh_tree()
    panel.tree.selection_set(first.row_id)
    active_calls: list[tuple[bool, str, Path]] = []
    panel.app = SimpleNamespace(
        _exam_preflight=lambda: (True, ""),
        _exam_set_active=lambda active, *, run_id, run_dir: active_calls.append(
            (active, run_id, run_dir)
        ),
    )
    panel._save_roi = lambda: True
    commands = []
    panel._dispatch = commands.extend
    # 测「入队后手动再开考」；不自动点是
    monkeypatch.setattr(
        exam_panel.messagebox, "askyesno", lambda *a, **k: False
    )

    panel._retest_current()

    assert panel.session.phase == "ready"
    assert panel.session.pointer == 0
    assert len(panel.session.rows) == 3
    assert panel.session.rows[-1].attempt_index == 2
    assert panel.tree.selection() == (panel.session.rows[-1].row_id,)

    panel._start_exam()

    assert panel.session.phase == "calling"
    assert panel.session.pointer == 1
    assert panel.session.current is panel.session.rows[1]
    assert active_calls == [(True, "run-existing", tmp_path)]
    assert any(command.kind == "ANNOUNCE" for command in commands)
    book.close()


def test_completed_retest_can_auto_start_when_confirmed(
    tmp_path: Path, monkeypatch
) -> None:
    candidates = [ExamCandidate(1, "S1", "甲")]
    book = ExamScorebook(path=None)
    book.load_candidates(candidates)
    first = book.rows[0]
    book.update_row(
        first,
        status="completed",
        combined_percent=88,
        combined_score=0.88,
    )

    session = ExamSession()
    session.rows = book.rows
    session.phase = "completed"
    session.pointer = 1
    session.run_id = "run-done"
    panel = _bare_panel(session)
    panel.scorebook = book
    panel.run_dir = tmp_path
    panel._row_by_id = {row.row_id: row for row in session.rows}
    panel._refresh_tree()
    panel.tree.selection_set(first.row_id)
    panel.app = SimpleNamespace(
        _exam_preflight=lambda: (True, ""),
        _exam_set_active=lambda *a, **k: None,
    )
    panel._save_roi = lambda: True
    commands: list = []
    panel._dispatch = commands.extend
    monkeypatch.setattr(exam_panel.messagebox, "askyesno", lambda *a, **k: True)

    panel._retest_current()

    assert panel.session.phase == "calling"
    assert panel.session.pointer == 1
    assert panel.session.current is not None
    assert panel.session.current.attempt_index == 2
    assert any(c.kind == "ANNOUNCE" for c in commands)
    book.close()


def test_retest_without_scorebook_explains_workflow(monkeypatch) -> None:
    session = ExamSession()
    session.load_roster([ExamCandidate(1, "S1", "甲")])
    panel = _bare_panel(session)
    panel._refresh_tree()
    panel.tree.selection_set(session.rows[0].row_id)
    errors: list[str] = []
    monkeypatch.setattr(
        exam_panel.messagebox,
        "showerror",
        lambda _t, message, **_k: errors.append(str(message)),
    )

    panel._retest_current()

    assert len(errors) == 1
    assert "成绩台账" in errors[0]
    assert "开始考试" in errors[0]


def test_retest_rejects_pure_pending_candidate(monkeypatch) -> None:
    candidates = [ExamCandidate(1, "S1", "甲")]
    book = ExamScorebook(path=None)
    book.load_candidates(candidates)
    session = ExamSession()
    session.rows = book.rows
    session.phase = "ready"
    panel = _bare_panel(session)
    panel.scorebook = book
    panel._row_by_id = {row.row_id: row for row in session.rows}
    panel._refresh_tree()
    panel.tree.selection_set(session.rows[0].row_id)
    infos: list[str] = []
    monkeypatch.setattr(
        exam_panel.messagebox,
        "showinfo",
        lambda _t, message, **_k: infos.append(str(message)),
    )

    panel._retest_current()

    assert len(panel.session.rows) == 1
    assert any("无需重考" in msg for msg in infos)
    book.close()


def test_retest_falls_back_to_session_current_when_no_selection(
    monkeypatch,
) -> None:
    candidates = [
        ExamCandidate(1, "S1", "甲"),
        ExamCandidate(2, "S2", "乙"),
    ]
    book = ExamScorebook(path=None)
    book.load_candidates(candidates)
    book.update_row(book.rows[0], status="failed", error_code="begin_failed")
    session = ExamSession()
    session.rows = book.rows
    session.phase = "wait_enter"
    session.pointer = 0
    panel = _bare_panel(session)
    panel.scorebook = book
    panel._row_by_id = {row.row_id: row for row in session.rows}
    panel._refresh_tree()
    # 故意不选中
    panel.tree.selected = []
    infos: list[str] = []
    monkeypatch.setattr(
        exam_panel.messagebox,
        "showinfo",
        lambda _t, message, **_k: infos.append(str(message)),
    )

    panel._retest_current()

    assert len(panel.session.rows) == 3
    assert panel.session.rows[-1].attempt_index == 2
    assert panel.session.rows[-1].candidate.student_id == "S1"
    assert any("队尾" in msg for msg in infos)
    book.close()


def test_nonempty_aborted_session_does_not_claim_roster_is_missing(
    monkeypatch,
) -> None:
    session = ExamSession()
    session.load_roster([ExamCandidate(1, "S1", "甲")])
    session.phase = "aborted"
    panel = _bare_panel(session)
    panel.app = SimpleNamespace(
        _exam_preflight=lambda: (_ for _ in ()).throw(
            AssertionError("phase validation must run before preflight")
        )
    )
    errors: list[str] = []
    monkeypatch.setattr(
        exam_panel.messagebox,
        "showerror",
        lambda _title, message, **_kwargs: errors.append(str(message)),
    )

    panel._start_exam()

    assert len(errors) == 1
    assert "中止" in errors[0]
    assert "导入名单" not in errors[0]
