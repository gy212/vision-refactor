# -*- coding: utf-8 -*-
"""关窗流程：组合根注入给 ``ShutdownCoordinator`` 的步骤（停会话、禁用录制、收尾片段、取消后处理、释放摄像头、
回填并关闭台账），以及有界等待后销毁窗口。取代 test_app_ui_lifecycle 中的关窗用例。"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from apps.tk import shutdown
from core.live import APP_CLOSE
from core.media import CameraService
from core.recording import SegmentFinalizer, SegmentMeta, SegmentUpdate
from core.recording import steps as recording_steps
from core.workflows import ScorebookRegistry, WorkflowHost
from tests.tk_app_doubles import (  # noqa: F401
    FakeLive,
    FinalizerDouble,
    SyncPool,
    attach_queued_close,
    bind_recording,
    build_app,
    frame,
    root,
)


def _closing_app(root, finalizer=None, *, live=None, pool=None):
    app = build_app(root, pool=pool, live=live or FakeLive())
    bind_recording(app, finalizer or FinalizerDouble(), plan_for=lambda _segment: None)
    return app


def _single_recording_app(root, tmp_path: Path, *, live=None):
    """单摄会话中、已开录并写入一帧；finalizer 是真实的。"""
    app = build_app(root, live=live or FakeLive("running"))
    bind_recording(app, SegmentFinalizer())
    app.flags.recording_ready = True
    service = app.recording.service
    service.begin_session({"front": (4, 3)}, 30.0)
    assert service.start(SegmentMeta(tmp_path))
    service.write({"front": frame()})
    return app


def _dual_app(root, tmp_path: Path):
    app = build_app(root, live=FakeLive("running"))
    finalizer = FinalizerDouble()
    bind_recording(app, finalizer)
    app.flags.dual_active = app.flags.recording_ready = True
    service = app.recording.service
    service.begin_session({"front": (4, 3), "side": (4, 3)}, 30.0)
    assert service.start(SegmentMeta(tmp_path))
    service.write({"front": frame(), "side": frame()})
    return app, finalizer


def _blocking_transcode(monkeypatch):
    entered, finish = threading.Event(), threading.Event()

    def transcode(path: Path, stop_evt=None) -> Path:
        entered.set()
        assert finish.wait(timeout=2.0)
        return Path(path)

    monkeypatch.setattr(recording_steps, "transcode_to_h264", transcode)
    return entered, finish


def test_close_permanently_closes_camera_pool(root):
    pool = SyncPool()
    app = _closing_app(root, pool=pool)
    attach_queued_close(app)
    app.close()
    app.shutdown.join_release(timeout=1.0)
    assert not app.shutdown.releasing
    assert pool.close_calls == 1


def test_close_poll_window_disables_recording_controls_and_rejects_toggle(root):
    app = _closing_app(root)
    release_prepare = threading.Event()
    app.cameras.service = CameraService(pool=SimpleNamespace(
        release_all=lambda: None, close=lambda: release_prepare.wait(2.0), wait_released=lambda _timeout: True,
    ))
    queued = attach_queued_close(app)
    panel = app.window.recording
    try:
        app.close()
        assert app.shutdown.releasing
        assert queued.after_calls[0][0] == shutdown.POLL_MS
        controls = (panel.record_btn, panel.record_stop_btn, panel.auto_compare_scale, panel.record_skeleton_check,
                    panel.record_dir_entry, panel.record_dir_btn)
        assert all(str(control.cget("state")) == "disabled" for control in controls)
        app.recording.toggle()
        assert app.recording.service.snapshot().segment is None
    finally:
        release_prepare.set()
        app.shutdown.join_release(2.0)
    assert not app.shutdown.releasing


def test_close_waits_for_single_camera_transcode_without_blocking_tk_loop(root, monkeypatch, tmp_path):
    entered, finish = _blocking_transcode(monkeypatch)
    app = _single_recording_app(root, tmp_path)
    app.recording.end()
    assert entered.wait(timeout=1.0)
    queued = attach_queued_close(app)
    app.close()
    assert app.session.live.stops == [APP_CLOSE]
    assert queued.destroy_calls == 0
    assert queued.after_calls[0][0] == shutdown.POLL_MS
    # 关窗的 cancel_all 不取消 finish_on_close 的单摄转码
    finish.set()
    assert app.recording.service.finalizer.wait_idle(2.0)
    app.shutdown.join_release(1.0)
    _delay, poll = queued.after_calls.pop(0)
    poll()
    assert queued.destroy_calls == 1


def test_close_destroys_window_after_bounded_deadline(root, monkeypatch):
    finalizer = FinalizerDouble()
    finalizer.busy = True
    app = _closing_app(root, finalizer, live=FakeLive("running", ends_on_stop=False))   # 会话线程一直不结束
    clock = [0.0]
    monkeypatch.setattr(shutdown.time, "monotonic", lambda: clock[0])
    queued = attach_queued_close(app)
    app.close()
    app.shutdown.join_release(1.0)
    assert queued.destroy_calls == 0
    assert app.session.live.waits == [shutdown.JOIN_SLICE_S, 0]
    clock[0] = shutdown.JOIN_TIMEOUT_S + 0.1
    _delay, poll = queued.after_calls.pop(0)
    poll()
    assert queued.destroy_calls == 1


def test_close_waits_for_transcode_submitted_by_capture_finally(root, monkeypatch, tmp_path):
    entered, finish = _blocking_transcode(monkeypatch)
    live = FakeLive("running", ends_on_stop=False)
    app = _single_recording_app(root, tmp_path, live=live)

    def session_finishes() -> None:
        # 会话线程收尾（FrameSink.close_session）在关窗等待期间才提交单摄片段
        if live.state != "idle":
            live.state = "idle"
            app.recording.close_session()

    live.on_wait = session_finishes
    queued = attach_queued_close(app)
    app.close()
    # 采集线程 finally 在关窗后才提交单摄片段：finish_on_close 仍被接受，poll 必须等它
    assert entered.wait(timeout=1.0)
    assert queued.destroy_calls == 0
    assert queued.after_calls[0][0] == shutdown.POLL_MS
    finish.set()
    assert app.recording.service.finalizer.wait_idle(2.0)
    app.shutdown.join_release(1.0)
    _delay, poll = queued.after_calls.pop(0)
    poll()
    assert queued.destroy_calls == 1


def test_close_poll_waits_for_finalizer_idle(root, monkeypatch):
    finalizer = FinalizerDouble()
    finalizer.busy = True
    app = _closing_app(root, finalizer)
    monkeypatch.setattr(shutdown.time, "monotonic", lambda: 0.0)
    queued = attach_queued_close(app)
    app.close()
    assert app.shutdown.join_release(1.0)
    assert queued.destroy_calls == 0
    finalizer.busy = False
    _delay, poll = queued.after_calls.pop(0)
    poll()
    assert queued.destroy_calls == 1


def test_close_closes_scorebooks_after_deadline_even_if_finalizer_busy(root, monkeypatch):
    finalizer = FinalizerDouble()
    finalizer.busy = True
    app = _closing_app(root, finalizer)
    closed: list[bool] = []
    app.scorebooks.close_all = lambda: closed.append(True)
    clock = [0.0]
    monkeypatch.setattr(shutdown.time, "monotonic", lambda: clock[0])
    queued = attach_queued_close(app)
    app.close()
    assert app.shutdown.join_release(1.0)
    assert closed == [] and queued.destroy_calls == 0
    clock[0] = shutdown.JOIN_TIMEOUT_S + 1.0
    _delay, poll = queued.after_calls.pop(0)
    poll()
    assert closed == [True]
    assert queued.destroy_calls == 1


def test_close_flush_backfills_scorebook_updates_posted_during_close(root, monkeypatch):
    finalizer = FinalizerDouble()
    finalizer.busy = True
    app = _closing_app(root, finalizer)
    applied: list[str] = []
    # 关窗时已移交的台账仍需回填
    app.scorebooks = app.workflows.scorebooks = SimpleNamespace(
        apply=lambda update: applied.append(update.segment_id), has_detached=True, has_processing=False,
        close_all=lambda: None,
    )
    monkeypatch.setattr(shutdown.time, "monotonic", lambda: 0.0)
    queued = attach_queued_close(app)
    app.close()
    app.shutdown.join_release(1.0)
    app.recording.post_update(SegmentUpdate("record_late", "completed", "done", combined_percent=80))
    finalizer.busy = False
    _delay, poll = queued.after_calls.pop(0)
    poll()
    assert applied == ["record_late"]
    assert queued.destroy_calls == 1


def test_close_finalizes_current_dual_segment_then_cancels_postprocessor(root, tmp_path):
    app, finalizer = _dual_app(root, tmp_path)
    attach_queued_close(app)
    app.close()
    app.shutdown.join_release(1.0)
    assert app.flags.closing is True
    assert app.session.live.stops == [APP_CLOSE]
    assert len(finalizer.jobs) == 1
    assert finalizer.cancel_calls == 1


def test_close_returns_while_inflight_finalize_finishes_before_cancelling(root, tmp_path):
    app, finalizer = _dual_app(root, tmp_path)
    attach_queued_close(app)
    submit_started, release_submit = threading.Event(), threading.Event()
    original_submit = finalizer.submit
    errors: list[BaseException] = []

    def submit(segment, plan) -> bool:
        accepted = original_submit(segment, plan)
        submit_started.set()
        assert release_submit.wait(2.0)
        return accepted

    finalizer.submit = submit

    def run() -> None:
        try:
            app.recording.close_session()
        except BaseException as exc:  # noqa: BLE001 - thread failures must reach test
            errors.append(exc)

    finalize_thread = threading.Thread(target=run)
    finalize_thread.start()
    assert submit_started.wait(1.0)
    app.close()
    # 结束+提交与 cancel_all 在同一把锁里串行：提交返回前不取消
    assert finalizer.cancel_calls == 0
    assert app.shutdown.deadline is not None
    assert app.shutdown.releasing
    release_submit.set()
    finalize_thread.join(2.0)
    app.shutdown.join_release(2.0)
    assert not finalize_thread.is_alive() and not app.shutdown.releasing
    assert errors == []
    assert len(finalizer.jobs) == 1
    assert finalizer.cancel_calls == 1


def test_close_deadline_is_set_before_blocking_cancel_work(root, tmp_path):
    app, finalizer = _dual_app(root, tmp_path)
    attach_queued_close(app)
    cancel_started, release_cancel = threading.Event(), threading.Event()
    original_cancel = finalizer.cancel_all

    def cancel_all() -> None:
        cancel_started.set()
        assert release_cancel.wait(2.0)
        original_cancel()

    finalizer.cancel_all = cancel_all
    app.close()
    assert app.shutdown.deadline is not None
    assert cancel_started.wait(1.0)
    assert app.shutdown.releasing
    release_cancel.set()
    app.shutdown.join_release(2.0)
    assert not app.shutdown.releasing


def test_close_extra_budget_covers_exam_backfill():
    from apps.tk.app import close_extra_budget_s

    registry = SimpleNamespace(has_detached=False, has_processing=False)
    assert close_extra_budget_s(1.5, registry) == 1.5
    registry.has_detached = True
    assert close_extra_budget_s(1.5, registry) == 8.0
    registry.has_processing = True
    assert close_extra_budget_s(1.5, registry) == 12.0


def test_real_window_close_destroys_root(root):
    app = _closing_app(root)
    app.close()
    deadline = time.monotonic() + 5
    while not app.ui.destroyed and time.monotonic() < deadline:
        root.update()
        time.sleep(0.01)
    assert app.ui.destroyed


def test_app_close_registers_open_panel_scorebook_for_flush_close(tmp_path):
    """主窗关闭时，仍打开的考试窗口成绩簿必须移交登记表，并在最后 flush/close。"""
    from apps.tk.controllers.exam import ExamController
    from core.workflows import ExamCandidate, ExamScorebook, WorkflowPorts

    flushes: list[str] = []
    closes: list[str] = []

    class _Book(ExamScorebook):
        def flush(self, *, timeout: float = 5.0) -> None:  # type: ignore[override]
            flushes.append("flush")

        def close(self) -> None:  # type: ignore[override]
            closes.append("close")

    book = _Book(path=tmp_path / "scores.xlsx")
    book.load_candidates([ExamCandidate(1, "S1", "甲")])
    book.update_row(book.rows[0], status="processing")
    registry = ScorebookRegistry()
    modes: list[bool] = []
    announcer_closed: list[str] = []
    host = SimpleNamespace(
        exam_scorebooks=registry, workflows=WorkflowHost(registry), set_exam_mode=modes.append,
        lock_manual_record=lambda _locked: None,
        exam_ports=lambda *, ui, announcer: WorkflowPorts(
            recorder=SimpleNamespace(), occupancy=SimpleNamespace(arm=lambda _armed: None),
            announcer=announcer, ui=ui,
        ),
        exam_closed=lambda c: setattr(host, "exam_panel", None) if host.exam_panel is c else None,
    )
    view = SimpleNamespace(status="", set_roi=lambda _roi: None, set_phase=lambda _t: None,
                           set_present=lambda _t: None, is_open=lambda: True, destroy=lambda: None)
    view.set_status = lambda text: setattr(view, "status", text)
    panel = ExamController(host, view, prefs={}, restore=False,
                           announcer=SimpleNamespace(close=lambda: announcer_closed.append("close")))
    panel.session.phase = "completed"
    panel.scorebook = book
    host.exam_panel = panel

    panel.prepare_for_app_close()

    assert announcer_closed == ["close"], "主窗关闭路径必须停 exam-announcer"
    assert panel.scorebook is None
    assert registry.has_detached and registry.has_processing
    assert host.exam_panel is None
    assert host.workflows.active is None
    assert modes == [False]
    assert view.status == "主会话已停止，考试已中止"
    assert closes == []
    registry.close_all()
    assert closes == ["close"]
    # 面板 flush、移交登记表时 flush、最后关闭前 flush
    assert flushes == ["flush", "flush", "flush"]
