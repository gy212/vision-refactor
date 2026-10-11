# -*- coding: utf-8 -*-
"""Tkinter 摄像头预打开、识别状态与转码关窗生命周期测试（无需真实 Tk/camera）。"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from apps import app_ui
from apps.camera_warmup import CameraWarmupError
from core.recording import (
    ProcessingPlan,
    RecordingService,
    SegmentFinalizer,
    SegmentMeta,
    SegmentRecorder,
    SegmentUpdate,
)
from core.recording import steps as recording_steps
from core.workflows import ScorebookRegistry


class _Var:
    def __init__(self, value=None) -> None:
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


class _Widget:
    def __init__(self) -> None:
        self.config: dict[str, object] = {}

    def configure(self, **kwargs) -> None:
        self.config.update(kwargs)


class _Cap:
    def __init__(self, *, opened: bool = True) -> None:
        self.opened = opened
        self.release_calls = 0

    def isOpened(self) -> bool:
        return self.opened

    def release(self) -> None:
        self.release_calls += 1


class _WarmCapture(_Cap):
    def __init__(self, value: int) -> None:
        super().__init__()
        self._condition = threading.Condition()
        self._frame = np.full((2, 3, 3), value, dtype=np.uint8)
        self._sent = False
        self._interrupted = False

    def read(self):
        with self._condition:
            if not self._sent:
                self._sent = True
                return True, self._frame.copy()
            while self.release_calls == 0 and not self._interrupted:
                self._condition.wait()
            return False, None

    def interrupt(self) -> None:
        with self._condition:
            self._interrupted = True
            self._condition.notify_all()

    def release(self) -> None:
        with self._condition:
            super().release()
            self._condition.notify_all()


class _QueuedRoot:
    def __init__(self) -> None:
        self.after_calls: list[tuple[int, object]] = []
        self.destroy_calls = 0

    def after(self, delay: int, callback) -> None:
        self.after_calls.append((delay, callback))

    def destroy(self) -> None:
        self.destroy_calls += 1


class _SyncPool:
    def __init__(self) -> None:
        self.sync_calls: list[tuple[object, object]] = []
        self.release_all_calls = 0
        self.close_calls = 0
        self.cancel_calls: list[tuple[str, int | None]] = []

    def sync(self, primary, secondary) -> None:
        self.sync_calls.append((primary, secondary))

    def release_all(self) -> None:
        self.release_all_calls += 1

    def cancel(self, role: str, *, generation: int | None = None) -> None:
        self.cancel_calls.append((role, generation))

    def close(self) -> None:
        self.close_calls += 1

    def wait_released(self, _timeout: float) -> bool:
        return True


def _camera_app(index: int = 0):
    app = object.__new__(app_ui.App)
    app._worker = None
    app._closing = False
    app._student_practice_active = False
    app._source_state = app_ui.InputSourceState()
    app._source_state.select_camera(index)
    return app


def _warmup_app(*, second_label: str = "摄像头 1"):
    app = _camera_app()
    app._camera_entries = [
        app_ui.CameraEntry(label="摄像头 0", index=0),
        app_ui.CameraEntry(label="摄像头 1", index=1),
    ]
    app.camera_choice_var_2 = _Var(second_label)
    app._camera_warmup_pool = _SyncPool()
    app._post_status = lambda _text: None
    return app


def _closing_app(finalizer=None):
    app = _camera_app()
    app._recording = RecordingService(
        finalizer or _FinalizerDouble(), plan_for=lambda _segment: None
    )
    app.exam_scorebooks = ScorebookRegistry()
    app._closing = False
    app._close_deadline = None
    app._stop_evt = threading.Event()
    app.root = _QueuedRoot()
    return app


def _enumeration_app():
    applied: list[tuple[list[object], bool]] = []
    app = SimpleNamespace(
        _enum_busy=threading.Event(),
        _closing=False,
        _camera_enum_result_queue=Queue(maxsize=1),
        _set_refresh_enabled=lambda: None,
        camera_combo=_Widget(),
        camera_choice_var=_Var(""),
        _apply_camera_entries=lambda entries, ok: applied.append((entries, ok)),
    )
    return app, applied


def _camera_selection_app(
    *, preferred_primary: int | None, preferred_secondary: int | None
):
    app = object.__new__(app_ui.App)
    app._preferred_primary_camera_index = preferred_primary
    app._preferred_secondary_camera_index = preferred_secondary
    app._camera_entries = []
    app._enum_busy = threading.Event()
    app._enum_busy.set()
    app._worker = None
    app._closing = False
    app.camera_combo = _Widget()
    app.camera_combo_2 = _Widget()
    app.camera_choice_var = _Var("")
    app.camera_choice_var_2 = _Var(app_ui.NO_SECOND_CAMERA)
    app.source_var = _Var("")
    app.source_hint_var = _Var("")
    app.status_var = _Var("")
    app._source_state = app_ui.InputSourceState()
    app._sync_camera_warmup = lambda: None
    app._set_refresh_enabled = lambda: None
    return app


def test_single_camera_selection_declares_primary_spec_only():
    app = _warmup_app(second_label=app_ui.NO_SECOND_CAMERA)

    app_ui.App._sync_camera_warmup(app)

    assert app._camera_warmup_pool.sync_calls == [
        (app_ui.CaptureSpec(0, 1280, 720), None)
    ]


def test_two_camera_selection_declares_both_specs():
    app = _warmup_app()

    app_ui.App._sync_camera_warmup(app)

    assert app._camera_warmup_pool.sync_calls == [
        (app_ui.CaptureSpec(0, 1280, 720), app_ui.CaptureSpec(1, 1280, 720))
    ]


def test_same_camera_selection_never_warms_second_role():
    app = _warmup_app(second_label="摄像头 0")

    app_ui.App._sync_camera_warmup(app)

    assert app._camera_warmup_pool.sync_calls == [
        (app_ui.CaptureSpec(0, 1280, 720), None)
    ]


def test_student_mode_declares_1080p_specs_without_reading_flag_later():
    app = _warmup_app()
    app._student_practice_active = True

    app_ui.App._sync_camera_warmup(app)
    app._student_practice_active = False

    assert app._camera_warmup_pool.sync_calls == [
        (app_ui.CaptureSpec(0, 1920, 1080), app_ui.CaptureSpec(1, 1920, 1080))
    ]


def test_running_session_blocks_warmup_changes_unless_allowed():
    app = _warmup_app()
    app._worker = SimpleNamespace(is_alive=lambda: True)

    app_ui.App._sync_camera_warmup(app)
    assert app._camera_warmup_pool.sync_calls == []

    app_ui.App._sync_camera_warmup(app, allow_running=True)
    assert len(app._camera_warmup_pool.sync_calls) == 1


def test_warmup_rejection_releases_and_reports_status():
    app = _warmup_app()
    statuses: list[str] = []
    app._post_status = statuses.append

    def reject(_primary, _secondary):
        raise CameraWarmupError("pool closed")

    app._camera_warmup_pool.sync = reject

    app_ui.App._sync_camera_warmup(app)

    assert app._camera_warmup_pool.release_all_calls == 1
    assert statuses == ["摄像头预热失败：pool closed"]


def test_session_state_freezes_capture_size_for_whole_start():
    app = _warmup_app()
    app._student_practice_active = True
    state = SimpleNamespace(capture_size=app_ui.App._camera_capture_size(app))
    app._student_practice_active = False

    assert app_ui.App._session_spec(state, "1") == app_ui.CaptureSpec(1, 1920, 1080)


def test_process_factory_forwards_frozen_spec(monkeypatch):
    calls = []

    def factory(index, **kwargs):
        calls.append((index, kwargs))
        return _Cap()

    monkeypatch.setattr(app_ui, "open_camera", factory)
    stop = threading.Event()
    app = _camera_app()
    app._student_practice_active = True  # 工厂不再读取模式标志

    app_ui.App._open_camera_process(app, 0, width=1280, height=720, stop_event=stop)

    assert calls == [(0, {"width": 1280, "height": 720, "stop_event": stop})]


def test_failed_session_cancels_only_its_observed_generations():
    app = _warmup_app()

    app_ui.App._cancel_session_warmups(app, (3, 7))
    app_ui.App._cancel_session_warmups(app, ())

    assert app._camera_warmup_pool.cancel_calls == [
        (app_ui.PRIMARY, 3),
        (app_ui.SECONDARY, 7),
    ]


def test_session_release_failure_is_reported_not_swallowed():
    class _StuckLease:
        def release(self) -> None:
            raise CameraWarmupError("摄像头 0 释放失败，已隔离")

    message = app_ui.App._release_session_camera(object(), _StuckLease())

    assert message is not None and "已隔离" in message


def test_camera_enumeration_result_is_applied_only_from_main_thread_queue(monkeypatch):
    app, applied = _enumeration_app()
    pending: list[object] = []
    entries = [object()]

    class _PendingThread:
        def __init__(self, *, target, daemon) -> None:
            pending.append(target)

        def start(self) -> None:
            pass

    monkeypatch.setattr(app_ui.threading, "Thread", _PendingThread)
    monkeypatch.setattr(app_ui, "enumerate_cameras", lambda: entries)

    app_ui.App._start_enumeration(app)
    pending[0]()

    assert applied == []
    app_ui.App._drain_camera_enum_results(app)
    assert applied == [(entries, True)]


def test_camera_enumeration_restores_persisted_primary_and_secondary() -> None:
    app = _camera_selection_app(preferred_primary=2, preferred_secondary=1)
    entries = [
        app_ui.CameraEntry(label="摄像头 0", index=0),
        app_ui.CameraEntry(label="摄像头 1: Side", index=1),
        app_ui.CameraEntry(label="摄像头 2: Front", index=2),
    ]

    app_ui.App._apply_camera_entries(app, entries, True)

    assert app.camera_choice_var.get() == "摄像头 2: Front"
    assert app.camera_choice_var_2.get() == "摄像头 1: Side"
    assert app.source_var.get() == "2"


def test_missing_preferred_cameras_fallback_without_forgetting_then_restore() -> None:
    app = _camera_selection_app(preferred_primary=2, preferred_secondary=1)

    app_ui.App._apply_camera_entries(
        app,
        [app_ui.CameraEntry(label="摄像头 0", index=0)],
        True,
    )

    assert app.camera_choice_var.get() == "摄像头 0"
    assert app.camera_choice_var_2.get() == app_ui.NO_SECOND_CAMERA
    assert app._preferred_primary_camera_index == 2
    assert app._preferred_secondary_camera_index == 1

    app_ui.App._apply_camera_entries(
        app,
        [
            app_ui.CameraEntry(label="摄像头 0", index=0),
            app_ui.CameraEntry(label="摄像头 1", index=1),
            app_ui.CameraEntry(label="摄像头 2", index=2),
        ],
        True,
    )

    assert app.camera_choice_var.get() == "摄像头 2"
    assert app.camera_choice_var_2.get() == "摄像头 1"


def test_explicit_camera_selections_are_persisted(monkeypatch) -> None:
    app = _camera_selection_app(preferred_primary=0, preferred_secondary=None)
    app._camera_entries = [
        app_ui.CameraEntry(label="摄像头 0", index=0),
        app_ui.CameraEntry(label="摄像头 1", index=1),
    ]
    saved: list[tuple[int | None, int | None]] = []
    monkeypatch.setattr(
        app_ui,
        "save_camera_selection",
        lambda primary, secondary: saved.append((primary, secondary)),
    )

    app.camera_choice_var.set("摄像头 1")
    app_ui.App._on_camera_selected(app)
    app.camera_choice_var_2.set("摄像头 0")
    app_ui.App._on_camera_2_selected(app)
    app.camera_choice_var_2.set(app_ui.NO_SECOND_CAMERA)
    app_ui.App._on_camera_2_selected(app)

    assert saved == [(1, None), (1, 0), (1, None)]
    assert app.source_var.get() == "1"


def test_camera_enumeration_return_after_close_does_not_touch_tk(monkeypatch):
    app, applied = _enumeration_app()
    pending: list[object] = []

    class _PendingThread:
        def __init__(self, *, target, daemon) -> None:
            pending.append(target)

        def start(self) -> None:
            pass

    monkeypatch.setattr(app_ui.threading, "Thread", _PendingThread)
    monkeypatch.setattr(app_ui, "enumerate_cameras", lambda: [object()])

    app_ui.App._start_enumeration(app)
    app._closing = True
    pending[0]()
    app_ui.App._drain_camera_enum_results(app)

    assert applied == []
    assert app._camera_enum_result_queue.empty()


def test_refresh_releases_all_camera_warmups_before_enumeration():
    app = _warmup_app()
    app._enum_busy = threading.Event()
    events: list[str] = []
    app._release_camera_warmups = lambda: events.append("release")
    app._start_enumeration = lambda: events.append("enumerate")

    app_ui.App._refresh_cameras(app)

    assert events == ["release", "enumerate"]


def test_video_worker_releases_camera_warmups_before_opening_file(monkeypatch):
    events: list[str] = []
    video_cap = _Cap(opened=False)
    app = SimpleNamespace(
        _release_camera_warmups=lambda: events.append("release"),
        _post_status=lambda text: events.append(text),
        _post_done=lambda **_kwargs: events.append("done"),
    )
    state = SimpleNamespace(source="input.mp4", source2=None, workers=1)

    monkeypatch.setattr(
        app_ui.cv2,
        "VideoCapture",
        lambda path: (events.append(f"open:{path}"), video_cap)[1],
    )

    app_ui.App._worker_loop(app, state)

    assert events[:2] == ["release", "open:input.mp4"]
    assert events[-1] == "done"
    assert video_cap.release_calls == 1


@pytest.mark.parametrize("rewarm", [True, False])
def test_post_done_rewarms_only_when_requested(rewarm):
    calls: list[bool] = []

    class _ImmediateRoot:
        def after(self, _delay, callback) -> None:
            callback()

    app = SimpleNamespace(
        _closing=False,
        root=_ImmediateRoot(),
        start_btn=_Widget(),
        stop_btn=_Widget(),
        _set_refresh_enabled=lambda: None,
        _set_running_controls=lambda _running: None,
        _sync_camera_warmup=lambda *, allow_running=False: calls.append(allow_running),
    )

    app_ui.App._post_done(app, rewarm=rewarm)

    assert calls == ([True] if rewarm else [])


class _SinglePool:
    def __init__(self, *, wait_error: BaseException | None = None) -> None:
        self.wait_error = wait_error
        self.calls: list[str] = []

    def wait_one(self, spec, *, timeout, stop_event):
        self.calls.append(f"wait:{spec.index}:{spec.width}x{spec.height}")
        if self.wait_error is not None:
            raise self.wait_error
        return SimpleNamespace(generation=5)

    def claim_one(self, spec, *, timeout, expected_generation, stop_event):
        self.calls.append(f"claim:{expected_generation}")
        return _Cap()


def test_camera_open_failure_restores_controls_without_rewarming():
    statuses = []
    completions = []
    pool = _SinglePool(wait_error=TimeoutError("摄像头响应超时"))
    app = SimpleNamespace(
        _camera_warmup_pool=pool,
        _stop_evt=threading.Event(),
        _post_status=statuses.append,
        _post_done=lambda **kwargs: completions.append(kwargs),
    )
    state = SimpleNamespace(
        source="0", source2=None, workers=1, record_skeleton=False,
        capture_size=(1280, 720), session_generation=6,
    )
    app_ui.App._worker_loop(app, state)
    assert pool.calls == ["wait:0:1280x720"]
    assert "摄像头响应超时" in statuses[-1]
    assert completions == [
        {"session_generation": 6, "final_status": statuses[-1], "rewarm": False}
    ]


def test_single_camera_stop_during_open_finishes_session_quietly():
    completions = []
    pool = _SinglePool(wait_error=app_ui.CameraWarmupStopped("stopped"))
    app = SimpleNamespace(
        _camera_warmup_pool=pool,
        _stop_evt=threading.Event(),
        _post_status=lambda _text: None,
        _post_done=lambda **kwargs: completions.append(kwargs),
    )
    state = SimpleNamespace(
        source="0", source2=None, workers=1, record_skeleton=False,
        capture_size=(1920, 1080), session_generation=2,
    )
    app_ui.App._worker_loop(app, state)
    assert pool.calls == ["wait:0:1920x1080"]
    assert completions == [{"session_generation": 2}]


def test_generation_aware_post_done_ignores_stale_and_deduplicates_current():
    events: list[str] = []

    class _ImmediateRoot:
        def after(self, _delay, callback) -> None:
            callback()

    app = SimpleNamespace(
        _closing=False,
        root=_ImmediateRoot(),
        start_btn=_Widget(),
        stop_btn=_Widget(),
        _set_refresh_enabled=lambda: events.append("refresh"),
        _set_running_controls=lambda _running: events.append("controls"),
        _sync_camera_warmup=lambda *, allow_running=False: events.append("warm"),
        _current_session_generation=4,
        _dual_done_generation=0,
        _active_dual_generation=4,
        _dual_recording_ready=True,
        _dual_preview_lock=threading.Lock(),
        _dual_preview_queue=Queue(maxsize=1),
        _dual_metrics_lock=threading.Lock(),
        _dual_first_render_events={4: object()},
        _dual_startup_metrics={4: object()},
        status_var=_Var("就绪"),
    )

    app_ui.App._post_done(app, session_generation=3)
    assert events == []

    app_ui.App._post_done(
        app, session_generation=4, final_status="初始化失败：boom"
    )
    app_ui.App._post_done(app, session_generation=4)

    assert events == ["refresh", "controls", "warm"]
    assert app._current_session_generation == 0
    assert app._dual_done_generation == 4
    assert app._dual_recording_ready is False
    assert app._dual_first_render_events == {}
    assert app._dual_startup_metrics == {}
    assert app.status_var.get() == "初始化失败：boom"


def test_recording_ready_callback_requires_current_live_generation():
    class _ImmediateRoot:
        def after(self, _delay, callback) -> None:
            callback()

    app = SimpleNamespace(
        _closing=False,
        root=_ImmediateRoot(),
        _stop_evt=threading.Event(),
        _current_session_generation=8,
        _active_dual_generation=8,
        _dual_recording_ready=False,
        _dual_preview_lock=threading.Lock(),
        record_btn=_Widget(),
    )

    app_ui.App._post_dual_recording_ready(app, 7)
    assert app._dual_recording_ready is False

    app_ui.App._post_dual_recording_ready(app, 8)
    assert app._dual_recording_ready is True
    assert app.record_btn.config["state"] == "normal"


def test_close_permanently_closes_dual_camera_warmup_pool():
    app = _closing_app()
    pool = _SyncPool()
    app._camera_warmup_pool = pool

    app_ui.App._on_close(app)
    app._close_prepare_worker.join(timeout=1.0)

    assert not app._close_prepare_worker.is_alive()
    assert pool.close_calls == 1


def test_start_resets_stale_match_text(monkeypatch):
    app = object.__new__(app_ui.App)
    app._worker = None
    app._stop_evt = threading.Event()
    app._collect_state = lambda: SimpleNamespace(source2=None)
    app.start_btn = _Widget()
    app.stop_btn = _Widget()
    app.status_var = _Var()
    app.actions_var = _Var()
    app.match_var = _Var("识别到：直拳 (0.91)")
    app.progress_var = _Var()
    app.progress_text_var = _Var()
    app.progress_bar = _Widget()
    app._set_refresh_enabled = lambda: None
    app._set_dual_preview_visible = lambda _visible: None
    app._set_running_controls = lambda _running: None
    app._worker_loop = lambda _state: None

    class _StartOnlyThread:
        def __init__(self, *, target, args, daemon) -> None:
            self.target = target
            self.args = args
            self.daemon = daemon
            self.started = False

        def start(self) -> None:
            self.started = True

        def is_alive(self) -> bool:
            return self.started

    monkeypatch.setattr(app_ui.threading, "Thread", _StartOnlyThread)

    app_ui.App._start(app)

    assert app.match_var.get() == "识别：待机"
    assert app._worker.started is True


def test_exam_preflight_rejects_skeleton_session():
    app = SimpleNamespace(
        _dual_active=True,
        _dual_recording_ready=True,
        _dual_record_skeleton=True,
        record_skeleton_var=_Var(True),
    )
    ok, msg = app_ui.App._exam_preflight(app)
    assert ok is False
    assert "骨架" in msg


def test_dual_layout_update_uses_queue_and_never_touches_tk_after_close():
    applied: list[str] = []
    app = SimpleNamespace(
        _closing=False,
        _dual_layout_queue=Queue(maxsize=1),
        _set_dual_preview_layout=lambda layout: applied.append(layout),
    )

    app_ui.App._post_dual_preview_layout(app, "stacked")
    app_ui.App._post_dual_preview_layout(app, "side_by_side")
    app_ui.App._drain_dual_preview_layout(app)
    assert applied == ["side_by_side"]

    app._closing = True
    app_ui.App._post_dual_preview_layout(app, "stacked")
    app_ui.App._drain_dual_preview_layout(app)
    assert applied == ["side_by_side"]


def test_close_poll_window_disables_recording_controls_and_rejects_toggle():
    app = _closing_app()
    release_prepare = threading.Event()
    controls = {
        name: _Widget()
        for name in (
            "record_btn",
            "record_stop_btn",
            "auto_compare_check",
            "record_skeleton_check",
            "record_dir_entry",
            "record_dir_btn",
        )
    }
    for name, control in controls.items():
        setattr(app, name, control)

    class _UnexpectedTkRead:
        def get(self):
            raise AssertionError("closing toggle must return before reading Tk state")

    app.record_dir_var = _UnexpectedTkRead()
    app._camera_warmup_pool = SimpleNamespace(
        release_all=lambda: None,
        close=lambda: release_prepare.wait(2.0),
        wait_released=lambda _timeout: True,
    )

    try:
        app_ui.App._on_close(app)

        assert app._close_prepare_worker.is_alive()
        assert app.root.after_calls[0][0] == app_ui._CLOSE_POLL_MS
        assert all(control.config["state"] == "disabled" for control in controls.values())
        app_ui.App._on_record_toggle(app)
    finally:
        release_prepare.set()
        app._close_prepare_worker.join(2.0)

    assert not app._close_prepare_worker.is_alive()


# ---------------------------------------------------------------- 录制服务（P4：RecordingService + SegmentFinalizer）


class _FakeWriter:
    def __init__(self) -> None:
        self.frames: list[object] = []
        self.released = False

    def write(self, frame) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


def _writer_factory(failing: frozenset[str] = frozenset()):
    """按视角决定 writer 是否创建失败；单路文件（record_<stamp>.mp4）算 front。"""

    def factory(path, _fps, _size):
        path = Path(path)
        view = "front" if path.stem.startswith("record_") else path.stem
        if view in failing:
            raise RuntimeError(f"{view} writer unavailable")
        return _FakeWriter(), path, "fake"

    return factory


class _FinalizerDouble:
    """只记录提交的片段与计划，不执行步骤。"""

    def __init__(self) -> None:
        self.jobs: list[SimpleNamespace] = []
        self.cancel_calls = 0
        self.busy = False

    def submit(self, segment, plan) -> bool:
        self.jobs.append(SimpleNamespace(segment=segment, plan=plan))
        return True

    def cancel_all(self) -> None:
        self.cancel_calls += 1

    def close_budget_s(self) -> float:
        return 0.0

    @property
    def idle(self) -> bool:
        return not self.busy


def _frame() -> np.ndarray:
    return np.zeros((3, 4, 3), dtype=np.uint8)


def _bind_recording(app, finalizer, *, failing: frozenset[str] = frozenset()) -> None:
    app._recording = RecordingService(
        finalizer,
        plan_for=lambda segment: app_ui.App._recording_plan(app, segment),
        recorder=SegmentRecorder(_writer_factory(failing)),
    )


def _single_recording_app(tmp_path: Path, *, failing: frozenset[str] = frozenset()):
    """单摄会话中、已开录并写入一帧；finalizer 是真实的。"""
    app = _closing_app()
    _bind_recording(app, SegmentFinalizer(), failing=failing)
    app._pending_record_errors = []
    app._record_error_shown = False
    app._dual_active = False
    app.record_btn = _Widget()
    app.recording_status_var = _Var("")
    app._recording.begin_session({"front": (4, 3)}, 30.0)
    assert app._recording.start(SegmentMeta(tmp_path))
    app._recording.write({"front": _frame()})
    return app


def _blocking_transcode(monkeypatch):
    entered = threading.Event()
    finish = threading.Event()
    threads: list[tuple[str, bool]] = []

    def _transcode(path: Path, stop_evt=None) -> Path:
        current = threading.current_thread()
        threads.append((current.name, current.daemon))
        entered.set()
        assert finish.wait(timeout=2.0)
        return Path(path)

    monkeypatch.setattr(recording_steps, "transcode_to_h264", _transcode)
    return entered, finish, threads


def test_single_camera_transcode_runs_on_non_daemon_finalizer_lane(monkeypatch, tmp_path):
    entered, finish, threads = _blocking_transcode(monkeypatch)
    app = _single_recording_app(tmp_path)

    app_ui.App._end_recording_segment(app)

    assert entered.wait(timeout=1.0)
    # V-4：单摄转码进 finalizer 的 io 线；该线程非 daemon，进程退出前跑完
    assert threads == [("finalizer/segment-finalizer/io", False)]
    assert not app._recording.finalizer.idle
    finish.set()
    assert app._recording.finalizer.wait_idle(2.0)


def test_close_waits_for_single_camera_transcode_without_blocking_tk_loop(monkeypatch, tmp_path):
    entered, finish, _threads = _blocking_transcode(monkeypatch)
    app = _single_recording_app(tmp_path)
    app_ui.App._end_recording_segment(app)
    assert entered.wait(timeout=1.0)

    app_ui.App._on_close(app)

    assert app._stop_evt.is_set()
    assert app.root.destroy_calls == 0
    assert app.root.after_calls[0][0] == app_ui._CLOSE_POLL_MS

    # 关窗的 cancel_all 不取消 finish_on_close 的单摄转码
    finish.set()
    assert app._recording.finalizer.wait_idle(2.0)
    app._close_prepare_worker.join(1.0)
    _delay, poll = app.root.after_calls.pop(0)
    poll()
    assert app.root.destroy_calls == 1


def test_close_destroys_window_after_bounded_deadline(monkeypatch):
    finalizer = _FinalizerDouble()
    finalizer.busy = True
    app = _closing_app(finalizer)
    clock = [0.0]

    class _NeverEndingWorker:
        def __init__(self) -> None:
            self.join_timeouts: list[float] = []

        def is_alive(self) -> bool:
            return True

        def join(self, timeout=None) -> None:
            self.join_timeouts.append(timeout)

    worker = _NeverEndingWorker()
    app._worker = worker
    monkeypatch.setattr(app_ui.time, "monotonic", lambda: clock[0])

    app_ui.App._on_close(app)
    app._close_prepare_worker.join(1.0)
    assert app.root.destroy_calls == 0
    assert worker.join_timeouts == [app_ui._CLOSE_JOIN_SLICE_S]

    clock[0] = app_ui._CLOSE_JOIN_TIMEOUT_S + 0.1
    _delay, poll = app.root.after_calls.pop(0)
    poll()
    assert app.root.destroy_calls == 1


def test_close_waits_for_transcode_submitted_by_capture_finally(monkeypatch, tmp_path):
    entered, finish, _threads = _blocking_transcode(monkeypatch)
    app = _single_recording_app(tmp_path)

    class _CaptureFinishingWorker:
        def __init__(self) -> None:
            self.alive = True

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout=None) -> None:
            self.alive = False
            app_ui.App._close_recording_session(app)

    app._worker = _CaptureFinishingWorker()
    app_ui.App._on_close(app)

    # 采集线程 finally 在关窗后才提交单摄片段：finish_on_close 仍被接受，poll 必须等它
    assert entered.wait(timeout=1.0)
    assert app.root.destroy_calls == 0
    assert app.root.after_calls[0][0] == app_ui._CLOSE_POLL_MS

    finish.set()
    assert app._recording.finalizer.wait_idle(2.0)
    app._close_prepare_worker.join(1.0)
    _delay, poll = app.root.after_calls.pop(0)
    poll()
    assert app.root.destroy_calls == 1


def test_close_poll_waits_for_finalizer_idle(monkeypatch):
    finalizer = _FinalizerDouble()
    finalizer.busy = True
    app = _closing_app(finalizer)
    app._close_deadline = app_ui._CLOSE_JOIN_TIMEOUT_S
    monkeypatch.setattr(app_ui.time, "monotonic", lambda: 0.0)

    app_ui.App._poll_close_workers(app)

    assert app.root.destroy_calls == 0
    assert app.root.after_calls[0][0] == app_ui._CLOSE_POLL_MS

    finalizer.busy = False
    _delay, poll = app.root.after_calls.pop(0)
    poll()
    assert app.root.destroy_calls == 1


def test_close_poll_closes_scorebooks_after_deadline_even_if_finalizer_busy(monkeypatch):
    finalizer = _FinalizerDouble()
    finalizer.busy = True
    app = _closing_app(finalizer)
    closed: list[bool] = []
    app.exam_scorebooks = SimpleNamespace(close_all=lambda: closed.append(True))
    app._close_deadline = 1.0
    monkeypatch.setattr(app_ui.time, "monotonic", lambda: 2.0)

    app_ui.App._poll_close_workers(app)

    assert closed == [True]
    assert app.root.destroy_calls == 1


def test_side_writer_failure_submits_pair_once_without_modal_and_single_session_unaffected(
    monkeypatch, tmp_path
):
    finalizer = _FinalizerDouble()
    app = SimpleNamespace(
        _pending_record_errors=[],
        _record_error_shown=False,
        _sync_record_stop_enabled=lambda _state: None,
        recording_status_var=_Var("录制中"),
        record_btn=_Widget(),
    )
    app._recording = RecordingService(
        finalizer,
        plan_for=lambda _segment: ProcessingPlan(()),
        recorder=SegmentRecorder(_writer_factory(frozenset({"side"}))),
    )
    app._recording.begin_session({"front": (4, 3), "side": (4, 3)}, 30.0)
    assert app._recording.start(SegmentMeta(tmp_path))
    app._recording.write({"front": _frame(), "side": _frame()})
    views = app._recording.snapshot().views
    assert (views["front"].state, views["side"].state) == ("recording", "idle")

    errors: list[str] = []
    monkeypatch.setattr(
        app_ui.messagebox, "showerror", lambda _title, message: errors.append(message)
    )

    app_ui.App._refresh_recording_status(app)

    # 双摄的写失败由结果 JSON 与比对状态栏承载，不弹框；两路一起结束并只提交一次
    assert app._recording.state == "idle"
    assert errors == []
    assert len(finalizer.jobs) == 1
    assert finalizer.jobs[0].segment.errors == {"side": "side writer unavailable"}
    app_ui.App._refresh_recording_status(app)
    assert app._record_error_shown is False
    assert len(finalizer.jobs) == 1

    # 双摄会话结束后只开主路，旧侧路错误不得误停或污染单摄录制
    app._recording.close_session()
    app._recording.begin_session({"front": (4, 3)}, 30.0)
    assert app._recording.start(SegmentMeta(tmp_path))

    app_ui.App._refresh_recording_status(app)

    assert app._recording.state == "recording"
    assert errors == []
    assert app.recording_status_var.get().startswith("录制中：")


def test_close_recording_session_keeps_single_camera_error_for_next_tk_tick(
    monkeypatch, tmp_path
):
    app = _single_recording_app(tmp_path, failing=frozenset({"front"}))
    errors: list[str] = []
    monkeypatch.setattr(
        app_ui.messagebox, "showerror", lambda _title, message: errors.append(message)
    )

    app_ui.App._close_recording_session(app)
    assert app._dual_active is False
    assert app._recording.snapshot().errors == {}

    app_ui.App._refresh_recording_status(app)

    assert errors == ["录制发生错误：\n第一路：front writer unavailable"]
    assert app._pending_record_errors == []


def _dual_submission_app(
    tmp_path: Path, *, write_frames: bool = True, failing: frozenset[str] = frozenset()
):
    finalizer = _FinalizerDouble()
    app = SimpleNamespace(
        _closing=False,
        _stop_evt=None,
        _exam_manual_locked=False,
        _exam_pending_row=None,
        _exam_run_id=None,
        _student_practice_active=False,
        _pending_record_errors=[],
        _record_error_shown=False,
        _dual_active=True,
        _dual_recording_ready=True,
        _dual_record_skeleton=False,
        _dual_auto_compare=True,
        _record_base_dir=tmp_path,
        _latest_compare_segment_id=None,
        exam_scorebooks=ScorebookRegistry(),
        _sync_record_stop_enabled=lambda _state: None,
        record_btn=_Widget(),
    )
    # 计划构建读取模板池；这些测试只验证提交机制，返回空池即可。
    app._current_template_lists = lambda: ([], [])
    _bind_recording(app, finalizer, failing=failing)
    app._recording.begin_session({"front": (4, 3), "side": (4, 3)}, 30.0)
    segment = app._recording.start(SegmentMeta(tmp_path))
    assert segment
    if write_frames:
        app._recording.write({"front": _frame(), "side": _frame()})
    return app, finalizer, segment.directory


def test_dual_record_stop_submits_exactly_once_and_allows_next_segment(tmp_path):
    app, finalizer, segment_dir = _dual_submission_app(tmp_path)

    app_ui.App._on_record_stop(app)

    assert app._recording.state == "idle"
    assert len(finalizer.jobs) == 1
    job = finalizer.jobs[0]
    assert job.segment.segment_id == segment_dir.name
    assert job.segment.source("front") == segment_dir / "front.mp4"
    assert job.segment.source("side") == segment_dir / "side.mp4"
    assert [job.segment.views[v].frames for v in ("front", "side")] == [1, 1]
    assert job.plan.result.record_skeleton is False
    assert job.plan.result.auto_compare is True

    app_ui.App._close_recording_session(app)
    assert len(finalizer.jobs) == 1
    assert app._dual_active is False


def test_dual_record_stop_submits_auto_compare_disabled_when_only_recording(tmp_path):
    app, finalizer, segment_dir = _dual_submission_app(tmp_path)
    app._dual_auto_compare = False

    app_ui.App._on_record_stop(app)

    assert len(finalizer.jobs) == 1
    job = finalizer.jobs[0]
    assert job.segment.segment_id == segment_dir.name
    assert job.plan.result.auto_compare is False


def test_record_toggle_starts_dual_segment_under_selected_base_dir(tmp_path, monkeypatch):
    app, finalizer, _segment_dir = _dual_submission_app(tmp_path / "old")
    app_ui.App._on_record_stop(app)
    new_base = tmp_path / "new"
    app.record_dir_var = _Var(str(new_base))
    saved: list[Path] = []
    monkeypatch.setattr(app_ui, "save_record_dir", lambda path: saved.append(Path(path)))

    app_ui.App._on_record_toggle(app)

    snapshot = app._recording.snapshot()
    assert snapshot.state == "recording"
    assert snapshot.segment.directory.parent.parent == new_base
    assert app._record_base_dir == new_base
    assert saved == [new_base]
    assert len(finalizer.jobs) == 1


def test_app_close_registers_open_panel_scorebook_for_flush_close(tmp_path):
    """主窗关闭时，仍打开的考试面板成绩簿必须移交登记表，并在最后 flush/close。"""
    from apps.exam_panel import ExamPanel
    from core.workflows import ExamCandidate, ExamScorebook

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
    app = SimpleNamespace(exam_scorebooks=registry, _exam_panel="sentinel")
    announcer_closed: list[str] = []
    panel = SimpleNamespace(
        app=app,
        scorebook=book,
        session=SimpleNamespace(phase="completed"),
        _tick_id="x",
        win=SimpleNamespace(after_cancel=lambda _i: None, destroy=lambda: None),
        phase_var=_Var(""),
        status_var=_Var(""),
        on_session_stop=lambda: None,
        _announcer=SimpleNamespace(close=lambda: announcer_closed.append("close")),
    )

    ExamPanel.prepare_for_app_close(panel)  # type: ignore[arg-type]

    assert announcer_closed == ["close"], "主窗关闭路径必须停 exam-announcer"
    assert panel.scorebook is None
    assert registry.has_detached
    assert registry.has_processing
    assert getattr(app, "_exam_panel", None) is None
    assert closes == []

    registry.close_all()
    assert closes == ["close"]
    # 面板 flush、移交登记表时 flush、最后关闭前 flush
    assert flushes == ["flush", "flush", "flush"]


def test_compare_update_ignores_old_segment_and_formats_latest_score():
    exam_updates: list = []
    app = SimpleNamespace(
        _closing=False,
        _latest_compare_segment_id="record_new",
        _compare_update_queue=Queue(),
        root=_QueuedRoot(),
        compare_segment_var=_Var("片段：record_new"),
        compare_status_var=_Var("自动比对：排队中"),
        compare_score_var=_Var("unchanged"),
        compare_error_var=_Var(""),
        exam_scorebooks=SimpleNamespace(
            apply=lambda update: exam_updates.append(update.segment_id),
            has_detached=False,
        ),
    )

    old = SegmentUpdate(
        segment_id="record_old",
        status="completed",
        message="done",
        front_score=0.1,
        side_score=0.2,
        combined_percent=15,
    )
    app_ui.App._post_recording_compare_update(app, old)
    assert app.root.after_calls == []
    app_ui.App._drain_recording_compare_updates(app)
    # 主界面比对条只跟最新片段；考试台账仍须收到旧片段终态
    assert app.compare_score_var.get() == "unchanged"
    assert exam_updates == ["record_old"]

    latest = SegmentUpdate(
        segment_id="record_new",
        status="completed",
        message="done",
        front_score=0.823,
        side_score=0.891,
        combined_percent=86,
    )
    app_ui.App._post_recording_compare_update(app, latest)
    assert app.root.after_calls == []
    app_ui.App._drain_recording_compare_updates(app)

    assert app.compare_status_var.get() == "自动比对：已完成"
    assert app.compare_score_var.get() == "正面：82.3%　侧面：89.1%　综合：86%"
    assert app.compare_error_var.get() == ""
    assert exam_updates == ["record_old", "record_new"]

    app._closing = True
    app_ui.App._post_recording_compare_update(app, latest)
    app_ui.App._drain_recording_compare_updates(app)
    assert app.root.after_calls == []
    assert exam_updates == ["record_old", "record_new"]


def test_compare_update_formats_failed_skipped_and_cancelled_states():
    app = SimpleNamespace(
        _closing=False,
        _latest_compare_segment_id="record_terminal",
        _compare_update_queue=Queue(),
        compare_segment_var=_Var("片段：record_terminal"),
        compare_status_var=_Var("自动比对：排队中"),
        compare_score_var=_Var("stale score"),
        compare_error_var=_Var(""),
        exam_scorebooks=ScorebookRegistry(),
    )
    cases = (
        ("failed", "失败", "video_unreadable", "录像无法读取"),
        ("skipped", "已跳过", "annotated_recording", "带骨架录像未自动比对"),
        ("cancelled", "已取消", "app_closing", "应用关闭"),
    )

    for status, label, code, message in cases:
        app_ui.App._post_recording_compare_update(
            app, SegmentUpdate("record_terminal", status, message, error_code=code)
        )
        app_ui.App._drain_recording_compare_updates(app)
        assert app.compare_status_var.get() == f"自动比对：{label}"
        assert app.compare_score_var.get() == "正面：-　侧面：-　综合：-"
        assert app.compare_error_var.get() == f"{code}：{message}"


def test_compare_update_already_queued_is_discarded_after_close_starts():
    app = SimpleNamespace(
        _closing=False,
        _latest_compare_segment_id="record_latest",
        _compare_update_queue=Queue(),
        compare_segment_var=_Var("片段：record_latest"),
        compare_status_var=_Var("自动比对：排队中"),
        compare_score_var=_Var("unchanged"),
        compare_error_var=_Var(""),
        exam_scorebooks=ScorebookRegistry(),
    )
    app_ui.App._post_recording_compare_update(
        app,
        SegmentUpdate(
            "record_latest",
            "completed",
            "done",
            front_score=0.9,
            side_score=0.9,
            combined_percent=90,
        ),
    )

    app._closing = True
    app_ui.App._drain_recording_compare_updates(app)

    assert app.compare_score_var.get() == "unchanged"


def test_main_stop_finally_submits_current_dual_segment_once(tmp_path):
    app, finalizer, _segment_dir = _dual_submission_app(tmp_path)
    app._stop_evt = threading.Event()
    app.stop_btn = _Widget()
    app.status_var = _Var()

    app_ui.App._stop(app)
    app_ui.App._close_recording_session(app)
    app_ui.App._close_recording_session(app)

    assert app._stop_evt.is_set()
    assert app.status_var.get() == "正在停止…"
    assert len(finalizer.jobs) == 1


def test_zero_frame_dual_segment_is_submitted_from_explicit_and_main_stop(tmp_path):
    for mode in ("explicit", "main_stop"):
        app, finalizer, _segment_dir = _dual_submission_app(
            tmp_path / mode, write_frames=False
        )
        if mode == "explicit":
            app_ui.App._on_record_stop(app)
        else:
            app._stop_evt = threading.Event()
            app.stop_btn = _Widget()
            app.status_var = _Var()
            app_ui.App._stop(app)
            app_ui.App._close_recording_session(app)

        assert len(finalizer.jobs) == 1
        segment = finalizer.jobs[0].segment
        assert [segment.views[v].frames for v in ("front", "side")] == [0, 0]


def test_second_dual_segment_can_start_before_first_result_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(app_ui, "save_record_dir", lambda _path: None)
    app, finalizer, _first_dir = _dual_submission_app(tmp_path)
    app_ui.App._on_record_stop(app)
    first = finalizer.jobs[0].segment

    app_ui.App._on_record_toggle(app)
    app._recording.write({"front": _frame(), "side": _frame()})
    app_ui.App._on_record_stop(app)

    assert len(finalizer.jobs) == 2
    second = finalizer.jobs[1].segment
    assert second.segment_id != first.segment_id
    assert second.source("front") == second.directory / "front.mp4"
    assert second.source("side") == second.directory / "side.mp4"
    assert [second.views[v].frames for v in ("front", "side")] == [1, 1]


def test_dual_submit_does_not_block_capture_writes(tmp_path):
    app, finalizer, _segment_dir = _dual_submission_app(tmp_path)
    original_submit = finalizer.submit
    write_returned: list[bool] = []

    def submit(segment, plan) -> bool:
        # 提交在服务锁内，但采集线程写帧不经这把锁，不被提交阻塞
        writer = threading.Thread(
            target=lambda: app._recording.write({"front": _frame(), "side": _frame()})
        )
        writer.start()
        writer.join(1.0)
        write_returned.append(not writer.is_alive())
        return original_submit(segment, plan)

    finalizer.submit = submit
    app_ui.App._on_record_stop(app)

    assert write_returned == [True]


def test_writer_error_submits_formal_job_once_without_modal(tmp_path, monkeypatch):
    app, finalizer, _segment_dir = _dual_submission_app(
        tmp_path, failing=frozenset({"side"})
    )
    app.recording_status_var = _Var("录制中")
    errors: list[str] = []
    monkeypatch.setattr(
        app_ui.messagebox, "showerror", lambda _title, message: errors.append(message)
    )

    app_ui.App._refresh_recording_status(app)
    app_ui.App._close_recording_session(app)

    assert errors == []
    assert len(finalizer.jobs) == 1
    assert finalizer.jobs[0].segment.errors == {"side": "side writer unavailable"}


def test_toggle_after_primary_writer_failure_finalizes_old_pair_before_new_segment(tmp_path):
    app, finalizer, segment_dir = _dual_submission_app(
        tmp_path, failing=frozenset({"front"})
    )

    app_ui.App._on_record_toggle(app)

    assert app._recording.state == "idle"
    assert app._recording.snapshot().segment is None
    assert len(finalizer.jobs) == 1
    segment = finalizer.jobs[0].segment
    assert segment.segment_id == segment_dir.name
    assert segment.source("front") == segment_dir / "front.mp4"
    assert segment.source("side") == segment_dir / "side.mp4"
    assert segment.errors == {"front": "front writer unavailable"}
    assert app.record_btn.config["text"] == app_ui.RECORD_BTN_TEXT["idle"]


def test_submit_callback_sees_segment_as_latest_before_submit_returns(tmp_path):
    app, finalizer, _segment_dir = _dual_submission_app(tmp_path)
    app._compare_update_queue = Queue()
    app.compare_segment_var = _Var("片段：-")
    app.compare_status_var = _Var("自动比对：待机")
    app.compare_score_var = _Var("正面：-　侧面：-　综合：-")
    app.compare_error_var = _Var("")
    observed_latest: list[str | None] = []

    def submit(segment, _plan) -> bool:
        update = SegmentUpdate(segment.segment_id, "queued", "queued")
        app_ui.App._post_recording_compare_update(app, update)
        app_ui.App._drain_recording_compare_updates(app)
        observed_latest.append(app._latest_compare_segment_id)
        return True

    finalizer.submit = submit
    app_ui.App._on_record_stop(app)

    latest = app._latest_compare_segment_id
    assert observed_latest == [latest]
    assert latest is not None
    assert app.compare_segment_var.get() == f"片段：{latest}"


def test_close_finalizes_current_dual_segment_then_cancels_postprocessor(tmp_path):
    app, finalizer, _segment_dir = _dual_submission_app(tmp_path)
    app._stop_evt = threading.Event()
    app._poll_close_workers = lambda: None
    app._close_deadline = None

    app_ui.App._on_close(app)
    app._close_prepare_worker.join(1.0)

    assert app._closing is True
    assert app._stop_evt.is_set()
    assert len(finalizer.jobs) == 1
    assert finalizer.cancel_calls == 1


def test_close_returns_while_inflight_finalize_finishes_before_cancelling(tmp_path):
    app, finalizer, _segment_dir = _dual_submission_app(tmp_path)
    app._stop_evt = threading.Event()
    app._poll_close_workers = lambda: None
    app._close_deadline = None
    submit_started = threading.Event()
    release_submit = threading.Event()
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
            app_ui.App._close_recording_session(app)
        except BaseException as exc:  # noqa: BLE001 - thread failures must reach test
            errors.append(exc)

    finalize_thread = threading.Thread(target=run)
    finalize_thread.start()
    assert submit_started.wait(1.0)

    app_ui.App._on_close(app)

    # 结束+提交与 cancel_all 在同一把锁里串行：提交返回前不取消
    assert finalizer.cancel_calls == 0
    assert app._close_deadline is not None
    assert app._close_prepare_worker.is_alive()
    release_submit.set()
    finalize_thread.join(2.0)
    app._close_prepare_worker.join(2.0)

    assert not finalize_thread.is_alive()
    assert not app._close_prepare_worker.is_alive()
    assert errors == []
    assert len(finalizer.jobs) == 1
    assert finalizer.cancel_calls == 1


def test_close_deadline_is_set_before_blocking_cancel_work(tmp_path):
    app, finalizer, _segment_dir = _dual_submission_app(tmp_path)
    app._stop_evt = threading.Event()
    app._poll_close_workers = lambda: None
    app._close_deadline = None
    cancel_started = threading.Event()
    release_cancel = threading.Event()
    original_cancel = finalizer.cancel_all

    def cancel_all() -> None:
        cancel_started.set()
        assert release_cancel.wait(2.0)
        original_cancel()

    finalizer.cancel_all = cancel_all

    app_ui.App._on_close(app)

    assert app._close_deadline is not None
    assert cancel_started.wait(1.0)
    assert app._close_prepare_worker.is_alive()
    release_cancel.set()
    app._close_prepare_worker.join(2.0)
    assert not app._close_prepare_worker.is_alive()


def test_recording_status_refresh_returns_without_recording_locks_during_close(tmp_path):
    app, finalizer, _segment_dir = _dual_submission_app(tmp_path)
    app._closing = True
    app._stop_evt = threading.Event()
    app.record_btn.configure(state="disabled")
    app.record_stop_btn = _Widget()
    app.record_stop_btn.configure(state="disabled")
    app.recording_status_var = _Var("closing")

    started = threading.Event()
    errors: list[BaseException] = []

    def refresh() -> None:
        started.set()
        try:
            app_ui.App._refresh_recording_status(app)
        except BaseException as exc:  # noqa: BLE001 - surface thread failures
            errors.append(exc)

    app._recording._lock.acquire()
    app._recording._recorder._lock.acquire()
    worker = threading.Thread(target=refresh)
    try:
        worker.start()
        assert started.wait(1.0)
        worker.join(0.2)
        assert not worker.is_alive()
    finally:
        app._recording._recorder._lock.release()
        app._recording._lock.release()
        worker.join(1.0)

    assert errors == []
    assert finalizer.jobs == []
    assert app.recording_status_var.get() == "closing"
    assert app.record_btn.config["state"] == "disabled"
    assert app.record_stop_btn.config["state"] == "disabled"


def test_recording_status_refresh_clears_worker_finalized_main_stop_state(tmp_path):
    app, finalizer, _segment_dir = _dual_submission_app(tmp_path)
    app._stop_evt = threading.Event()
    app._stop_evt.set()
    app.recording_status_var = _Var("录制中：旧片段")
    app.record_stop_btn = _Widget()
    app.record_stop_btn.configure(state="normal")
    app._sync_record_stop_enabled = lambda state: app_ui.App._sync_record_stop_enabled(
        app, state
    )

    app_ui.App._close_recording_session(app)
    assert app._recording.state == "idle"

    app_ui.App._refresh_recording_status(app)

    assert len(finalizer.jobs) == 1
    assert app.recording_status_var.get() == ""
    assert app.record_stop_btn.config["state"] == "disabled"


def test_record_stop_returns_without_recording_lock_after_stop_requested(tmp_path):
    app, finalizer, _segment_dir = _dual_submission_app(tmp_path)
    app._stop_evt = threading.Event()
    app._stop_evt.set()
    app.record_btn.configure(state="disabled")
    app.record_stop_btn = _Widget()
    app.record_stop_btn.configure(state="disabled")

    started = threading.Event()
    errors: list[BaseException] = []

    def stop_recording() -> None:
        started.set()
        try:
            app_ui.App._on_record_stop(app)
        except BaseException as exc:  # noqa: BLE001 - surface thread failures
            errors.append(exc)

    app._recording._lock.acquire()
    worker = threading.Thread(target=stop_recording)
    try:
        worker.start()
        assert started.wait(1.0)
        worker.join(0.2)
        assert not worker.is_alive()
    finally:
        app._recording._lock.release()
        worker.join(1.0)

    assert errors == []
    assert finalizer.jobs == []
    assert app._recording.state == "recording"
    assert app.record_btn.config["state"] == "disabled"
    assert app.record_stop_btn.config["state"] == "disabled"
