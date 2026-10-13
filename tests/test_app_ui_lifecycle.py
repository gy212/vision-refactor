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
from core.live import LiveControls, SessionEvent, SessionOutcome
from core.media import CameraService, CameraWarmupError, CameraWarmupStopped, CatalogSnapshot
from core.recording import (
    ProcessingPlan,
    RecordingService,
    SegmentFinalizer,
    SegmentMeta,
    SegmentRecorder,
    SegmentUpdate,
)
from core.recording import recorder as recording_recorder
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


class _FakeLive:
    """会话管理器替身：记录 stop/wait_idle。默认 stop 后会话线程立即结束；
    ``ends_on_stop=False`` 模拟迟迟不结束的会话线程，``on_wait`` 模拟它在等待期间收尾。"""

    def __init__(self, state: str = "idle", *, on_wait=None, ends_on_stop: bool = True) -> None:
        self.state = state
        self.ends_on_stop = ends_on_stop
        self.stops: list[str] = []
        self.waits: list[float] = []
        self.started: list[object] = []
        self.on_wait = on_wait
        self.token = 1
        self.config = None
        self.preview = None

    def active(self) -> bool:
        return self.state != "idle"

    def is_current(self, token: int) -> bool:
        return token == self.token

    def start(self, config, *, start_click=None):
        self.started.append(config)
        self.config = config
        self.state = "starting"
        return self.token

    def stop(self, reason: str = "user_stop") -> bool:
        self.stops.append(reason)
        if self.state in ("idle", "stopping"):
            return False
        self.state = "idle" if self.ends_on_stop else "stopping"
        return True

    def wait_idle(self, timeout: float) -> bool:
        self.waits.append(timeout)
        if self.on_wait is not None:
            self.on_wait()
        return self.state == "idle"


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
    app._live = _FakeLive()
    app._live_controls = LiveControls()
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
    app._fake_pool = _SyncPool()
    app._camera_service = CameraService(pool=app._fake_pool)
    return app


def _closing_app(finalizer=None):
    app = _camera_app()
    app._recording = RecordingService(
        finalizer or _FinalizerDouble(), plan_for=lambda _segment: None
    )
    app.exam_scorebooks = ScorebookRegistry()
    app._closing = False
    app._close_deadline = None
    app.root = _QueuedRoot()
    return app


def _enumeration_app(pending):
    applied: list[tuple[list[object], bool]] = []
    app = SimpleNamespace(
        _camera_service=SimpleNamespace(refresh=lambda on_done: pending.append(on_done)),
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
    app._live = _FakeLive()
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

    assert app._fake_pool.sync_calls == [
        (app_ui.CaptureSpec(0, 1280, 720), None)
    ]


def test_two_camera_selection_declares_both_specs():
    app = _warmup_app()

    app_ui.App._sync_camera_warmup(app)

    assert app._fake_pool.sync_calls == [
        (app_ui.CaptureSpec(0, 1280, 720), app_ui.CaptureSpec(1, 1280, 720))
    ]


def test_same_camera_selection_never_warms_second_role():
    app = _warmup_app(second_label="摄像头 0")

    app_ui.App._sync_camera_warmup(app)

    assert app._fake_pool.sync_calls == [
        (app_ui.CaptureSpec(0, 1280, 720), None)
    ]


def test_student_mode_declares_1080p_specs_without_reading_flag_later():
    app = _warmup_app()
    app._student_practice_active = True

    app_ui.App._sync_camera_warmup(app)
    app._student_practice_active = False

    # D-6：学生练习严格要求 1080p。
    assert app._fake_pool.sync_calls == [
        (
            app_ui.CaptureSpec(0, 1920, 1080, policy="require"),
            app_ui.CaptureSpec(1, 1920, 1080, policy="require"),
        )
    ]


def test_enumerated_device_names_are_frozen_into_specs_for_open_check():
    app = _warmup_app()
    app._camera_entries = [
        app_ui.CameraEntry(label="摄像头 0: Front", index=0, name="Front"),
        app_ui.CameraEntry(label="摄像头 1: Side", index=1, name="Side"),
    ]
    app.camera_choice_var_2 = _Var("摄像头 1: Side")

    app_ui.App._sync_camera_warmup(app)

    # V-3：打开时在采集进程内复核设备名。
    assert app._fake_pool.sync_calls == [
        (
            app_ui.CaptureSpec(0, 1280, 720, expected_name="Front"),
            app_ui.CaptureSpec(1, 1280, 720, expected_name="Side"),
        )
    ]


def test_running_session_blocks_warmup_changes_unless_allowed():
    app = _warmup_app()
    app._live = _FakeLive("running")

    app_ui.App._sync_camera_warmup(app)
    assert app._fake_pool.sync_calls == []

    app_ui.App._sync_camera_warmup(app, allow_running=True)
    assert len(app._fake_pool.sync_calls) == 1


def test_warmup_rejection_releases_and_reports_status():
    app = _warmup_app()
    app.status_var = _Var("")

    def reject(_primary, _secondary):
        raise CameraWarmupError("pool closed")

    app._fake_pool.sync = reject

    app_ui.App._sync_camera_warmup(app)

    assert app._fake_pool.release_all_calls == 1
    assert app.status_var.get() == "摄像头预热失败：pool closed"


def test_session_state_freezes_capture_size_for_whole_start():
    app = _warmup_app()
    app._student_practice_active = True
    state = SimpleNamespace(capture_size=app_ui.App._camera_capture_size(app))
    app._student_practice_active = False

    assert app_ui.App._session_spec(state, "1") == app_ui.CaptureSpec(1, 1920, 1080)

    frozen = SimpleNamespace(
        capture_size=(1920, 1080), capture_policy="require", camera_names=((1, "Side"),)
    )
    assert app_ui.App._session_spec(frozen, "1") == app_ui.CaptureSpec(
        1, 1920, 1080, expected_name="Side", policy="require"
    )


def test_process_factory_forwards_frozen_spec(monkeypatch):
    calls = []

    def factory(index, **kwargs):
        calls.append((index, kwargs))
        return _Cap()

    monkeypatch.setattr(app_ui, "ProcessCamera", factory)
    stop = threading.Event()
    app = _camera_app()
    app._student_practice_active = True  # 工厂不再读取模式标志

    app_ui.App._open_camera_process(app, 0, width=1280, height=720, stop_event=stop)

    assert calls == [(0, {
        "width": 1280, "height": 720, "stop_event": stop,
        "expected_name": None, "policy": "prefer",
    })]


def test_camera_enumeration_result_is_applied_only_from_main_thread_queue():
    pending: list[object] = []
    app, applied = _enumeration_app(pending)
    entries = [app_ui.CameraEntry(label="摄像头 0", index=0)]

    app_ui.App._start_enumeration(app)
    pending[0](CatalogSnapshot(tuple(entries), ok=True))

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


def test_camera_enumeration_return_after_close_does_not_touch_tk():
    pending: list[object] = []
    app, applied = _enumeration_app(pending)

    app_ui.App._start_enumeration(app)
    app._closing = True
    pending[0](CatalogSnapshot((app_ui.CameraEntry(label="摄像头 0", index=0),), ok=True))
    app_ui.App._drain_camera_enum_results(app)

    assert applied == []
    assert app._camera_enum_result_queue.empty()


def test_refresh_keeps_camera_warmups_while_enumerating():
    # V-6：刷新不再先放弃预热、等设备释放；兜底探测由 DeviceCatalog 跳过占用的设备。
    app = _warmup_app()
    app._enum_busy = threading.Event()
    events: list[str] = []
    app._release_camera_warmups = lambda: events.append("release")
    app._start_enumeration = lambda: events.append("enumerate")

    app_ui.App._refresh_cameras(app)

    assert events == ["enumerate"]


def _start_app(state):
    app = _camera_app()
    app._collect_state = lambda: state
    app.start_btn = _Widget()
    app.stop_btn = _Widget()
    app.status_var = _Var()
    app.actions_var = _Var()
    app.match_var = _Var("识别到：直拳 (0.91)")
    app.progress_var = _Var()
    app.progress_text_var = _Var()
    app.progress_bar = _Widget()
    app.rotate_var = _Var("0°")
    app.rotate_var_2 = _Var("0°")
    app._current_auto_compare = lambda prefer_ui=False: True
    app._set_refresh_enabled = lambda: None
    app._set_dual_preview_visible = lambda _visible: None
    app._set_running_controls = lambda _running: None
    return app


def test_video_session_releases_camera_warmups_before_starting():
    events: list[str] = []
    state = app_ui.UiState(source="input.mp4", pose_variant="full", workers=1, enable_hands=True)
    app = _start_app(state)
    app._release_camera_warmups = lambda: events.append("release")
    original_start = app._live.start
    app._live.start = lambda config, **kw: (events.append("start"), original_start(config, **kw))[1]

    app_ui.App._start(app)

    assert events == ["release", "start"]
    assert app._live.started[0].is_file and str(app._live.started[0].video_path) == "input.mp4"


def test_start_resets_stale_match_text():
    state = app_ui.UiState(source="0", pose_variant="full", workers=1, enable_hands=True)
    app = _start_app(state)

    app_ui.App._start(app)

    assert app.match_var.get() == "识别：待机"
    assert len(app._live.started) == 1 and not app._live.started[0].dual
    assert app.stop_btn.config["state"] == "normal"


def test_start_refused_while_previous_session_is_finishing():
    app = _start_app(None)
    app._live.state = "stopping"

    app_ui.App._start(app)

    assert app._live.started == []
    assert app.status_var.get() == "上一会话仍在收尾，请稍候再点「开始」"


def _ended_app(config):
    calls: list[bool] = []
    app = _camera_app()
    app._live.config = config
    app.start_btn = _Widget()
    app.stop_btn = _Widget()
    app.status_var = _Var("就绪")
    app._set_refresh_enabled = lambda: None
    app._set_running_controls = lambda _running: None
    app._sync_camera_warmup = lambda *, allow_running=False: calls.append(allow_running)
    return app, calls


@pytest.mark.parametrize(
    ("outcome", "is_file", "rewarm"),
    [("stopped", False, True), ("failed", False, False), ("failed", True, True)],
)
def test_session_end_rewarms_unless_camera_session_failed(outcome, is_file, rewarm):
    """V-1：摄像头会话失败时设备已隔离/释放，不再声明预热；视频会话结束总是恢复预热。"""
    app, calls = _ended_app(SimpleNamespace(is_file=is_file, dual=False))
    status = "初始化失败：boom" if outcome == "failed" else "已停止"

    app_ui.App._on_session_ended(app, SessionOutcome(outcome, status))

    assert calls == ([True] if rewarm else [])
    assert app.status_var.get() == status
    assert app.start_btn.config["state"] == "normal"
    assert app.stop_btn.config["state"] == "disabled"


def test_session_events_from_stale_sessions_are_ignored():
    app, calls = _ended_app(SimpleNamespace(is_file=False, dual=True))
    app._live_events = Queue()
    app._dual_recording_ready = True
    app._live.token = 4

    app._live_events.put(SessionEvent("ended", 3, SessionOutcome("stopped", "旧会话")))
    app_ui.App._drain_session_events(app)
    assert calls == [] and app._dual_recording_ready is True

    app._live_events.put(SessionEvent("ended", 4, SessionOutcome("failed", "初始化失败：boom")))
    app_ui.App._drain_session_events(app)
    assert app._dual_recording_ready is False
    assert app.status_var.get() == "初始化失败：boom"


def test_close_permanently_closes_dual_camera_warmup_pool():
    app = _closing_app()
    pool = _SyncPool()
    app._camera_service = CameraService(pool=pool)

    app_ui.App._on_close(app)
    app._close_prepare_worker.join(timeout=1.0)

    assert not app._close_prepare_worker.is_alive()
    assert pool.close_calls == 1


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
    app._camera_service = CameraService(pool=SimpleNamespace(
        release_all=lambda: None,
        close=lambda: release_prepare.wait(2.0),
        wait_released=lambda _timeout: True,
    ))

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
    app._live = _FakeLive("running")
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
    # V-4：单摄转码进 finalizer 单独的 io-finish 线；该线程非 daemon，进程退出前跑完
    assert threads == [("finalizer/segment-finalizer/io-finish", False)]
    assert not app._recording.finalizer.idle
    finish.set()
    assert app._recording.finalizer.wait_idle(2.0)


def test_close_waits_for_single_camera_transcode_without_blocking_tk_loop(monkeypatch, tmp_path):
    entered, finish, _threads = _blocking_transcode(monkeypatch)
    app = _single_recording_app(tmp_path)
    app_ui.App._end_recording_segment(app)
    assert entered.wait(timeout=1.0)

    app_ui.App._on_close(app)

    assert app._live.stops == [app_ui.APP_CLOSE]
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

    app._live = _FakeLive("running", ends_on_stop=False)   # 会话线程一直不结束
    monkeypatch.setattr(app_ui.time, "monotonic", lambda: clock[0])

    app_ui.App._on_close(app)
    app._close_prepare_worker.join(1.0)
    assert app.root.destroy_calls == 0
    assert app._live.waits == [app_ui._CLOSE_JOIN_SLICE_S, 0]

    clock[0] = app_ui._CLOSE_JOIN_TIMEOUT_S + 0.1
    _delay, poll = app.root.after_calls.pop(0)
    poll()
    assert app.root.destroy_calls == 1


def test_close_waits_for_transcode_submitted_by_capture_finally(monkeypatch, tmp_path):
    entered, finish, _threads = _blocking_transcode(monkeypatch)
    app = _single_recording_app(tmp_path)

    def session_finishes() -> None:
        # 会话线程收尾（FrameSink.close_session）在关窗等待期间才提交单摄片段
        if app._live.state != "idle":
            app._live.state = "idle"
            app_ui.App._close_recording_session(app)

    app._live = _FakeLive("running", on_wait=session_finishes, ends_on_stop=False)
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
        _live=_FakeLive("running"),
        _live_controls=LiveControls(),
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
    app.stop_btn = _Widget()
    app.status_var = _Var()

    app_ui.App._stop(app)
    app_ui.App._close_recording_session(app)
    app_ui.App._close_recording_session(app)

    assert app._live.stops == ["user_stop"]
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
            app.stop_btn = _Widget()
            app.status_var = _Var()
            app_ui.App._stop(app)
            app_ui.App._close_recording_session(app)

        assert len(finalizer.jobs) == 1
        segment = finalizer.jobs[0].segment
        assert [segment.views[v].frames for v in ("front", "side")] == [0, 0]


def test_second_dual_segment_can_start_before_first_result_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(app_ui, "save_record_dir", lambda _path: None)
    # 两个片段的时间戳钉死，不依赖时钟在两次调用之间前进
    stamps = iter(["20260101_000000_000001", "20260101_000000_000002"])
    monkeypatch.setattr(recording_recorder, "new_stamp", lambda: next(stamps))
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
    app._poll_close_workers = lambda: None
    app._close_deadline = None

    app_ui.App._on_close(app)
    app._close_prepare_worker.join(1.0)

    assert app._closing is True
    assert app._live.stops == [app_ui.APP_CLOSE]
    assert len(finalizer.jobs) == 1
    assert finalizer.cancel_calls == 1


def test_close_returns_while_inflight_finalize_finishes_before_cancelling(tmp_path):
    app, finalizer, _segment_dir = _dual_submission_app(tmp_path)
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
    app._live.state = "stopping"
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
    app._live.state = "stopping"
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
