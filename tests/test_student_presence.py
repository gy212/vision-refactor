"""学生练习经 WorkflowHost 驱动真实 PracticeSession 与 PresenceRecordingCycle；仅替换摄像头、录像落盘和语音设备。"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from apps import app_ui
from core.foundation import prefs as user_prefs
from core.live import LiveControls, PresenceSample, RunningInfo
from core.workflows import (
    PRACTICE,
    CommandExecutor,
    ModeManager,
    PracticeSession,
    ScorebookRegistry,
    WorkflowHost,
    WorkflowPorts,
)


class _FakeLive:
    """会话管理器替身：只提供 App 读取的状态、token 与配置。"""

    def __init__(self, state="running", dual=True):
        self.state = state
        self.token = 1
        self.config = SimpleNamespace(dual=dual)

    def active(self):
        return self.state != "idle"


def practice_app(monkeypatch, *, ready=True):
    app = app_ui.App.__new__(app_ui.App)
    app._closing = False
    app._live = _FakeLive()
    app._live_controls = LiveControls()
    app._dual_recording_ready = ready
    app.modes = ModeManager()
    app.modes.enter(PRACTICE)
    app.exam_scorebooks = ScorebookRegistry()
    app.workflows = WorkflowHost(app.exam_scorebooks)
    app.workflows.session_started(app._live.token)
    app.exam_panel = None
    app.root = SimpleNamespace(after=lambda _ms, callback: callback())
    app.student_status_var = Mock()
    for name in ("student_start_btn", "student_end_btn", "student_judge_btn", "student_exit_btn"):
        setattr(app, name, Mock())
    app.feedback_controls = SimpleNamespace(identity=lambda: {"studentId": "001"}, set_recording=Mock())
    app._student_announcer = Mock()
    app._student_dual_cameras_ready = lambda: (True, "")
    app._recording = SimpleNamespace(state="idle")
    app._practice = PracticeSession(app.feedback_controls.identity)
    ports = WorkflowPorts(
        recorder=app_ui._SegmentPort(app),
        occupancy=app_ui._OccupancyPort(app, "practice"),
        announcer=app._student_announcer,
        ui=app_ui._PracticeUi(app),
        analysis=app_ui._PracticeAnalysis(app),
    )
    app.workflows.activate(app._practice, CommandExecutor(ports, clock=lambda: app_ui.time.monotonic()))

    def begin(_self, _intent=None):
        app._recording.state = "recording"
        return SimpleNamespace(segment_id="record_1", directory=Path("record_1"))

    def end(_self, **_kwargs):
        app._recording.state = "idle"

    monkeypatch.setattr(app_ui.App, "_begin_recording_segment", begin)
    monkeypatch.setattr(app_ui.App, "_end_recording_segment", end)
    monkeypatch.setattr(app_ui.model_manager, "is_installed", lambda _model: True)
    user_prefs.PreferenceStore().set(user_prefs.EXAM_ROI, (.3, .2, .7, .9))
    monkeypatch.setattr(app_ui.time, "monotonic", lambda: 100.0)
    return app


def occupancy(app, present, now):
    app_ui.App._on_session_presence(app, PresenceSample(present, now))


def test_cold_start_requests_student_format_without_blind_release(monkeypatch):
    app = practice_app(monkeypatch, ready=False)
    app._live.state = "idle"
    events = []
    app._release_camera_warmups = lambda: events.append("release")

    def start():
        # 学生模式下冻结的请求尺寸为 1080p；格式不一致的预热由池在释放后重开。
        events.append(("start", app_ui.App._camera_capture_size(app)))
        app._live.state = "starting"

    app._start = start
    app._on_practice_start()
    assert events == [("start", (1920, 1080))]
    assert app._practice.phase == "waiting_session"


def test_wait_enter_record_leave_and_rearm(monkeypatch):
    app = practice_app(monkeypatch)
    app._on_practice_start()
    occ = app._live_controls.occupancy.get()
    assert occ.roi == (.3, .2, .7, .9)
    assert "practice" in occ.armed_by
    assert app._recording.state == "idle"
    app.student_start_btn.configure.assert_called_with(state="disabled")
    app.student_end_btn.configure.assert_called_with(state="normal")
    app.student_judge_btn.configure.assert_called_with(state="disabled")
    # 旧排队帧不能使新一轮立即开始。
    occupancy(app, True, 90.0)
    occupancy(app, True, 91.0)
    occupancy(app, True, 100.0)
    occupancy(app, False, 100.4)
    occupancy(app, True, 101.0)
    occupancy(app, True, 101.7)
    app._student_announcer.announce.assert_not_called()
    occupancy(app, True, 101.9)
    assert app._recording.state == "recording"
    assert app._practice.identity == {"studentId": "001"}
    occupancy(app, True, 102.0)
    app._student_announcer.announce.assert_called_once_with("开始")
    # 中途回来重置离场计时。
    occupancy(app, False, 102.1)
    occupancy(app, True, 103.0)
    occupancy(app, False, 103.5)
    occupancy(app, False, 105.49)
    assert app._recording.state == "recording"
    occupancy(app, False, 105.5)
    assert app._recording.state == "idle"
    assert app._practice.presence.phase == "idle"
    assert "practice" not in app._live_controls.occupancy.get().armed_by
    occupancy(app, False, 108.0)
    occupancy(app, True, 109.0)
    assert [c.args[0] for c in app._student_announcer.announce.call_args_list] == ["开始", "结束"]
    # 下一段仍需主动点开始；不会覆盖刚完成、待评判的片段。
    monkeypatch.setattr(app_ui.time, "monotonic", lambda: 110.0)
    app._on_practice_start()
    occupancy(app, True, 110.0)
    occupancy(app, True, 110.9)
    occupancy(app, False, 111.0)
    occupancy(app, False, 113.0)
    assert app._recording.state == "idle"  # 离场2秒，不受考试最短录制3秒限制
    assert app._student_announcer.announce.call_count == 4


@pytest.mark.parametrize("cancel", [False, True])
def test_camera_ready_only_arms_detection_and_cancel_blocks_late_callback(monkeypatch, cancel):
    app = practice_app(monkeypatch, ready=False)
    app._on_practice_start()
    assert app._practice.waiting and app._practice.presence.phase == "idle"
    if cancel:
        app._on_practice_end()
    app_ui.App._on_session_running(app, RunningInfo({}, {}))
    assert app._recording.state == "idle"
    assert (app._practice.presence.phase == "idle") == cancel
    app._student_announcer.announce.assert_not_called()


def test_manual_end_and_stop_do_not_allow_late_detection(monkeypatch):
    app = practice_app(monkeypatch)
    app._on_practice_start()
    occupancy(app, True, 100.0)
    occupancy(app, True, 100.9)
    app._on_practice_end()
    occupancy(app, False, 101.0)
    occupancy(app, False, 103.0)
    assert app._student_announcer.announce.call_count == 2
    app._on_practice_start()
    app._live.state = "stopping"
    occupancy(app, True, 104.0)
    occupancy(app, True, 105.0)
    assert app._recording.state == "idle"


def test_failed_recording_does_not_announce_or_retry(monkeypatch):
    app = practice_app(monkeypatch)
    monkeypatch.setattr(app_ui.App, "_begin_recording_segment", lambda _self, _intent=None: None)
    monkeypatch.setattr(app_ui.messagebox, "showerror", Mock())
    app._on_practice_start()
    occupancy(app, True, 100.0)
    occupancy(app, True, 100.9)
    occupancy(app, True, 102.0)
    assert app._practice.presence.phase == "idle"
    assert not app._practice.waiting
    app._student_announcer.announce.assert_not_called()
    app_ui.messagebox.showerror.assert_called_once()


def test_windows_announcer_launches_without_console(monkeypatch):
    from core.workflows import announcer
    calls = []
    monkeypatch.setattr(announcer.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(announcer.sys, "platform", "win32")
    monkeypatch.setattr(announcer.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    announcer._default_windows_speak("开始'测试")
    command = calls[0][0][0]
    assert command[command.index("-WindowStyle") + 1] == "Hidden"
    assert "开始''测试" in command[-1]
    assert calls[0][1]["creationflags"] == 0x08000000
    assert calls[0][1]["stdin"] == announcer.subprocess.DEVNULL
