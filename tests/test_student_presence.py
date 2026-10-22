"""学生练习经 WorkflowHost 驱动真实 PracticeSession 与 PresenceRecordingCycle；仅替换摄像头、录像落盘和语音设备。"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from apps.tk.controllers import practice as practice_mod
from apps.tk.controllers import recording as recording_mod
from core.foundation import prefs as user_prefs
from core.live import PresenceSample, RunningInfo, SessionEvent
from tests.tk_app_doubles import practice_app as _practice_app
from tests.tk_app_doubles import root  # noqa: F401


def practice_app(root, monkeypatch, *, ready=True):
    """练习流程真实；开录/停录替换为只改状态，模型视为已安装，时钟钉在 100 秒。"""
    app = _practice_app(root, ready=ready, identity=lambda: {"studentId": "001"})
    app.recording.service = SimpleNamespace(state="idle")

    def begin(_intent=None, on_confirm=None):
        app.recording.service.state = "recording"
        segment = SimpleNamespace(segment_id="record_1", directory=Path("record_1"))
        if on_confirm is not None:   # 两路写出首帧的确认（真实录制经 Tk 线程回送）
            on_confirm(segment, None)
        return segment

    def end(**_kwargs):
        app.recording.service.state = "idle"

    app.recording.begin = begin
    app.recording.end = end
    monkeypatch.setattr(recording_mod, "is_installed", lambda _model: True)
    user_prefs.PreferenceStore().set(user_prefs.EXAM_ROI, (.3, .2, .7, .9))
    monkeypatch.setattr(practice_mod.time, "monotonic", lambda: 100.0)
    return app


def occupancy(app, present, now):
    app.session.handle(SessionEvent("presence", app.session.live.token, PresenceSample(present, now)))


def _buttons(app):
    block = app.window.practice
    return {name: str(getattr(block, name).cget("state")) for name in ("start_btn", "end_btn", "judge_btn")}


def test_cold_start_requests_student_format_without_blind_release(root, monkeypatch):
    app = practice_app(root, monkeypatch, ready=False)
    app.session.live.state = "idle"
    events = []
    app.cameras.release_warmups = lambda: events.append("release")

    def start():
        # 学生模式下冻结的请求尺寸为 1080p；格式不一致的预热由池在释放后重开。
        events.append(("start", app.cameras.capture_size()))
        app.session.live.state = "starting"

    app.session.start = start
    app.practice.start()
    assert events == [("start", (1920, 1080))]
    assert app.practice.practice.phase == "waiting_session"


def test_wait_enter_record_leave_and_rearm(root, monkeypatch):
    app = practice_app(root, monkeypatch)
    app.practice.start()
    occ = app.controls.occupancy.get()
    assert occ.roi == (.3, .2, .7, .9)
    assert "practice" in occ.armed_by
    assert app.recording.state == "idle"
    assert _buttons(app) == {"start_btn": "disabled", "end_btn": "normal", "judge_btn": "disabled"}
    # 旧排队帧不能使新一轮立即开始。
    occupancy(app, True, 90.0)
    occupancy(app, True, 91.0)
    occupancy(app, True, 100.0)
    occupancy(app, False, 100.4)
    occupancy(app, True, 101.0)
    occupancy(app, True, 101.7)
    app.practice.announcer.announce.assert_not_called()
    occupancy(app, True, 101.9)
    assert app.recording.state == "recording"
    assert app.practice.practice.identity == {"studentId": "001"}
    occupancy(app, True, 102.0)
    app.practice.announcer.announce.assert_called_once_with("开始")
    # 中途回来重置离场计时。
    occupancy(app, False, 102.1)
    occupancy(app, True, 103.0)
    occupancy(app, False, 103.5)
    occupancy(app, False, 105.49)
    assert app.recording.state == "recording"
    occupancy(app, False, 105.5)
    assert app.recording.state == "idle"
    assert app.practice.practice.presence.phase == "idle"
    assert "practice" not in app.controls.occupancy.get().armed_by
    occupancy(app, False, 108.0)
    occupancy(app, True, 109.0)
    assert [c.args[0] for c in app.practice.announcer.announce.call_args_list] == ["开始", "结束"]
    # 下一段仍需主动点开始；不会覆盖刚完成、待评判的片段。
    monkeypatch.setattr(practice_mod.time, "monotonic", lambda: 110.0)
    app.practice.start()
    occupancy(app, True, 110.0)
    occupancy(app, True, 110.9)
    occupancy(app, False, 111.0)
    occupancy(app, False, 113.0)
    assert app.recording.state == "idle"  # 离场2秒，不受考试最短录制3秒限制
    assert app.practice.announcer.announce.call_count == 4


@pytest.mark.parametrize("cancel", [False, True])
def test_camera_ready_only_arms_detection_and_cancel_blocks_late_callback(root, monkeypatch, cancel):
    app = practice_app(root, monkeypatch, ready=False)
    app.practice.start()
    assert app.practice.practice.waiting and app.practice.practice.presence.phase == "idle"
    if cancel:
        app.practice.end()
    app.session.handle(SessionEvent("running", app.session.live.token, RunningInfo({}, {})))
    assert app.recording.state == "idle"
    assert (app.practice.practice.presence.phase == "idle") == cancel
    app.practice.announcer.announce.assert_not_called()


def test_manual_end_and_stop_do_not_allow_late_detection(root, monkeypatch):
    app = practice_app(root, monkeypatch)
    app.practice.start()
    occupancy(app, True, 100.0)
    occupancy(app, True, 100.9)
    app.practice.end()
    occupancy(app, False, 101.0)
    occupancy(app, False, 103.0)
    assert app.practice.announcer.announce.call_count == 2
    app.practice.start()
    app.session.live.state = "stopping"
    occupancy(app, True, 104.0)
    occupancy(app, True, 105.0)
    assert app.recording.state == "idle"


def test_failed_recording_does_not_announce_or_retry(root, monkeypatch):
    app = practice_app(root, monkeypatch)
    app.recording.begin = lambda _intent=None, on_confirm=None: None
    showerror = Mock()
    monkeypatch.setattr(practice_mod.messagebox, "showerror", showerror)
    app.practice.start()
    occupancy(app, True, 100.0)
    occupancy(app, True, 100.9)
    occupancy(app, True, 102.0)
    assert app.practice.practice.presence.phase == "idle"
    assert not app.practice.practice.waiting
    app.practice.announcer.announce.assert_not_called()
    showerror.assert_called_once()


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
