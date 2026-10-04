"""学生练习复用真实 PresenceGate；仅替换摄像头、录像落盘和语音设备。"""
from queue import Queue
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from apps import app_ui, exam_panel


def practice_app(monkeypatch, *, ready=True):
    app = app_ui.App.__new__(app_ui.App)
    app._student_practice_active = True
    app._student_pending_record = False
    app._student_presence_gate = None
    app._student_presence_started_at = 0.0
    app._student_judging = False
    app._student_segment_ready = True
    app._student_last_segment_dir = None
    app._record_stamp = None
    app._closing = False
    app._stop_evt = Event()
    app._dual_recording_ready = ready
    app._dual_preview_lock = Lock()
    app._exam_lock = Lock()
    app._current_session_generation = app._active_dual_generation = 1
    app._exam_occupancy_queue = Queue()
    app._exam_panel = SimpleNamespace(on_occupancy=Mock())
    app._worker = SimpleNamespace(is_alive=lambda: True)
    app.root = SimpleNamespace(after=lambda _ms, callback: callback())
    app.student_status_var = Mock()
    for name in ("student_start_btn", "student_end_btn", "student_judge_btn", "student_exit_btn",
                 "auto_compare_var", "auto_compare_scale_var", "record_skeleton_var", "enable_hands_var"):
        setattr(app, name, Mock())
    app.feedback_controls = SimpleNamespace(identity=lambda: {"studentId": "001"}, set_recording=Mock())
    app._student_announcer = Mock()
    app._student_dual_cameras_ready = lambda: (True, "")
    app._rec = SimpleNamespace(state="idle")

    def begin(_self):
        app._rec.state = "recording"
        return True

    def end(_self, **_kwargs):
        app._rec.state = "idle"

    monkeypatch.setattr(app_ui.App, "_begin_recording_segment", begin)
    monkeypatch.setattr(app_ui.App, "_end_recording_segment", end)
    monkeypatch.setattr(app_ui.model_manager, "is_installed", lambda _model: True)
    monkeypatch.setattr(exam_panel, "load_exam_prefs", lambda: {"exam_roi_norm": [.3, .2, .7, .9]})
    monkeypatch.setattr(app_ui.time, "monotonic", lambda: 100.0)
    return app


def occupancy(app, present, now):
    app._exam_occupancy_queue.put((present, now))
    app._drain_exam_occupancy()


def test_cold_start_requests_student_format_without_blind_release(monkeypatch):
    app = practice_app(monkeypatch, ready=False)
    app._worker = None
    events = []
    app._release_camera_warmups = lambda: events.append("release")

    def start():
        # 学生模式下冻结的请求尺寸为 1080p；格式不一致的预热由池在释放后重开。
        events.append(("start", app_ui.App._camera_capture_size(app)))
        app._worker = SimpleNamespace(is_alive=lambda: True)

    app._start = start
    app._student_start()
    assert events == [("start", (1920, 1080))]
    assert app._student_pending_record


def test_wait_enter_record_leave_and_rearm(monkeypatch):
    app = practice_app(monkeypatch)
    app._student_start()
    assert app._exam_roi == (.3, .2, .7, .9)
    assert app._rec.state == "idle"
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
    assert app._rec.state == "recording"
    assert app._student_feedback_identity == {"studentId": "001"}
    occupancy(app, True, 102.0)
    app._student_announcer.announce.assert_called_once_with("开始")
    # 中途回来重置离场计时。
    occupancy(app, False, 102.1)
    occupancy(app, True, 103.0)
    occupancy(app, False, 103.5)
    occupancy(app, False, 105.49)
    assert app._rec.state == "recording"
    occupancy(app, False, 105.5)
    assert app._rec.state == "idle"
    assert app._student_presence_gate is None
    occupancy(app, False, 108.0)
    occupancy(app, True, 109.0)
    assert [c.args[0] for c in app._student_announcer.announce.call_args_list] == ["开始", "结束"]
    app._exam_panel.on_occupancy.assert_not_called()
    # 下一段仍需主动点开始；不会覆盖刚完成、待评判的片段。
    monkeypatch.setattr(app_ui.time, "monotonic", lambda: 110.0)
    app._student_start()
    occupancy(app, True, 110.0)
    occupancy(app, True, 110.9)
    occupancy(app, False, 111.0)
    occupancy(app, False, 113.0)
    assert app._rec.state == "idle"  # 离场2秒，不受考试最短录制3秒限制
    assert app._student_announcer.announce.call_count == 4


@pytest.mark.parametrize("cancel", [False, True])
def test_camera_ready_only_arms_detection_and_cancel_blocks_late_callback(monkeypatch, cancel):
    app = practice_app(monkeypatch, ready=False)
    app._student_start()
    assert app._student_pending_record and app._student_presence_gate is None
    if cancel:
        app._student_end()
    app._post_dual_recording_ready(1)
    assert app._rec.state == "idle"
    assert (app._student_presence_gate is None) == cancel
    app._student_announcer.announce.assert_not_called()


def test_manual_end_and_stop_do_not_allow_late_detection(monkeypatch):
    app = practice_app(monkeypatch)
    app._student_start()
    occupancy(app, True, 100.0)
    occupancy(app, True, 100.9)
    app._student_end()
    occupancy(app, False, 101.0)
    occupancy(app, False, 103.0)
    assert app._student_announcer.announce.call_count == 2
    app._student_start()
    app._stop_evt.set()
    occupancy(app, True, 104.0)
    occupancy(app, True, 105.0)
    assert app._rec.state == "idle"


def test_failed_recording_does_not_announce_or_retry(monkeypatch):
    app = practice_app(monkeypatch)
    monkeypatch.setattr(app_ui.App, "_begin_recording_segment", lambda _self: False)
    monkeypatch.setattr(app_ui.messagebox, "showerror", Mock())
    app._student_start()
    occupancy(app, True, 100.0)
    occupancy(app, True, 100.9)
    occupancy(app, True, 102.0)
    assert app._student_presence_gate is None
    assert not app._student_pending_record
    app._student_announcer.announce.assert_not_called()
    app_ui.messagebox.showerror.assert_called_once()
def test_windows_announcer_launches_without_console(monkeypatch):
    from core import exam_announcer
    calls = []
    monkeypatch.setattr(exam_announcer.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(exam_announcer.sys, "platform", "win32")
    monkeypatch.setattr(exam_announcer.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    exam_announcer._default_windows_speak("开始'测试")
    command = calls[0][0][0]
    assert command[command.index("-WindowStyle") + 1] == "Hidden"
    assert "开始''测试" in command[-1]
    assert calls[0][1]["creationflags"] == 0x08000000
    assert calls[0][1]["stdin"] == exam_announcer.subprocess.DEVNULL
