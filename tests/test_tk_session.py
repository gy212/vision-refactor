# -*- coding: utf-8 -*-
"""会话控制器：开始时冻结界面设置、开始/停止、会话事件回到 Tk 线程、预览渲染、运行态控件联动与旋转。

取代 test_app_ui_dual_camera、test_app_ui_lifecycle、test_app_controls 中对半成品 App 的同名用例。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from apps.tk.controllers import session as session_mod
from apps.tk.controllers.session import UiState, next_rotation_cw, session_config, session_spec
from apps.tk.views.capture_panel import NO_SECOND_CAMERA
from apps.tk.views.recording_panel import RECORD_BTN_TEXT
from core.live import CameraInput, LiveControls, PreviewStore, SessionEvent, SessionOutcome, StartupMetrics
from core.media import PRIMARY, SECONDARY, CameraService, CaptureSpec
from core.media import CameraDevice as CameraEntry
from core.workflows import PRACTICE
from tests.tk_app_doubles import FakeLive, build_app, root  # noqa: F401


def _state_app(root, *, source="0", second_label=NO_SECOND_CAMERA, skeleton=False, auto_compare=True):
    app = build_app(root, live=FakeLive())
    app.cameras.entries = [CameraEntry(label="摄像头 0", index=0), CameraEntry(label="摄像头 1", index=1)]
    if source.isdigit():
        app.cameras.source.select_camera(int(source))
    else:
        app.cameras.source.select_video(source)
    app.window.capture.camera_choice_2.set(second_label)
    app.window.recording.record_skeleton_var.set(skeleton)
    app.window.recording.auto_compare_var.set(auto_compare)
    return app


def _state(widget) -> str:
    return str(widget.cget("state"))


# ---------------------------------------------------------------- 冻结界面设置
def test_collect_state_requires_input_source(root):
    app = build_app(root, live=FakeLive())
    with pytest.raises(ValueError, match="请先选择摄像头或视频"):
        app.session.collect_state()


def test_collect_state_sets_source2_when_two_distinct_cameras_selected(root):
    state = _state_app(root, second_label="摄像头 1").session.collect_state()
    assert (state.source, state.source2, state.enable_hands) == ("0", "1", True)


def test_collect_state_rejects_same_camera_for_both_views(root):
    with pytest.raises(ValueError, match="两个摄像头不能选同一个"):
        _state_app(root, second_label="摄像头 0").session.collect_state()


def test_collect_state_ignores_second_camera_for_video_file_source(root):
    state = _state_app(root, source="input.mp4", second_label="摄像头 1", skeleton=True).session.collect_state()
    # 全局骨架开关与双摄无关；文件源同样透传勾选状态
    assert (state.source, state.source2, state.record_skeleton) == ("input.mp4", None, True)


@pytest.mark.parametrize("second", [NO_SECOND_CAMERA, "摄像头 1"])
@pytest.mark.parametrize("skeleton", [False, True])
def test_collect_state_captures_skeleton_for_single_and_dual(root, second, skeleton):
    assert _state_app(root, second_label=second, skeleton=skeleton).session.collect_state().record_skeleton is skeleton


def test_collect_state_auto_compare_only_applies_to_dual(root):
    assert _state_app(root, second_label="摄像头 1", auto_compare=False).session.collect_state().auto_compare is False


def test_collect_state_keeps_auto_compare_true_for_single_camera(root):
    assert _state_app(root, auto_compare=False).session.collect_state().auto_compare is True


def test_collect_state_reads_runtime_rotation_and_ui_settings(root):
    app = _state_app(root, second_label="摄像头 1")
    app.session.set_rotation(90, PRIMARY, announce=False)
    app.session.set_rotation(270, SECONDARY, announce=False)
    capture = app.window.capture
    capture.pose_var.set("heavy")
    capture.workers_var.set(4)
    capture.enable_hands_var.set(False)
    state = app.session.collect_state()
    assert (state.rotate, state.rotate2) == (90, 270)
    assert (state.pose_variant, state.workers, state.enable_hands) == ("heavy", 4, False)


def test_ui_defaults_record_skeleton_off_and_auto_compare_on(root):
    panel = build_app(root).window.recording
    assert panel.record_skeleton_var.get() is False
    assert panel.auto_compare_var.get() is True


@pytest.mark.parametrize(("student", "size", "policy"), [(False, (1280, 720), "prefer"), (True, (1920, 1080), "require")])
def test_dual_session_config_freezes_capture_specs(student, size, policy):
    """开始时冻结的请求尺寸与策略原样进入两路 CaptureSpec（学生练习 1080p、require，D-6）。"""
    state = UiState(source="0", source2="1", pose_variant="full", workers=4, enable_hands=True,
                    capture_size=size, capture_policy=policy, camera_names=((1, "Side Cam"),))
    config = session_config(state)
    assert config.dual and config.roles == (PRIMARY, SECONDARY)
    assert config.cameras[PRIMARY] == CaptureSpec(0, *size, policy=policy)
    assert config.cameras[SECONDARY] == CaptureSpec(1, *size, expected_name="Side Cam", policy=policy)
    assert not config.parallel                     # 双摄不走并行推理


def test_session_spec_uses_frozen_size_policy_and_name(root):
    app = build_app(root)
    app.modes.enter(PRACTICE)
    size = app.cameras.capture_size()
    app.modes.leave()
    base = UiState(source="1", pose_variant="full", workers=1, enable_hands=True)
    from dataclasses import replace
    assert session_spec(replace(base, capture_size=size), "1") == CaptureSpec(1, 1920, 1080)
    frozen = replace(base, capture_size=(1920, 1080), capture_policy="require", camera_names=((1, "Side"),))
    assert session_spec(frozen, "1") == CaptureSpec(1, 1920, 1080, expected_name="Side", policy="require")


def test_session_input_opens_camera_stream_without_touching_pool(root):
    """开始只登记设备句柄：等待与接管在会话线程的 CameraStream 上进行，Tk 线程不调用池。"""

    class _Pool:
        def __getattr__(self, name):
            pytest.fail(f"Tk thread must not call pool.{name}")

    app = build_app(root)
    app.cameras.service = CameraService(pool=_Pool())
    config = session_config(UiState(source="0", source2="1", pose_variant="full", workers=1, enable_hands=True))
    assert isinstance(app.cameras.make_input(config), CameraInput)


# ---------------------------------------------------------------- 开始与停止
def _start_app(root, state):
    app = build_app(root, live=FakeLive())
    app.session.collect_state = lambda: state
    return app


def test_video_session_releases_camera_warmups_before_starting(root):
    events: list[str] = []
    app = _start_app(root, UiState(source="input.mp4", pose_variant="full", workers=1, enable_hands=True))
    app.cameras.release_warmups = lambda: events.append("release")
    live = app.session.live
    original = live.start
    live.start = lambda config, **kw: (events.append("start"), original(config, **kw))[1]
    app.session.start()
    assert events == ["release", "start"]
    assert live.started[0].is_file and str(live.started[0].video_path) == "input.mp4"


def test_start_resets_stale_match_text_and_enters_running_controls(root):
    app = _start_app(root, UiState(source="0", pose_variant="full", workers=1, enable_hands=True))
    app.window.status.match_var.set("识别到：直拳 (0.91)")
    app.session.start()
    assert app.window.status.match_var.get() == "识别：待机"
    assert len(app.session.live.started) == 1 and not app.session.live.started[0].dual
    capture = app.window.capture
    assert _state(capture.stop_btn) == "normal"
    assert _state(capture.start_btn) == "disabled" and str(capture.start_btn.cget("text")) == "停止"
    assert app.flags.recording_ready is True   # 单摄开始即可录制


def test_start_refused_while_previous_session_is_finishing(root):
    app = _start_app(root, None)
    app.session.live.state = "stopping"
    app.session.start()
    assert app.session.live.started == []
    assert app.window.status.status_var.get() == "上一会话仍在收尾，请稍候再点「开始」"


def test_start_with_bad_config_shows_error_and_does_not_start(root, monkeypatch):
    errors = []
    monkeypatch.setattr(session_mod.messagebox, "showerror", lambda title, text: errors.append((title, text)))
    app = build_app(root, live=FakeLive())
    app.session.start()
    assert errors == [("配置错误", "请先选择摄像头或视频")]
    assert app.session.live.started == []


def test_stop_does_not_cancel_background_compare_queue(root):
    app = build_app(root, live=FakeLive("running"))
    cancelled = []
    app.recording.service.cancel_all = lambda: cancelled.append(True)
    app.session.stop()
    assert app.session.live.stops == ["user_stop"]
    assert cancelled == []
    assert app.window.status.status_var.get() == "正在停止…"


# ---------------------------------------------------------------- 会话事件
def _ended_app(root, config):
    app = build_app(root, live=FakeLive(config=config))
    calls: list[bool] = []
    app.cameras.sync_warmup = lambda *, allow_running=False: calls.append(allow_running)
    return app, calls


@pytest.mark.parametrize(("outcome", "is_file", "rewarm"), [("stopped", False, True), ("failed", False, False),
                                                            ("failed", True, True)])
def test_session_end_rewarms_unless_camera_session_failed(root, outcome, is_file, rewarm):
    """V-1：摄像头会话失败时设备已隔离/释放，不再声明预热；视频会话结束总是恢复预热。"""
    app, calls = _ended_app(root, SimpleNamespace(is_file=is_file, dual=False))
    status = "初始化失败：boom" if outcome == "failed" else "已停止"
    app.session.handle(SessionEvent("ended", 1, SessionOutcome(outcome, status)))
    assert calls == ([True] if rewarm else [])
    assert app.window.status.status_var.get() == status
    assert _state(app.window.capture.start_btn) == "normal"
    assert _state(app.window.capture.stop_btn) == "disabled"


def test_session_events_reach_tk_thread_and_stale_sessions_are_ignored(root):
    app, calls = _ended_app(root, SimpleNamespace(is_file=False, dual=True))
    app.flags.recording_ready = True
    app.session.live.token = 4
    app.session.on_event(SessionEvent("ended", 3, SessionOutcome("stopped", "旧会话")))
    assert not app.session.finished()   # 已投递、尚未在 Tk 线程处理
    app.ui.pump()
    assert calls == [] and app.flags.recording_ready is True
    assert app.session.finished()

    app.session.on_event(SessionEvent("ended", 4, SessionOutcome("failed", "初始化失败：boom")))
    app.ui.pump()
    assert app.flags.recording_ready is False
    assert app.window.status.status_var.get() == "初始化失败：boom"


def test_progress_event_shows_file_progress_and_hides_camera_progress(root):
    app = build_app(root, live=FakeLive("running"))
    app.session.handle(SessionEvent("progress", 1, (5, 10)))
    assert app.window.status.progress_text_var.get() == "进度：5/10（50.0%）"
    app.session.handle(SessionEvent("progress", 1, (5, 0)))
    assert app.window.status.progress_text_var.get() == ""


def test_match_shows_only_while_session_live(root):
    app = build_app(root, live=FakeLive("running"))
    app.session.handle(SessionEvent("match", 1, "识别到：直拳 (0.87)"))
    assert app.window.status.match_var.get() == "识别到：直拳 (0.87)"
    app.session.live.state = "stopping"
    app.session.handle(SessionEvent("match", 1, "识别到：摆拳 (0.50)"))
    assert app.window.status.match_var.get() == "识别到：直拳 (0.87)"


def test_session_running_enables_dual_recording_only_for_live_dual_session(root):
    app = build_app(root, live=FakeLive("stopping", config=SimpleNamespace(dual=True, is_file=False)))
    record_btn = app.window.recording.record_btn
    app.session.handle(SessionEvent("running", 1, None))
    assert app.flags.recording_ready is False and _state(record_btn) == "disabled"

    app.session.live.state = "running"
    app.session.handle(SessionEvent("running", 1, None))
    assert app.flags.recording_ready is True and app.flags.dual_active is True
    assert _state(record_btn) == "normal" and str(record_btn.cget("text")) == RECORD_BTN_TEXT["idle"]


# ---------------------------------------------------------------- 预览渲染
def _preview_app(root, *, record_skeleton=False):
    metrics = StartupMetrics(1, primary_index=0, secondary_index=1, inference=record_skeleton, start_click=1.0)
    store = PreviewStore(1, lambda _role: (0, 0), metrics=metrics)
    app = build_app(root, live=FakeLive("running", preview=store, config=SimpleNamespace(dual=True, is_file=False)))
    shown: list[set[str]] = []
    original = app.window.preview.show
    app.window.preview.show = lambda images, sizes: (shown.append(set(images)), original(images, sizes))
    return app, store, shown


def test_render_shows_dual_frame_atomically_with_status_once_per_stage(root):
    app, store, shown = _preview_app(root, record_skeleton=True)
    image = np.full((2, 3, 3), 7, dtype=np.uint8)
    status = app.window.status
    store.publish("raw", {PRIMARY: image, SECONDARY: image}, {PRIMARY: "front"})
    app.ui.pump()
    assert shown == [{PRIMARY, SECONDARY}]
    assert status.actions_var.get() == "front"
    assert status.status_var.get().startswith("双摄画面已显示")
    assert status.status_var.get().endswith("，模型加载中…")
    assert store.wait_rendered(0)

    status.status_var.set("运行中…")
    store.publish("raw", {PRIMARY: image, SECONDARY: image}, {PRIMARY: "again"})
    app.ui.pump()
    assert status.status_var.get() == "运行中…"      # 同一阶段只提示一次
    assert len(shown) == 2

    store.advance("annotated")
    store.publish("annotated", {PRIMARY: image, SECONDARY: image}, {})
    app.ui.pump()
    assert status.status_var.get().startswith("运行中…（双摄首个标注帧")
    assert status.actions_var.get() == "-"


def test_render_shows_nothing_after_stop_requested(root):
    app, store, shown = _preview_app(root)
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    store.publish("raw", {PRIMARY: image, SECONDARY: image}, {})
    app.session.live.state = "stopping"
    app.ui.pump()
    assert shown == []


# ---------------------------------------------------------------- 运行态控件
def test_running_controls_when_running_and_back(root):
    """运行中禁用摄像头与模型选择、录制按钮启用并置「开始录制」、开始键置「停止」；动作分析始终可用。"""
    app = build_app(root, live=FakeLive("running"))
    app.cameras.entries = [CameraEntry(label="摄像头 0", index=0)]
    app.flags.recording_ready = True
    capture, panel = app.window.capture, app.window.recording
    app.session.set_running_controls(True)
    assert _state(capture.camera_combo) == _state(capture.camera_combo_2) == "disabled"
    assert _state(capture.model_combo) == "disabled"
    assert _state(panel.record_skeleton_check) == "disabled"
    assert _state(panel.auto_compare_scale) == "normal"     # 自动比对滑块运行中可快速切换
    assert _state(panel.record_btn) == "normal" and str(panel.record_btn.cget("text")) == "开始录制"
    assert _state(app.window.compare_btn) == "normal"
    assert str(capture.start_btn.cget("text")) == "停止" and _state(capture.start_btn) == "disabled"
    assert _state(capture.stop_btn) == "normal"
    assert _state(capture.refresh_btn) == "disabled"

    app.session.live.state = "idle"
    app.session.set_running_controls(False)
    assert _state(panel.record_btn) == "disabled"
    assert _state(app.window.compare_btn) == "normal"
    assert str(capture.start_btn.cget("text")) == "开始" and _state(capture.start_btn) == "normal"
    assert _state(capture.stop_btn) == "disabled"
    assert _state(capture.model_combo) == "readonly"
    assert _state(panel.record_skeleton_check) == "normal"
    assert _state(capture.camera_combo) == _state(capture.camera_combo_2) == "readonly"
    assert app.window.status.status_var.get() == "就绪"
    assert _state(capture.refresh_btn) == "normal"


def test_dual_loading_keeps_record_control_disabled(root):
    """双摄两路首帧就绪前其余运行态联动照常，录制按钮保持禁用（就绪后由 running 事件启用）。"""
    app = build_app(root, live=FakeLive("running"))
    app.flags.recording_ready = False
    app.session.set_running_controls(True)
    assert _state(app.window.recording.record_btn) == "disabled"
    assert str(app.window.capture.start_btn.cget("text")) == "停止"


# ---------------------------------------------------------------- 旋转
@pytest.mark.parametrize(("degrees", "expected"), [(0, 90), (90, 180), (180, 270), (270, 0), (45, 90)])
def test_next_rotation_cw_steps_quarter_turns(degrees, expected):
    assert next_rotation_cw(degrees) == expected


def test_runtime_rotation_primary_and_secondary_are_isolated(root):
    app = build_app(root)
    app.session.set_rotation(90, PRIMARY, announce=False)
    app.session.set_rotation(180, SECONDARY, announce=False)
    app.session.rotate_clockwise(PRIMARY)
    assert app.controls.rotation_of(PRIMARY) == 180
    assert app.controls.rotation_of(SECONDARY) == 180
    assert app.window.status.status_var.get() == "主摄旋转 → 180°（点击预览可继续顺时针 90°）"


def test_clicking_hidden_second_preview_does_not_rotate(root):
    app = build_app(root)
    app.window.preview._clicked(SECONDARY)
    assert app.controls.rotation_of(SECONDARY) == 0
    app.window.preview.set_dual_visible(True)
    app.window.preview._clicked(SECONDARY)
    assert app.controls.rotation_of(SECONDARY) == 90


def test_live_controls_are_shared_with_exam_rotation(root):
    app = build_app(root)
    app.session.set_rotation(270, PRIMARY, announce=False)
    assert app.exam.exam_primary_rotation() == 270
    assert isinstance(app.controls, LiveControls)
