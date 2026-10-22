# -*- coding: utf-8 -*-
"""录制控制器：录制按钮三态、片段提交（双摄成对、只提交一次）、写错误提示、比对状态回显与考试预检。

取代 test_app_controls 与 test_app_ui_lifecycle 中对半成品 App 的录制用例；关窗相关的在 test_tk_close.py。
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from apps.tk.controllers import recording as recording_mod
from apps.tk.views.recording_panel import RECORD_BTN_TEXT
from core.recording import ProcessingPlan, SegmentFinalizer, SegmentMeta, SegmentUpdate
from core.recording import recorder as recording_recorder
from core.recording import steps as recording_steps
from core.workflows import EXAM
from tests.tk_app_doubles import FakeLive, FinalizerDouble, bind_recording, build_app, frame, root  # noqa: F401


def _text(widget) -> str:
    return str(widget.cget("text"))


def _state(widget) -> str:
    return str(widget.cget("state"))


def _recording_app(root, tmp_path: Path, finalizer=None, *, failing=frozenset(), plan_for=None):
    app = build_app(root, live=FakeLive("running"))
    app.window.recording.record_dir_var.set(str(tmp_path))
    app.recording.record_base_dir = tmp_path
    app.flags.recording_ready = True
    finalizer = finalizer if finalizer is not None else FinalizerDouble()
    bind_recording(app, finalizer, failing=failing, plan_for=plan_for)
    return app, finalizer


def _dual_app(root, tmp_path: Path, *, write_frames: bool = True, failing=frozenset()):
    """双摄会话中、已开录（可选写入一帧）；finalizer 只记录提交。"""
    app, finalizer = _recording_app(root, tmp_path, failing=failing)
    app.flags.dual_active = True
    service = app.recording.service
    service.begin_session({"front": (4, 3), "side": (4, 3)}, 30.0)
    segment = service.start(SegmentMeta(tmp_path))
    assert segment
    if write_frames:
        service.write({"front": frame(), "side": frame()})
    return app, finalizer, segment.directory


def _states(app) -> list[str]:
    return [snap.state for snap in app.recording.service.snapshot().views.values()]


# ---------------------------------------------------------------- 三态按钮
def test_record_btn_text_mapping():
    assert RECORD_BTN_TEXT == {"idle": "开始录制", "recording": "暂停录制", "paused": "继续录制", "stopping": "正在结束…"}


def test_toggle_drives_button_text_cycle(root, tmp_path):
    app, _ = _recording_app(root, tmp_path, plan_for=lambda _segment: None)
    app.recording.service.begin_session({"front": (640, 480)}, 30.0)
    button = app.window.recording.record_btn
    for expected in ("recording", "paused", "recording", "paused", "recording"):
        app.recording.toggle()
        assert app.recording.state == expected
        assert _text(button) == RECORD_BTN_TEXT[expected]


def test_toggle_keeps_both_camera_recorders_in_sync(root, tmp_path):
    app, _ = _recording_app(root, tmp_path, plan_for=lambda _segment: None)
    app.recording.service.begin_session({"front": (640, 480), "side": (480, 640)}, 30.0)
    for expected in ("recording", "paused", "recording"):
        app.recording.toggle()
        assert _states(app) == [expected, expected]


def test_toggle_noop_before_session(root, tmp_path):
    app, _ = _recording_app(root, tmp_path, plan_for=lambda _segment: None)
    app.recording.toggle()
    assert app.recording.state == "idle"
    assert _text(app.window.recording.record_btn) == RECORD_BTN_TEXT["idle"]


@pytest.mark.parametrize("blocker", ["stopping", "dual_loading", "exam_locked", "closing"])
def test_toggle_rejected_when_session_not_accepting(root, tmp_path, blocker):
    app, _ = _recording_app(root, tmp_path, plan_for=lambda _segment: None)
    app.recording.service.begin_session({"front": (640, 480), "side": (480, 640)}, 30.0)
    if blocker == "stopping":
        app.session.live.state = "stopping"
    elif blocker == "dual_loading":
        app.flags.recording_ready = False
    elif blocker == "exam_locked":
        app.flags.exam_locked = True
    else:
        app.flags.closing = True
    app.recording.toggle()
    assert app.recording.service.snapshot().segment is None
    assert _states(app) == ["idle", "idle"]


def test_toggle_starts_dual_segment_under_selected_base_dir(root, tmp_path, monkeypatch):
    app, finalizer, _ = _dual_app(root, tmp_path / "old")
    app.recording.stop_clicked()
    new_base = tmp_path / "new"
    app.window.recording.record_dir_var.set(str(new_base))
    saved: list[Path] = []
    monkeypatch.setattr(recording_mod, "save_record_dir", lambda path: saved.append(Path(path)))
    app.recording.toggle()
    snapshot = app.recording.service.snapshot()
    assert snapshot.state == "recording"
    assert snapshot.segment.directory.parent.parent == new_base
    assert app.recording.record_base_dir == new_base
    assert saved == [new_base]
    assert len(finalizer.jobs) == 1
    assert app.window.status.compare_status_var.get() == "自动比对：录制中"
    assert _state(app.window.recording.record_stop_btn) == "normal"


def test_empty_record_dir_keeps_cached_base_dir(root, tmp_path, monkeypatch):
    app, _finalizer, _ = _dual_app(root, tmp_path)
    app.recording.stop_clicked()
    app.window.recording.record_dir_var.set("")
    monkeypatch.setattr(recording_mod, "save_record_dir", lambda _path: None)
    app.recording.toggle()
    assert app.recording.service.snapshot().segment.directory.parent.parent == tmp_path


# ---------------------------------------------------------------- 片段提交
def test_dual_record_stop_submits_exactly_once_and_allows_next_segment(root, tmp_path):
    app, finalizer, segment_dir = _dual_app(root, tmp_path)
    app.recording.stop_clicked()
    assert app.recording.state == "idle"
    assert len(finalizer.jobs) == 1
    job = finalizer.jobs[0]
    assert job.segment.segment_id == segment_dir.name
    assert job.segment.source("front") == segment_dir / "front.mp4"
    assert job.segment.source("side") == segment_dir / "side.mp4"
    assert [job.segment.views[v].frames for v in ("front", "side")] == [1, 1]
    assert job.plan.result.record_skeleton is False
    assert job.plan.result.auto_compare is True
    app.recording.close_session()
    assert len(finalizer.jobs) == 1
    assert app.flags.dual_active is False


def test_dual_record_stop_submits_auto_compare_disabled_when_only_recording(root, tmp_path):
    app, finalizer, segment_dir = _dual_app(root, tmp_path)
    app.flags.auto_compare = False
    app.recording.stop_clicked()
    assert finalizer.jobs[0].segment.segment_id == segment_dir.name
    assert finalizer.jobs[0].plan.result.auto_compare is False


def test_exam_mode_forces_auto_compare_in_plan(root, tmp_path):
    app, finalizer, _ = _dual_app(root, tmp_path)
    app.flags.auto_compare = False
    app.modes.enter(EXAM)
    app.recording.stop_clicked()
    assert finalizer.jobs[0].plan.result.auto_compare is True


def test_plan_uses_template_pools_cached_on_tk_thread(root, tmp_path):
    app, finalizer, _ = _dual_app(root, tmp_path)
    app.window.recording.add_pool_path("front", tmp_path / "front_a.npz")
    app.recording.pools.changed()
    assert app.window.recording.template_status_var.get().startswith("模板池：正面 1 个 · 侧面 0 个")
    app.recording.stop_clicked()
    assert app.recording.pools.lists() == ([tmp_path / "front_a.npz"], [])
    assert finalizer.jobs[0].plan is not None


def test_main_stop_finally_submits_current_dual_segment_once(root, tmp_path):
    app, finalizer, _ = _dual_app(root, tmp_path)
    app.session.stop()
    app.recording.close_session()
    app.recording.close_session()
    assert app.session.live.stops == ["user_stop"]
    assert app.window.status.status_var.get() == "正在停止…"
    assert len(finalizer.jobs) == 1


@pytest.mark.parametrize("mode", ["explicit", "main_stop"])
def test_zero_frame_dual_segment_is_submitted_from_explicit_and_main_stop(root, tmp_path, mode):
    app, finalizer, _ = _dual_app(root, tmp_path, write_frames=False)
    if mode == "explicit":
        app.recording.stop_clicked()
    else:
        app.session.stop()
        app.recording.close_session()
    assert len(finalizer.jobs) == 1
    assert [finalizer.jobs[0].segment.views[v].frames for v in ("front", "side")] == [0, 0]


def test_second_dual_segment_can_start_before_first_result_finishes(root, tmp_path, monkeypatch):
    monkeypatch.setattr(recording_mod, "save_record_dir", lambda _path: None)
    stamps = iter(["20260101_000000_000001", "20260101_000000_000002"])   # 两个片段的时间戳钉死
    monkeypatch.setattr(recording_recorder, "new_stamp", lambda: next(stamps))
    app, finalizer, _ = _dual_app(root, tmp_path)
    app.recording.stop_clicked()
    first = finalizer.jobs[0].segment
    app.recording.toggle()
    app.recording.service.write({"front": frame(), "side": frame()})
    app.recording.stop_clicked()
    assert len(finalizer.jobs) == 2
    second = finalizer.jobs[1].segment
    assert second.segment_id != first.segment_id
    assert second.source("front") == second.directory / "front.mp4"
    assert [second.views[v].frames for v in ("front", "side")] == [1, 1]


def test_dual_submit_does_not_block_capture_writes(root, tmp_path):
    app, finalizer, _ = _dual_app(root, tmp_path)
    original_submit = finalizer.submit
    write_returned: list[bool] = []

    def submit(segment, plan) -> bool:
        # 提交在服务锁内，但采集线程写帧不经这把锁，不被提交阻塞
        writer = threading.Thread(target=lambda: app.recording.service.write({"front": frame(), "side": frame()}))
        writer.start()
        writer.join(1.0)
        write_returned.append(not writer.is_alive())
        return original_submit(segment, plan)

    finalizer.submit = submit
    app.recording.stop_clicked()
    assert write_returned == [True]


def test_toggle_after_primary_writer_failure_finalizes_old_pair_before_new_segment(root, tmp_path):
    app, finalizer, segment_dir = _dual_app(root, tmp_path, failing=frozenset({"front"}))
    app.recording.toggle()
    assert app.recording.state == "idle"
    assert app.recording.service.snapshot().segment is None
    segment = finalizer.jobs[0].segment
    assert segment.segment_id == segment_dir.name
    assert segment.errors == {"front": "front writer unavailable"}
    assert _text(app.window.recording.record_btn) == RECORD_BTN_TEXT["idle"]


def test_single_camera_transcode_runs_on_non_daemon_finalizer_lane(root, tmp_path, monkeypatch):
    entered, finish = threading.Event(), threading.Event()
    threads: list[tuple[str, bool]] = []

    def transcode(path: Path, stop_evt=None) -> Path:
        current = threading.current_thread()
        threads.append((current.name, current.daemon))
        entered.set()
        assert finish.wait(timeout=2.0)
        return Path(path)

    monkeypatch.setattr(recording_steps, "transcode_to_h264", transcode)
    app, _ = _recording_app(root, tmp_path, SegmentFinalizer())
    app.recording.service.begin_session({"front": (4, 3)}, 30.0)
    assert app.recording.service.start(SegmentMeta(tmp_path))
    app.recording.service.write({"front": frame()})
    app.recording.end()
    assert entered.wait(timeout=1.0)
    # V-4：单摄转码进 finalizer 单独的 io-finish 线；该线程非 daemon，进程退出前跑完
    assert threads == [("finalizer/segment-finalizer/io-finish", False)]
    assert not app.recording.service.finalizer.idle
    finish.set()
    assert app.recording.service.finalizer.wait_idle(2.0)


# ---------------------------------------------------------------- 写错误与录制状态
def test_side_writer_failure_submits_pair_once_without_modal_and_single_session_unaffected(root, tmp_path, monkeypatch):
    errors: list[str] = []
    monkeypatch.setattr(recording_mod.messagebox, "showerror", lambda _title, message: errors.append(message))
    app, finalizer = _recording_app(root, tmp_path, failing=frozenset({"side"}),
                                    plan_for=lambda _segment: ProcessingPlan(()))
    service = app.recording.service
    service.begin_session({"front": (4, 3), "side": (4, 3)}, 30.0)
    assert service.start(SegmentMeta(tmp_path))
    service.write({"front": frame(), "side": frame()})
    views = service.snapshot().views
    assert (views["front"].state, views["side"].state) == ("recording", "idle")

    app.recording.refresh_status()
    # 双摄的写失败由结果 JSON 与比对状态栏承载，不弹框；两路一起结束并只提交一次
    assert app.recording.state == "idle"
    assert errors == []
    assert len(finalizer.jobs) == 1
    assert finalizer.jobs[0].segment.errors == {"side": "side writer unavailable"}
    app.recording.refresh_status()
    assert len(finalizer.jobs) == 1

    # 双摄会话结束后只开主路，旧侧路错误不得误停或污染单摄录制
    service.close_session()
    service.begin_session({"front": (4, 3)}, 30.0)
    assert service.start(SegmentMeta(tmp_path))
    app.recording.refresh_status()
    assert app.recording.state == "recording"
    assert errors == []
    assert app.window.status.recording_status_var.get().startswith("录制中：")


def test_close_session_keeps_single_camera_error_for_next_tick(root, tmp_path, monkeypatch):
    errors: list[str] = []
    monkeypatch.setattr(recording_mod.messagebox, "showerror", lambda _title, message: errors.append(message))
    app, _ = _recording_app(root, tmp_path, SegmentFinalizer(), failing=frozenset({"front"}))
    service = app.recording.service
    service.begin_session({"front": (4, 3)}, 30.0)
    assert service.start(SegmentMeta(tmp_path))
    service.write({"front": frame()})
    app.recording.close_session()
    assert app.flags.dual_active is False
    assert service.snapshot().errors == {}
    app.recording.refresh_status()
    assert errors == ["录制发生错误：\n第一路：front writer unavailable"]
    assert app.recording.pending_errors == []


def test_writer_error_submits_formal_job_once_without_modal(root, tmp_path, monkeypatch):
    errors: list[str] = []
    monkeypatch.setattr(recording_mod.messagebox, "showerror", lambda _title, message: errors.append(message))
    app, finalizer, _ = _dual_app(root, tmp_path, failing=frozenset({"side"}))
    app.recording.refresh_status()
    app.recording.close_session()
    assert errors == []
    assert len(finalizer.jobs) == 1
    assert finalizer.jobs[0].segment.errors == {"side": "side writer unavailable"}


def test_recording_status_shows_both_paths_while_recording(root, tmp_path):
    app, _, segment_dir = _dual_app(root, tmp_path)
    app.recording.refresh_status()
    text = app.window.status.recording_status_var.get()
    assert text.startswith("录制中：正面 ") and "；侧面 " in text
    assert _state(app.window.recording.record_stop_btn) == "normal"


def test_recording_status_refresh_returns_without_recording_locks_during_close(root, tmp_path):
    app, finalizer, _ = _dual_app(root, tmp_path)
    app.flags.closing = True
    status = app.window.status.recording_status_var
    status.set("closing")
    started = threading.Event()
    errors: list[BaseException] = []

    def refresh() -> None:
        started.set()
        try:
            app.recording.refresh_status()
        except BaseException as exc:  # noqa: BLE001 - surface thread failures
            errors.append(exc)

    service = app.recording.service
    service._lock.acquire()
    service._recorder._lock.acquire()
    worker = threading.Thread(target=refresh)
    try:
        worker.start()
        assert started.wait(1.0)
        worker.join(0.2)
        assert not worker.is_alive()
    finally:
        service._recorder._lock.release()
        service._lock.release()
        worker.join(1.0)
    assert errors == [] and finalizer.jobs == []
    assert status.get() == "closing"


def test_recording_status_refresh_clears_worker_finalized_main_stop_state(root, tmp_path):
    app, finalizer, _ = _dual_app(root, tmp_path)
    app.session.live.state = "stopping"
    app.window.status.recording_status_var.set("录制中：旧片段")
    app.window.recording.record_stop_btn.configure(state="normal")
    app.recording.close_session()
    assert app.recording.state == "idle"
    app.recording.refresh_status()
    assert len(finalizer.jobs) == 1
    assert app.window.status.recording_status_var.get() == ""
    assert _state(app.window.recording.record_stop_btn) == "disabled"


def test_record_stop_returns_without_recording_lock_after_stop_requested(root, tmp_path):
    app, finalizer, _ = _dual_app(root, tmp_path)
    app.session.live.state = "stopping"
    started = threading.Event()
    errors: list[BaseException] = []

    def stop_recording() -> None:
        started.set()
        try:
            app.recording.stop_clicked()
        except BaseException as exc:  # noqa: BLE001 - surface thread failures
            errors.append(exc)

    app.recording.service._lock.acquire()
    worker = threading.Thread(target=stop_recording)
    try:
        worker.start()
        assert started.wait(1.0)
        worker.join(0.2)
        assert not worker.is_alive()
    finally:
        app.recording.service._lock.release()
        worker.join(1.0)
    assert errors == [] and finalizer.jobs == []
    assert app.recording.state == "recording"


# ---------------------------------------------------------------- 比对状态回显
def _update_app(root, tmp_path, latest: str):
    app, _ = _recording_app(root, tmp_path)
    app.recording.latest_segment_id = latest
    applied: list[str] = []
    app.scorebooks.apply = lambda update: applied.append(update.segment_id)
    return app, applied


def test_compare_update_ignores_old_segment_and_formats_latest_score(root, tmp_path):
    app, exam_updates = _update_app(root, tmp_path, "record_new")
    status = app.window.status
    status.compare_score_var.set("unchanged")
    app.recording.post_update(SegmentUpdate("record_old", "completed", "done", front_score=0.1, side_score=0.2,
                                            combined_percent=15))
    assert exam_updates == []          # 只投递，Tk 线程排空时才应用
    app.ui.pump()
    # 主界面比对条只跟最新片段；考试台账仍须收到旧片段终态
    assert status.compare_score_var.get() == "unchanged"
    assert exam_updates == ["record_old"]

    latest = SegmentUpdate("record_new", "completed", "done", front_score=0.823, side_score=0.891,
                           combined_percent=86)
    app.recording.post_update(latest)
    app.ui.pump()
    assert status.compare_status_var.get() == "自动比对：已完成"
    assert status.compare_score_var.get() == "正面：82.3%　侧面：89.1%　综合：86%"
    assert status.compare_error_var.get() == ""
    assert exam_updates == ["record_old", "record_new"]

    app.flags.closing = True           # 关窗后没有考试台账要回填：不再投递
    app.recording.post_update(latest)
    app.ui.pump()
    assert exam_updates == ["record_old", "record_new"]


def test_compare_update_formats_failed_skipped_and_cancelled_states(root, tmp_path):
    app, _ = _update_app(root, tmp_path, "record_terminal")
    status = app.window.status
    for state, label, code, message in (("failed", "失败", "video_unreadable", "录像无法读取"),
                                        ("skipped", "已跳过", "annotated_recording", "带骨架录像未自动比对"),
                                        ("cancelled", "已取消", "app_closing", "应用关闭")):
        status.compare_score_var.set("stale score")
        app.recording.post_update(SegmentUpdate("record_terminal", state, message, error_code=code))
        app.ui.pump()
        assert status.compare_status_var.get() == f"自动比对：{label}"
        assert status.compare_score_var.get() == "正面：-　侧面：-　综合：-"
        assert status.compare_error_var.get() == f"{code}：{message}"


def test_compare_update_in_progress_keeps_previous_score(root, tmp_path):
    app, _ = _update_app(root, tmp_path, "record_x")
    app.window.status.compare_score_var.set("kept")
    app.recording.post_update(SegmentUpdate("record_x", "transcoding", "…"))
    app.ui.pump()
    assert app.window.status.compare_status_var.get() == "自动比对：正在转码"
    assert app.window.status.compare_score_var.get() == "kept"


def test_compare_update_already_queued_only_backfills_scorebook_after_close_starts(root, tmp_path):
    app, exam_updates = _update_app(root, tmp_path, "record_latest")
    app.window.status.compare_score_var.set("unchanged")
    app.recording.post_update(SegmentUpdate("record_latest", "completed", "done", front_score=0.9, side_score=0.9,
                                            combined_percent=90))
    app.flags.closing = True
    app.ui.close()                     # 关窗：比对更新以 during_close 投递，仍回填台账
    app.ui.pump()
    assert app.window.status.compare_score_var.get() == "unchanged"
    assert exam_updates == ["record_latest"]


def test_submit_callback_sees_segment_as_latest_before_submit_returns(root, tmp_path):
    app, finalizer, _ = _dual_app(root, tmp_path)
    observed_latest: list[str | None] = []

    def submit(segment, _plan) -> bool:
        app.recording.post_update(SegmentUpdate(segment.segment_id, "queued", "queued"))
        app.ui.pump()
        observed_latest.append(app.recording.latest_segment_id)
        return True

    finalizer.submit = submit
    app.recording.stop_clicked()
    latest = app.recording.latest_segment_id
    assert observed_latest == [latest] and latest is not None
    assert app.window.status.compare_segment_var.get() == f"片段：{latest}"


# ---------------------------------------------------------------- 考试锁定与预检
def test_exam_lock_disables_manual_controls_and_forces_compare(root, tmp_path):
    app, _ = _recording_app(root, tmp_path)
    panel = app.window.recording
    panel.record_skeleton_var.set(True)
    app.recording.set_auto_compare(False)
    app.exam.lock_manual_record(True)
    assert all(_state(w) == "disabled" for w in (panel.record_btn, panel.record_stop_btn,
                                                  panel.record_skeleton_check, panel.auto_compare_scale))
    assert panel.record_skeleton_var.get() is False and panel.auto_compare_var.get() is True
    assert app.flags.auto_compare is True
    app.exam.lock_manual_record(False)
    assert _state(panel.auto_compare_scale) == "normal"
    assert _state(panel.record_btn) == "normal"   # 会话进行中且可录制


def test_auto_compare_slider_snaps_and_updates_hint(root):
    app = build_app(root)
    app.recording.auto_scale_moved("0.3")
    panel = app.window.recording
    assert panel.auto_compare_var.get() is False and panel.auto_compare_scale_var.get() == 0.0
    assert panel.auto_compare_hint_var.get() == "仅录制落盘，不自动比对"
    assert app.flags.auto_compare is False
    panel.auto_compare_var.set(True)   # 变量写入同步滑块
    assert panel.auto_compare_scale_var.get() == 1.0 and app.flags.auto_compare is True


def test_exam_preflight_rejects_skeleton_session(root):
    app = build_app(root)
    app.flags.dual_active = app.flags.recording_ready = True
    app.flags.record_skeleton = True
    problems = app.exam.exam_preflight()
    assert problems and "骨架" in problems[0]


# ---------------------------------------------------------------- D-18 第 8 层：开录确认与结束写盘（D-27）
def test_stopping_disables_record_button_until_drained(root, tmp_path, monkeypatch):
    """结束不等写完：stopping 期间按钮显示"正在结束…"且不可用，点击无效；写完后的 tick 复位。"""
    app, _finalizer = _recording_app(root, tmp_path)
    service = app.recording.service
    state = {"value": "stopping"}
    monkeypatch.setattr(type(service), "state", property(lambda _self: state["value"]))
    calls = []
    monkeypatch.setattr(service, "toggle_pause", lambda: calls.append("toggle") or "paused")
    app.recording.sync_record_state("stopping")
    panel = app.window.recording
    assert _text(panel.record_btn) == "正在结束…" and _state(panel.record_btn) == "disabled"
    app.recording.toggle()
    assert calls == []
    state["value"] = "idle"
    app.recording.refresh_status()
    assert _text(panel.record_btn) == "开始录制" and _state(panel.record_btn) == "normal"


def test_manual_start_failure_resets_button_and_tells_the_user(root, tmp_path, monkeypatch):
    app, _finalizer = _recording_app(root, tmp_path)
    shown = []
    monkeypatch.setattr(recording_mod.messagebox, "showerror", lambda title, text, **_k: shown.append((title, text)))
    app.recording.service.begin_session({"front": (4, 3), "side": (4, 3)}, 30.0)
    segment = app.recording.begin()
    assert segment and _text(app.window.recording.record_btn) == "暂停录制"
    app.recording.service.finish(discard=True)              # 录制端在开录失败时已结束整段
    app.recording._confirmed(segment, "侧面：写盘失败", None)
    assert shown and "侧面：写盘失败" in shown[0][1]
    assert _text(app.window.recording.record_btn) == "开始录制"
    forwarded = []
    app.recording._confirmed(segment, "侧面：写盘失败", lambda seg, err: forwarded.append(err))
    assert forwarded == ["侧面：写盘失败"] and len(shown) == 1   # 工作流开录的失败由工作流提示
