"""规则边界、证据门、历史生命周期与真实Tk入口的针对性回归。"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import pytest

from core import action_feedback as af
from core import feedback_geometry as fg
from core.feedback_history import FeedbackHistory, validate_identity


def pose_series(n=32):
    lm = np.zeros((n, 33, 4), dtype=np.float32)
    lm[:, :, 3] = 1
    points = {0:(.5,.15), 9:(.47,.2), 10:(.53,.2), 11:(.4,.3), 12:(.6,.3),
              13:(.4,.45), 14:(.6,.45), 15:(.55,.45), 16:(.75,.45),
              23:(.43,.6), 24:(.57,.6), 25:(.4,.78), 26:(.6,.78),
              27:(.35,.96), 28:(.65,.96), 29:(.38,.98), 30:(.68,.98),
              31:(.3,.96), 32:(.6,.96)}
    for index, xy in points.items():
        lm[:, index, :2] = xy
    return {"landmarks": lm, "valid_mask": np.ones((n,33), bool),
            "frames": list(range(n)), "times": [i/12 for i in range(n)]}


def result(action="front_straight", stance="left"):
    return fg.evaluate_series(action, stance, {"front": pose_series(), "side": pose_series()})


def fake_analyzer(action, stance, front, side, **kwargs):
    return result(action, stance)


def video(path: Path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (96, 96))
    assert writer.isOpened()
    for i in range(16):
        writer.write(np.full((96, 96, 3), i * 10, dtype=np.uint8))
    writer.release()
    return path


@pytest.mark.parametrize("action", list(af.ACTIONS))
def test_catalog_covers_phases_sides_sources_and_no_scores(action):
    checks = af.build_checks(action, "left")
    assert len({c["id"] for c in checks}) == len(checks)
    assert checks[0]["phase"] == "start" and checks[-1]["phase"] == "end"
    assert all(c["source"] and c["standard"] and c["bodyPart"] for c in checks)
    assert all("score" not in c and "penalty" not in c for c in checks)
    right = af.build_checks(action, "right")
    assert next(c for c in checks if c["code"] == "stance_elbow")["bodyPart"].endswith("（左）")
    assert next(c for c in right if c["code"] == "stance_elbow")["bodyPart"].endswith("（右）")
    if action.endswith("combo"):
        segments = list(dict.fromkeys(c["segment"] for c in checks if c["phase"] in {"motion", "finish"}))
        assert segments[0].startswith("front_") and segments[1].startswith("rear_")


def test_rules_use_selected_front_arm_and_keep_uncalibrated_checks_pending():
    series = pose_series()
    series["landmarks"][:, 15, :2] = (.4, .6)  # 左肘180度；右肘90度。
    left = fg.evaluate_series("stance", "left", {"front": series, "side": series})
    right = fg.evaluate_series("stance", "right", {"front": series, "side": series})
    get = lambda data, code: next(c for c in data["checks"] if c["code"] == code)
    assert get(left, "stance_elbow")["status"] == "candidate"
    assert get(right, "stance_elbow")["status"] == "not_observed"
    assert get(left, "stance_height")["status"] == "pending_rule"
    assert get(left, "stance_width")["status"] == "pending_rule"
    assert left["backend"] == "mediapipe" and left["delegate"] == "cpu"
    assert not left["scoreAuthorized"]


def test_missing_landmarks_or_view_or_motion_are_not_a_pass():
    series = pose_series()
    series["valid_mask"][:] = False
    data = fg.evaluate_series("front_straight", "left", {"front": series})
    assert data["needsRerecord"]
    assert all(c["status"] in {"unable", "pending_rule"} for c in data["checks"])
    data = result()
    assert all(c["status"] in {"unable", "pending_rule"} for c in data["checks"] if c["phase"] == "motion")


def _check(data, code="stance_elbow", phase="start"):
    return next(c for c in data["checks"] if c["code"] == code and c["phase"] == phase)


def test_fusion_prefers_side_for_elbow_and_preserves_both_views():
    good, bad = pose_series(), pose_series()
    bad["landmarks"][:, 15, :2] = (.4, .6)
    data = fg.evaluate_series("stance", "left", {"front": bad, "side": good})
    check = _check(data)
    assert check["status"] == "not_observed"  # 正面投影异常不推翻更适用的侧面。
    assert check["fusion"]["support"] == pytest.approx(.2)
    assert {m["view"] for m in check["measurements"]} == {"front", "side"}
    assert {e["view"] for e in data["evidence"]} == {"front", "side"}
    reverse = fg.evaluate_series("stance", "left", {"front": good, "side": bad})
    assert _check(reverse)["status"] == "candidate"
    assert _check(reverse)["fusion"]["support"] == pytest.approx(.8)


def test_missing_or_occluded_view_is_not_promoted_to_full_weight():
    good, hidden = pose_series(), pose_series()
    hidden["valid_mask"][:, 13] = False
    for views in ({"front": good}, {"front": good, "side": hidden}):
        check = _check(fg.evaluate_series("stance", "left", views))
        assert check["status"] == "unable"
        assert check["fusion"]["support"] is None
        assert not check["fusion"]["effectiveWeights"]
        assert check["measurements"]  # 保留可用单侧观测给教师。


def test_quality_reduces_weight_and_equal_conflict_stays_undecided(monkeypatch):
    front, side = pose_series(), pose_series()
    side["landmarks"][:, 15, :2] = (.4, .6)
    side["valid_mask"][[1, 3, 5, 7], 13] = False
    monkeypatch.setattr(fg, "phase_windows", lambda *args: {("stance", "start"): np.arange(10)})
    weights = fg._default_front_weights()
    weights["stance"]["stance_elbow"] = .3
    cfg = fg.FeedbackConfig(front_weights=weights)
    check = _check(fg.evaluate_series("stance", "left", {"front": front, "side": side}, cfg))
    assert check["status"] == "unable"
    assert check["fusion"]["effectiveWeights"]["side"] == pytest.approx(.7 * .6 / (.3 + .7 * .6))
    assert "接近" in check["reason"]
    weights["stance"]["stance_elbow"] = .5
    side["valid_mask"][:] = True
    check = _check(fg.evaluate_series("stance", "left", {"front": front, "side": side}, cfg))
    assert check["status"] == "unable" and check["fusion"]["support"] == .5


def test_different_action_stages_are_not_fused():
    front, side = pose_series(), pose_series()
    side["times"] = [t + 2 for t in side["times"]]
    check = _check(fg.evaluate_series("stance", "left", {"front": front, "side": side}))
    assert check["status"] == "unable" and "时间不对应" in check["reason"]


def test_action_profiles_and_primary_view_limit_are_explicit():
    cfg = fg.FeedbackConfig()
    values = [fg._view_weights({"segment": action, "code": "guard_low"}, cfg)["front"]
              for action in ("front_straight", "front_hook", "front_low_kick")]
    assert values == [.7, .75, .65]
    check = _check(result("stance"), "stance_torso")
    assert check["fusion"]["method"] == "primary_view"
    assert check["fusion"]["baseWeights"] == {"front": 1, "side": 0}
    assert "不能称为双视角一致" in af.format_fusion(check)


def test_weight_configuration_validation_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(fg, "repo_root", lambda: tmp_path)
    weights = fg._default_front_weights()
    weights["stance"]["stance_elbow"] = .25
    path = tmp_path / "feedback_config.json"
    path.write_text(json.dumps({"front_weights": weights}), encoding="utf-8")
    assert fg.load_feedback_config().front_weights["stance"]["stance_elbow"] == .25
    for value in (0, 1, float("nan"), True):
        weights["stance"]["stance_elbow"] = value
        with pytest.raises(ValueError):
            fg.FeedbackConfig(front_weights=weights)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="配置无效"):
        fg.load_feedback_config()
    data = result("stance")
    record = {"result": data, "studentId": "001", "studentName": "", "createdAt": "now", "expiresAt": "later"}
    report = af.format_report(record)
    assert "实际权重" in report and "已检查未发现的问题项" in report
    assert "非成绩、非准确率" in report


def test_same_video_cannot_be_used_as_both_cameras(tmp_path):
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"fixture")
    store = FeedbackHistory(tmp_path / "history")
    with pytest.raises(ValueError, match="同一个视频"):
        store.add(validate_identity("001", "", "stance", "left"), source, source)
    assert list(store.root.iterdir()) == []


def test_movement_windows_require_return_and_keep_front_rear_order():
    series = pose_series(40)
    # 每只手一次明确离开并回收：起峰后回到初始护位。
    for start, wrist in ((7,15),(23,16)):
        series["landmarks"][start:start+7, wrist, 0] += np.array([.10,.16,.22,.28,.22,.16,.10])
    windows = fg.phase_windows(series, "straight_combo", "left", fg.FeedbackConfig())
    assert windows[("front_straight","finish")][1] == 10
    assert windows[("rear_straight","finish")][1] == 26
    assert fg.phase_windows(series, "straight_combo", "right", fg.FeedbackConfig()).keys() == {("stance","start"),("stance","end")}
    series["valid_mask"][10,15] = False  # 短暂漏点保留原无效标记，可以识别前后已观测到的运动。
    assert ("front_straight","motion") in fg.phase_windows(series,"straight_combo","left",fg.FeedbackConfig())
    assert not series["valid_mask"][10,15]
    series["valid_mask"][9:13,15] = False  # 长遮挡不能拼接成完整动作。
    assert ("front_straight","motion") not in fg.phase_windows(series,"straight_combo","left",fg.FeedbackConfig())


def test_actual_video_sampling_uses_pipeline_and_does_not_repeat_lost_pose(tmp_path):
    from types import SimpleNamespace
    path = video(tmp_path / "front.avi")
    calls = []
    closed = []

    class Pipeline:
        def infer(self, frame, *, timestamp_ms):
            calls.append(timestamp_ms)
            if len(calls) == 2:
                return None, []
            return [SimpleNamespace(x=.5,y=.5,z=0,visibility=1) for _ in range(33)], []
        def close(self):
            closed.append(True)

    data = fg.extract_series(path, fg.FeedbackConfig(), lambda: False, Pipeline)
    assert data["frames"][0] == 0 and data["frames"][-1] == 15
    assert data["valid_mask"][0].all() and not data["valid_mask"][1].any()
    assert calls == sorted(set(calls)) and closed == [True]
    with pytest.raises(InterruptedError):
        af.analyze_feedback("stance", "left", path, None, stopped=lambda: True, pipeline_factory=Pipeline)


@pytest.mark.parametrize("variant", ["lite", "full", "heavy"])
def test_feedback_pipeline_uses_selected_cpu_model_without_hands(monkeypatch, tmp_path, variant):
    (tmp_path / f"pose_landmarker_{variant}.task").write_bytes(b"fixture")
    monkeypatch.setattr(fg, "models_dir", lambda: tmp_path)
    captured = []
    monkeypatch.setattr(fg, "MediaPipePipeline", lambda **kw: captured.append(kw))
    fg._pipeline(fg.FeedbackConfig(pose_variant=variant))
    config = captured[0]["cfg"]
    assert config.pose_variant == variant and config.delegate == "cpu" and not config.enable_hands
    assert fg.FeedbackConfig().pose_variant == "full"


def test_completed_hip_turn_and_arm_swing_are_not_missing_actions():
    series = pose_series(32)
    series["landmarks"][7:14, 27, 0] += np.array([.10,.16,.22,.28,.22,.16,.10])
    series["landmarks"][10, 15, 1] += .2  # 高峰处已甩手
    series["landmarks"][10, 24, 2] += .1  # 高峰处已翻胯
    data = fg.evaluate_series("front_low_kick", "left", {"front": series, "side": series})
    checked = [c for c in data["checks"] if c["code"] in {"hip_turn", "arm_swing"}]
    assert checked and all(c["status"] == "not_observed" for c in checked)


def test_personal_history_review_export_reanalysis_and_retention(tmp_path):
    now = [datetime(2026, 9, 18, tzinfo=timezone.utc)]
    store = FeedbackHistory(tmp_path / "history", clock=lambda: now[0])
    source = tmp_path / "source.avi"
    source.write_bytes(b"mock video, inference is injected")
    identity = validate_identity("001", "张同学", "front_straight", "left")
    saved = store.add(identity, source, None, analyzer=fake_analyzer)
    assert store.list("002") == [] and len(store.list("001")) == 1
    issue = next(c for c in saved["result"]["checks"] if c["status"] == "candidate")
    changed = store.review(saved["id"], issue["id"], "confirmed", "李老师", "已核对")
    assert changed["result"]["reviewHistory"][-1]["before"] == "pending"
    out = store.export(saved["id"], tmp_path / "报告.txt")
    assert "教师已确认" in out.read_text(encoding="utf-8-sig")
    store.review(saved["id"], issue["id"], "revoked", "李老师", "误报")
    assert "教师已确认" not in af.format_report(store.get(saved["id"]))
    now[0] += timedelta(days=1)
    updated = store.reanalyze(saved["id"], analyzer=fake_analyzer)
    assert updated["expiresAt"] == saved["expiresAt"]
    assert len(updated["previousResults"]) == 1
    assert all(c["review"] == "pending" for c in updated["result"]["checks"])
    now[0] += timedelta(days=13)
    assert store.list("001") == []
    assert source.is_file() and not (store.root / saved["id"]).exists()


def test_cancel_failure_and_invalid_path_never_publish_history(tmp_path):
    store = FeedbackHistory(tmp_path / "history")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"mock")
    identity = validate_identity("001", "", "stance", "right")
    with pytest.raises(InterruptedError):
        store.add(identity, source, None, stopped=lambda: True)
    assert store.list("001") == [] and list(store.root.iterdir()) == []
    with pytest.raises(ValueError):
        store.get("../source")
    with pytest.raises(ValueError):
        validate_identity("../other", "", "stance", "left")
    saved = store.add(identity, source, None, analyzer=fake_analyzer)
    with pytest.raises(ValueError):
        store.export(saved["id"], store.root / "report.txt")
    (store.root / saved["id"] / saved["videos"]["front"]).unlink()
    with pytest.raises(ValueError, match="原视频"):
        store.reanalyze(saved["id"], analyzer=fake_analyzer)


def test_real_tk_panel_background_analysis_history_and_identity_switch(tmp_path):
    import tkinter as tk
    from apps.feedback_panel import FeedbackControls

    root = tk.Tk()
    root.withdraw()
    messages, statuses = [], []
    busy = []
    controls = FeedbackControls(root, root=root, set_busy=busy.append, show_result=messages.append,
                                set_status=statuses.append, can_analyze=lambda: True)
    controls.store = FeedbackHistory(tmp_path / "history")
    controls.student_id.set("001")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"mock")
    identity = controls.identity()
    try:
        controls._start(lambda: controls.store.add(identity, source, None, analyzer=fake_analyzer))
        deadline = time.monotonic() + 5
        while controls.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(.01)
        assert not controls.busy and busy == [True, False]
        assert messages and "动作问题说明" in messages[-1]
        controls.open_history()
        root.update()
        assert len(controls.history_tree.get_children()) == 1
        controls.history_tree.selection_set(controls.current["id"])
        root.update()
        assert controls.check_tree.get_children()
        controls.student_id.set("002")
        assert controls.current is None and str(controls.export_button.cget("state")) == "disabled"
    finally:
        controls.close()
        root.destroy()


def test_late_cancel_does_not_write_or_replace_result(tmp_path):
    store = FeedbackHistory(tmp_path / "history")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"mock")
    identity = validate_identity("001", "", "stance", "left")
    stop = threading.Event()

    def cancel_after_inference(*args, **kwargs):
        stop.set()
        return fake_analyzer(*args, **kwargs)

    with pytest.raises(InterruptedError):
        store.add(identity, source, None, analyzer=cancel_after_inference, stopped=stop.is_set)
    assert store.list("001") == []
    stop.clear()
    saved = store.add(identity, source, None, analyzer=fake_analyzer)
    with pytest.raises(InterruptedError):
        store.reanalyze(saved["id"], analyzer=cancel_after_inference, stopped=stop.is_set)
    assert store.get(saved["id"])["revision"] == 1


@pytest.mark.parametrize("waiting", [False, True])
def test_app_judges_identity_bound_at_recording_start(waiting):
    from types import SimpleNamespace
    from apps.app_ui import App

    captured = []
    identity = validate_identity("recorded_student", "", "rear_low_kick", "right")
    app = SimpleNamespace(_student_practice_active=True, _student_judging=False, _student_segment_ready=True,
        _student_pending_record=waiting,
        _rec=SimpleNamespace(state="idle"), _student_feedback_identity=identity, _student_feedback_record_id=None,
        _student_front_video=Path("front.mp4"), _student_side_video=Path("side.mp4"),
        feedback_controls=SimpleNamespace(analyze_recording=lambda *args, **kw: captured.append((args, kw))))
    App._student_judge(app)
    if waiting:
        assert captured == []
        return
    assert captured[0][0][0] == identity
    captured[0][1]["on_saved"]({"id": "saved_id"})
    assert app._student_feedback_record_id == "saved_id"


def test_corrupt_record_is_reported_without_deleting_unmanaged_files(tmp_path):
    store = FeedbackHistory(tmp_path / "history")
    folder = store.root / ("a" * 32)
    folder.mkdir()
    (folder / "record.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="格式损坏"):
        store.list("001")
    assert (folder / "record.json").is_file()


def combo_series(n=80):
    series = pose_series(n)
    series["times"] = [i / 30 for i in range(n)]
    for start, wrist in ((12, 15), (30, 16)):
        series["landmarks"][start:start + 5, wrist, 0] += [.10, .20, .28, .20, .10]
    return series


def test_fast_combo_shared_timing_keeps_independent_measurements_and_masks():
    side = combo_series()
    front = pose_series(80)  # 正面投影不明显；同步侧面提供动作时间段。
    front["times"] = side["times"][:]
    data = fg.evaluate_series("straight_combo", "left", {"front": front, "side": side})
    assert data["diagnostics"]["front"]["phaseSource"] == "side_synchronized_timing"
    check = next(c for c in data["checks"] if c["code"] == "shoulder_level" and c["segment"] == "front_straight")
    assert check["status"] in {"candidate", "not_observed"}
    assert len(check["measurements"]) == 2
    assert all(m["values"] for m in check["measurements"])
    front["valid_mask"][11:19, 15] = False
    data = fg.evaluate_series("straight_combo", "left", {"front": front, "side": side})
    check = next(c for c in data["checks"] if c["code"] == "shoulder_level" and c["segment"] == "front_straight")
    assert check["status"] == "unable"  # 时间段可共享，缺失腕点不能从侧面借。
    assert "正面" in check["reason"] and "有效帧不足" in check["reason"]


def test_occluded_far_side_does_not_block_visible_arm_but_hidden_arm_stays_unable():
    front, side = pose_series(), pose_series()
    side["valid_mask"][:, [12, 14, 16, 24]] = False
    data = fg.evaluate_series("stance", "left", {"front": front, "side": side})
    assert _check(data)["status"] == "not_observed"
    assert _check(data, "stance_rear_guard")["status"] == "unable"
    record = {"result": data, "studentId": "test", "studentName": "", "createdAt": "now", "expiresAt": "later"}
    text = af.format_report(record)
    assert "右腕0%" in text and "关键点有效比例" in text and "前手肘角：中位" in text


def test_phase_search_ignores_lost_entry_and_exit_and_keeps_repeated_combos():
    series = combo_series(100)
    series["valid_mask"][:5] = False
    series["valid_mask"][-6:] = False
    for start, wrist in ((55, 15), (73, 16)):
        series["landmarks"][start:start + 5, wrist, 0] += [.10, .20, .28, .20, .10]
    windows = fg.phase_windows(series, "straight_combo", "left", fg.FeedbackConfig())
    assert windows[("stance", "start")][0] >= 5
    assert windows[("stance", "end")][-1] < 94
    assert len(fg.contiguous_runs(windows[("front_straight", "finish")])) == 2
    assert len(fg.contiguous_runs(windows[("rear_straight", "finish")])) == 2


def test_pair_alignment_uses_multiple_peaks_and_rejects_count_disagreement():
    front, side = combo_series(), combo_series()
    side["times"] = [t + .2 for t in side["times"]]
    data = fg.evaluate_series("straight_combo", "left", {"front": front, "side": side})
    assert data["alignment"]["sideOffsetSeconds"] == pytest.approx(-.2)
    assert data["alignment"]["anchors"] == 2 and data["alignment"]["reliable"]
    for start, wrist in ((50, 15), (67, 16)):
        front["landmarks"][start:start + 5, wrist, 0] += [.10, .20, .28, .20, .10]
    data = fg.evaluate_series("straight_combo", "left", {"front": front, "side": side})
    assert not data["alignment"]["reliable"]
    assert all(c["status"] in {"unable", "pending_rule"} for c in data["checks"])


def test_incomplete_combo_is_not_overwritten_by_other_view():
    front, side = combo_series(), combo_series()
    front["landmarks"][30:35, 16, 0] = .75
    data = fg.evaluate_series("straight_combo", "left", {"front": front, "side": side})
    assert data["diagnostics"]["front"]["phaseSource"] == "own_view"
    assert "次数或顺序" in data["diagnostics"]["front"]["reason"]
    assert all(c["status"] in {"unable", "pending_rule"} for c in data["checks"] if c["phase"] == "motion")


def test_1080p_30fps_analysis_keeps_every_frame(tmp_path):
    path = tmp_path / "full-resolution.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (1920, 1080))
    assert writer.isOpened()
    try:
        for _ in range(12):
            writer.write(np.zeros((1080, 1920, 3), np.uint8))
    finally:
        writer.release()
    seen = []

    class Pipeline:
        def infer(self, frame, *, timestamp_ms):
            seen.append((frame.shape, timestamp_ms))
            return None, []
        def close(self):
            pass

    series = fg.extract_series(path, fg.FeedbackConfig(), lambda: False, Pipeline)
    assert series["frames"] == list(range(12))
    assert all(shape == (1080, 1920, 3) for shape, _ in seen)
    assert series["capture"]["sourceFps"] == 30 and series["capture"]["stride"] == 1
    assert series["capture"]["analyzedFrames"] == 12 and not series["valid_mask"].any()


def test_real_tk_evidence_frames_and_model_choice(tmp_path, monkeypatch):
    import tkinter as tk
    import apps.feedback_panel as panel
    root = tk.Tk()
    root.withdraw()
    controls = panel.FeedbackControls(root, root=root, set_busy=lambda _: None, show_result=lambda _: None,
                                      set_status=lambda _: None, can_analyze=lambda: True)
    controls.store = FeedbackHistory(tmp_path / "history")
    controls.student_id.set("evidence-test")
    controls.action.set(af.ACTIONS["stance"])
    try:
        source = video(tmp_path / "source.avi")
        side = video(tmp_path / "side.avi")
        saved = controls.store.add(controls.identity(), source, side, analyzer=fake_analyzer)
        controls.current = saved
        controls.open_history()
        root.update()
        controls.history_tree.selection_set(saved["id"])
        root.update()
        controls.check_tree.selection_set(_check(saved["result"])["id"])
        controls.show_evidence()
        root.update()
        window = next(w for w in root.winfo_children() if isinstance(w, tk.Toplevel) and w.title().startswith("证据帧"))
        assert len(window._photos) == 6
        captured = []
        monkeypatch.setattr(panel, "analyze_feedback", lambda *args, **kwargs: captured.append(kwargs["config"]))
        controls.analysis_model.set("Lite（快速对比）")
        analyze = controls._analyzer()
        controls.analysis_model.set("Full（精细分析）")
        analyze()
        assert captured[0].pose_variant == "lite"  # 主线程选择快照，不在工作线程读取Tk变量。
    finally:
        controls.close()
        root.destroy()
