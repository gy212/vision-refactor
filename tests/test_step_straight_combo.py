# -*- coding: utf-8 -*-
"""上步前后手直拳组合（D-24、D-25）：侧面逐次切分、标定后的检查清单、练习次数与考试逐次评分。

合成序列是一个侧面左式考生（面向 +x）：起始护位 → 前手直拳 → 后手直拳 → 护位，按参数重复若干次。
"""
from __future__ import annotations

from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from core.evaluation.matching import engine
from core.evaluation.phases import project_windows
from core.evaluation.practice import evaluate_series
from core.evaluation.rule_checks import NEEDS_SIDE, judge_views, rule_series
from core.evaluation.rules.sanda_v5 import (
    ACTIONS,
    CALIBRATED,
    UNCALIBRATED,
    FeedbackConfig,
    action_config,
    action_segments,
    build_checks,
)
from core.evaluation.scoring import exam_pool
from core.evaluation.step_reps import EXPECTED_REPETITIONS, step_combo_windows

FPS = 30.0
ACTION = "step_straight_combo"
# 左式：前手是左手（15），后手是右手（16）。
FRONT_WRIST, REAR_WRIST = 15, 16


def _base_pose() -> np.ndarray:
    pose = np.zeros((33, 4), np.float32)
    pose[:, :2] = (0.5, 0.5)
    pose[:, 3] = 1.0
    pose[0, :2] = (0.56, 0.2)  # 鼻子在髋前方：面向 +x
    pose[[11, 12], :2] = (0.5, 0.3)
    pose[[13, 14], :2] = (0.53, 0.42)
    pose[[15, 16], :2] = (0.55, 0.3)
    pose[[23, 24], :2] = (0.5, 0.6)
    pose[[25, 26], :2] = (0.5, 0.75)
    pose[[27, 28], :2] = (0.5, 0.9)
    return pose


def _series(reps: int, *, seconds: float = 6.0) -> dict:
    """``reps`` 次组合：每次前手直拳 0.3 s，紧接后手直拳 0.3 s，两次之间回护位 1.2 s。"""
    times = np.arange(int(seconds * FPS)) / FPS
    punches = []
    for k in range(reps):
        t0 = 1.5 + 2.0 * k
        punches += [(FRONT_WRIST, t0, t0 + .3), (REAR_WRIST, t0 + .4, t0 + .7)]
    frames = np.repeat(_base_pose()[None], len(times), axis=0)
    for wrist, t0, t1 in punches:
        inside = (times >= t0) & (times <= t1)
        reach = .2 * (1 - np.abs(2 * (times[inside] - t0) / (t1 - t0) - 1))
        frames[inside, wrist, 0] += reach
        frames[inside, wrist - 2, 0] += reach / 2
    return rule_series(frames, range(len(times)), FPS, 1.0)


def test_two_reps_are_split_with_both_punches() -> None:
    diagnostics: dict = {}
    windows = step_combo_windows(_series(2), "left", FeedbackConfig(), diagnostics)
    assert diagnostics["reason"] == "已检测到完整动作"
    assert [r["index"] for r in diagnostics["repetitions"]] == [1, 2]
    for rep, t0 in zip(diagnostics["repetitions"], (1.5, 3.5)):
        assert rep["frontPeak"] == pytest.approx(t0 + .15, abs=.05)
        assert rep["rearPeak"] == pytest.approx(t0 + .55, abs=.05)
        assert rep["start"] < rep["frontPeak"] < rep["rearPeak"] < rep["end"]
    for key in (("front_straight", "finish"), ("front_straight", "motion"), ("rear_straight", "finish"),
                ("rear_straight", "motion"), ("stance", "start"), ("stance", "end")):
        assert key in windows


@pytest.mark.parametrize("reps", [1, 3])
def test_wrong_rep_count_is_judged_on_what_was_found(reps: int) -> None:
    diagnostics: dict = {}
    windows = step_combo_windows(_series(reps, seconds=8.0), "left", FeedbackConfig(), diagnostics)
    assert len(diagnostics["repetitions"]) == reps
    assert diagnostics["reason"] == f"检测到 {reps} 次动作（应为 {EXPECTED_REPETITIONS} 次），按检测到的次数判定"
    assert ("rear_straight", "finish") in windows


def test_no_rear_punch_is_reported() -> None:
    diagnostics: dict = {}
    step_combo_windows(_series(0), "left", FeedbackConfig(), diagnostics)
    assert diagnostics["reason"] == "未检测到后手直拳"
    assert diagnostics["repetitions"] == []


def test_front_view_shares_side_timing_and_needs_side() -> None:
    side = _series(2)
    front = rule_series(np.repeat(_base_pose()[None], len(side["times"]), axis=0), side["frames"], FPS, 1.0)
    judged = judge_views({"front": front, "side": side}, ACTION, "left", FeedbackConfig())
    assert judged["front"][1]["phaseSource"] == "side_synchronized_timing"
    assert len(judged["front"][1]["repetitions"]) == 2
    alone = judge_views({"front": front}, ACTION, "left", FeedbackConfig())
    assert alone["front"][1]["reason"] == NEEDS_SIDE


def test_project_windows_maps_runs_by_time() -> None:
    source = np.arange(10) / 10.0
    target = np.arange(20) / 20.0
    projected = project_windows({("a", "b"): np.array([2, 3, 4])}, source, target)
    assert projected[("a", "b")].tolist() == [4, 5, 6, 7, 8]
    assert project_windows({("a", "b"): np.array([9])}, source, target[:5]) == {}


def test_calibrated_checks_are_live_and_the_rest_pending() -> None:
    calibration = CALIBRATED[ACTION]
    checks = build_checks(ACTION, "left")
    live = [c for c in checks if (c["segment"], c["role"], c["code"]) in calibration.live]
    assert {(c["segment"], c["role"], c["code"]) for c in live} == set(calibration.live)
    assert all(not c["blockedReason"] for c in live)
    assert all(c["blockedReason"] for c in checks if c not in live)
    assert any(c["blockedReason"] == UNCALIBRATED for c in checks)
    assert {c["primaryView"] for c in live if c["code"] in ("guard_shoulder", "guard_elbow")} == {"front"}
    assert {c["primaryView"] for c in live if c["code"] == "shoulder_level"} == {"side"}
    # 旧动作不受标定影响。
    assert all(c["blockedReason"] != UNCALIBRATED for c in build_checks("straight_combo", "left"))
    assert "guard_shoulder" not in {c["code"] for c in build_checks("straight_combo", "left")}
    assert action_segments(ACTION) == action_segments("straight_combo")
    assert ACTIONS[ACTION] == "上步前后手直拳组合"


def test_action_config_applies_calibrated_params_only_to_its_action() -> None:
    cfg = FeedbackConfig()
    tuned = action_config(cfg, ACTION)
    assert tuned.elbow_raise == pytest.approx(-.18) and tuned.shoulder_height == pytest.approx(.17)
    assert action_config(cfg, "straight_combo") is cfg
    with pytest.raises(ValueError):
        FeedbackConfig(elbow_raise=1.5)


def test_practice_reports_repetition_count() -> None:
    side = _series(1, seconds=5.0)
    front = rule_series(np.repeat(_base_pose()[None], len(side["times"]), axis=0), side["frames"], FPS, 1.0)
    result = evaluate_series(ACTION, "left", {"front": front, "side": side})
    assert result["repetitions"]["expected"] == 2 and result["repetitions"]["detected"] == 1
    assert "检测到 1 次动作，应做 2 次" in result["summary"]
    assert result["config"]["elbow_raise"] == pytest.approx(-.18)
    assert "repetitions" not in evaluate_series("straight_combo", "left", {"front": front, "side": side})


def test_match_video_windows_matches_each_range_with_whole_video_frames(monkeypatch) -> None:
    seq = np.zeros((100, 3), np.float32)
    seen = []
    monkeypatch.setattr(engine, "template_features", lambda *a, **k: (seq, 30.0, None, "heavy", 1))

    def match(window, templates, **_kw):
        seen.append(len(window))
        return [SimpleNamespace(template=SimpleNamespace(path=Path(f"t{i}.npz")), start_frame=1, end_frame=5,
                                cost=1.0, avg_cost=.1, score=.5 + i / 10) for i in range(len(templates))]

    monkeypatch.setattr(engine, "match_templates", match)
    out = engine.match_video_windows(["a", "b"], "v.mp4", [(10, 40), (60, 200), (120, 130)],
                                     load=lambda p: SimpleNamespace(layout="pose33_v3", path=p))
    assert seen == [31, 40]  # 第二个范围截到视频末尾，第三个超出视频被跳过
    assert len(out) == 2 and [len(per) for per in out] == [2, 2]
    assert [(r.start_frame, r.end_frame) for r in out[0]] == [(11, 15), (61, 65)]
    assert [r.score for r in out[1]] == [.6, .6]


def _result(score: float, start: int, end: int) -> SimpleNamespace:
    return SimpleNamespace(score=score, start_frame=start, end_frame=end)


@pytest.fixture
def pool(monkeypatch, tmp_path):
    def no_rules():
        raise ValueError("规则配置不参与本组用例")

    monkeypatch.setattr(exam_pool, "load_feedback_config", no_rules)
    calls: dict = {"windows": [], "plain": []}

    def compare(templates, video, **_kw):
        calls["plain"].append([Path(t).stem for t in templates])
        return [_result(.4, 0, 30) for _ in templates]

    def compare_windows(templates, video, windows, **_kw):
        calls["windows"].append((Path(video).name, list(windows)))
        return [[_result(.6 + .2 * i, a + 1, b - 1) for i, (a, b) in enumerate(windows)] for _ in templates]

    def raw_series(video, **_kw):
        return SimpleNamespace(meta={"fps": FPS})

    tpls = [tmp_path / "上步前后手直拳组合.npz", tmp_path / "直拳.npz"]
    return SimpleNamespace(calls=calls, compare=compare, compare_windows=compare_windows, raw_series=raw_series,
                           tpls=tpls)


def test_exam_scores_each_rep_and_averages(monkeypatch, pool) -> None:
    monkeypatch.setattr(exam_pool, "side_repetitions", lambda series, cfg: [(1.0, 2.0), (3.0, 4.0)])
    res = exam_pool.score_exam_pools(pool.tpls, pool.tpls, Path("f.mp4"), Path("s.mp4"), compare=pool.compare,
                                     raw_series=pool.raw_series, stop_evt=Event(),
                                     compare_windows=pool.compare_windows)
    assert pool.calls["windows"] == [("f.mp4", [(15, 75), (75, 135)]), ("s.mp4", [(15, 75), (75, 135)])]
    assert pool.calls["plain"] == [["直拳"], ["直拳"]]
    step = [d for d in res.action_scores if d["name"] == "上步前后手直拳组合"]
    assert len(step) == 2
    for detail in step:
        assert detail["score"] == pytest.approx(.7)
        assert [r["score"] for r in detail["repetitions"]] == pytest.approx([.6, .8])
        assert (detail["start"], detail["end"]) == (16, 134)
        assert "repetition_note" not in detail
    assert res.front_score == pytest.approx((.7 + .4) / 2)


def test_exam_notes_incomplete_reps_and_missing_window_compare(monkeypatch, pool) -> None:
    monkeypatch.setattr(exam_pool, "side_repetitions", lambda series, cfg: [(1.0, 2.0)])
    res = exam_pool.score_exam_pools(pool.tpls[:1], [], Path("f.mp4"), Path("s.mp4"), compare=pool.compare,
                                     raw_series=pool.raw_series, stop_evt=Event(),
                                     compare_windows=pool.compare_windows)
    assert res.action_scores[0]["score"] == pytest.approx(.6)
    assert res.action_scores[0]["repetition_note"].startswith("检测到 1 次动作")

    monkeypatch.setattr(exam_pool, "side_repetitions", lambda series, cfg: [])
    res = exam_pool.score_exam_pools(pool.tpls[:1], [], Path("f.mp4"), Path("s.mp4"), compare=pool.compare,
                                     raw_series=pool.raw_series, stop_evt=Event(),
                                     compare_windows=pool.compare_windows)
    assert res.action_scores[0]["score"] == pytest.approx(.4)
    assert res.action_scores[0]["repetition_note"] == "未切出动作次数，按整段匹配一次"

    res = exam_pool.score_exam_pools(pool.tpls[:1], [], Path("f.mp4"), Path("s.mp4"), compare=pool.compare,
                                     raw_series=pool.raw_series, stop_evt=Event())
    assert res.action_scores[0]["repetition_note"] == "未接入逐次比对，按整段匹配一次"


def test_exam_rule_target_is_whole_clip_for_step_combo(monkeypatch) -> None:
    calls = []

    def fake_slice(series, **kw):
        calls.append(kw)
        return np.zeros((1, 33, 4)), {"start_frame": kw.get("start_frame", 0), "end_frame": kw.get("end_frame", 99)}

    monkeypatch.setattr(exam_pool, "slice_pose_raw_series", fake_slice)
    monkeypatch.setattr(exam_pool, "series_from_raw", lambda lm, meta: {})
    series = SimpleNamespace(meta={"fps": FPS})
    detail = {"name": "x", "start": 200, "end": 260}
    whole = exam_pool.rule_target(series, detail, ACTION, "front")
    windowed = exam_pool.rule_target(series, detail, "straight_combo", "front")
    assert calls == [{}, {"start_frame": 170, "end_frame": 290}]  # 上步组合取整段；其它动作按匹配窗口放宽 1 s
    assert (whole["start"], whole["end"]) == (0, 99) and (windowed["start"], windowed["end"]) == (170, 290)
