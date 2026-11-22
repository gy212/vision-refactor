# -*- coding: utf-8 -*-
"""学生练习检查项全局统一开关：目录收集、跨动作复用、全关拦截、阶段计时保留与报告只读口径。"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from core import evaluation as ev
from core.evaluation import (
    analyze_practice,
    build_checks,
    catalog_checks,
    evaluate_series,
    select_practice_checks,
)
from core.evaluation.rule_checks import rule_series
from core.foundation import load_practice_disabled_checks, save_practice_disabled_checks
from core.reporting import build_report_model, format_report
from core.reporting.html import render_html_report
from tests.eval_diff import synth_practice as sp


def _punch_views(*, with_rear: bool = False) -> dict:
    fps = 30.0
    times = np.arange(int(5.0 * fps)) / fps
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[:, :2] = (0.5, 0.5)
    pose[:, 3] = 1.0
    pose[0, :2] = (0.56, 0.2)
    pose[[11, 12], :2] = (0.5, 0.3)
    pose[[13, 14], :2] = (0.53, 0.42)
    pose[[15, 16], :2] = (0.55, 0.3)
    pose[[23, 24], :2] = (0.5, 0.6)
    pose[[25, 26], :2] = (0.5, 0.75)
    pose[[27, 28], :2] = (0.5, 0.9)
    frames = np.repeat(pose[None], len(times), axis=0)
    punches = [(15, 1.5, 1.8)]
    if with_rear:
        punches.append((16, 1.9, 2.2))
    for wrist, t0, t1 in punches:
        inside = (times >= t0) & (times <= t1)
        reach = 0.2 * (1 - np.abs(2 * (times[inside] - t0) / (t1 - t0) - 1))
        frames[inside, wrist, 0] += reach
        frames[inside, wrist - 2, 0] += reach / 2
    series = rule_series(frames, range(len(times)), fps, 1.0)
    return {"front": deepcopy(series), "side": deepcopy(series)}


def _synth_views(action: str, stance: str, tmp_path) -> dict:
    cfg = ev.FeedbackConfig()
    views = {}
    for i, view in enumerate(("front", "side")):
        lm, detected = sp.landmarks(action, stance, view, 10 + i)
        video = sp.write_avi(tmp_path / f"{action}_{stance}_{view}.avi")
        views[view] = ev.extract_series(
            video, cfg, lambda: False, sp.scripted_pipeline(lm, detected)
        )
    return views


def _wrap_record(action: str, stance: str, result: dict) -> dict:
    return {
        "schemaVersion": 1,
        "id": "d" * 32,
        "studentId": "2026001",
        "studentName": "李同学",
        "action": action,
        "stance": stance,
        "createdAt": "2026-11-20T12:00:00Z",
        "expiresAt": "2026-12-04T12:00:00Z",
        "revision": 1,
        "status": "ready",
        "videos": {"front": "front.mp4", "side": "side.mp4"},
        "result": result,
    }


def test_catalog_checks_returns_151_unique_items_with_stance_bookends():
    left_catalog = catalog_checks("left")
    right_catalog = catalog_checks("right")

    assert len(left_catalog) == 151
    assert len(right_catalog) == 151
    assert len({c["id"] for c in left_catalog}) == 151
    assert [c["id"] for c in left_catalog] == [c["id"] for c in right_catalog]

    assert all(c["segment"] == "stance" and c["phase"] == "start" for c in left_catalog[:8])
    assert all(c["segment"] == "stance" and c["phase"] == "end" for c in left_catalog[-8:])
    assert all(c["segment"] != "stance" for c in left_catalog[8:-8])

    with pytest.raises(ValueError, match="请选择支持的左式/右式"):
        catalog_checks("invalid")


def test_global_disabled_checks_reuse_across_actions_and_stances(tmp_path):
    disabled = (
        "stance:start:front:stance_elbow",
        "front_straight:finish:front:punch_extension",
    )
    save_practice_disabled_checks(disabled)
    loaded = load_practice_disabled_checks()

    for action in ("front_straight", "step_straight_combo"):
        for stance in ("left", "right"):
            views = _synth_views(action, stance, tmp_path)
            result = evaluate_series(action, stance, deepcopy(views), disabled_checks=loaded)
            expected_all = build_checks(action, stance)

            returned_ids = [c["id"] for c in result["checks"]]
            assert "stance:start:front:stance_elbow" not in returned_ids
            assert "front_straight:finish:front:punch_extension" not in returned_ids
            assert len(returned_ids) == len(expected_all) - 2
            assert result["disabledCheckIds"] == [
                c["id"] for c in expected_all if c["id"] in set(disabled)
            ]
            assert result["totalCatalogChecks"] == len(expected_all)
            assert {c["status"] for c in result["checks"]} <= {
                "candidate",
                "not_observed",
                "unable",
            }


def test_analyze_practice_applies_disabled_checks_and_default_omits_new_keys(tmp_path):
    front_lm, front_det = sp.landmarks("front_straight", "left", "front", 1)
    side_lm, side_det = sp.landmarks("front_straight", "left", "side", 2)
    front_avi = sp.write_avi(tmp_path / "front.avi")
    side_avi = sp.write_avi(tmp_path / "side.avi")

    def factory():
        raise AssertionError("should pass explicit pipeline_factory per test")

    _ = factory

    def run_for(disabled_checks):
        calls = []

        def fake_extract(path, cfg, stopped, pipeline_factory=None):
            calls.append(path.name)
            if path.name == "front.avi":
                return ev.extract_series(
                    path, cfg, stopped, sp.scripted_pipeline(front_lm, front_det)
                )
            return ev.extract_series(path, cfg, stopped, sp.scripted_pipeline(side_lm, side_det))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("core.evaluation.practice.extract_series", fake_extract)
            res = analyze_practice(
                "front_straight",
                "left",
                front_avi,
                side_avi,
                config=ev.FeedbackConfig(),
                disabled_checks=disabled_checks,
            )
        assert calls == ["front.avi", "side.avi"]
        return res

    baseline = run_for(())
    assert "disabledCheckIds" not in baseline
    assert "totalCatalogChecks" not in baseline

    filtered = run_for(["stance:start:front:stance_elbow"])
    assert filtered["disabledCheckIds"] == ["stance:start:front:stance_elbow"]
    assert filtered["totalCatalogChecks"] == len(build_checks("front_straight", "left"))
    assert "stance:start:front:stance_elbow" not in [c["id"] for c in filtered["checks"]]


def test_reject_all_disabled_before_inference_and_ignore_unrelated_ids(tmp_path):
    stance_ids = [c["id"] for c in build_checks("stance", "left")]

    enabled, skipped, total = select_practice_checks(
        "stance",
        "left",
        disabled_checks=[stance_ids[0], "stale:obsolete:check:id"],
    )
    assert total == len(stance_ids)
    assert skipped == [stance_ids[0]]
    assert len(enabled) == len(stance_ids) - 1
    with pytest.raises(ValueError, match="当前动作至少需要保留 1 个开启的检查项"):
        select_practice_checks("stance", "left", disabled_checks=stance_ids)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("不应启动关键点提取")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("core.evaluation.practice.extract_series", fail_if_called)
        with pytest.raises(ValueError, match="当前动作至少需要保留 1 个开启的检查项"):
            analyze_practice(
                "stance",
                "left",
                tmp_path / "missing_front.avi",
                tmp_path / "missing_side.avi",
                config=ev.FeedbackConfig(),
                disabled_checks=stance_ids,
            )

    views = _synth_views("stance", "left", tmp_path)
    with pytest.raises(ValueError, match="当前动作至少需要保留 1 个开启的检查项"):
        evaluate_series("stance", "left", views, disabled_checks=stance_ids)

    unrelated = (
        "rear_low_kick:finish:rear:kick_height",
        "stale:obsolete:check:id",
    )
    fs_views = _synth_views("front_straight", "left", tmp_path)
    result = evaluate_series("front_straight", "left", fs_views, disabled_checks=unrelated)
    assert len(result["checks"]) == len(build_checks("front_straight", "left"))
    assert "disabledCheckIds" not in result
    assert "totalCatalogChecks" not in result


def test_phase_windows_preserved_and_reports_lock_saved_scope_counts():
    stance_only_ids = [c["id"] for c in build_checks("stance", "left")]
    views = _punch_views()
    baseline = evaluate_series("front_straight", "left", deepcopy(views))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "core.evaluation.practice.paired_torso_measurements",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("无多视角启用规则时不应计算 paired_torso_measurements")
            ),
        )
        result = evaluate_series(
            "front_straight", "left", deepcopy(views), disabled_checks=stance_only_ids
        )

    assert {c["segment"] for c in result["checks"]} == {"front_straight"}
    assert result["phaseWindows"] == baseline["phaseWindows"]
    for view in ("front", "side"):
        phases = {(pw["segment"], pw["phase"]) for pw in result["phaseWindows"][view]}
        assert ("stance", "start") in phases
        assert ("stance", "end") in phases
        assert ("front_straight", "motion") in phases
        assert ("front_straight", "finish") in phases

    base_rec = _wrap_record("front_straight", "left", baseline)
    assert "本次已按设置跳过" not in build_report_model(base_rec).headline.subtitle
    assert "检查项范围：" not in format_report(base_rec)

    combo_views = _punch_views(with_rear=True)
    combo_base = evaluate_series("step_straight_combo", "left", deepcopy(combo_views))
    combo_res = evaluate_series(
        "step_straight_combo",
        "left",
        deepcopy(combo_views),
        disabled_checks=stance_only_ids,
    )
    assert combo_res["repetitions"] == combo_base["repetitions"]

    record = _wrap_record("front_straight", "left", result)
    enabled_n = len(result["checks"])
    skipped_n = len(stance_only_ids)
    total_n = enabled_n + skipped_n

    # 后续修改全局偏好，绝不影响已保存记录的只读报告口径
    save_practice_disabled_checks(["front_straight:finish:front:punch_extension"])

    expected_note = f"本次已按设置跳过 {skipped_n} 个检查项（启用 {enabled_n}/{total_n} 项）。"
    model = build_report_model(record)
    assert model.counts.total_checks == enabled_n
    assert model.counts.verify_conservation()
    assert expected_note in model.headline.subtitle

    text_report = format_report(record)
    assert f"检查项范围：{expected_note}" in text_report

    html_report = render_html_report(record)
    assert expected_note in html_report

    # totalCatalogChecks 缺失或非法时回退为 enabled_n + skipped_n
    fallback_rec = deepcopy(record)
    fallback_rec["result"]["totalCatalogChecks"] = "invalid"
    assert expected_note in build_report_model(fallback_rec).headline.subtitle
    assert f"检查项范围：{expected_note}" in format_report(fallback_rec)
