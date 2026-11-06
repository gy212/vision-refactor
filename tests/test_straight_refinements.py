"""直拳逐项改进的站稳取帧、独立姿势判断与提醒合并契约。"""

from copy import deepcopy

import numpy as np
import pytest

from core.evaluation.rules.sanda_v5 import FeedbackConfig, action_config
from core.evaluation.rules.spec import RuleContext
from core.evaluation.rules.straight_posture import measure_posture
from core.evaluation.stance_windows import settled_stance_windows
from core.reporting import (
    build_report_model,
    format_report,
    group_action_issues,
    render_html_report,
)
from tests.test_feedback_html_export import make_html_test_record
from tests.test_step_straight_combo import ACTION, _base_pose, _series


def test_stance_uses_first_settled_period_and_prefers_visible_points() -> None:
    series = _series(0, seconds=5)
    series["landmarks"][45:60, 16, 0] += np.arange(15) * 0.02
    windows = {("stance", "start"): np.arange(7)}
    reps = [{"start": 3.0}]
    original = deepcopy(windows)

    settled = settled_stance_windows(series, windows, reps, FeedbackConfig())
    chosen = settled[("stance", "start")]
    assert chosen[0] > 7 and chosen[-1] < 45
    # 中间的嘴部看不清时，在同一段站姿里选可见画面，不能跳到后面的预摆停顿。
    series["valid_mask"][:31, [9, 10]] = False
    clearer = settled_stance_windows(series, windows, reps, FeedbackConfig())
    assert clearer[("stance", "start")][0] >= 31
    assert clearer[("stance", "start")][-1] < 45
    np.testing.assert_array_equal(windows[("stance", "start")], original[("stance", "start")])
    assert reps == [{"start": 3.0}]


def test_stance_selection_does_not_search_for_a_correct_guard() -> None:
    series = _series(0, seconds=3)
    windows = {("stance", "start"): np.arange(7)}
    reps = [{"start": 2.0}]
    before = settled_stance_windows(series, windows, reps, FeedbackConfig())
    # 一只手持续偏低，另一只手已抬起。仍应检查同一段站姿，不寻找手更高的时刻。
    series["landmarks"][:, 16, 1] = 0.55
    after = settled_stance_windows(series, windows, reps, FeedbackConfig())
    np.testing.assert_array_equal(before[("stance", "start")], after[("stance", "start")])
    # 全部点看不到时，不能拿旧窗口冒充已站稳。
    series["valid_mask"][:] = False
    assert settled_stance_windows(series, windows, reps, FeedbackConfig()) == {}


def test_wrist_height_checks_shoulder_and_does_not_require_nose() -> None:
    lm = np.repeat(_base_pose()[None], 3, axis=0)
    lm[:, 15, 1] = [0.2, 0.3, 0.4]
    mask = np.ones(lm.shape[:2], bool)
    mask[:, 0] = False
    ctx = RuleContext(view="front", action=ACTION, stance="left", params=FeedbackConfig())

    bad, valid, _ = measure_posture("stance_height", "front", lm, mask, ctx)

    assert bad.tolist() == [False, False, True]
    assert valid.all()
    mask[:, 15] = False
    assert not measure_posture("stance_height", "front", lm, mask, ctx)[1].any()


def test_straight_arm_does_not_automatically_mean_shoulder_height() -> None:
    lm = np.repeat(_base_pose()[None], 2, axis=0)
    lm[0, 13, :2], lm[0, 15, :2] = (0.625, 0.425), (0.75, 0.55)
    lm[1, 13, :2], lm[1, 15, :2] = (0.625, 0.35), (0.75, 0.3)
    mask = np.ones(lm.shape[:2], bool)
    ctx = RuleContext(
        view="side", action=ACTION, stance="left", params=action_config(FeedbackConfig(), ACTION)
    )

    extension, valid, _ = measure_posture("punch_extension", "front", lm, mask, ctx)
    height, height_valid, _ = measure_posture("shoulder_level", "front", lm, mask, ctx)

    assert valid.all() and height_valid.all()
    assert extension.tolist() == [False, True]
    assert height.tolist() == [True, False]
    mask[:, 13] = False
    assert not measure_posture("punch_extension", "front", lm, mask, ctx)[1].any()
    assert measure_posture("shoulder_level", "front", lm, mask, ctx)[1].all()


def test_grouped_reminders_keep_evidence_and_each_phase_verdict() -> None:
    record = make_html_test_record(action=ACTION)
    check = record["result"]["checks"][0]
    check.update(code="guard_elbow", phase="motion", phaseLabel="出拳中", review="confirmed")
    finish = {
        **deepcopy(check),
        "id": "finish",
        "phase": "finish",
        "phaseLabel": "打到头时",
        "review": "pending",
        "evidence": ["side:20"],
    }
    record["result"]["checks"] = [check, finish]
    model = build_report_model(record)
    original = deepcopy(model.all_checks)

    grouped = group_action_issues(model.candidate_checks, ACTION)

    assert len(grouped) == 1
    assert grouped[0].evidence_refs == ["front:10", "side:20"]
    assert grouped[0].review == "pending" and grouped[0].review_display == ""
    assert "出拳中" in grouped[0].phase_label and "打到头时" in grouped[0].phase_label
    assert model.all_checks == original
    assert model.counts.candidate_count == 2
    assert render_html_report(record).count('class="issue-card"') == 1
    # 结束时已恢复，就只提醒过程，不能复制过程的结论。
    finish["status"] = "not_observed"
    model = build_report_model(record)
    assert len(group_action_issues(model.candidate_checks, ACTION)) == 1
    assert model.all_checks[1].status == "not_observed"
    assert len(group_action_issues(original, "straight_combo")) == 2


def test_unrecognized_combo_keeps_its_own_last_visible_stance() -> None:
    series = _series(1, seconds=5)
    # 另一面能看到更晚的站姿，本面从第130张起看不清。
    series["valid_mask"][130:] = False
    series["landmarks"][80:130, 15, 1] = 0.42
    windows = {("stance", "start"): np.arange(7), ("stance", "end"): np.arange(137, 144)}
    repetitions = []

    revised = settled_stance_windows(series, windows, repetitions, FeedbackConfig())

    ending = revised[("stance", "end")]
    assert ending[0] >= 80 and ending[-1] < 130
    lm, mask = series["landmarks"][ending], series["valid_mask"][ending]
    ctx = RuleContext(view="front", action=ACTION, stance="left", params=FeedbackConfig())
    bad, valid, _ = measure_posture("stance_height", "front", lm, mask, ctx)
    assert bad.all() and valid.all()
    assert repetitions == []


def test_end_wrist_height_is_enabled_without_copying_start_qualification() -> None:
    from core.evaluation import build_checks

    for model in ("full", "heavy"):
        checks = {c["id"]: c for c in build_checks(ACTION, "left", pose_variant=model)}
        assert checks["stance:end:front:stance_height"]["availability"]["state"] == "enabled"
        assert checks["stance:end:front:stance_height"]["primaryView"] == "front"
        assert checks["stance:start:front:stance_height"]["availability"]["state"] == "trial"


def test_guard_return_is_not_back_pull_and_body_tilt_does_not_change_verdict() -> None:
    from core.evaluation.rules.straight_motion import measure_guard_back

    baseline = np.repeat(_base_pose()[None], 3, axis=0)
    baseline[:, 0, 0] = 0.8
    baseline[:, 14, :2] = (0.8, 0.5)
    lm = baseline.copy()
    # 前两张收回到身旁。最后一张肘才明显退到身体后面。
    lm[:, 14, :2] = [(0.5, 0.5), (0.55, 0.5), (0.4, 0.5)]
    mask = np.ones(lm.shape[:2], bool)
    ctx = RuleContext(view="side", action=ACTION, stance="left", params=FeedbackConfig())
    bad, valid, _ = measure_guard_back("rear", lm, mask, baseline, mask, ctx)
    assert bad.tolist() == [False, False, True]
    assert valid.all()

    angle = np.deg2rad(20)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    tilted = lm.copy()
    tilted[:, :, :2] = lm[:, :, :2] @ rotation.T + (0.3, -0.1)
    moved = measure_guard_back("rear", tilted, mask, baseline, mask, ctx)
    np.testing.assert_array_equal(moved[0], bad)
    assert moved[1].all()
    mask[:, 14] = False
    assert not measure_guard_back("rear", lm, mask, baseline, mask, ctx)[1].any()


def test_guard_back_is_enabled_only_for_verified_model_and_phase() -> None:
    from core.evaluation import build_checks

    for model in ("full", "heavy"):
        checks = {c["id"]: c for c in build_checks(ACTION, "left", pose_variant=model)}
        motion = checks["front_straight:motion:rear:guard_back"]
        finish = checks["front_straight:finish:rear:guard_back"]
        assert motion["availability"]["state"] == "enabled"
        assert motion["primaryView"] == "side"
        assert (finish["availability"]["state"] == "enabled") == (model == "heavy")
        assert checks["rear_straight:motion:front:guard_back"]["availability"]["state"] == "trial"
        assert checks["rear_straight:motion:front:guard_back"]["primaryView"] == "front"


def test_punch_extension_checks_each_punch_at_its_fullest_reach() -> None:
    from core.evaluation.rules.straight_posture import measure_extension_at_peak

    lm = np.repeat(_base_pose()[None], 6, axis=0)
    lm[:, 11, :2], lm[:, 13, :2], lm[:, 15, :2] = (0.4, 0.3), (0.6, 0.3), (0.6, 0.5)
    lm[1, 15, :2] = (0.8, 0.3)
    mask = np.ones(lm.shape[:2], bool)
    ctx = RuleContext(
        view="side",
        action=ACTION,
        stance="left",
        params=action_config(FeedbackConfig(), ACTION),
        phase_windows={"front_straight:finish": np.array([10, 11, 12, 30, 31, 32])},
    )
    bad, valid, _ = measure_extension_at_peak("front", lm, mask, ctx)
    assert bad.tolist() == [False, False, False, True, True, True]
    assert valid.all()
    mask[1, 13] = False
    bad, valid, _ = measure_extension_at_peak("front", lm, mask, ctx)
    assert bad[valid].all()


def test_front_view_can_check_guard_elbow_and_missing_points_are_not_passes() -> None:
    from core.evaluation.rules.straight_motion import measure_guard_back

    lm = np.repeat(_base_pose()[None], 3, axis=0)
    lm[:, 11, :3], lm[:, 12, :3] = (0.4, 0.3, 0), (0.6, 0.3, 0)
    lm[:, 23, :3], lm[:, 24, :3] = (0.4, 0.6, 0), (0.6, 0.6, 0)
    lm[:, 0, :3] = (0.5, 0.15, -0.1)
    lm[:, 13, :3] = (0.4, 0.5, -0.1)
    baseline = lm.copy()
    lm[2, 13, 2] = 0.1
    mask = np.ones(lm.shape[:2], bool)
    ctx = RuleContext(view="front", action=ACTION, stance="left", params=FeedbackConfig())
    bad, valid, _ = measure_guard_back("front", lm, mask, baseline, mask, ctx)
    assert bad.tolist() == [False, False, True]
    assert valid.all()
    mask[:, 13] = False
    assert not measure_guard_back("front", lm, mask, baseline, mask, ctx)[1].any()


def test_trial_checks_retain_visible_trial_label_in_saved_reports() -> None:
    record = make_html_test_record(action=ACTION)
    check = record["result"]["checks"][0]
    check["availability"] = {"state": "trial", "label": "试用中", "reason": "可能报错或漏查。"}
    model = build_report_model(record)
    trial = next(c for c in model.all_checks if c.id == check["id"])
    assert trial.status_display.startswith("试用")
    assert trial.reason.startswith("试用检查。")
    assert "试用检查" in render_html_report(record)


@pytest.mark.parametrize("status", ["candidate", "unable", "not_observed"])
def test_text_report_preserves_trial_label_and_reason_for_each_result(status) -> None:
    record = make_html_test_record(action=ACTION)
    check = record["result"]["checks"][0]
    check["status"] = status
    check["availability"] = {"state": "trial", "label": "试用中", "reason": "可能报错或漏查。"}
    record["result"].update(
        summary="分析完成", fusionMethod="action_rule_quality_weighted_v1", diagnostics={}
    )
    original = deepcopy(record)

    text = format_report(record)

    assert check["name"] + "（试用）" in text
    assert check["availability"]["reason"] in text
    assert record == original


@pytest.mark.parametrize(
    ("code", "angles"),
    [
        ("stance_elbow", [100, 85, 65]),
        ("stance_knee", [165, 172, 178]),
        ("shoulder_level", [90, 102, 112]),
    ],
)
def test_straight_tolerance_accepts_small_deviation_and_keeps_clear_errors(code, angles) -> None:
    from core.evaluation.rules.sanda_v5 import rule_profile

    lm = np.repeat(_base_pose()[None], 3, axis=0)
    angles = np.deg2rad(angles)
    if code == "stance_elbow":
        lm[:, 11, :2], lm[:, 13, :2] = (0.5, 0.3), (0.5, 0.5)
        lm[:, 15, :2] = np.column_stack((0.5 + 0.2 * np.sin(angles), 0.5 - 0.2 * np.cos(angles)))
    elif code == "stance_knee":
        for hip, knee, ankle in ((23, 25, 27), (24, 26, 28)):
            lm[:, hip, :2], lm[:, knee, :2] = (0.5, 0.3), (0.5, 0.5)
            lm[:, ankle, :2] = np.column_stack(
                (0.5 + 0.2 * np.sin(angles), 0.5 - 0.2 * np.cos(angles))
            )
    else:
        lm[:, 11, :2], lm[:, 23, :2] = (0.5, 0.3), (0.5, 0.6)
        lm[:, 15, :2] = np.column_stack((0.5 + 0.2 * np.sin(angles), 0.3 + 0.2 * np.cos(angles)))
    mask = np.ones(lm.shape[:2], bool)
    cfg = FeedbackConfig()
    profile = rule_profile(ACTION, "left", cfg)
    rule = next(s for s in profile.rules if s.code.endswith(":" + code))
    ctx = RuleContext(view="side", action=ACTION, stance="left", params=cfg)
    bad, valid, _ = rule.check(lm, mask, lm, mask, ctx)
    assert valid.all()
    assert bad.tolist() == [False, False, True]
    mask[:, rule.required[0]] = False
    assert not rule.check(lm, mask, lm, mask, ctx)[1].any()
    # 新余量仅用于上步直拳；其他动作继续用原设置。
    other = action_config(cfg, "straight_combo")
    assert (other.stance_elbow_min, other.stance_knee_max, other.punch_level_deviation) == (
        90,
        170,
        5,
    )
