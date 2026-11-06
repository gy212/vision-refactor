"""公共规则在动作编排、执行和保存结果之间保持同一份业务定义。"""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from core.evaluation.action_catalog import ACTION_DEFINITIONS, ACTIONS, action_definition
from core.evaluation.rules.engine import evaluate_rules
from core.evaluation.rules.catalog_measurements import measure_catalog
from core.evaluation.rules.qualification import apply_rule_qualification
from core.evaluation.rules.sanda_v5 import FeedbackConfig, build_checks, rule_profile
from core.evaluation.rules.spec import RuleContext
from core.evaluation.rules.straight_measures import measure_straight
from core.evaluation.rules.straight_policy import apply_straight_availability
from tests.test_step_straight_combo import _base_pose


def _elbow_outcome(action: str):
    cfg = FeedbackConfig()
    profile = rule_profile(action, "left", cfg)
    rule = next(rule for rule in profile.rules if rule.code == "stance:start:front:stance_elbow")
    lm = np.repeat(_base_pose()[None], 3, axis=0)
    angle = np.deg2rad(85)
    lm[:, 11, :2], lm[:, 13, :2] = (0.5, 0.3), (0.5, 0.5)
    lm[:, 15, :2] = (0.5 + 0.2 * np.sin(angle), 0.5 - 0.2 * np.cos(angle))
    ctx = RuleContext(
        view="side",
        action=action,
        stance="left",
        params=cfg,
        baseline=np.arange(3),
        phase_windows={rule.scope: np.arange(3)},
    )
    return evaluate_rules(
        lm, np.ones(lm.shape[:2], bool), replace(profile, rules=(rule,)), context=ctx
    )[0]


@pytest.mark.parametrize("stance", ["left", "right"])
@pytest.mark.parametrize("model", ["full", "heavy"])
def test_all_action_checks_save_the_same_definition_used_for_judging(stance, model):
    cfg = FeedbackConfig(pose_variant=model)
    for action in ACTIONS:
        checks = {check["id"]: check for check in build_checks(action, stance, pose_variant=model)}
        rules = rule_profile(action, stance, cfg).rules
        assert set(checks) == {rule.code for rule in rules}
        for rule in rules:
            check = checks[rule.code]
            assert check["strategy"] == rule.strategy
            assert check["name"] == rule.title
            assert check["source"] == rule.source
            assert check.get("primaryView", "any") == rule.view
            assert check["requiredLandmarks"] == list(rule.required)
            assert check["highlightJoints"] == list(rule.highlight_joints)
            assert check["requiredViews"] == list(rule.required_views)
            assert check["evidenceJointsByView"] == {
                view: list(joints) for view, joints in rule.evidence_joints_by_view.items()
            }


def test_identical_standard_is_reused_by_stance_single_punch_and_combination():
    outcomes = [_elbow_outcome(action) for action in ("stance", "rear_straight", "straight_combo")]
    assert {outcome.state for outcome in outcomes} == {"violation"}
    assert len({outcome.spec.strategy for outcome in outcomes}) == 1
    assert all(outcome.measurements == outcomes[0].measurements for outcome in outcomes)


def test_timing_and_scoring_choices_cannot_change_rule_definitions(monkeypatch):
    before = build_checks("rear_straight", "left")
    definition = action_definition("rear_straight")
    monkeypatch.setitem(
        ACTION_DEFINITIONS,
        "rear_straight",
        replace(definition, timing="side_combo", scoring="per_repetition", rule_input="whole_clip"),
    )
    assert build_checks("rear_straight", "left") == before
    assert _elbow_outcome("rear_straight").state == "violation"


def test_explicit_standard_can_be_reused_without_changing_action_timing(monkeypatch):
    original = action_definition("rear_straight")
    assert _elbow_outcome("rear_straight").state == "violation"
    monkeypatch.setitem(
        ACTION_DEFINITIONS, "rear_straight", replace(original, rule_standard="teacher_straight")
    )
    assert _elbow_outcome("rear_straight").state == "pending_rule"
    assert _elbow_outcome("step_straight_combo").state == "pass"
    checks = build_checks("rear_straight", "left")
    assert (
        next(c for c in checks if c["code"] == "stance_elbow")["parameterOverrides"][
            "stance_elbow_min"
        ]
        == 80
    )
    assert {c["availability"]["state"] for c in checks} == {"validation_pending"}
    assert action_definition("rear_straight").timing == original.timing
    assert action_definition("rear_straight").scoring == original.scoring


@pytest.mark.parametrize(
    "qualification", ["legacy_catalog", "step_straight_validation", "unregistered"]
)
def test_front_punch_reuses_only_its_members_and_does_not_inherit_combo_qualification(
    monkeypatch, qualification
):
    original = action_definition("front_straight")
    monkeypatch.setitem(
        ACTION_DEFINITIONS,
        "front_straight",
        replace(original, rule_standard="teacher_straight", qualification=qualification),
    )
    checks = build_checks("front_straight", "left")
    assert {c["segment"] for c in checks} == {"stance", "front_straight"}
    assert not {"prepare", "recovered"} & {c["phase"] for c in checks}
    assert {c["availability"]["state"] for c in checks} == {"validation_pending"}
    assert all("无适用登记" in c["availability"]["evidence"] for c in checks)


def test_changing_qualification_keeps_standard_and_members(monkeypatch):
    before = build_checks("step_straight_combo", "left")
    monkeypatch.setitem(
        ACTION_DEFINITIONS,
        "step_straight_combo",
        replace(action_definition("step_straight_combo"), qualification="unregistered"),
    )
    after = build_checks("step_straight_combo", "left")
    state_fields = {"availability", "blockedReason"}
    assert [{k: v for k, v in c.items() if k not in state_fields} for c in before] == [
        {k: v for k, v in c.items() if k not in state_fields} for c in after
    ]
    assert {c["availability"]["state"] for c in after} == {"validation_pending"}


def test_unregistered_binding_cannot_claim_the_legacy_execution_exemption():
    checks = build_checks("front_straight", "left")
    apply_rule_qualification(checks, "new_straight", "catalog", "legacy_catalog", "full")
    assert {c["availability"]["state"] for c in checks} == {"validation_pending", "unsupported"}
    assert all("无适用登记" in c["availability"]["evidence"] for c in checks)


def test_qualification_changes_only_execution_state_and_keeps_all_definitions():
    checks = build_checks("step_straight_combo", "right", pose_variant="full")
    originals = deepcopy(checks)
    apply_straight_availability(checks, "heavy")
    assert len(checks) == len(originals)
    for old, new in zip(originals, checks):
        assert {k: v for k, v in old.items() if k not in {"availability", "blockedReason"}} == {
            k: v for k, v in new.items() if k not in {"availability", "blockedReason"}
        }
    target = "front_straight:finish:rear:guard_back"
    assert next(c for c in originals if c["id"] == target)["availability"]["state"] == "trial"
    assert next(c for c in checks if c["id"] == target)["availability"]["state"] == "enabled"


def test_paired_measurement_declares_its_reference_view_separately_from_display_view():
    rules = rule_profile("step_straight_combo", "left", FeedbackConfig()).rules
    width = next(rule for rule in rules if rule.code == "stance:start::stance_width")
    assert width.view == "front"
    assert width.required_views == ("front", "side")
    assert set(width.evidence_joints_by_view["front"]) == {11, 12, 29, 30, 31, 32}
    assert width.evidence_joints_by_view["side"] == (11, 12)
    elbow = next(rule for rule in rules if rule.code == "stance:start:front:stance_elbow")
    assert elbow.highlight_joints == (11, 13, 15)
    assert elbow.required_views == ("side",)


def test_shared_height_retains_the_two_existing_elbow_thresholds():
    lm = np.repeat(_base_pose()[None], 3, axis=0).astype(float)
    lm[:, 13, 1] = lm[:, 11, 1] + np.array([-0.06, 0, 0.06])
    mask = np.ones(lm.shape[:2], bool)
    cfg = FeedbackConfig()
    details = {}
    catalog_bad, catalog_valid = measure_catalog(
        "guard_elbow", "front", lm, mask, lm, mask, "left", cfg, details
    )
    teacher_bad, teacher_valid, teacher_details = measure_straight(
        "punch_preelbow",
        "front",
        lm,
        mask,
        lm,
        mask,
        RuleContext(view="side", action="step_straight_combo", stance="left", params=cfg),
    )
    np.testing.assert_allclose(details["肘高于肩"][0], [0.2, 0, -0.2], atol=1e-7)
    np.testing.assert_array_equal(details["肘高于肩"][0], teacher_details["起动前肘高于肩"][0])
    assert catalog_bad.tolist() == [True, False, False]
    assert teacher_bad.tolist() == [True, True, False]
    assert catalog_valid.all() and teacher_valid.all()


def test_shared_displacement_retains_distinct_baseline_and_facing_requirements():
    baseline = np.repeat(_base_pose()[None], 4, axis=0).astype(float)
    baseline[:, 0, 0] = baseline[:, 23, 0] + 0.01
    lm = baseline.copy()
    lm[:, 15, 0] -= 0.09
    mask = np.ones(lm.shape[:2], bool)
    cfg = FeedbackConfig()
    details = {}
    _, catalog_valid = measure_catalog(
        "guard_back", "front", lm, mask, baseline, mask, "left", cfg, details
    )
    bad, valid, measured = measure_straight(
        "punch_prepull",
        "front",
        lm,
        mask,
        baseline,
        mask,
        RuleContext(view="side", action="step_straight_combo", stance="left", params=cfg),
    )
    assert not catalog_valid.any()  # 旧目录要求明确侧身朝向，教师标准仅要求方向非零。
    assert valid.all() and bad.all()
    np.testing.assert_allclose(measured["起动前向后预摆距离"][0], 0.3, atol=1e-7)

    baseline[:, 0, 0] = baseline[:, 23, 0] + 0.06
    baseline_mask = mask.copy()
    baseline_mask[0, 15] = False
    _, catalog_valid = measure_catalog(
        "arm_swing", "front", lm, mask, baseline, baseline_mask, "left", cfg, {}
    )
    _, teacher_valid, _ = measure_straight(
        "punch_predrop",
        "front",
        lm,
        mask,
        baseline,
        baseline_mask,
        RuleContext(view="side", action="step_straight_combo", stance="left", params=cfg),
    )
    assert not catalog_valid.any()  # 旧目录要求整段基线有效，教师标准保留有效基线帧。
    assert teacher_valid.all()
