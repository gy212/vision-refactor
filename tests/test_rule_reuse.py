"""公共规则在动作编排、执行和保存结果之间保持同一份业务定义与公共/私有边界契约。"""

from dataclasses import replace

import numpy as np
import pytest

from core.evaluation.action_catalog import (
    ACTION_DEFINITIONS,
    ACTIONS,
    action_definition,
)
from core.evaluation.rules.engine import evaluate_rules
from core.evaluation.rules.guard_rules import (
    guard_view,
    measure_guard_back,
    measure_guard_elbow,
)
from core.evaluation.rules.punch_rules import (
    measure_punch_predrop,
    measure_punch_prepull,
)
from core.evaluation.rules.rule_definitions import (
    PRIVATE_DOMAINS,
    PUBLIC_DOMAINS,
    SEGMENT_CHECK_BLUEPRINTS,
    STRATEGY_REGISTRY,
)
from core.evaluation.rules.sanda_v5 import (
    FeedbackConfig,
    build_checks,
    catalog_checks,
    rule_profile,
)
from core.evaluation.rules.spec import RuleContext
from core.evaluation.rules.straight_rules import measure_punch_preelbow
from core.evaluation.view_weights import family_of
from core.reporting import build_report_model, group_action_issues
from tests.test_feedback_html_export import make_html_test_record
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
@pytest.mark.parametrize("model", ["lite", "full", "heavy"])
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
    outcomes = [
        _elbow_outcome(action)
        for action in (
            "stance",
            "rear_straight",
            "straight_combo",
            "step_straight_combo",
            "rear_hook",
        )
    ]
    assert {outcome.state for outcome in outcomes} == {"pass"}
    assert len({outcome.spec.strategy for outcome in outcomes}) == 1
    assert all(outcome.measurements == outcomes[0].measurements for outcome in outcomes)


def test_timing_and_scoring_choices_cannot_change_rule_definitions(monkeypatch):
    before = build_checks("rear_straight", "left")
    definition = action_definition("rear_straight")
    monkeypatch.setitem(
        ACTION_DEFINITIONS,
        "rear_straight",
        replace(
            definition,
            timing="side_combo",
            scoring="per_repetition",
            rule_input="whole_clip",
        ),
    )
    assert build_checks("rear_straight", "left") == before
    assert _elbow_outcome("rear_straight").state == "pass"


def test_same_base_segment_has_identical_checks_in_every_composition():
    reference = build_checks("step_straight_combo", "left")
    ref_segments = {"stance", *action_definition("step_straight_combo").segments}
    for action in ACTIONS:
        checks = build_checks(action, "left")
        members = {"stance", *action_definition(action).segments}
        expected = [check for check in reference if check["segment"] in members]
        actual = [check for check in checks if check["segment"] in ref_segments]
        assert actual == expected
    assert _elbow_outcome("step_straight_combo").state == "pass"


def test_front_punch_contains_only_its_members_and_all_models_are_enabled_identically():
    checks = build_checks("front_straight", "left")
    assert {c["segment"] for c in checks} == {"stance", "front_straight"}
    assert not {"prepare", "recovered"} & {c["phase"] for c in checks}
    for action in ACTIONS:
        base = build_checks(action, "left", pose_variant="full")
        assert build_checks(action, "left", pose_variant="lite") == base
        assert build_checks(action, "left", pose_variant="heavy") == base
        assert all("availability" not in c and "blockedReason" not in c for c in base)

    # 验证同一检查项在 lite/full/heavy 下实际执行结论完全一致
    lm = np.repeat(_base_pose()[None], 3, axis=0)
    angle = np.deg2rad(60)
    lm[:, 11, :2], lm[:, 13, :2] = (0.5, 0.3), (0.5, 0.5)
    lm[:, 15, :2] = (0.5 + 0.2 * np.sin(angle), 0.5 - 0.2 * np.cos(angle))
    outcomes = {}
    for variant in ("lite", "full", "heavy"):
        cfg = FeedbackConfig(pose_variant=variant)
        profile = rule_profile("front_straight", "left", cfg)
        rule = next(r for r in profile.rules if r.code == "stance:start:front:stance_elbow")
        ctx = RuleContext(
            view="side",
            action="front_straight",
            stance="left",
            params=cfg,
            baseline=np.arange(3),
            phase_windows={rule.scope: np.arange(3)},
        )
        outcomes[variant] = evaluate_rules(
            lm,
            np.ones(lm.shape[:2], bool),
            replace(profile, rules=(rule,)),
            context=ctx,
        )[0]
        assert outcomes[variant].state == "violation"
    assert outcomes["lite"].state == outcomes["full"].state == outcomes["heavy"].state


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
    ctx = RuleContext(view="side", action="step_straight_combo", stance="left", params=cfg)
    guard_bad, guard_valid, guard_details = measure_guard_elbow("front", lm, mask, ctx)
    teacher_bad, teacher_valid, teacher_details = measure_punch_preelbow("front", lm, mask, ctx)
    np.testing.assert_allclose(guard_details["肘高于肩"][0], [0.2, 0, -0.2], atol=1e-7)
    np.testing.assert_array_equal(
        guard_details["肘高于肩"][0], teacher_details["起动前肘高于肩"][0]
    )
    assert guard_bad.tolist() == [True, False, False]
    assert teacher_bad.tolist() == [True, True, False]
    assert guard_valid.all() and teacher_valid.all()


def test_shared_displacement_retains_distinct_baseline_and_facing_requirements():
    baseline = np.repeat(_base_pose()[None], 4, axis=0).astype(float)
    baseline[:, 0, 0] = baseline[:, 23, 0] + 0.01
    lm = baseline.copy()
    lm[:, 15, 0] -= 0.09
    mask = np.ones(lm.shape[:2], bool)
    ctx = RuleContext(
        view="side",
        action="step_straight_combo",
        stance="left",
        params=FeedbackConfig(),
    )
    _, guard_valid, _ = measure_guard_back("front", lm, mask, baseline, mask, ctx)
    bad, valid, measured = measure_punch_prepull("front", lm, mask, baseline, mask, ctx)
    assert not guard_valid.any()  # 公共护手后拉要求明确侧身朝向，预摆仅要求方向非零。
    assert valid.all() and bad.all()
    np.testing.assert_allclose(measured["起动前向后预摆距离"][0], 0.3, atol=1e-7)

    baseline[:, 0, 0] = baseline[:, 23, 0] + 0.06
    baseline_mask = mask.copy()
    baseline_mask[0, 15] = False
    _, teacher_valid, _ = measure_punch_predrop("front", lm, mask, baseline, baseline_mask, ctx)
    assert teacher_valid.all()  # 一帧参照缺失不否决其余有效参照。


def test_public_private_boundary_enforces_cross_action_sharing():
    """核心铁律结构契约：跨动作族重叠项一律归属公共域，私有域只含本动作族独有策略。"""
    strategy_families: dict[str, set[str]] = {}
    code_families: dict[str, set[str]] = {}
    code_domains: dict[str, set[str]] = {}
    for segment, blueprints in SEGMENT_CHECK_BLUEPRINTS.items():
        fam = family_of(segment)
        for bp in blueprints:
            spec = STRATEGY_REGISTRY[bp.strategy]
            strategy_families.setdefault(bp.strategy, set()).add(fam)
            code_families.setdefault(bp.code, set()).add(fam)
            code_domains.setdefault(bp.code, set()).add(spec.domain)

    # 注册表零死策略：所有已注册策略都必须在蓝图中被使用。
    assert set(strategy_families) == set(STRATEGY_REGISTRY)

    for strategy, families in strategy_families.items():
        domain = STRATEGY_REGISTRY[strategy].domain
        if len(families) >= 2:
            assert domain in PUBLIC_DOMAINS, (
                f"跨动作族重叠策略 {strategy} ({families}) 必须归属公共域，实际为 {domain}"
            )
        if domain in PRIVATE_DOMAINS:
            assert families == {domain}, (
                f"私有域策略 {strategy} ({domain}) 只能由本动作族引用，实际被 {families} 引用"
            )

    # 跨动作族重叠的检查码（除各拳种专用齐肩槽位 shoulder_level 外）一律只绑定公共域策略。
    for code, families in code_families.items():
        if len(families) >= 2 and code != "shoulder_level":
            assert code_domains[code] <= PUBLIC_DOMAINS, (
                f"跨动作族检查码 {code} ({families}) 包含非公共域策略：{code_domains[code]}"
            )


def test_straight_hook_and_kick_share_common_guard_and_punch_definitions():
    """直拳、摆拳、低鞭腿共用同一公共护手策略与视角规律，并显式保留各自已确认阈值。"""
    attack_segments = (
        "front_straight",
        "rear_straight",
        "front_hook",
        "rear_hook",
        "front_low_kick",
        "rear_low_kick",
    )
    by_seg = {
        seg: [c for c in catalog_checks("left") if c["segment"] == seg] for seg in attack_segments
    }
    for seg, checks in by_seg.items():
        expected_guard_role = "rear" if seg.startswith("front_") else "front"
        expected_view = guard_view(expected_guard_role)
        for code, strategy in (
            ("guard_shoulder", "wrist_shoulder_drop"),
            ("guard_back", "elbow_behind_body"),
        ):
            items = [c for c in checks if c["code"] == code]
            assert len(items) == 2, f"{seg}:{code} 应覆盖 motion 与 finish"
            assert {c["role"] for c in items} == {expected_guard_role}
            assert {c["strategy"] for c in items} == {strategy}
            assert {c["primaryView"] for c in items} == {expected_view}

    # 验证摆拳与步法/姿态/衔接接入公共层、低鞭腿甩手收口到 finish 后的唯一目录数与各动作检查数。
    assert len(catalog_checks("left")) == 149
    assert len(build_checks("front_hook", "left")) == 36
    assert len(build_checks("rear_hook", "left")) == 41
    assert len(build_checks("hook_combo", "left")) == 62
    assert len(build_checks("step_hook_combo", "left")) == 66

    front_hook_checks = by_seg["front_hook"]
    rear_hook_checks = by_seg["rear_hook"]
    for seg_checks, attack_role in (
        (front_hook_checks, "front"),
        (rear_hook_checks, "rear"),
    ):
        assert not any(
            c["code"].startswith("guard_") and c["role"] == attack_role for c in seg_checks
        ), "摆拳出拳手不得误挂任何护手检查项"
        assert not {"guard_low", "guard_recover"} & {c["code"] for c in seg_checks}
        assert [c["phase"] for c in seg_checks if c["code"] == "shoulder_level"] == ["finish"]

    rear_straight_ids = {c["id"] for c in by_seg["rear_straight"]}
    assert {
        "rear_straight:motion:front:guard_jaw",
        "rear_straight:finish:front:guard_jaw",
    } <= rear_straight_ids

    rear_hook_ids = {c["id"] for c in rear_hook_checks}
    assert {
        "rear_hook:prepare:rear:punch_predrop",
        "rear_hook:prepare:rear:punch_prepull",
        "rear_hook:motion:front:guard_back",
        "rear_hook:finish:front:guard_back",
        "rear_hook:motion:front:guard_jaw",
        "rear_hook:finish:front:guard_jaw",
        "rear_hook:recovered:rear:punch_recover",
        "rear_hook:finish:rear:hook_arm_bend",
        "rear_hook:finish:rear:hook_forearm_level",
        "rear_hook:finish:rear:hook_over_center",
    } <= rear_hook_ids
    assert "rear_hook:prepare:rear:punch_preelbow" not in rear_hook_ids

    # 摆拳齐肩复用公共 wrist_shoulder_height_gap（0.20 + 双视角融合 any），公共护手抬肘统一为 -0.18。
    hook_level = next(c for c in rear_hook_checks if c["code"] == "shoulder_level")
    assert hook_level["strategy"] == "wrist_shoulder_height_gap"
    assert "primaryView" not in hook_level
    assert hook_level["parameterOverrides"]["shoulder_height"] == pytest.approx(0.20)
    hook_elbow = next(c for c in rear_hook_checks if c["code"] == "guard_elbow")
    assert hook_elbow["parameterOverrides"]["elbow_raise"] == pytest.approx(-0.18)
    assert hook_elbow["primaryView"] == "front"


def test_hook_common_guard_issues_group_across_motion_and_finish():
    """摆拳接入公共护手后，motion 与 finish 同类护手提醒合并出卡，不重复提醒。"""
    for segment, role, code in (
        ("front_hook", "rear", "guard_shoulder"),
        ("rear_hook", "front", "guard_jaw"),
    ):
        record = make_html_test_record(action=segment)
        base = record["result"]["checks"][0]
        base.update(
            id=f"{segment}:motion:{role}:{code}",
            segment=segment,
            code=code,
            role=role,
            phase="motion",
            phaseLabel="动作过程",
            status="candidate",
            review="pending",
        )
        finish = {
            **base,
            "id": f"{segment}:finish:{role}:{code}",
            "phase": "finish",
            "phaseLabel": "动作结束瞬间",
        }
        record["result"]["checks"] = [base, finish]
        model = build_report_model(record)
        grouped = group_action_issues(model.candidate_checks, segment)
        assert len(grouped) == 1
        assert len(grouped[0].members) == 2


def test_catalog_standards_are_plain_language_and_paired_torso_label_simplified():
    """护栏：全量 149 条目录标准均为大白话（零程序黑话，46 条唯一共享标准），且双摄肩宽比显示名已简化。"""
    from core.evaluation.paired_torso import paired_torso_check
    from core.evaluation.rules.spec import RuleContext

    forbidden_jargon = ("躯干长", "正交机位", "垂向位置", "归一肩宽比", "锁定角")
    left_catalog = catalog_checks("left")
    right_catalog = catalog_checks("right")
    assert len(left_catalog) == 149
    for item in (*left_catalog, *right_catalog):
        std = item["standard"]
        assert std, f"{item['id']} 缺少标准文案"
        for term in forbidden_jargon:
            assert term not in std, f"{item['id']} 的标准仍含程序黑话 {term!r}: {std}"

    unique_standards = {item["standard"] for item in left_catalog}
    assert unique_standards == {item["standard"] for item in right_catalog}
    assert len(unique_standards) == 46

    # 验证 paired_torso 测量项显示名与单位同步简化（不再出现“双视角归一肩宽比/正交机位近似”）
    lm = np.zeros((4, 33, 4), dtype=float)
    mask = np.ones((4, 33), dtype=bool)
    ctx = RuleContext(
        view="front",
        action="stance",
        stance="left",
        phase_windows={"stance:start": np.arange(4)},
        paired_torso=(np.full(4, 2.0), np.ones(4, dtype=bool)),
    )
    bad, valid, details = paired_torso_check("stance:start")(lm, mask, lm, mask, ctx)
    assert bad.all() and valid.all()
    assert list(details.keys()) == ["正面与侧面肩宽比"]
    assert details["正面与侧面肩宽比"][1] == "倍"
