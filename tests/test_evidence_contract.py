"""公开评估结果的证据来源须完整进入 HTML 和结果页共用的选图能力。"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.evaluation.rules.engine import evaluate_rules
from core.evaluation import evaluate_series
from core.evaluation.rules.spec import PhaseVerdict, RuleContext, RuleProfile, RuleSpec
from core.reporting import (
    build_report_model,
    group_action_issues,
    format_report,
    render_html_report,
    select_evidence,
)
from tests.test_feedback_html_export import make_html_test_record
from tests.test_step_straight_combo import _series


def _record_with_evidence(*, grouped=False, paired=False):
    record = make_html_test_record(action="step_straight_combo" if grouped else "rear_straight")
    original = record["result"]["checks"][0]
    original.update(
        id="rear_straight:motion:front:guard_elbow",
        segment="rear_straight",
        code="guard_elbow",
        phase="motion",
        phaseLabel="动作过程",
        evidence=["front:10", "front:11", "front:12", "side:10", "side:11", "side:12"],
        measurements=[
            {
                "view": view,
                "issueObserved": observed,
                "violationRatio": float(observed),
                "validFrames": 30,
                "totalFrames": 30,
                "validRatio": 1.0,
                "values": [],
            }
            for view, observed in (("front", False), ("side", True))
        ],
        fusion={
            "baseWeights": {"front": 0.2, "side": 0.8},
            "effectiveWeights": {},
            "method": "weighted_support",
            "support": 0.8,
        },
    )
    checks = [original]
    if grouped:
        finish = deepcopy(original)
        finish.update(
            id="rear_straight:finish:front:guard_elbow",
            phase="finish",
            phaseLabel="动作结束瞬间",
            evidence=[f"{v}:{f}" for v in ("front", "side") for f in (20, 21, 22)],
        )
        checks.append(finish)
    for check in checks:
        check["evidenceDetails"] = [
            {
                "ref": ref,
                "checkId": check["id"],
                "phaseLabel": check["phaseLabel"],
                "view": ref.partition(":")[0],
                "highlightJoints": [11, 13, 15],
                "purpose": "reference"
                if paired and ref.startswith("front:")
                else "violation"
                if ref.startswith("side:")
                else "observation",
                "weight": 0.8 if ref.startswith("side:") else 0.2,
            }
            for ref in check["evidence"]
        ]
    record["result"]["checks"] = checks
    record["result"]["evidence"] = [
        {
            "id": ref,
            "view": ref.partition(":")[0],
            "frame": int(ref.partition(":")[2]),
            "timeSeconds": int(ref.partition(":")[2]) / 30,
        }
        for check in checks
        for ref in check["evidence"]
    ]
    return record


def _render_requests(monkeypatch, record, **kwargs):
    requested = []

    def capture(path, frame, **options):
        requested.append((path.stem, frame, options.get("highlight_joints")))
        return np.zeros((12, 12, 3), np.uint8)

    monkeypatch.setattr(
        "core.reporting.html.renderer.resolve_record_video_path",
        lambda root, record_id, view, videos: Path(view + ".mp4"),
    )
    monkeypatch.setattr("core.reporting.html.renderer.extract_evidence_frame", capture)
    rendered = render_html_report(record, history_root=Path.cwd(), **kwargs)
    return requested, rendered


def test_final_html_uses_supporting_view_and_saved_measured_joints(monkeypatch):
    record = _record_with_evidence()
    requested, rendered = _render_requests(monkeypatch, record)

    assert requested == [("side", 10, {11, 13, 15}), ("front", 10, {11, 13, 15})]
    assert "侧面问题依据" in rendered and "正面观测画面" in rendered
    assert "旧记录未保存测量部位" not in rendered


def test_grouped_phases_keep_each_members_views_and_review(monkeypatch):
    record = _record_with_evidence(grouped=True, paired=True)
    record["result"]["checks"][0]["review"] = "confirmed"
    original = deepcopy(record)
    requested, rendered = _render_requests(monkeypatch, record)
    model = build_report_model(record)
    grouped = group_action_issues(model.candidate_checks, record["action"])

    assert [(view, frame) for view, frame, _ in requested] == [
        ("side", 10),
        ("front", 10),
        ("side", 20),
        ("front", 20),
    ]
    assert len(grouped[0].members) == 2
    assert [member.review for member in grouped[0].members] == ["confirmed", "pending"]
    assert "动作过程：教师已确认" in rendered and "动作结束瞬间：待教师复核" in rendered
    assert "正面测量参照" in rendered
    assert model.counts.candidate_count == 2
    assert record == original


def test_image_budget_names_omitted_phase_and_view(monkeypatch):
    record = _record_with_evidence(grouped=True)
    requested, rendered = _render_requests(monkeypatch, record, max_images=2)

    assert len(requested) == 2
    assert "动作结束瞬间 · 侧面问题依据" in rendered
    assert "图片数量已达上限，本阶段此机位图片已省略" in rendered


def test_legacy_saved_weights_select_views_without_current_rule_inference(monkeypatch):
    record = _record_with_evidence()
    del record["result"]["checks"][0]["evidenceDetails"]
    original = deepcopy(record)

    def unexpected(*args, **kwargs):
        raise AssertionError("读取旧检查不能查询当前规则定义")

    monkeypatch.setattr("core.evaluation.rules.sanda_v5.build_checks", unexpected)
    requested, rendered = _render_requests(monkeypatch, record)

    assert [(view, frame) for view, frame, _ in requested] == [("side", 10), ("front", 10)]
    assert all(joints is None for _, _, joints in requested)
    assert "旧记录未保存测量部位，未加问题标记" in rendered
    assert record == original


def test_shared_frame_keeps_each_checks_own_markers(monkeypatch):
    record = _record_with_evidence()
    first = record["result"]["checks"][0]
    first["evidence"] = ["side:10"]
    first["evidenceDetails"] = [first["evidenceDetails"][3]]
    second = deepcopy(first)
    second["id"] = "rear_straight:motion:rear:guard_elbow"
    second["evidenceDetails"][0].update(checkId=second["id"], highlightJoints=[12, 14, 16])
    record["result"]["checks"] = [first, second]

    requested, _ = _render_requests(monkeypatch, record)

    assert requested == [("side", 10, {11, 13, 15}), ("side", 10, {12, 14, 16})]
    assert "highlightJoints" not in record["result"]["evidence"][0]


def test_engine_retains_middle_repetition_evidence_without_changing_verdict():
    windows = [np.arange(i, i + 3) for i in (0, 5, 10, 15)]
    indices = np.concatenate(windows)

    def measured(lm, mask, baseline, baseline_mask, context):
        return np.ones(len(lm), bool), np.ones(len(lm), bool), {}

    spec = RuleSpec(
        "test", "测试", measured, (11, 13, 15), "side", "punch:finish", PhaseVerdict(3, 0.5, 0.3)
    )
    outcome = evaluate_rules(
        np.zeros((18, 33, 4)),
        np.ones((18, 33), bool),
        RuleProfile("test", (spec,)),
        context=RuleContext(
            "side", "rear_straight", phase_windows={spec.scope: indices}, baseline=np.arange(3)
        ),
    )[0]

    assert outcome.state == "violation" and outcome.violation_ratio == 1.0
    assert outcome.valid_frames == outcome.total_frames == 12
    assert outcome.evidence_frames == tuple(indices)


def test_saved_repetitions_are_preserved_by_shared_selection():
    record = _record_with_evidence()
    check = record["result"]["checks"][0]
    for detail in check["evidenceDetails"]:
        detail["repetition"] = int(detail["ref"].partition(":")[2]) - 9
    model = build_report_model(record)

    selected = select_evidence(model.all_checks[0], model.evidence_catalog)

    assert {(item.repetition, item.view) for item in selected} == {
        (number, view) for number in (1, 2, 3) for view in ("front", "side")
    }


def test_instance_sampling_keeps_aggregate_numbers_and_does_not_invent_local_violations():
    def measured(lm, mask, baseline, baseline_mask, context):
        return np.arange(len(lm)) < 4, np.ones(len(lm), bool), {}

    spec = RuleSpec(
        "test", "测试", measured, (11,), "side", "punch:motion", PhaseVerdict(3, 0.5, 0.3)
    )
    context = RuleContext(
        "side", "rear_straight", phase_windows={spec.scope: np.arange(8)}, baseline=np.arange(3)
    )
    frames = np.zeros((8, 33, 4))
    valid = np.ones((8, 33), bool)
    profile = RuleProfile("test", (spec,))
    before = evaluate_rules(frames, valid, profile, context=context)[0]
    after = evaluate_rules(
        frames,
        valid,
        profile,
        context=replace(
            context, phase_instances={spec.scope: ((1, np.arange(4)), (2, np.arange(4, 8)))}
        ),
    )[0]

    assert replace(after, evidence_frames=(), evidence_instances=()) == replace(
        before, evidence_frames=(), evidence_instances=()
    )
    assert {(item.repetition, item.purpose) for item in after.evidence_instances} == {
        (1, "violation"),
        (2, "observation"),
    }


def test_tk_uses_shared_selection_and_never_borrows_unrelated_frames(monkeypatch):
    pytest.importorskip("tkinter")
    from apps.tk.views.practice_result.evidence_view import EvidenceView

    record = _record_with_evidence()
    check = record["result"]["checks"][0]
    check["evidence"] = ["side:10", "side:11"]
    model = build_report_model(record)
    requested = []

    class Value:
        def set(self, value):
            self.value = value

        def get(self):
            return self.value

    boxes = {
        view: SimpleNamespace(
            view=view, view_label=view, title_var=Value(), show_text=lambda text: None
        )
        for view in ("front", "side")
    }
    panel = SimpleNamespace(
        boxes=boxes,
        reset_frames=lambda: None,
        _show_evidence=lambda box, record, selected, ev, path: requested.append(
            (box.view, selected.ref)
        ),
    )
    EvidenceView.load_frames(panel, record, model.all_checks[0], model.evidence_catalog, Path.cwd())

    assert requested == [("side", "side:10")]


def test_paired_measurement_keeps_reference_view_from_same_sample_time():
    from core.evaluation.rules.adapters import register_paired_evidence

    record = _record_with_evidence()
    check = record["result"]["checks"][0]
    check.update(
        evidence=["front:10"],
        evidenceDetails=[check["evidenceDetails"][0]],
        requiredViews=["front", "side"],
        highlightJoints=[11, 12, 29, 30, 31, 32],
        evidenceJointsByView={"front": [11, 12, 29, 30, 31, 32], "side": [11, 12]},
    )
    catalog = {ev["id"]: ev for ev in record["result"]["evidence"]}
    side = {
        "times": [10 / 30, 11 / 30],
        "frames": [100, 101],
        "landmarks": np.zeros((2, 33, 4)),
        "valid_mask": np.zeros((2, 33), bool),
    }
    side["valid_mask"][:, [11, 12]] = True

    register_paired_evidence(check, {"side": side}, catalog)

    assert check["evidence"] == ["front:10", "side:100"]
    assert check["evidenceDetails"][-1]["purpose"] == "reference"
    assert check["evidenceDetails"][-1]["highlightJoints"] == [11, 12]
    assert "purpose" not in catalog["side:100"]


def test_public_evaluation_to_html_keeps_side_judgment_and_correct_arm(monkeypatch):
    series = _series(1)
    angle = np.deg2rad(65)
    series["landmarks"][:, 11, :2] = (0.5, 0.3)
    series["landmarks"][:, 13, :2] = (0.5, 0.5)
    series["landmarks"][:, 15, :2] = (0.5 + 0.2 * np.sin(angle), 0.5 - 0.2 * np.cos(angle))
    result = evaluate_series("rear_straight", "left", {"front": series, "side": deepcopy(series)})
    check = next(c for c in result["checks"] if c["id"] == "stance:start:front:stance_elbow")
    assert check["status"] == "candidate"
    assert all(d["strategy"] == "elbow_angle_range" for d in check["evidenceDetails"])
    record = make_html_test_record(action="rear_straight")
    record["result"] = {**result, "checks": [check]}

    requested, _ = _render_requests(monkeypatch, record)

    assert [view for view, _, _ in requested] == ["side", "front"]
    assert all(joints == {11, 13, 15} for _, _, joints in requested)


def test_public_paired_width_records_reference_shoulders_when_feet_are_occluded():
    front, side = _series(1), _series(1)
    front["landmarks"][:, [29, 30, 31, 32], 0] = 0.5
    for series in (front, side):
        series["landmarks"][:, 11, 0] = 0.45
        series["landmarks"][:, 12, 0] = 0.55
    side["valid_mask"][:, [29, 30, 31, 32]] = False

    result = evaluate_series("step_straight_combo", "left", {"front": front, "side": side})
    check = next(c for c in result["checks"] if c["id"] == "stance:start::stance_width")

    assert check["status"] == "candidate"
    references = [d for d in check["evidenceDetails"] if d["purpose"] == "reference"]
    assert references and all(d["view"] == "side" for d in references)
    assert all(d["highlightJoints"] == [11, 12] for d in references)


def test_reanalysis_preserves_old_rule_evidence_and_review(tmp_path):
    from datetime import datetime, timezone
    from core.services import HistoryStore

    original = _record_with_evidence()["result"]
    original["checks"][0]["review"] = "confirmed"
    source = tmp_path / "front.mp4"
    source.write_bytes(b"injected analysis")
    store = HistoryStore(
        tmp_path / "history", clock=lambda: datetime(2026, 11, 6, tzinfo=timezone.utc)
    )
    identity = {"studentId": "001", "studentName": "", "action": "rear_straight", "stance": "left"}
    first = store.add(identity, source, None, analyzer=lambda *a, **k: deepcopy(original))
    changed = deepcopy(original)
    changed["ruleVersion"] = "new-version"
    changed["checks"][0]["evidenceDetails"][0]["highlightJoints"] = [12, 14, 16]
    changed["checks"][0]["review"] = "pending"

    updated = store.reanalyze(first["id"], analyzer=lambda *a, **k: changed)
    reread = store.get(first["id"])

    assert reread["schemaVersion"] == 1 and reread["revision"] == updated["revision"] == 2
    assert reread["previousResults"][0]["result"] == original
    assert reread["result"] == changed


def test_unpaired_front_punch_does_not_change_combination_evidence_number():
    series = _series(1)
    times = np.asarray(series["times"])
    inside = (times >= 0.6) & (times <= 0.9)
    reach = 0.2 * (1 - np.abs(2 * (times[inside] - 0.6) / 0.3 - 1))
    series["landmarks"][inside, 15, 0] += reach
    series["landmarks"][inside, 13, 0] += reach / 2

    result = evaluate_series(
        "step_straight_combo", "left", {"front": series, "side": deepcopy(series)}
    )

    assert result["repetitions"]["detected"] == 1
    for segment in ("front_straight", "rear_straight"):
        details = [
            detail
            for check in result["checks"]
            if check["segment"] == segment and check["phase"] == "finish"
            for detail in check.get("evidenceDetails", [])
        ]
        assert details and {detail["repetition"] for detail in details} == {1}


def test_touching_motion_windows_keep_all_four_repetitions_in_final_html(monkeypatch):
    series = _series(0, seconds=4)
    for start in (30, 35, 40, 45):
        series["landmarks"][start : start + 4, 16, 0] += 0.18
        series["landmarks"][start : start + 4, 14, 0] += 0.09
    series["landmarks"][29:50, 15, 0] = 0.30  # 四次打拳时护手都相对原护位后拉。
    result = evaluate_series("rear_straight", "left", {"front": deepcopy(series), "side": series})
    check = next(c for c in result["checks"] if c["id"] == "rear_straight:motion:front:guard_back")

    assert check["status"] == "candidate"
    assert {detail["repetition"] for detail in check["evidenceDetails"]} == {1, 2, 3, 4}
    assert {
        detail["repetition"] for detail in check["evidenceDetails"] if detail["ref"] == "side:34"
    } == {1, 2}  # 共用边界帧保留两次来源，不通过时间反推覆盖。
    record = make_html_test_record(action="rear_straight")
    record["result"] = {**result, "checks": [check]}

    requested, rendered = _render_requests(monkeypatch, record)

    assert {29, 34, 39, 44} <= {frame for _, frame, _ in requested}
    for repetition in (1, 2, 3, 4):
        assert f"第{repetition}次 · 动作过程 · 侧面问题依据" in rendered
    no_root = render_html_report(record)
    assert "第3次 · 动作过程 · 侧面问题依据" in no_root
    assert "未指定历史目录，跳过截帧" in no_root


@pytest.mark.parametrize("checks_present", [False, True])
def test_old_missing_or_empty_checklist_never_imports_todays_rules(checks_present, monkeypatch):
    record = {
        "schemaVersion": 1,
        "id": "b" * 32,
        "studentId": "002",
        "action": "step_straight_combo",
        "stance": "left",
        "result": {"schemaVersion": 1, "summary": "当时保存的结果摘要"},
    }
    if checks_present:
        record["result"]["checks"] = []
    original = deepcopy(record)

    def unexpected(*args, **kwargs):
        raise AssertionError("读取旧记录不能查询当前检查目录")

    monkeypatch.setattr("core.evaluation.rules.sanda_v5.build_checks", unexpected)
    model = build_report_model(record)
    html = render_html_report(record)
    text = format_report(record)

    assert model.counts.total_checks == 0 and model.all_checks == []
    for output in (html, text, model.headline.subtitle):
        assert "当时保存的结果摘要" in output
        assert "未保存详细检查清单" in output
        assert "80°" not in output and "试用" not in output
    assert record == original
