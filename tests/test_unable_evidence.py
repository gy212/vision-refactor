"""无法判断的真实取帧与失败原因必须贯通所有分析出口。"""

from copy import deepcopy

import pytest

from core.evaluation import FeedbackConfig, evaluate_series
from core.evaluation.rule_checks import judge_views
from core.reporting import build_report_model, select_evidence
from tests.test_evidence_contract import _render_requests
from tests.test_feedback_html_export import make_html_test_record
from tests.test_rule_evidence_entries import CHECK_ID, _batch, _check, _exam, _paired_views


def test_unable_keeps_actual_frames_and_missing_points_across_outputs(monkeypatch):
    views = _paired_views()
    # 脚尖无法识别不影响侧面动作计时；应保留正面实际选中的站姿画面。
    views["front"]["valid_mask"][:, 31] = False
    result = evaluate_series("step_straight_combo", "left", deepcopy(views))
    practice = next(check for check in result["checks"] if check["id"] == CHECK_ID)
    judged = _check(
        judge_views(deepcopy(views), "step_straight_combo", "left", FeedbackConfig())["front"][0]
    )
    exam = _check(_exam(deepcopy(views))[0]["checks"])
    # 回放视频入口从原始visibility重新派生mask。
    views["front"]["landmarks"][:, 31, 3] = 0
    batch = _check(_batch(views).front_rule_checks)

    assert practice["status"] == "unable"
    assert "左脚尖0/" in practice["reason"]
    assert practice["evidence"]
    for item in (judged, exam, batch):
        assert item.state == "unable"
        assert "左脚尖0/" in item.detail
        assert item.evidence_frames
        assert item.evidence_catalog
        selected = [detail for detail in item.evidence_details if detail["view"] == "front"]
        assert all(detail["purpose"] == "diagnostic" for detail in selected)
        assert all(
            not frame["validMask"][31]
            for frame in item.evidence_catalog
            if frame["view"] == "front"
        )

    record = make_html_test_record(action="step_straight_combo")
    record["result"] = result
    record["result"]["checks"] = [practice]
    model = build_report_model(record)
    picks = select_evidence(model.unable_checks[0], model.evidence_catalog)
    assert any(pick.purpose == "diagnostic" for pick in picks)
    requested, rendered = _render_requests(monkeypatch, record)
    assert requested
    assert "未能判定的取帧" in rendered
    assert "左脚尖0/" in rendered
    assert "查看所选画面与测量" in rendered


@pytest.mark.parametrize("view", ["front", "side"])
def test_few_valid_samples_keep_computed_values_and_exact_window(view):
    from dataclasses import replace
    import numpy as np
    from core.evaluation.rules.engine import evaluate_rules
    from core.evaluation.rules.spec import PhaseVerdict, RuleContext, RuleProfile, RuleSpec

    def check(lm, mask, baseline, baseline_mask, ctx):
        return lm[:, 15, 0] > 2, mask[:, 15], {"手腕位置": (lm[:, 15, 0], "单位")}

    lm = np.ones((9, 33, 4))
    lm[:, 15, 0] = np.arange(9)
    mask = np.ones((9, 33), bool)
    mask[4:7, 15] = False
    spec = RuleSpec(
        "probe", "手腕", check, (15,), view, "rear_straight:finish", PhaseVerdict(3, 0.6, 0.3)
    )
    ctx = RuleContext(view, "rear_straight", "left", {spec.scope: np.arange(2, 7)}, np.arange(2))
    outcome = evaluate_rules(lm, mask, RuleProfile("probe", (spec,)), context=ctx)[0]
    assert outcome.state == "unable"
    assert outcome.valid_frames == 2
    assert outcome.measurements[0]["min"] == 2
    assert outcome.measurements[0]["max"] == 3
    assert outcome.evidence_frames == (2, 4, 6)
    assert "左腕2/5帧" in outcome.detail
    assert all(item.purpose == "diagnostic" for item in outcome.evidence_instances)
    # 阶段只覆盖两帧也留原片；未定位阶段不借用邻近画面。
    short = replace(ctx, phase_windows={spec.scope: np.array([2, 3])})
    partial = evaluate_rules(lm, mask, RuleProfile("probe", (spec,)), context=short)[0]
    assert partial.evidence_frames == (2, 3)
    absent = evaluate_rules(
        lm, mask, RuleProfile("probe", (spec,)), context=replace(ctx, phase_windows={})
    )[0]
    assert not absent.evidence_frames
    assert absent.skip_reason == "phase_not_found"
