"""练习、考试与批量最终结果共同保存双面测量的完整证据。"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from core.evaluation import FeedbackConfig, compare_dual_streams, evaluate_series
from core.evaluation.rule_checks import judge_view, judge_views
from core.evaluation.scoring.exam_pool import rule_reports
from core.foundation.jsonable import to_jsonable
from tests import golden_harness as H
from tests.test_step_straight_combo import _series

ACTION = "step_straight_combo"
CHECK_ID = "stance:start::stance_width"
FIXTURES = Path(__file__).parent / "fixtures" / "pose33_v3"


def _paired_views():
    front, side = _series(1), _series(1)
    front["landmarks"][:, [29, 30, 31, 32], 0] = 0.5
    for series in (front, side):
        series["landmarks"][:, 11, 0] = 0.45
        series["landmarks"][:, 12, 0] = 0.55
    return {"front": front, "side": side}


def _check(checks):
    return next(check for check in checks if check.check_id == CHECK_ID)


def _exam(views):
    targets = {
        view: [
            {
                "name": "上步前后手直拳组合",
                "action": ACTION,
                "view": view,
                "series": series,
                "start": 0,
                "end": len(series["frames"]) - 1,
            }
        ]
        for view, series in views.items()
    }
    return rule_reports(targets["front"], targets["side"], FeedbackConfig())


def _batch(views):
    names = {view: f"evidence://{view}.mp4" for view in views}
    H.clear_registry()
    for view, series in views.items():
        H.register_video(names[view], series["landmarks"], width=720, height=720)
    try:
        with H.replay_context():
            return compare_dual_streams(
                FIXTURES / "front_template.npz",
                FIXTURES / "side_template.npz",
                names["front"],
                names["side"],
                pose_variant="full",
                enable_rules=True,
                rule_action=ACTION,
            )
    finally:
        H.clear_registry()


def _assert_complete(details, catalog):
    catalog = {frame["id"]: frame for frame in catalog}
    assert {item["view"] for item in details} == {"front", "side"}
    primaries = {item["ref"]: item for item in details if item["view"] == "front"}
    references = [item for item in details if item["view"] == "side"]
    assert primaries and references
    for detail in details:
        frame = catalog[detail["ref"]]
        assert detail["frame"] == frame["frame"]
        assert detail["timeSeconds"] == frame["timeSeconds"]
        assert len(frame["landmarks"]) == len(frame["validMask"]) == 33
        assert detail["requiredViews"] == ["front", "side"]
    for reference in references:
        primary = primaries[reference["referenceFor"]]
        assert reference["purpose"] == "reference"
        assert reference["timeSeconds"] == primary["timeSeconds"]
        assert reference["highlightJoints"] == [11, 12]
        assert primary["timingSource"] == "side"
        assert reference["timingSource"] == "own_view"


def test_all_final_outputs_keep_the_same_paired_measurement_evidence():
    views = _paired_views()
    practice = evaluate_series(ACTION, "left", deepcopy(views))
    practice_check = next(check for check in practice["checks"] if check["id"] == CHECK_ID)
    judged = _check(judge_views(deepcopy(views), ACTION, "left", FeedbackConfig())["front"][0])
    exam = _check(_exam(deepcopy(views))[0]["checks"])
    batch = _check(_batch(deepcopy(views)).front_rule_checks)

    assert practice_check["status"] == "candidate"
    _assert_complete(practice_check["evidenceDetails"], practice["evidence"])
    for check in (judged, exam, batch):
        assert check.state == "violation"
        saved = json.loads(json.dumps(to_jsonable(check)))
        _assert_complete(saved["evidence_details"], saved["evidence_catalog"])
        # 入口权重可不同，但不能改变原画面、部位、来源及参照。
        observed = [
            {key: value for key, value in item.items() if key != "weight"}
            for item in saved["evidence_details"]
        ]
        expected = [
            {key: value for key, value in item.items() if key != "weight"}
            for item in practice_check["evidenceDetails"]
        ]
        assert observed == expected


@pytest.mark.parametrize("missing", ["time", "points", "view"])
def test_missing_paired_data_never_produces_reference_or_pass(missing):
    views = _paired_views()
    if missing == "time":
        views["side"]["times"] = [time + 0.001 for time in views["side"]["times"]]
    elif missing == "points":
        views["side"]["valid_mask"][:, [11, 12]] = False
    else:
        views.pop("side")
    judged = _check(judge_views(views, ACTION, "left", FeedbackConfig())["front"][0])
    assert judged.state == "unable"
    assert not judged.evidence_details and not judged.evidence_catalog


def test_single_view_rule_output_records_own_original_times():
    series = _series(1)
    checks, _ = judge_view(series, "side", ACTION, "left", FeedbackConfig())
    details = [detail for check in checks for detail in check.evidence_details]
    assert details
    assert all(detail["timingSource"] == "own_view" for detail in details)
    assert all(detail["timeSeconds"] == series["times"][detail["frame"]] for detail in details)
    assert all(detail["purpose"] != "reference" for detail in details)
