"""相同直拳在单拳、组合及所有分析入口中真正使用同一机位和测量。"""

from copy import deepcopy
from pathlib import Path

import pytest

from core.evaluation import ACTIONS, FeedbackConfig, compare_dual_streams, evaluate_series
from core.evaluation.rule_checks import judge_views
from core.evaluation.scoring.exam_pool import rule_reports
from core.foundation.jsonable import to_jsonable
from tests import golden_harness as H
from tests.test_shared_motion_timing import _single_punch
from tests.test_step_straight_combo import _series


FIXTURES = Path(__file__).parent / "fixtures" / "pose33_v3"


def _visible_side_punch(role):
    front, side = _series(1), _series(1)
    shoulder, elbow, wrist = (11, 13, 15) if role == "front" else (12, 14, 16)
    # 侧面直接可见一直线的肩、肘、腕；正面同一手肘被遮住。
    side["landmarks"][:, elbow, :2] = (
        side["landmarks"][:, shoulder, :2] + side["landmarks"][:, wrist, :2]
    ) / 2
    front["valid_mask"][:, elbow] = False
    front["landmarks"][:, elbow, 3] = 0
    return {"front": front, "side": side}


def _exam(views, action):
    targets = {
        view: [
            {
                "name": ACTIONS[action],
                "action": action,
                "view": view,
                "series": series,
                "start": 0,
                "end": len(series["frames"]) - 1,
            }
        ]
        for view, series in views.items()
    }
    return rule_reports(targets["front"], targets["side"], FeedbackConfig())


def _batch(views, action):
    names = {view: f"consumer://{view}.mp4" for view in views}
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
                rule_action=action,
            )
    finally:
        H.clear_registry()


@pytest.mark.parametrize(
    ("action", "role"),
    [
        ("front_straight", "front"),
        ("rear_straight", "rear"),
        ("straight_combo", "front"),
        ("straight_combo", "rear"),
        ("step_straight_combo", "front"),
        ("step_straight_combo", "rear"),
    ],
)
def test_visible_punch_is_judged_in_every_composition_and_public_entry(action, role):
    views = _visible_side_punch(role)
    check_id = f"{role}_straight:finish:{role}:punch_extension"
    practice = evaluate_series(action, "left", deepcopy(views))
    check = next(item for item in practice["checks"] if item["id"] == check_id)
    assert check["status"] == "not_observed"
    assert check["fusion"]["baseWeights"] == {"front": 0.0, "side": 1.0}
    assert {item["view"] for item in check["measurements"]} == {"side"}
    assert check["availability"]["state"] == "trial"

    judged = judge_views(deepcopy(views), action, "left", FeedbackConfig())
    exam = _exam(deepcopy(views), action)
    batch = _batch(deepcopy(views), action)
    outputs = [
        judged["side"][0],
        tuple(item for report in exam for item in report["checks"]),
        batch.side_rule_checks,
    ]
    expected_frames = [item["frame"] for item in check["evidenceDetails"]]
    for checks in outputs:
        result = next(item for item in checks if item.check_id == check_id)
        assert result.state == "pass"
        assert result.view == "side"
        assert result.valid_frames == result.total_frames >= 3
        assert "试用" in result.name
        assert [item["frame"] for item in result.evidence_details] == expected_frames
        assert result.measurements == tuple(check["measurements"][0]["values"])


@pytest.mark.parametrize("case", ["unpaired_front", "missing_side_reference"])
def test_review_cases_keep_the_same_events_and_verdicts_in_every_output(case):
    if case == "unpaired_front":
        series = _series(1)
        series["landmarks"][:, [14, 16], :] = _series(0)["landmarks"][:, [14, 16], :]
        views = {"front": deepcopy(series), "side": deepcopy(series)}
        action, source, segment = "step_straight_combo", "side", "front_straight"
    else:
        views = {"front": _single_punch(), "side": _single_punch()}
        views["side"]["valid_mask"][:, 0] = False
        views["side"]["landmarks"][:, 0, 3] = 0
        action, source, segment = "rear_straight", "front", "rear_straight"
    practice = evaluate_series(action, "left", deepcopy(views))
    judged = judge_views(deepcopy(views), action, "left", FeedbackConfig())
    exam = _exam(deepcopy(views), action)
    batch = _batch(deepcopy(views), action)
    expected = practice["sharedActionRecord"]
    assert expected["timingSource"] == source
    assert len(expected["eventDetails"]) == 1
    assert expected["eventDetails"][0]["segment"] == segment
    phases = [item for item in expected["phaseInstances"] if item["segment"] == segment]
    assert {"motion", "finish"} <= {item["phase"] for item in phases}
    assert to_jsonable(judged[source][1]["sharedActionRecord"]) == to_jsonable(expected)
    assert to_jsonable(exam[0]["sharedActionRecord"]) == to_jsonable(expected)
    assert to_jsonable(batch.shared_action_records) == [to_jsonable(expected)]
    check_id = f"{segment}:finish:{segment.split('_')[0]}:punch_extension"
    practice_check = next(item for item in practice["checks"] if item["id"] == check_id)
    for output in (
        judged["side"][0],
        tuple(item for report in exam for item in report["checks"]),
        batch.side_rule_checks,
    ):
        check = next(item for item in output if item.check_id == check_id)
        assert check.state in {"pass", "violation"}
        assert practice_check["status"] == (
            "candidate" if check.state == "violation" else "not_observed"
        )
        assert check.evidence_details
