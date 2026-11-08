"""相同直拳在单拳、组合及所有分析入口中真正使用同一机位和测量。"""

from copy import deepcopy
from pathlib import Path

import pytest

from core.evaluation import ACTIONS, FeedbackConfig, compare_dual_streams, evaluate_series
from core.evaluation.rule_checks import judge_views
from core.evaluation.scoring.exam_pool import rule_reports
from tests import golden_harness as H
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
