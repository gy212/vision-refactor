"""基础动作的阶段、身份及检查不受组合缺失成员影响。"""

import json

import numpy as np
import pytest

from core.evaluation import FeedbackConfig, evaluate_series
from core.evaluation.action_events import ActionEvent
from core.evaluation.event_phases import observed_phase_windows
from core.evaluation.motion_events import MotionEvent
from tests.test_step_straight_combo import _series


@pytest.mark.parametrize("rear_reference_visible", [True, False])
def test_front_only_keeps_identical_phases_and_checks_in_every_action(rear_reference_visible):
    series = _series(1)
    series["landmarks"][:, [14, 16]] = _series(0)["landmarks"][:, [14, 16]]
    if not rear_reference_visible:
        series["valid_mask"][:, 16] = False
    results = {
        action: evaluate_series(action, "left", {"front": series, "side": series})
        for action in ("front_straight", "straight_combo", "step_straight_combo")
    }
    reference = results["front_straight"]
    reference_data = reference["diagnostics"]["side"]
    assert [event["eventId"] for event in reference_data["eventDetails"]] == ["front_straight:1"]

    def front_checks(result):
        return json.dumps(
            [
                {
                    key: check[key]
                    for key in ("id", "status", "reason", "measurements", "evidenceDetails")
                }
                for check in result["checks"]
                if check["segment"] == "front_straight"
            ],
            sort_keys=True,
        )

    for result in results.values():
        data = result["diagnostics"]["side"]
        assert data["eventDetails"] == reference_data["eventDetails"]
        assert data["phaseInstances"] == reference_data["phaseInstances"]
        assert front_checks(result) == front_checks(reference)
    combined = results["step_straight_combo"]["diagnostics"]["side"]
    assert combined["repetitions"] == []
    assert combined["unpairedEventIds"] == ["front_straight:1"]


def test_missing_front_member_uses_observed_rear_interval_without_inventing_front():
    series = _series(1)
    series["landmarks"][:, [13, 15]] = _series(0)["landmarks"][:, [13, 15]]
    result = evaluate_series("step_straight_combo", "left", {"side": series})
    data = result["diagnostics"]["side"]
    rear = data["eventDetails"][0]
    combination = data["repetitions"][0]
    assert rear["eventId"] == "rear_straight:1"
    assert combination["frontEventId"] is None
    assert combination["rearEventId"] == rear["eventId"]
    assert combination["missingSegments"] == ["front_straight"]
    assert (combination["start"], combination["end"]) == (rear["start"], rear["end"])
    assert {instance["eventId"] for instance in data["phaseInstances"]} == {rear["eventId"]}


def test_short_confirmed_event_keeps_its_actual_finish_evidence():
    times = np.arange(12) / 30
    event = ActionEvent("front_straight", MotionEvent(6, 6, 7, "returned"))
    diagnostics = {"phaseInstances": []}
    windows = observed_phase_windows([event], times, [np.arange(3)], FeedbackConfig(), diagnostics)
    np.testing.assert_array_equal(windows[("front_straight", "finish")], [6, 7])
    finish = next(
        instance for instance in diagnostics["phaseInstances"] if instance["phase"] == "finish"
    )
    assert finish["eventId"] == diagnostics["eventDetails"][0]["eventId"]
    assert (finish["start"], finish["end"]) == (times[6], times[7])
