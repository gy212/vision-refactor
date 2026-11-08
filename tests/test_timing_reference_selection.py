"""计时选源先核对观测真正需要的起始参照，再执行唯一动作识别。"""

from copy import deepcopy

import pytest

from core.evaluation import evaluate_series
from core.evaluation import action_timing
from tests.test_shared_motion_timing import _single_punch


@pytest.mark.parametrize("action", ["rear_straight", "straight_combo"])
@pytest.mark.parametrize("missing_only_at_start", [False, True])
def test_unusable_side_reference_selects_executable_front_before_detection(
    monkeypatch, action, missing_only_at_start
):
    front, side = _single_punch(), _single_punch()
    end = 7 if missing_only_at_start else len(side["times"])
    side["valid_mask"][:end, 0] = False
    side["landmarks"][:end, 0, 3] = 0
    expected = evaluate_series(action, "left", {"front": deepcopy(front)})
    detected = []
    original = action_timing.phase_windows

    def capture(series, *args):
        detected.append(series)
        return original(series, *args)

    monkeypatch.setattr(action_timing, "phase_windows", capture)
    result = evaluate_series(action, "left", {"front": front, "side": side})

    assert len(detected) == 1 and detected[0] is front
    assert result["alignment"]["timingSource"] == "front"
    observed = result["diagnostics"]["front"]["eventDetails"]
    assert len(observed) == 1
    assert observed == expected["diagnostics"]["front"]["eventDetails"]
    selected = result["alignment"]["sourceSelection"]["visibility"]
    assert selected["side"]["validFrames"] == len(side["times"])
    assert not selected["side"]["referenceReady"]
    failure = selected["side"]["references"]["rear_straight"]
    assert failure["directionReferenceFrames"] == 0
    assert "鼻点或髋点" in failure["reason"]
    assert selected["front"]["referenceReady"]


def test_side_only_timing_policy_does_not_borrow_front_after_reference_failure(monkeypatch):
    front, side = _single_punch(), _single_punch()
    side["valid_mask"][:, 0] = False
    detected = []
    original = action_timing.step_combo_windows

    def capture(series, *args):
        detected.append(series)
        return original(series, *args)

    monkeypatch.setattr(action_timing, "step_combo_windows", capture)
    result = evaluate_series("step_straight_combo", "left", {"front": front, "side": side})

    assert len(detected) == 1 and detected[0] is side
    assert result["alignment"]["timingSource"] == "side"
    assert result["diagnostics"]["side"]["eventDetails"] == []
    assert "鼻点或髋点" in result["diagnostics"]["side"]["reason"]
    assert not result["alignment"]["sourceSelection"]["visibility"]["side"]["referenceReady"]
