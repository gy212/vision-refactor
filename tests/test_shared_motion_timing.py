"""公共动作识别与逐阶段对应的业务回归。"""

from copy import deepcopy

import numpy as np
import pytest

from core.evaluation import FeedbackConfig, evaluate_series
from core.evaluation.rule_checks import synchronized_windows
from core.evaluation import action_timing
from core.evaluation.punch_events import reach_events
from tests.test_step_straight_combo import _series


def _single_punch(*, downward=False, held=False, lost=False):
    series = _series(0)
    times = np.asarray(series["times"])
    inside = (times >= 1.5) & (times <= 2.1)
    displacement = 0.2 * (1 - np.abs(2 * (times[inside] - 1.5) / 0.6 - 1))
    axis = 1 if downward else 0
    series["landmarks"][inside, 16, axis] += displacement
    if held:
        series["landmarks"][times >= 1.8, 16, 0] = 0.75
    if lost:
        series["valid_mask"][(times >= 1.7) & (times <= 2.0), 16] = False
    return series


@pytest.mark.parametrize("action", ["rear_straight", "step_straight_combo"])
def test_public_detector_rejects_release_and_retains_bad_punch(action):
    down = _single_punch(downward=True)
    result = evaluate_series(action, "left", {"front": down, "side": deepcopy(down)})
    assert result["diagnostics"]["side"]["eventDetails"] == []
    punch = _single_punch()
    # 肘保持弯曲不妨碍识别“发生了一次拳”，姿势由对应检查判断。
    result = evaluate_series(action, "left", {"front": punch, "side": deepcopy(punch)})
    events = result["diagnostics"]["side"]["eventDetails"]
    rear = [event for event in events if event["segment"] == "rear_straight"]
    assert len(rear) == 1 and rear[0]["ending"] == "returned"
    assert rear[0]["start"] < rear[0]["peak"] < rear[0]["end"]


@pytest.mark.parametrize("action", ["rear_straight", "step_straight_combo"])
def test_public_detector_keeps_held_and_rejects_gap_bridge(action):
    held = _single_punch(held=True)
    result = evaluate_series(action, "left", {"side": held})
    events = result["diagnostics"]["side"]["eventDetails"]
    assert len(events) == 1 and events[0]["ending"] == "held"
    assert "未观察到收回" in result["diagnostics"]["side"]["reason"]
    lost = _single_punch(lost=True)
    result = evaluate_series(action, "left", {"side": lost})
    assert result["diagnostics"]["side"]["eventDetails"] == []


@pytest.mark.parametrize("front_count,side_count", [(4, 2), (4, 4)])
def test_dense_actions_share_one_detection_and_instance_identity(
    monkeypatch, front_count, side_count
):
    def rapid(count):
        series = _series(0, seconds=4)
        for start in range(30, 50, 20 // count):
            width = 20 // count - 1
            series["landmarks"][start : start + width, 16, 0] += 0.18
            series["landmarks"][start : start + width, 14, 0] += 0.09
        return series

    views = {"front": rapid(front_count), "side": rapid(side_count)}
    called = []
    detect = action_timing.phase_windows

    def observe(series, *args):
        called.append(series)
        return detect(series, *args)

    monkeypatch.setattr(action_timing, "phase_windows", observe)
    result = evaluate_series("rear_straight", "left", views)
    assert len(called) == 1 and called[0] is views["side"]
    assert result["alignment"]["eventCounts"] == {"side": {"rear_straight": side_count}}
    source, target = result["diagnostics"]["side"], result["diagnostics"]["front"]
    assert target["eventDetails"] == []
    assert target["projectedPhaseInstances"] == source["phaseInstances"]
    assert target["projectedEventDetails"] == source["eventDetails"]
    assert result["phaseWindows"]["front"] == result["phaseWindows"]["side"]
    checks = {c["id"]: c for c in result["checks"]}
    assert checks["rear_straight:motion:front:guard_low"]["status"] != "unable"
    assert "eventPairs" not in result["alignment"]


def test_source_visibility_is_checked_before_detection_and_missing_data_stays_unable(monkeypatch):
    front, side = _single_punch(), _series(0)
    side["valid_mask"][:, 16] = False
    result = evaluate_series("rear_straight", "left", {"front": front, "side": side})
    assert result["alignment"]["timingSource"] == "front"
    assert len(result["diagnostics"]["front"]["eventDetails"]) == 1
    check = next(
        c for c in result["checks"] if c["id"] == "rear_straight:finish:rear:punch_extension"
    )
    assert check["status"] == "unable"
    assert "有效帧不足" in check["reason"]


def test_source_with_no_event_is_not_replaced_by_a_more_active_other_view():
    result = evaluate_series(
        "rear_straight", "left", {"front": _single_punch(), "side": _series(0)}
    )
    assert result["alignment"]["timingSource"] == "side"
    assert result["diagnostics"]["side"]["eventDetails"] == []
    assert result["diagnostics"]["front"]["projectedEventDetails"] == []
    check = next(
        c for c in result["checks"] if c["id"] == "rear_straight:finish:rear:punch_extension"
    )
    assert check["status"] == "unable"


def test_shared_phase_still_requires_target_time_coverage():
    front, side = _series(0), _series(0)
    side["times"] = [time + 10 for time in side["times"]]
    result = evaluate_series("stance", "left", {"front": front, "side": side})
    elbow = next(
        check for check in result["checks"] if check["id"] == "stance:start:front:stance_elbow"
    )
    assert elbow["status"] == "unable"
    assert result["phaseWindows"]["side"] == []


def test_projected_events_are_not_reported_as_independent_observations():
    _, diagnostics, alignment, sources = synchronized_windows(
        {"front": _series(0), "side": _series(2)}, "step_straight_combo", "left", FeedbackConfig()
    )
    assert diagnostics["front"]["eventDetails"] == []
    assert diagnostics["front"]["projectedEventDetails"] == diagnostics["side"]["eventDetails"]
    assert diagnostics["front"]["timingSource"] == "side"
    assert sources["front"] == "side_synchronized_timing"
    assert "front" not in alignment["eventCounts"]


def test_single_rear_punch_has_no_front_combination_requirement():
    series = _single_punch()
    windows, diagnostics, _, _ = synchronized_windows(
        {"side": series}, "rear_straight", "left", FeedbackConfig()
    )
    assert ("rear_straight", "finish") in windows["side"]
    assert not any(segment == "front_straight" for segment, _ in windows["side"])
    assert diagnostics["side"]["events"] == ["rear_straight"]


def test_side_punch_keeps_approved_gap_tolerance():
    times = np.arange(40) / 30
    reach = np.zeros(40)
    reach[6:30] = 0.6
    reach[15:19] = np.nan
    events = reach_events(
        reach, np.zeros(40), times, np.arange(5), FeedbackConfig(), rise=0.3, fall=0.15
    )
    assert [(event.start, event.peak, event.end, event.ending) for event in events] == [
        (6, 6, 30, "returned")
    ]


def test_projection_uses_same_absolute_timestamps_with_different_sampling():
    side, front = _single_punch(), _single_punch()
    retained = np.arange(15, len(front["times"]), 2)
    for key in ("landmarks", "valid_mask"):
        front[key] = front[key][retained]
    for key in ("times", "frames"):
        front[key] = [front[key][i] for i in retained]
    windows, diagnostics, alignment, _ = synchronized_windows(
        {"front": front, "side": side}, "rear_straight", "left", FeedbackConfig()
    )
    assert alignment["sideOffsetSeconds"] == 0
    assert diagnostics["front"]["projectedPhaseInstances"] == diagnostics["side"]["phaseInstances"]
    for key in [("rear_straight", "motion"), ("rear_straight", "finish")]:
        source_times = np.asarray(side["times"])[windows["side"][key]]
        target_times = np.asarray(front["times"])[windows["front"][key]]
        assert target_times.min() >= source_times.min() and target_times.max() <= source_times.max()


def test_combination_stage_keeps_its_index_with_an_unpaired_earlier_punch():
    series = _series(1)
    times = np.asarray(series["times"])
    inside = (times >= 0.6) & (times <= 0.9)
    series["landmarks"][inside, 15, 0] += 0.2 * (1 - np.abs(2 * (times[inside] - 0.6) / 0.3 - 1))
    _, diagnostics, _, _ = synchronized_windows(
        {"side": series}, "step_straight_combo", "left", FeedbackConfig()
    )
    side = diagnostics["side"]
    assert len(side["repetitions"]) == 1
    assert (
        len([event for event in side["eventDetails"] if event["segment"] == "front_straight"]) == 2
    )
    assert {instance["repetition"] for instance in side["phaseInstances"]} == {1}
    assert {instance["phase"] for instance in side["phaseInstances"]} >= {
        "prepare",
        "motion",
        "finish",
        "recovered",
    }
