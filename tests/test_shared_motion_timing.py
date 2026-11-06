"""公共动作识别与逐阶段对应的业务回归。"""

from copy import deepcopy

import numpy as np
import pytest

from core.evaluation import FeedbackConfig, evaluate_series
from core.evaluation.rule_checks import synchronized_windows
from core.evaluation.phase_alignment import event_correspondence, share_phase_timing
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


def test_count_conflict_keeps_independent_start_and_blocks_unmatched_motion():
    result = evaluate_series("rear_straight", "left", {"front": _series(2), "side": _series(1)})
    assert not result["alignment"]["reliable"]
    assert result["alignment"]["eventCounts"] == {
        "front": {"rear_straight": 2},
        "side": {"rear_straight": 1},
    }
    assert len(result["alignment"]["unmatchedEvents"]) == 1
    checks = {check["id"]: check for check in result["checks"]}
    assert checks["stance:start:front:stance_elbow"]["status"] in {"candidate", "not_observed"}
    assert checks["rear_straight:finish:rear:punch_extension"]["status"] == "unable"
    assert "次数不一致" in checks["rear_straight:finish:rear:punch_extension"]["reason"]


def test_start_stance_still_requires_matching_times_for_two_view_rule():
    front, side = _series(0), _series(0)
    side["times"] = [time + 2 for time in side["times"]]
    result = evaluate_series("stance", "left", {"front": front, "side": side})
    elbow = next(
        check for check in result["checks"] if check["id"] == "stance:start:front:stance_elbow"
    )
    assert elbow["status"] == "unable"
    assert "时间不对应" in elbow["reason"]


def test_projected_events_are_not_reported_as_independent_observations():
    _, diagnostics, alignment, sources = synchronized_windows(
        {"front": _series(0), "side": _series(2)}, "step_straight_combo", "left", FeedbackConfig()
    )
    assert diagnostics["front"]["eventDetails"] == []
    assert diagnostics["front"]["projectedEventDetails"] == diagnostics["side"]["eventDetails"]
    assert diagnostics["front"]["timingSource"] == "side"
    assert sources["front"] == "side_synchronized_timing"
    assert alignment["eventCounts"]["front"] == {}


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


@pytest.mark.parametrize("source,target", [("front", "side"), ("side", "front")])
def test_projected_event_times_use_target_clock(source, target):
    views = {"front": {"times": np.arange(30) / 10}, "side": {"times": np.arange(30) / 10 + 0.2}}
    windows = {view: {("stance", "start"): np.arange(3)} for view in views}
    windows[source][("rear_straight", "motion")] = np.arange(10, 21)
    event = {
        "segment": "rear_straight",
        "start": float(views[source]["times"][10]),
        "peak": float(views[source]["times"][15]),
        "end": float(views[source]["times"][20]),
        "ending": "returned",
    }
    instance = {
        "segment": "rear_straight",
        "phase": "motion",
        "repetition": 1,
        "start": event["start"],
        "end": event["end"],
    }
    diagnostics = {
        source: {
            "events": ["rear_straight"],
            "eventDetails": [event],
            "phaseInstances": [instance],
        },
        target: {"events": [], "eventDetails": []},
    }
    alignment = {"reliable": True, "sideOffsetSeconds": -0.2}
    sources = share_phase_timing(views, windows, alignment, diagnostics, FeedbackConfig())
    projected = diagnostics[target]["projectedEventDetails"][0]
    for key, index in (("start", 10), ("peak", 15), ("end", 20)):
        assert projected[key] == pytest.approx(views[target]["times"][index])
        assert event[key] == pytest.approx(views[source]["times"][index])
    assert sources[target] == source + "_synchronized_timing"
    phase = diagnostics[target]["projectedPhaseInstances"][0]
    assert phase["start"] == pytest.approx(views[target]["times"][10])
    assert phase["end"] == pytest.approx(views[target]["times"][20])


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


@pytest.mark.parametrize("front_peaks,side_peaks", [([1.0, 1.3], [1.2]), ([1.2], [1.0, 1.3])])
def test_event_correspondence_requires_unique_match_from_both_views(front_peaks, side_peaks):
    def observed(peaks):
        return {
            "eventDetails": [
                {
                    "segment": "rear_straight",
                    "start": peak - 0.1,
                    "peak": peak,
                    "end": peak + 0.1,
                    "ending": "returned",
                }
                for peak in [*peaks, 3.0]
            ]
        }

    result = event_correspondence(
        {"front": observed(front_peaks), "side": observed(side_peaks)}, 0, 0.4
    )
    assert result["eventPairs"] == [
        {"segment": "rear_straight", "frontIndex": len(front_peaks), "sideIndex": len(side_peaks)}
    ]
    assert len(result["unmatchedEvents"]) == 3
    assert all(event["reason"] == "对应候选不唯一" for event in result["unmatchedEvents"])
