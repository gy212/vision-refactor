"""公共动作识别与逐阶段对应的业务回归。"""

from copy import deepcopy

from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
import pytest

from core.evaluation import FeedbackConfig, action_timing, evaluate_series, motion_input
from core.evaluation import stance_windows
from core.evaluation.motion_input import prepare_motion_input
from core.evaluation.punch_events import reach_events
from core.evaluation.rule_checks import synchronized_windows
from core.evaluation.stance_windows import settled_stance_windows, stable_guard_windows
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
    assert checks["rear_straight:motion:front:guard_shoulder"]["status"] != "unable"
    assert "eventPairs" not in result["alignment"]


def test_source_visibility_is_checked_before_detection_and_missing_data_stays_unable(
    monkeypatch,
):
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
        {"front": _series(0), "side": _series(2)},
        "step_straight_combo",
        "left",
        FeedbackConfig(),
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


@pytest.mark.parametrize("extra_start", [0.6, 3.0])
def test_unpaired_punch_keeps_its_own_stages_before_or_after_combination(extra_start):
    series = _series(1)
    times = np.asarray(series["times"])
    inside = (times >= extra_start) & (times <= extra_start + 0.3)
    series["landmarks"][inside, 15, 0] += 0.2 * (
        1 - np.abs(2 * (times[inside] - extra_start) / 0.3 - 1)
    )
    _, diagnostics, _, _ = synchronized_windows(
        {"side": series}, "step_straight_combo", "left", FeedbackConfig()
    )
    side = diagnostics["side"]
    assert len(side["repetitions"]) == 1
    assert (
        len([event for event in side["eventDetails"] if event["segment"] == "front_straight"]) == 2
    )
    events = {event["eventId"]: event for event in side["eventDetails"]}
    assert set(events) == {"front_straight:1", "front_straight:2", "rear_straight:1"}
    for identity, event in events.items():
        instances = [
            instance for instance in side["phaseInstances"] if instance["eventId"] == identity
        ]
        assert {instance["phase"] for instance in instances} >= {"motion", "finish"}
        assert {instance["repetition"] for instance in instances} == {event["repetition"]}
    for combination in side["repetitions"]:
        assert combination["index"] == 1
        for key in ("frontEventId", "rearEventId"):
            assert combination[key] is None or combination[key] in events
    unpaired = (
        {"front_straight:1", "front_straight:2"} if extra_start < 1.5 else {"front_straight:2"}
    )
    assert set(side["unpairedEventIds"]) == unpaired
    assert {instance["phase"] for instance in side["phaseInstances"]} >= {
        "prepare",
        "motion",
        "finish",
        "recovered",
    }


@pytest.mark.parametrize("segment", ["front_straight", "rear_straight"])
def test_single_and_combined_punches_share_observed_events_and_stage_times(segment):
    series = _series(2)
    results = {}
    for action in (segment, "straight_combo", "step_straight_combo"):
        windows, diagnostics, _, _ = synchronized_windows(
            {"side": series}, action, "left", FeedbackConfig()
        )
        results[action] = (windows["side"], diagnostics["side"])
    reference_windows, reference = results[segment]
    assert len(reference["eventDetails"]) == 2
    assert {"prepare", "recovered"} <= {
        phase for current, phase in reference_windows if current == segment
    }
    for windows, diagnostics in results.values():
        observed = [e for e in diagnostics["eventDetails"] if e["segment"] == segment]
        assert observed == reference["eventDetails"]
        for phase in ("motion", "finish"):
            np.testing.assert_array_equal(
                windows[(segment, phase)], reference_windows[(segment, phase)]
            )


def test_missing_combination_member_preserves_the_observed_punch():
    windows, diagnostics, _, _ = synchronized_windows(
        {"side": _single_punch()}, "straight_combo", "left", FeedbackConfig()
    )
    assert "不完整" in diagnostics["side"]["reason"]
    assert ("front_straight", "finish") not in windows["side"]
    assert ("rear_straight", "finish") in windows["side"]
    assert ("rear_straight", "prepare") in windows["side"]
    assert ("rear_straight", "recovered") in windows["side"]


def test_unseen_direction_reference_does_not_create_a_punch():
    series = _single_punch()
    series["valid_mask"][:, 0] = False
    _, diagnostics, _, _ = synchronized_windows(
        {"side": series}, "rear_straight", "left", FeedbackConfig()
    )
    assert diagnostics["side"]["eventDetails"] == []
    assert "鼻点或髋点" in diagnostics["side"]["reason"]


def _reference_stable_guard_windows(
    lm: np.ndarray, mask: np.ndarray, torso: np.ndarray, edge: int, cfg: FeedbackConfig
) -> list[np.ndarray]:
    """逐帧语义参考实现，用于独立验证向量化护位滑窗的逐窗口一致性。"""
    n = len(lm)
    if edge <= 0 or n < edge:
        return []
    with np.errstate(all="ignore"):
        wrists = (lm[:, [15, 16], :2] - lm[:, [11, 12], :2]) / np.maximum(
            torso[:, None, None], 1e-4
        )
        expected = []
        for start in range(n - edge + 1):
            ix = np.arange(start, start + edge)
            if not (torso[ix] > 1e-4).all():
                continue
            visible = 0
            raised = False
            for arm, (sh, wr, hip) in enumerate(((11, 15, 23), (12, 16, 24))):
                if not mask[ix][:, [sh, wr, hip]].all():
                    continue
                visible += 1
                relative = wrists[ix, arm]
                w_dev = np.max(np.linalg.norm(relative - np.median(relative, axis=0), axis=1))
                h_dev = np.max(
                    np.linalg.norm(lm[ix, hip, :2] - np.median(lm[ix, hip, :2], axis=0), axis=1)
                )
                if (
                    w_dev > cfg.return_distance / 2
                    or h_dev > 0.15 * np.median(torso[ix])
                    or not (lm[ix, wr, 1] < lm[ix, hip, 1] + 0.1 * torso[ix]).all()
                ):
                    visible = 0
                    break
                raised |= bool(
                    (lm[ix, wr, 1] <= lm[ix, sh, 1] + cfg.guard_wrist_height * torso[ix]).all()
                )
            if visible and raised:
                expected.append(ix)
    return expected


_ADVERSARIAL_FLOATS = st.one_of(
    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    st.sampled_from(
        [
            0.0,
            -0.0,
            1e-4,
            1e-4 + 1e-7,
            1e-4 - 1e-7,
            0.25,
            0.5,
            1.0,
            float("nan"),
            float("inf"),
            float("-inf"),
            1e20,
            3e38,
            -3e38,
        ]
    ),
)


@settings(max_examples=550, deadline=None)
@given(
    t=st.integers(min_value=0, max_value=22),
    edge=st.integers(min_value=1, max_value=9),
    lm_dtype=st.sampled_from([np.float32, np.float64]),
    torso_dtype=st.sampled_from([np.float32, np.float64]),
    readonly=st.booleans(),
    strided=st.booleans(),
    data=st.data(),
)
def test_stable_guard_windows_matches_framewise_reference_under_adversarial_inputs(
    t: int,
    edge: int,
    lm_dtype: type,
    torso_dtype: type,
    readonly: bool,
    strided: bool,
    data: st.DataObject,
):
    cfg = FeedbackConfig()
    lm = np.zeros((t, 33, 4), dtype=lm_dtype)
    mask = np.zeros((t, 33), dtype=bool)
    if t > 0:
        pts = data.draw(arrays(lm_dtype, (t, 6, 2), elements=_ADVERSARIAL_FLOATS))
        lm[:, [11, 12, 15, 16, 23, 24], :2] = pts
        arm_mode = data.draw(st.sampled_from(["all_false", "left_only", "right_only", "random"]))
        if arm_mode == "left_only":
            mask[:, [11, 15, 23]] = True
        elif arm_mode == "right_only":
            mask[:, [12, 16, 24]] = True
        elif arm_mode == "random":
            mask[:, [11, 12, 15, 16, 23, 24]] = data.draw(
                arrays(bool, (t, 6), elements=st.booleans())
            )
        torso = data.draw(arrays(torso_dtype, (t,), elements=_ADVERSARIAL_FLOATS))
    else:
        torso = np.zeros((0,), dtype=torso_dtype)
    if strided and t > 0:
        lm = np.repeat(lm, 2, axis=0)[::2]
        mask = np.repeat(mask, 2, axis=0)[::2]
        torso = np.repeat(torso, 2, axis=0)[::2]
    if readonly:
        lm.setflags(write=False)
        mask.setflags(write=False)
        torso.setflags(write=False)

    actual = stable_guard_windows(lm, mask, torso, edge, cfg)
    expected = _reference_stable_guard_windows(lm, mask, torso, edge, cfg)

    assert isinstance(actual, tuple)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert not got.flags.writeable
        np.testing.assert_array_equal(got, want)


def test_stable_guard_windows_preserves_ieee754_nan_demorgan_and_arm_semantics():
    cfg = FeedbackConfig()
    edge = 4

    def base_guard(t: int = 6):
        lm = np.zeros((t, 33, 4), dtype=np.float64)
        lm[:, 11, :2] = (0.4, 0.30)
        lm[:, 12, :2] = (0.6, 0.30)
        lm[:, 15, :2] = (0.42, 0.28)
        lm[:, 16, :2] = (0.58, 0.28)
        lm[:, 23, :2] = (0.42, 0.60)
        lm[:, 24, :2] = (0.58, 0.60)
        mask = np.ones((t, 33), dtype=bool)
        torso = np.full(t, 0.30, dtype=np.float64)
        return lm, mask, torso

    # hip_x 或 wr_x 为 NaN 但 y 坐标满足条件时，原语义 NaN > thr 为 False，窗口必须保留。
    for joint in (23, 15):
        lm, mask, torso = base_guard()
        lm[2, joint, 0] = np.nan
        got = stable_guard_windows(lm, mask, torso, edge, cfg)
        want = _reference_stable_guard_windows(lm, mask, torso, edge, cfg)
        assert len(got) == len(want) == 3
        for a, e in zip(got, want):
            np.testing.assert_array_equal(a, e)

    # y 坐标或躯干长度含 NaN 时，必须拒绝对应窗口。
    lm, mask, torso = base_guard()
    lm[:, 15, 1] = np.nan
    assert stable_guard_windows(lm, mask, torso, edge, cfg) == ()

    lm, mask, torso = base_guard()
    torso[:] = np.nan
    assert stable_guard_windows(lm, mask, torso, edge, cfg) == ()

    # 单臂可见时，远侧遮挡臂的异常坐标不得否定近侧稳定护位；但双臂可见时移动手否定稳定。
    lm, mask, torso = base_guard()
    lm[:, 16, 0] = np.linspace(0.5, 1.5, len(lm))
    assert stable_guard_windows(lm, mask, torso, edge, cfg) == ()
    mask[:, [12, 16, 24]] = False
    assert len(stable_guard_windows(lm, mask, torso, edge, cfg)) == 3
    assert stable_guard_windows(lm, mask, torso, 0, cfg) == ()

    # T=0、T=1、T < edge、全 False 掩码均安全返回空或精确单窗。
    assert (
        stable_guard_windows(
            np.zeros((0, 33, 4), dtype=np.float32),
            np.zeros((0, 33), dtype=bool),
            np.zeros((0,), dtype=np.float64),
            edge,
            cfg,
        )
        == ()
    )
    lm1, mask1, torso1 = base_guard(t=1)
    got1 = stable_guard_windows(lm1, mask1, torso1, 1, cfg)
    assert len(got1) == 1 and got1[0].tolist() == [0]
    assert stable_guard_windows(lm1, mask1, torso1, edge, cfg) == ()
    lm3, mask3, torso3 = base_guard(t=3)
    assert stable_guard_windows(lm3, mask3, torso3, edge, cfg) == ()
    mask_none = np.zeros_like(mask3)
    assert stable_guard_windows(lm3, mask_none, torso3, 2, cfg) == ()


def test_stable_guard_and_settled_windows_enforce_readonly_contract():
    series = _series(0, seconds=3)
    lm = series["landmarks"].copy()
    mask = series["valid_mask"].copy()
    times = np.asarray(series["times"], dtype=float)
    lm.setflags(write=False)
    mask.setflags(write=False)
    times.setflags(write=False)
    readonly_series = {**series, "landmarks": lm, "valid_mask": mask, "times": times}

    cfg = FeedbackConfig()
    prepared = prepare_motion_input(readonly_series, cfg)
    assert isinstance(prepared.stable, tuple)
    assert len(prepared.stable) > 0
    for win in prepared.stable:
        assert not win.flags.writeable
        with pytest.raises(ValueError):
            win[0] = 999

    windows = {("stance", "start"): np.arange(7), ("stance", "end"): np.arange(70, 77)}
    settled = settled_stance_windows(
        readonly_series, windows, [{"start": 2.0}], cfg, stable=prepared.stable
    )
    assert ("stance", "start") in settled
    with pytest.raises(ValueError):
        settled[("stance", "start")][0] = 999


@pytest.mark.parametrize(
    "bad_times",
    [
        np.full(30, 1.0),
        np.linspace(2.0, 0.0, 30),
        np.full(30, np.nan),
        np.full(30, np.inf),
        np.full(30, -np.inf),
        np.arange(30, dtype=float) * 1e-320,
    ],
)
def test_settled_stance_windows_handles_degenerate_timestamps_without_crashing(
    bad_times,
):
    series = _series(0, seconds=1)
    series["times"] = bad_times
    cfg = FeedbackConfig()
    windows = {
        ("stance", "start"): np.arange(5),
        ("stance", "end"): np.arange(20, 25),
        ("rear_straight", "motion"): np.arange(10, 15),
    }

    prepared = prepare_motion_input(series, cfg)
    assert prepared.stable == ()

    direct = settled_stance_windows(series, windows, [{"start": 0.5}], cfg)
    reused = settled_stance_windows(series, windows, [{"start": 0.5}], cfg, stable=prepared.stable)
    assert ("stance", "start") not in direct
    assert ("stance", "end") not in direct
    assert ("rear_straight", "motion") in direct
    assert set(direct) == set(reused)
    for key in direct:
        np.testing.assert_array_equal(direct[key], reused[key])


def test_shared_timeline_computes_stable_guard_windows_once_per_view(monkeypatch):
    calls: list[str] = []
    real_stable = stance_windows.stable_guard_windows

    def spy_motion(*args, **kwargs):
        calls.append("motion_input")
        return real_stable(*args, **kwargs)

    def spy_settled(*args, **kwargs):
        calls.append("stance_windows")
        return real_stable(*args, **kwargs)

    monkeypatch.setattr(motion_input, "stable_guard_windows", spy_motion)
    monkeypatch.setattr(stance_windows, "stable_guard_windows", spy_settled)

    views = {"front": _single_punch(), "side": _single_punch()}
    windows, diagnostics, _, _ = synchronized_windows(
        views, "rear_straight", "left", FeedbackConfig()
    )
    assert calls == ["motion_input", "motion_input"]
    assert ("stance", "start") in windows["front"]
    assert ("stance", "start") in windows["side"]
    with pytest.raises(ValueError):
        windows["side"][("stance", "start")][0] = 999
    assert len(diagnostics["side"]["eventDetails"]) == 1
