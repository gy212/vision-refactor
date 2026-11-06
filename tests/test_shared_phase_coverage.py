"""公开入口必须用同一动作实例的实际录像覆盖，缺失次数不能被相邻动作代替。"""

from copy import deepcopy

import numpy as np
import pytest

from core.evaluation import FeedbackConfig, evaluate_series, judge_views
from tests.test_step_straight_combo import _series


def _rapid_punches():
    series = _series(0, seconds=4)
    for start in (30, 35, 40, 45):
        series["landmarks"][start : start + 4, 16, 0] += 0.18
        series["landmarks"][start : start + 4, 14, 0] += 0.09
    return series


def _select(series, indices):
    result = deepcopy(series)
    for key in ("landmarks", "valid_mask"):
        result[key] = result[key][indices]
    for key in ("times", "frames"):
        result[key] = [result[key][i] for i in indices]
    return result


@pytest.mark.parametrize(
    "removed,repetition",
    [
        (range(44, 120), 4),  # 末尾整次缺失，前一次距离不足旧的 0.4 秒容差。
        (range(34, 39), 2),  # 录像头尾均完整，但中间整次缺失。
        ([31], 1),  # 同次阶段内部缺帧：相邻数组下标不代表原视频时间连续。
        ([44, 45, 46], 4),  # 最后一次只剩末帧，不能按部分画面宣布完整通过。
        ([44], 4),  # 同帧率本应存在的阶段首帧不能用采样误差掩盖。
    ],
)
def test_missing_instance_blocks_practice_and_shared_judging(removed, repetition):
    side = _rapid_punches()
    retained = np.setdiff1d(np.arange(len(side["times"])), list(removed))
    views = {"front": _select(side, retained), "side": side}
    practice = evaluate_series("rear_straight", "left", views)
    judged = judge_views(views, "rear_straight", "left", FeedbackConfig())
    check_id = "rear_straight:finish:front:guard_low"
    check = next(item for item in practice["checks"] if item["id"] == check_id)
    assert check["status"] == "unable"
    assert f"第 {repetition} 次" in check["reason"]
    assert "正面录像" in check["reason"]
    front = next(item for item in judged["front"][0] if item.check_id == check_id)
    assert front.state == "unable" and front.skip_reason == "phase_not_found"
    assert f"第 {repetition} 次" in front.detail
    side_check = next(item for item in judged["side"][0] if item.check_id == check_id)
    assert side_check.state == "pass"
    # 仍保存实际已有的测量，不将整段录像伪装成没有任何画面。
    assert front.valid_frames > 0 and front.measurements
    coverage = practice["alignment"]["phases"]["rear_straight:finish"]
    assert not coverage["reliable"]
    missing = [i for i in coverage["views"]["front"]["instances"] if not i["covered"]]
    assert [i["repetition"] for i in missing] == [repetition]
    assert coverage["views"]["side"]["complete"]


def test_front_only_rule_uses_same_missing_instance_gate():
    side = _series(2)
    front = _select(side, np.flatnonzero(np.asarray(side["times"]) < 3.9))
    views = {"front": front, "side": side}
    practice = evaluate_series("step_straight_combo", "left", views)
    judged = judge_views(views, "step_straight_combo", "left", FeedbackConfig())
    check_id = "rear_straight:finish:front:guard_shoulder"
    check = next(item for item in practice["checks"] if item["id"] == check_id)
    assert check["fusion"]["method"] == "primary_view"
    assert check["status"] == "unable" and "第 2 次" in check["reason"]
    exported = next(item for item in judged["front"][0] if item.check_id == check_id)
    assert exported.state == "unable" and "第 2 次" in exported.detail
    source_only = next(
        item
        for item in practice["checks"]
        if item["id"] == "rear_straight:finish:rear:shoulder_level"
    )
    assert source_only["status"] == "not_observed"


@pytest.mark.parametrize("stride,offset", [(1, 0.01), (2, 0.0), (2, 0.01)])
def test_real_sampling_boundaries_cover_the_same_instances(stride, offset):
    side = _rapid_punches()
    front = _select(side, np.arange(0, len(side["times"]), stride))
    front["times"] = [time + offset for time in front["times"]]
    views = {"front": front, "side": side}
    practice = evaluate_series("rear_straight", "left", views)
    judged = judge_views(views, "rear_straight", "left", FeedbackConfig())
    check_id = "rear_straight:finish:front:guard_low"
    check = next(item for item in practice["checks"] if item["id"] == check_id)
    assert check["status"] == "not_observed"
    assert next(item for item in judged["front"][0] if item.check_id == check_id).state == "pass"
    assert practice["alignment"]["phases"]["rear_straight:finish"]["reliable"]
    projected = practice["diagnostics"]["front"]["projectedPhaseInstances"]
    assert projected == practice["diagnostics"]["side"]["phaseInstances"]
