"""四个入口的最终 JSON 保留同一次识别的完整动作记录，包括空识别和缺失阶段。"""

from copy import deepcopy
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from batch.batch_dual_compare import _write_pose33_jsonl
from core.evaluation import (
    FeedbackConfig,
    compare_dual_templates,
    evaluate_series,
    judge_view,
    judge_views,
)
from core.evaluation import action_timing
from core.evaluation.scoring.exam_pool import rule_reports, score_exam_pools
from core.foundation import to_jsonable
from core.pose import Pose33RawSeries
from core.recording import write_result_json
from core.recording.steps import result_payload
from tests import golden_harness as H
from tests.test_rule_evidence_entries import ACTION, FIXTURES, _batch, _paired_views
from tests.test_step_straight_combo import _series


def _views(case):
    views = _paired_views()
    if case == "zero_events":
        views = {view: _series(0) for view in views}
    elif case == "missing_phase":
        # 来源的前后手都完整；正面只录到第一拳中途。
        for key in ("landmarks", "valid_mask", "frames", "times"):
            views["front"][key] = views["front"][key][:49]
    return views


def _exam_result(views):
    def raw_series(video, **_kwargs):
        series = views[Path(video).stem]
        return Pose33RawSeries(
            series["landmarks"],
            {
                "fps": 30.0,
                "width": 720,
                "height": 720,
                "frame_count": len(series["frames"]),
                "start_frame": 0,
            },
        )

    def compare(templates, video, **_kwargs):
        return [
            SimpleNamespace(
                score=0.8, start_frame=0, end_frame=len(views[Path(video).stem]["frames"]) - 1
            )
            for _ in templates
        ]

    template = Path("上步前后手直拳组合.npz")
    return score_exam_pools(
        [template] if "front" in views else [],
        [template] if "side" in views else [],
        Path("front.mp4"),
        Path("side.mp4"),
        compare=compare,
        raw_series=raw_series,
        stop_evt=Event(),
    )


@pytest.mark.parametrize("case", ["complete", "zero_events", "missing_phase"])
def test_final_files_keep_one_complete_record_for_each_paired_action(case, tmp_path, monkeypatch):
    views = _views(case)
    calls = []
    detect = action_timing.step_combo_windows

    def detect_once(*args, **kwargs):
        calls.append(args[0])
        return detect(*args, **kwargs)

    monkeypatch.setattr(action_timing, "step_combo_windows", detect_once)
    practice = evaluate_series(ACTION, "left", deepcopy(views))
    assert len(calls) == 1
    judged = judge_views(deepcopy(views), ACTION, "left", FeedbackConfig())
    assert len(calls) == 2
    exam = _exam_result(deepcopy(views))
    assert len(calls) == 3
    batch = _batch(deepcopy(views))
    assert len(calls) == 4

    practice_path, exam_path, batch_path = (
        tmp_path / name for name in ("record.json", "result.json", "compare_results.jsonl")
    )
    practice_path.write_text(json.dumps(to_jsonable(practice)), encoding="utf-8")
    write_result_json(exam_path, {"result": result_payload(exam)})
    _write_pose33_jsonl(batch_path, [(batch, {}, [], [], {})])
    expected = json.loads(practice_path.read_text())["sharedActionRecord"]
    exam_reports = json.loads(exam_path.read_text())["result"]["rule_checks"]
    batch_saved = json.loads(batch_path.read_text())
    assert judged["front"][1]["sharedActionRecord"] is judged["side"][1]["sharedActionRecord"]
    assert to_jsonable(judged["front"][1]["sharedActionRecord"]) == expected
    assert exam_reports[0]["sharedActionRecord"] == expected
    assert [report["sharedActionRecordReport"] for report in exam_reports] == [0, 0]
    assert "sharedActionRecord" not in exam_reports[1]
    assert batch_saved["shared_action_records"] == [expected]
    assert all("sharedActionRecord" not in check for check in batch_saved["front_rule_checks"])
    assert expected["sourceSelection"] == practice["alignment"]["sourceSelection"]
    assert expected["eventDetails"] == practice["diagnostics"]["side"]["eventDetails"]
    assert expected["phaseInstances"] == practice["diagnostics"]["side"]["phaseInstances"]
    assert expected["phases"] == to_jsonable(practice["alignment"]["phases"])
    if case == "zero_events":
        assert expected["eventDetails"] == expected["phaseInstances"] == []
        assert expected["timingSource"] == "side"
        assert expected["views"]["side"]["reason"]
    else:
        assert len(expected["eventDetails"]) == 2
        assert len(expected["phaseInstances"]) >= 4


@pytest.mark.parametrize("view", ["front", "side"])
def test_single_view_and_missing_timing_source_still_save_the_record(view, tmp_path):
    series = _series(1)
    practice = evaluate_series(ACTION, "left", {view: deepcopy(series)})
    _, diagnostics = judge_view(deepcopy(series), view, ACTION, "left", FeedbackConfig())
    exam = _exam_result({view: deepcopy(series)})
    path = tmp_path / "result.json"
    write_result_json(path, {"result": result_payload(exam)})
    record = practice["sharedActionRecord"]
    assert diagnostics["sharedActionRecord"] == record
    assert json.loads(path.read_text())["result"]["rule_checks"][0][
        "sharedActionRecord"
    ] == to_jsonable(record)
    assert record["inputViews"] == [view]
    if view == "front":
        assert record["timingSource"] is None
        assert record["sourceSelection"]["selectedView"] is None
        assert record["eventDetails"] == record["phaseInstances"] == []
        assert record["reason"]
    else:
        assert len(record["eventDetails"]) == 2


def test_single_video_two_segments_save_their_independent_records(tmp_path):
    video = "records://single.mp4"
    H.clear_registry()
    H.register_video(video, _series(2)["landmarks"], width=720, height=720)
    try:
        with H.replay_context():
            result = compare_dual_templates(
                FIXTURES / "front_template.npz",
                FIXTURES / "side_template.npz",
                video,
                pose_variant="full",
                enable_rules=True,
                rule_action=ACTION,
            )
    finally:
        H.clear_registry()
    path = tmp_path / "compare_results.jsonl"
    _write_pose33_jsonl(path, [(result, {}, [], [], {})])
    records = json.loads(path.read_text())["shared_action_records"]
    assert [record["inputViews"] for record in records] == [["front"], ["side"]]
    assert records[0]["timingSource"] is None
    assert records[1]["timingSource"] == "side"


def test_exam_repeated_actions_reference_their_own_record():
    views = _paired_views()
    targets = {
        view: [
            {
                "name": "上步前后手直拳组合",
                "action": ACTION,
                "view": view,
                "series": deepcopy(series),
                "start": 0,
                "end": len(series["frames"]) - 1,
            }
            for _ in range(2)
        ]
        for view, series in views.items()
    }
    reports = rule_reports(targets["front"], targets["side"], FeedbackConfig())
    assert [report["sharedActionRecordReport"] for report in reports] == [0, 1, 0, 1]
    assert ["sharedActionRecord" in report for report in reports] == [True, True, False, False]
