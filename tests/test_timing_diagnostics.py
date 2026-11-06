"""动作次数和对应关系由共同说明进入文本、HTML 与 Tk。"""

from copy import deepcopy

import pytest

from core.reporting import (
    build_report_model,
    format_report,
    format_timing_diagnostics,
    render_html_report,
)
from tests.test_feedback_html_export import make_html_test_record


def _diagnostics():
    events = [
        {"segment": "rear_straight", "start": 1.0, "peak": 1.2, "end": 1.4, "ending": "returned"},
        {"segment": "rear_straight", "start": 2.0, "peak": 2.2, "end": 2.4, "ending": "held"},
    ]
    diagnostic = {
        "bodyValidFrames": 90,
        "totalFrames": 90,
        "jointValidRatios": [1.0] * 33,
        "reason": "已检测到动作",
    }
    views = {
        "front": {**diagnostic, "eventDetails": events},
        "side": {**diagnostic, "eventDetails": [events[0]]},
    }
    alignment = {
        "method": "video_timestamps",
        "sideOffsetSeconds": 0.0,
        "reliable": False,
        "reason": "正侧面检测到的动作次数不一致",
        "eventCounts": {"front": {"rear_straight": 2}, "side": {"rear_straight": 1}},
        "eventPairs": [{"segment": "rear_straight", "frontIndex": 0, "sideIndex": 0}],
        "unmatchedEvents": [{"view": "front", "index": 1, **events[1]}],
        "phases": {"rear_straight:finish": {"reliable": False, "reason": "该阶段次数不一致"}},
    }
    return views, alignment


def _record(diagnostics, alignment):
    record = make_html_test_record(action="rear_straight")
    record["result"].update(
        diagnostics=diagnostics, alignment=alignment, capture={}, summary="分析完成"
    )
    return record


def test_saved_event_counts_and_times_enter_text_and_final_html():
    diagnostic, alignment = _diagnostics()
    record = _record(diagnostic, alignment)
    original = deepcopy(record)
    expected = format_timing_diagnostics(diagnostic, alignment)

    text = format_report(record)
    html = render_html_report(record)

    assert all(line in text and line in html for line in expected)
    assert "正面独立检测：后手直拳2次" in html
    assert "侧面独立检测：后手直拳1次" in html
    assert "开始2.000秒，峰值2.200秒，结束2.400秒；未观察到收回" in html
    assert "未独立对应：正面事件2" in html
    assert "已对应：后手直拳，正面事件1与侧面事件1" in html
    assert "阶段对应受限：后手直拳/动作结束瞬间" in html
    assert record == original


def test_projected_view_is_not_reported_as_independent_zero_count():
    diagnostic, alignment = _diagnostics()
    events = diagnostic["side"]["eventDetails"]
    diagnostic["front"].update(eventDetails=[], projectedEventDetails=events, timingSource="side")
    alignment.update(
        eventCounts={"front": {}, "side": {"rear_straight": 1}},
        eventPairs=[],
        unmatchedEvents=[{"view": "side", "index": 0, **events[0]}],
        reliable=True,
        reason="",
        phases={},
    )

    lines = "\n".join(format_timing_diagnostics(diagnostic, alignment))

    assert "正面阶段时间来自侧面，本路未独立计次" in lines
    assert "正面共用事件1" in lines
    assert "供正面共用阶段时间，未作独立事件配对" in lines
    assert "正面独立检测：0次" not in lines
    assert "次数不一致" not in lines


def test_legacy_missing_event_data_is_not_fabricated():
    assert format_timing_diagnostics({"front": {"reason": "旧说明"}}, {}) == []
    lines = format_timing_diagnostics({}, {"method": "video_timestamps"})
    assert lines == ["时间对应：录像时间。"]
    lines = format_timing_diagnostics({"front": {"eventDetails": []}}, {})
    assert lines == ["正面独立检测：0次。"]


def test_shared_timeline_is_reported_once_in_text_html_and_tk_model():
    diagnostic, _ = _diagnostics()
    events = diagnostic["side"]["eventDetails"]
    diagnostic["front"].update(eventDetails=[], projectedEventDetails=events, timingSource="side")
    diagnostic["side"]["timingSource"] = "own_view"
    alignment = {
        "method": "shared_clock",
        "timingSource": "side",
        "sourceSelection": {"reason": "按动作机位优先级及所需点可见性选择"},
        "eventCounts": {"side": {"rear_straight": 1}},
        "phases": {},
        "reliable": True,
    }
    record = _record(diagnostic, alignment)
    original = deepcopy(record)
    lines = format_timing_diagnostics(diagnostic, alignment)
    for output in (format_report(record), render_html_report(record)):
        assert all(line in output for line in lines)
        assert "动作次数和阶段时间统一来自侧面" in output
        assert "共用动作记录：后手直拳1次" in output
        assert "独立检测" not in output and "未独立对应" not in output
    model = build_report_model(record)
    assert format_timing_diagnostics(model.diagnostics, model.alignment) == lines
    assert record == original


def test_final_html_escapes_recorded_event_names_and_reasons():
    diagnostic, alignment = _diagnostics()
    injected = '<script>alert("event")</script>'
    diagnostic["front"]["eventDetails"][0]["segment"] = injected
    alignment["reason"] = injected

    html = render_html_report(_record(diagnostic, alignment))

    assert "<script>" not in html
    assert "&lt;script&gt;alert" in html


def test_tk_uses_the_same_timing_description_and_no_invented_visibility():
    pytest.importorskip("tkinter")
    from apps.tk.views.practice_result.diagnostics import diagnostics_lines

    diagnostic, alignment = _diagnostics()
    model = build_report_model(_record(diagnostic, alignment))
    lines = diagnostics_lines(model)

    assert all(line in lines for line in format_timing_diagnostics(diagnostic, alignment))
    old_model = build_report_model(_record({}, {}))
    old_text = "\n".join(diagnostics_lines(old_model))
    assert "未记录" in old_text
    assert "98%" not in old_text and "95%" not in old_text and "正常" not in old_text
