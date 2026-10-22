# -*- coding: utf-8 -*-
"""录像时间轴 ``Timeline`` 与采集记录 ``capture.json``（设计文档 §8.8，D-18 第 3 层）。

核心要求：输出第 k 帧只显示采集于该时刻之前（含）最近的源帧，新画面不复制到过去（PR #34 的缺陷）。
"""
from __future__ import annotations

import json
import math
import random

import numpy as np
import pytest

from core.media import (
    CAPTURE_FILE,
    CapturedFrame,
    SegmentLayout,
    Timeline,
    capture_document,
    open_video,
    open_video_writer,
    read_capture_record,
    write_capture_record,
)

S = 1_000_000_000
FPS = 30.0
PERIOD = S / FPS
T0 = 100 * S


def frame(seq: int, t_ns: float, value: int = 0, role: str = "primary") -> CapturedFrame:
    return CapturedFrame(role, seq, int(t_ns), np.full((48, 64, 3), value, np.uint8))


def run(timeline: Timeline, frames, *, start=T0, stop=None):
    """开录于 ``start``（首帧取 ``frames[0]``），依次推入其余帧，结束于 ``stop``；返回全部输出帧。"""
    timeline.start(start, frames[0])
    out = []
    for item in frames[1:]:
        out += timeline.push(item)
    out += timeline.stop(stop if stop is not None else frames[-1].captured_at_ns + 1)
    return out


def assert_never_future(timeline: Timeline, sources: list[CapturedFrame], start: int) -> None:
    """每个输出槽的源帧采集于该槽时刻之前（含），且是那时最近的一帧（单个录制区间）。"""
    times = sorted(f.captured_at_ns for f in sources)
    for k, (_, t_ns) in enumerate(timeline.slot_sources):
        wall = start + k * PERIOD
        assert t_ns <= wall + 1e-6, f"slot {k} shows a frame from the future"
        latest = max(t for t in times if t <= wall + 1e-6)
        assert t_ns == latest, f"slot {k} skipped a newer past frame"


# --------------------------------------------------------------------------- #
# 内容出现的时间
# --------------------------------------------------------------------------- #


def test_black_then_white_is_not_copied_backwards():
    """评审用例：黑帧 100.0 s、白帧 101.0 s，30 fps：前 30 帧黑，白从第 31 帧开始。"""
    timeline = Timeline(FPS)
    out = run(timeline, [frame(1, T0, 0), frame(2, T0 + S, 255)], stop=T0 + 2 * S)
    values = [int(item.bgr[0, 0, 0]) for item in out]
    assert len(values) == 60
    assert values[:30] == [0] * 30
    assert values[30:] == [255] * 30


def test_black_then_white_decoded_from_a_written_video(tmp_path):
    """解码检查：经真实 writer 写出再读回，内容出现的时间与上一项相同。"""
    timeline = Timeline(FPS)
    out = run(timeline, [frame(1, T0, 0), frame(2, T0 + S, 255)], stop=T0 + 2 * S)
    writer, path, _codec = open_video_writer(tmp_path / "bw.mp4", fps=FPS, size=(64, 48))
    try:
        for item in out:
            writer.write(item.bgr)
    finally:
        writer.release()
    with open_video(path) as source:
        assert abs(source.meta.fps_reported - FPS) < 0.01
        decoded = [float(item.bgr.mean()) for item in source]
    assert len(decoded) == 60
    assert all(value < 40 for value in decoded[:30])
    assert all(value > 215 for value in decoded[30:])


@pytest.mark.parametrize("seed", range(8))
def test_random_jitter_and_rates_never_use_a_future_frame(seed):
    rng = random.Random(seed)
    rate = rng.choice([5.0, 8.0, 12.0, 20.0, 30.0, 45.0, 60.0])
    t = T0 - rng.uniform(0, S / rate)
    sources = []
    while t < T0 + 3 * S:
        sources.append(frame(len(sources) + 1, t))
        t += S / rate * rng.uniform(0.6, 1.4)
    timeline = Timeline(FPS)
    run(timeline, sources, stop=T0 + 3 * S)
    assert_never_future(timeline, sources, T0)


def test_late_frame_captured_before_start_replaces_the_seed():
    """开录后才处理到、但采集于边界之前的帧比 seed 新：第一个槽用它。"""
    timeline = Timeline(FPS)
    timeline.start(T0, frame(1, T0 - 50_000_000, 10))
    timeline.push(frame(2, T0 - 10_000_000, 20))
    out = timeline.push(frame(3, T0 + 40_000_000, 30))
    assert [int(item.bgr[0, 0, 0]) for item in out] == [20, 20]   # 槽 0（T0）与槽 1（T0+33ms）


# --------------------------------------------------------------------------- #
# 时长、补帧与跳帧
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rate", [8.0, 12.0, 15.0, 20.0])
def test_slow_source_keeps_real_duration_by_repeating_past_frames(rate):
    duration = 10 * S
    sources = [frame(i + 1, T0 + i * S / rate) for i in range(int(10 * rate) + 1)]
    timeline = Timeline(FPS)
    out = run(timeline, sources, stop=T0 + duration)
    assert len(out) == math.ceil(duration / PERIOD)          # 时长误差不超过一个输出帧
    stats = timeline.stats
    assert stats.slots == len(out)
    assert stats.used_frames == stats.source_frames - stats.skipped_frames
    assert stats.repeated_slots == stats.slots - stats.used_frames
    assert stats.max_source_gap_ns == pytest.approx(S / rate, abs=2)


def test_fast_source_skips_frames_and_counts_them():
    sources = [frame(i + 1, T0 + i * S / 60.0) for i in range(601)]
    timeline = Timeline(FPS)
    out = run(timeline, sources, stop=T0 + 10 * S)
    assert len(out) == 300
    assert timeline.stats.repeated_slots == 0
    assert timeline.stats.skipped_frames == pytest.approx(300, abs=2)


def test_nominal_rate_offset_from_the_grid_has_no_repeats_under_small_jitter():
    """源帧落在两个输出时刻之间时，±5 ms 抖动不造成补帧或跳帧。"""
    rng = random.Random(1)
    sources = [frame(i + 1, T0 + (i - 0.5) * PERIOD + rng.uniform(-5e6, 5e6)) for i in range(300)]
    timeline = Timeline(FPS)
    run(timeline, sources, stop=T0 + 10 * S)
    assert timeline.stats.repeated_slots == 0
    assert timeline.stats.skipped_frames == 0


# --------------------------------------------------------------------------- #
# 暂停、继续与边界
# --------------------------------------------------------------------------- #


def test_pause_excludes_the_paused_interval_and_resume_uses_a_fresh_frame():
    timeline = Timeline(FPS)
    timeline.start(T0, frame(1, T0, 1))
    out = timeline.push(frame(2, T0 + S, 2))
    out += timeline.pause(T0 + 2 * S)
    assert len(out) == 60
    # 暂停期间的帧不写出；继续时用边界之前最近的那一帧
    for seq, t in ((3, T0 + 3 * S, ), (4, T0 + 4 * S)):
        assert timeline.push(frame(seq, t, seq)) == []
    out += timeline.resume(T0 + 5 * S)
    out += timeline.push(frame(5, T0 + 6 * S, 5))
    out += timeline.stop(T0 + 7 * S)
    values = [int(item.bgr[0, 0, 0]) for item in out]
    assert len(values) == 120                                  # 2 s + 2 s，暂停的 3 s 不计
    assert values[60:90] == [4] * 30                           # 继续后先是暂停期间最后一帧
    assert values[90:] == [5] * 30
    assert timeline.runs[0].end_ns == T0 + 2 * S and timeline.runs[1].start_ns == T0 + 5 * S


def test_frames_after_the_resume_boundary_pushed_while_paused_are_replayed():
    timeline = Timeline(FPS)
    timeline.start(T0, frame(1, T0, 1))
    timeline.pause(T0 + S)
    timeline.push(frame(2, T0 + 2 * S - 1, 2))                 # 继续边界之前
    timeline.push(frame(3, T0 + 2 * S + 500_000_000, 3))       # 继续边界之后，但先于 resume 处理到
    out = timeline.resume(T0 + 2 * S)
    out += timeline.stop(T0 + 3 * S)
    values = [int(item.bgr[0, 0, 0]) for item in out]
    assert values[:15] == [2] * 15 and values[15:] == [3] * 15


def test_two_views_share_boundaries_and_end_with_equal_frame_counts():
    """正侧采集率不同（8 与 25 fps），共用边界，帧数相同；各自按自己的采集时刻取帧。"""
    counts = {}
    for view, rate in (("front", 8.0), ("side", 25.0)):
        timeline = Timeline(FPS)
        timeline.start(T0, frame(0, T0 - 1, role=view))
        sources = [frame(i, T0 + i * S / rate, role=view) for i in range(1, int(11.5 * rate))]
        for boundary, action in ((T0 + 4 * S, timeline.pause), (T0 + 6 * S, timeline.resume), (None, None)):
            while sources and (boundary is None or sources[0].captured_at_ns < boundary):
                timeline.push(sources.pop(0))
            if action is not None:
                action(boundary)
        timeline.stop(int(T0 + 11.5 * S))
        counts[view] = timeline.stats.slots
    assert counts["front"] == counts["side"] == math.ceil(9.5 * S / PERIOD)


def test_consecutive_segments_start_from_their_own_boundary():
    first, second = Timeline(FPS), Timeline(FPS)
    out1 = run(first, [frame(1, T0, 1), frame(2, T0 + S, 2)], stop=T0 + 2 * S)
    out2 = run(second, [frame(3, T0 + 2 * S, 3), frame(4, T0 + 3 * S, 4)], start=T0 + 2 * S, stop=T0 + 4 * S)
    assert len(out1) == len(out2) == 60
    assert {item.seq for item in out2} == {3, 4}                # 上一段的帧不进下一段


def test_start_requires_a_frame_from_before_the_boundary():
    with pytest.raises(ValueError):
        Timeline(FPS).start(T0, None)
    with pytest.raises(ValueError):
        Timeline(FPS).start(T0, frame(1, T0 + 1))
    with pytest.raises(ValueError):
        Timeline(0)


def test_out_of_order_and_post_stop_frames_are_ignored():
    timeline = Timeline(FPS)
    timeline.start(T0, frame(1, T0))
    timeline.push(frame(2, T0 + 100_000_000))
    assert timeline.push(frame(3, T0 + 50_000_000)) == []
    timeline.stop(T0 + 200_000_000)
    assert timeline.push(frame(4, T0 + 300_000_000)) == []
    assert timeline.stop(T0 + S) == []
    assert timeline.stats.slots == 6


def test_stop_while_paused_ends_at_the_pause_boundary():
    timeline = Timeline(FPS)
    timeline.start(T0, frame(1, T0))
    timeline.pause(T0 + S)
    timeline.push(frame(2, T0 + 2 * S))
    assert timeline.stop(T0 + 5 * S) == []
    assert timeline.stats.slots == 30


# --------------------------------------------------------------------------- #
# capture.json
# --------------------------------------------------------------------------- #


def _timelines():
    result = {}
    for view in ("front", "side"):
        timeline = Timeline(FPS)
        run(timeline, [frame(1, T0, role=view), frame(2, T0 + S, role=view)], stop=T0 + 2 * S)
        result[view] = timeline
    return result


def test_capture_record_round_trip(tmp_path):
    document = capture_document(_timelines(), {"front": {"queue_peak": 3}})
    path = tmp_path / CAPTURE_FILE
    write_capture_record(path, document)
    loaded = read_capture_record(path)
    assert loaded == json.loads(json.dumps(document))
    front = loaded["views"]["front"]
    assert loaded["schema"] == 1 and loaded["clock"] == "perf_counter_ns" and loaded["fps"] == FPS
    assert front["runs"] == [[T0, T0 + 2 * S]]
    assert front["stats"]["slots"] == 60 and front["queue_peak"] == 3
    assert front["slot_sources"][0] == [1, T0] and front["slot_sources"][30] == [2, T0 + S]


def test_capture_record_absent_or_foreign_reads_as_none(tmp_path):
    assert read_capture_record(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert read_capture_record(tmp_path / "bad.json") is None
    (tmp_path / "v2.json").write_text(json.dumps({"schema": 2, "views": {}}), encoding="utf-8")
    assert read_capture_record(tmp_path / "v2.json") is None


def test_capture_record_rejects_mixed_frame_rates():
    timelines = _timelines()
    other = Timeline(25.0)
    run(other, [frame(1, T0), frame(2, T0 + S)], stop=T0 + 2 * S)
    timelines["side"] = other
    with pytest.raises(ValueError):
        capture_document(timelines)


def test_capture_path_sits_inside_multi_segment_and_beside_single_segment(tmp_path):
    layout = SegmentLayout(tmp_path)
    stamp = "20261020_101500_000001"
    assert layout.capture_path(stamp, multi=True) == layout.segment_dir(stamp) / "capture.json"
    single = layout.capture_path(stamp, multi=False)
    assert single == layout.single_path(stamp).with_name(f"record_{stamp}.capture.json")
    assert SegmentLayout.segment_id(layout.single_path(stamp)) == f"record_{stamp}"
