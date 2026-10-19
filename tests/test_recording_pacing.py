# -*- coding: utf-8 -*-
"""D-18：录像按真实时间写出。``FramePacer`` 的时间轴，以及 ``SegmentRecorder`` 在慢速流水线下的录像时长与正侧同步。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.media import FramePacer
from core.recording import SegmentMeta, SegmentRecorder

FPS = 30.0


def _feed(pacer: FramePacer, rate: float, seconds: float, start: float = 100.0, jitter: float = 0.0) -> list[int]:
    rng = np.random.default_rng(0)
    n = int(round(rate * seconds))
    stamps = [start + i / rate + (rng.uniform(-jitter, jitter) if jitter else 0.0) for i in range(n)]
    return [pacer.repeats(t) for t in stamps]


def test_pacer_at_nominal_rate_writes_each_frame_once_even_with_jitter():
    pacer = FramePacer(FPS)
    counts = _feed(pacer, FPS, 4.0, jitter=0.01)
    assert counts == [1] * len(counts)


@pytest.mark.parametrize("rate", [8.0, 12.0, 15.0, 20.0])
def test_pacer_slow_pipeline_fills_to_real_duration(rate):
    """处理只有 8～20 fps 时，写出的帧数按 30 fps 折算仍等于真实时长（误差 < 3%）。"""
    pacer = FramePacer(FPS)
    seconds = 10.0
    counts = _feed(pacer, rate, seconds)
    duration = sum(counts) / FPS
    real = (int(round(rate * seconds)) - 1) / rate        # 首帧到末帧的真实间隔
    assert abs(duration - real) / real < 0.03
    assert min(counts) >= 1                               # 慢于名义帧率时不丢帧


def test_pacer_fast_source_skips_frames_already_covered():
    pacer = FramePacer(FPS)
    counts = _feed(pacer, 60.0, 2.0)
    assert sum(counts) == pytest.approx(60, abs=1)
    assert counts.count(0) == pytest.approx(60, abs=1)


def test_pacer_without_timestamps_writes_once():
    pacer = FramePacer(FPS)
    assert [pacer.repeats(None) for _ in range(5)] == [1] * 5
    assert pacer.written == 5


def test_pacer_resync_excludes_paused_time_and_reset_starts_new_timeline():
    pacer = FramePacer(FPS)
    first = _feed(pacer, 15.0, 2.0, start=0.0)
    pacer.resync()                                         # 暂停 60 秒后继续
    second = _feed(pacer, 15.0, 2.0, start=62.0)
    assert sum(first) + sum(second) == pytest.approx(4 * FPS, abs=4)
    pacer.reset()
    assert pacer.written == 0 and pacer.repeats(500.0) == 1


def test_pacer_rejects_invalid_fps():
    with pytest.raises(ValueError):
        FramePacer(0.0)


class _Writer:
    def __init__(self) -> None:
        self.frames = 0

    def write(self, frame) -> None:
        self.frames += 1

    def release(self) -> None:
        pass


def _recorder(writers: dict[str, _Writer]) -> SegmentRecorder:
    def factory(path: Path, fps: float, size):
        writer = _Writer()
        writers[Path(path).stem] = writer
        return writer, Path(path), "fake"

    return SegmentRecorder(writer_factory=factory)


def test_recorder_slow_pipeline_records_real_duration_and_keeps_views_in_step(tmp_path):
    """假的慢速流水线：12 fps 写 6 秒。修复前每路只有 72 帧（30 fps 回放 2.4 秒），修复后约 180 帧且正侧相同。"""
    writers: dict[str, _Writer] = {}
    rec = _recorder(writers)
    rec.begin_session({"front": (6, 4), "side": (4, 6)}, FPS)
    assert rec.start_segment(SegmentMeta(tmp_path))
    rate, seconds = 12.0, 6.0
    for i in range(int(rate * seconds)):
        rec.write({"front": np.zeros((4, 6, 3), np.uint8), "side": np.zeros((6, 4, 3), np.uint8)},
                  captured_at=1000.0 + i / rate)
    segment = rec.end_segment()
    frames = {view: segment.views[view].frames for view in ("front", "side")}
    assert frames["front"] == frames["side"] == writers["front"].frames == writers["side"].frames
    real = (rate * seconds - 1) / rate
    assert abs(frames["front"] / FPS - real) / real < 0.03


def test_recorder_pause_is_not_recorded_and_new_segment_restarts_timeline(tmp_path):
    rec = _recorder({})
    rec.begin_session({"front": (6, 4)}, FPS)
    assert rec.start_segment(SegmentMeta(tmp_path))
    img = {"front": np.zeros((4, 6, 3), np.uint8)}
    for i in range(30):                                   # 2 秒，15 fps
        rec.write(img, captured_at=i / 15.0)
    assert rec.toggle_pause() == "paused"
    for i in range(30):                                   # 暂停期间的帧不写、不推进时间轴
        rec.write(img, captured_at=10.0 + i / 15.0)
    assert rec.toggle_pause() == "recording"
    for i in range(30):
        rec.write(img, captured_at=50.0 + i / 15.0)
    first = rec.end_segment()
    assert first.views["front"].frames == pytest.approx(4 * FPS, abs=4)

    assert rec.start_segment(SegmentMeta(tmp_path))
    rec.write(img, captured_at=900.0)
    rec.write(img, captured_at=900.0 + 1 / 15.0)
    assert rec.end_segment().views["front"].frames == 3


def test_recorder_without_timestamps_writes_each_frame_once(tmp_path):
    rec = _recorder({})
    rec.begin_session({"front": (6, 4)}, 25.0)
    assert rec.start_segment(SegmentMeta(tmp_path))
    for _ in range(7):
        rec.write({"front": np.zeros((4, 6, 3), np.uint8)})
    assert rec.end_segment().views["front"].frames == 7
