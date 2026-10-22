# -*- coding: utf-8 -*-
"""D-18 第 8 层：摄像头会话的开录确认、异步结束与显式暂停（D-27、D-28，设计文档 §8.8 控制）。

用第 6 层测试的假帧来源（按真实时间以 30 fps 出帧）与假 writer 驱动真实的 ``SegmentRecorder``/``RecordingService``。
"""
from __future__ import annotations

import threading
import time

from core.media import read_capture_record
from core.recording import (
    START_CONFIRM_S,
    BeginError,
    ProcessingPlan,
    RecordingService,
    SegmentFinalizer,
    SegmentMeta,
    SegmentRecorder,
)
from tests.test_recording_worker import FakeFeed, Writers, _recorder, feed  # noqa: F401 - feed 是 fixture


class Confirms:
    """记下开录确认：(片段 id, 原因, 回送时刻)。"""

    def __init__(self) -> None:
        self.items: list[tuple[str, str | None, float]] = []
        self.event = threading.Event()

    def __call__(self, segment, error) -> None:
        self.items.append((segment.segment_id, error, time.perf_counter()))
        self.event.set()


def test_start_is_confirmed_only_after_both_views_wrote_their_first_frame(tmp_path, feed):
    writers = Writers()
    recorder = _recorder(feed(), writers)
    confirms = Confirms()
    started = time.perf_counter()
    ref = recorder.start_segment(SegmentMeta(tmp_path), confirms)
    assert ref and not confirms.items                    # 立即返回，确认还没到
    assert confirms.event.wait(START_CONFIRM_S + 1.0)
    [(segment_id, error, at)] = confirms.items
    assert segment_id == ref.segment_id and error is None
    assert at - started > 0.03                           # 至少等到下一帧确认首个输出槽
    assert all(writers.last(view).frames for view in ("front", "side"))
    recorder.end_segment()
    recorder.close_session()


def test_start_fails_and_ends_the_segment_when_a_view_has_no_frames(tmp_path, feed):
    source = feed()
    writers = Writers()
    recorder = _recorder(source, writers)
    source.break_role("secondary")                       # 侧面在开录前断流
    time.sleep(0.2)
    finished = []
    confirms = Confirms()
    recorder.start_segment(SegmentMeta(tmp_path), confirms)
    assert confirms.event.wait(START_CONFIRM_S + 1.0)
    [(_sid, error, _at)] = confirms.items
    assert error is not None and error.startswith("侧面")
    assert recorder.wait_idle(5.0)
    assert recorder.state == "idle"                      # 整段已结束（落盘、不提交），可以重试
    assert recorder.request_end(on_finished=finished.append) is None
    recorder.close_session()


def test_start_times_out_when_no_frame_is_written(tmp_path, feed, monkeypatch):
    import core.recording.recorder as recorder_mod

    monkeypatch.setattr(recorder_mod, "START_CONFIRM_S", 0.3)
    source = feed()
    recorder = _recorder(source, Writers())
    source.close()                                       # 两路都停止出帧，但没有报错
    time.sleep(0.1)
    confirms = Confirms()
    recorder.start_segment(SegmentMeta(tmp_path), confirms)
    assert confirms.event.wait(3.0)
    error = confirms.items[0][1]
    assert error is not None
    assert recorder.wait_idle(5.0) and recorder.state == "idle"
    recorder.close_session()


def test_end_does_not_wait_and_blocks_the_next_start_until_drained(tmp_path, feed):
    """结束只截断边界：立即返回、进入 stopping，stopping 期间不能开下一段；写完后交出一次片段并回到 idle。"""
    writers = Writers(delay=0.004)
    recorder = _recorder(feed(), writers)
    confirms = Confirms()
    ref = recorder.start_segment(SegmentMeta(tmp_path), confirms)
    assert confirms.event.wait(START_CONFIRM_S + 1.0)
    time.sleep(0.5)
    finished = []
    t0 = time.perf_counter()
    ending = recorder.request_end(on_finished=finished.append)
    returned_in = time.perf_counter() - t0
    assert ending is ref and returned_in < 0.05
    assert recorder.state == "stopping"
    blocked = recorder.start_segment(SegmentMeta(tmp_path))
    assert isinstance(blocked, BeginError) and blocked.reason == "stopping"
    assert "结束" in blocked.message
    assert recorder.wait_idle(5.0)
    assert recorder.state == "idle" and len(finished) == 1
    segment = finished[0]
    assert segment.ref is ref and not segment.errors
    assert segment.views["front"].frames == segment.views["side"].frames > 0
    assert read_capture_record(ref.capture_path) is not None
    assert recorder.start_segment(SegmentMeta(tmp_path))    # 写完之后可以开下一段
    recorder.close_session()


def test_late_confirmation_of_an_ended_segment_is_dropped(tmp_path, feed):
    recorder = _recorder(feed(), Writers())
    confirms = Confirms()
    recorder.start_segment(SegmentMeta(tmp_path), confirms)
    recorder.request_end(discard=True)                   # 确认之前就结束（考试跳过、练习取消）
    assert recorder.wait_idle(5.0)
    time.sleep(0.3)
    assert confirms.items == []
    recorder.close_session()


def test_explicit_pause_and_resume_exclude_the_pause_from_the_output(tmp_path, feed):
    writers = Writers()
    recorder = _recorder(feed(), writers)
    ref = recorder.start_segment(SegmentMeta(tmp_path))
    assert recorder.pause() == "paused"                  # 录制中→暂停
    time.sleep(0.6)
    assert recorder.pause() == "paused"                  # 已暂停时不变
    time.sleep(0.6)
    assert recorder.resume() == "recording"
    assert recorder.resume() == "recording"              # 已在录制时不变
    time.sleep(0.6)
    finished = recorder.end_segment()
    recorder.close_session()
    assert finished.ref is ref
    record = read_capture_record(ref.capture_path)["views"]["front"]
    assert abs(record["stats"]["slots"] - 18) <= 3       # 两段 0.6 s；中间 1.2 s 的暂停不进录像


def test_service_submits_once_after_the_drain_and_close_waits(tmp_path, feed):
    submitted = []
    finalizer = SegmentFinalizer()
    service = RecordingService(
        finalizer, plan_for=lambda seg: submitted.append(seg) or ProcessingPlan(()),
        recorder=SegmentRecorder(Writers()),
    )
    service.begin_session({"front": (6, 4), "side": (6, 4)}, 30.0, feed=feed(), rotation_of=lambda _r: 0)
    assert service.start(SegmentMeta(tmp_path))
    time.sleep(0.4)
    ref = service.finish()
    assert ref is not None and submitted == []           # 结束立即返回，写完才提交
    assert service.wait_idle(5.0)
    assert [seg.ref for seg in submitted] == [ref]
    assert service.start(SegmentMeta(tmp_path))
    time.sleep(0.3)
    outcome = service.close_session()                    # 会话结束：等写完并提交
    assert outcome.segment is not None and outcome.submitted
    assert len(submitted) == 2
    assert finalizer.close(2.0)
