# -*- coding: utf-8 -*-
"""P4 录制层的新契约：片段路径往返、多路同边界、finalizer 终态唯一与关窗取消、单摄关窗后转码、台账脱离面板后回填。"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from core.foundation.tasks import CancelToken
from core.media import SegmentLayout
from core.media.segment_layout import new_stamp, segment_id_for_stamp
from core.recording import (
    ProcessingPlan,
    RecordingService,
    ResultSpec,
    SegmentFinalizer,
    SegmentMeta,
    SegmentRecorder,
    SegmentUpdate,
    StepContext,
)
from core.workflows import ExamCandidate, ExamScorebook, ScorebookRegistry
from tests.segment_doubles import dual_segment, single_segment


class _Writer:
    def __init__(self, path: Path, on_write=None) -> None:
        self.path = path
        self.frames = 0
        self._on_write = on_write

    def write(self, _frame) -> None:
        if self._on_write is not None:
            self._on_write(self.path)
        self.frames += 1

    def release(self) -> None:
        pass


def _factory(*, failing: frozenset[str] = frozenset(), on_write=None):
    def factory(path, _fps, _size):
        path = Path(path)
        if path.stem in failing:
            raise RuntimeError(f"{path.stem} writer unavailable")
        return _Writer(path, on_write), path, "fake"

    return factory


def _frames(views) -> dict[str, np.ndarray]:
    return {view: np.zeros((3, 4, 3), dtype=np.uint8) for view in views}


# ---------------------------------------------------------------- SegmentLayout


def test_segment_layout_generated_paths_resolve_back_to_same_segment(tmp_path):
    stamp = new_stamp(datetime(2026, 10, 11, 23, 59, 59, 123456))
    layout = SegmentLayout(tmp_path)
    segment_id = segment_id_for_stamp(stamp)

    assert stamp == "20261011_235959_123456"
    assert layout.day_dir(stamp) == tmp_path / "20261011"
    assert layout.single_path(stamp) == tmp_path / "20261011" / f"{segment_id}.mp4"
    assert layout.view_path(stamp, "side") == layout.segment_dir(stamp) / "side.mp4"
    for path in (
        layout.single_path(stamp),
        layout.segment_dir(stamp),
        layout.view_path(stamp, "front"),
        layout.view_path(stamp, "side"),
    ):
        assert SegmentLayout.segment_id(path) == segment_id

    segment_dir = layout.segment_dir(stamp)
    segment_dir.mkdir(parents=True)
    for name in ("front.avi", "front.mp4", "side.avi", "notes.txt"):
        (segment_dir / name).write_bytes(b"x")
    assert SegmentLayout.resolve_videos(segment_dir) == (
        segment_dir / "front.mp4",
        segment_dir / "side.avi",
    )
    assert SegmentLayout.resolve_contained(tmp_path, "20261011", segment_id, "front.mp4") == (
        segment_dir / "front.mp4"
    ).resolve()


@pytest.mark.parametrize("name", ["", ".", "..", "../x.mp4", "a/b.mp4", "a\\b.mp4", None, 3])
def test_segment_layout_rejects_names_that_leave_the_directory(tmp_path, name):
    assert SegmentLayout.child(tmp_path, name) is None
    assert SegmentLayout.resolve_contained(tmp_path, name) is None


def test_segment_layout_directory_keys_are_stable_for_same_directory(tmp_path):
    directory = tmp_path / "20261011" / "record_x"
    directory.mkdir(parents=True)

    assert SegmentLayout.directory_keys(directory) == SegmentLayout.directory_keys(
        directory / ".." / "record_x"
    )


# ---------------------------------------------------------------- SegmentRecorder


def test_three_views_share_segment_boundaries_with_inflight_write(tmp_path):
    """片段结束要等进行中的一轮写完：每一路帧数相同，不会一路在片段内、一路落到片段外。"""
    views = ("front", "side", "top")
    first_write = threading.Event()
    release = threading.Event()

    def on_write(path: Path) -> None:
        if path.stem == "front" and not first_write.is_set():
            first_write.set()
            assert release.wait(2.0)

    recorder = SegmentRecorder(_factory(on_write=on_write))
    recorder.begin_session({view: (4, 3) for view in views}, 30.0)
    ref = recorder.start_segment(SegmentMeta(tmp_path))
    assert ref and ref.views == views

    writer = threading.Thread(target=lambda: recorder.write(_frames(views)))
    writer.start()
    assert first_write.wait(1.0)
    finished: list = []
    ender = threading.Thread(target=lambda: finished.append(recorder.end_segment()))
    ender.start()
    ender.join(0.2)
    assert ender.is_alive(), "片段结束必须等这一轮各路写完"

    release.set()
    writer.join(2.0)
    ender.join(2.0)
    segment = finished[0]
    assert [segment.views[v].frames for v in views] == [1, 1, 1]
    assert {segment.source(v).parent for v in views} == {segment.directory}
    assert recorder.state == "idle"


def test_one_view_write_failure_closes_all_views_together(tmp_path):
    recorder = SegmentRecorder(_factory(failing=frozenset({"side"})))
    recorder.begin_session({"front": (4, 3), "side": (4, 3), "top": (4, 3)}, 30.0)
    assert recorder.start_segment(SegmentMeta(tmp_path))
    recorder.write(_frames(("front", "side", "top")))

    assert recorder.broken()
    segment = recorder.end_segment()

    assert segment.errors == {"side": "side writer unavailable"}
    assert [segment.views[v].frames for v in ("front", "side", "top")] == [1, 0, 1]
    assert recorder.snapshot().errors == {}
    assert recorder.state == "idle"
    assert not recorder.broken()
    # 下一段照常各路一起开始
    assert recorder.start_segment(SegmentMeta(tmp_path))


def test_start_realigns_views_when_one_is_still_recording(tmp_path):
    recorder = SegmentRecorder(_factory())
    recorder.begin_session({"front": (4, 3), "side": (4, 3)}, 30.0)
    # 模拟主路写失败复位 idle、侧路仍 recording 的分叉状态
    recorder._controllers["side"].request_toggle()

    assert recorder.start_segment(SegmentMeta(tmp_path))
    assert {view: snap.state for view, snap in recorder.snapshot().views.items()} == {
        "front": "recording",
        "side": "recording",
    }


# ---------------------------------------------------------------- SegmentFinalizer


class _BlockingStep:
    lane = "io"

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def run(self, ctx: StepContext, cancel: CancelToken):
        self.entered.set()
        assert self.release.wait(2.0)
        if cancel.cancelled or ctx.stop_evt.is_set():
            raise InterruptedError("cancelled")
        return None


def _dual(tmp_path: Path, name: str):
    directory = tmp_path / name
    directory.mkdir(parents=True)
    return dual_segment(name, directory, directory / "front.mp4", directory / "side.mp4")


def test_finalizer_publishes_single_terminal_after_cancel_and_cancels_queued_on_close(tmp_path):
    updates: list[SegmentUpdate] = []
    finalizer = SegmentFinalizer(on_update=updates.append)
    step = _BlockingStep()
    running = _dual(tmp_path, "record_running")
    queued = _dual(tmp_path, "record_queued")
    plan = ProcessingPlan((step,), result=ResultSpec())

    assert finalizer.submit(running, plan)
    assert finalizer.submit(queued, ProcessingPlan((), result=ResultSpec()))
    assert step.entered.wait(1.0)

    finalizer.cancel_all()
    finalizer.cancel_all()  # 重复取消不产生第二个终态
    assert not finalizer.submit(_dual(tmp_path, "record_late"), plan)
    step.release.set()
    assert finalizer.wait_idle(2.0)

    terminals = [(u.segment_id, u.status, u.error_code) for u in updates if u.status != "queued"]
    assert terminals == [
        ("record_running", "cancelled", "app_closing"),
        ("record_queued", "cancelled", "app_closing"),
    ]
    for segment in (running, queued):
        payload = json.loads((segment.directory / "result.json").read_text(encoding="utf-8"))
        assert payload["status"] == "cancelled"
        assert payload["error"]["code"] == "app_closing"


def test_finalizer_rejects_duplicate_segment_and_directory(tmp_path):
    finalizer = SegmentFinalizer()
    segment = _dual(tmp_path, "record_a")
    same_dir = dual_segment("record_b", segment.directory, Path("f.mp4"), Path("s.mp4"))
    plan = ProcessingPlan(())

    assert finalizer.submit(segment, plan)
    assert not finalizer.submit(segment, plan)
    assert not finalizer.submit(same_dir, plan)
    assert finalizer.wait_idle(1.0)


def test_single_camera_transcode_is_accepted_and_finished_after_close(tmp_path):
    """V-4：单摄转码计划 finish_on_close，关窗取消后仍接受、执行，且执行线不是守护线程。"""
    ran: list[tuple[str, bool]] = []

    class _Transcode:
        lane = "io"

        def run(self, ctx, cancel):
            current = threading.current_thread()
            ran.append((ctx.segment_id, current.daemon))

    finalizer = SegmentFinalizer()
    finalizer.cancel_all()

    assert not finalizer.submit(_dual(tmp_path, "record_dual"), ProcessingPlan((_Transcode(),)))
    assert finalizer.submit(
        single_segment("record_single", tmp_path / "record_single.mp4"),
        ProcessingPlan((_Transcode(),), finish_on_close=True),
    )
    assert finalizer.close(2.0)
    assert ran == [("record_single", False)]


def test_service_finish_and_close_cancel_never_interleave(tmp_path):
    """结束+提交与 cancel_all 串行：提交前取消则拒绝提交，不会留下既没提交也没取消终态的片段。"""
    updates: list[SegmentUpdate] = []
    finalizer = SegmentFinalizer(on_update=updates.append)
    service = RecordingService(
        finalizer,
        plan_for=lambda _segment: ProcessingPlan((), result=ResultSpec()),
        recorder=SegmentRecorder(_factory()),
    )
    service.begin_session({"front": (4, 3), "side": (4, 3)}, 30.0)
    assert service.start(SegmentMeta(tmp_path))
    service.cancel_all()

    assert service.finish() is not None     # 结束不等待；写完时提交被拒绝
    assert service.wait_idle(1.0)
    assert finalizer.wait_idle(1.0)
    assert updates == []


# ---------------------------------------------------------------- ScorebookRegistry


def test_scorebook_backfills_after_panel_detached(tmp_path):
    book = ExamScorebook(path=tmp_path / "scores.xlsx")
    book.load_candidates([ExamCandidate(1, "S1", "甲"), ExamCandidate(2, "S2", "乙")])
    first, second = book.rows[:2]
    book.bind_segment(first, segment_id="record_1", segment_dir=str(tmp_path / "record_1"))
    book.bind_segment(second, segment_id="record_2", segment_dir=str(tmp_path / "record_2"))
    book.update_row(first, status="processing")
    book.update_row(second, status="processing")

    registry = ScorebookRegistry()
    registry.register("run-1", book)
    registry.detach(book)  # 面板关闭：台账交给登记表
    assert registry.has_detached and registry.has_processing

    applied = registry.apply(
        SegmentUpdate(
            "record_1", "completed", "done",
            front_score=0.8, side_score=0.9, combined_percent=85, exam_run_id="run-1",
        )
    )
    assert applied == (book, first)
    assert first.status == "completed" and first.combined_percent == 85
    # 没带 run_id 的更新按片段 id 在全部台账里找
    assert registry.apply(SegmentUpdate("record_2", "failed", "bad", error_code="x")) == (book, second)
    assert second.status == "failed"
    assert not registry.has_processing
    assert registry.apply(SegmentUpdate("record_unknown", "completed", "done")) is None

    registry.close_all()
    assert (tmp_path / "scores.xlsx").is_file()
    assert not registry.has_detached
