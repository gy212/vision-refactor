# -*- coding: utf-8 -*-
"""D-26：录像只存原始画面，带骨架的成片在录后生成——``RenderSkeleton`` 步骤、处理计划、单摄失败上报、
``annotate_video`` 与 CLI 摄像头 ``--out``。"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import apps.cli.main as cli_main
import core.pose.annotate as annotate_mod
from core.foundation.tasks import Cancelled, CancelToken
from core.media import PRIMARY, open_video_writer
from core.pose import EngineSpec, annotate_video
from core.recording import (
    PostprocessError,
    ProcessingPlan,
    RenderSkeleton,
    SegmentFinalizer,
    SegmentUpdate,
    StepContext,
    Transcode,
    skeleton_engine,
)
from core.workflows import PlanSettings, build_segment_plan
from tests.segment_doubles import single_segment
from tests.test_recording_worker import FakeFeed

ENGINE = skeleton_engine("lite", False)


def _ctx(tmp_path: Path, views=("front",)) -> StepContext:
    sources = {}
    for view in views:
        sources[view] = tmp_path / f"{view}.mp4"
        sources[view].write_bytes(b"raw")
    return StepContext("record_x", tmp_path, tuple(views), sources, {v: 3 for v in views}, {},
                       threading.Event(), lambda *_a: None)


def _render_to(content: bytes, *, fail: Exception | None = None):
    calls = []

    def render(source: Path, target: Path, engine: EngineSpec, cancel: CancelToken) -> Path:
        calls.append((source, engine))
        actual = target.with_suffix(".avi")       # 模拟 H.264 不可用时 writer 回退 AVI
        actual.write_bytes(content)
        if fail is not None:
            raise fail
        return actual

    render.calls = calls
    return render


def _transcode(path: Path, _stop) -> Path:
    out = path.with_suffix(".mp4")
    out.write_bytes(path.read_bytes())
    path.unlink()
    return out


def test_render_replaces_raw_recording_with_skeleton_video_under_the_same_name(tmp_path):
    ctx = _ctx(tmp_path, ("front", "side"))
    render = _render_to(b"skeleton")
    step = RenderSkeleton(ENGINE, render=render, transcode=_transcode, validator=lambda p: p.is_file())

    assert step.run(ctx, CancelToken()) is None
    for view in ("front", "side"):
        assert ctx.video(view) == tmp_path / f"{view}.mp4"
        assert (tmp_path / f"{view}.mp4").read_bytes() == b"skeleton"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["front.mp4", "side.mp4"]
    assert [engine for _src, engine in render.calls] == [ENGINE, ENGINE]


@pytest.mark.parametrize("case", ["render", "invalid"])
def test_render_failure_keeps_raw_recording_and_removes_temporaries(tmp_path, case):
    ctx = _ctx(tmp_path)
    render = _render_to(b"half", fail=RuntimeError("推理失败") if case == "render" else None)
    step = RenderSkeleton(ENGINE, render=render, transcode=_transcode, validator=lambda _p: False)

    with pytest.raises(PostprocessError) as exc:
        step.run(ctx, CancelToken())
    assert exc.value.code == "skeleton_render_failed" and "已保留原始录像" in str(exc.value)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["front.mp4"]
    assert (tmp_path / "front.mp4").read_bytes() == b"raw"


def test_render_cancel_keeps_raw_recording(tmp_path):
    ctx = _ctx(tmp_path)
    step = RenderSkeleton(ENGINE, render=_render_to(b"half", fail=Cancelled("app_close")), transcode=_transcode)
    with pytest.raises(Cancelled):
        step.run(ctx, CancelToken())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["front.mp4"]


def _segment(multi: bool):
    if multi:
        return SimpleNamespace(multi=True, ref=SimpleNamespace(meta=SimpleNamespace(context=None)))
    return SimpleNamespace(multi=False)


def test_plans_render_skeleton_only_when_recording_skeleton():
    plain = build_segment_plan(_segment(False), PlanSettings(record_skeleton=False, auto_compare=False))
    assert [type(s) for s in plain.steps] == [Transcode] and plain.finish_on_close

    single = build_segment_plan(_segment(False), PlanSettings(record_skeleton=True, auto_compare=False,
                                                              pose_variant="heavy", enable_hands=False))
    assert [type(s) for s in single.steps] == [Transcode, RenderSkeleton]
    assert single.steps[1].engine == skeleton_engine("heavy", False)
    assert not single.finish_on_close      # 成片可能很久：关窗时取消，保留原始录像

    dual = build_segment_plan(_segment(True), PlanSettings(record_skeleton=True, auto_compare=True))
    kinds = [type(s).__name__ for s in dual.steps]
    assert kinds.index("RenderSkeleton") == kinds.index("ValidateVideos") + 1
    assert kinds.index("CompareGate") == kinds.index("RenderSkeleton") + 1
    assert "RenderSkeleton" not in [type(s).__name__ for s in build_segment_plan(
        _segment(True), PlanSettings(record_skeleton=False, auto_compare=True)).steps]


def test_single_segment_failure_is_reported_without_result_json(tmp_path):
    updates: list[SegmentUpdate] = []
    finalizer = SegmentFinalizer(on_update=updates.append)

    class Failing:
        lane = "analysis"

        def run(self, ctx, cancel):
            raise PostprocessError("skeleton_render_failed", "正面骨架成片失败（已保留原始录像）：x")

    class Passing:
        lane = "io"

        def run(self, ctx, cancel):
            return None

    assert finalizer.submit(single_segment("record_ok", tmp_path / "a.mp4"), ProcessingPlan((Passing(),)))
    assert finalizer.submit(single_segment("record_bad", tmp_path / "b.mp4"), ProcessingPlan((Failing(),)))
    assert finalizer.wait_idle(2.0)
    assert [(u.segment_id, u.status, u.error_code) for u in updates] == [
        ("record_bad", "failed", "skeleton_render_failed")]
    assert not list(tmp_path.glob("*.json"))


class FakeEngine:
    def __init__(self) -> None:
        self.stamps: list[int] = []
        self.closed = False

    def annotate(self, frame, *, timestamp_ms=None):
        self.stamps.append(timestamp_ms)
        out = frame.copy()
        out[:, :8] = 255
        return out, []

    def close(self) -> None:
        self.closed = True


def _video(path: Path, frames: int = 6) -> Path:
    writer, actual, _codec = open_video_writer(path, fps=30.0, size=(32, 24))
    for value in range(frames):
        writer.write(np.full((24, 32, 3), value * 10, dtype=np.uint8))
    writer.release()
    return actual


def test_annotate_video_draws_every_frame_with_increasing_timestamps(tmp_path, monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(annotate_mod, "create_engine", lambda _spec: engine)
    source = _video(tmp_path / "raw.avi")
    path, _codec, frames = annotate_video(source, tmp_path / "out.avi", ENGINE)

    assert frames == 6 and engine.closed and engine.stamps == sorted(set(engine.stamps))
    capture = cv2.VideoCapture(str(path))
    count = 0
    while (got := capture.read())[0]:
        assert int(got[1][:, :8].min()) > 200
        count += 1
    capture.release()
    assert count == 6


def test_cli_camera_out_records_raw_then_renders_skeleton_video(tmp_path, monkeypatch):
    """CLI 摄像头 ``--out``：会话期间按真实时间录原始画面到临时文件，会话后画骨架写成 ``--out`` 并删掉临时文件。"""
    rendered: list[tuple[Path, Path]] = []

    def fake_annotate(raw, out, engine, *, open_writer, progress):
        rendered.append((Path(raw), Path(out)))
        Path(out).write_bytes(b"skeleton")
        return Path(out), "mp4v", 1

    monkeypatch.setattr(cli_main, "annotate_video", fake_annotate)
    out = tmp_path / "out.mp4"
    export = cli_main.CliExport(str(out), ENGINE)
    feed = FakeFeed(roles=(PRIMARY,), shape=(24, 32, 3))
    try:
        export.begin_session({"front": (32, 24)}, 30.0, feed=feed)
        time.sleep(0.4)
        export.close_session()
    finally:
        feed.close()
    assert export.error is None and not rendered      # 成片在会话之后才生成
    export.finish(progress=False)
    raw = rendered[0][0]
    assert rendered[0][1] == out and raw.name == ".out.raw.avi" and not raw.exists()
    assert out.read_bytes() == b"skeleton" and export.error is None
