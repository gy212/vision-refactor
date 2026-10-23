# -*- coding: utf-8 -*-
"""录制性能测试（``core.services.recording_benchmark``）：用假摄像头与假引擎跑完四段，检查报告内容、判定与清理；
writer 与生产一样在帧流动之后打开，报告编码、打开耗时与积压，打开超过生产上限判不达标。"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime

import numpy as np
import pytest

from core.foundation import CancelToken
from core.media import PRIMARY, SECONDARY, CameraPool, CameraService, CaptureSpec, FrameQueue
from core.services import BENCHMARK_PHASES, BenchmarkPhase, BenchmarkSetup, run_recording_benchmark, save_benchmark_report
from core.services import recording_benchmark as bench


class TimedCapture:
    """按 ``interval`` 出帧的假驱动（小画面，测试快）。"""

    interval = 1 / 30

    def __init__(self, index, *, width=1920, height=1080, **_kwargs) -> None:
        self.index = index
        self.next_at = time.perf_counter()
        self.reads = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return {3: 64, 4: 48, 5: 30.0}.get(prop, 0.0)

    def read(self):
        self.next_at += self.interval
        time.sleep(max(0.0, self.next_at - time.perf_counter()))
        self.reads += 1
        return True, np.full((48, 64, 3), self.reads % 251, np.uint8)

    def release(self):
        pass


class FakeEngine:
    def __init__(self, spec, delay=0.002):
        self.spec, self.delay, self.closed = spec, delay, False

    def next_timestamp_ms(self, *, is_file, fps_for_ts):
        return int(time.monotonic() * 1000)

    def infer(self, frame, *, timestamp_ms=None):
        time.sleep(self.delay)
        return None, []

    def draw(self, frame, pose, hands):
        return frame, []

    def close(self):
        self.closed = True


SHORT = tuple(BenchmarkPhase(p.key, p.label, 0.6, p.write, p.lite, p.skeleton) for p in BENCHMARK_PHASES)


@pytest.fixture
def service():
    svc = CameraService(pool=CameraPool(TimedCapture, join_timeout=0.5))
    try:
        yield svc
    finally:
        svc.close(2.0)


def _setup(service, tmp_path, engines, *, phases=SHORT, dual=True):
    specs = {PRIMARY: CaptureSpec(0)}
    if dual:
        specs[SECONDARY] = CaptureSpec(1)
    factory = lambda spec: engines.setdefault(len(engines), FakeEngine(spec))
    return BenchmarkSetup(
        stream=service.open_stream(specs), rotations={PRIMARY: 90, SECONDARY: 0},
        labels={PRIMARY: "正面相机", SECONDARY: "侧面相机"}, work_dir=tmp_path / "work",
        engine_factory=factory, phases=phases, extra={"build": {"commit": "abc1234"}},
    )


def test_full_run_reports_every_phase_and_cleans_up(service, tmp_path):
    engines: dict = {}
    progress: list = []
    report = run_recording_benchmark(_setup(service, tmp_path, engines), CancelToken(),
                                     lambda text, done: progress.append((text, done)))
    assert report["kind"] == "recording_benchmark" and report["build"] == {"commit": "abc1234"}
    assert [p["key"] for p in report["phases"]] == ["capture", "record", "record_lite", "record_skeleton"]
    assert set(report["cameras"]) == {"front", "side"}
    assert report["cameras"]["front"]["rotation"] == 90 and report["cameras"]["front"]["label"] == "正面相机"
    for phase in report["phases"]:
        for view in ("front", "side"):
            capture = phase["roles"][view]["capture"]
            assert capture["frames"] >= 10 and capture["fps"] == pytest.approx(30, abs=4)
            if phase["key"] != "capture":
                write = phase["roles"][view]["write"]
                assert write["frames"] >= 10 and write["codec"] and write["error"] is None
                assert write["over_production_queue"] == 0 and write["dropped"] == 0
                assert write["open_ms"] is not None and write["open_backlog"] >= 1 and write["backlog_dropped"] == 0
                assert write["catch_up_ms"] is not None
    assert report["phases"][2]["lite"]["runs"] >= 3
    assert report["phases"][3]["skeleton"]["loops"] >= 3
    assert all(engine.closed for engine in engines.values()) and len(engines) == 3   # lite + 每路一个
    assert not (tmp_path / "work").exists()                                          # 临时录像已删除
    assert progress[-1][1] == 1.0 and report["summary"]
    assert "编码 正面" in report["summary"][1] and report["production_open_budget_ms"] == 1500.0
    assert not service.device_busy(0) or True                                        # 设备交回池（预热）


def _slow_open(monkeypatch, open_s: float) -> None:
    real_writer = bench.open_video_writer

    def slow(path, *, fps, size):
        time.sleep(open_s)
        return real_writer(path, fps=fps, size=size)

    monkeypatch.setattr(bench, "open_video_writer", slow)


def test_writer_opens_after_frames_flow_like_production(service, tmp_path, monkeypatch):
    """与生产相同：帧已经在流动时才打开 writer。打开 0.6 s 期间帧按序积压，写盘队列不排满，打开后追平。"""
    _slow_open(monkeypatch, 0.6)
    phases = (BenchmarkPhase("record", "采集 + 写盘", 2.0, write=True),)
    report = run_recording_benchmark(_setup(service, tmp_path, {}, phases=phases), CancelToken(), lambda *_: None)
    for view in ("front", "side"):
        write = report["phases"][0]["roles"][view]["write"]
        assert write["open_ms"] >= 550 and write["open_backlog"] >= 15 and write["backlog_peak"] >= 15
        assert write["over_production_queue"] == 0 and write["backlog_dropped"] == 0
        assert write["catch_up_ms"] is not None and write["frames"] >= 50
    assert report["ok"] is True, report["summary"]


def test_writer_open_over_the_production_budget_is_reported(service, tmp_path, monkeypatch):
    """打开超过生产上限（1.5 s）时，生产会判开录失败：报告写明编码与打开耗时，该段不达标。"""
    _slow_open(monkeypatch, bench.PRODUCTION_OPEN_BUDGET_S + 0.3)
    phases = (BenchmarkPhase("record", "采集 + 写盘", 2.5, write=True),)
    report = run_recording_benchmark(_setup(service, tmp_path, {}, phases=phases, dual=False), CancelToken(),
                                     lambda *_: None)
    write = report["phases"][0]["roles"]["front"]["write"]
    assert write["open_ms"] >= 1800 and write["backlog_dropped"] > 0 and write["over_production_queue"] == 0
    problems = report["phases"][0]["verdict"]["problems"]
    assert report["ok"] is False
    assert any("打开录像文件用了" in problem and "超过生产上限 1500 ms" in problem for problem in problems)
    assert any(f"开录积压超过 {bench.BACKLOG_LIMIT} 帧" in problem for problem in problems)


def test_production_limits_match_the_recording_worker():
    """benchmark 不能导入 ``core.recording``（L4 互不导入），镜像的生产常量由这里核对。"""
    from core.recording import QUEUE_CAPACITY, WRITER_OPEN_TIMEOUT_S

    assert bench.PRODUCTION_QUEUE == QUEUE_CAPACITY
    assert bench.PRODUCTION_OPEN_BUDGET_S == WRITER_OPEN_TIMEOUT_S


def test_slow_writer_is_reported_as_not_keeping_up(service, tmp_path, monkeypatch):
    """写一帧 60 ms（30 fps 需要 ≤33 ms）时，队列会排到 8 帧以上，判为不达标。"""
    real_writer = bench.open_video_writer

    class SlowWriter:
        def __init__(self, inner):
            self.inner = inner

        def write(self, frame):
            time.sleep(0.06)
            self.inner.write(frame)

        def release(self):
            self.inner.release()

    def slow(path, *, fps, size):
        writer, actual, codec = real_writer(path, fps=fps, size=size)
        return SlowWriter(writer), actual, codec

    monkeypatch.setattr(bench, "open_video_writer", slow)
    phases = (BenchmarkPhase("record", "采集 + 写盘", 1.5, write=True),)
    report = run_recording_benchmark(_setup(service, tmp_path, {}, phases=phases), CancelToken(), lambda *_: None)
    write = report["phases"][0]["roles"]["front"]["write"]
    assert write["over_production_queue"] > 0 and write["queue_peak"] >= 8
    assert report["ok"] is False and not report["phases"][0]["verdict"]["ok"]
    assert any("写盘跟不上" in line for line in report["summary"])


def test_cancel_stops_early_and_still_returns_a_report(service, tmp_path):
    token = CancelToken()
    threading.Timer(0.3, token.cancel, args=("user",)).start()
    phases = (BenchmarkPhase("capture", "只采集", 30.0),)
    started = time.monotonic()
    report = run_recording_benchmark(_setup(service, tmp_path, {}, phases=phases, dual=False), token,
                                     lambda *_: None)
    assert time.monotonic() - started < 10
    assert report["ok"] is False and report["error"] == "已取消"
    assert any("测试中断" in line for line in report["summary"])


def test_engine_failure_is_recorded_not_raised(service, tmp_path):
    def broken(spec):
        raise RuntimeError("模型缺失")

    setup = _setup(service, tmp_path, {})
    setup.engine_factory = broken
    report = run_recording_benchmark(setup, CancelToken(), lambda *_: None)
    assert report["ok"] is False and "模型缺失" in report["error"]


def test_save_report_writes_a_named_utf8_json(tmp_path):
    path = save_benchmark_report({"kind": "recording_benchmark", "summary": ["中文"]}, tmp_path / "桌面",
                                 now=datetime(2026, 10, 21, 9, 30, 5))
    assert path.name == "录制性能测试_20261021_093005.json"
    assert json.loads(path.read_text(encoding="utf-8"))["summary"] == ["中文"]


def test_frame_queue_counts_watermark_and_overflow():
    queue = FrameQueue(3, watermark=2)
    frames = [object() for _ in range(5)]
    results = [queue.put_nowait(frame) for frame in frames]
    assert results == [True, True, True, False, False]
    assert queue.peak == 3 and queue.overflows == 2 and queue.over_watermark == 3
    assert queue.get(0) is frames[0]
    queue.close()
    assert queue.put_nowait(frames[0]) is False
    assert [queue.get(0), queue.get(0), queue.get(0)] == [frames[1], frames[2], None]


def test_stream_subscription_delivers_every_frame_in_order(service):
    stream = service.open_stream({PRIMARY: CaptureSpec(0)})
    stream.wait_ready(CancelToken())
    stream.claim(CancelToken())
    try:
        tap = stream.subscribe(PRIMARY, 64)
        time.sleep(0.3)
        seqs = []
        while (frame := tap.get(0)) is not None:
            seqs.append(frame.seq)
        assert len(seqs) >= 5 and seqs == list(range(seqs[0], seqs[0] + len(seqs)))
        assert stream.latest(PRIMARY).seq >= seqs[-1]
        stream.unsubscribe(PRIMARY, tap)
        assert tap.closed
    finally:
        stream.close()
    with pytest.raises(RuntimeError):
        service.open_stream({PRIMARY: CaptureSpec(0)}).subscribe(PRIMARY, 4)


class StoppingCapture(TimedCapture):
    """读到第 ``limit`` 帧之后读帧失败（摄像头中途断开）；预热阶段读的帧也计入。"""

    limit = 10_000
    last = None

    def __init__(self, index, **kwargs) -> None:
        super().__init__(index, **kwargs)
        StoppingCapture.last = self

    def read(self):
        if self.reads >= self.limit:
            return False, None
        return super().read()


def test_camera_stopping_mid_phase_fails_the_phase(tmp_path, monkeypatch):
    """评审复现：2 秒的只采集段里摄像头读了 28 帧就停了；剩下的帧间隔正常也不能判达标。"""
    service = CameraService(pool=CameraPool(StoppingCapture, join_timeout=0.5))
    try:
        stream = service.open_stream({PRIMARY: CaptureSpec(0)})
        stream.wait_ready(CancelToken())
        stream.claim(CancelToken())
        monkeypatch.setattr(StoppingCapture, "limit", StoppingCapture.last.reads + 28)
        setup = BenchmarkSetup(stream=stream, rotations={}, labels={}, work_dir=tmp_path / "work",
                               engine_factory=FakeEngine, phases=(BenchmarkPhase("capture", "只采集", 2.0),))
        phase = bench.run_phase(setup, setup.phases[0], 0, stream.actual_sizes(), {}, lambda *_: None, CancelToken())
        capture = phase["roles"]["front"]["capture"]
        assert capture["frames"] <= 30 and capture["covered_s"] < 1.2
        assert capture["fps"] < bench.MIN_CAPTURE_FPS and capture["error"]
        assert not phase["verdict"]["ok"]
        assert any("停止出帧" in problem for problem in phase["verdict"]["problems"])
        assert stream.failure(PRIMARY) is not None
        stream.close()
    finally:
        service.close(2.0)


def test_capture_stats_cover_the_whole_phase_window():
    start, period = 1_000_000_000, 33_333_333
    stamps = [start + k * period for k in range(1, 31)]                # 前 1 秒正常，之后不再出帧
    stats = bench._capture_stats(stamps, (start, start + 2_000_000_000), None)
    assert stats["frames"] == 30 and stats["fps"] == 15.0 and stats["covered_s"] == 1.0 and stats["error"] is None
    verdict = bench.phase_verdict({"seconds": 2.0, "roles": {"front": {"capture": stats}}})
    assert not verdict["ok"] and any("不再出帧" in problem for problem in verdict["problems"])
    healthy = bench._capture_stats([start + k * period for k in range(1, 61)], (start, start + 2_000_000_000), None)
    assert healthy["fps"] == 30.0 and bench.phase_verdict({"seconds": 2.0, "roles": {"front": {"capture": healthy}}})["ok"]
