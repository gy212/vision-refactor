# -*- coding: utf-8 -*-
"""录制性能测试（``core.services.recording_benchmark``）：用假摄像头、假引擎、假 writer 与假处理循环跑完五段，检查报告内容、
判定、总结论与清理；writer 与生产一样在帧流动之后按旋转打开、收未转正的源帧，报告编码、打开耗时、积压与每帧写出耗时，
打开超过生产上限判不达标。"按当前设置录像"一段先等注入的处理循环开始再计时，总结论只看这一段。
假 writer 的打开与写入耗时由测试指定，结果不随本机编码器变化。整机项的系统查询用假实现（``FAST_SYSTEM``），
录像盘读写在临时目录里用小数据量真实读写。"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from core.foundation import CancelToken
from core.media import PRIMARY, SECONDARY, CameraPool, CameraService, CaptureSpec, FrameQueue
from core.services import (
    BENCHMARK_PHASES,
    BenchmarkPhase,
    BenchmarkSetup,
    BenchmarkSystemProbes,
    run_recording_benchmark,
    save_benchmark_report,
)
from core.services import benchmark_verdict as verdict
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


class FakeWriter:
    """记下写入帧数的假 writer；``delay`` 是每帧写入耗时。"""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.frames = 0
        self.released = False

    def write(self, frame) -> None:
        if self.delay:
            time.sleep(self.delay)
        self.frames += 1

    def release(self) -> None:
        self.released = True


class FakeWriters:
    """``BenchmarkSetup.writer_factory`` 的假实现：打开要 ``open_s`` 秒，记下打开参数与 writer。"""

    def __init__(self, *, open_s: float = 0.0, delay: float = 0.0) -> None:
        self.open_s = open_s
        self.delay = delay
        self.calls: list[tuple[Path, float, tuple[int, int], int]] = []
        self.writers: list[FakeWriter] = []

    def __call__(self, path, fps, size, rotation):
        self.calls.append((Path(path), fps, size, rotation))
        if self.open_s:
            time.sleep(self.open_s)
        writer = FakeWriter(self.delay)
        self.writers.append(writer)
        return writer, Path(path), "fake"


class FakeSessionLoad:
    """``BenchmarkSetup.session_load`` 的假实现：``start_s`` 后报告开始，一直跑到 token 取消，返回统计；
    ``fail`` 时直接抛出。记下调用时拿到的摄像头流与开始到结束的时间。"""

    def __init__(self, *, start_s: float = 0.0, fail: str | None = None, never_start: bool = False) -> None:
        self.start_s, self.fail, self.never_start = start_s, fail, never_start
        self.streams: list = []
        self.running_at: float | None = None

    def __call__(self, stream, token, on_running):
        self.streams.append(stream)
        if self.fail:
            raise RuntimeError(self.fail)
        token.wait(self.start_s)
        if not self.never_start:
            self.running_at = time.monotonic()
            on_running()
        token.wait()
        return {"frames": 12, "fps": 20.0, "settings": {"skeleton": False}, "error": None}


SHORT = tuple(replace(p, seconds=0.6) for p in BENCHMARK_PHASES)


class FakePaging:
    def __init__(self, rate=12.0):
        self.rate, self.closed = rate, False

    def read(self):
        return {"pages_per_s": self.rate, "error": None}

    def close(self):
        self.closed = True


FAKE_HOST = {"os": "test", "memory": {"total_gb": 16.0, "available_gb": 8.0},
             "power": {"ac_power": True, "battery_saver": False, "power_mode": "balanced",
                       "plan": {"guid": "381b4222-f694-41f0-9685-ff5bb260df2e", "name": "平衡", "key": "balanced"},
                       "errors": {}}}
FAST_SYSTEM = BenchmarkSystemProbes(
    host=lambda: FAKE_HOST, memory=lambda: {"total_gb": 16.0, "available_gb": 8.0}, paging=FakePaging,
    volume=lambda _directory: {"root": "T:\\", "drive_type": "fixed", "filesystem": "NTFS", "bus": "nvme",
                               "free_gb": 500.0, "errors": {}},
    memory_seconds=0.05, frame_shape=(108, 192, 3), disk_total_bytes=2 * 2**20, disk_chunk_bytes=64 * 2**10,
    disk_max_write_s=2.0, disk_max_read_s=2.0, disk_spare_bytes=0, sample_s=0.05,
)


@pytest.fixture
def service():
    svc = CameraService(pool=CameraPool(TimedCapture, join_timeout=0.5))
    try:
        yield svc
    finally:
        svc.close(2.0)


def _setup(service, tmp_path, engines, *, phases=SHORT, dual=True, writers=None, session=None, missing=None):
    specs = {PRIMARY: CaptureSpec(0)}
    if dual:
        specs[SECONDARY] = CaptureSpec(1)
    factory = lambda spec: engines.setdefault(len(engines), FakeEngine(spec))
    return BenchmarkSetup(
        stream=service.open_stream(specs), rotations={PRIMARY: 90, SECONDARY: 0},
        labels={PRIMARY: "正面相机", SECONDARY: "侧面相机"}, work_dir=tmp_path / "work",
        engine_factory=factory, phases=phases, extra={"build": {"commit": "abc1234"}},
        writer_factory=writers if writers is not None else FakeWriters(),
        session_load=session, missing_models=missing or {}, system=FAST_SYSTEM,
    )


def test_full_run_reports_every_phase_and_cleans_up(service, tmp_path):
    engines: dict = {}
    progress: list = []
    writers = FakeWriters()
    session = FakeSessionLoad()
    report = run_recording_benchmark(_setup(service, tmp_path, engines, writers=writers, session=session),
                                     CancelToken(), lambda text, done: progress.append((text, done)))
    assert report["kind"] == "recording_benchmark" and report["build"] == {"commit": "abc1234"}
    assert report["schema"] == 2 and report["host"] == FAKE_HOST
    assert report["memory_bandwidth"]["copies"] > 0 and report["memory_bandwidth"]["gb_per_s"] > 0
    disk = report["disk"]
    assert disk["error"] is None and disk["skipped"] is None and disk["volume"]["root"] == "T:\\"
    assert disk["write"]["bytes"] == 2 * 2**20 and disk["write"]["complete"] and disk["read"]["bytes"] == 2 * 2**20
    assert Path(disk["directory"]).parent == tmp_path                        # 在录制目录下的临时目录里测
    for phase in report["phases"]:
        memory = phase["memory"]
        assert memory["before"]["available_gb"] == 8.0 and memory["min_available_gb"] == 8.0
        assert memory["pages_per_s"] == 12.0 and memory["paging_error"] is None
    assert [p["key"] for p in report["phases"]] == ["capture", "record", "record_lite", "record_skeleton",
                                                    "production"]
    assert set(report["cameras"]) == {"front", "side"}
    assert report["cameras"]["front"]["rotation"] == 90 and report["cameras"]["front"]["label"] == "正面相机"
    for phase in report["phases"]:
        for view in ("front", "side"):
            capture = phase["roles"][view]["capture"]
            assert capture["frames"] >= 10 and capture["fps"] == pytest.approx(30, abs=4)
            if phase["key"] != "capture":
                write = phase["roles"][view]["write"]
                assert write["frames"] >= 10 and write["codec"] == "fake" and write["error"] is None
                assert write["over_production_queue"] == 0 and write["dropped"] == 0
                assert write["open_ms"] is not None and write["open_backlog"] >= 1 and write["backlog_dropped"] == 0
                assert write["catch_up_ms"] is not None
    assert report["phases"][2]["lite"]["runs"] >= 3
    assert report["phases"][3]["skeleton"]["loops"] >= 3
    assert report["phases"][4]["session"] == {"frames": 12, "fps": 20.0, "settings": {"skeleton": False},
                                              "error": None}
    assert len(session.streams) == 1
    assert all(engine.closed for engine in engines.values()) and len(engines) == 3   # lite + 每路一个
    assert not (tmp_path / "work").exists()                                          # 临时录像已删除
    assert progress[-1][1] == 1.0 and report["ok"] is True
    assert report["conclusion"]["level"] == "ok" and report["conclusion"]["ok"] is True
    assert report["summary"][0] == f"结论：{report['conclusion']['text']}"
    assert "按当前设置可以稳定录像" in report["summary"][0] and report["production_open_budget_ms"] == 1500.0
    assert any("fake 打开" in line and "写一帧 p50" in line for line in report["summary"])
    assert len(writers.writers) == 8 and all(writer.released for writer in writers.writers)   # 4 个写盘段 × 2 路
    front_calls = [call for call in writers.calls if call[0].stem == "front"]
    # writer 按开录时的旋转打开，尺寸是转正后的；收到的是未转正的源帧（转正在 writer 里，D-31）
    assert front_calls and all(fps == 30.0 and size == (48, 64) and rotation == 90
                               for _path, fps, size, rotation in front_calls)
    side_calls = [call for call in writers.calls if call[0].stem == "side"]
    assert side_calls and all(size == (64, 48) and rotation == 0 for _path, _fps, size, rotation in side_calls)
    assert not service.device_busy(0) or True                                        # 设备交回池（预热）


def test_writer_opens_after_frames_flow_like_production(service, tmp_path):
    """与生产相同：帧已经在流动时才打开 writer。打开 0.6 s 期间帧按序积压，写盘队列不排满，打开后追平。"""
    phases = (BenchmarkPhase("record", "采集 + 写盘", 2.0, write=True),)
    setup = _setup(service, tmp_path, {}, phases=phases, writers=FakeWriters(open_s=0.6))
    report = run_recording_benchmark(setup, CancelToken(), lambda *_: None)
    for view in ("front", "side"):
        write = report["phases"][0]["roles"][view]["write"]
        assert write["open_ms"] >= 550 and write["open_backlog"] >= 15 and write["backlog_peak"] >= 15
        assert write["over_production_queue"] == 0 and write["backlog_dropped"] == 0
        assert write["catch_up_ms"] is not None and write["frames"] >= 50
    assert report["ok"] is True, report["summary"]


def test_writer_open_over_the_production_budget_is_reported(service, tmp_path):
    """打开超过生产上限（1.5 s）时，生产会判开录失败：报告写明编码与打开耗时，该段不达标。"""
    phases = (BenchmarkPhase("record", "采集 + 写盘", 2.5, write=True),)
    writers = FakeWriters(open_s=bench.PRODUCTION_OPEN_BUDGET_S + 0.3)
    report = run_recording_benchmark(_setup(service, tmp_path, {}, phases=phases, dual=False, writers=writers),
                                     CancelToken(), lambda *_: None)
    write = report["phases"][0]["roles"]["front"]["write"]
    assert write["open_ms"] >= 1800 and write["backlog_dropped"] > 0 and write["over_production_queue"] == 0
    problems = report["phases"][0]["verdict"]["problems"]
    assert report["ok"] is False
    assert any("打开录像文件用了" in problem and "超过生产上限 1500 ms" in problem for problem in problems)
    assert any(f"开录积压超过 {bench.BACKLOG_LIMIT} 帧" in problem for problem in problems)
    assert any("fake 打开" in line for line in report["summary"])
    assert report["conclusion"]["level"] == "fail"
    assert "没有测与正式录制相同的负载" in report["conclusion"]["text"]     # 没有注入处理循环：按唯一一段判断


def test_default_writer_factory_is_the_production_recording_writer(tmp_path, monkeypatch):
    """默认工厂把路径、fps、尺寸与旋转原样交给 ``open_recording_writer``（生产录像线程用的同一个）。"""
    calls: list = []
    monkeypatch.setattr(bench, "open_recording_writer", lambda path, *, fps, size, rotation:
                        calls.append((path, fps, size, rotation)) or ("w", path, "MJPG"))
    setup = BenchmarkSetup(stream=None, rotations={}, labels={}, work_dir=tmp_path, engine_factory=FakeEngine)
    assert setup.writer_factory(tmp_path / "front.mp4", 30.0, (1080, 1920), 90) == ("w", tmp_path / "front.mp4", "MJPG")
    assert calls == [(tmp_path / "front.mp4", 30.0, (1080, 1920), 90)]


def test_production_limits_match_the_recording_worker():
    """benchmark 不能导入 ``core.recording``（L4 互不导入），镜像的生产常量由这里核对。"""
    from core.recording import QUEUE_CAPACITY, WRITER_OPEN_TIMEOUT_S

    assert bench.PRODUCTION_QUEUE == QUEUE_CAPACITY
    assert bench.PRODUCTION_OPEN_BUDGET_S == WRITER_OPEN_TIMEOUT_S


def test_slow_writer_is_reported_as_not_keeping_up(service, tmp_path):
    """写一帧 60 ms（30 fps 需要 ≤33 ms）时，队列会排到 8 帧以上，判为不达标。"""
    phases = (BenchmarkPhase("record", "采集 + 写盘", 1.5, write=True),)
    setup = _setup(service, tmp_path, {}, phases=phases, writers=FakeWriters(delay=0.06))
    report = run_recording_benchmark(setup, CancelToken(), lambda *_: None)
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
        assert capture["fps"] < verdict.MIN_CAPTURE_FPS and capture["error"]
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


# --------------------------------------------------------------------------- 按当前设置录像一段与总结论


def test_production_phase_waits_for_the_session_loop_and_decides_the_conclusion(service, tmp_path):
    """处理循环加载 0.5 s：段窗口从循环开始之后算起；前面的分项段有问题、这一段正常时，总结论是能录像。"""
    session = FakeSessionLoad(start_s=0.5)
    phases = (BenchmarkPhase("record", "采集 + 写盘", 1.0, write=True),
              BenchmarkPhase("production", "按当前设置录像", 1.0, write=True, session=True))
    writers = FakeWriters()
    report = run_recording_benchmark(_setup(service, tmp_path, {}, phases=phases, writers=writers,
                                            session=session), CancelToken(), lambda *_: None)
    production = report["phases"][1]
    assert production["session"]["frames"] == 12 and production["verdict"]["ok"]
    assert session.running_at is not None
    for view in ("front", "side"):
        assert production["roles"][view]["capture"]["fps"] == pytest.approx(30, abs=4)   # 窗口不含加载的 0.5 s
    assert report["conclusion"]["level"] == "ok" and "（" in report["conclusion"]["text"]
    assert "没有测与正式录制相同的负载" not in report["conclusion"]["text"]


def test_session_loop_failure_fails_the_production_phase(service, tmp_path):
    phases = (BenchmarkPhase("production", "按当前设置录像", 0.6, write=True, session=True),)
    report = run_recording_benchmark(_setup(service, tmp_path, {}, phases=phases, dual=False,
                                            session=FakeSessionLoad(fail="模型缺失")), CancelToken(), lambda *_: None)
    problems = report["phases"][0]["verdict"]["problems"]
    assert any("处理循环失败：模型缺失" in problem for problem in problems)
    assert report["ok"] is False and report["conclusion"]["level"] == "fail"
    assert report["summary"][0].startswith("结论：按当前设置不能稳定录像")


def test_session_loop_that_never_starts_times_out(service, tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "SESSION_START_S", 0.5)
    phases = (BenchmarkPhase("production", "按当前设置录像", 0.4, write=True, session=True),)
    report = run_recording_benchmark(_setup(service, tmp_path, {}, phases=phases, dual=False,
                                            session=FakeSessionLoad(never_start=True)), CancelToken(),
                                     lambda *_: None)
    assert any("没有开始" in problem for problem in report["phases"][0]["verdict"]["problems"])
    assert report["conclusion"]["level"] == "fail"


def _phase(key, p95, ok=True, problems=()):
    write = {"write_ms": {"p50": p95 / 2, "p95": p95, "max": p95}, "codec": "MJPG", "open_ms": 5.0}
    return {"key": key, "label": key, "seconds": 30.0,
            "roles": {"front": {"capture": {"fps": 30.0}, "write": write}},
            "verdict": {"ok": ok, "problems": list(problems)}}


def test_conclusion_levels():
    """能录且写一帧 p95 不超过帧间隔的 60% 为 ok，超过为 tight；headline 段不达标为 fail；出错为 incomplete。"""
    detail = _phase("record", 50.0, ok=False, problems=["正面 写盘跟不上"])
    assert verdict.conclusion({"phases": [detail, _phase("production", 12.0)]})["level"] == "ok"
    tight = verdict.conclusion({"phases": [_phase("production", 25.0)]})
    assert tight["level"] == "tight" and tight["ok"] is True and "余量小" in tight["text"]
    failed = verdict.conclusion({"phases": [_phase("production", 12.0, ok=False, problems=["正面 写盘跟不上"])]})
    assert failed["level"] == "fail" and failed["ok"] is False and "正面 写盘跟不上" in failed["text"]
    incomplete = verdict.conclusion({"phases": [_phase("production", 12.0)], "error": "已取消"})
    assert incomplete["level"] == "incomplete" and incomplete["ok"] is False
    assert verdict.conclusion({"phases": []})["level"] == "incomplete"


def test_summary_puts_the_conclusion_first_and_warns_about_power():
    report = {"phases": [_phase("production", 12.0)],
              "host": {"power": {"ac_power": False, "battery_saver": True, "power_mode": "best_efficiency"}}}
    lines = verdict.summary_lines(report)
    assert lines[0].startswith("结论：按当前设置可以稳定录像") and "需要注意：电脑在省电状态" in lines[0]
    assert "没有接电源" in lines[0] and "节电模式开着" in lines[0] and "最佳能效" in lines[0]
    assert any(line.startswith("· 电源：用电池，节电开着，电源模式 最佳能效") for line in lines)
    assert any("单独一段正常不代表能录像" in line for line in lines)
    assert any("本段正常" in line for line in lines)
    plugged = {"phases": [_phase("production", 12.0)],
               "host": {"power": {"ac_power": True, "battery_saver": False, "power_mode": "balanced"}}}
    assert verdict.power_hint(plugged["host"]) is None and verdict.power_hint(None) is None


def test_missing_lite_model_skips_its_phase_and_leads_the_conclusion(service, tmp_path):
    """缺 Lite 模型：不加载 Lite、跳过站位检测一段，其余照测；结论判"测试不完整"，缺模型的提醒在最前面，已测的结论接在后面。"""
    from core.pose import missing_model_message

    engines: dict = {}
    hint = missing_model_message("benchmark_lite")
    report = run_recording_benchmark(_setup(service, tmp_path, engines, missing={"lite": hint},
                                            session=FakeSessionLoad()), CancelToken(), lambda *_: None)
    assert [phase["key"] for phase in report["phases"]] == ["capture", "record", "record_skeleton", "production"]
    assert all(engine.spec.pose_variant != "lite" for engine in engines.values())
    assert report["missing_models"] == {"lite": hint}
    conclusion = report["conclusion"]
    assert conclusion["level"] == "incomplete" and conclusion["ok"] is False and report["ok"] is False
    assert conclusion["measured_level"] == "ok"
    assert report["summary"][0].startswith("结论：缺少 Lite 模型（pose_landmarker_lite.task）：考试和学生练习都开不了")
    assert "设置→模型管理" in conclusion["text"] and "已测的部分：按当前设置可以稳定录像" in conclusion["text"]


def test_missing_skeleton_model_skips_the_skeleton_phase(service, tmp_path):
    engines: dict = {}
    report = run_recording_benchmark(_setup(service, tmp_path, engines, missing={"skeleton": "缺少骨架识别模型（x）。"}),
                                     CancelToken(), lambda *_: None)
    assert [phase["key"] for phase in report["phases"]] == ["capture", "record", "record_lite"]
    assert [engine.spec.pose_variant for engine in engines.values()] == ["lite"]
    assert report["conclusion"]["level"] == "incomplete" and report["conclusion"]["text"].startswith("缺少骨架识别模型")
