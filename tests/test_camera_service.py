# -*- coding: utf-8 -*-
"""CameraService / CameraStream（P5）：取消适配、租约归还池（V-1）、名称复核（V-3）、
1080p 严格模式（D-6）、刷新跳过占用设备（V-6）、诊断行与 CLI 冒烟（模拟驱动 + 真实 spawn）。"""
from __future__ import annotations

import functools
import multiprocessing as mp
import threading
import time

import numpy as np
import pytest

from core.foundation import Cancelled, CancelToken
from core.media import (
    CapturedFrame,
    PRIMARY,
    SECONDARY,
    CameraPool,
    CameraService,
    CaptureSpec,
    DeviceCatalog,
)
from core.media.camera import capture_process, devices, pool as pool_module, service as service_module


def _eventually(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


class StreamingCapture:
    """持续出帧的假驱动；记录打开与释放，用于观察设备是否被重新打开。"""

    opened: list["StreamingCapture"] = []

    def __init__(self, index, *, width=1280, height=720, **_kwargs) -> None:
        self.index, self.size = index, (width, height)
        self.release_calls = 0
        self.reads = 0
        StreamingCapture.opened.append(self)

    def isOpened(self) -> bool:
        return self.release_calls == 0

    def get(self, prop):
        return {3: self.size[0], 4: self.size[1], 5: 30.0}.get(prop, 0.0)

    def read(self):
        if self.release_calls:
            return False, None
        time.sleep(0.002)
        self.reads += 1
        return True, np.full((self.size[1] // 80, self.size[0] // 80, 3), self.reads % 255, np.uint8)

    def release(self) -> None:
        self.release_calls += 1


@pytest.fixture
def service():
    StreamingCapture.opened = []
    svc = CameraService(pool=CameraPool(StreamingCapture, join_timeout=0.2))
    try:
        yield svc
    finally:
        svc.close(2.0)


def _session(svc, specs, *, keep_warm):
    stream = svc.open_stream(specs)
    stream.wait_ready(CancelToken())
    stream.claim(CancelToken())
    frames = stream.read()
    stream.close(keep_warm=keep_warm)
    return frames


# --------------------------------------------------------------------------- #
# CameraStream：取消适配与代次
# --------------------------------------------------------------------------- #


def test_cancelled_token_turns_pool_stop_into_cancelled(service):
    token = CancelToken()
    token.cancel("mode_switch")
    stream = service.open_stream({PRIMARY: CaptureSpec(0)})

    with pytest.raises(Cancelled) as info:
        stream.wait_ready(token)

    assert info.value.reason == "mode_switch"


def test_unclaimed_stream_cancels_only_its_observed_generations():
    class _Pool:
        def __init__(self) -> None:
            self.cancel_calls = []

        def wait_pair(self, primary, secondary, *, timeout, stop_event):
            frame = np.zeros((2, 3, 3), np.uint8)
            return (
                CapturedFrame("primary", 1, 0, frame, 3),
                CapturedFrame("secondary", 1, 0, frame, 7),
            )

        def cancel(self, role, *, generation=None):
            self.cancel_calls.append((role, generation))

    fake = _Pool()
    stream = CameraService(pool=fake).open_stream(
        {PRIMARY: CaptureSpec(0), SECONDARY: CaptureSpec(1)}
    )
    stream.wait_ready(CancelToken())
    stream.close()
    stream.close()  # 幂等

    assert fake.cancel_calls == [(PRIMARY, 3), (SECONDARY, 7)]


def test_read_failure_names_the_failed_role_in_dual_stream(service):
    stream = service.open_stream({PRIMARY: CaptureSpec(0), SECONDARY: CaptureSpec(1)})
    stream.wait_ready(CancelToken())
    stream.claim(CancelToken())
    try:
        # 两路预热线程的打开顺序不确定，按设备号找第二路。
        next(cap for cap in StreamingCapture.opened if cap.index == 1).release()  # 第二路断流

        # 断流前已采到的帧照常交出；reader 读到断流后关闭槽，之后的 read 抛出该路的错误。
        with pytest.raises(service_module.CameraReadError, match="第2路"):
            for _ in range(100):
                stream.read()
    finally:
        stream.close()


# --------------------------------------------------------------------------- #
# V-1：会话正常结束且规格不变时，租约归还池继续预热
# --------------------------------------------------------------------------- #


def test_normal_end_returns_lease_to_pool_without_reopening(service):
    specs = {PRIMARY: CaptureSpec(0), SECONDARY: CaptureSpec(1)}
    _session(service, specs, keep_warm=True)

    # 设备没有释放，仍由池的预热 reader 持有，并继续出新帧。
    assert [cap.release_calls for cap in StreamingCapture.opened] == [0, 0]
    _eventually(lambda: service.open_stream(specs).snapshot() is not None)
    service.prepare(specs)  # 会话结束后的 rewarm：规格一致即复用

    _session(service, specs, keep_warm=True)
    assert len(StreamingCapture.opened) == 2


def test_failed_or_explicit_release_closes_the_device(service):
    _session(service, {PRIMARY: CaptureSpec(0)}, keep_warm=False)

    assert StreamingCapture.opened[0].release_calls == 1
    assert not service.device_busy(0)


def test_spec_change_after_return_reopens_with_new_format(service):
    _session(service, {PRIMARY: CaptureSpec(0, 1280, 720)}, keep_warm=True)
    service.prepare({PRIMARY: CaptureSpec(0, 1920, 1080)})

    stream = service.open_stream({PRIMARY: CaptureSpec(0, 1920, 1080)})
    stream.wait_ready(CancelToken())

    assert StreamingCapture.opened[0].release_calls == 1
    assert StreamingCapture.opened[-1].size == (1920, 1080)
    stream.close()


def test_return_is_refused_when_role_was_redeclared_during_session(service):
    stream = service.open_stream({PRIMARY: CaptureSpec(0)})
    stream.wait_ready(CancelToken())
    stream.claim(CancelToken())
    service.prepare({PRIMARY: CaptureSpec(1)})  # 会话期间角色已换成另一台设备

    stream.close(keep_warm=True)

    assert StreamingCapture.opened[0].release_calls == 1


def test_closed_pool_never_takes_lease_back(service):
    stream = service.open_stream({PRIMARY: CaptureSpec(0)})
    stream.wait_ready(CancelToken())
    stream.claim(CancelToken())
    service.close(1.0)

    stream.close(keep_warm=True)

    assert StreamingCapture.opened[0].release_calls == 1


# --------------------------------------------------------------------------- #
# V-3 名称复核与 D-6 严格尺寸（采集子进程内的判定）
# --------------------------------------------------------------------------- #


def test_device_name_mismatch_fails_with_refresh_hint(monkeypatch):
    monkeypatch.setattr(devices, "media_foundation_names", lambda: ["Side", "Front"])

    capture_process._check_device_name(1, "Front")
    capture_process._check_device_name(1, None)
    with pytest.raises(RuntimeError, match="刷新设备列表"):
        capture_process._check_device_name(0, "Front")


def test_device_name_is_not_checked_without_media_foundation(monkeypatch):
    monkeypatch.setattr(devices, "media_foundation_names", lambda: None)

    capture_process._check_device_name(0, "Front")


def test_require_policy_rejects_smaller_frames_prefer_accepts():
    frame = np.zeros((720, 1280, 3), np.uint8)

    capture_process._check_size(frame, 1920, 1080, "prefer")
    capture_process._check_size(frame, 1280, 720, "require")
    with pytest.raises(RuntimeError, match="设备不支持 1920×1080"):
        capture_process._check_size(frame, 1920, 1080, "require")


def test_native_require_tries_every_format_before_failing(monkeypatch):
    import cv2

    opened = []

    class SmallCapture:
        def __init__(self, device, backend) -> None:
            opened.append(self)
            self.released = False

        def isOpened(self):
            return True

        def set(self, prop, value):
            return False  # 驱动拒绝 1080p

        def read(self):
            return True, np.zeros((720, 1280, 3), np.uint8)

        def release(self):
            self.released = True

    monkeypatch.setattr(cv2, "VideoCapture", SmallCapture)
    monkeypatch.setattr(capture_process, "backend_candidates", lambda index: [(cv2.CAP_MSMF, index)])

    with pytest.raises(RuntimeError, match="设备不支持"):
        capture_process._open_native(0, 1920, 1080, "require")
    assert len(opened) == 2 and all(cap.released for cap in opened)  # MJPG 与默认格式都试过


class SmallDriver:
    """子进程里的模拟驱动：无论请求多大都只给 640×360。"""

    def __init__(self, index, *, width, height):
        pass

    def get(self, prop):
        return 30

    def read(self):
        return True, np.zeros((360, 640, 3), np.uint8)

    def release(self):
        pass


def test_process_camera_require_fails_and_reaps_child():
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(RuntimeError, match="设备不支持 1920×1080"):
        capture_process.ProcessCamera(0, width=1920, height=1080, policy="require", _factory=SmallDriver)
    assert {p.pid for p in mp.active_children()} == before

    camera = capture_process.ProcessCamera(0, width=1920, height=1080, _factory=SmallDriver)
    try:
        assert (camera.get(3), camera.get(4)) == (640, 360)  # prefer 接受实际尺寸
    finally:
        camera.release()


def test_pool_forwards_name_and_policy_only_to_factories_that_accept_them():
    seen = []

    def factory(index, *, width, height, stop_event, expected_name=None, policy="prefer"):
        seen.append((index, expected_name, policy))
        return StreamingCapture(index, width=width, height=height)

    def legacy(index, *, width, height):
        seen.append((index, "legacy"))
        return StreamingCapture(index, width=width, height=height)

    for capture_factory, spec in (
        (factory, CaptureSpec(0, 1920, 1080, expected_name="Front", policy="require")),
        (legacy, CaptureSpec(1, expected_name="Side", policy="require")),
    ):
        svc = CameraService(pool=CameraPool(capture_factory, join_timeout=0.2))
        try:
            svc.open_stream({PRIMARY: spec}).wait_ready(CancelToken())
        finally:
            svc.close(1.0)

    assert seen == [(0, "Front", "require"), (1, "legacy")]


def test_capture_spec_rejects_unknown_policy():
    with pytest.raises(ValueError):
        CaptureSpec(0, policy="strict")


# --------------------------------------------------------------------------- #
# V-6：刷新跳过本程序占用的设备
# --------------------------------------------------------------------------- #


def test_fallback_probe_skips_busy_devices_and_marks_them():
    probed = []

    def probe(index):
        probed.append(index)
        return index == 2

    found = devices.enumerate_cameras(3, probe, lambda: [], busy=lambda index: index == 0)

    assert probed == [1, 2, 3]
    assert [(d.index, d.label, d.busy) for d in found] == [
        (0, "摄像头 0（使用中）", True),
        (2, "摄像头 2", False),
    ]


def test_named_enumeration_records_identity_without_probing():
    found = devices.enumerate_cameras(5, lambda _i: pytest.fail("no probe"), lambda: ["USB", "USB", "Side"])

    assert [(d.index, d.name, d.name_unique) for d in found] == [
        (0, "USB", False),
        (1, "USB", False),
        (2, "Side", True),
    ]


def test_service_refresh_uses_pool_occupancy_and_keeps_warm_reader(service):
    service.prepare({PRIMARY: CaptureSpec(0)})
    _eventually(lambda: StreamingCapture.opened and StreamingCapture.opened[0].reads > 0)
    probed = []
    service._catalog = DeviceCatalog(names=lambda: [], probe=lambda i: probed.append(i) or False, scan_limit=1)
    delivered = []

    handle = service.refresh(on_done=delivered.append)

    assert handle.join(2.0)
    assert probed == [1]
    assert [(d.index, d.busy) for d in delivered[0].devices] == [(0, True)]
    assert service.catalog() == delivered[0]
    assert StreamingCapture.opened[0].release_calls == 0  # 刷新没有打断预热


def test_catalog_failure_keeps_previous_snapshot():
    def boom(*_args):
        raise RuntimeError("driver crashed")

    catalog = DeviceCatalog(names=lambda: ["Front"])
    first = catalog.refresh()
    catalog._names, catalog._probe = (lambda: []), boom

    failed = catalog.refresh()

    assert first.ok and first.name_of(0) == "Front"
    assert not failed.ok and failed.devices == first.devices
    assert catalog.snapshot() == first


# --------------------------------------------------------------------------- #
# 诊断行
# --------------------------------------------------------------------------- #


def test_pool_events_are_written_as_diag_lines(capsys):
    service_module.log_pool_event({
        "event": "ready", "role": PRIMARY, "index": 0, "requested": "1280x720",
        "generation": 2, "actual": "1280x720", "open_ms": 812.5,
    })
    service_module.log_pool_event({
        "event": "release_failed", "role": SECONDARY, "index": 1, "requested": "1280x720",
        "generation": 4, "error": "stuck",
    })

    lines = capsys.readouterr().err.splitlines()
    assert "event=camera.ready role=primary device=0 generation=2" in lines[0]
    assert "elapsed_ms=812.5" in lines[0]
    assert "event=device.state to=quarantined cause=release_failed" in lines[1]
    assert "error=stuck" in lines[1]


# --------------------------------------------------------------------------- #
# CLI 实时冒烟：模拟驱动 + 真实 spawn，经 CameraService 打开
# --------------------------------------------------------------------------- #


class CliDriver:
    def __init__(self, index, *, width, height):
        self.size = (width, height)
        self.calls = 0

    def get(self, prop):
        return 30

    def read(self):
        self.calls += 1
        return True, np.full((self.size[1], self.size[0], 3), self.calls % 255, np.uint8)

    def release(self):
        pass


def test_cli_camera_source_opens_through_camera_service(monkeypatch):
    from apps.cli import main as cli
    from core.live import CameraInput

    monkeypatch.setattr(
        pool_module, "ProcessCamera", functools.partial(capture_process.ProcessCamera, _factory=CliDriver)
    )
    before = {p.pid for p in mp.active_children()}

    source = cli.open_input("0")
    token = CancelToken()
    try:
        assert isinstance(source, CameraInput)
        source.wait_ready(token)
        source.start(token)
        # CLI 按默认采集尺寸打开（D-17：1080p，策略仍为 prefer）
        assert source.actual_sizes() == {PRIMARY: (1920, 1080)}
        got = source.read(token)
        assert got.frames[PRIMARY].shape == (1080, 1920, 3)
        assert got.captured_at_ns > 0
    finally:
        source.close(keep_warm=True)

    assert not source.stream.claimed
    _eventually(lambda: {p.pid for p in mp.active_children()} == before, timeout=5.0)


def test_cli_camera_open_failure_is_reported_and_service_closed(monkeypatch):
    from apps.cli import main as cli

    monkeypatch.setattr(
        pool_module, "ProcessCamera",
        functools.partial(capture_process.ProcessCamera, policy="require", _factory=SmallDriver),
    )
    before = {p.pid for p in mp.active_children()}

    with pytest.raises(RuntimeError, match="打开输入源失败"):
        cli.run("0", show=False)
    _eventually(lambda: {p.pid for p in mp.active_children()} == before, timeout=5.0)


class DisconnectingCliDriver(CliDriver):
    """出几帧后断流（之后一直读不到画面）。"""

    def read(self):
        self.calls += 1
        if self.calls > 20:
            return False, None
        return True, np.full((self.size[1], self.size[0], 3), self.calls % 255, np.uint8)


def test_cli_run_releases_capture_writer_and_windows_on_disconnect(monkeypatch):
    from apps.cli import main as cli
    import core.live.stages as live_stages

    monkeypatch.setattr(
        pool_module, "ProcessCamera",
        functools.partial(capture_process.ProcessCamera, _factory=DisconnectingCliDriver),
    )
    before = {p.pid for p in mp.active_children()}

    class _Pipe:
        def next_timestamp_ms(self, **_kwargs):
            return 0

        def infer(self, frame, *, timestamp_ms):
            return None, []

        def close(self):
            pass

    class _Writer:
        def __init__(self, *_args):
            self.release_calls = 0

        def isOpened(self):
            return True

        def write(self, _frame):
            pass

        def release(self):
            self.release_calls += 1

    inputs, writers, destroyed = [], [], []
    cli_open = cli.open_input
    monkeypatch.setattr(cli, "open_input", lambda source: inputs.append(cli_open(source)) or inputs[-1])
    monkeypatch.setattr(live_stages, "create_engine", lambda *_a, **_k: _Pipe())
    monkeypatch.setattr(cli.cv2, "VideoWriter", lambda *a: writers.append(_Writer(*a)) or writers[-1])
    monkeypatch.setattr(cli.cv2, "destroyAllWindows", lambda: destroyed.append(True))

    with pytest.raises(RuntimeError, match="运行失败"):
        cli.run("0", show=False, out_path="unused.mp4")

    assert [w.release_calls for w in writers] == [1]
    assert destroyed == [True]
    assert not inputs[0].stream.claimed
    _eventually(lambda: {p.pid for p in mp.active_children()} == before, timeout=5.0)


def test_no_thread_left_running_after_service_close(service):
    _session(service, {PRIMARY: CaptureSpec(0)}, keep_warm=True)

    assert service.close(2.0)
    _eventually(lambda: not any(
        t.name.startswith("camera-warmup-") and t.is_alive() for t in threading.enumerate()
    ))
