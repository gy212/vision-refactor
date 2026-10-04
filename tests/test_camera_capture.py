"""真实 spawn + 模拟驱动：不访问摄像头，覆盖原生阻塞和重开。"""
import multiprocessing as mp
import threading
import time

import numpy as np
import pytest

from apps import camera_capture


class FakeCapture:
    def __init__(self, index, **kwargs):
        self.index = index
        self.calls = 0
        if index == 1:
            time.sleep(60)  # 模拟 native open 永不返回

    def get(self, prop):
        return 30

    def read(self):
        self.calls += 1
        if self.calls > 1 and self.index == 2:
            time.sleep(60)  # 模拟 native read 永不返回
        if self.calls > 1 and self.index == 4:
            raise RuntimeError("(-215:Assertion failed) _step >= minstep in cv::Mat::Mat")
        # 负 stride 只用于检查跨进程输出必定是独立连续数组。
        return True, np.full((3, 4, 3), self.calls, np.uint8)[:, ::-1]

    def release(self):
        if self.index == 3:
            time.sleep(60)  # 模拟 native release 卡死


class HealthyCapture(FakeCapture):
    def __init__(self, index, **kwargs):
        super().__init__(0, **kwargs)


def test_frames_own_memory_and_capture_can_reopen():
    for _ in range(2):
        camera = camera_capture.ProcessCamera(0, _factory=FakeCapture)
        try:
            _, first = camera.read()
            _, second = camera.read()
            assert first.flags.c_contiguous and first.flags.owndata
            assert np.all(first == 1) and np.all(second == 2)
            assert (camera.get(3), camera.get(4)) == (4, 3)
        finally:
            camera.release()
        assert not camera.isOpened()


@pytest.mark.parametrize("index", [2, 4])
def test_read_hang_or_stride_error_reaps_process_and_allows_reopen(index):
    camera = camera_capture.ProcessCamera(index, read_timeout=0.2, _factory=FakeCapture)
    pid = camera._process.pid
    camera.read()  # 已验证的首帧
    try:
        with pytest.raises((TimeoutError, RuntimeError)):
            camera.read()
        assert not camera.isOpened()
        assert pid not in {child.pid for child in mp.active_children()}
        recovered = camera_capture.ProcessCamera(index, _factory=HealthyCapture)
        try:
            assert recovered.read()[0]
        finally:
            recovered.release()
    finally:
        camera.release()


def test_cancel_blocked_open_reaps_child():
    stop = threading.Event()
    before = {p.pid for p in mp.active_children()}
    timer = threading.Timer(0.8, stop.set)
    timer.start()
    try:
        with pytest.raises(RuntimeError, match="取消"):
            camera_capture.ProcessCamera(1, stop_event=stop, _factory=FakeCapture)
        assert {p.pid for p in mp.active_children()} == before
    finally:
        timer.cancel()


def test_blocked_release_is_bounded_and_idempotent():
    camera = camera_capture.ProcessCamera(3, _factory=FakeCapture)
    pid = camera._process.pid
    started = time.monotonic()
    camera.release()
    camera.release()
    assert time.monotonic() - started < 3
    assert not camera.isOpened()
    assert pid not in {child.pid for child in mp.active_children()}


def test_native_stride_failure_retries_format_before_exposing_capture(monkeypatch):
    import cv2

    opened = []

    class FormatCapture:
        def __init__(self, index, backend):
            self.fail = not opened
            self.released = False
            self.settings = []
            opened.append(self)

        def isOpened(self):
            return True

        def set(self, prop, value):
            self.settings.append((prop, value))

        def read(self):
            if self.fail:
                raise cv2.error("(-215:Assertion failed) _step >= minstep in cv::Mat::Mat")
            return True, np.zeros((2, 3, 3), np.uint8)

        def release(self):
            self.released = True

    monkeypatch.setattr(cv2, "VideoCapture", FormatCapture)
    monkeypatch.setattr(camera_capture, "_camera_sources", lambda index: [(cv2.CAP_MSMF, index)])
    cap, first = camera_capture._open_native(0, 1920, 1080)
    assert opened[0].released and cap is opened[1]
    assert all((cv2.CAP_PROP_FRAME_WIDTH, 1920) in c.settings for c in opened)
    assert first.flags.c_contiguous
    cap.release()


def test_transient_empty_frames_do_not_end_session(monkeypatch):
    class StartupCapture:
        calls = 0

        def read(self):
            self.calls += 1
            if self.calls < 3:
                return False, None
            return True, np.zeros((2, 3, 3), np.uint8)

    monkeypatch.setattr(camera_capture.time, "sleep", lambda _: None)
    capture = StartupCapture()
    assert camera_capture._read_frame(capture).shape == (2, 3, 3)
    assert capture.calls == 3


def test_ambiguous_camera_names_never_fall_back_to_another_index(monkeypatch):
    import cv2
    from apps import camera_enum
    monkeypatch.setattr(camera_enum, "_enumerate_media_foundation_names",
                        lambda: ["USB Camera", "USB Camera"])
    assert list(camera_capture._camera_sources(1)) == [(cv2.CAP_MSMF, 1)]


def test_process_pair_survives_warmup_thread_handoff_and_reopens():
    from apps.camera_warmup import CameraWarmupPool, CaptureSpec

    def factory(index, *, width, height, stop_event):
        return camera_capture.ProcessCamera(
            index, width=width, height=height, stop_event=stop_event,
            _factory=HealthyCapture)

    pool = CameraWarmupPool(capture_factory=factory, join_timeout=0.1)
    try:
        for size in ((1280, 720), (1920, 1080)):
            specs = (CaptureSpec(0, *size), CaptureSpec(1, *size))
            pool.wait_pair(*specs, timeout=10, stop_event=threading.Event())
            leases = pool.claim_pair(*specs, timeout=3)
            try:
                # claim 已设置预热 stop_event；新线程接管后仍须能正常读帧。
                for lease in leases:
                    assert lease.read()[0]
                    assert pool.device_busy(lease.spec.index)
            finally:
                for lease in leases:
                    lease.release()
            assert not pool.device_busy(0) and not pool.device_busy(1)
    finally:
        pool.close()


def test_tk_preview_student_switch_with_process_cameras(monkeypatch, tmp_path):
    tk = pytest.importorskip("tkinter")
    from apps import app_ui, desktop_smoke
    from core import paths

    monkeypatch.setattr(app_ui.App, "_start_enumeration", lambda self: None)
    monkeypatch.setattr(paths, "outputs_dir", lambda: tmp_path)
    monkeypatch.setattr(app_ui, "outputs_dir", lambda: tmp_path)
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display is unavailable: {exc}")
    root.withdraw()
    app = None
    try:
        app = app_ui.App(root)
        root.update()
        desktop_smoke._check_camera_mode_switch(app, root)
    finally:
        if app is not None:
            app._on_close()
        try:
            root.destroy()
        except tk.TclError:
            pass
