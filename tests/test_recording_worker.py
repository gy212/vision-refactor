# -*- coding: utf-8 -*-
"""D-18 第 6 层：摄像头会话的每路录像线程（``ViewWorker``）经 ``SegmentRecorder`` 驱动。

假帧来源按真实时间以 30 fps 产出带采集时刻的帧（像素值编码序号），用真实的 ``FrameQueue``；
检查：帧数与时长、两路帧数相同、不把边界之后的帧写进片段、暂停扣除、段内方向固定（D-29）、
队列溢出与断流使该段失败（D-27）、``capture.json``，以及真实 writer 写出后解码检查画面内容的时间。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.media import (
    PRIMARY,
    SECONDARY,
    CameraReadError,
    CapturedFrame,
    FrameQueue,
    capture_clock_ns,
    open_video_writer,
    read_capture_record,
)
from core.recording import RecordingService, SegmentFinalizer, SegmentMeta, SegmentRecorder

P_NS = 1e9 / 30


class FakeFeed:
    """每路一个生产线程：按 ``fps`` 产出帧，``captured_at_ns`` 取产出时刻；``phase`` 让两路错开。"""

    def __init__(self, roles=(PRIMARY, SECONDARY), *, fps=30.0, shape=(4, 6, 3), phase_s=None) -> None:
        self._fps = fps
        self._shape = shape
        self._taps: dict[str, list[FrameQueue]] = {role: [] for role in roles}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._failed: dict[str, BaseException] = {}
        self._halt: dict[str, threading.Event] = {role: threading.Event() for role in roles}
        phases = phase_s or {role: 0.013 * i for i, role in enumerate(roles)}
        self._threads = [
            threading.Thread(target=self._produce, args=(role, phases[role]), daemon=True) for role in roles
        ]
        for thread in self._threads:
            thread.start()

    def _produce(self, role: str, phase: float) -> None:
        seq = 0
        period = 1.0 / self._fps
        next_t = time.perf_counter() + phase
        while not self._stop.is_set() and not self._halt[role].is_set():
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            seq += 1
            image = np.full(self._shape, seq % 256, dtype=np.uint8)
            frame = CapturedFrame(role, seq, capture_clock_ns(), image, 7)
            with self._lock:
                taps = list(self._taps[role])
            for tap in taps:
                tap.put_nowait(frame)
            next_t += period
        with self._lock:
            taps, self._taps[role] = list(self._taps[role]), []
        for tap in taps:
            tap.close()

    def subscribe(self, role, capacity, *, watermark=None):
        tap = FrameQueue(capacity, watermark=watermark)
        with self._lock:
            self._taps[role].append(tap)
        return tap

    def unsubscribe(self, role, tap):
        with self._lock:
            if tap in self._taps[role]:
                self._taps[role].remove(tap)
        tap.close()

    def failure(self, role):
        return self._failed.get(role)

    def break_role(self, role: str) -> None:
        """模拟该路断流：reader 报错停止，队列关闭。"""
        self._failed[role] = CameraReadError(role, numbered=True)
        self._halt[role].set()

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(1.0)


class RecordingWriter:
    def __init__(self, path: Path, delay: float = 0.0) -> None:
        self.path = path
        self.frames: list[np.ndarray] = []
        self.delay = delay
        self.released = False

    def write(self, frame) -> None:
        if self.delay:
            time.sleep(self.delay)
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


class Writers:
    """writer 工厂：按视角（文件名）记下每次打开的 writer。"""

    def __init__(self, delay: float = 0.0) -> None:
        self.opened: dict[str, list[RecordingWriter]] = {}
        self.order: list[RecordingWriter] = []   # 按打开顺序（单路片段的文件名是片段 id）
        self.delay = delay

    def __call__(self, path, _fps, _size):
        writer = RecordingWriter(Path(path), self.delay)
        self.opened.setdefault(Path(path).stem, []).append(writer)
        self.order.append(writer)
        return writer, Path(path), "fake"

    def last(self, view: str) -> RecordingWriter:
        return self.opened[view][-1]


@pytest.fixture
def feed():
    created: list[FakeFeed] = []

    def make(**kwargs) -> FakeFeed:
        created.append(FakeFeed(**kwargs))
        return created[-1]

    yield make
    for item in created:
        item.close()


def _recorder(feed: FakeFeed, writers, *, rotation=None, views=("front", "side")) -> SegmentRecorder:
    recorder = SegmentRecorder(writers)
    sizes = {view: (6, 4) for view in views}
    recorder.begin_session(sizes, 30.0, feed=feed, rotation_of=rotation or (lambda _role: 0))
    return recorder


def test_segment_length_matches_wall_time_and_views_have_equal_frames(tmp_path, feed):
    writers = Writers()
    source = feed()
    recorder = _recorder(source, writers)
    ref = recorder.start_segment(SegmentMeta(tmp_path))
    assert ref
    started = time.perf_counter()
    time.sleep(1.0)
    elapsed = time.perf_counter() - started      # 片段止于结束边界；之后等录像线程写完不算进时长
    finished = recorder.end_segment()
    recorder.close_session()

    front, side = finished.views["front"], finished.views["side"]
    assert front.error is None and side.error is None
    assert front.frames == side.frames
    assert abs(front.frames - elapsed * 30) <= 2
    assert len(writers.last("front").frames) == front.frames and writers.last("front").released


def test_no_frame_captured_after_a_boundary_is_written(tmp_path, feed):
    """capture.json 的每个输出槽都只用采集于该槽时刻之前的源帧，且不晚于结束边界（不把新帧复制到过去）。"""
    source = feed()
    recorder = _recorder(source, Writers())
    ref = recorder.start_segment(SegmentMeta(tmp_path))
    time.sleep(0.5)
    recorder.toggle_pause()
    time.sleep(0.3)
    recorder.toggle_pause()
    time.sleep(0.5)
    finished = recorder.end_segment()
    recorder.close_session()

    record = read_capture_record(ref.capture_path)
    assert record is not None and set(record["views"]) == {"front", "side"}
    for view, data in record["views"].items():
        runs = data["runs"]
        assert len(runs) == 2 and all(end is not None for _start, end in runs)
        slots = [[start + k * P_NS for k in range(len(data["slot_sources"]))] for start, _ in runs[:1]][0]
        offset = 0.0
        k = 0
        for start, end in runs:
            while k < len(data["slot_sources"]) and start + k * P_NS - offset < end:
                assert data["slot_sources"][k][1] <= start + k * P_NS - offset + 1
                k += 1
            offset += end - start
        assert k == len(slots)
        assert data["stats"]["slots"] == finished.views[view].frames
        assert data["queue"]["capacity"] == 8 and data["queue"]["overflows"] == 0
        assert data["frame_clock"]["frames"] > 0 and data["role"] in (PRIMARY, SECONDARY)
    paused_total = sum(end - start for start, end in record["views"]["front"]["runs"])
    assert abs(finished.views["front"].frames - paused_total / P_NS) <= 1
    assert finished.views["front"].frames == finished.views["side"].frames


def test_rotation_is_fixed_at_segment_start(tmp_path, feed):
    """D-29：录制中改旋转不影响本段的方向与尺寸，下一段才生效。"""
    rotation = {PRIMARY: 90}
    writers = Writers()
    source = feed(roles=(PRIMARY,))
    recorder = _recorder(source, writers, rotation=lambda role: rotation[role], views=("front",))
    assert recorder.start_segment(SegmentMeta(tmp_path))
    time.sleep(0.2)
    rotation[PRIMARY] = 0
    time.sleep(0.2)
    recorder.end_segment()
    first = writers.order[-1]
    assert first.frames and {frame.shape[:2] for frame in first.frames} == {(6, 4)}
    assert recorder.start_segment(SegmentMeta(tmp_path))
    time.sleep(0.2)
    recorder.end_segment()
    recorder.close_session()
    second = writers.order[-1]
    assert second is not first and {frame.shape[:2] for frame in second.frames} == {(4, 6)}


def test_slow_writer_overflows_the_queue_and_fails_the_segment(tmp_path, feed):
    """D-27：写盘跟不上采集时队列满，该段失败（已写文件保留），不无限缓存、不静默丢帧。"""
    writers = Writers(delay=0.1)
    source = feed()
    recorder = _recorder(source, writers)
    assert recorder.start_segment(SegmentMeta(tmp_path))
    deadline = time.monotonic() + 3.0
    while not recorder.broken() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert recorder.broken()
    finished = recorder.end_segment()
    recorder.close_session()
    errors = finished.errors
    assert errors and all("录像队列已满（8 帧）" in error for error in errors.values())
    assert all(view.path is not None for view in finished.views.values())


def test_stream_failure_mid_segment_fails_that_view(tmp_path, feed):
    source = feed()
    recorder = _recorder(source, Writers())
    assert recorder.start_segment(SegmentMeta(tmp_path))
    time.sleep(0.3)
    source.break_role(SECONDARY)
    deadline = time.monotonic() + 2.0
    while not recorder.broken() and time.monotonic() < deadline:
        time.sleep(0.02)
    finished = recorder.end_segment()
    recorder.close_session()
    assert finished.errors.keys() == {"side"}
    assert "第2路摄像头读帧失败" in finished.errors["side"]


def test_stop_during_a_stall_completes_after_the_grace_period(tmp_path, feed):
    """结束边界之后迟迟没有新帧（卡顿）时，按已到的帧执行结束，不无限等待。"""
    source = feed(roles=(PRIMARY,))
    recorder = _recorder(source, Writers(), views=("front",))
    assert recorder.start_segment(SegmentMeta(tmp_path))
    time.sleep(0.4)
    source._halt[PRIMARY].set()      # 卡住：不再出帧，也不关闭队列
    started = time.perf_counter()
    finished = recorder.end_segment()
    assert time.perf_counter() - started < 2.0
    recorder.close_session()
    assert finished.views["front"].frames > 0


def test_service_session_close_submits_the_open_segment(tmp_path, feed):
    submitted: list = []

    class Finalizer(SegmentFinalizer):
        def submit(self, segment, plan):
            submitted.append(segment)
            return True

    service = RecordingService(Finalizer(), plan_for=lambda _s: object(), recorder=SegmentRecorder(Writers()))
    source = feed()
    service.begin_session({"front": (6, 4), "side": (6, 4)}, 30.0, feed=source)
    assert service.start(SegmentMeta(tmp_path))
    time.sleep(0.3)
    outcome = service.close_session()
    assert outcome.submitted and submitted[0].views["front"].frames == submitted[0].views["side"].frames > 0


def test_real_writer_output_shows_each_frame_no_earlier_than_it_was_captured(tmp_path, feed):
    """解码检查：真实 writer（MJPG）写出后，第 k 帧画面的源序号单调不减，且源帧采集于该输出时刻之前。"""
    source = feed(roles=(PRIMARY,), shape=(48, 64, 3))
    recorder = SegmentRecorder(lambda path, fps, size: open_video_writer(Path(path).with_suffix(".avi"),
                                                                          fps=fps, size=size))
    recorder.begin_session({"front": (64, 48)}, 30.0, feed=source)
    ref = recorder.start_segment(SegmentMeta(tmp_path))
    time.sleep(1.0)
    finished = recorder.end_segment()
    recorder.close_session()

    path = finished.views["front"].path
    capture = cv2.VideoCapture(str(path))
    values = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        values.append(int(round(float(image.mean()))))
    capture.release()
    record = read_capture_record(ref.capture_path)["views"]["front"]
    expected = [seq % 256 for seq, _at in record["slot_sources"]]
    assert len(values) == finished.views["front"].frames == len(expected)
    assert all(abs(a - b) <= 2 for a, b in zip(values, expected))
    start = record["runs"][0][0]
    assert all(at <= start + k * P_NS + 1 for k, (_seq, at) in enumerate(record["slot_sources"]))


class PacedCapture:
    """按 30 fps 节拍出帧的假驱动（经真实的 ``CameraPool``/``CameraStream``/``CaptureReader``）。"""

    def __init__(self, index, *, width=64, height=48, **_kwargs) -> None:
        self.size = (width, height)
        self.released = False
        self.reads = 0
        self._next = time.perf_counter()

    def isOpened(self) -> bool:
        return not self.released

    def get(self, prop):
        return {3: self.size[0], 4: self.size[1], 5: 30.0}.get(prop, 0.0)

    def read(self):
        if self.released:
            return False, None
        self._next += 1 / 30
        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        self.reads += 1
        return True, np.full((self.size[1], self.size[0], 3), self.reads % 256, np.uint8)

    def release(self) -> None:
        self.released = True


def test_slow_processing_loop_does_not_slow_the_recording(tmp_path):
    """解耦验收：处理循环慢到约 5 fps（站位检测每帧 0.2 s）时，录像仍收到完整的源帧序列，两路帧数相同。"""
    from core.live import CameraInput, LiveSessionManager, camera_session
    from core.media import CameraPool, CameraService, CaptureSpec

    service = CameraService(pool=CameraPool(PacedCapture, join_timeout=0.2))
    recording = RecordingService(SegmentFinalizer(), plan_for=lambda _s: None, recorder=SegmentRecorder(Writers()))
    running = threading.Event()
    ended = threading.Event()

    def on_event(event):
        if event.kind == "running":
            running.set()
        elif event.kind == "ended":
            ended.set()

    def slow_probe(_pose, _roi):
        time.sleep(0.2)
        return True

    manager = LiveSessionManager(
        make_input=lambda config: CameraInput(service.open_stream(config.cameras)),
        on_event=on_event, sink=recording, probe=slow_probe,
    )
    manager.controls.arm_occupancy("test", True)
    import core.live.stages as stages_mod
    original = stages_mod.create_engine

    class Lite:
        def next_timestamp_ms(self, **_kw):
            return int(time.monotonic() * 1000)

        def infer(self, frame, *, timestamp_ms=None):
            return None, []

        def close(self):
            pass

    stages_mod.create_engine = lambda *_a, **_k: Lite()
    try:
        manager.start(camera_session({PRIMARY: CaptureSpec(0, 64, 48), SECONDARY: CaptureSpec(1, 64, 48)}))
        assert running.wait(10.0)
        assert recording.start(SegmentMeta(tmp_path))
        time.sleep(1.5)
        outcome = recording.finish()
        manager.stop()
        assert ended.wait(10.0)
    finally:
        stages_mod.create_engine = original
        service.close(2.0)
    finished = outcome.segment
    record = read_capture_record(finished.ref.capture_path)["views"]
    for view in ("front", "side"):
        stats = record[view]["stats"]
        assert finished.views[view].error is None
        assert abs(stats["slots"] - 45) <= 3
        assert stats["used_frames"] >= stats["slots"] - 3      # 几乎每个输出帧都是新的源帧，不是 5 fps 的重复
    assert finished.views["front"].frames == finished.views["side"].frames
