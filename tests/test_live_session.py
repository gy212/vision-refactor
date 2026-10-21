# -*- coding: utf-8 -*-
"""实时会话（core/live）的生命周期与 §8.2 行为：用假输入、假引擎驱动 ``LiveSessionManager``/``SessionRunner``。"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import core.live.stages as stages_mod
from core.foundation.tasks import Cancelled
from core.live import (
    APP_CLOSE,
    IDLE,
    InputFrames,
    LiveControls,
    LiveSessionManager,
    PresenceSample,
    camera_session,
    video_session,
)
from core.live.preview import PreviewStore
from core.media import apply_rotation, rotated_size
from core.media import PRIMARY, SECONDARY, CameraReadError, CaptureSpec

WAIT = 5.0


class FakeEngine:
    built = 0
    gate: threading.Event | None = None

    def __init__(self, spec) -> None:
        type(self).built += 1
        self.spec = spec
        self.closed = False
        if type(self).gate is not None:
            type(self).gate.wait(WAIT)

    def next_timestamp_ms(self, *, is_file, fps_for_ts):
        return 0

    def infer(self, frame, *, timestamp_ms=None):
        return None, []

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch, tmp_path):
    FakeEngine.built = 0
    FakeEngine.gate = None
    monkeypatch.setattr(stages_mod, "create_engine", lambda spec, models_dir=None: FakeEngine(spec))
    monkeypatch.setattr(stages_mod, "models_dir", lambda: tmp_path)
    monkeypatch.setattr(stages_mod, "templates_dir", lambda: tmp_path)
    yield


def frame(value: int = 0, shape=(4, 6, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


class FakeInput:
    """按 ``roles`` 产出帧；``fail_after`` 帧后读帧失败，``frames`` 读完后返回 None（视频）。"""

    def __init__(self, roles=(PRIMARY,), *, is_file=False, frames=None, fail_after=None,
                 ready_gate: threading.Event | None = None, shape=(4, 6, 3)) -> None:
        self.roles = tuple(roles)
        self._shape = shape
        self.is_file = is_file
        self.calls: list[str] = []
        self.keep_warm: bool | None = None
        self._frames = frames
        self._fail_after = fail_after
        self._ready_gate = ready_gate
        self._read = 0
        self._seq = 0
        self.feed = None if is_file else object()   # 交给录制层的帧来源（录像线程从中订阅）

    @property
    def reads(self) -> int:
        return self._read

    def wait_ready(self, cancel):
        self.calls.append("wait_ready")
        if self._ready_gate is not None:
            while not self._ready_gate.wait(0.01):
                cancel.raise_if_cancelled()
        cancel.raise_if_cancelled()
        return {} if self.is_file else {role: frame(1) for role in self.roles}

    def first_cursor(self):
        return {} if self.is_file else {role: 0 for role in self.roles}

    def snapshot(self, after):
        self._seq += 1
        return {role: frame(2) for role in self.roles}, {role: self._seq for role in self.roles}

    def start(self, cancel):
        self.calls.append("start")

    def actual_sizes(self):
        return {role: (6, 4) for role in self.roles}

    def fps(self):
        return 25.0

    def total_frames(self):
        return self._frames or 0

    def read(self, cancel):
        cancel.raise_if_cancelled()
        if self._fail_after is not None and self._read >= self._fail_after:
            raise CameraReadError(self.roles[-1], numbered=len(self.roles) > 1)
        if self._frames is not None and self._read >= self._frames:
            return None
        self._read += 1
        time.sleep(0.002)
        return InputFrames({role: frame(3, self._shape) for role in self.roles}, None if self.is_file else time.perf_counter_ns())

    def capture_feed(self):
        return self.feed

    def close(self, *, keep_warm):
        self.calls.append("close")
        self.keep_warm = keep_warm


class FakeSink:
    def __init__(self) -> None:
        self.calls: list = []
        self.writes: list[dict] = []
        self.fail_write = False
        self.fail_begin = False
        self.feed = None
        self.rotation_of = None

    def begin_session(self, sizes, fps, *, feed=None, rotation_of=None):
        if self.fail_begin:
            raise RuntimeError("begin failed")
        self.calls.append(("begin", dict(sizes), fps))
        self.feed, self.rotation_of = feed, rotation_of

    def write(self, frames):
        if self.fail_write:
            raise RuntimeError("write failed")
        self.writes.append(dict(frames))

    def close_session(self):
        self.calls.append(("close",))


class Harness:
    def __init__(self, source: FakeInput, *, probe=None, preview=True) -> None:
        self.source = source
        self.sink = FakeSink()
        self.events: list = []
        self.states: list[str] = []
        self.ended = threading.Event()
        self.controls = LiveControls()
        self.manager = LiveSessionManager(
            make_input=lambda _config: source,
            on_event=self._on_event,
            controls=self.controls,
            sink=self.sink,
            probe=probe,
            preview_target=(lambda _role: (0, 0)) if preview else None,
        )

    def _on_event(self, event) -> None:
        self.events.append(event)
        self.states.append(self.manager.state)
        if event.kind == "ended":
            self.ended.set()

    def kinds(self) -> list[str]:
        return [event.kind for event in self.events if event.kind not in ("status", "progress")]

    def outcome(self):
        assert self.ended.wait(WAIT), "session did not end"
        return next(e.payload for e in self.events if e.kind == "ended")

    def statuses(self) -> list[str]:
        return [event.payload for event in self.events if event.kind == "status"]

    def wait_for(self, predicate, timeout=WAIT) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError("condition not reached")


def single(skeleton=True, **kwargs):
    return camera_session({PRIMARY: CaptureSpec(0)}, skeleton=skeleton, **kwargs)


def dual(skeleton=False, **kwargs):
    return camera_session({PRIMARY: CaptureSpec(0), SECONDARY: CaptureSpec(1)}, skeleton=skeleton, **kwargs)


# ---- 生命周期 ----


def test_stop_immediately_after_start_never_claims_or_records():
    gate = threading.Event()
    h = Harness(FakeInput(ready_gate=gate))
    token = h.manager.start(single())
    assert token == 1 and h.manager.state == "starting"
    assert h.manager.stop("user_stop")
    assert h.manager.state == "stopping"
    outcome = h.outcome()
    assert outcome.outcome == "stopped"
    assert "start" not in h.source.calls and h.source.calls[-1] == "close"
    assert h.sink.calls == []
    assert h.kinds() == ["ended"]
    assert h.manager.state == IDLE


def test_start_while_stopping_is_refused_until_ended():
    gate = threading.Event()
    h = Harness(FakeInput(ready_gate=gate))
    h.manager.start(single())
    h.manager.stop()
    assert h.manager.start(single()) is None
    h.outcome()
    assert h.manager.state == IDLE
    second = FakeInput()
    h.manager._make_input = lambda _config: second
    h.ended.clear()
    assert h.manager.start(single()) == 2
    h.manager.stop()
    h.outcome()


def test_stop_during_model_load_releases_without_claim_and_closes_engines():
    FakeEngine.gate = threading.Event()
    h = Harness(FakeInput())
    h.manager.start(single(skeleton=True))
    h.wait_for(lambda: FakeEngine.built == 1)
    h.manager.stop()
    FakeEngine.gate.set()
    outcome = h.outcome()
    assert outcome.outcome == "stopped"
    assert "start" not in h.source.calls
    assert h.source.keep_warm is True       # 正常停止：交回预热（V-1）
    assert h.sink.calls == []


def test_runtime_read_failure_fails_session_and_releases_device():
    h = Harness(FakeInput(fail_after=3))
    h.manager.start(single(skeleton=False))
    outcome = h.outcome()
    assert outcome.failed and outcome.error_type == "CameraReadError"
    assert outcome.status == "运行失败：摄像头读帧失败，请检查连接后重新开始"
    assert h.source.keep_warm is False
    assert h.sink.calls[-1] == ("close",)
    assert h.kinds() == ["running", "ended"]


def test_dual_failure_uses_numbered_dual_message_and_both_views_converge():
    h = Harness(FakeInput((PRIMARY, SECONDARY), fail_after=4))
    h.manager.start(dual())
    outcome = h.outcome()
    assert outcome.status == "双摄像头运行失败：第2路摄像头读帧失败，请检查连接后重新开始"
    assert outcome.frames == 4 and h.sink.writes == []
    assert h.sink.calls[0] == ("begin", {"front": (6, 4), "side": (6, 4)}, 30.0)


def test_state_order_is_deterministic():
    h = Harness(FakeInput())
    h.manager.start(single(skeleton=False))
    h.wait_for(lambda: "running" in h.kinds())
    assert h.manager.state == "running"
    h.manager.stop()
    h.outcome()
    assert h.kinds() == ["running", "ended"]
    assert h.states[-1] == IDLE


def test_events_from_an_old_session_are_dropped():
    h = Harness(FakeInput())
    h.manager.start(single(skeleton=False))
    h.wait_for(lambda: "running" in h.kinds())
    h.manager.stop()
    h.outcome()
    count = len(h.events)
    from core.live.runner import SessionEvent

    h.manager._dispatch(SessionEvent("status", 1, "late"))      # 已结束的会话
    h.manager._dispatch(SessionEvent("status", 99, "foreign"))   # 不是当前 token
    assert len(h.events) == count


def test_rotation_applies_to_recording_size_and_is_read_live_by_recording():
    h = Harness(FakeInput())
    h.controls.rotation[PRIMARY].set(90)
    h.manager.start(single(skeleton=False))
    h.wait_for(lambda: h.source.reads >= 2)
    h.manager.stop()
    h.outcome()
    assert h.sink.calls[0][1] == {"front": (4, 6)}
    assert h.sink.rotation_of(PRIMARY) == 90     # 录像线程在开录时读取（D-29）


# ---- §8.2 表 ----


def test_skeleton_off_builds_no_engine_and_previews_raw_frames():
    h = Harness(FakeInput((PRIMARY, SECONDARY)))
    h.manager.start(dual(skeleton=False))
    h.wait_for(lambda: h.source.reads >= 3)
    store = h.manager.preview
    h.manager.stop()
    h.outcome()
    assert FakeEngine.built == 0
    assert store.stage == "raw"


@pytest.mark.parametrize("skeleton", [False, True])
def test_camera_sessions_hand_the_feed_to_recording_and_never_write_from_the_loop(skeleton, monkeypatch):
    """D-18/D-26：摄像头会话的录像由录制层的录像线程从摄像头取帧（原始画面），处理循环不写录像；
    录制会话先于输入关闭，录像线程能用仍在出帧的摄像头写完结束边界。"""
    monkeypatch.setattr(stages_mod, "render_pose", lambda img, *_a, **_k: (np.full_like(img, 9), []))
    h = Harness(FakeInput((PRIMARY, SECONDARY), shape=(80, 120, 3)))
    order: list[str] = []
    h.sink.close_session = lambda: order.append("sink")
    close = h.source.close
    h.source.close = lambda *, keep_warm: (order.append("source"), close(keep_warm=keep_warm))
    h.manager.start(dual(skeleton=skeleton))
    h.wait_for(lambda: h.source.reads >= 3)
    h.manager.stop()
    h.outcome()
    assert h.sink.feed is h.source.feed and h.sink.writes == []
    assert order == ["sink", "source"]


def test_parallel_camera_drops_into_ordered_results(monkeypatch):
    monkeypatch.setattr(
        stages_mod, "default_pipeline_factory", lambda **_kwargs: (lambda: FakeEngine(None)),
    )
    h = Harness(FakeInput(frames=None))
    h.manager.start(single(skeleton=True, workers=2))
    h.wait_for(lambda: h.source.reads >= 5)
    h.manager.stop()
    outcome = h.outcome()
    assert outcome.outcome == "stopped"
    assert FakeEngine.built == 2
    assert "运行中…（实时多线程：2，预览平滑开启）" in h.statuses()


def test_parallel_video_keeps_every_frame_in_order(monkeypatch):
    monkeypatch.setattr(
        stages_mod, "default_pipeline_factory", lambda **_kwargs: (lambda: FakeEngine(None)),
    )
    h = Harness(FakeInput(is_file=True, frames=12))
    h.manager.start(video_session("clip.mp4", workers=3))
    outcome = h.outcome()
    assert outcome.outcome == "stopped" and outcome.frames == 12
    assert len(h.sink.writes) == 12


def test_stopping_parallel_video_with_full_queue_leaves_no_reader(monkeypatch):
    """PR #17 评审 1：输入队列满、读帧线程阻塞在 submit 时停止，会话要很快结束且不留读帧线程。"""
    class Slow(FakeEngine):
        def infer(self, frame, *, timestamp_ms=None):
            time.sleep(0.3)   # 停止时 worker 仍在推理，结束后不再取帧，读帧线程卡在满队列上
            return None, []

    monkeypatch.setattr(stages_mod, "default_pipeline_factory", lambda **_kwargs: (lambda: Slow(None)))
    h = Harness(FakeInput(is_file=True, frames=200))
    h.manager.start(video_session("clip.mp4", workers=2))
    h.wait_for(lambda: len(h.sink.writes) >= 1)
    started = time.monotonic()
    h.manager.stop()
    outcome = h.outcome()
    assert outcome.outcome == "stopped"
    assert time.monotonic() - started < 1.0
    assert not [t.name for t in threading.enumerate() if t.name.endswith("/runner/reader")]


@pytest.mark.parametrize("hud, expect_text", [("cli", True), ("tk", False)])
def test_parallel_video_writer_receives_hud_only_for_cli(monkeypatch, hud, expect_text):
    """PR #17 评审 2：CLI 并行导出先画标签/FPS 再写；Tk 并行视频沿用写完再画。"""
    monkeypatch.setattr(
        stages_mod, "default_pipeline_factory", lambda **_kwargs: (lambda: FakeEngine(None)),
    )
    h = Harness(FakeInput(is_file=True, frames=6, shape=(120, 320, 3)))
    copies: list[np.ndarray] = []
    h.sink.write = lambda frames: copies.append(next(iter(frames.values())).copy())
    h.manager.start(video_session("clip.mp4", workers=2, hud=hud))
    assert h.outcome().frames == 6
    assert len(copies) == 6
    assert all(bool((f != 3).any()) == expect_text for f in copies)


def test_video_file_always_infers_and_ends_at_eof():
    h = Harness(FakeInput(is_file=True, frames=7))
    h.manager.start(video_session("clip.mp4"))
    outcome = h.outcome()
    assert outcome.outcome == "stopped" and outcome.frames == 7
    assert FakeEngine.built == 1
    progress = [e.payload for e in h.events if e.kind == "progress"]
    assert (5, 7) in progress and progress[-1] == (7, 7)
    assert h.sink.calls[0] == ("begin", {"front": (6, 4)}, 25.0)


def test_config_table_matches_legacy_loops():
    seq = single(skeleton=True)
    assert (seq.inference, seq.smoothing, seq.online_match, seq.parallel, seq.loading_raw_preview) == (
        True, True, True, False, True)
    par = single(skeleton=True, workers=4)
    assert par.parallel and par.smoothing and not par.loading_raw_preview and not par.hud_hint
    off = single(skeleton=False, workers=4)
    assert not off.inference and not off.parallel
    d7 = dual(skeleton=True)
    assert not d7.smoothing and not d7.online_match and not d7.occupancy   # D-7
    assert dual(skeleton=False).occupancy
    video = video_session("x.mp4", workers=3)
    assert video.inference and video.parallel and not video.record_hud and not video.smoothing
    assert video_session("x.mp4", workers=3, hud="cli").record_hud   # CLI 并行导出也带 HUD


def test_loading_preview_shows_raw_then_annotated_without_regressing():
    """D-8：单摄模型加载期间发布裸帧，模型就绪后只进不退到标注帧。"""
    FakeEngine.gate = threading.Event()
    h = Harness(FakeInput())
    h.manager.start(single(skeleton=True))
    h.wait_for(lambda: FakeEngine.built == 1)
    store = h.manager.preview
    shown = store.take_newer(0)
    assert shown is not None and shown.stage == "raw"
    store.mark_rendered(shown)
    assert "模型加载中…（裸帧预览）" in h.statuses()
    FakeEngine.gate.set()
    h.wait_for(lambda: store.stage == "annotated")
    assert store.publish("raw", {PRIMARY: frame()}, {}) is False
    h.wait_for(lambda: (f := store.take_newer(shown.seq)) is not None and f.stage == "annotated")
    h.manager.stop()
    h.outcome()


def test_occupancy_samples_every_stride_frames_when_armed():
    calls = []

    def probe(pose, roi):
        calls.append(roi)
        return True

    h = Harness(FakeInput((PRIMARY, SECONDARY)), probe=probe)
    h.controls.arm_occupancy("exam", True)
    h.controls.set_roi((0.1, 0.1, 0.9, 0.9))
    h.manager.start(dual(skeleton=False))
    h.wait_for(lambda: h.source.reads >= 7)
    h.manager.stop()
    outcome = h.outcome()
    samples = [e.payload for e in h.events if e.kind == "presence"]
    assert samples and all(isinstance(s, PresenceSample) and s.present for s in samples)
    assert calls[0] == (0.1, 0.1, 0.9, 0.9)
    assert len(samples) == (outcome.frames + 2) // 3
    assert FakeEngine.built == 1        # 独立的 Lite 引擎


def test_occupancy_not_run_for_skeleton_sessions():
    h = Harness(FakeInput((PRIMARY, SECONDARY)), probe=lambda *_: True)
    h.controls.arm_occupancy("exam", True)
    h.manager.start(dual(skeleton=True))
    h.wait_for(lambda: h.source.reads >= 4)
    h.manager.stop()
    h.outcome()
    assert not [e for e in h.events if e.kind == "presence"]


def test_metrics_emitted_once_with_legacy_fields(monkeypatch):
    lines = []
    import core.live.metrics as metrics_mod

    monkeypatch.setattr(metrics_mod.diag, "event", lambda name, **fields: lines.append((name, fields)))
    h = Harness(FakeInput((PRIMARY, SECONDARY)))
    h.manager.start(dual(skeleton=False), start_click=time.monotonic())
    h.wait_for(lambda: "running" in h.kinds())
    store = h.manager.preview
    h.wait_for(lambda: store.take_newer(0) is not None or bool(lines))
    h.manager.stop()
    h.outcome()
    assert len(lines) == 1
    name, fields = lines[0]
    assert name == "session.metrics"
    assert fields["primary_index"] == 0 and fields["secondary_index"] == 1
    for key in ("pair_ready", "first_pair_enqueued", "first_pair_rendered", "pipeline_ready",
                "first_annotated_enqueued", "first_annotated_rendered"):
        assert f"start_to_{key}_ms" in fields


def test_preview_store_is_single_slot_and_monotonic():
    store = PreviewStore(1, lambda _r: (0, 0))
    store.publish("raw", {PRIMARY: frame(1)}, {})
    store.publish("raw", {PRIMARY: frame(2)}, {PRIMARY: "-"})
    latest = store.take_newer(0)
    assert latest.seq == 2 and store.take_newer(latest.seq) is None
    store.advance("annotated")
    assert not store.publish("raw", {PRIMARY: frame()}, {})
    store.close()
    assert not store.publish("annotated", {PRIMARY: frame()}, {})


def test_cancelled_wait_ready_reports_stopped():
    class Refusing(FakeInput):
        def wait_ready(self, cancel):
            raise Cancelled("user_stop")

    h = Harness(Refusing())
    h.manager.start(single())
    assert h.outcome().outcome == "stopped"


# ---- 双摄会话（取代旧 _worker_loop_dual_camera 的白盒测试） ----


def test_dual_wait_failure_reports_warmup_error_once_and_never_claims():
    class Unavailable(FakeInput):
        def wait_ready(self, cancel):
            raise RuntimeError("camera 1 unavailable")

    h = Harness(Unavailable((PRIMARY, SECONDARY)))
    h.manager.start(dual())
    outcome = h.outcome()
    assert outcome.status == "双摄像头预热失败：camera 1 unavailable"
    assert [e.kind for e in h.events].count("ended") == 1
    assert "start" not in h.source.calls and h.source.keep_warm is False
    assert h.sink.calls == []


def test_dual_skeleton_builds_two_engines_and_closes_both_on_infer_failure(monkeypatch):
    class Failing(FakeEngine):
        def infer(self, frame, *, timestamp_ms=None):
            raise RuntimeError("annotate failed")

    engines = []

    def factory(spec, models_dir=None):
        engines.append(Failing(spec))
        return engines[-1]

    monkeypatch.setattr(stages_mod, "create_engine", factory)
    h = Harness(FakeInput((PRIMARY, SECONDARY)))
    h.manager.start(dual(skeleton=True))
    outcome = h.outcome()
    assert outcome.status == "双摄像头运行失败：annotate failed"
    assert len(engines) == 2 and all(e.closed for e in engines)
    assert all(e.spec.running_mode == "video" for e in engines)
    assert h.sink.calls[-1] == ("close",) and h.source.keep_warm is False


def test_second_engine_failure_closes_first_and_never_claims_or_records(monkeypatch):
    built = []

    def factory(spec, models_dir=None):
        if built:
            raise RuntimeError("second pipeline failed")
        built.append(FakeEngine(spec))
        return built[0]

    monkeypatch.setattr(stages_mod, "create_engine", factory)
    h = Harness(FakeInput((PRIMARY, SECONDARY)))
    h.manager.start(dual(skeleton=True))
    outcome = h.outcome()
    assert outcome.status == "初始化失败：second pipeline failed"
    assert built[0].closed
    assert "start" not in h.source.calls and h.sink.calls == []
    assert h.source.keep_warm is False


def test_recording_begin_failure_fails_session_and_releases_without_engines():
    h = Harness(FakeInput((PRIMARY, SECONDARY)))
    h.sink.fail_begin = True
    h.manager.start(dual(skeleton=False))
    outcome = h.outcome()
    assert outcome.failed and outcome.status == "双摄像头接管失败：begin failed"
    assert FakeEngine.built == 0
    assert ("close",) not in h.sink.calls
    assert h.source.calls[-1] == "close" and h.source.keep_warm is False


def test_dual_loading_keeps_raw_pairs_flowing_without_writes_then_annotates():
    FakeEngine.gate = threading.Event()
    h = Harness(FakeInput((PRIMARY, SECONDARY)))
    h.manager.start(dual(skeleton=True))
    h.wait_for(lambda: FakeEngine.built >= 1)
    store = h.manager.preview
    shown: list = []
    for _ in range(2):
        seq = shown[-1].seq if shown else 0
        h.wait_for(lambda: (f := store.take_newer(seq)) is not None and not shown.append(f))
        assert shown[-1].stage == "raw" and set(shown[-1].images) == {PRIMARY, SECONDARY}
        store.mark_rendered(shown[-1])
    assert "模型加载中…（双摄裸帧预览）" in h.statuses()
    assert h.sink.calls == [] and "running" not in h.kinds()
    FakeEngine.gate.set()
    h.wait_for(lambda: store.stage == "annotated")
    h.manager.stop()
    h.outcome()
    assert h.kinds()[0] == "running"


def test_dual_rotation_is_per_role_for_sizes_and_read_live_by_recording():
    h = Harness(FakeInput((PRIMARY, SECONDARY)))
    h.controls.rotation[SECONDARY].set(90)
    h.manager.start(dual(skeleton=False))
    h.wait_for(lambda: h.source.reads >= 2)
    h.controls.rotation[PRIMARY].set(270)          # 运行中改旋转：录制层下一段开录时读到新值（D-29）
    h.manager.stop()
    h.outcome()
    assert h.sink.calls[0] == ("begin", {"front": (6, 4), "side": (4, 6)}, 30.0)
    assert (h.sink.rotation_of(PRIMARY), h.sink.rotation_of(SECONDARY)) == (270, 90)


def test_video_file_ignores_camera_rotation():
    h = Harness(FakeInput(is_file=True, frames=3))
    h.controls.rotation[PRIMARY].set(90)
    h.manager.start(video_session("clip.mp4", workers=1))
    h.outcome()
    assert h.sink.calls[0] == ("begin", {"front": (6, 4)}, 25.0)
    assert h.sink.writes[0]["front"].shape[:2] == (4, 6) and h.sink.feed is None


def test_parallel_camera_applies_rotation(monkeypatch):
    monkeypatch.setattr(
        stages_mod, "default_pipeline_factory", lambda **_kwargs: (lambda: FakeEngine(None)),
    )
    h = Harness(FakeInput())
    h.controls.rotation[PRIMARY].set(90)
    h.manager.start(single(skeleton=True, workers=2))
    h.wait_for(lambda: h.source.reads >= 2)
    h.manager.stop()
    h.outcome()
    assert h.sink.calls[0][1] == {"front": (4, 6)}


def test_claim_failure_emits_failed_metrics_once(monkeypatch):
    lines = []
    import core.live.metrics as metrics_mod

    monkeypatch.setattr(metrics_mod.diag, "event", lambda name, **fields: lines.append((name, fields)))

    class ClaimFails(FakeInput):
        def start(self, cancel):
            raise RuntimeError("claim failed")

    h = Harness(ClaimFails((PRIMARY, SECONDARY)))
    h.manager.start(dual(skeleton=False), start_click=time.monotonic())
    outcome = h.outcome()
    assert outcome.status == "双摄像头接管失败：claim failed"
    assert len(lines) == 1
    assert lines[0][1]["outcome"] == "failed" and lines[0][1]["error_type"] == "RuntimeError"


def test_release_failure_is_reported_not_swallowed():
    class StuckRelease(FakeInput):
        def close(self, *, keep_warm):
            super().close(keep_warm=keep_warm)
            raise RuntimeError("摄像头 0 释放失败，已隔离")

    h = Harness(StuckRelease())
    h.manager.start(single(skeleton=False))
    h.wait_for(lambda: "running" in h.kinds())
    h.manager.stop()
    outcome = h.outcome()
    assert outcome.failed and outcome.error_type == "CameraReleaseError"
    assert outcome.status == "摄像头释放失败：摄像头 0 释放失败，已隔离"
    assert h.statuses()[-1] == outcome.status


def test_open_failure_releases_without_keeping_warm():
    class Timeout(FakeInput):
        def wait_ready(self, cancel):
            raise TimeoutError("摄像头响应超时")

    h = Harness(Timeout())
    h.manager.start(single(skeleton=False))
    outcome = h.outcome()
    assert outcome.status == "打开输入源失败：摄像头响应超时"
    assert h.source.keep_warm is False and "start" not in h.source.calls


@pytest.mark.parametrize(("reason", "keep_warm"), [("user_stop", True), (APP_CLOSE, False)])
def test_normal_stop_keeps_device_warm_but_app_close_releases(reason, keep_warm):
    h = Harness(FakeInput())
    h.manager.start(single(skeleton=False))
    h.wait_for(lambda: "running" in h.kinds())
    h.manager.stop(reason)
    assert h.outcome().outcome == "stopped"
    assert h.source.keep_warm is keep_warm


# --- 旋转与尺寸适配（原 test_app_ui_dual_camera.py，与 Tk 无关）---------------------------

@pytest.mark.parametrize(
    ("degrees", "turns"),
    [(90, -1), (180, 2), (270, 1)],
)
def test_apply_rotation_matches_clockwise_degree_contract(degrees, turns):
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

    rotated = apply_rotation(frame, degrees)

    np.testing.assert_array_equal(rotated, np.rot90(frame, k=turns))


@pytest.mark.parametrize("degrees", [0, 45])
def test_apply_rotation_returns_original_frame_when_no_rotation_applies(degrees):
    frame = np.zeros((2, 3, 3), dtype=np.uint8)

    assert apply_rotation(frame, degrees) is frame


@pytest.mark.parametrize(
    ("degrees", "expected"),
    [(0, (1280, 720)), (90, (720, 1280)), (180, (1280, 720)), (270, (720, 1280))],
)
def test_rotated_size_matches_rotated_frame_dimensions(degrees, expected):
    assert rotated_size(1280, 720, degrees) == expected
