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
    IDLE,
    LiveControls,
    LiveSessionManager,
    PresenceSample,
    camera_session,
    video_session,
)
from core.live.preview import PreviewStore
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
        return {role: frame(3, self._shape) for role in self.roles}

    def close(self, *, keep_warm):
        self.calls.append("close")
        self.keep_warm = keep_warm


class FakeSink:
    def __init__(self) -> None:
        self.calls: list = []
        self.writes: list[dict] = []

    def begin_session(self, sizes, fps):
        self.calls.append(("begin", dict(sizes), fps))

    def update_sizes(self, sizes):
        pass

    def write(self, frames):
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
    assert len(h.sink.writes) == 4
    assert all(set(w) == {"front", "side"} for w in h.sink.writes)
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


def test_rotation_applies_to_recording_size_and_frames():
    h = Harness(FakeInput())
    h.controls.rotation[PRIMARY].set(90)
    h.manager.start(single(skeleton=False))
    h.wait_for(lambda: len(h.sink.writes) >= 2)
    h.manager.stop()
    h.outcome()
    assert h.sink.calls[0][1] == {"front": (4, 6)}
    assert h.sink.writes[0]["front"].shape[:2] == (6, 4)


# ---- §8.2 表 ----


def test_skeleton_off_builds_no_engine_and_previews_raw_frames():
    h = Harness(FakeInput((PRIMARY, SECONDARY)))
    h.manager.start(dual(skeleton=False))
    h.wait_for(lambda: len(h.sink.writes) >= 3)
    store = h.manager.preview
    h.manager.stop()
    h.outcome()
    assert FakeEngine.built == 0
    assert store.stage == "raw"
    # 双摄骨架关：录制原始帧，HUD 只在预览帧上
    written = h.sink.writes[0]["front"]
    assert int(written.max()) == 3


def test_single_camera_recording_keeps_hud_even_without_skeleton():
    """单摄录像一直带 HUD（旧单摄循环写的是叠加后的帧）；双摄骨架关时录原始帧。"""
    h = Harness(FakeInput(shape=(80, 120, 3)))
    h.manager.start(single(skeleton=False))
    h.wait_for(lambda: len(h.sink.writes) >= 2)
    h.manager.stop()
    h.outcome()
    assert int(h.sink.writes[0]["front"].max()) == 255
    assert dual(skeleton=False).record_overlay is False


def test_parallel_camera_drops_into_ordered_results(monkeypatch):
    monkeypatch.setattr(
        stages_mod, "default_pipeline_factory", lambda **_kwargs: (lambda: FakeEngine(None)),
    )
    h = Harness(FakeInput(frames=None))
    h.manager.start(single(skeleton=True, workers=2))
    h.wait_for(lambda: len(h.sink.writes) >= 5)
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
    h.wait_for(lambda: len(h.sink.writes) >= 7)
    h.manager.stop()
    h.outcome()
    samples = [e.payload for e in h.events if e.kind == "presence"]
    assert samples and all(isinstance(s, PresenceSample) and s.present for s in samples)
    assert calls[0] == (0.1, 0.1, 0.9, 0.9)
    assert len(samples) == (len(h.sink.writes) + 2) // 3
    assert FakeEngine.built == 1        # 独立的 Lite 引擎


def test_occupancy_not_run_for_skeleton_sessions():
    h = Harness(FakeInput((PRIMARY, SECONDARY)), probe=lambda *_: True)
    h.controls.arm_occupancy("exam", True)
    h.manager.start(dual(skeleton=True))
    h.wait_for(lambda: len(h.sink.writes) >= 4)
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
