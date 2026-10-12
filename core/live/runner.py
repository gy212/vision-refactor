# -*- coding: utf-8 -*-
"""``SessionRunner``：实时会话唯一的逐帧循环（设计文档 §8.2），Tk 与 CLI 共用。

一次会话在 runner 线程里依次：等输入就绪 → 构造推理 stage（期间裸帧泵发布预览，D-8）→ 接管输入 →
登记录制尺寸 → 逐帧"读 → 转正 → 推理 → 后续 stages → 写录像 → 发布预览"→ 收尾。收尾总是关闭 stages、
关闭输入（正常结束时把摄像头交回预热，V-1）、关闭录制会话，最后发出 ``ended``。
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

import numpy as np

from core.foundation.tasks import Cancelled, CancelToken, TaskGroup
from core.live.config import LiveControls, SessionConfig
from core.live.inputs import FrameInput
from core.live.metrics import StartupMetrics
from core.live.preview import PreviewStore
from core.live.stages import (
    FramePacket,
    OverlayStage,
    PoseStage,
    StageContext,
    TransformStage,
    rotated_size,
)
from core.media import PRIMARY, SECONDARY, VideoOpenError, view_for_role
from core.pose import actions_text

RAW_PUMP_INTERVAL_S = 0.05
FIRST_RENDER_WAIT_S = 0.25
PARALLEL_GET_TIMEOUT_S = 0.2

_SINGLE_ERRORS = {
    "warmup": "打开输入源失败",
    "pipeline": "初始化失败",
    "claim": "打开输入源失败",
    "runtime": "运行失败",
}
_DUAL_ERRORS = {
    "warmup": "双摄像头预热失败",
    "pipeline": "初始化失败",
    "claim": "双摄像头接管失败",
    "runtime": "双摄像头运行失败",
}


class FrameSink(Protocol):
    """录制端口（由 ``recording.RecordingService`` 适配实现）；帧按视角名（front/side）组织。"""

    def begin_session(self, sizes: Mapping[str, tuple[int, int]], fps: float) -> None: ...
    def update_sizes(self, sizes: Mapping[str, tuple[int, int]]) -> None: ...
    def write(self, frames: Mapping[str, np.ndarray]) -> None: ...
    def close_session(self) -> None: ...


@dataclass(frozen=True)
class SessionEvent:
    """会话事件。``kind``：status、progress、running、match、presence、ended。"""

    kind: str
    token: int
    payload: object = None


@dataclass(frozen=True)
class SessionOutcome:
    outcome: str                     # stopped（含视频读完）/ failed
    status: str | None
    error_type: str | None = None
    frames: int = 0
    error: BaseException | None = field(default=None, compare=False)   # 失败的原始异常（CLI 据此重新抛出）

    @property
    def failed(self) -> bool:
        return self.outcome == "failed"


@dataclass(frozen=True)
class RunningInfo:
    """会话进入运行：各路实际宽高（未旋转）与录制登记的尺寸（已旋转，按视角名）。"""

    actual_sizes: Mapping[str, tuple[int, int]]
    record_sizes: Mapping[str, tuple[int, int]]


@dataclass
class _Loop:
    frames: int = 0
    t0: float = field(default_factory=time.monotonic)


class SessionRunner:
    """一个会话一个实例；``run`` 在会话线程里执行，返回终态并发出 ``ended`` 事件。"""

    def __init__(
        self,
        config: SessionConfig,
        source: FrameInput,
        pose: PoseStage | None,
        stages: list,
        sink: FrameSink | None,
        preview: PreviewStore | None,
        events: Callable[[SessionEvent], None],
        token: int,
        cancel: CancelToken,
        *,
        controls: LiveControls | None = None,
        metrics: StartupMetrics | None = None,
        display: Callable[[FramePacket], bool] | None = None,
    ) -> None:
        self.config = config
        self._source = source
        self._pose = pose
        self._stages = stages
        self._sink = sink
        self._preview = preview
        self._events = events
        self._token = token
        self._cancel = cancel
        self._controls = controls or LiveControls()
        self._metrics = metrics
        self._display = display
        self._transform = TransformStage()
        self._overlay = OverlayStage()
        self._opened: list = []
        self._tasks = TaskGroup(f"live-{token}")
        self._phase = "warmup"
        self._sink_open = False
        self._loop = _Loop()
        self.stats: dict[str, float] = {"captured": 0, "dropped": 0, "capture_started": 0.0, "capture_ended": 0.0}
        self.ctx = StageContext(config, self._controls, self._emit)

    # ---- 事件 ----

    def _emit(self, kind: str, payload: object = None) -> None:
        self._events(SessionEvent(kind, self._token, payload))

    def _status(self, text: str) -> None:
        self._emit("status", text)

    # ---- 主流程 ----

    def run(self) -> SessionOutcome:
        outcome = SessionOutcome("failed", None)
        try:
            self._start()
            self._phase = "runtime"
            self._run_loop()
            outcome = SessionOutcome("stopped", "已停止", frames=self._loop.frames)
            self._status("已停止")
            self._emit("progress", (self._loop.frames, self._source.total_frames()))
        except Cancelled:
            outcome = SessionOutcome("stopped", "已停止", frames=self._loop.frames)
            self._status("已停止")
        except Exception as exc:
            if self._cancel.cancelled:
                outcome = SessionOutcome("stopped", "已停止", frames=self._loop.frames)
            else:
                traceback.print_exc()
                outcome = SessionOutcome(
                    "failed", self._failure_text(exc), type(exc).__name__, self._loop.frames, error=exc,
                )
            self._status(outcome.status or "")
        finally:
            outcome = self._shutdown(outcome)
            self._emit("ended", outcome)
        return outcome

    def _failure_text(self, exc: Exception) -> str:
        if isinstance(exc, VideoOpenError):
            return f"无法打开输入源：{exc.path}"
        prefixes = _DUAL_ERRORS if self.config.dual else _SINGLE_ERRORS
        return f"{prefixes[self._phase]}：{exc}"

    def _start(self) -> None:
        first = self._source.wait_ready(self._cancel)
        self._mark("pair_ready")
        self._cancel.raise_if_cancelled()
        self.ctx.fps = self._source.fps() if self.config.is_file else self.config.nominal_fps
        if first and (self.config.loading_raw_preview or self._pose is None):
            self._publish("raw", self._rotate(first), {})
        if self._pose is None:
            self._mark("pipeline_ready")
        self._phase = "pipeline"
        self._open_stages()
        self._cancel.raise_if_cancelled()
        self._phase = "claim"
        self._source.start(self._cancel)
        self._cancel.raise_if_cancelled()
        if self._pose is not None and self._preview is not None:
            self._preview.advance("annotated")
        actual = self._source.actual_sizes()
        record_sizes = {
            view_for_role(role): (size if self.config.is_file else rotated_size(*size, self._controls.rotation_of(role)))
            for role, size in actual.items()
        }
        if self._sink is not None:
            self._sink.begin_session(record_sizes, self.ctx.fps)
            self._sink_open = True
        if self._metrics is not None:
            self._metrics.set_outcome("running")
        self._emit("running", RunningInfo(actual, record_sizes))
        self._status(self._running_text(actual))
        self._emit("progress", (0, self._source.total_frames()))

    def _running_text(self, actual: Mapping[str, tuple[int, int]]) -> str:
        cfg = self.config
        if cfg.dual:
            (w, h), (w2, h2) = actual[PRIMARY], actual[SECONDARY]
            return f"运行中…（双摄像头 {w}×{h} / {w2}×{h2}）"
        if cfg.parallel and cfg.is_file:
            return f"运行中…（离线多线程：{cfg.workers}）"
        if cfg.parallel:
            return f"运行中…（实时多线程：{cfg.workers}，预览平滑开启）"
        return "运行中…" if cfg.inference else "运行中…（裸帧预览）"

    def _open_stages(self) -> None:
        if self._pose is not None:
            pump = self._start_raw_pump()
            try:
                self._opened.append(self._pose)
                self._pose.open(self.ctx)
                self._mark("pipeline_ready")
                if pump is not None:
                    self._wait_first_render()
                    if self._preview is not None:
                        self._preview.advance("annotated")
            finally:
                if pump is not None:
                    pump.token.cancel("loaded")
                    pump.join(1.0)
            if pump is not None and pump.error is not None:
                raise RuntimeError(str(pump.error)) from pump.error
        for stage in self._stages:
            self._opened.append(stage)
            stage.open(self.ctx)

    def _start_raw_pump(self):
        if not self.config.loading_raw_preview or self._preview is None:
            return None
        self._status("模型加载中…（双摄裸帧预览）" if self.config.dual else "模型加载中…（裸帧预览）")
        cursor = self._source.first_cursor()
        if not cursor:
            return None

        def pump(token: CancelToken) -> None:
            while not token.cancelled and not self._cancel.cancelled:
                latest = self._source.snapshot(cursor)
                if latest is not None:
                    frames, seen = latest
                    cursor.update(seen)
                    self._publish("raw", self._rotate(frames), {})
                token.wait(RAW_PUMP_INTERVAL_S)

        return self._tasks.spawn("raw-preview", pump, owner="runner", close_budget_s=1.0)

    def _wait_first_render(self) -> None:
        deadline = time.monotonic() + FIRST_RENDER_WAIT_S
        while not self._cancel.cancelled and time.monotonic() < deadline:
            if self._preview.wait_rendered(min(0.02, max(0.0, deadline - time.monotonic()))):
                return

    # ---- 逐帧 ----

    def _run_loop(self) -> None:
        self._loop = _Loop()
        if self._pose is not None and self._pose.parallel is not None:
            self._run_parallel()
            return
        frames_iter = self._latest_frames() if self.config.latest_frame else self._read_frames(self._cancel)
        for index, frames, captured_at in frames_iter:
            packet = self._transform.process(FramePacket(index, frames, captured_at=captured_at), self.ctx)
            self._update_sizes(packet.frames)
            if self._pose is not None:
                self._pose.process(packet, self.ctx)
            if not self._finish_frame(packet):
                return

    def _read_frames(self, token: CancelToken):
        """逐帧读取输入，产出 ``(序号, 各路帧, 采集时刻)``；读完、取消或到达 ``limit_frames`` 时结束。"""
        stats = self.stats
        stats["capture_started"] = time.monotonic()
        index = 0
        limit = self.config.limit_frames
        try:
            while not token.cancelled and not self._cancel.cancelled:
                if limit is not None and index >= limit:
                    return
                frames = self._source.read(self._cancel)
                if frames is None:
                    return
                stats["captured"] += 1
                yield index, frames, time.monotonic()
                index += 1
        finally:
            stats["capture_ended"] = time.monotonic()

    def _latest_frames(self):
        """采集在后台任务里进行，只保留最新一帧；推理取到的总是最新帧，被覆盖的计入 ``dropped``。"""
        slot: list = []
        done = threading.Event()
        lock = threading.Lock()

        def capture(token: CancelToken) -> None:
            try:
                for item in self._read_frames(token):
                    with lock:
                        if slot:
                            self.stats["dropped"] += 1
                            slot.clear()
                        slot.append(item)
            finally:
                done.set()

        reader = self._tasks.spawn("latest-reader", capture, owner="runner", close_budget_s=2.0)
        try:
            while not self._cancel.cancelled:
                with lock:
                    item = slot.pop() if slot else None
                if item is None:
                    if done.is_set():
                        with lock:
                            if not slot:
                                break
                        continue
                    self._cancel.wait(0.001)
                    continue
                yield item
        finally:
            reader.token.cancel("loop_end")
            reader.join(2.0)
        if reader.error is not None and not self._cancel.cancelled:
            raise reader.error

    def _run_parallel(self) -> None:
        engine = self._pose.parallel
        stamps: dict[int, int] = {}
        stamped = threading.Condition()
        capture_t0 = time.monotonic()

        def read(token: CancelToken) -> None:
            try:
                for index, frames, captured_at in self._read_frames(token):
                    frames = self._transform.process(FramePacket(index, frames), self.ctx).frames
                    self._update_sizes(frames)
                    if self.config.is_file:
                        ts = int(index * 1000.0 / max(1e-6, self.ctx.fps))
                    else:
                        ts = int((captured_at - capture_t0) * 1000.0)
                    submitted = engine.submit(frames[PRIMARY])
                    if submitted is not None:
                        with stamped:
                            stamps[submitted] = ts
                            stamped.notify_all()
            finally:
                engine.signal_input_done()

        reader = self._tasks.spawn("reader", read, owner="runner", close_budget_s=2.0)
        try:
            while not self._cancel.cancelled:
                res = engine.get(timeout=PARALLEL_GET_TIMEOUT_S)
                if res is None:
                    if engine.is_drained():
                        break
                    continue
                if res.frame is None:
                    continue
                # 丢帧不占序号：另存采集时刻，才能保留满队列丢帧造成的真实时间间隙。
                deadline = time.monotonic() + 0.2
                with stamped:
                    while res.index not in stamps and time.monotonic() < deadline:
                        stamped.wait(timeout=max(0.0, deadline - time.monotonic()))
                    ts = stamps.pop(res.index, int(res.index * 1000.0 / self.ctx.fps))
                packet = FramePacket(res.index, {PRIMARY: res.frame}, timestamp_ms=ts,
                                     pose={PRIMARY: res.pose_landmarks}, hands={PRIMARY: res.hands or []})
                if not self._finish_frame(packet):
                    break
        finally:
            engine.close()
            reader.token.cancel("loop_end")
            reader.join(2.0)
        if reader.error is not None and not self._cancel.cancelled:
            raise reader.error
        error = engine.take_error()
        if error is not None:
            raise error

    def _finish_frame(self, packet: FramePacket) -> bool:
        """推理之后的 stages、HUD、写录像、发布预览；CLI 窗口要求退出时返回 False。"""
        if self._pose is None:
            packet.out = {role: frame.copy() for role, frame in packet.frames.items()}
        for stage in self._stages:
            stage.process(packet, self.ctx)
        loop = self._loop
        loop.frames += 1
        total = self._source.total_frames()
        if self.config.is_file and total > 0 and (loop.frames % 5 == 0 or loop.frames == total):
            self._emit("progress", (loop.frames, total))
        fps = loop.frames / max(1e-6, time.monotonic() - loop.t0)
        suffix = f" (x{self.config.workers})" if self.config.parallel and not self.config.is_file else ""
        self.ctx.fps_text = f"{self.config.hud_title}: {fps:.1f}{suffix}"
        record = packet.out if self.config.record_overlay else packet.frames
        if self.config.record_hud:
            self._overlay.process(packet, self.ctx)
            self._write(record)
        else:
            self._write(record)
            self._overlay.process(packet, self.ctx)
        stage = "annotated" if self._pose is not None else "raw"
        self._publish(stage, packet.out, {role: actions_text(a) for role, a in packet.actions.items()})
        if self._display is not None:
            return self._display(packet)
        return True

    # ---- 辅助 ----

    def _rotate(self, frames) -> dict:
        packet = FramePacket(0, dict(frames))
        return self._transform.process(packet, self.ctx).frames

    def _update_sizes(self, frames) -> None:
        if self._sink is not None and self._sink_open and not self.config.is_file:
            self._sink.update_sizes({
                view_for_role(role): (int(frame.shape[1]), int(frame.shape[0])) for role, frame in frames.items()
            })

    def _write(self, frames) -> None:
        if self._sink is not None and self._sink_open:
            self._sink.write({view_for_role(role): frame for role, frame in frames.items()})

    def _publish(self, stage: str, frames, actions) -> None:
        if self._preview is not None:
            self._preview.publish(stage, frames, actions)

    def _mark(self, name: str) -> None:
        if self._metrics is not None:
            self._metrics.mark(name)

    def _shutdown(self, outcome: SessionOutcome) -> SessionOutcome:
        self._tasks.cancel(reason="session_end")
        self._tasks.join_until(time.monotonic() + 2.0)
        for stage in reversed(self._opened):
            try:
                stage.close()
            except Exception:
                pass
        try:
            self._source.close(keep_warm=not outcome.failed)
        except Exception as exc:
            traceback.print_exc()
            error = f"摄像头释放失败：{exc}"
            outcome = SessionOutcome(
                "failed", outcome.status if outcome.error_type else error,
                outcome.error_type or "CameraReleaseError", outcome.frames, error=outcome.error or exc,
            )
            self._status(outcome.status or error)
        if self._sink is not None and self._sink_open:
            self._sink_open = False
            self._sink.close_session()
        if self._preview is not None:
            self._preview.close()
        if self._metrics is not None:
            self._metrics.finish(outcome.outcome, outcome.error_type)
        return outcome


__all__ = ["FrameSink", "RunningInfo", "SessionEvent", "SessionOutcome", "SessionRunner"]
