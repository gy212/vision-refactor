# -*- coding: utf-8 -*-
"""站位采样：按固定时间间隔取主路最新帧跑 Lite 站位检测，发出带源帧采集时刻的 ``presence`` 事件（D-30，设计文档 §8.8）。

采样线程与处理循环分开：循环慢、预览不渲染时采样照常；没有新帧时不发结果（没有新结果不等于离场）。
样本时刻是源帧的采集时刻，换算到工作流使用的 ``time.monotonic``；推理期间站位设置变了（停用、重新启用、
改 ROI）的结果丢弃，不把上一流程的结果交给下一流程。引擎在启用后第一次需要时构造、停用后关闭，
构造、推理与关闭都在采样线程里。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

from core.foundation.paths import models_dir
from core.foundation.tasks import CancelToken
from core.live.config import LiveControls, PresenceProbe
from core.media import PRIMARY, CapturedFrame, apply_rotation, monotonic_of_capture
from core.pose import EngineSpec, create_engine

PRESENCE_INTERVAL_S = 0.1          # 约 10 次/秒（D-30）
IDLE_POLL_S = 0.05                 # 未启用或没有新帧时再看一次的间隔


@dataclass(frozen=True)
class PresenceSample:
    """站位检测的一次采样：人是否在 ROI 内；``captured_at`` 是源帧的采集时刻（``time.monotonic`` 秒），
    ``seq`` 是该帧在主路的序号。"""

    present: bool
    captured_at: float
    seq: int = 0


class LatestFeed(Protocol):
    """采样用的帧来源（``CameraStream``）：各路此刻的最新帧，不推进其它消费方的游标。"""

    def latest(self, role: str) -> CapturedFrame | None: ...


class PresenceSampler:
    """一个会话一个实例；``run`` 在会话的任务线程里执行，直到会话取消。"""

    def __init__(self, probe: PresenceProbe, *, interval_s: float = PRESENCE_INTERVAL_S) -> None:
        self._probe = probe
        self._interval_s = interval_s
        self._engine = None
        self._origin_ns: int | None = None
        self._last_ts = -1
        self._last_frame: tuple[int, int] | None = None

    def run(self, token: CancelToken, feed: LatestFeed, controls: LiveControls,
            emit: Callable[[str, object], None]) -> None:
        try:
            while not token.cancelled:
                started = time.monotonic()
                sampled = self._sample(feed, controls, emit)
                wait = self._interval_s - (time.monotonic() - started) if sampled else IDLE_POLL_S
                token.wait(max(0.0, wait))
        finally:
            self._close_engine()

    def _sample(self, feed: LatestFeed, controls: LiveControls, emit: Callable[[str, object], None]) -> bool:
        """采一次；返回是否推理了新帧（据此决定等一个间隔还是很快再看）。"""
        version = controls.occupancy.version
        settings = controls.occupancy.get()
        if not settings.armed:
            self._close_engine()
            return False
        frame = feed.latest(PRIMARY)
        if frame is None or (frame.generation, frame.seq) == self._last_frame:
            return False
        self._last_frame = (frame.generation, frame.seq)
        try:
            engine = self._ensure_engine()
            image = apply_rotation(frame.bgr, controls.rotation_of(PRIMARY))
            pose, _ = engine.infer(image, timestamp_ms=self._timestamp_ms(frame.captured_at_ns))
            present = bool(self._probe(pose, settings.roi))
        except Exception:
            return True
        if controls.occupancy.version != version:   # 推理期间停用、重新启用或改了 ROI：结果作废
            return True
        emit("presence", PresenceSample(present, monotonic_of_capture(frame.captured_at_ns), frame.seq))
        return True

    def _ensure_engine(self):
        if self._engine is None:
            self._engine = create_engine(
                EngineSpec(pose_variant="lite", running_mode="video", enable_hands=False),
                models_dir=models_dir(),
            )
            self._origin_ns = None
            self._last_ts = -1
        return self._engine

    def _timestamp_ms(self, captured_at_ns: int) -> int:
        """VIDEO 模式的时间戳：按采集时刻计，自引擎构造后的第一帧起，严格递增。"""
        if self._origin_ns is None:
            self._origin_ns = captured_at_ns
        ts = max(self._last_ts + 1, (captured_at_ns - self._origin_ns) // 1_000_000)
        self._last_ts = ts
        return ts

    def _close_engine(self) -> None:
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception:
                pass
            self._engine = None


__all__ = ["IDLE_POLL_S", "LatestFeed", "PRESENCE_INTERVAL_S", "PresenceSample", "PresenceSampler"]
