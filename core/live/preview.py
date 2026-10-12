# -*- coding: utf-8 -*-
"""会话预览的交接槽 ``PreviewStore``：runner 写入最新一帧（已转 RGB、已缩放），界面每次只取比已显示更新的一帧。

- 每会话一个 store、一个槽，序号单调递增；新帧覆盖未取走的旧帧。
- 阶段只进不退：进入 ``annotated`` 后不再接受 ``raw`` 帧（加载期裸帧泵与正式循环的交接）。
- 界面显示一帧后调用 ``mark_rendered``，供启动耗时统计与 runner 的"首帧已显示"等待。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping

import cv2
import numpy as np

from core.live.metrics import StartupMetrics
from core.media import PRIMARY

STAGES = ("raw", "annotated")
_RANK = {stage: rank for rank, stage in enumerate(STAGES)}

TargetSize = Callable[[str], tuple[int, int]]


def fit_preview_rgb(frame_bgr: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """BGR→RGB 并按目标区域等比缩放；目标尚未布局（≤1 像素）时不缩放。不改动传入的帧。"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pw, ph = target
    if pw > 1 and ph > 1:
        ih, iw = rgb.shape[:2]
        scale = min(pw / iw, ph / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return rgb


@dataclass(frozen=True)
class PreviewFrame:
    """一帧预览。``images`` 是各路 RGB 图；``sizes`` 是各路缩放前的宽高（双摄布局据此选择）。"""

    token: int
    seq: int
    stage: str
    images: Mapping[str, np.ndarray]
    sizes: Mapping[str, tuple[int, int]]
    actions: Mapping[str, str]
    enqueued_at: float

    @property
    def actions_text(self) -> str:
        return self.actions.get(PRIMARY, "-")


class PreviewStore:
    """一次会话的预览槽。``publish`` 在 runner 线程调用，``take_newer``/``mark_rendered`` 在界面线程调用。"""

    def __init__(self, token: int, target_size: TargetSize, *, metrics: StartupMetrics | None = None) -> None:
        self.token = token
        self.metrics = metrics
        self._target_size = target_size
        self._lock = threading.Lock()
        self._stage = STAGES[0]
        self._seq = 0
        self._slot: PreviewFrame | None = None
        self._closed = False
        self._first_render = threading.Event()

    @property
    def stage(self) -> str:
        with self._lock:
            return self._stage

    def advance(self, stage: str) -> None:
        """进入更高的阶段并丢掉槽里旧阶段的帧；不能后退。"""
        with self._lock:
            if _RANK[stage] > _RANK[self._stage]:
                self._stage = stage
                if self._slot is not None and self._slot.stage != stage:
                    self._slot = None

    def publish(self, stage: str, frames: Mapping[str, np.ndarray], actions: Mapping[str, str]) -> bool:
        """转换并写入一帧；会话已关闭或阶段与当前不符时丢弃，返回是否写入。"""
        if stage not in _RANK:
            raise ValueError(f"unsupported preview stage: {stage}")
        with self._lock:
            if self._closed or stage != self._stage:
                return False
        images = {role: fit_preview_rgb(frame, self._target_size(role)) for role, frame in frames.items()}
        sizes = {role: (int(frame.shape[1]), int(frame.shape[0])) for role, frame in frames.items()}
        now = time.monotonic()
        with self._lock:
            if self._closed or stage != self._stage:
                return False
            self._seq += 1
            self._slot = PreviewFrame(self.token, self._seq, stage, images, sizes, dict(actions), now)
        if self.metrics is not None:
            self.metrics.mark("first_pair_enqueued", now)
            if stage == "annotated":
                self.metrics.mark("first_annotated_enqueued", now)
        return True

    def take_newer(self, shown_seq: int) -> PreviewFrame | None:
        """取走槽里序号大于 ``shown_seq`` 且阶段不落后的帧；没有则 None。"""
        with self._lock:
            frame = self._slot
            if frame is None or frame.seq <= shown_seq or _RANK[frame.stage] < _RANK[self._stage]:
                return None
            self._slot = None
            return frame

    def mark_rendered(self, frame: PreviewFrame) -> None:
        """界面已显示 ``frame``：记启动耗时，唤醒等待首帧显示的 runner。"""
        now = time.monotonic()
        if self.metrics is not None:
            self.metrics.mark("first_pair_rendered", now)
            if frame.stage == "annotated":
                self.metrics.mark("first_annotated_rendered", now)
            self.metrics.maybe_emit()
        self._first_render.set()

    def wait_rendered(self, timeout: float) -> bool:
        return self._first_render.wait(timeout)

    def close(self) -> None:
        """会话结束：不再接受新帧，清空槽，放开等待中的 runner。"""
        with self._lock:
            self._closed = True
            self._slot = None
        self._first_render.set()


__all__ = ["PreviewFrame", "PreviewStore", "STAGES", "fit_preview_rgb"]
