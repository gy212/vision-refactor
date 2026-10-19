# -*- coding: utf-8 -*-
"""录制服务 ``RecordingService``：开始、暂停、结束片段，并在同一把锁内把结束的片段连同处理计划提交给 finalizer。

"结束片段 + 提交"在同一把锁内完成，关窗时的 ``cancel_all`` 不会插进两者之间，因此每个结束的片段要么已提交、
要么在关窗后被拒绝提交，不会出现"既没提交也没有取消终态"的片段。写帧不经这把锁，提交不阻塞采集线程。
处理计划由构建方给出：显式传入 ``plan``，或由构造时注入的 ``plan_for`` 按片段生成（返回 None 表示不提交）。
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from core.recording.controller import RecordingState
from core.recording.finalizer import ProcessingPlan, SegmentFinalizer
from core.recording.recorder import (
    BeginError,
    FinishedSegment,
    RecorderSnapshot,
    SegmentMeta,
    SegmentRecorder,
    SegmentRef,
)

PlanBuilder = Callable[[FinishedSegment], "ProcessingPlan | None"]


@dataclass(frozen=True)
class FinishOutcome:
    """结束片段的结果：结束的片段（没有进行中的片段时为 None），以及是否已提交后处理。"""

    segment: FinishedSegment | None
    submitted: bool = False


class RecordingService:
    def __init__(
        self,
        finalizer: SegmentFinalizer,
        plan_for: PlanBuilder,
        recorder: SegmentRecorder | None = None,
    ) -> None:
        self._finalizer = finalizer
        self._plan_for = plan_for
        self._recorder = recorder or SegmentRecorder()
        self._lock = threading.RLock()

    @property
    def finalizer(self) -> SegmentFinalizer:
        return self._finalizer

    # ------------------------------------------------------------------ 会话（采集线程）
    def begin_session(self, sizes: Mapping[str, tuple[int, int]], fps: float) -> None:
        with self._lock:
            self._recorder.begin_session(sizes, fps)

    def update_sizes(self, sizes: Mapping[str, tuple[int, int]]) -> None:
        self._recorder.update_sizes(sizes)

    def write(self, frames: Mapping[str, np.ndarray], captured_at: float | None = None) -> None:
        self._recorder.write(frames, captured_at)

    def close_session(self, *, plan: ProcessingPlan | None = None) -> FinishOutcome:
        """会话结束：关闭各路 writer，仍在进行的片段照常提交。"""
        with self._lock:
            return self._submit(self._recorder.close_session(), plan)

    # ------------------------------------------------------------------ 片段（UI 线程）
    def start(self, meta: SegmentMeta) -> SegmentRef | BeginError:
        with self._lock:
            return self._recorder.start_segment(meta)

    def toggle_pause(self) -> RecordingState:
        """recording⇄paused；各路切换后分叉时结束该片段并返回 idle。"""
        with self._lock:
            state = self._recorder.toggle_pause()
            if state is None:
                self._submit(self._recorder.end_segment(), None)
                return "idle"
            return state

    def finish(self, *, discard: bool = False, plan: ProcessingPlan | None = None) -> FinishOutcome:
        """结束当前片段；``discard`` 时只落盘不提交。"""
        with self._lock:
            segment = self._recorder.end_segment(discard=discard)
            if discard:
                return FinishOutcome(segment, False)
            return self._submit(segment, plan)

    def finish_if_broken(self) -> FinishOutcome | None:
        """多路片段已分叉（写失败或状态不一致）时结束并提交它；未分叉返回 None。"""
        with self._lock:
            if not self._recorder.broken():
                return None
            return self._submit(self._recorder.end_segment(), None)

    def cancel_all(self) -> None:
        with self._lock:
            self._finalizer.cancel_all()

    # ------------------------------------------------------------------ 查询
    @property
    def state(self) -> RecordingState:
        return self._recorder.state

    @property
    def views(self) -> tuple[str, ...]:
        return self._recorder.views

    @property
    def session_sizes(self) -> dict[str, tuple[int, int]]:
        return self._recorder.session_sizes

    def snapshot(self) -> RecorderSnapshot:
        return self._recorder.snapshot()

    # ------------------------------------------------------------------ 内部
    def _submit(self, segment: FinishedSegment | None, plan: ProcessingPlan | None) -> FinishOutcome:
        if segment is None or not segment.has_activity:
            return FinishOutcome(segment, False)
        plan = plan or self._plan_for(segment)
        if plan is None:
            return FinishOutcome(segment, False)
        return FinishOutcome(segment, self._finalizer.submit(segment, plan))
