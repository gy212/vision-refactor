# -*- coding: utf-8 -*-
"""录制服务 ``RecordingService``：开始、暂停、结束片段，并把结束的片段连同处理计划提交给 finalizer。

摄像头会话的开录与结束是异步的（D-27，设计文档 §8.8 控制）：``start`` 立即返回片段，各路写出首帧后经
``on_confirm`` 回送结果；``finish`` 截断结束边界后立即返回，各路写完时（录制的任务线程里）才在这把锁内
构建计划并提交一次。会话结束的 ``close_session`` 等写完再返回。提交与关窗时的 ``cancel_all`` 用同一把锁，
每个结束的片段要么已提交、要么在关窗后被拒绝提交。写帧不经这把锁（摄像头的帧由各路录像线程取，
视频文件的帧由会话线程写），提交不阻塞采集。
处理计划由构建方给出：显式传入 ``plan``，或由构造时注入的 ``plan_for`` 按片段生成（返回 None 表示不提交）。
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from core.recording.finalizer import ProcessingPlan, SegmentFinalizer
from core.recording.recorder import (
    BeginError,
    FinishedSegment,
    RecorderSnapshot,
    RotationOf,
    SegmentMeta,
    SegmentRecorder,
    SegmentRef,
    SegmentState,
    StartConfirm,
)
from core.recording.worker import FrameFeed

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
    def begin_session(self, sizes: Mapping[str, tuple[int, int]], fps: float, *,
                      feed: FrameFeed | None = None, rotation_of: RotationOf | None = None) -> None:
        """摄像头会话给出 ``feed``，各路录像线程从中取帧；视频文件会话没有，由会话线程 ``write``。"""
        self._recorder.begin_session(sizes, fps, feed=feed, rotation_of=rotation_of)

    def write(self, frames: Mapping[str, np.ndarray]) -> None:
        self._recorder.write(frames)

    def close_session(self, *, plan: ProcessingPlan | None = None) -> FinishOutcome:
        """会话结束：结束仍在进行的片段并照常提交，等各路写完再关闭；不持锁等待（写完的回调要取这把锁）。"""
        outcome: list[FinishOutcome] = []
        finished = self._recorder.close_session(on_finished=lambda seg: outcome.append(self._submit(seg, plan)))
        return outcome[0] if outcome else FinishOutcome(finished, False)

    # ------------------------------------------------------------------ 片段（UI 线程）
    def start(self, meta: SegmentMeta, on_confirm: StartConfirm | None = None) -> SegmentRef | BeginError:
        """开录；``on_confirm(片段, 失败原因或 None)`` 在各路写出首帧或开录失败后回送一次（录制的任务线程）。"""
        with self._lock:
            return self._recorder.start_segment(meta, on_confirm)

    def pause(self) -> SegmentState:
        with self._lock:
            return self._recorder.pause()

    def resume(self) -> SegmentState:
        with self._lock:
            return self._recorder.resume()

    def toggle_pause(self) -> SegmentState:
        """recording⇄paused；各路切换后分叉时结束该片段并返回其后的状态。"""
        with self._lock:
            state = self._recorder.toggle_pause()
            if state is None:
                self._end(discard=False, plan=None)
                return self._recorder.state
            return state

    def finish(self, *, discard: bool = False, plan: ProcessingPlan | None = None) -> SegmentRef | None:
        """结束当前片段，不等待写完；``discard`` 时只落盘不提交。返回正在结束的片段（没有则 None）。"""
        with self._lock:
            return self._end(discard=discard, plan=plan)

    def finish_if_broken(self) -> SegmentRef | None:
        """多路片段已分叉（写失败或状态不一致）时结束它（写完后提交）；未分叉返回 None。"""
        with self._lock:
            if not self._recorder.broken():
                return None
            return self._end(discard=False, plan=None)

    def wait_idle(self, timeout: float) -> bool:
        """等正在结束的片段写完并提交（关窗收尾、测试用）。"""
        return self._recorder.wait_idle(timeout)

    def cancel_all(self) -> None:
        with self._lock:
            self._finalizer.cancel_all()

    # ------------------------------------------------------------------ 查询
    @property
    def state(self) -> SegmentState:
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
    def _end(self, *, discard: bool, plan: ProcessingPlan | None) -> SegmentRef | None:
        def finished(segment: FinishedSegment) -> None:
            if not discard:
                self._submit(segment, plan)

        return self._recorder.request_end(discard=discard, on_finished=finished)

    def _submit(self, segment: FinishedSegment | None, plan: ProcessingPlan | None) -> FinishOutcome:
        with self._lock:
            if segment is None or not segment.has_activity:
                return FinishOutcome(segment, False)
            plan = plan or self._plan_for(segment)
            if plan is None:
                return FinishOutcome(segment, False)
            return FinishOutcome(segment, self._finalizer.submit(segment, plan))
