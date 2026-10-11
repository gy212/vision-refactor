# -*- coding: utf-8 -*-
"""录后处理执行器 ``SegmentFinalizer``：按处理计划在 io 线与 analysis 线上执行步骤，保证终态唯一、可取消。

一个任务 = 已结束的片段 + ``ProcessingPlan``。io 线做裁剪、转码、可读性校验；analysis 线串行做比对评分；
两条线各自 FIFO，可以并行（一个片段在比对时，下一个片段可以转码）。计划带 ``result`` 时，每次状态变化都原子写
``<片段目录>/result.json`` 并通知订阅者；同一片段只发布一次终态，取消后的再次取消只补写路径不再通知。
``cancel_all`` 取消尚未终态的任务并拒绝新提交，发布 ``app_closing`` 取消终态；``finish_on_close`` 的任务
（单摄转码，V-4）不取消，io 线是非守护线程，进程退出前会做完。由 ``DualRecordingPostProcessor`` 演化而来（P4）。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.foundation.atomic_io import atomic_write_json
from core.foundation.tasks import Cancelled, CancelToken, TaskGroup
from core.media.segment_layout import SegmentLayout
from core.recording.recorder import FinishedSegment
from core.recording.steps import (
    TERMINAL_STATUSES,
    SegmentStatus,
    Step,
    StepContext,
    PostprocessError,
    Terminal,
)

_EXAM_KEYS = ("exam_run_id", "student_id", "student_name", "student_order", "attempt_index")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class ResultSpec:
    """结果 JSON 中由计划构建方给定的字段。"""

    record_skeleton: bool = False
    auto_compare: bool = True
    front_template: Path = Path("")
    side_template: Path = Path("")
    exam: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ProcessingPlan:
    """处理计划：步骤列表；``result`` 为 None 时不写结果 JSON、不发更新（单摄）。"""

    steps: tuple[Step, ...]
    result: ResultSpec | None = None
    finish_on_close: bool = False
    close_budget_s: float = 0.0


@dataclass(frozen=True)
class SegmentUpdate:
    segment_id: str
    status: SegmentStatus
    message: str
    front_score: float | None = None
    side_score: float | None = None
    combined_percent: int | None = None
    error_code: str | None = None
    # 逐动作子分明细：({"name","view","score","start","end"}, ...)
    action_scores: tuple[dict[str, Any], ...] = ()
    # 当前最新的视频路径（学生练习等消费方用）
    front_video: Path | None = None
    side_video: Path | None = None
    # 考试片段所属的场次（按它回填台账）；非考试片段为 None
    exam_run_id: str | None = None


UpdateCallback = Callable[[SegmentUpdate], None]
ResultWriter = Callable[[Path, dict[str, Any]], None]


def write_result_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload, trailing_newline=True)


@dataclass(eq=False)
class _Job:
    segment: FinishedSegment
    plan: ProcessingPlan
    created_at: str
    token: CancelToken
    stop_evt: threading.Event = field(default_factory=threading.Event)
    ctx: StepContext | None = None
    index: int = 0
    started: bool = False


class _Lane:
    """一条 FIFO 执行线。有活时才有线程，队列空了线程就结束（不常驻、不阻塞进程退出）。"""

    def __init__(self, name: str, tasks: TaskGroup, run: Callable[[_Job], None], *, daemon: bool) -> None:
        self.name = name
        self._tasks = tasks
        self._run = run
        self._daemon = daemon
        self._lock = threading.Lock()
        self._queue: deque[_Job] = deque()
        self._running = False

    def put(self, job: _Job) -> None:
        with self._lock:
            self._queue.append(job)
            if self._running:
                return
            self._running = True
        self._tasks.spawn(self.name, self._drain, owner="segment-finalizer", daemon=self._daemon)

    @property
    def idle(self) -> bool:
        with self._lock:
            return not self._running and not self._queue

    def jobs(self) -> list[_Job]:
        with self._lock:
            return list(self._queue)

    def _drain(self, _token: CancelToken) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    self._running = False
                    return
                job = self._queue.popleft()
            self._run(job)


class SegmentFinalizer:
    def __init__(
        self,
        *,
        on_update: UpdateCallback | None = None,
        write_result: ResultWriter = write_result_json,
        tasks: TaskGroup | None = None,
    ) -> None:
        self._on_update = on_update
        self._write_result = write_result
        tasks = tasks or TaskGroup("finalizer")
        self._lanes = {
            "io": _Lane("io", tasks, lambda job: self._advance(job, "io"), daemon=False),
            "analysis": _Lane("analysis", tasks, lambda job: self._advance(job, "analysis"), daemon=True),
        }
        self._lock = threading.RLock()
        self._persistence_lock = threading.Lock()
        self._submitted: set[str] = set()
        self._dir_owners: dict[Any, str] = {}
        self._jobs: dict[str, _Job] = {}
        self._closing_cancelled: set[str] = set()
        self._terminal: dict[str, SegmentStatus] = {}
        self._closed = False

    # ------------------------------------------------------------------ 公共接口
    def submit(self, segment: FinishedSegment, plan: ProcessingPlan) -> bool:
        """提交一个片段；同一片段 id 或同一片段目录只接受一次。关闭后只接受 ``finish_on_close`` 的计划。"""
        try:
            keys = SegmentLayout.directory_keys(segment.directory) if segment.directory is not None else ()
        except (OSError, RuntimeError):
            return False
        with self._lock:
            if self._closed and not plan.finish_on_close:
                return False
            if segment.segment_id in self._submitted or any(key in self._dir_owners for key in keys):
                return False
            job = _Job(segment, plan, utc_timestamp(), CancelToken())
            job.ctx = self._context(job)
            self._submitted.add(segment.segment_id)
            for key in keys:
                self._dir_owners[key] = segment.segment_id
            self._jobs[segment.segment_id] = job
            self._lanes["io"].put(job)
        return True

    def cancel_all(self) -> None:
        """取消尚未终态的任务（``finish_on_close`` 的除外）并拒绝之后的提交；不等待。"""
        with self._lock:
            self._closed = True
            victims = [
                job for sid, job in self._jobs.items()
                if sid not in self._terminal and not job.plan.finish_on_close
            ]
            self._closing_cancelled.update(job.segment.segment_id for job in victims)
        for job in victims:
            job.stop_evt.set()
            job.token.cancel("app_close")

    def close(self, timeout: float) -> bool:
        """``cancel_all`` 后最多等 ``timeout`` 秒到两条线都空闲；返回是否已空闲。"""
        self.cancel_all()
        return self.wait_idle(timeout)

    def wait_idle(self, timeout: float) -> bool:
        end = time.monotonic() + max(0.0, float(timeout))
        while not self.idle:
            remaining = end - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.01, remaining))
        return True

    @property
    def idle(self) -> bool:
        return all(lane.idle for lane in self._lanes.values())

    def close_budget_s(self) -> float:
        """尚未终态的任务里最大的关窗预算（如考试裁剪）。"""
        with self._lock:
            pending = [job for sid, job in self._jobs.items() if sid not in self._terminal]
        return max((job.plan.close_budget_s for job in pending), default=0.0)

    # ------------------------------------------------------------------ 执行
    def _advance(self, job: _Job, lane: str) -> None:
        try:
            if not job.started:
                job.started = True
                with self._lock:
                    closing = self._closed and not job.plan.finish_on_close
                if closing:
                    self._publish_terminal_safely(job, "cancelled", "应用关闭，后台比对已取消", "app_closing")
                    return
            steps = job.plan.steps
            while job.index < len(steps):
                step = steps[job.index]
                if step.lane != lane:
                    self._lanes[step.lane].put(job)
                    return
                if job.token.cancelled or job.stop_evt.is_set():
                    raise Cancelled(job.token.reason)
                outcome = step.run(job.ctx, job.token)
                job.index += 1
                if outcome is not None:
                    self._finish(job, outcome)
                    return
            if job.plan.result is not None:
                self._finish(job, Terminal("completed", "后台处理完成"))
        except (Cancelled, InterruptedError):
            self._publish_cancelled(job)
        except PostprocessError as exc:
            if self._is_closing_cancelled(job):
                self._publish_cancelled(job)
            else:
                self._publish_terminal_safely(job, "failed", str(exc), exc.code)
        except Exception as exc:  # noqa: BLE001 - 后台任务必须失败隔离
            if self._is_closing_cancelled(job):
                self._publish_cancelled(job)
            else:
                self._publish_terminal_safely(job, "failed", f"后台比对失败：{exc}", "compare_failed")

    def _context(self, job: _Job) -> StepContext:
        segment = job.segment
        views = tuple(segment.views)
        return StepContext(
            segment_id=segment.segment_id,
            segment_dir=segment.directory,
            views=views,
            sources={view: segment.source(view) for view in views},
            frames={view: rec.frames for view, rec in segment.views.items()},
            errors=segment.errors,
            stop_evt=job.stop_evt,
            publish=lambda status, message: self._publish(job, status, message),
        )

    def _finish(self, job: _Job, outcome: Terminal) -> None:
        if job.plan.result is None:
            return
        self._publish(job, outcome.status, outcome.message, error_code=outcome.error_code, result=outcome.result)

    def _is_closing_cancelled(self, job: _Job) -> bool:
        with self._lock:
            return job.segment.segment_id in self._closing_cancelled

    def _publish_cancelled(self, job: _Job) -> None:
        closing = self._is_closing_cancelled(job)
        self._publish_terminal_safely(
            job,
            "cancelled",
            "应用关闭，后台比对已取消" if closing else "后台比对已取消",
            "app_closing" if closing else "cancelled",
        )

    # ------------------------------------------------------------------ 发布
    def _payload(self, job: _Job, status: SegmentStatus, message: str, error_code: str | None,
                 result: dict[str, Any] | None) -> dict[str, Any]:
        spec = job.plan.result
        ctx = job.ctx
        warnings = list(ctx.warnings)
        for note in ctx.notes:
            if note not in warnings:
                warnings.append(str(note))
        exam = dict(spec.exam or {})
        has_exam = any(exam.get(key) is not None for key in _EXAM_KEYS) or bool(ctx.notes)
        return {
            "schema_version": 1,
            "status": status,
            "segment_id": job.segment.segment_id,
            "created_at": job.created_at,
            "completed_at": utc_timestamp() if status in TERMINAL_STATUSES else None,
            "record_skeleton": bool(spec.record_skeleton),
            "auto_compare": bool(spec.auto_compare),
            "front_video_path": str(ctx.video("front")),
            "side_video_path": str(ctx.video("side")),
            "front_template_path": str(spec.front_template),
            "side_template_path": str(spec.side_template),
            "warnings": warnings,
            "result": result,
            "error": None if error_code is None else {"code": error_code, "message": message},
            "exam": {key: exam.get(key) for key in _EXAM_KEYS} if has_exam else None,
        }

    def _publish(
        self,
        job: _Job,
        status: SegmentStatus,
        message: str,
        *,
        error_code: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        if job.plan.result is None or job.segment.directory is None:
            return
        segment_id = job.segment.segment_id
        payload = self._payload(job, status, message, error_code, result)
        update: SegmentUpdate | None = None
        with self._persistence_lock:
            enrichment = self._check_publishable(segment_id, status)
            if enrichment is None:
                return
            try:
                self._write_result(job.segment.directory / "result.json", payload)
            except Exception as exc:
                if enrichment:
                    return
                raise PostprocessError("result_write_failed", f"结果文件写入失败：{exc}") from exc
            with self._lock:  # 终检与终态提交同锁，cancel_all 不能插进中间
                enrichment = self._check_publishable(segment_id, status)
                if enrichment is None:
                    return
                if status in TERMINAL_STATUSES:
                    self._terminal[segment_id] = status
            if not enrichment:
                update = _update_from_payload(payload, message)
        if update is not None:
            self._notify(update)

    def _check_publishable(self, segment_id: str, status: SegmentStatus) -> bool | None:
        """None：已有终态，丢弃；True：取消后的再次取消（只补写路径，不通知）；False：正常发布。

        已被 ``cancel_all`` 取消的片段只能发布 cancelled，其他状态抛 ``Cancelled``。
        """
        with self._lock:
            existing = self._terminal.get(segment_id)
            enrichment = existing == "cancelled" and status == "cancelled"
            if existing is not None and not enrichment:
                return None
            if segment_id in self._closing_cancelled and status != "cancelled":
                raise Cancelled("app_close")
            return enrichment

    def _publish_terminal_safely(self, job: _Job, status: SegmentStatus, message: str,
                                 error_code: str | None = None) -> None:
        try:
            self._publish(job, status, message, error_code=error_code)
        except Cancelled:
            try:
                self._publish(job, "cancelled", "应用关闭，后台比对已取消", error_code="app_closing")
            except Exception as exc:  # noqa: BLE001 - 持久化失败不得杀死执行线
                self._notify_write_failure(job, exc)
        except Exception as exc:  # noqa: BLE001 - 持久化失败不得杀死执行线
            self._notify_write_failure(job, exc)

    def _notify_write_failure(self, job: _Job, exc: BaseException) -> None:
        if job.plan.result is None:
            return
        self._notify(SegmentUpdate(
            segment_id=job.segment.segment_id,
            status="failed",
            message=f"结果文件写入失败：{exc}",
            error_code="result_write_failed",
            exam_run_id=(job.plan.result.exam or {}).get("exam_run_id"),
        ))

    def _notify(self, update: SegmentUpdate) -> None:
        if self._on_update is None:
            return
        try:
            self._on_update(update)
        except Exception:
            pass


def _update_from_payload(payload: dict[str, Any], message: str) -> SegmentUpdate:
    result = payload.get("result") or {}
    error = payload.get("error") or {}
    front_raw = payload.get("front_video_path")
    side_raw = payload.get("side_video_path")
    return SegmentUpdate(
        segment_id=str(payload["segment_id"]),
        status=payload["status"],
        message=message,
        front_score=result.get("front_score"),
        side_score=result.get("side_score"),
        combined_percent=result.get("combined_percent"),
        error_code=error.get("code"),
        action_scores=tuple(result.get("action_scores") or ()),
        front_video=Path(str(front_raw)) if front_raw else None,
        side_video=Path(str(side_raw)) if side_raw else None,
        exam_run_id=(payload.get("exam") or {}).get("exam_run_id"),
    )
