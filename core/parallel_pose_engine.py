# -*- coding: utf-8 -*-
"""多核并行姿态推理引擎（IMAGE 模式）。

背景与动机
----------
单条 MediaPipe ``PoseLandmarker`` 在 VIDEO 模式下是串行的：每帧只用到底层 C++/TFLite
的少数线程，在多核机器上（例如 32 逻辑核）整机 CPU 利用率很低（~10%），而 heavy 模型
单帧 ~55ms，实时只能跑到 ~15-18fps。实测把多条 IMAGE 模式管线并行起来可近线性扩展
（4 worker ≈ 45fps，6 worker ≈ 57fps）。

本模块把"读帧 → 多 worker 并行推理 → 按帧序重排输出"这套逻辑抽成一个可复用、可测试的
引擎，供 CLI（``apps/main.py``）与 UI（``apps/app_ui.py``）的离线与实时路径共用。

取舍
----
- IMAGE 模式逐帧独立推理，**没有 VIDEO 模式的时序跟踪/平滑**，骨架会比单线程 VIDEO
  更抖。这是用吞吐换平滑的显式取舍，调用方需要知情（默认路径仍保持单线程 VIDEO）。
- 实时摄像头：用有界输入队列 + ``drop_when_full`` 丢弃新帧来约束端到端延迟；离线文件：
  关闭丢帧、用阻塞背压保证每帧都被处理。
- 输出严格按 ``submit`` 分配的帧序号重排，保证录制/显示帧顺序正确。被丢弃的帧不分配
  序号，因此不会在重排序列里留下空洞。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Callable, Protocol

import numpy as np

from core.vision_pipeline import MediaPipePipeline, PipelineConfig


class _AnnotatePipeline(Protocol):
    """引擎依赖 ``annotate``/``close``；``defer_draw`` 模式下额外依赖 ``infer``。"""

    def annotate(
        self, frame_bgr: np.ndarray, *, timestamp_ms: int | None = None
    ) -> tuple[np.ndarray, list[str]]:
        ...

    def infer(self, frame_bgr: np.ndarray, *, timestamp_ms: int | None = None):
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class InferResult:
    """单帧推理结果（按帧序号 ``index`` 重排后交付）。"""

    index: int
    annotated: np.ndarray | None
    actions: list[str]
    delegate: dict[str, object] | None = None
    # defer_draw=True 时携带 raw landmarks + 原始帧，供消费端在重排出口单线程 smooth+draw；
    # 此时 ``annotated`` 为 None、``actions`` 为空（动作分类在 draw 阶段做）。
    pose_landmarks: object | None = None
    hands: object | None = None
    frame: np.ndarray | None = None


PipelineFactory = Callable[[], _AnnotatePipeline]


def default_pipeline_factory(
    *, models_dir: Path, pose_variant: str, enable_hands: bool, delegate: str = "cpu",
    draw_joint_angles: bool = False,
) -> PipelineFactory:
    """返回一个创建 IMAGE 模式 ``MediaPipePipeline`` 的工厂。

    每个 worker 必须拥有独立的管线实例（MediaPipe landmarker 非线程安全），因此这里
    返回工厂而非实例。
    """

    def _factory() -> _AnnotatePipeline:
        return MediaPipePipeline(
            models_dir=models_dir,
            cfg=PipelineConfig(
                pose_variant=pose_variant,
                running_mode="image",
                enable_hands=enable_hands,
                delegate=delegate,
                draw_joint_angles=draw_joint_angles,
            ),
        )

    return _factory


class ParallelPoseEngine:
    """多 worker 并行姿态推理引擎。

    用法::

        engine = ParallelPoseEngine(pipeline_factory=..., workers=4, drop_when_full=True)
        engine.start()
        # 生产者线程：
        idx = engine.submit(frame)            # 返回分配的帧序号；丢帧时返回 None
        engine.signal_input_done()            # 所有帧已提交后调用
        # 消费者线程：
        while True:
            res = engine.get(timeout=0.2)     # 按帧序返回 InferResult
            if res is None:
                if engine.is_drained():
                    break
                continue
            ...
        engine.close()
    """

    def __init__(
        self,
        *,
        pipeline_factory: PipelineFactory,
        workers: int,
        drop_when_full: bool = False,
        queue_factor: int = 2,
        max_reorder_lag: int | None = None,
        defer_draw: bool = False,
    ) -> None:
        self._factory = pipeline_factory
        self._workers = max(1, int(workers))
        self._drop_when_full = bool(drop_when_full)
        self._defer_draw = bool(defer_draw)
        maxsize = max(1, self._workers * max(1, int(queue_factor)))
        # 输入队列：(idx, frame)；None 作为关闭哨兵。
        self._frame_q: "Queue[tuple[int, np.ndarray] | None]" = Queue(maxsize=maxsize)
        # worker 原始输出（乱序）。
        self._raw_q: "Queue[InferResult]" = Queue(maxsize=maxsize)
        # 重排后有序输出。
        self._ordered_q: "Queue[InferResult]" = Queue(maxsize=maxsize)
        # 重排允许的最大滞后：若等待的 next_idx 迟迟不到、而缓冲里已堆积超过该阈值，
        # 则跳过缺失帧继续输出，避免单帧卡死拖垮实时延迟。None 表示严格按序不跳。
        self._max_reorder_lag = max_reorder_lag

        self._next_submit_idx = 0
        self._submit_lock = threading.Lock()

        self._stop_evt = threading.Event()
        self._input_done_evt = threading.Event()

        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._delegate_payloads: list[dict[str, object]] = []
        self._delegate_lock = threading.Lock()

        self._worker_threads: list[threading.Thread] = []
        self._collector_thread: threading.Thread | None = None
        self._started = False

        # 统计
        self._submitted = 0
        self._dropped = 0
        self._emitted = 0
        self._stats_lock = threading.Lock()

    # ----- 生命周期 -----
    def start(self) -> None:
        if self._started:
            raise RuntimeError("ParallelPoseEngine already started")
        self._started = True
        self._worker_threads = [
            threading.Thread(target=self._worker_loop, name=f"pose-worker-{i}", daemon=True)
            for i in range(self._workers)
        ]
        for t in self._worker_threads:
            t.start()
        self._collector_thread = threading.Thread(
            target=self._collector_loop, name="pose-collector", daemon=True
        )
        self._collector_thread.start()

    def submit(self, frame: np.ndarray) -> int | None:
        """提交一帧。返回分配的帧序号；当 ``drop_when_full`` 且队列满时丢弃并返回 None。

        被丢弃的帧不消耗序号，因此重排序列保持连续、不留空洞。
        """
        if not self._started:
            raise RuntimeError("ParallelPoseEngine not started")
        with self._submit_lock:
            idx = self._next_submit_idx
            item = (idx, frame)
            if self._drop_when_full:
                try:
                    self._frame_q.put_nowait(item)
                except Full:
                    with self._stats_lock:
                        self._dropped += 1
                    return None
            else:
                self._frame_q.put(item)
            self._next_submit_idx += 1
            with self._stats_lock:
                self._submitted += 1
            return idx

    def signal_input_done(self) -> None:
        """通知引擎不会再有新帧；worker 处理完队列后退出。

        必须非阻塞：worker 在 ``frame_q.get`` 超时后会检查 ``_input_done_evt`` 自行退出，
        因此哨兵只是加速退出的优化。若在此处用阻塞 ``put`` 等待满队列，会与 worker→raw_q、
        collector→ordered_q 的背压链形成死锁（生产者线程被卡住，消费者无法开始排空）。
        """
        self._input_done_evt.set()
        for _ in range(self._workers):
            try:
                self._frame_q.put_nowait(None)
            except Full:
                # 队列满：依赖 _input_done_evt + worker 端超时退出，不在此阻塞。
                pass

    def get(self, timeout: float | None = None) -> InferResult | None:
        """取一帧有序结果；超时返回 None。"""
        try:
            return self._ordered_q.get(timeout=timeout)
        except Empty:
            return None

    def is_drained(self) -> bool:
        """输入已结束且所有线程退出、无残留输出 —— 可安全停止消费循环。"""
        if not self._input_done_evt.is_set():
            return False
        workers_alive = any(t.is_alive() for t in self._worker_threads)
        collector_alive = self._collector_thread.is_alive() if self._collector_thread else False
        return (not workers_alive) and (not collector_alive) and self._ordered_q.empty()

    def close(self) -> None:
        """停止所有线程并释放底层管线。"""
        self._stop_evt.set()
        # 解除可能阻塞在 put 上的生产者/哨兵
        self._input_done_evt.set()
        if self._collector_thread is not None:
            self._collector_thread.join(timeout=2.0)
        for t in self._worker_threads:
            t.join(timeout=2.0)

    def take_error(self) -> BaseException | None:
        with self._error_lock:
            return self._error

    @property
    def delegate_payload(self) -> dict[str, object]:
        with self._delegate_lock:
            payloads = [dict(item) for item in self._delegate_payloads]
        return _merge_delegate_payloads(payloads)

    @property
    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return {
                "submitted": self._submitted,
                "dropped": self._dropped,
                "emitted": self._emitted,
            }

    # ----- 内部线程 -----
    def _record_error(self, exc: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = exc
        self._stop_evt.set()

    def _record_delegate_payload(self, pipe: _AnnotatePipeline) -> dict[str, object] | None:
        payload = _delegate_payload_from_pipeline(pipe)
        if payload:
            with self._delegate_lock:
                self._delegate_payloads.append(payload)
        return payload or None

    def _worker_loop(self) -> None:
        pipe: _AnnotatePipeline | None = None
        delegate_payload: dict[str, object] | None = None
        try:
            pipe = self._factory()
            delegate_payload = self._record_delegate_payload(pipe)
            while not self._stop_evt.is_set():
                try:
                    item = self._frame_q.get(timeout=0.1)
                except Empty:
                    if self._input_done_evt.is_set():
                        break
                    continue
                if item is None:
                    break
                idx, frame = item
                if self._defer_draw:
                    pose_landmarks, hands = pipe.infer(frame, timestamp_ms=None)
                    result = InferResult(
                        index=idx,
                        annotated=None,
                        actions=[],
                        delegate=delegate_payload,
                        pose_landmarks=pose_landmarks,
                        hands=hands,
                        frame=frame,
                    )
                else:
                    annotated, actions = pipe.annotate(frame, timestamp_ms=None)
                    result = InferResult(
                        index=idx,
                        annotated=annotated,
                        actions=list(actions),
                        delegate=delegate_payload,
                    )
                # 可中断的有界 put：raw_q 满时不永久阻塞，响应 stop。
                while not self._stop_evt.is_set():
                    try:
                        self._raw_q.put(result, timeout=0.1)
                        break
                    except Full:
                        continue
        except BaseException as exc:  # noqa: BLE001 - 传播到消费侧
            self._record_error(exc)
        finally:
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:
                    pass

    def _collector_loop(self) -> None:
        """把乱序的 worker 结果按 ``index`` 重排后送入有序输出队列。"""
        pending: dict[int, InferResult] = {}
        next_idx = 0
        try:
            while True:
                # 先尽量输出已就绪的连续帧。
                while next_idx in pending:
                    self._emit(pending.pop(next_idx))
                    next_idx += 1

                # 重排滞后保护：缺失帧迟迟不到但缓冲已堆积过多 → 跳过它。
                if (
                    self._max_reorder_lag is not None
                    and pending
                    and (min(pending) - next_idx) >= 0
                    and len(pending) > self._max_reorder_lag
                ):
                    next_idx = min(pending)
                    continue

                # 终止条件：worker 全退出且原始队列排空、缓冲也清空。
                workers_alive = any(t.is_alive() for t in self._worker_threads)
                if not workers_alive and self._raw_q.empty() and not pending:
                    break
                if self._stop_evt.is_set() and self._raw_q.empty():
                    break

                try:
                    res = self._raw_q.get(timeout=0.1)
                except Empty:
                    continue
                pending[res.index] = res
        except BaseException as exc:  # noqa: BLE001
            self._record_error(exc)

    def _emit(self, res: InferResult) -> None:
        with self._stats_lock:
            self._emitted += 1
        # 输出队列满时不能永久阻塞（消费方可能已停止）：带超时重试 + 关停检查。
        while not self._stop_evt.is_set():
            try:
                self._ordered_q.put(res, timeout=0.1)
                return
            except Full:
                continue


def _delegate_payload_from_pipeline(pipe: object) -> dict[str, object]:
    requested = getattr(pipe, "requested_delegate", None)
    active = getattr(pipe, "active_delegate", None)
    reason = getattr(pipe, "delegate_fallback_reason", None)
    if requested is None and active is None and reason is None:
        return {}
    payload: dict[str, object] = {
        "requested": str(requested or active or "cpu"),
        "active": str(active or requested or "cpu"),
        "fallback": bool(reason),
    }
    if reason:
        payload["fallbackReason"] = str(reason)
    return payload


def _merge_delegate_payloads(payloads: list[dict[str, object]]) -> dict[str, object]:
    if not payloads:
        return {}
    requested = str(payloads[0].get("requested") or "cpu")
    active_values = {str(item.get("active") or requested) for item in payloads}
    fallback_reasons = [
        str(item.get("fallbackReason"))
        for item in payloads
        if item.get("fallbackReason")
    ]
    payload: dict[str, object] = {
        "requested": requested,
        "active": active_values.pop() if len(active_values) == 1 else "mixed",
        "fallback": any(bool(item.get("fallback")) for item in payloads),
    }
    if fallback_reasons:
        payload["fallbackReason"] = "; ".join(dict.fromkeys(fallback_reasons))
    return payload
