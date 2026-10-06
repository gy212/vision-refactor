# -*- coding: utf-8 -*-
"""唯一的"解码 → 推理"循环 ``extract_pose_sequence``、提取预设 ``ExtractionProfile`` 与作用域缓存 ``pose_cache``（设计文档 §6.3）。

``workers > 1`` 时沿用旧的分段方案：每段从 ``seg_start - 15`` 定位开始读，前 15 帧只做跟踪预热，
结果标记 ``tracking="segmented"``。``with pose_cache():`` 内同一视频、同一引擎与预设只推理一次；
顺序跟踪的请求不会命中分段结果。
"""
from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from core.foundation.tasks import CancelToken, Cancelled, TaskGroup, from_stopped
from core.media.video_source import DecodeOptions, VideoOpenError, open_video, probe_video
from core.pose.engine import EngineSpec, create_engine
from core.pose.landmarks import LANDMARK_COUNT, landmarks_to_array
from core.pose.sequence import PoseSequence, SegmentWarmup, SourceMeta
from core.pose.validity import DEFAULT_VALID_CONF_THR, MEDIAPIPE_VALIDITY_POLICY

SEGMENT_WARMUP_FRAMES = 15
ProgressFn = Callable[[int, int], None]
EngineFactory = Callable[[], object]


@dataclass(frozen=True)
class ExtractionProfile:
    """名称 + 解码选项 + 引擎构造方式。``lazy_engine``：读到第一帧才构造引擎（练习分析 v4）。"""

    name: str
    decode: DecodeOptions = DecodeOptions()
    lazy_engine: bool = False
    download_models: bool = True


FORMAL = ExtractionProfile("formal")


def practice_v4_profile(*, sample_fps: float, max_edge: int) -> ExtractionProfile:
    """练习分析 v4：抽帧与长边上限取自 ``FeedbackConfig``（可被 ``feedback_config.json`` 覆盖）。"""
    return ExtractionProfile(
        "practice_v4",
        DecodeOptions(sample_fps=sample_fps or None, max_edge=max_edge, timestamps="round_monotonic", strict=True),
        lazy_engine=True,
        download_models=False,
    )


PRACTICE_V4 = practice_v4_profile(sample_fps=0.0, max_edge=1920)


def formal_engine(pose_variant: str) -> EngineSpec:
    """正式评估与模板特征：VIDEO 模式、不跑手部。"""
    return EngineSpec(pose_variant=pose_variant, running_mode="video", enable_hands=False)


def practice_engine(pose_variant: str) -> EngineSpec:
    """练习分析：CPU、VIDEO 模式、不跑手部。"""
    return EngineSpec(pose_variant=pose_variant, enable_hands=False, delegate="cpu", running_mode="video")


# --------------------------------------------------------------------------- #
# 作用域缓存
# --------------------------------------------------------------------------- #

_CACHE: contextvars.ContextVar[dict | None] = contextvars.ContextVar("pose_cache", default=None)


@contextmanager
def pose_cache() -> Iterator[None]:
    """作用域内缓存提取结果；离开作用域即丢弃。只对当前线程（上下文）生效。"""
    token = _CACHE.set({})
    try:
        yield
    finally:
        _CACHE.reset(token)


def _file_key(video: Path) -> tuple:
    try:
        st = video.stat()
        return (str(video.resolve()), st.st_size, st.st_mtime_ns)
    except OSError:
        return (str(video), None, None)


def _cache_key(video: Path, engine: EngineSpec, profile: ExtractionProfile, workers: int) -> tuple:
    decode = replace(profile.decode, end=None)
    tracking = ("segmented", workers) if workers > 1 else ("sequential",)
    return (_file_key(video), engine, profile.name, decode, profile.lazy_engine, tracking)


def _cache_lookup(store: dict, key: tuple, end: int | None) -> PoseSequence | None:
    """顺序跟踪是因果的：读到更远（或读到结尾）的结果截前缀即等于只读到 ``end`` 的结果。"""
    for cached_end, seq in store.get(key, ()):
        reached_eof = cached_end is not None and len(seq) <= cached_end
        if cached_end == end or (end is None and reached_eof):
            return seq
        if key[-1][0] == "sequential" and end is not None:
            if cached_end is None or cached_end >= end or reached_eof:
                return seq.prefix(end)
    return None


# --------------------------------------------------------------------------- #
# 提取
# --------------------------------------------------------------------------- #


def _close(engine) -> None:
    close = getattr(engine, "close", None)
    if callable(close):
        close()


def _meta(source_meta, engine: EngineSpec, profile: ExtractionProfile, tracking: str, *,
          frames_scanned: int = 0, stride: int = 1, inference_size=(0, 0)) -> SourceMeta:
    return SourceMeta(
        video=str(source_meta.path), fps=source_meta.fps, fps_reported=source_meta.fps_reported,
        frame_count=source_meta.frame_count, width=source_meta.width, height=source_meta.height,
        engine=engine, profile=profile.name, tracking=tracking,
        validity_policy=MEDIAPIPE_VALIDITY_POLICY, valid_conf_thr=float(DEFAULT_VALID_CONF_THR),
        frames_scanned=frames_scanned, stride=stride, inference_size=inference_size,
    )


def _sequential(video: Path, engine: EngineSpec, profile: ExtractionProfile, factory: EngineFactory,
                cancel: CancelToken | None, progress: ProgressFn | None) -> PoseSequence:
    rows: list[np.ndarray] = []
    detected: list[bool] = []
    frame_index: list[int] = []
    t_ms: list[int] = []
    source_wh: list[tuple[int, int]] = []
    inference_size = (0, 0)
    pipe = None
    with open_video(video, profile.decode, cancel) as source:
        total = source.meta.frame_count
        try:
            if not profile.lazy_engine:
                pipe = factory()
            for frame in source:
                if pipe is None:
                    pipe = factory()
                pose_landmarks, _hands = pipe.infer(frame.bgr, timestamp_ms=frame.t_ms)
                hit = pose_landmarks is not None
                rows.append(landmarks_to_array(pose_landmarks) if hit else np.zeros((LANDMARK_COUNT, 4)))
                detected.append(hit)
                frame_index.append(frame.index)
                t_ms.append(frame.t_ms)
                source_wh.append(frame.source_size)
                shape = getattr(frame.bgr, "shape", None)
                if shape is not None:
                    inference_size = (int(shape[1]), int(shape[0]))
                done = len(rows)
                if progress is not None and ((done % 10 == 0) or (total > 0 and done == total)):
                    progress(done, total)
        finally:
            if pipe is not None:
                _close(pipe)
        meta = _meta(source.meta, engine, profile, "sequential", frames_scanned=source.frames_scanned,
                     stride=source.stride(), inference_size=inference_size)
    n = len(rows)
    return PoseSequence(
        landmarks=np.stack(rows, axis=0) if rows else np.zeros((0, LANDMARK_COUNT, 4)),
        detected=np.asarray(detected, dtype=bool),
        decoded=np.ones((n,), dtype=bool),
        frame_index=np.asarray(frame_index, dtype=np.int64),
        t_ms=np.asarray(t_ms, dtype=np.int64),
        source_wh=np.asarray(source_wh, dtype=np.int64).reshape(n, 2),
        meta=meta,
    )


def _segments(total: int, workers: int) -> list[tuple[int, int]]:
    segs = []
    for k in range(workers):
        s = (k * total) // workers
        e = ((k + 1) * total) // workers - 1
        if e >= s:
            segs.append((s, e))
    return segs


def _segmented(video: Path, engine: EngineSpec, profile: ExtractionProfile, factory: EngineFactory,
               workers: int, source_meta, cancel: CancelToken | None,
               progress: ProgressFn | None) -> PoseSequence:
    total = source_meta.frame_count
    landmarks = np.zeros((total, LANDMARK_COUNT, 4))
    detected = np.zeros((total,), dtype=bool)
    decoded = np.zeros((total,), dtype=bool)
    t_ms = np.zeros((total,), dtype=np.int64)
    shared = from_stopped(lambda: cancel is not None and cancel.cancelled)
    errors: list[BaseException] = []
    warmups: dict[int, SegmentWarmup] = {}
    lock = threading.Lock()
    done = 0

    def advance() -> None:
        nonlocal done
        with lock:
            done += 1
            d = done
        if progress is not None and ((d % 10 == 0) or d == total):
            progress(d, total)

    def fail(exc: BaseException) -> None:
        with lock:
            errors.append(exc)
        shared.cancel("error")

    def run(seg_start: int, seg_end: int) -> None:
        warm_start = max(0, seg_start - SEGMENT_WARMUP_FRAMES)
        options = replace(profile.decode, start=warm_start, end=seg_end, seek=True)
        warm_rows: list[np.ndarray] = []
        warm_hit: list[bool] = []
        pipe = None
        try:
            if shared.cancelled:
                return
            try:
                source = open_video(video, options, shared)
            except VideoOpenError:
                raise RuntimeError(f"无法打开视频分段：{video}") from None
            with source:
                pipe = factory()
                for frame in source:
                    pose_landmarks, _hands = pipe.infer(frame.bgr, timestamp_ms=frame.t_ms)
                    hit = pose_landmarks is not None
                    row = landmarks_to_array(pose_landmarks) if hit else np.zeros((LANDMARK_COUNT, 4))
                    if frame.index >= seg_start:
                        landmarks[frame.index] = row
                        detected[frame.index] = hit
                        decoded[frame.index] = True
                        t_ms[frame.index] = frame.t_ms
                        advance()
                    else:
                        warm_rows.append(row)
                        warm_hit.append(hit)
        except Cancelled:
            pass  # 其它段失败或外部取消：本段停止读帧，与旧实现跳出循环一致
        except BaseException as exc:  # 线程里的失败必须交给调用方
            fail(exc)
        finally:
            if pipe is not None:
                try:
                    _close(pipe)
                except BaseException as exc:
                    fail(exc)
            warmups[seg_start] = SegmentWarmup(
                seg_start, seg_end,
                np.stack(warm_rows, axis=0) if warm_rows else np.zeros((0, LANDMARK_COUNT, 4)),
                np.asarray(warm_hit, dtype=bool),
            )

    group = TaskGroup("pose-extract")
    handles = [group.spawn(f"segment-{s}", lambda _t, s=s, e=e: run(s, e), owner="extract_pose_sequence")
               for s, e in _segments(total, workers)]
    for handle in handles:
        handle.join()
    if errors:
        raise errors[0]
    shared.raise_if_cancelled()
    return PoseSequence(
        landmarks=landmarks, detected=detected, decoded=decoded,
        frame_index=np.arange(total, dtype=np.int64), t_ms=t_ms,
        source_wh=np.zeros((total, 2), dtype=np.int64),
        meta=_meta(source_meta, engine, profile, "segmented"),
        segments=tuple(warmups[s] for s in sorted(warmups)),
    )


def extract_pose_sequence(video: Path | str, engine: EngineSpec, profile: ExtractionProfile = FORMAL, *,
                          workers: int = 1, cancel: CancelToken | None = None,
                          progress: ProgressFn | None = None,
                          engine_factory: EngineFactory | None = None) -> PoseSequence:
    """解码并逐帧推理，返回 ``PoseSequence``。取消时抛 ``Cancelled``；视频打不开抛 ``VideoOpenError``。

    ``workers > 1`` 且容器给出帧数时分段并行（``tracking="segmented"``），否则顺序跟踪。
    ``engine_factory`` 用于注入假引擎（测试、练习分析的 ``pipeline_factory``）；注入时不走缓存。
    """
    video = Path(str(video))
    workers = max(1, int(workers))
    factory = engine_factory or (lambda: create_engine(engine, download=profile.download_models))
    store = _CACHE.get() if engine_factory is None else None
    source_meta = None
    if workers > 1:
        source_meta = probe_video(video)
        if source_meta.frame_count <= 0:
            workers = 1  # 帧数未知无法分段，退回顺序跟踪
    key = _cache_key(video, engine, profile, workers) if store is not None else None
    if store is not None:
        hit = _cache_lookup(store, key, profile.decode.end)
        if hit is not None:
            return hit
    if workers > 1:
        seq = _segmented(video, engine, profile, factory, workers, source_meta, cancel, progress)
    else:
        seq = _sequential(video, engine, profile, factory, cancel, progress)
    if store is not None:
        store.setdefault(key, []).append((profile.decode.end, seq))
    return seq
