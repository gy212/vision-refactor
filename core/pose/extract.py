# -*- coding: utf-8 -*-
"""唯一的"解码 → 推理"循环 ``extract_pose_sequence``、提取预设 ``ExtractionProfile`` 与作用域缓存 ``pose_cache``（设计文档 §6.3）。

``workers > 1`` 时沿用旧的分段方案：每段从 ``seg_start - 15`` 定位开始读，前 15 帧只做跟踪预热，
结果标记 ``tracking="segmented"``。分段按分段边界定位读取，保不住 ``profile.decode`` 的
``start``/``end``/``sample_fps`` 语义，这些选项非默认值时退回顺序跟踪（和帧数未知时一样）。
``with pose_cache():`` 内同一视频、同一引擎与预设只推理一次；顺序跟踪的请求不会命中分段结果。
前缀复用只在"第 t 行 = 第 t 帧"成立时发生（``start=0``、不抽帧、不定位、不缩放），其余情形
只认终点相同的命中；是否读到结尾由 ``VideoSource.exhausted`` 判定，不从行数反推。
"""

from __future__ import annotations

import contextvars
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from core.foundation.paths import models_dir as default_models_dir
from core.foundation.tasks import CancelToken, Cancelled, TaskGroup, from_stopped
from core.media.video_source import DecodeOptions, VideoOpenError, open_video, probe_video
from core.pose.engine import EngineSpec, create_engine
from core.pose.landmarks import LANDMARK_COUNT, landmarks_to_array
from core.pose.sequence import PoseSequence, SegmentWarmup, SourceMeta
from core.pose.validity import DEFAULT_VALID_CONF_THR, MEDIAPIPE_VALIDITY_POLICY

SEGMENT_WARMUP_FRAMES = 15
MIN_CPUS_FOR_RECORDING_PARALLEL = 8
ProgressFn = Callable[[int, int], None]
EngineFactory = Callable[[], object]
ParallelFlag = bool | Callable[[], bool]
# (media.Frame, pose_landmarks 或 None)，顺序跟踪时逐帧回调
FrameHook = Callable[[object, object], None]


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
        DecodeOptions(
            sample_fps=sample_fps or None,
            max_edge=max_edge,
            timestamps="round_monotonic",
            strict=True,
        ),
        lazy_engine=True,
        download_models=False,
    )


PRACTICE_V4 = practice_v4_profile(sample_fps=0.0, max_edge=1920)


def formal_engine(pose_variant: str) -> EngineSpec:
    """正式评估与模板特征：VIDEO 模式、不跑手部。"""
    return EngineSpec(pose_variant=pose_variant, running_mode="video", enable_hands=False)


def practice_engine(pose_variant: str) -> EngineSpec:
    """练习分析：CPU、VIDEO 模式、不跑手部。"""
    return EngineSpec(
        pose_variant=pose_variant, enable_hands=False, delegate="cpu", running_mode="video"
    )


# --------------------------------------------------------------------------- #
# 作用域缓存
# --------------------------------------------------------------------------- #

_CACHE: contextvars.ContextVar[dict | None] = contextvars.ContextVar("pose_cache", default=None)


@contextmanager
def pose_cache() -> Iterator[None]:
    """作用域内缓存提取结果；离开作用域即丢弃。只对当前线程（上下文）生效。

    可重入：已在作用域内时沿用外层缓存，评分函数自带的作用域不会切断调用方更大范围的复用。
    """
    if _CACHE.get() is not None:
        yield
        return
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


def _prefix_safe(decode) -> bool:
    """前缀复用只在"第 t 行 = 第 t 帧"成立时安全：从 0 开始、不抽帧、不定位、不缩放（FORMAL 这类读法）。

    其余读法（``start>0``、抽帧、``seek``、缩放）行号与帧号不再一一对应，``prefix(end)`` 截的是
    "前 end+1 行"而不是"读到第 end 帧"，这类请求只认终点相同的命中（见 ``_cache_lookup``）。
    """
    return (
        decode.start == 0 and not decode.sample_fps and not decode.seek and decode.max_edge is None
    )


def _cache_lookup(store: dict, key: tuple, end: int | None) -> PoseSequence | None:
    """顺序跟踪是因果的：读到更远（或读到结尾）的结果截前缀即等于只读到 ``end`` 的结果。

    ``reached_eof`` 必须由 ``VideoSource.exhausted`` 判定（读不到帧才算结尾），不能从行数反推：
    分段结果的行数等于容器帧数但末尾可能是未解码的占位行；抽帧结果的行数远小于帧号，行数巧合
    等于 ``end`` 不代表读到了第 ``end`` 帧。
    """
    decode = key[3]  # _cache_key 里 replace(profile.decode, end=None) 之后的 DecodeOptions
    sequential = key[-1][0] == "sequential"
    safe_prefix = sequential and _prefix_safe(decode)
    for cached_end, reached_eof, seq in store.get(key, ()):
        if cached_end == end or (end is None and reached_eof):
            return seq
        if (
            safe_prefix
            and end is not None
            and (reached_eof or (cached_end is not None and cached_end >= end))
        ):
            return seq.prefix(end)
    return None


# --------------------------------------------------------------------------- #
# 提取
# --------------------------------------------------------------------------- #


def _close(engine) -> None:
    close = getattr(engine, "close", None)
    if callable(close):
        close()


def _meta(
    source_meta,
    engine: EngineSpec,
    profile: ExtractionProfile,
    tracking: str,
    *,
    frames_scanned: int = 0,
    stride: int = 1,
    inference_size=(0, 0),
) -> SourceMeta:
    return SourceMeta(
        video=str(source_meta.path),
        fps=source_meta.fps,
        fps_reported=source_meta.fps_reported,
        frame_count=source_meta.frame_count,
        width=source_meta.width,
        height=source_meta.height,
        engine=engine,
        profile=profile.name,
        tracking=tracking,
        validity_policy=MEDIAPIPE_VALIDITY_POLICY,
        valid_conf_thr=float(DEFAULT_VALID_CONF_THR),
        frames_scanned=frames_scanned,
        stride=stride,
        inference_size=inference_size,
    )


def _sequential(
    video: Path,
    engine: EngineSpec,
    profile: ExtractionProfile,
    factory: EngineFactory,
    cancel: CancelToken | None,
    progress: ProgressFn | None,
    on_frame: FrameHook | None = None,
    *,
    infer_gate: _InferGate | None = None,
) -> tuple[PoseSequence, bool]:
    """返回 ``(序列, exhausted)``；``exhausted`` 为真表示因读不到帧结束，而非到达 ``decode.end``。"""
    rows: list[np.ndarray] = []
    detected, frame_index, t_ms, source_wh = [], [], [], []
    inference_size = (0, 0)
    pipe = None
    with open_video(video, profile.decode, cancel) as source:
        total = source.meta.frame_count
        try:
            if not profile.lazy_engine:
                if cancel is not None:
                    cancel.raise_if_cancelled()
                pipe = factory()
            for frame in source:
                if pipe is None:
                    pipe = factory()
                with infer_gate.acquire(cancel) if infer_gate is not None else nullcontext():
                    pose_landmarks, _hands = pipe.infer(frame.bgr, timestamp_ms=frame.t_ms)
                hit = pose_landmarks is not None
                row = landmarks_to_array(pose_landmarks) if hit else np.zeros((LANDMARK_COUNT, 4))
                rows.append(row)
                detected.append(hit)
                frame_index.append(frame.index)
                t_ms.append(frame.t_ms)
                source_wh.append(frame.source_size)
                if on_frame is not None:
                    on_frame(frame, pose_landmarks)
                shape = getattr(frame.bgr, "shape", None)
                if shape is not None:
                    inference_size = (int(shape[1]), int(shape[0]))
                done = len(rows)
                if progress is not None and ((done % 10 == 0) or (total > 0 and done == total)):
                    progress(done, total)
        finally:
            if pipe is not None:
                _close(pipe)
        meta = _meta(
            source.meta,
            engine,
            profile,
            "sequential",
            frames_scanned=source.frames_scanned,
            stride=source.stride(),
            inference_size=inference_size,
        )
        exhausted = source.exhausted
    n = len(rows)
    return (
        PoseSequence(
            landmarks=np.stack(rows, axis=0) if rows else np.zeros((0, LANDMARK_COUNT, 4)),
            detected=np.asarray(detected, dtype=bool),
            decoded=np.ones((n,), dtype=bool),
            frame_index=np.asarray(frame_index, dtype=np.int64),
            t_ms=np.asarray(t_ms, dtype=np.int64),
            source_wh=np.asarray(source_wh, dtype=np.int64).reshape(n, 2),
            meta=meta,
        ),
        exhausted,
    )


def _segments(total: int, workers: int) -> list[tuple[int, int]]:
    segs = []
    for k in range(workers):
        s = (k * total) // workers
        e = ((k + 1) * total) // workers - 1
        if e >= s:
            segs.append((s, e))
    return segs


def _run_segment(
    video: Path,
    seg_range: tuple[int, int],
    decode,
    factory: EngineFactory,
    shared: CancelToken,
    buffers: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    callbacks: tuple[Callable[[], None], Callable[[BaseException], None]],
    warmups: dict[int, SegmentWarmup],
) -> None:
    seg_start, seg_end = seg_range
    landmarks, detected, decoded, t_ms, source_wh = buffers
    advance, fail = callbacks
    warm_start = max(0, seg_start - SEGMENT_WARMUP_FRAMES)
    options = replace(decode, start=warm_start, end=seg_end, seek=True)
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
                    source_wh[frame.index] = frame.source_size
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
            seg_start,
            seg_end,
            np.stack(warm_rows, axis=0) if warm_rows else np.zeros((0, LANDMARK_COUNT, 4)),
            np.asarray(warm_hit, dtype=bool),
        )


def _segmented(
    video: Path,
    engine: EngineSpec,
    profile: ExtractionProfile,
    factory: EngineFactory,
    workers: int,
    source_meta,
    cancel: CancelToken | None,
    progress: ProgressFn | None,
) -> PoseSequence:
    total = source_meta.frame_count
    landmarks = np.zeros((total, LANDMARK_COUNT, 4))
    detected = np.zeros((total,), dtype=bool)
    decoded = np.zeros((total,), dtype=bool)
    t_ms = np.zeros((total,), dtype=np.int64)
    source_wh = np.zeros((total, 2), dtype=np.int64)
    buffers = (landmarks, detected, decoded, t_ms, source_wh)
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

    callbacks = (advance, fail)
    group = TaskGroup("pose-extract")
    handles = [
        group.spawn(
            f"segment-{s}",
            lambda _t, r=(s, e): _run_segment(
                video, r, profile.decode, factory, shared, buffers, callbacks, warmups
            ),
            owner="extract_pose_sequence",
        )
        for s, e in _segments(total, workers)
    ]
    for handle in handles:
        handle.join()
    if errors:
        raise errors[0]
    shared.raise_if_cancelled()
    return PoseSequence(
        landmarks=landmarks,
        detected=detected,
        decoded=decoded,
        frame_index=np.arange(total, dtype=np.int64),
        t_ms=t_ms,
        source_wh=source_wh,
        meta=_meta(source_meta, engine, profile, "segmented"),
        segments=tuple(warmups[s] for s in sorted(warmups)),
    )


def _segmentable(decode) -> bool:
    """分段方案（每段按 ``seg_start``/``seg_end`` 定位读取）无法保持 ``start``/``end``/``sample_fps``
    的语义：定位是按分段边界算的，不是按 ``decode.start``；抽帧的步长、保底末帧也是整段才有意义。
    这些选项非默认值时退回顺序跟踪，和"帧数未知退回顺序"一样处理（§6.3 实现补充）。
    """
    return decode.start == 0 and decode.end is None and not decode.sample_fps


def extract_pose_sequence(
    video: Path | str,
    engine: EngineSpec,
    profile: ExtractionProfile = FORMAL,
    *,
    workers: int = 1,
    cancel: CancelToken | None = None,
    progress: ProgressFn | None = None,
    engine_factory: EngineFactory | None = None,
    on_frame: FrameHook | None = None,
) -> PoseSequence:
    """解码并逐帧推理，返回 ``PoseSequence``。取消时抛 ``Cancelled``；视频打不开抛 ``VideoOpenError``。

    ``workers > 1`` 且容器给出帧数、且 ``profile.decode`` 的 ``start``/``end``/``sample_fps`` 都是默认值时
    分段并行（``tracking="segmented"``）；否则顺序跟踪——分段方案按分段边界定位读取，无法保持这些选项
    的语义（§6.3 实现补充）。``engine_factory`` 用于注入假引擎（测试、练习分析的 ``pipeline_factory``）；
    注入时不走缓存。``on_frame`` 在每帧推理后拿到解码帧与关键点（如边提取边写骨架视频，免得再解码一遍）；
    给了它就顺序跟踪、不走缓存，保证每帧都回调。模型目录只解析一次再交给各段的引擎（冻结包首次运行时
    避免多个分段线程各自解析、同时播种模型）。
    """
    video = Path(str(video))
    workers = 1 if on_frame is not None else max(1, int(workers))
    factory = _resolve_factory(engine, profile, engine_factory)
    store = _CACHE.get() if engine_factory is None and on_frame is None else None
    source_meta = None
    if workers > 1 and _segmentable(profile.decode):
        source_meta = probe_video(video)
        if source_meta.frame_count <= 0:
            workers = 1  # 帧数未知无法分段，退回顺序跟踪
    else:
        workers = 1  # start/end/sample_fps 非默认值，分段保不住语义，退回顺序跟踪
    key = _cache_key(video, engine, profile, workers) if store is not None else None
    if store is not None:
        hit = _cache_lookup(store, key, profile.decode.end)
        if hit is not None:
            return hit
    if workers > 1:
        seq = _segmented(video, engine, profile, factory, workers, source_meta, cancel, progress)
        reached_eof = True  # 分段方案要求容器给出帧数，覆盖全部帧即读到结尾
    else:
        seq, reached_eof = _sequential(video, engine, profile, factory, cancel, progress, on_frame)
    if store is not None:
        store.setdefault(key, []).append((profile.decode.end, reached_eof, seq))
    return seq


def can_parallel_pose_pair(
    *, foreground_recording: bool = False, cpu_count: int | None = None
) -> bool:
    """判断当前是否允许双摄两路并发提点。

    前台未在实时录制时允许并发；若前台正在实时录制，则仅在逻辑核数 ``>= 8`` 时并发，
    弱 CPU 考场机自动退回单路串行，避免抢占前台 30fps 录像算力。
    """
    if not foreground_recording:
        return True
    if cpu_count is None:
        cpus = os.cpu_count()
    else:
        if isinstance(cpu_count, bool) or not isinstance(cpu_count, int):
            raise TypeError(f"cpu_count 必须为 int 或 None，实际 {type(cpu_count).__name__}")
        cpus = cpu_count
    return (cpus or 0) >= MIN_CPUS_FOR_RECORDING_PARALLEL


def _same_video_path(first: Path, second: Path) -> bool:
    """判断两路路径是否指向同一物理文件（或同一虚拟回放键）。"""
    if str(first) == str(second):
        return True
    try:
        return first.samefile(second)
    except OSError:
        return _file_key(first) == _file_key(second)


def _parallel_enabled(parallel: ParallelFlag) -> bool:
    return bool(parallel()) if callable(parallel) else bool(parallel)


def _record_elapsed(elapsed_out: list[float] | None, first_s: float, second_s: float) -> None:
    if elapsed_out is not None:
        elapsed_out[0] = float(first_s)
        elapsed_out[1] = float(second_s)


def _resolve_factory(
    engine: EngineSpec, profile: ExtractionProfile, engine_factory: EngineFactory | None
) -> EngineFactory:
    if engine_factory is not None:
        return engine_factory
    directory = default_models_dir()
    return lambda: create_engine(engine, models_dir=directory, download=profile.download_models)


def _mirror_cache_entries(store: dict, src_key: tuple, dst_key: tuple) -> None:
    """把硬链接同一物理文件的缓存项同步登记到另一路径键，避免下游按另一路径重提。"""
    if dst_key == src_key or src_key not in store:
        return
    dst = store.setdefault(dst_key, [])
    existing = {(end, eof, id(seq)) for end, eof, seq in dst}
    for entry in store[src_key]:
        sig = (entry[0], entry[1], id(entry[2]))
        if sig not in existing:
            dst.append(entry)
            existing.add(sig)


class _DualProgress:
    """双路进度聚合器（并行与串行共用）：两路总帧数均确认 > 0 且未越界前报不定进度（total=0）。"""

    def __init__(self, progress: ProgressFn) -> None:
        self._progress = progress
        self._lock = threading.Lock()
        self._done, self._total, self._last_done = [0, 0], [0, 0], 0

    def for_slot(self, slot: int) -> ProgressFn:
        def report(done: int, total: int) -> None:
            with self._lock:
                done_i, total_i = int(done), int(total)
                self._done[slot] = max(self._done[slot], done_i)
                if total_i <= 0 or done_i > total_i:
                    self._total[slot] = -1
                elif self._total[slot] >= 0:
                    self._total[slot] = total_i
                combined_done = max(self._last_done, self._done[0] + self._done[1])
                self._last_done = combined_done
                combined_total = self._total[0] + self._total[1]
                valid = (
                    self._total[0] > 0 and self._total[1] > 0 and combined_done <= combined_total
                )
                self._progress(combined_done, combined_total if valid else 0)

        return report


class _InferGate:
    """动态并发推理闸门：谓词为真时双路并发直通，切为假时等在途帧退出且同时至多 1 路推理。"""

    def __init__(self, allow_parallel: Callable[[], bool]) -> None:
        self._allow_parallel = allow_parallel
        self._cond = threading.Condition(threading.Lock())
        self._active = 0

    @contextmanager
    def acquire(self, cancel: CancelToken | None) -> Iterator[None]:
        with self._cond:
            while self._active > 0 and not self._allow_parallel():
                if cancel is not None:
                    cancel.raise_if_cancelled()
                self._cond.wait(0.05)
            if cancel is not None:
                cancel.raise_if_cancelled()
            self._active += 1
        try:
            yield
        finally:
            with self._cond:
                self._active -= 1
                self._cond.notify_all()


def _ordered_factories(
    factory: EngineFactory, shared: CancelToken
) -> tuple[EngineFactory, EngineFactory, threading.Event, list[float]]:
    """保证 slot 0 首次调用注入工厂（或 0 帧提前结束）严格早于 slot 1，首帧后两路无锁并行。"""
    lock = threading.Lock()
    first_done = threading.Event()
    wait_s = [0.0]

    def first() -> object:
        try:
            with lock:
                return factory()
        finally:
            first_done.set()

    def second() -> object:
        t_wait = time.monotonic()
        while not first_done.wait(0.05):
            shared.raise_if_cancelled()
        shared.raise_if_cancelled()
        with lock:
            wait_s[0] += time.monotonic() - t_wait
            return factory()

    return first, second, first_done, wait_s


@dataclass(frozen=True)
class _PairPlan:
    engine: EngineSpec
    profile: ExtractionProfile
    factory: EngineFactory
    store: dict | None
    keys: tuple[tuple | None, tuple | None]
    cancel: CancelToken | None
    progress: ProgressFn | None
    elapsed_out: list[float] | None

    def save(self, slot: int, eof: bool, seq: PoseSequence) -> None:
        if self.store is not None and self.keys[slot] is not None:
            self.store.setdefault(self.keys[slot], []).append((self.profile.decode.end, eof, seq))


def _extract_pair_sequential(
    first: Path,
    second: Path,
    plan: _PairPlan,
    hits: tuple[PoseSequence | None, PoseSequence | None],
    *,
    same_file: bool,
) -> tuple[PoseSequence, PoseSequence]:
    hit1, hit2 = hits
    dual = (
        _DualProgress(plan.progress)
        if plan.progress is not None and not same_file and hit1 is None and hit2 is None
        else None
    )

    def run_one(
        video: Path, hit: PoseSequence | None, slot: int
    ) -> tuple[PoseSequence, bool, float]:
        if hit is not None:
            return hit, False, 0.0
        prog = dual.for_slot(slot) if dual is not None else plan.progress
        started = time.monotonic()
        seq, eof = _sequential(video, plan.engine, plan.profile, plan.factory, plan.cancel, prog)
        dur = time.monotonic() - started
        if plan.cancel is not None:
            plan.cancel.raise_if_cancelled()
        return seq, eof, dur

    seq1, eof1, elapsed1 = run_one(first, hit1, 0)
    if same_file:
        if hit1 is None:
            plan.save(0, eof1, seq1)
            if plan.store is not None and plan.keys[0] is not None and plan.keys[1] is not None:
                _mirror_cache_entries(plan.store, plan.keys[0], plan.keys[1])
        _record_elapsed(plan.elapsed_out, elapsed1, 0.0)
        return seq1, seq1

    seq2, eof2, elapsed2 = run_one(second, hit2, 1)
    if hit1 is None:
        plan.save(0, eof1, seq1)
    if hit2 is None:
        plan.save(1, eof2, seq2)
    _record_elapsed(plan.elapsed_out, elapsed1, elapsed2)
    return seq1, seq2


def _extract_pair_parallel(
    first: Path,
    second: Path,
    plan: _PairPlan,
    *,
    order_factory: bool,
    allow_parallel: Callable[[], bool] | None,
) -> tuple[PoseSequence, PoseSequence]:
    shared = plan.cancel.child() if plan.cancel is not None else CancelToken()
    results: list[PoseSequence | None] = [None, None]
    exhausted, elapsed = [False, False], [0.0, 0.0]
    errors: list[BaseException | None] = [None, None]
    dual = _DualProgress(plan.progress) if plan.progress is not None else None
    infer_gate = _InferGate(allow_parallel) if allow_parallel is not None else None
    if order_factory:
        f0, f1, first_done, second_wait = _ordered_factories(plan.factory, shared)
        factories: tuple[EngineFactory, EngineFactory] = (f0, f1)
    else:
        first_done, second_wait, factories = None, [0.0], (plan.factory, plan.factory)

    def run_view(slot: int, video: Path) -> None:
        try:
            started = time.monotonic()
            slot_prog = dual.for_slot(slot) if dual is not None else None
            seq, eof = _sequential(
                video,
                plan.engine,
                plan.profile,
                factories[slot],
                shared,
                slot_prog,
                infer_gate=infer_gate,
            )
            dur = time.monotonic() - started
            elapsed[slot] = max(0.0, dur - second_wait[0]) if slot == 1 else dur
            results[slot], exhausted[slot] = seq, eof
        except Cancelled as exc:
            shared.cancel(exc.reason or "cancelled")
        except BaseException as exc:
            errors[slot] = exc
            shared.cancel("error")
        finally:
            if slot == 0 and first_done is not None:
                first_done.set()

    group = TaskGroup("pose-pair")
    handles = []
    try:
        for slot, video in ((0, first), (1, second)):
            handles.append(
                group.spawn(
                    f"view-{slot}",
                    lambda _t, s=slot, v=video: run_view(s, v),
                    owner="extract_pose_pair",
                )
            )
    except BaseException:
        shared.cancel("error")
        for handle in handles:
            try:
                handle.join()
            except Cancelled:
                continue
        raise
    for handle in handles:
        handle.join()
    if plan.cancel is not None:
        plan.cancel.raise_if_cancelled()
    for exc in errors:
        if exc is not None:
            raise exc
    shared.raise_if_cancelled()
    if results[0] is None or results[1] is None:
        raise RuntimeError("双摄并行提点未返回完整结果")
    plan.save(0, exhausted[0], results[0])
    plan.save(1, exhausted[1], results[1])
    _record_elapsed(plan.elapsed_out, elapsed[0], elapsed[1])
    return results[0], results[1]


def extract_pose_pair(
    first_video: Path | str,
    second_video: Path | str,
    engine: EngineSpec,
    profile: ExtractionProfile = FORMAL,
    *,
    parallel: ParallelFlag = True,
    cancel: CancelToken | None = None,
    progress: ProgressFn | None = None,
    engine_factory: EngineFactory | None = None,
    elapsed_out: list[float] | None = None,
) -> tuple[PoseSequence, PoseSequence]:
    """双摄正/侧两路视频提点唯一公共原语（固定 ``workers=1`` 顺序跟踪，与串行 0 ULP 一致）。

    先查当前调用线程的 ``pose_cache``（含同物理文件硬链接双键互通）：若任一路已命中、两路指向
    同一物理文件或并发开关/谓词 ``parallel`` 为假，直接顺序提取不额外起线程；两路均未缓存且允许并发时
    通过 ``TaskGroup`` 并发跑两路独立引擎（``parallel`` 为动态回调时支持运行期逐帧退避至单路），
    完成后按精确缓存键写回父线程 ``pose_cache``，并可选将两路独立墙钟耗时写入 ``elapsed_out``。
    """
    if elapsed_out is not None and len(elapsed_out) < 2:
        raise ValueError(f"elapsed_out 长度至少为 2，实际 {len(elapsed_out)}")
    if cancel is not None:
        cancel.raise_if_cancelled()
    first, second = Path(str(first_video)), Path(str(second_video))
    same_file = _same_video_path(first, second)
    store = _CACHE.get() if engine_factory is None else None
    key1 = key2 = hit1 = hit2 = None
    if store is not None:
        key1 = _cache_key(first, engine, profile, 1)
        key2 = key1 if str(first) == str(second) else _cache_key(second, engine, profile, 1)
        hit1 = _cache_lookup(store, key1, profile.decode.end)
        if same_file:
            if hit1 is None and key2 != key1:
                hit1 = _cache_lookup(store, key2, profile.decode.end)
                if hit1 is not None:
                    _mirror_cache_entries(store, key2, key1)
            elif hit1 is not None:
                _mirror_cache_entries(store, key1, key2)
            if hit1 is not None:
                _record_elapsed(elapsed_out, 0.0, 0.0)
                return hit1, hit1
        else:
            hit2 = _cache_lookup(store, key2, profile.decode.end)
            if hit1 is not None and hit2 is not None:
                _record_elapsed(elapsed_out, 0.0, 0.0)
                return hit1, hit2
    factory = _resolve_factory(engine, profile, engine_factory)
    plan = _PairPlan(engine, profile, factory, store, (key1, key2), cancel, progress, elapsed_out)
    if (not _parallel_enabled(parallel)) or same_file or (hit1 is not None) or (hit2 is not None):
        return _extract_pair_sequential(first, second, plan, (hit1, hit2), same_file=same_file)
    allow = parallel if callable(parallel) else None
    return _extract_pair_parallel(
        first, second, plan, order_factory=engine_factory is not None, allow_parallel=allow
    )
