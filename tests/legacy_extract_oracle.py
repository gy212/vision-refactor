# -*- coding: utf-8 -*-
"""P2 差分测试的旧实现快照（只读 oracle）：迁移前 4 个"解码→推理"循环，逐字取自 main@5721f1d。

只在 tests/ 中使用，配合 golden harness 回放，证明新的 ``extract_pose_sequence`` 与旧循环逐元素一致。
P9 随门面一并删除。
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable

import cv2
import numpy as np

from core.pose.layouts import POSE33_V3, FeatureLayoutSpec
from core.pose.normalize import normalize_pose_xy
from core.pose.validity import DEFAULT_VALID_CONF_THR, MEDIAPIPE_VALIDITY_POLICY, derive_valid_mask
from core.pose.view_score import pose_view_score
from tests.golden_harness import _FakePipeline as MediaPipePipeline


def PipelineConfig(**kwargs):  # noqa: N802 - 旧代码的构造签名
    return kwargs


def models_dir() -> Path:
    return Path("models")


ProgressCb = Callable[[str, int, int], None]


def _extract_pose_features(
    video_path: Path,
    *,
    pose_variant: str,
    workers: int = 1,
    normalizer: Callable = normalize_pose_xy,
    compute_view: bool = False,
    layout: FeatureLayoutSpec = POSE33_V3,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """
    Extract normalized pose features for the entire video.
    Returns (features[T,J,2], fps, view_scores[T] or None), where J/D come from `layout`.

    本期默认 `layout=POSE33_V3`（即 (22,2)），行为与旧路径一致；缺帧补零按 layout
    shape 生成，不再写死 (22,2)。
    """
    stop_evt = stop_evt or Event()
    zero_frame_shape = tuple(int(x) for x in layout.shape)
    workers = max(1, int(workers))
    cap = cv2.VideoCapture(str(video_path))  # golden harness 替换为回放
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"无法打开视频：{video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    except BaseException:
        cap.release()
        raise

    def _emit(done: int) -> None:
        if progress_cb is not None:
            progress_cb("提取骨架特征", done, total)

    # Single-thread path: VIDEO mode (more stable landmarks).
    if workers == 1:
        pipe: MediaPipePipeline | None = None
        try:
            models_dir_path = models_dir()
            pipe = MediaPipePipeline(
                models_dir=models_dir_path,
                cfg=PipelineConfig(pose_variant=pose_variant, running_mode="video", enable_hands=False),
            )
            feats: list[np.ndarray] = []
            views: list[float] | None = [] if compute_view else None
            last_view = 0.0
            i = 0
            while not stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                ts = int(i * 1000.0 / fps)
                pose_landmarks, _hands = pipe.infer(frame, timestamp_ms=ts)
                f = normalizer(pose_landmarks)
                if f is None:
                    if feats:
                        f = feats[-1].copy()
                    else:
                        f = np.zeros(zero_frame_shape, dtype=np.float32)
                feats.append(f)
                if views is not None:
                    v = pose_view_score(pose_landmarks)
                    if v is None:
                        v = last_view
                    else:
                        last_view = float(v)
                    views.append(float(v))
                i += 1
                if (i % 10 == 0) or (total > 0 and i == total):
                    _emit(i)

            if stop_evt.is_set():
                raise InterruptedError("姿态特征提取已取消")
            if not feats:
                raise RuntimeError("视频为空或无法读取帧")
            view_arr = np.array(views, dtype=np.float32) if views is not None else None
            return np.stack(feats, axis=0), fps, view_arr
        finally:
            try:
                cap.release()
            finally:
                if pipe is not None:
                    close = getattr(pipe, "close", None)
                    if callable(close):
                        close()

    # Multi-thread path: split the video into segments and process each segment sequentially
    # with VIDEO mode (tracking enabled within each segment). This keeps features comparable
    # to the template (VIDEO mode) and avoids the huge mismatch caused by IMAGE mode.
    if total <= 0:
        # Can't segment without a known frame count; fall back to single-thread.
        cap.release()
        return _extract_pose_features(
            video_path,
            pose_variant=pose_variant,
            workers=1,
            normalizer=normalizer,
            compute_view=compute_view,
            layout=layout,
            progress_cb=progress_cb,
            stop_evt=stop_evt,
        )

    cap.release()

    models_dir_path = models_dir()
    feat_arr = np.zeros((total, *zero_frame_shape), dtype=np.float32)
    view_arr = np.zeros((total,), dtype=np.float32) if compute_view else None

    done_lock = Lock()
    error_lock = Lock()
    worker_errors: list[BaseException] = []
    done = 0

    def _inc_done(n: int = 1) -> None:
        nonlocal done
        with done_lock:
            done += n
            d = done
        if (d % 10 == 0) or d == total:
            _emit(d)

    overlap = 15  # warm-up frames per segment to stabilize tracking

    def _record_worker_error(exc: BaseException) -> None:
        with error_lock:
            worker_errors.append(exc)
        stop_evt.set()

    def seg_worker(seg_start: int, seg_end: int) -> None:
        cap2: cv2.VideoCapture | None = None
        pipe: MediaPipePipeline | None = None
        try:
            if stop_evt.is_set():
                return
            warm_start = max(0, seg_start - overlap)
            cap2 = cv2.VideoCapture(str(video_path))
            if not cap2.isOpened():
                raise RuntimeError(f"无法打开视频分段：{video_path}")
            # Seek to warm_start; if seek fails, OpenCV will usually continue from 0.
            cap2.set(cv2.CAP_PROP_POS_FRAMES, float(warm_start))

            pipe = MediaPipePipeline(
                models_dir=models_dir_path,
                cfg=PipelineConfig(
                    pose_variant=pose_variant,
                    running_mode="video",
                    enable_hands=False,
                ),
            )

            last: np.ndarray | None = None
            last_view: float | None = None
            i = warm_start
            while (not stop_evt.is_set()) and i <= seg_end:
                ok, frame = cap2.read()
                if not ok:
                    break
                ts = int(i * 1000.0 / fps)
                pose_landmarks, _hands = pipe.infer(frame, timestamp_ms=ts)
                f = normalizer(pose_landmarks)
                if f is None:
                    f = (
                        last.copy()
                        if last is not None
                        else np.zeros(zero_frame_shape, dtype=np.float32)
                    )
                else:
                    last = f

                if i >= seg_start:
                    feat_arr[i] = f
                    if view_arr is not None:
                        v = pose_view_score(pose_landmarks)
                        if v is None:
                            v = float(last_view or 0.0)
                        else:
                            last_view = float(v)
                        view_arr[i] = float(v)
                    _inc_done(1)
                i += 1
        except BaseException as exc:  # thread failures must reach the caller
            _record_worker_error(exc)
        finally:
            if cap2 is not None:
                try:
                    cap2.release()
                except BaseException as exc:
                    _record_worker_error(exc)
            if pipe is not None:
                close = getattr(pipe, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException as exc:
                        _record_worker_error(exc)

    # Partition [0, total) across workers
    segs: list[tuple[int, int]] = []
    for k in range(workers):
        s = (k * total) // workers
        e = ((k + 1) * total) // workers - 1
        if e >= s:
            segs.append((s, e))

    threads = [Thread(target=seg_worker, args=segs[i], daemon=True) for i in range(len(segs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if worker_errors:
        raise worker_errors[0]
    if stop_evt.is_set():
        raise InterruptedError("姿态特征提取已取消")
    if done == 0:
        raise RuntimeError("视频为空或无法读取帧")
    return feat_arr, fps, view_arr


def _extract_pose_raw_prefix(
    video_path: Path,
    *,
    pose_variant: str,
    end_frame: int | None = None,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))  # golden harness 替换为回放
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    end_i = int(end_frame) if end_frame is not None else (n_frames - 1 if n_frames > 0 else 10**9)
    end_i = max(0, end_i)

    models_dir_path = models_dir()
    pipe = MediaPipePipeline(
        models_dir=models_dir_path,
        cfg=PipelineConfig(pose_variant=pose_variant, running_mode="video", enable_hands=False),
    )

    out: list[np.ndarray] = []
    last: np.ndarray | None = None

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(i * 1000.0 / fps)
        pose_landmarks, _hands = pipe.infer(frame, timestamp_ms=ts)
        if pose_landmarks is None:
            arr = last.copy() if last is not None else np.zeros((33, 4), dtype=np.float32)
        else:
            arr = np.zeros((33, 4), dtype=np.float32)
            for j in range(min(33, len(pose_landmarks))):
                lm = pose_landmarks[j]
                arr[j, 0] = float(getattr(lm, "x", 0.0))
                arr[j, 1] = float(getattr(lm, "y", 0.0))
                arr[j, 2] = float(getattr(lm, "z", 0.0))
                arr[j, 3] = float(getattr(lm, "visibility", 0.0))
            last = arr

        out.append(arr)
        if i >= end_i:
            break
        i += 1

    cap.release()
    if not out:
        raise RuntimeError(f"未能从视频提取姿态：{video_path}")

    landmarks = np.stack(out, axis=0).astype(np.float32, copy=False)
    return landmarks


def extract_pose_and_view_scores(
    video_path: Path,
    *,
    pose_variant: str,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cap = cv2.VideoCapture(str(video_path))  # golden harness 替换为回放
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    start_i = int(start_frame) if start_frame is not None else 0
    end_i = int(end_frame) if end_frame is not None else (n_frames - 1 if n_frames > 0 else 10**9)
    start_i = max(0, start_i)
    end_i = max(start_i, end_i)

    models_dir_path = models_dir()
    pipe = MediaPipePipeline(
        models_dir=models_dir_path,
        cfg=PipelineConfig(pose_variant=pose_variant, running_mode="video", enable_hands=False),
    )

    lms: list[np.ndarray] = []
    vs: list[float] = []

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts = int(i * 1000.0 / fps)
        pose_landmarks, _hands = pipe.infer(frame, timestamp_ms=ts)

        if start_i <= i <= end_i:
            if pose_landmarks is None:
                arr = np.zeros((33, 4), dtype=np.float32)
                lms.append(arr)
                vs.append(float("nan"))
            else:
                arr = np.zeros((33, 4), dtype=np.float32)
                for j in range(min(33, len(pose_landmarks))):
                    lm = pose_landmarks[j]
                    arr[j, 0] = float(getattr(lm, "x", 0.0))
                    arr[j, 1] = float(getattr(lm, "y", 0.0))
                    arr[j, 2] = float(getattr(lm, "z", 0.0))
                    arr[j, 3] = float(getattr(lm, "visibility", 0.0))
                lms.append(arr)
                s = pose_view_score(pose_landmarks)
                vs.append(float("nan") if (s is None or not np.isfinite(s)) else float(s))

        if i >= end_i:
            break
        i += 1

    cap.release()
    if not lms:
        raise RuntimeError(f"未能从视频提取姿态：{video_path}")

    meta = {
        "video": str(video_path),
        "fps": float(fps),
        "frame_count": int(n_frames),
        "width": int(w),
        "height": int(h),
        "pose_variant": str(pose_variant),
        "start_frame": int(start_i),
        "end_frame": int(start_i + len(lms) - 1),
        "landmark_layout": "pose33_normalized_xyzw(visibility)",
        # valid_mask 契约（YOLO 迁移 Issue #5）：声明有效性策略与阈值，便于下游区分
        # 可信/待标定。本入口的 meta 会被序列化进 JSON 报告，故只落地标量字段；
        # valid_mask 由下游按同一 `valid_conf_thr` 现场推导（逐位等价）。
        "validity_policy": MEDIAPIPE_VALIDITY_POLICY,
        "valid_conf_thr": float(DEFAULT_VALID_CONF_THR),
    }
    return np.stack(lms, axis=0), np.asarray(vs, dtype=np.float32), meta


def extract_series(path: Path, cfg, stopped: Callable[[], bool], pipeline_factory=None) -> dict:
    """默认原帧率逐帧推理，只保留关键点。丢失帧全无效，绝不复用上一帧。"""
    started = time.monotonic()
    cap = cv2.VideoCapture(str(path))
    pipe = None
    try:
        if not cap.isOpened():
            raise ValueError(f"视频无法读取，请重新录制：{path.name}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("视频帧率不可用，请重新录制")
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total = int(total) if math.isfinite(total) else 0
        stride = max(1, math.ceil(fps / cfg.sample_fps)) if cfg.sample_fps else 1
        points, frames, times = [], [], []
        index = 0
        timestamp = -1
        while cap.grab():
            if stopped():
                raise InterruptedError("分析已取消")
            if index % stride == 0 or index == total - 1:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    raise ValueError("视频存在不可解码的帧，请重新录制")
                h, w = frame.shape[:2]
                if max(h, w) > cfg.max_edge:
                    scale = cfg.max_edge / max(h, w)
                    frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))))
                if pipe is None:
                    pipe = pipeline_factory()
                timestamp = max(timestamp + 1, round(index * 1000 / fps))
                landmarks, _ = pipe.infer(frame, timestamp_ms=timestamp)
                arr = np.zeros((33, 4), dtype=np.float32)
                if landmarks is not None:
                    for j, lm in enumerate(landmarks[:33]):
                        # x和z统一为图像高度单位，防止非正方形画面把角度算歪。
                        arr[j] = (lm.x * w / h, lm.y, lm.z * w / h, getattr(lm, "visibility", 0))
                points.append(arr)
                frames.append(index)
                times.append(index / fps)
            index += 1
        if not points:
            raise ValueError("视频没有可读取的帧，请重新录制")
        lm = np.stack(points)
        mask = derive_valid_mask(lm) & np.isfinite(lm[:, :, :3]).all(axis=2)
        mask &= (lm[:, :, 0] >= 0) & (lm[:, :, 0] <= w / h) & (lm[:, :, 1] >= 0) & (lm[:, :, 1] <= 1)
        return {"landmarks": lm, "valid_mask": mask, "frames": frames, "times": times,
                "capture": {"sourceFps": fps, "sourceFrames": index, "analyzedFrames": len(frames),
                            "stride": stride, "width": w, "height": h,
                            "inferenceWidth": frame.shape[1], "inferenceHeight": frame.shape[0],
                            "analysisSeconds": round(time.monotonic() - started, 3)}}
    finally:
        cap.release()
        if pipe is not None:
            pipe.close()
