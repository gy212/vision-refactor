# -*- coding: utf-8 -*-
"""样本姿态缓存：按清单逐视频提取 MediaPipe 正式路径姿态（heavy、FORMAL 顺序跟踪），多进程、可续跑。

每段视频一个 ``<pair_id>__<view>.npz``：``landmarks``（float32 归一化坐标）、``detected``、``decoded``、
``frame_index``、``t_ms``、``source_wh`` 与 JSON 字符串 ``meta``。缓存键（版本、模型、旋转、视频相对路径与字节数）
与请求不一致的文件视为缺失、重新提取；写入经原子替换，中断不留半成品。每次提取的结果追加到 ``cache_log.jsonl``。
读回时重建 ``PoseSequence``，规则输入与练习分析 ``extract_series`` 同口径：``raw(ZERO, ASPECT_CORRECTED)`` + ``rule_series``。
"""
from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.sample_eval.manifest import SamplePair
from core.evaluation import rule_series
from core.foundation import atomic_target, load_npz, models_dir
from core.media import apply_rotation, rotated_size
from core.pose import (
    FORMAL,
    Coords,
    GapPolicy,
    PoseSequence,
    SourceMeta,
    create_engine,
    extract_pose_sequence,
    formal_engine,
    model_available,
    pose_model_path,
)

CACHE_VERSION = 1
POSE_VARIANT = "heavy"
VIEWS = ("front", "side")
KEY_FIELDS = ("cache_version", "pose_variant", "rotation", "video_rel", "video_bytes")


@dataclass(frozen=True)
class CacheJob:
    """一段视频的提取任务（参数可 pickle，供子进程使用）。"""

    pair_id: str
    view: str
    video: str
    video_rel: str
    out: str
    rotation: int
    models_dir: str = ""
    pose_variant: str = POSE_VARIANT

    def key(self) -> dict:
        return {"cache_version": CACHE_VERSION, "pose_variant": self.pose_variant, "rotation": self.rotation,
                "video_rel": self.video_rel, "video_bytes": Path(self.video).stat().st_size}


class RotatingEngine:
    """推理前把帧顺时针旋转（录反方向的样本专用），不改生产路径。"""

    def __init__(self, inner, degrees: int) -> None:
        self._inner = inner
        self._degrees = degrees

    def infer(self, frame_bgr, *, timestamp_ms=None):
        return self._inner.infer(apply_rotation(frame_bgr, self._degrees), timestamp_ms=timestamp_ms)

    def close(self) -> None:
        self._inner.close()


def cache_path(cache_dir: Path, pair_id: str, view: str) -> Path:
    return Path(cache_dir) / f"{pair_id}__{view}.npz"


def _read_meta(arrays: dict) -> dict:
    return json.loads(str(arrays["meta"]))


def is_fresh(job: CacheJob) -> bool:
    """缓存存在且缓存键与任务一致。"""
    path = Path(job.out)
    if not path.is_file():
        return False
    try:
        meta = _read_meta(load_npz(path))
    except (OSError, ValueError, KeyError):
        return False
    key = job.key()
    return all(meta.get(name) == key[name] for name in KEY_FIELDS)


def require_model(models: Path, pose_variant: str = POSE_VARIANT) -> None:
    """提取前在主进程确认模型已就位：子进程一律不下载，避免多个进程同时下载写坏模型文件。"""
    if not model_available(pose_variant, models):
        raise SystemExit(f"缺少 MediaPipe {pose_variant} 模型：{pose_model_path(pose_variant, models)}。"
                         "请先在桌面程序的模型管理中安装，或用 --models-dir 指向已有模型的目录")


def plan_jobs(pairs: list[SamplePair], samples_root: Path, cache_dir: Path,
              models: Path | None = None) -> tuple[list[CacheJob], list[str]]:
    """要评估的对（``rule_actions`` 非空）的全部视频任务，以及缺失视频的说明。"""
    models = Path(models) if models is not None else models_dir()
    jobs, missing = [], []
    for pair in pairs:
        if not pair.rule_actions:
            continue
        for view in VIEWS:
            video = pair.video(samples_root, view)
            if not video.is_file():
                missing.append(f"{pair.pair_id}/{view}: {video}")
                continue
            jobs.append(CacheJob(pair.pair_id, view, str(video), f"{pair.segment_dir}/{view}.mp4",
                                 str(cache_path(cache_dir, pair.pair_id, view)),
                                 pair.front_rotation if view == "front" else 0, str(models)))
    return jobs, missing


def _meta_dict(job: CacheJob, seq: PoseSequence, seconds: float) -> dict:
    m = seq.meta
    width, height = rotated_size(m.width, m.height, job.rotation)
    return {**job.key(), "video": job.video, "fps": m.fps, "fps_reported": m.fps_reported,
            "frame_count": m.frame_count, "width": width, "height": height, "frames_scanned": m.frames_scanned,
            "stride": m.stride, "inference_size": list(m.inference_size), "profile": m.profile,
            "tracking": m.tracking, "validity_policy": m.validity_policy, "valid_conf_thr": m.valid_conf_thr,
            "infer_sec": round(seconds, 2)}


def extract_one(job: CacheJob) -> dict:
    """子进程入口（模块顶层、参数可 pickle）：提取一段视频并原子写入缓存。"""
    started = time.perf_counter()
    spec = formal_engine(job.pose_variant)

    def factory():
        engine = create_engine(spec, models_dir=Path(job.models_dir) if job.models_dir else models_dir(), download=False)
        return RotatingEngine(engine, job.rotation) if job.rotation else engine

    seq = extract_pose_sequence(Path(job.video), spec, FORMAL, engine_factory=factory)
    seconds = time.perf_counter() - started
    source_wh = seq.source_wh[:, ::-1].copy() if job.rotation in (90, 270) else seq.source_wh
    meta = _meta_dict(job, seq, seconds)
    Path(job.out).parent.mkdir(parents=True, exist_ok=True)
    with atomic_target(job.out) as tmp, open(tmp, "wb") as fh:
        np.savez_compressed(fh, landmarks=seq.landmarks.astype(np.float32), detected=seq.detected,
                            decoded=seq.decoded, frame_index=seq.frame_index, t_ms=seq.t_ms, source_wh=source_wh,
                            meta=np.array(json.dumps(meta, ensure_ascii=False)))
    return {"frames": len(seq), "detected": int(np.count_nonzero(seq.detected)), "infer_sec": round(seconds, 1)}


def build_cache(jobs: list[CacheJob], cache_dir: Path, workers: int,
                echo: Callable[[str], None] = print) -> dict[str, int]:
    """提取缺失或过期的缓存；长视频先跑。返回 ``{"reused", "extracted", "failed"}`` 计数，失败写进日志不中断。"""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    todo = sorted((j for j in jobs if not is_fresh(j)), key=lambda j: Path(j.video).stat().st_size, reverse=True)
    counts = {"reused": len(jobs) - len(todo), "extracted": 0, "failed": 0}
    if not todo:
        return counts
    require_model(Path(todo[0].models_dir) if todo[0].models_dir else models_dir(), todo[0].pose_variant)
    echo(f"姿态缓存：需提取 {len(todo)} 段，复用 {counts['reused']} 段，{workers} 进程")
    started = time.time()
    with (Path(cache_dir) / "cache_log.jsonl").open("a", encoding="utf-8") as log, \
            ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(extract_one, job): job for job in todo}
        for done, fut in enumerate(as_completed(futures), start=1):
            job = futures[fut]
            rec = {"pair_id": job.pair_id, "view": job.view, "video": job.video, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            try:
                rec.update(status="ok", **fut.result())
                counts["extracted"] += 1
            except Exception as exc:  # noqa: BLE001 - 记录失败原因，继续其它视频
                rec.update(status="failed", error=repr(exc), tb=traceback.format_exc()[-1500:])
                counts["failed"] += 1
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            echo(f"[{done}/{len(todo)}] {time.time() - started:.0f}s {job.pair_id}/{job.view} {rec['status']}")
    return counts


def load_sequence(path: Path) -> PoseSequence:
    """缓存文件 → ``PoseSequence``（``meta.engine`` 只记模型名）。"""
    arrays = load_npz(path)
    meta = _read_meta(arrays)
    source = SourceMeta(video=meta["video"], fps=float(meta["fps"]), fps_reported=float(meta["fps_reported"]),
                        frame_count=int(meta["frame_count"]), width=int(meta["width"]), height=int(meta["height"]),
                        engine=meta["pose_variant"], profile=meta["profile"], tracking=meta["tracking"],
                        validity_policy=meta["validity_policy"], valid_conf_thr=float(meta["valid_conf_thr"]),
                        frames_scanned=int(meta["frames_scanned"]), stride=int(meta["stride"]),
                        inference_size=tuple(meta["inference_size"]))
    return PoseSequence(landmarks=arrays["landmarks"], detected=arrays["detected"].astype(bool),
                        decoded=arrays["decoded"].astype(bool), frame_index=arrays["frame_index"],
                        t_ms=arrays["t_ms"], source_wh=arrays["source_wh"], meta=source)


def practice_input(seq: PoseSequence) -> dict:
    """``PoseSequence`` → 规则输入序列，与练习分析 ``extract_series`` 同口径（丢失帧全 0、无效）。"""
    width, height = (int(v) for v in seq.source_wh[-1])
    fps = float(seq.meta.fps_reported or seq.meta.fps or 30.0)
    series = rule_series(seq.raw(GapPolicy.ZERO, Coords.ASPECT_CORRECTED).copy(), seq.frame_index, fps, width / height)
    series["capture"] = {"sourceFps": fps, "sourceFrames": seq.meta.frames_scanned,
                         "analyzedFrames": len(series["frames"]), "stride": seq.meta.stride, "width": width,
                         "height": height, "inferenceWidth": width, "inferenceHeight": height,
                         "analysisSeconds": 0.0}
    return series


def load_views(pair: SamplePair, cache_dir: Path) -> dict[str, dict]:
    """一对样本已缓存视角的规则输入 ``{"front"/"side": series}``；缺失的视角不出现。"""
    out = {}
    for view in VIEWS:
        path = cache_path(cache_dir, pair.pair_id, view)
        if path.is_file():
            out[view] = practice_input(load_sequence(path))
    return out
