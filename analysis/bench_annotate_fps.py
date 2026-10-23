# -*- coding: utf-8 -*-
"""S5 GPU recheck benchmark harness for YOLO migration Issue #23.

This script is intentionally independent from the production entry points. It
only reads videos, calls existing MediaPipe / YOLO boundary APIs, and writes
benchmark artifacts under ``outputs/``. It must not be imported by the main
runtime path. 用例与样本见 ``bench_annotate_cases``，YOLO 被测对象见
``bench_annotate_yolo``，统计与 CSV 见 ``bench_annotate_metrics``。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.bench_annotate_cases import (  # noqa: E402
    BenchCase,
    Sample,
    case_record,
    default_cases,
    default_yolo_model_path,
    format_warmup_shape,
    infer_yolo_delegate,
    load_samples,
)
from analysis.bench_annotate_metrics import FrameLoopStats, collect_env, timing_metrics, write_csv  # noqa: E402
from analysis.bench_annotate_yolo import YoloPreviewAnnotator, empty_yolo_metrics  # noqa: E402
from core.foundation import models_dir as repo_models_dir  # noqa: E402
from core.media import VideoOpenError, open_video  # noqa: E402
from core.pose import EngineSpec, create_engine  # noqa: E402
from core.pose.backends.yolo import DEFAULT_YOLO26L_MODEL_NAME, DEFAULT_YOLO_VALID_CONF_THR  # noqa: E402


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:  # noqa: BLE001
        return None


def _make_runner(
    case: BenchCase,
    *,
    models_dir: Path,
    pose_variant: str,
    yolo_model: Path,
    valid_conf_thr: float,
) -> Any:
    if case.backend == "mediapipe":
        return create_engine(
            EngineSpec(
                pose_variant=pose_variant,
                running_mode="video",
                enable_hands=case.enable_hands,
                delegate=case.delegate,
            ),
            models_dir=models_dir,
        )
    if case.backend == "yolo":
        return YoloPreviewAnnotator(
            model_path=yolo_model,
            models_dir=models_dir,
            device=case.device,
            enable_hands=case.enable_hands,
            imgsz=case.imgsz,
            half=case.half,
            adapter_warmup=case.adapter_warmup,
            warmup_shape=case.warmup_shape,
            valid_conf_thr=valid_conf_thr,
        )
    raise ValueError(f"Unsupported backend: {case.backend}")


@contextmanager
def _video_frames(path: Path) -> Iterator[Iterator[np.ndarray]]:
    """按顺序解码视频的 BGR 帧（``core.media`` 唯一解码入口）；打不开时抛 ``VideoOpenError``。"""
    source = open_video(path)
    try:
        yield (frame.bgr for frame in source)
    finally:
        source.close()


def _run_frame_loop(
    *,
    frames: Iterable[np.ndarray],
    runner: Any,
    fps_for_ts: float,
    frame_limit: int | None,
) -> FrameLoopStats:
    count = 0
    wall_latencies: list[float] = []
    infer_latencies: list[float] = []
    peak_rss_bytes = _process_rss_bytes()
    for frame in frames:
        if frame_limit is not None and count >= frame_limit:
            break
        timestamp_ms = int(count * 1000.0 / max(1e-6, fps_for_ts))
        start = time.perf_counter()
        runner.annotate(frame, timestamp_ms=timestamp_ms)
        wall_latencies.append(time.perf_counter() - start)
        infer_sec = getattr(runner, "last_yolo_raw_infer_sec", None)
        if infer_sec is not None:
            infer_latencies.append(float(infer_sec))
        current_rss_bytes = _process_rss_bytes()
        if current_rss_bytes is not None:
            peak_rss_bytes = max(peak_rss_bytes or 0, current_rss_bytes)
        count += 1
    return FrameLoopStats(
        frames=count,
        wall_sec=float(sum(wall_latencies)),
        wall_latencies_sec=tuple(wall_latencies),
        infer_latencies_sec=tuple(infer_latencies),
        peak_rss_bytes=peak_rss_bytes,
    )


def _run_yolo_raw_warmup(
    *,
    frames: Iterable[np.ndarray],
    runner: YoloPreviewAnnotator,
    frame_limit: int,
) -> FrameLoopStats:
    infer_latencies: list[float] = []
    for frame in frames:
        if len(infer_latencies) >= frame_limit:
            break
        infer_latencies.append(runner.warmup_raw_infer(frame))
    return FrameLoopStats(
        frames=len(infer_latencies),
        wall_sec=float(sum(infer_latencies)),
        wall_latencies_sec=(),
        infer_latencies_sec=tuple(infer_latencies),
    )


def _warm_mediapipe(sample: Sample, case: BenchCase, runner_kwargs: dict, fps_for_ts: float,
                    warmup_frames: int) -> FrameLoopStats:
    """MediaPipe 用一个独立 runner 预热，正式计时换新 runner，手部 VIDEO 时间戳从 0 开始。"""
    warmup_runner = _make_runner(case, **runner_kwargs)
    try:
        with _video_frames(sample.path) as frames:
            return _run_frame_loop(frames=frames, runner=warmup_runner, fps_for_ts=fps_for_ts,
                                   frame_limit=warmup_frames)
    finally:
        if hasattr(warmup_runner, "close"):
            warmup_runner.close()


def _warm_yolo(sample: Sample, runner: Any, warmup_frames: int) -> FrameLoopStats:
    """只在同一 adapter 上预热 YOLO 推理，把 CUDA/模型冷启动留在计时之外，然后清零统计。"""
    with _video_frames(sample.path) as frames:
        stats = _run_yolo_raw_warmup(frames=frames, runner=runner, frame_limit=warmup_frames)
    runner.reset_metrics()
    return stats


def _ok_record(sample: Sample, case: BenchCase, runner: Any, init_sec: float, rss: tuple[int | None, ...],
               timed: FrameLoopStats, warmup: FrameLoopStats, yolo_metrics: dict) -> dict[str, Any]:
    rss_before_bytes, rss_after_init_bytes = rss
    peak_candidates = [v for v in (rss_after_init_bytes, timed.peak_rss_bytes) if v is not None]
    peak_rss_bytes = max(peak_candidates) if peak_candidates else None
    rss_delta_bytes = (
        max(0, peak_rss_bytes - rss_before_bytes)
        if peak_rss_bytes is not None and rss_before_bytes is not None
        else None
    )
    return {
        "sample_id": sample.sample_id,
        "video": str(sample.path),
        "case": case.name,
        "backend": case.backend,
        "hands": case.enable_hands,
        "device": case.device,
        "delegate": case.delegate,
        "active_delegate": str(getattr(runner, "active_delegate", case.delegate)),
        "delegate_fallback_reason": getattr(runner, "delegate_fallback_reason", None),
        "imgsz": case.imgsz,
        "half": case.half,
        "adapter_warmup": case.adapter_warmup,
        "warmup_shape": format_warmup_shape(case.warmup_shape),
        "status": "ok",
        "expected_frames": sample.expected_frames,
        "init_sec": round(init_sec, 4),
        "peak_rss_mb": round(peak_rss_bytes / (1024 * 1024), 3) if peak_rss_bytes is not None else None,
        "rss_delta_mb": round(rss_delta_bytes / (1024 * 1024), 3) if rss_delta_bytes is not None else None,
        **timing_metrics(timed=timed, warmup=warmup),
        **yolo_metrics,
    }


def bench_sample(
    sample: Sample,
    case: BenchCase,
    *,
    models_dir: Path,
    pose_variant: str,
    yolo_model: Path,
    limit_frames: int | None,
    warmup_frames: int,
    valid_conf_thr: float,
) -> dict[str, Any]:
    """一个样本 × 一个用例：预热（不计时）、初始化 runner、正式计时一遍，返回一条记录。"""
    try:
        with open_video(sample.path) as probe:
            src_fps = probe.meta.fps_reported
    except VideoOpenError:
        return case_record(sample, case, status="error", error=f"Cannot open video: {sample.path}")
    fps_for_ts = src_fps if src_fps > 1e-3 else 30.0
    runner_kwargs = dict(models_dir=models_dir, pose_variant=pose_variant, yolo_model=yolo_model,
                         valid_conf_thr=valid_conf_thr)
    warmup_frames = max(0, int(warmup_frames))

    warmup_stats = FrameLoopStats(0, 0.0, ())
    if warmup_frames > 0 and case.backend == "mediapipe":
        try:
            warmup_stats = _warm_mediapipe(sample, case, runner_kwargs, fps_for_ts, warmup_frames)
        except VideoOpenError:
            return case_record(sample, case, status="skipped", error="warmup_video_open_failed")

    rss_before_bytes = _process_rss_bytes()
    init_start = time.perf_counter()
    runner = _make_runner(case, **runner_kwargs)
    init_sec = time.perf_counter() - init_start
    rss_after_init_bytes = _process_rss_bytes()
    try:
        if warmup_frames > 0 and case.backend == "yolo":
            try:
                warmup_stats = _warm_yolo(sample, runner, warmup_frames)
            except VideoOpenError:
                return case_record(sample, case, status="skipped", error="warmup_video_open_failed")
        try:
            with _video_frames(sample.path) as frames:
                timed_stats = _run_frame_loop(frames=frames, runner=runner, fps_for_ts=fps_for_ts,
                                              frame_limit=limit_frames)
        except VideoOpenError:
            return case_record(sample, case, status="skipped", error="timed_video_open_failed")
        yolo_metrics = runner.metrics() if case.backend == "yolo" else empty_yolo_metrics()
    finally:
        if hasattr(runner, "close"):
            runner.close()
    return _ok_record(sample, case, runner, init_sec, (rss_before_bytes, rss_after_init_bytes),
                      timed_stats, warmup_stats, yolo_metrics)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S5 GPU annotate FPS benchmark harness (Issue #23)")
    parser.add_argument("--samples", default=None, help="（--env-only 以外必填）样本清单 JSON（schema 见 git 历史中的 docs/yolo_eval_samples.json；样本登记见 docs/样本清单.md）")
    parser.add_argument("--asset-root", default=None, help="Root for ignored local videos when samples paths are relative")
    parser.add_argument("--models-dir", default=None)
    parser.add_argument(
        "--yolo-model",
        default=None,
        help=f"YOLO model path (default: models/{DEFAULT_YOLO26L_MODEL_NAME})",
    )
    parser.add_argument("--pose-variant", default="full", choices=["lite", "full", "heavy"])
    parser.add_argument("--device", default="cuda", help="YOLO device, e.g. cuda, 0, cpu")
    parser.add_argument("--yolo-delegate", default=None, help="Optional benchmark label, e.g. pytorch, onnxruntime, tensorrt")
    parser.add_argument("--valid-conf-thr", type=float, default=DEFAULT_YOLO_VALID_CONF_THR)
    parser.add_argument("--limit-frames", type=int, default=None, help="Optional smoke-test frame cap per sample/case")
    parser.add_argument("--warmup-frames", type=int, default=10, help="Warmup frames per sample/case, excluded from timed metrics")
    parser.add_argument(
        "--include-mediapipe-gpu",
        action="store_true",
        help="Opt-in MediaPipe BaseOptions.Delegate.GPU cases for P4 feasibility spikes",
    )
    parser.add_argument(
        "--mediapipe-only",
        action="store_true",
        help="Run only MediaPipe CPU/GPU cases (skip every YOLO benchmark case)",
    )
    parser.add_argument("--out", default="outputs/gpu_recheck")
    parser.add_argument("--env-only", action="store_true", help="Write environment JSON and exit")
    args = parser.parse_args(argv)
    if not args.env_only and not args.samples:
        parser.error("除 --env-only 外必须用 --samples 指定样本清单")
    return args


def _bench_all(args: argparse.Namespace, env: dict[str, Any]) -> tuple[list[Sample], list[dict[str, Any]]]:
    """按样本 × 用例逐个跑基准；缺视频、缺 CUDA、运行异常都记一条非 ok 记录。"""
    asset_root = Path(args.asset_root).resolve() if args.asset_root else None
    samples = load_samples(Path(args.samples), asset_root=asset_root)
    models_dir = Path(args.models_dir).resolve() if args.models_dir else repo_models_dir()
    yolo_model = Path(args.yolo_model).resolve() if args.yolo_model else default_yolo_model_path(models_dir)
    yolo_delegate = str(args.yolo_delegate) if args.yolo_delegate else infer_yolo_delegate(yolo_model)
    bench_cases = default_cases(args.device, yolo_delegate=yolo_delegate,
                                include_mediapipe_gpu=bool(args.include_mediapipe_gpu))
    if args.mediapipe_only:
        bench_cases = [case for case in bench_cases if case.backend == "mediapipe"]

    records: list[dict[str, Any]] = []
    for sample in samples:
        if not sample.path.exists():
            records += [case_record(sample, case, status="error", error="video_missing") for case in bench_cases]
            continue
        for case in bench_cases:
            if case.requires_cuda and not env.get("torch_cuda_available", False):
                records.append(case_record(sample, case, status="skipped", error="torch_cuda_unavailable"))
                continue
            print(f"[bench] {sample.sample_id} / {case.name}", flush=True)
            try:
                records.append(bench_sample(
                    sample, case, models_dir=models_dir, pose_variant=args.pose_variant, yolo_model=yolo_model,
                    limit_frames=args.limit_frames, warmup_frames=max(0, int(args.warmup_frames)),
                    valid_conf_thr=args.valid_conf_thr))
            except Exception as exc:  # noqa: BLE001
                records.append(case_record(sample, case, status="error", error=repr(exc)))
    return samples, records


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = collect_env()
    env_path = out_dir / "gpu_recheck_env.json"
    env_path.write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bench] environment written: {env_path}")
    if args.env_only:
        return 0

    samples, records = _bench_all(args, env)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": env,
        "status": "ok",
        "requested_device": args.device,
        "include_mediapipe_gpu": bool(args.include_mediapipe_gpu),
        "mediapipe_only": bool(args.mediapipe_only),
        "warmup_frames": int(args.warmup_frames),
        "samples_total": len(samples),
        "records": records,
    }
    json_path = out_dir / "annotate_fps_gpu_recheck.json"
    csv_path = out_dir / "annotate_fps_gpu_recheck.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, records)
    print(f"[bench] written: {json_path}")
    print(f"[bench] written: {csv_path}")
    return 0 if all(rec.get("status") == "ok" for rec in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
