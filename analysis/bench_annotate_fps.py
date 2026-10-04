# -*- coding: utf-8 -*-
"""S5 GPU recheck benchmark harness for YOLO migration Issue #23.

This script is intentionally independent from the production entry points. It
only reads videos, calls existing MediaPipe / YOLO boundary APIs, and writes
benchmark artifacts under ``outputs/``. It must not be imported by the main
runtime path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.paths import models_dir as repo_models_dir  # noqa: E402
from core.body_core_compare import DEFAULT_YOLO26L_MODEL_NAME  # noqa: E402
from core.vision_pipeline import MediaPipePipeline, PipelineConfig  # noqa: E402
from core.yolo_adapter import (  # noqa: E402
    BODY_CORE_V1_VALID_INDICES,
    DEFAULT_YOLO_VALID_CONF_THR,
    FrameResult,
    YoloPoseAdapter,
    evaluate_multi_person_gate,
)


BODY_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    path: Path
    expected_frames: int | None = None


@dataclass(frozen=True)
class BenchCase:
    name: str
    backend: str
    enable_hands: bool
    device: str
    delegate: str = "cpu"
    requires_cuda: bool = False
    imgsz: int = 640
    half: bool = False
    adapter_warmup: bool = False
    warmup_shape: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class FrameLoopStats:
    frames: int
    wall_sec: float
    wall_latencies_sec: tuple[float, ...]
    infer_latencies_sec: tuple[float, ...] = ()
    peak_rss_bytes: int | None = None


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:  # noqa: BLE001
        return None


def collect_env() -> dict[str, Any]:
    env: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        env["torch_cuda_available"] = bool(torch.cuda.is_available())
        env["torch_cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        env["torch_cuda_device_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception as exc:  # noqa: BLE001
        env["torch_error"] = repr(exc)
        env["torch_cuda_available"] = False
        env["torch_cuda_device_count"] = 0

    try:
        import ultralytics

        env["ultralytics"] = ultralytics.__version__
    except Exception as exc:  # noqa: BLE001
        env["ultralytics_error"] = repr(exc)

    try:
        import mediapipe as mp

        env["mediapipe"] = mp.__version__
    except Exception as exc:  # noqa: BLE001
        env["mediapipe_error"] = repr(exc)

    env["opencv"] = cv2.__version__
    for module_name, key in (
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
        ("onnxslim", "onnxslim"),
        ("tensorrt", "tensorrt"),
    ):
        try:
            module = __import__(module_name)
            env[key] = getattr(module, "__version__", "installed")
            if module_name == "onnxruntime":
                env["onnxruntime_providers"] = module.get_available_providers()
        except Exception as exc:  # noqa: BLE001
            env[f"{key}_error"] = repr(exc)
    env.update(_nvidia_smi_info())
    return env


def _nvidia_smi_info() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return {"nvidia_smi_available": False}
    info: dict[str, Any] = {
        "nvidia_smi_available": proc.returncode == 0,
        "nvidia_smi_returncode": proc.returncode,
    }
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    match = re.search(r"Driver Version:\s*([0-9.]+).*?CUDA Version:\s*([0-9.]+)", text, re.S)
    if match:
        info["nvidia_driver_version"] = match.group(1)
        info["nvidia_driver_cuda_version"] = match.group(2)
    gpu_match = re.search(r"\|\s*0\s+([^|]+?)\s{2,}", text)
    if gpu_match:
        info["nvidia_gpu_0"] = " ".join(gpu_match.group(1).split())
    return info


def load_samples(samples_json: Path, *, asset_root: Path | None = None) -> list[Sample]:
    data = json.loads(samples_json.read_text(encoding="utf-8"))
    items = data.get("samples", []) if isinstance(data, dict) else data
    base = asset_root.resolve() if asset_root is not None else samples_json.resolve().parent.parent
    samples: list[Sample] = []
    for idx, item in enumerate(items):
        if isinstance(item, dict):
            raw_path = Path(str(item["path"]))
            sample_id = str(item.get("id") or raw_path.stem or f"sample_{idx}")
            expected_frames = item.get("frames")
        else:
            raw_path = Path(str(item))
            sample_id = raw_path.stem or f"sample_{idx}"
            expected_frames = None
        path = raw_path if raw_path.is_absolute() else base / raw_path
        samples.append(
            Sample(
                sample_id=sample_id,
                path=path,
                expected_frames=int(expected_frames) if expected_frames is not None else None,
            )
        )
    return samples


def _infer_yolo_delegate(model_path: Path) -> str:
    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        return "onnxruntime"
    if suffix == ".engine":
        return "tensorrt"
    return "pytorch"


def _default_yolo_model_path(models_dir: Path) -> Path:
    return Path(models_dir) / DEFAULT_YOLO26L_MODEL_NAME


def _default_cases(
    device: str,
    *,
    yolo_delegate: str = "pytorch",
    include_mediapipe_gpu: bool = False,
) -> list[BenchCase]:
    device_norm = str(device).lower()
    cuda_requested = device_norm.startswith("cuda") or device_norm.isdigit()
    delegate = str(yolo_delegate)
    pytorch_delegate = delegate == "pytorch"
    yolo_cases: list[BenchCase] = []
    if cuda_requested and pytorch_delegate:
        yolo_cases.extend(
            [
                BenchCase("yolo_cpu_body_only", "yolo", False, "cpu", delegate),
                BenchCase("yolo_cpu_body_mp_hands", "yolo", True, "cpu", delegate),
            ]
        )
    yolo_cases.extend(
        [
            BenchCase(
                "yolo_body_only",
                "yolo",
                False,
                device,
                delegate,
                requires_cuda=cuda_requested,
            ),
            BenchCase(
                "yolo_body_mp_hands",
                "yolo",
                True,
                device,
                delegate,
                requires_cuda=cuda_requested,
            ),
        ]
    )
    if cuda_requested and pytorch_delegate:
        yolo_cases.extend(
            [
                BenchCase(
                    "yolo_body_only_fp16_640",
                    "yolo",
                    False,
                    device,
                    "pytorch",
                    requires_cuda=True,
                    imgsz=640,
                    half=True,
                    adapter_warmup=True,
                    warmup_shape=(640, 640, 3),
                ),
                BenchCase(
                    "yolo_body_mp_hands_fp16_640",
                    "yolo",
                    True,
                    device,
                    "pytorch",
                    requires_cuda=True,
                    imgsz=640,
                    half=True,
                    adapter_warmup=True,
                    warmup_shape=(640, 640, 3),
                ),
                BenchCase(
                    "yolo_body_only_fp16_512",
                    "yolo",
                    False,
                    device,
                    "pytorch",
                    requires_cuda=True,
                    imgsz=512,
                    half=True,
                    adapter_warmup=True,
                    warmup_shape=(512, 512, 3),
                ),
                BenchCase(
                    "yolo_body_mp_hands_fp16_512",
                    "yolo",
                    True,
                    device,
                    "pytorch",
                    requires_cuda=True,
                    imgsz=512,
                    half=True,
                    adapter_warmup=True,
                    warmup_shape=(512, 512, 3),
                ),
            ]
        )
    mediapipe_cases = [
        BenchCase("mediapipe_pose_only", "mediapipe", False, "cpu", "cpu"),
        BenchCase("mediapipe_pose_hands", "mediapipe", True, "cpu", "cpu"),
    ]
    if include_mediapipe_gpu:
        mediapipe_cases.extend(
            [
                BenchCase("mediapipe_gpu_pose_only", "mediapipe", False, "gpu", "gpu"),
                BenchCase("mediapipe_gpu_pose_hands", "mediapipe", True, "gpu", "gpu"),
            ]
        )
    return [*mediapipe_cases, *yolo_cases]


class YoloPreviewAnnotator:
    def __init__(
        self,
        *,
        model_path: Path,
        models_dir: Path,
        device: str,
        enable_hands: bool,
        imgsz: int,
        half: bool,
        adapter_warmup: bool,
        warmup_shape: tuple[int, int, int] | None,
        valid_conf_thr: float,
    ) -> None:
        self.adapter = YoloPoseAdapter(
            model_path=model_path,
            valid_conf_thr=valid_conf_thr,
            device=device,
            imgsz=imgsz,
            half=half,
            warmup=adapter_warmup,
            warmup_shape=warmup_shape,
        )
        self.enable_hands = bool(enable_hands)
        self.hand_landmarker: Any = None
        self.frames = 0
        self.yolo_raw_infer_sec = 0.0
        self.yolo_raw_infer_latencies_sec: list[float] = []
        self.last_yolo_raw_infer_sec: float | None = None
        self.detected_frames = 0
        self.body_core_full_valid_frames = 0
        self._body_core_rows: list[np.ndarray | None] = []
        self.num_persons_per_frame: list[int] = []
        if self.enable_hands:
            import mediapipe as mp

            hand_path = models_dir / "hand_landmarker.task"
            if not hand_path.exists():
                raise FileNotFoundError(f"Missing hand landmarker model: {hand_path}")
            self._mp = mp
            self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
                mp.tasks.vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_path)),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_hands=2,
                )
            )

    def annotate(self, frame_bgr: np.ndarray, *, timestamp_ms: int) -> np.ndarray:
        out = frame_bgr.copy()
        t_infer = time.perf_counter()
        frame_result = self.adapter.infer_frame(frame_bgr)
        raw_sec = time.perf_counter() - t_infer
        self.last_yolo_raw_infer_sec = raw_sec
        self.yolo_raw_infer_sec += raw_sec
        self.yolo_raw_infer_latencies_sec.append(raw_sec)
        self.frames += 1
        meta = frame_result.meta or {}
        num_persons = int(meta.get("num_persons") or 0)
        self.num_persons_per_frame.append(num_persons)
        if frame_result.pose33 is not None and num_persons > 0:
            self.detected_frames += 1
        body_core = _body_core_xy_from_frame(frame_result, self.adapter.valid_conf_thr)
        if body_core is not None:
            self.body_core_full_valid_frames += 1
        self._body_core_rows.append(body_core)

        _draw_yolo_body(out, frame_result, self.adapter.valid_conf_thr)
        if self.hand_landmarker is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            hand_res = self.hand_landmarker.detect_for_video(mp_image, int(timestamp_ms))
            hands = hand_res.hand_landmarks if getattr(hand_res, "hand_landmarks", None) else []
            MediaPipePipeline._draw_hands(out, hands, out.shape[1], out.shape[0])
        return out

    def reset_metrics(self) -> None:
        self.frames = 0
        self.yolo_raw_infer_sec = 0.0
        self.yolo_raw_infer_latencies_sec = []
        self.last_yolo_raw_infer_sec = None
        self.detected_frames = 0
        self.body_core_full_valid_frames = 0
        self._body_core_rows = []
        self.num_persons_per_frame = []
        self.adapter.reset_tracker()

    def warmup_raw_infer(self, frame_bgr: np.ndarray) -> float:
        t_infer = time.perf_counter()
        self.adapter.infer_frame(frame_bgr)
        raw_sec = time.perf_counter() - t_infer
        self.last_yolo_raw_infer_sec = raw_sec
        self.yolo_raw_infer_sec += raw_sec
        self.yolo_raw_infer_latencies_sec.append(raw_sec)
        self.frames += 1
        return raw_sec

    def close(self) -> None:
        if self.hand_landmarker is not None:
            self.hand_landmarker.close()

    def metrics(self) -> dict[str, Any]:
        miss_rate = 1.0 - (self.detected_frames / self.frames) if self.frames > 0 else None
        full_valid_rate = (
            self.body_core_full_valid_frames / self.frames if self.frames > 0 else None
        )
        gate = evaluate_multi_person_gate(self.num_persons_per_frame)
        return {
            "yolo_raw_infer_sec": round(self.yolo_raw_infer_sec, 4),
            "yolo_raw_infer_fps": (
                round(self.frames / self.yolo_raw_infer_sec, 3)
                if self.yolo_raw_infer_sec > 0 else 0.0
            ),
            "yolo_miss_rate": _round_or_none(miss_rate),
            "yolo_body_core_full_valid_rate": _round_or_none(full_valid_rate),
            "yolo_body_core_missing_rate": _round_or_none(
                1.0 - full_valid_rate if full_valid_rate is not None else None
            ),
            "yolo_body_core_jitter_median": _round_or_none(
                _body_core_jitter_median(self._body_core_rows)
            ),
            "max_persons": max(self.num_persons_per_frame) if self.num_persons_per_frame else 0,
            "multi_person_frames": int(sum(1 for n in self.num_persons_per_frame if n > 1)),
            "review_required": bool(gate["review_required"]),
            "gate_status": gate["gate_status"],
        }


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _format_warmup_shape(shape: tuple[int, int, int] | None) -> str | None:
    if shape is None:
        return None
    return "x".join(str(int(v)) for v in shape)


def _percentile_ms(values_sec: tuple[float, ...] | list[float], pct: float) -> float | None:
    if not values_sec:
        return None
    arr_ms = np.asarray(values_sec, dtype=np.float64) * 1000.0
    return round(float(np.percentile(arr_ms, pct)), 3)


def _latency_stats(prefix: str, values_sec: tuple[float, ...] | list[float]) -> dict[str, float | None]:
    return {
        f"{prefix}_p50": _percentile_ms(values_sec, 50),
        f"{prefix}_p90": _percentile_ms(values_sec, 90),
        f"{prefix}_p99": _percentile_ms(values_sec, 99),
    }


def _timing_metrics(
    *,
    timed: FrameLoopStats,
    warmup: FrameLoopStats,
) -> dict[str, Any]:
    cold_first_infer = (
        warmup.infer_latencies_sec[0]
        if warmup.infer_latencies_sec
        else (warmup.wall_latencies_sec[0] if warmup.wall_latencies_sec else None)
    )
    cold_first_annotate = warmup.wall_latencies_sec[0] if warmup.wall_latencies_sec else None
    return {
        "frames": timed.frames,
        "warmup_frames": warmup.frames,
        "timed_frames": timed.frames,
        "cold_first_infer_sec": _round_or_none(cold_first_infer),
        "cold_first_annotate_sec": _round_or_none(cold_first_annotate),
        "annotate_wall_sec": round(timed.wall_sec, 4),
        "annotate_fps": round(timed.frames / timed.wall_sec, 3) if timed.wall_sec > 0 else 0.0,
        **_latency_stats("timed_latency_ms", timed.wall_latencies_sec),
        **_latency_stats("yolo_raw_infer_latency_ms", timed.infer_latencies_sec),
    }


def _body_core_xy_from_frame(frame_result: FrameResult, valid_conf_thr: float) -> np.ndarray | None:
    pose = frame_result.pose33
    if not pose:
        return None

    points: list[tuple[float, float]] = []
    for idx in BODY_CORE_V1_VALID_INDICES:
        lm = pose[idx]
        conf = lm.confidence if lm.confidence is not None else lm.visibility
        if lm.synthetic or conf < valid_conf_thr:
            return None
        points.append((float(lm.x), float(lm.y)))
    return np.asarray(points, dtype=np.float32)


def _body_core_jitter_median(rows: list[np.ndarray | None]) -> float | None:
    diffs: list[np.ndarray] = []
    prev: np.ndarray | None = None
    for row in rows:
        if row is None:
            prev = None
            continue
        if prev is not None and prev.shape == row.shape:
            diffs.append(np.linalg.norm(row - prev, axis=1))
        prev = row
    if not diffs:
        return None
    return float(np.median(np.concatenate(diffs)))


def _empty_yolo_metrics() -> dict[str, Any]:
    return {
        "yolo_raw_infer_sec": None,
        "yolo_raw_infer_fps": None,
        "yolo_miss_rate": None,
        "yolo_body_core_full_valid_rate": None,
        "yolo_body_core_missing_rate": None,
        "yolo_body_core_jitter_median": None,
        "max_persons": None,
        "multi_person_frames": None,
        "review_required": None,
        "gate_status": None,
    }


def _draw_yolo_body(out_bgr: np.ndarray, frame_result: FrameResult, valid_conf_thr: float) -> None:
    pose = frame_result.pose33
    if not pose:
        return
    h, w = out_bgr.shape[:2]
    valid_indices: set[int] = set()
    for idx in BODY_CORE_V1_VALID_INDICES:
        lm = pose[idx]
        conf = lm.confidence if lm.confidence is not None else lm.visibility
        if lm.synthetic or conf < valid_conf_thr:
            continue
        valid_indices.add(idx)
    for i, j in BODY_CONNECTIONS:
        if i not in valid_indices or j not in valid_indices:
            continue
        a, b = pose[i], pose[j]
        cv2.line(
            out_bgr,
            (int(a.x * w), int(a.y * h)),
            (int(b.x * w), int(b.y * h)),
            (0, 140, 255),
            2,
            cv2.LINE_AA,
        )
    for idx in valid_indices:
        lm = pose[idx]
        cv2.circle(out_bgr, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1, cv2.LINE_AA)


def _make_runner(
    case: BenchCase,
    *,
    models_dir: Path,
    pose_variant: str,
    yolo_model: Path,
    valid_conf_thr: float,
) -> Any:
    if case.backend == "mediapipe":
        return MediaPipePipeline(
            models_dir=models_dir,
            cfg=PipelineConfig(
                pose_variant=pose_variant,
                running_mode="video",
                enable_hands=case.enable_hands,
                delegate=case.delegate,
            ),
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


def _run_frame_loop(
    *,
    cap: cv2.VideoCapture,
    runner: Any,
    fps_for_ts: float,
    frame_limit: int | None,
) -> FrameLoopStats:
    frames = 0
    wall_latencies: list[float] = []
    infer_latencies: list[float] = []
    peak_rss_bytes = _process_rss_bytes()
    while True:
        if frame_limit is not None and frames >= frame_limit:
            break
        ok, frame = cap.read()
        if not ok:
            break
        timestamp_ms = int(frames * 1000.0 / max(1e-6, fps_for_ts))
        start = time.perf_counter()
        runner.annotate(frame, timestamp_ms=timestamp_ms)
        wall_sec = time.perf_counter() - start
        wall_latencies.append(wall_sec)
        infer_sec = getattr(runner, "last_yolo_raw_infer_sec", None)
        if infer_sec is not None:
            infer_latencies.append(float(infer_sec))
        current_rss_bytes = _process_rss_bytes()
        if current_rss_bytes is not None:
            peak_rss_bytes = max(peak_rss_bytes or 0, current_rss_bytes)
        frames += 1
    return FrameLoopStats(
        frames=frames,
        wall_sec=float(sum(wall_latencies)),
        wall_latencies_sec=tuple(wall_latencies),
        infer_latencies_sec=tuple(infer_latencies),
        peak_rss_bytes=peak_rss_bytes,
    )


def _run_yolo_raw_warmup(
    *,
    cap: cv2.VideoCapture,
    runner: YoloPreviewAnnotator,
    frame_limit: int,
) -> FrameLoopStats:
    frames = 0
    infer_latencies: list[float] = []
    while frames < frame_limit:
        ok, frame = cap.read()
        if not ok:
            break
        infer_latencies.append(runner.warmup_raw_infer(frame))
        frames += 1
    return FrameLoopStats(
        frames=frames,
        wall_sec=float(sum(infer_latencies)),
        wall_latencies_sec=(),
        infer_latencies_sec=tuple(infer_latencies),
    )


def _skip_record(sample: Sample, case: BenchCase, reason: str) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "video": str(sample.path),
        "case": case.name,
        "backend": case.backend,
        "hands": case.enable_hands,
        "device": case.device,
        "delegate": case.delegate,
        "imgsz": case.imgsz,
        "half": case.half,
        "adapter_warmup": case.adapter_warmup,
        "warmup_shape": _format_warmup_shape(case.warmup_shape),
        "status": "skipped",
        "error": reason,
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
    cap = cv2.VideoCapture(str(sample.path))
    if not cap.isOpened():
        return {
            "sample_id": sample.sample_id,
            "video": str(sample.path),
            "case": case.name,
            "backend": case.backend,
            "hands": case.enable_hands,
            "device": case.device,
            "delegate": case.delegate,
            "imgsz": case.imgsz,
            "half": case.half,
            "adapter_warmup": case.adapter_warmup,
            "warmup_shape": _format_warmup_shape(case.warmup_shape),
            "status": "error",
            "error": f"Cannot open video: {sample.path}",
        }

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_for_ts = src_fps if src_fps > 1e-3 else 30.0
    cap.release()

    warmup_stats = FrameLoopStats(0, 0.0, ())
    if warmup_frames > 0 and case.backend == "mediapipe":
        warmup_cap = cv2.VideoCapture(str(sample.path))
        if not warmup_cap.isOpened():
            return _skip_record(sample, case, "warmup_video_open_failed")
        warmup_runner = _make_runner(
            case,
            models_dir=models_dir,
            pose_variant=pose_variant,
            yolo_model=yolo_model,
            valid_conf_thr=valid_conf_thr,
        )
        try:
            warmup_stats = _run_frame_loop(
                cap=warmup_cap,
                runner=warmup_runner,
                fps_for_ts=fps_for_ts,
                frame_limit=max(0, int(warmup_frames)),
            )
        finally:
            warmup_cap.release()
            if hasattr(warmup_runner, "close"):
                warmup_runner.close()

    rss_before_bytes = _process_rss_bytes()
    init_start = time.perf_counter()
    runner = _make_runner(
        case,
        models_dir=models_dir,
        pose_variant=pose_variant,
        yolo_model=yolo_model,
        valid_conf_thr=valid_conf_thr,
    )
    init_sec = time.perf_counter() - init_start
    rss_after_init_bytes = _process_rss_bytes()
    active_delegate = str(getattr(runner, "active_delegate", case.delegate))
    delegate_fallback_reason = getattr(runner, "delegate_fallback_reason", None)

    if warmup_frames > 0 and case.backend == "yolo":
        # Warm only YOLO inference on the same adapter so CUDA/model cold-start
        # stays outside timed metrics, while MediaPipe Hands timestamps remain
        # untouched before the formal VIDEO pass starts from 0.
        warmup_cap = cv2.VideoCapture(str(sample.path))
        if not warmup_cap.isOpened():
            if hasattr(runner, "close"):
                runner.close()
            return _skip_record(sample, case, "warmup_video_open_failed")
        try:
            warmup_stats = _run_yolo_raw_warmup(
                cap=warmup_cap,
                runner=runner,
                frame_limit=max(0, int(warmup_frames)),
            )
            runner.reset_metrics()
        finally:
            warmup_cap.release()

    timed_cap = cv2.VideoCapture(str(sample.path))
    if not timed_cap.isOpened():
        if hasattr(runner, "close"):
            runner.close()
        return _skip_record(sample, case, "timed_video_open_failed")

    try:
        timed_stats = _run_frame_loop(
            cap=timed_cap,
            runner=runner,
            fps_for_ts=fps_for_ts,
            frame_limit=limit_frames,
        )
        yolo_metrics = runner.metrics() if case.backend == "yolo" else _empty_yolo_metrics()
    finally:
        timed_cap.release()
        if hasattr(runner, "close"):
            runner.close()

    timing = _timing_metrics(timed=timed_stats, warmup=warmup_stats)
    peak_rss_candidates = [
        value
        for value in (rss_after_init_bytes, timed_stats.peak_rss_bytes)
        if value is not None
    ]
    peak_rss_bytes = max(peak_rss_candidates) if peak_rss_candidates else None
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
        "active_delegate": active_delegate,
        "delegate_fallback_reason": delegate_fallback_reason,
        "imgsz": case.imgsz,
        "half": case.half,
        "adapter_warmup": case.adapter_warmup,
        "warmup_shape": _format_warmup_shape(case.warmup_shape),
        "status": "ok",
        "expected_frames": sample.expected_frames,
        "init_sec": round(init_sec, 4),
        "peak_rss_mb": round(peak_rss_bytes / (1024 * 1024), 3) if peak_rss_bytes is not None else None,
        "rss_delta_mb": round(rss_delta_bytes / (1024 * 1024), 3) if rss_delta_bytes is not None else None,
        **timing,
        **yolo_metrics,
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    cols = [
        "sample_id",
        "case",
        "backend",
        "hands",
        "device",
        "delegate",
        "active_delegate",
        "delegate_fallback_reason",
        "imgsz",
        "half",
        "adapter_warmup",
        "warmup_shape",
        "status",
        "frames",
        "warmup_frames",
        "timed_frames",
        "expected_frames",
        "init_sec",
        "peak_rss_mb",
        "rss_delta_mb",
        "cold_first_infer_sec",
        "cold_first_annotate_sec",
        "annotate_wall_sec",
        "annotate_fps",
        "timed_latency_ms_p50",
        "timed_latency_ms_p90",
        "timed_latency_ms_p99",
        "yolo_raw_infer_sec",
        "yolo_raw_infer_fps",
        "yolo_raw_infer_latency_ms_p50",
        "yolo_raw_infer_latency_ms_p90",
        "yolo_raw_infer_latency_ms_p99",
        "yolo_miss_rate",
        "yolo_body_core_full_valid_rate",
        "yolo_body_core_missing_rate",
        "yolo_body_core_jitter_median",
        "max_persons",
        "multi_person_frames",
        "review_required",
        "gate_status",
        "error",
        "video",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S5 GPU annotate FPS benchmark harness (Issue #23)")
    parser.add_argument("--samples", default="docs/yolo_eval_samples.json")
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

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = collect_env()
    env_path = out_dir / "gpu_recheck_env.json"
    env_path.write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bench] environment written: {env_path}")
    if args.env_only:
        return 0

    asset_root = Path(args.asset_root).resolve() if args.asset_root else None
    samples = load_samples(Path(args.samples), asset_root=asset_root)
    models_dir = Path(args.models_dir).resolve() if args.models_dir else repo_models_dir()
    yolo_model = Path(args.yolo_model).resolve() if args.yolo_model else _default_yolo_model_path(models_dir)
    yolo_delegate = str(args.yolo_delegate) if args.yolo_delegate else _infer_yolo_delegate(yolo_model)
    bench_cases = _default_cases(
        args.device,
        yolo_delegate=yolo_delegate,
        include_mediapipe_gpu=bool(args.include_mediapipe_gpu),
    )
    if args.mediapipe_only:
        bench_cases = [case for case in bench_cases if case.backend == "mediapipe"]

    records: list[dict[str, Any]] = []
    for sample in samples:
        if not sample.path.exists():
            for case in bench_cases:
                records.append(
                    {
                        "sample_id": sample.sample_id,
                        "video": str(sample.path),
                        "case": case.name,
                        "backend": case.backend,
                        "hands": case.enable_hands,
                        "device": case.device,
                        "delegate": case.delegate,
                        "imgsz": case.imgsz,
                        "half": case.half,
                        "adapter_warmup": case.adapter_warmup,
                        "warmup_shape": _format_warmup_shape(case.warmup_shape),
                        "status": "error",
                        "error": "video_missing",
                    }
                )
            continue
        for case in bench_cases:
            if case.requires_cuda and not env.get("torch_cuda_available", False):
                records.append(_skip_record(sample, case, "torch_cuda_unavailable"))
                continue
            print(f"[bench] {sample.sample_id} / {case.name}", flush=True)
            try:
                records.append(
                    bench_sample(
                        sample,
                        case,
                        models_dir=models_dir,
                        pose_variant=args.pose_variant,
                        yolo_model=yolo_model,
                        limit_frames=args.limit_frames,
                        warmup_frames=max(0, int(args.warmup_frames)),
                        valid_conf_thr=args.valid_conf_thr,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                records.append(
                    {
                        "sample_id": sample.sample_id,
                        "video": str(sample.path),
                        "case": case.name,
                        "backend": case.backend,
                        "hands": case.enable_hands,
                        "device": case.device,
                        "delegate": case.delegate,
                        "imgsz": case.imgsz,
                        "half": case.half,
                        "adapter_warmup": case.adapter_warmup,
                        "warmup_shape": _format_warmup_shape(case.warmup_shape),
                        "status": "error",
                        "error": repr(exc),
                    }
                )

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
    _write_csv(csv_path, records)
    print(f"[bench] written: {json_path}")
    print(f"[bench] written: {csv_path}")
    return 0 if all(rec.get("status") == "ok" for rec in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
