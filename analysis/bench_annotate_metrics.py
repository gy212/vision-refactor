# -*- coding: utf-8 -*-
"""S5 标注帧率基准的统计与产物：运行环境采集、逐帧耗时的分位数与冷启动指标、CSV 列表与写出。

只供 ``analysis/bench_annotate_fps.py`` 使用，不进入生产入口。
"""

from __future__ import annotations

import csv
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameLoopStats:
    frames: int
    wall_sec: float
    wall_latencies_sec: tuple[float, ...]
    infer_latencies_sec: tuple[float, ...] = ()
    peak_rss_bytes: int | None = None


CSV_COLUMNS = (
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
)


def _torch_env() -> dict[str, Any]:
    env: dict[str, Any] = {}
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
    return env


def collect_env() -> dict[str, Any]:
    env: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    env.update(_torch_env())
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


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


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


def timing_metrics(
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
        "cold_first_infer_sec": round_or_none(cold_first_infer),
        "cold_first_annotate_sec": round_or_none(cold_first_annotate),
        "annotate_wall_sec": round(timed.wall_sec, 4),
        "annotate_fps": round(timed.frames / timed.wall_sec, 3) if timed.wall_sec > 0 else 0.0,
        **_latency_stats("timed_latency_ms", timed.wall_latencies_sec),
        **_latency_stats("yolo_raw_infer_latency_ms", timed.infer_latencies_sec),
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
