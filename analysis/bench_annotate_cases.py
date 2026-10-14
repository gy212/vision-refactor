# -*- coding: utf-8 -*-
"""S5 标注帧率基准的输入：样本清单读取与基准用例（后端、设备、手部、精度）的默认组合。

只供 ``analysis/bench_annotate_fps.py`` 使用，不进入生产入口。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.pose.backends.yolo import DEFAULT_YOLO26L_MODEL_NAME


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


def infer_yolo_delegate(model_path: Path) -> str:
    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        return "onnxruntime"
    if suffix == ".engine":
        return "tensorrt"
    return "pytorch"


def default_yolo_model_path(models_dir: Path) -> Path:
    return Path(models_dir) / DEFAULT_YOLO26L_MODEL_NAME


def format_warmup_shape(shape: tuple[int, int, int] | None) -> str | None:
    if shape is None:
        return None
    return "x".join(str(int(v)) for v in shape)


def case_record(sample: Sample, case: BenchCase, *, status: str, error: str) -> dict[str, Any]:
    """没有计时结果的一条记录（跳过或出错）：样本与用例的标识字段加状态和原因。"""
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
        "warmup_shape": format_warmup_shape(case.warmup_shape),
        "status": status,
        "error": error,
    }


def _fp16_cases(device: str, imgsz: int) -> list[BenchCase]:
    """CUDA + PyTorch 下的 FP16 用例：身体、身体加 MediaPipe 手部，各带同尺寸预热。"""
    return [
        BenchCase(
            f"yolo_{name}_fp16_{imgsz}",
            "yolo",
            hands,
            device,
            "pytorch",
            requires_cuda=True,
            imgsz=imgsz,
            half=True,
            adapter_warmup=True,
            warmup_shape=(imgsz, imgsz, 3),
        )
        for name, hands in (("body_only", False), ("body_mp_hands", True))
    ]


def _yolo_cases(device: str, delegate: str) -> list[BenchCase]:
    device_norm = str(device).lower()
    cuda_requested = device_norm.startswith("cuda") or device_norm.isdigit()
    cuda_pytorch = cuda_requested and delegate == "pytorch"
    cases: list[BenchCase] = []
    if cuda_pytorch:
        cases += [
            BenchCase("yolo_cpu_body_only", "yolo", False, "cpu", delegate),
            BenchCase("yolo_cpu_body_mp_hands", "yolo", True, "cpu", delegate),
        ]
    cases += [
        BenchCase("yolo_body_only", "yolo", False, device, delegate, requires_cuda=cuda_requested),
        BenchCase("yolo_body_mp_hands", "yolo", True, device, delegate, requires_cuda=cuda_requested),
    ]
    if cuda_pytorch:
        cases += _fp16_cases(device, 640) + _fp16_cases(device, 512)
    return cases


def default_cases(
    device: str,
    *,
    yolo_delegate: str = "pytorch",
    include_mediapipe_gpu: bool = False,
) -> list[BenchCase]:
    """MediaPipe CPU（可选 GPU）在前，YOLO 用例在后；CUDA + PyTorch 时多出 CPU 对照与 FP16 用例。"""
    mediapipe_cases = [
        BenchCase("mediapipe_pose_only", "mediapipe", False, "cpu", "cpu"),
        BenchCase("mediapipe_pose_hands", "mediapipe", True, "cpu", "cpu"),
    ]
    if include_mediapipe_gpu:
        mediapipe_cases += [
            BenchCase("mediapipe_gpu_pose_only", "mediapipe", False, "gpu", "gpu"),
            BenchCase("mediapipe_gpu_pose_hands", "mediapipe", True, "gpu", "gpu"),
        ]
    return [*mediapipe_cases, *_yolo_cases(device, str(yolo_delegate))]
