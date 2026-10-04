# -*- coding: utf-8 -*-
"""S5 GPU recheck decision gate regression tests (YOLO migration Issue #23)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis import bench_annotate_fps  # noqa: E402
from core.yolo_adapter import BODY_CORE_V1_VALID_INDICES, FrameResult, Landmark  # noqa: E402


REPORT = _REPO_ROOT / "docs" / "yolo_gpu_recheck_report.md"


def test_gpu_recheck_report_has_cuda_fixed_no_go_decision():
    assert REPORT.exists(), f"GPU 复测报告缺失：{REPORT}"
    text = REPORT.read_text(encoding="utf-8")

    for keyword in (
        "预注册阈值",
        "CUDA 修正版",
        "torch=2.11.0+cu128",
        "torch.cuda.is_available() = True",
        "6/6 样本真实 GPU 复测",
        "Hands 关 FPS 比仅 1/6",
        "Hands 开 FPS 比 0/6",
        "YOLO raw FPS 0/6",
        "body_core 抖动 3/6",
        "真实 GPU 数据 no-go",
        "no-go",
        "#25",
        "#26",
        "#28",
        "全部不切默认",
    ):
        assert keyword in text, f"GPU 复测报告缺少关键口径：{keyword}"

    assert "analysis/bench_annotate_fps.py" in text
    assert "yolo_raw_infer_fps" in text
    assert "旧 CPU torch 前提已纠正" in text

    stale_final_claims = (
        "0/6 个样本产生有效 GPU",
        "skip_reason = torch_cuda_unavailable",
        "GPU runtime 可用性：`False`",
    )
    for stale in stale_final_claims:
        assert stale not in text, f"GPU 复测报告仍保留 CPU torch 无效前提作为最终结论：{stale}"


def test_benchmark_env_only_writes_probe_json(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bench_annotate_fps,
        "collect_env",
        lambda: {
            "python": "test",
            "opencv": "test",
            "torch_cuda_available": False,
        },
    )

    out_dir = tmp_path / "gpu_recheck"
    rc = bench_annotate_fps.main(["--out", str(out_dir), "--env-only"])

    assert rc == 0
    env_path = out_dir / "gpu_recheck_env.json"
    assert env_path.exists()
    env = json.loads(env_path.read_text(encoding="utf-8"))
    for key in ("python", "opencv", "torch_cuda_available"):
        assert key in env


def test_body_core_jitter_helper_uses_only_consecutive_full_valid_frames():
    row_0 = bench_annotate_fps._body_core_xy_from_frame(_frame_with_shift(0.0), 0.6)
    row_1 = bench_annotate_fps._body_core_xy_from_frame(_frame_with_shift(0.01), 0.6)
    assert row_0 is not None
    assert row_1 is not None

    jitter = bench_annotate_fps._body_core_jitter_median([row_0, row_1, None, row_1])
    assert jitter == pytest.approx(0.01, abs=1e-6)


def test_csv_schema_contains_gpu_recheck_metric_fields(tmp_path):
    csv_path = tmp_path / "bench.csv"
    bench_annotate_fps._write_csv(
        csv_path,
        [
            {
                "sample_id": "s",
                "case": "yolo_body_only",
                "backend": "yolo",
                "device": "cuda",
                "delegate": "pytorch",
                "active_delegate": "pytorch",
                "delegate_fallback_reason": None,
                "imgsz": 640,
                "half": True,
                "adapter_warmup": True,
                "warmup_shape": "640x640x3",
                "status": "ok",
                "frames": 1,
                "warmup_frames": 10,
                "timed_frames": 1,
                "peak_rss_mb": 100.0,
                "rss_delta_mb": 10.0,
                "timed_latency_ms_p50": 10.0,
                "timed_latency_ms_p90": 10.0,
                "timed_latency_ms_p99": 10.0,
                "yolo_raw_infer_latency_ms_p50": 8.0,
                "yolo_raw_infer_latency_ms_p90": 8.0,
                "yolo_raw_infer_latency_ms_p99": 8.0,
                "yolo_raw_infer_fps": 99.0,
                "yolo_miss_rate": 0.0,
                "yolo_body_core_jitter_median": 0.01,
                "max_persons": 1,
                "multi_person_frames": 0,
                "review_required": False,
                "gate_status": "ok",
            }
        ],
    )

    text = csv_path.read_text(encoding="utf-8-sig")
    assert "yolo_raw_infer_fps" in text
    assert "yolo_miss_rate" in text
    assert "warmup_frames" in text
    assert "active_delegate" in text
    assert "delegate_fallback_reason" in text
    assert "imgsz" in text
    assert "adapter_warmup" in text
    assert "warmup_shape" in text
    assert "timed_frames" in text
    assert "peak_rss_mb" in text
    assert "rss_delta_mb" in text
    assert "timed_latency_ms_p50" in text
    assert "yolo_raw_infer_latency_ms_p99" in text
    assert "max_persons" in text
    assert "review_required" in text
    assert "yolo_body_core_jitter_median" in text


def test_default_cases_keep_cpu_baseline_when_cuda_is_requested():
    cases = {case.name: case for case in bench_annotate_fps._default_cases("cuda")}

    assert cases["mediapipe_pose_only"].backend == "mediapipe"
    assert cases["mediapipe_pose_only"].device == "cpu"
    assert cases["mediapipe_pose_only"].delegate == "cpu"
    assert not cases["mediapipe_pose_only"].requires_cuda

    assert cases["yolo_cpu_body_only"].backend == "yolo"
    assert cases["yolo_cpu_body_only"].device == "cpu"
    assert cases["yolo_cpu_body_only"].delegate == "pytorch"
    assert not cases["yolo_cpu_body_only"].requires_cuda

    assert cases["yolo_body_only"].device == "cuda"
    assert cases["yolo_body_only"].requires_cuda
    assert cases["yolo_body_only"].imgsz == 640
    assert not cases["yolo_body_only"].half
    assert not cases["yolo_body_only"].adapter_warmup

    assert cases["yolo_body_only_fp16_640"].device == "cuda"
    assert cases["yolo_body_only_fp16_640"].requires_cuda
    assert cases["yolo_body_only_fp16_640"].imgsz == 640
    assert cases["yolo_body_only_fp16_640"].half
    assert cases["yolo_body_only_fp16_640"].adapter_warmup
    assert cases["yolo_body_only_fp16_640"].warmup_shape == (640, 640, 3)

    assert cases["yolo_body_only_fp16_512"].imgsz == 512
    assert cases["yolo_body_only_fp16_512"].half
    assert cases["yolo_body_only_fp16_512"].warmup_shape == (512, 512, 3)


def test_mediapipe_gpu_cases_are_explicit_opt_in():
    default_cases = {case.name: case for case in bench_annotate_fps._default_cases("cuda")}
    assert "mediapipe_gpu_pose_only" not in default_cases
    assert "mediapipe_gpu_pose_hands" not in default_cases

    gpu_cases = {
        case.name: case
        for case in bench_annotate_fps._default_cases("cuda", include_mediapipe_gpu=True)
    }

    assert gpu_cases["mediapipe_pose_only"].delegate == "cpu"
    assert gpu_cases["mediapipe_pose_hands"].delegate == "cpu"
    assert gpu_cases["mediapipe_gpu_pose_only"].backend == "mediapipe"
    assert gpu_cases["mediapipe_gpu_pose_only"].device == "gpu"
    assert gpu_cases["mediapipe_gpu_pose_only"].delegate == "gpu"
    assert gpu_cases["mediapipe_gpu_pose_only"].enable_hands is False
    assert gpu_cases["mediapipe_gpu_pose_hands"].delegate == "gpu"
    assert gpu_cases["mediapipe_gpu_pose_hands"].enable_hands is True
    assert not gpu_cases["mediapipe_gpu_pose_hands"].requires_cuda


def test_mediapipe_only_cli_filters_every_yolo_case(tmp_path):
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": "missing",
                        "path": "missing.mp4",
                        "view": "front",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    exit_code = bench_annotate_fps.main(
        [
            "--samples",
            str(samples_path),
            "--asset-root",
            str(tmp_path),
            "--out",
            str(out_dir),
            "--include-mediapipe-gpu",
            "--mediapipe-only",
        ]
    )

    assert exit_code == 1
    payload = json.loads((out_dir / "annotate_fps_gpu_recheck.json").read_text(encoding="utf-8"))
    assert payload["mediapipe_only"] is True
    assert {record["case"] for record in payload["records"]} == {
        "mediapipe_pose_only",
        "mediapipe_pose_hands",
        "mediapipe_gpu_pose_only",
        "mediapipe_gpu_pose_hands",
    }
    assert {record["backend"] for record in payload["records"]} == {"mediapipe"}


def test_mediapipe_runner_passes_delegate_to_pipeline(tmp_path, monkeypatch):
    captured_cfgs = []

    class FakePipeline:
        def __init__(self, *, models_dir, cfg) -> None:
            captured_cfgs.append(cfg)

    monkeypatch.setattr(bench_annotate_fps, "MediaPipePipeline", FakePipeline)

    runner = bench_annotate_fps._make_runner(
        bench_annotate_fps.BenchCase("mediapipe_gpu_pose_only", "mediapipe", False, "gpu", "gpu"),
        models_dir=tmp_path,
        pose_variant="full",
        yolo_model=tmp_path / "model.pt",
        valid_conf_thr=0.6,
    )

    assert isinstance(runner, FakePipeline)
    assert captured_cfgs
    assert captured_cfgs[0].delegate == "gpu"
    assert captured_cfgs[0].enable_hands is False


def test_exported_yolo_model_delegate_is_inferred_from_suffix():
    assert bench_annotate_fps._infer_yolo_delegate(Path("yolo11n-pose.pt")) == "pytorch"
    assert bench_annotate_fps._infer_yolo_delegate(Path("yolo11n-pose.onnx")) == "onnxruntime"
    assert bench_annotate_fps._infer_yolo_delegate(Path("yolo11n-pose.engine")) == "tensorrt"


def test_exported_yolo_default_cases_skip_pytorch_only_fp16_variants():
    cases = {
        case.name: case
        for case in bench_annotate_fps._default_cases("cuda", yolo_delegate="onnxruntime")
    }

    assert cases["yolo_body_only"].delegate == "onnxruntime"
    assert cases["yolo_body_mp_hands"].delegate == "onnxruntime"
    assert "yolo_cpu_body_only" not in cases
    assert "yolo_cpu_body_mp_hands" not in cases
    assert "yolo_body_only_fp16_640" not in cases
    assert "yolo_body_mp_hands_fp16_640" not in cases
    assert "yolo_body_only_fp16_512" not in cases
    assert "yolo_body_mp_hands_fp16_512" not in cases


def test_timing_metrics_separate_warmup_and_timed_latency():
    warmup = bench_annotate_fps.FrameLoopStats(
        frames=2,
        wall_sec=0.9,
        wall_latencies_sec=(0.5, 0.4),
        infer_latencies_sec=(0.3, 0.2),
    )
    timed = bench_annotate_fps.FrameLoopStats(
        frames=3,
        wall_sec=0.06,
        wall_latencies_sec=(0.01, 0.02, 0.03),
        infer_latencies_sec=(0.004, 0.006, 0.008),
    )

    metrics = bench_annotate_fps._timing_metrics(timed=timed, warmup=warmup)

    assert metrics["frames"] == 3
    assert metrics["timed_frames"] == 3
    assert metrics["warmup_frames"] == 2
    assert metrics["cold_first_infer_sec"] == 0.3
    assert metrics["cold_first_annotate_sec"] == 0.5
    assert metrics["annotate_fps"] == 50.0
    assert metrics["timed_latency_ms_p50"] == 20.0
    assert metrics["timed_latency_ms_p90"] == 28.0
    assert metrics["yolo_raw_infer_latency_ms_p50"] == 6.0


def test_frame_loop_records_peak_process_rss(monkeypatch):
    rss_values = iter((100, 120, 115))

    class FakeCap:
        def __init__(self) -> None:
            self.frames = 0

        def read(self):
            if self.frames >= 2:
                return False, None
            self.frames += 1
            return True, object()

    class FakeRunner:
        last_yolo_raw_infer_sec = None

        def annotate(self, frame, *, timestamp_ms: int) -> None:
            pass

    monkeypatch.setattr(bench_annotate_fps, "_process_rss_bytes", lambda: next(rss_values))

    stats = bench_annotate_fps._run_frame_loop(
        cap=FakeCap(),
        runner=FakeRunner(),
        fps_for_ts=30.0,
        frame_limit=None,
    )

    assert stats.peak_rss_bytes == 120


def test_bench_sample_yolo_warmup_is_reset_before_timed_metrics(tmp_path, monkeypatch):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake video")
    sample = bench_annotate_fps.Sample("s", video_path, expected_frames=2)
    calls: list[str] = []

    class FakeCap:
        def __init__(self, path: str) -> None:
            self.path = path
            self.index = 0

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            return 30.0

        def read(self):
            if self.index >= 2:
                return False, None
            self.index += 1
            return True, object()

        def release(self) -> None:
            pass

    class FakeYoloRunner:
        last_yolo_raw_infer_sec: float | None = None
        active_delegate = "pytorch"
        delegate_fallback_reason = None

        def __init__(self) -> None:
            self.frames = 0
            self.yolo_raw_infer_sec = 0.0
            self.closed = False

        def warmup_raw_infer(self, frame) -> float:
            calls.append("warmup")
            self.frames += 1
            self.yolo_raw_infer_sec += 0.5
            self.last_yolo_raw_infer_sec = 0.5
            return 0.5

        def reset_metrics(self) -> None:
            calls.append("reset")
            self.frames = 0
            self.yolo_raw_infer_sec = 0.0
            self.last_yolo_raw_infer_sec = None

        def annotate(self, frame, *, timestamp_ms: int):
            calls.append(f"timed:{timestamp_ms}")
            self.frames += 1
            self.yolo_raw_infer_sec += 0.01
            self.last_yolo_raw_infer_sec = 0.01
            return frame

        def metrics(self) -> dict[str, float | None]:
            calls.append("metrics")
            return {
                "yolo_raw_infer_sec": round(self.yolo_raw_infer_sec, 4),
                "yolo_raw_infer_fps": round(self.frames / self.yolo_raw_infer_sec, 3),
                "yolo_miss_rate": 0.0,
                "yolo_body_core_full_valid_rate": 1.0,
                "yolo_body_core_missing_rate": 0.0,
                "yolo_body_core_jitter_median": 0.0,
                "max_persons": 1,
                "multi_person_frames": 0,
                "review_required": False,
                "gate_status": "ok",
            }

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(bench_annotate_fps.cv2, "VideoCapture", FakeCap)
    monkeypatch.setattr(bench_annotate_fps, "time", _FakeTime())
    monkeypatch.setattr(
        bench_annotate_fps,
        "_make_runner",
        lambda *args, **kwargs: FakeYoloRunner(),
    )

    record = bench_annotate_fps.bench_sample(
        sample,
        bench_annotate_fps.BenchCase("yolo_body_only", "yolo", False, "cuda", "pytorch"),
        models_dir=tmp_path,
        pose_variant="full",
        yolo_model=tmp_path / "model.pt",
        limit_frames=1,
        warmup_frames=1,
        valid_conf_thr=0.6,
    )

    assert calls == ["warmup", "reset", "timed:0", "metrics"]
    assert record["status"] == "ok"
    assert record["warmup_frames"] == 1
    assert record["timed_frames"] == 1
    assert record["cold_first_infer_sec"] == 0.5
    assert record["yolo_raw_infer_sec"] == 0.01
    assert record["yolo_raw_infer_fps"] == 100.0
    assert record["yolo_raw_infer_latency_ms_p50"] == 10.0
    assert record["max_persons"] == 1
    assert record["multi_person_frames"] == 0
    assert record["review_required"] is False
    assert record["gate_status"] == "ok"

    json.dumps(
        {
            "status": "ok",
            "records": [record],
        },
        ensure_ascii=False,
    )
    for field in (
        "backend",
        "device",
        "delegate",
        "active_delegate",
        "delegate_fallback_reason",
        "imgsz",
        "half",
        "adapter_warmup",
        "warmup_shape",
        "warmup_frames",
        "timed_frames",
        "init_sec",
        "peak_rss_mb",
        "rss_delta_mb",
        "cold_first_infer_sec",
        "timed_latency_ms_p50",
        "timed_latency_ms_p90",
        "timed_latency_ms_p99",
        "yolo_raw_infer_latency_ms_p50",
        "yolo_raw_infer_latency_ms_p90",
        "yolo_raw_infer_latency_ms_p99",
        "yolo_raw_infer_fps",
        "yolo_miss_rate",
        "max_persons",
        "multi_person_frames",
        "review_required",
        "gate_status",
        "yolo_body_core_jitter_median",
    ):
        assert field in record


class _FakeTime:
    def __init__(self) -> None:
        self.value = 0.0

    def perf_counter(self) -> float:
        self.value += 0.01
        return self.value

    def strftime(self, fmt: str) -> str:
        return "2026-06-01 00:00:00"


def _frame_with_shift(shift: float) -> FrameResult:
    landmarks = [
        Landmark(x=0.0, y=0.0, confidence=None, synthetic=True)
        for _ in range(33)
    ]
    for order, idx in enumerate(BODY_CORE_V1_VALID_INDICES):
        landmarks[idx] = Landmark(
            x=0.1 + shift + order * 0.001,
            y=0.2 + order * 0.001,
            confidence=0.9,
            synthetic=False,
        )
    return FrameResult(pose33=tuple(landmarks), meta={"num_persons": 1})
