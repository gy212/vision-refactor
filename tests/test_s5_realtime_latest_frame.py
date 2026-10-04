# -*- coding: utf-8 -*-
"""S5 latest-frame realtime smoke tests (Issue #42)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apps import main as app_main  # noqa: E402


def test_latest_frame_queue_keeps_only_newest_and_counts_drops():
    q = app_main.LatestFrameQueue()

    q.put(app_main.LatestFrameItem(0, "old", 1.0, 0))
    q.put(app_main.LatestFrameItem(1, "new", 2.0, 33))

    latest = q.get_latest()

    assert latest is not None
    assert latest.index == 1
    assert latest.frame == "new"
    assert q.get_latest() is None
    assert q.dropped_frames == 1


def test_latest_frame_metrics_report_p90_latency():
    metrics = app_main.RealtimeLatestFrameMetrics(
        captured_frames=4,
        processed_frames=3,
        rendered_frames=3,
        dropped_frames=1,
        capture_fps=40.0,
        infer_fps=30.0,
        render_fps=25.0,
        latency_p90_ms=12.345,
    )

    payload = app_main._metrics_to_dict(metrics)

    assert payload["capture_fps"] == 40.0
    assert payload["infer_fps"] == 30.0
    assert payload["render_fps"] == 25.0
    assert payload["latency_p90_ms"] == 12.345
    assert payload["dropped_frames"] == 1


def test_cli_latest_frame_smoke_is_opt_in(monkeypatch, capsys):
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--source",
            "sample.mp4",
            "--realtime-latest-frame",
            "--limit-frames",
            "5",
            "--no-show",
            "--no-hands",
        ],
    )
    monkeypatch.setattr(
        app_main,
        "run_realtime_latest_frame_smoke",
        lambda source, **kwargs: calls.append({"source": source, **kwargs})
        or app_main.RealtimeLatestFrameMetrics(
            captured_frames=5,
            processed_frames=4,
            rendered_frames=4,
            dropped_frames=1,
            capture_fps=50.0,
            infer_fps=40.0,
            render_fps=40.0,
            latency_p90_ms=10.0,
        ),
    )

    app_main.main()

    assert calls == [
        {
            "source": "sample.mp4",
            "show": False,
            "pose_variant": "full",
            "enable_hands": False,
            "delegate": "cpu",
            "limit_frames": 5,
        }
    ]
    out = capsys.readouterr().out
    assert "latency_p90_ms" in out
    assert "dropped_frames" in out


def test_default_cli_still_uses_run_not_latest_frame(monkeypatch):
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(sys, "argv", ["main.py", "--source", "0", "--no-show"])
    monkeypatch.setattr(app_main, "run", lambda source, **kwargs: calls.append({"source": source, **kwargs}))
    monkeypatch.setattr(
        app_main,
        "run_realtime_latest_frame_smoke",
        lambda *args, **kwargs: calls.append({"source": "latest"}),
    )

    app_main.main()

    assert calls == [
        {
            "source": "0",
            "show": False,
            "out_path": None,
            "pose_variant": "full",
            "enable_hands": True,
            "delegate": "cpu",
        }
    ]


def test_cli_delegate_gpu_is_explicit_opt_in(monkeypatch):
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(sys, "argv", ["main.py", "--source", "0", "--no-show", "--delegate", "gpu"])
    monkeypatch.setattr(app_main, "run", lambda source, **kwargs: calls.append({"source": source, **kwargs}))

    app_main.main()

    assert calls == [
        {
            "source": "0",
            "show": False,
            "out_path": None,
            "pose_variant": "full",
            "enable_hands": True,
            "delegate": "gpu",
        }
    ]


def test_latest_frame_smoke_runs_with_fake_capture_and_pipeline(monkeypatch):
    pipeline_cfgs = []

    class FakeCap:
        def __init__(self) -> None:
            self.index = 0
            self.released = False

        def isOpened(self) -> bool:
            return True

        def set(self, prop: int, value: float) -> None:
            pass

        def get(self, prop: int) -> float:
            return 30.0

        def read(self):
            if self.index >= 4:
                return False, None
            self.index += 1
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    class FakePipeline:
        def __init__(self, *, models_dir, cfg) -> None:
            pipeline_cfgs.append(cfg)

        def annotate(self, frame, *, timestamp_ms: int):
            return frame.copy(), ["SQUAT"]

    monkeypatch.setattr(app_main, "_open_capture", lambda source: FakeCap())
    monkeypatch.setattr(app_main, "models_dir", lambda: Path("models"))
    monkeypatch.setattr(app_main, "MediaPipePipeline", FakePipeline)
    monkeypatch.setattr(app_main.cv2, "destroyAllWindows", lambda: None)

    metrics = app_main.run_realtime_latest_frame_smoke(
        "sample.mp4",
        show=False,
        pose_variant="heavy",
        enable_hands=False,
        delegate="gpu",
        limit_frames=4,
    )

    assert metrics.captured_frames == 4
    assert metrics.processed_frames >= 1
    assert metrics.rendered_frames >= 1
    assert metrics.latency_p90_ms is not None
    assert pipeline_cfgs
    assert pipeline_cfgs[0].pose_variant == "heavy"
    assert pipeline_cfgs[0].enable_hands is False
    assert pipeline_cfgs[0].running_mode == "video"
    assert pipeline_cfgs[0].delegate == "gpu"


def test_latest_frame_smoke_reraises_worker_failures(monkeypatch):
    class FakeCap:
        def isOpened(self) -> bool:
            return True

        def set(self, prop: int, value: float) -> None:
            pass

        def get(self, prop: int) -> float:
            return 30.0

        def read(self):
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self) -> None:
            pass

    class BrokenPipeline:
        def __init__(self, *, models_dir, cfg) -> None:
            raise ValueError("pipeline init failed")

    monkeypatch.setattr(app_main, "_open_capture", lambda source: FakeCap())
    monkeypatch.setattr(app_main, "models_dir", lambda: Path("models"))
    monkeypatch.setattr(app_main, "MediaPipePipeline", BrokenPipeline)
    monkeypatch.setattr(app_main.cv2, "destroyAllWindows", lambda: None)

    try:
        app_main.run_realtime_latest_frame_smoke(
            "sample.mp4",
            show=False,
            limit_frames=1,
        )
    except RuntimeError as exc:
        assert isinstance(exc.__cause__, ValueError)
        assert str(exc.__cause__) == "pipeline init failed"
    else:
        raise AssertionError("worker failure should be re-raised")


def test_default_run_and_ui_worker_are_not_replaced_by_latest_frame_path():
    main_tree = ast.parse(Path(app_main.__file__).read_text(encoding="utf-8"))
    run_func = next(
        node for node in ast.walk(main_tree) if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == "LatestFrameQueue"
        for node in ast.walk(run_func)
    )

    app_ui_source = (_REPO_ROOT / "apps" / "app_ui.py").read_text(encoding="utf-8")
    assert "LatestFrameQueue" not in app_ui_source
    assert "realtime_latest_frame" not in app_ui_source
