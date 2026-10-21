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

from apps.cli import main as app_main  # noqa: E402
import core.live.stages as live_stages  # noqa: E402
from core.live import InputFrames  # noqa: E402
from core.media import PRIMARY  # noqa: E402


class FakeVideoInput:
    """视频文件形态的假输入：读 ``count`` 帧后结束；``delay`` 模拟采集间隔。"""

    is_file = True
    roles = (PRIMARY,)

    def __init__(self, count=None, delay=0.0) -> None:
        self.count = count
        self.delay = delay
        self.read_calls = 0
        self.closed = False

    def wait_ready(self, cancel):
        return {}

    def first_cursor(self):
        return {}

    def snapshot(self, after):
        return None

    def start(self, cancel):
        pass

    def actual_sizes(self):
        return {PRIMARY: (8, 8)}

    def fps(self):
        return 30.0

    def total_frames(self):
        return self.count or 0

    def read(self, cancel):
        if self.count is not None and self.read_calls >= self.count:
            return None
        self.read_calls += 1
        if self.delay:
            import time

            time.sleep(self.delay)
        return InputFrames({PRIMARY: np.zeros((8, 8, 3), dtype=np.uint8)})

    def capture_feed(self):
        return None

    def close(self, *, keep_warm):
        self.closed = True


def test_latest_frame_mode_only_infers_newest_and_counts_drops(monkeypatch):
    """采集快于推理时，推理只取最新帧，被覆盖的帧计入 dropped。"""
    import time

    class SlowPipeline:
        def next_timestamp_ms(self, **_kwargs):
            return 0

        def infer(self, frame, *, timestamp_ms=None):
            time.sleep(0.01)
            return None, []

        def close(self):
            pass

    monkeypatch.setattr(app_main, "open_input", lambda source: FakeVideoInput(count=40))
    monkeypatch.setattr(live_stages, "create_engine", lambda spec, *, models_dir=None: SlowPipeline())
    monkeypatch.setattr(app_main.cv2, "destroyAllWindows", lambda: None)

    metrics = app_main.run_realtime_latest_frame_smoke("sample.mp4", show=False, limit_frames=40)

    assert metrics.captured_frames == 40
    assert metrics.dropped_frames > 0
    assert metrics.processed_frames + metrics.dropped_frames == 40
    assert metrics.rendered_frames == metrics.processed_frames


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

    class FakePipeline:
        def __init__(self, *, models_dir, cfg) -> None:
            pipeline_cfgs.append(cfg)

        def next_timestamp_ms(self, **_kwargs):
            return 0

        def infer(self, frame, *, timestamp_ms=None):
            return None, []

        def close(self):
            pass

    source = FakeVideoInput(count=None)
    monkeypatch.setattr(app_main, "open_input", lambda _source: source)
    monkeypatch.setattr(live_stages, "models_dir", lambda: Path("models"))
    monkeypatch.setattr(live_stages, "create_engine", lambda spec, *, models_dir=None: FakePipeline(models_dir=models_dir, cfg=spec))
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
    assert source.closed
    assert pipeline_cfgs
    assert pipeline_cfgs[0].pose_variant == "heavy"
    assert pipeline_cfgs[0].enable_hands is False
    assert pipeline_cfgs[0].running_mode == "video"
    assert pipeline_cfgs[0].delegate == "gpu"


def test_latest_frame_smoke_reraises_worker_failures(monkeypatch):
    class BrokenPipeline:
        def __init__(self, *, models_dir, cfg) -> None:
            raise ValueError("pipeline init failed")

    monkeypatch.setattr(app_main, "open_input", lambda _source: FakeVideoInput())
    monkeypatch.setattr(live_stages, "models_dir", lambda: Path("models"))
    monkeypatch.setattr(live_stages, "create_engine", lambda spec, *, models_dir=None: BrokenPipeline(models_dir=models_dir, cfg=spec))
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
        isinstance(node, ast.keyword) and node.arg == "latest_frame"
        for node in ast.walk(run_func)
    )

    # Tk 桌面：入口壳与 apps/tk/ 下的组合根、控制器、视图都不接 latest-frame 冒烟路径
    tk_files = [_REPO_ROOT / "apps" / "app_ui.py", *sorted((_REPO_ROOT / "apps" / "tk").rglob("*.py"))]
    for path in tk_files:
        source = path.read_text(encoding="utf-8")
        assert "latest_frame" not in source, path
        assert "realtime_latest_frame" not in source, path
