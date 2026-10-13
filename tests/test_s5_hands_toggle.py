# -*- coding: utf-8 -*-
"""S5 realtime Hands toggle tests (Issue #41)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apps import app_ui  # noqa: E402
from apps.cli import main as app_main  # noqa: E402


def test_cli_no_hands_defaults_to_hands_enabled(monkeypatch):
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(sys, "argv", ["main.py", "--source", "0", "--no-show"])
    monkeypatch.setattr(
        app_main,
        "run",
        lambda source, **kwargs: calls.append({"source": source, **kwargs}),
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


def test_cli_no_hands_disables_single_thread_preview(monkeypatch):
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(sys, "argv", ["main.py", "--source", "0", "--no-show", "--no-hands"])
    monkeypatch.setattr(
        app_main,
        "run",
        lambda source, **kwargs: calls.append({"source": source, **kwargs}),
    )

    app_main.main()

    assert calls[0]["enable_hands"] is False


def test_cli_no_hands_disables_parallel_offline_pipeline(monkeypatch):
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--source", "sample.mp4", "--workers", "2", "--no-show", "--no-hands"],
    )
    monkeypatch.setattr(
        app_main,
        "run_parallel",
        lambda source, **kwargs: calls.append({"source": source, **kwargs}),
    )

    app_main.main()

    assert len(calls) == 1
    assert calls[0]["source"] == "sample.mp4"
    assert calls[0]["enable_hands"] is False
    assert calls[0]["pose_variant"] == "full"
    assert calls[0]["workers"] == 2


def test_ui_state_defaults_to_hands_enabled():
    app = object.__new__(app_ui.App)
    app.source_var = _Var("0")
    app.pose_var = _Var("full")
    app.workers_var = _Var(2)
    app.enable_hands_var = _Var(True)
    app.save_var = _Var(False)
    app.out_var = _Var("")
    app._source_state = app_ui.InputSourceState()
    app._source_state.select_camera(0)
    app._camera_entries = []

    state = app_ui.App._collect_state(app)

    assert state.enable_hands is True
    assert state.source == "0"
    assert state.pose_variant == "full"


def test_ui_state_can_disable_hands():
    app = object.__new__(app_ui.App)
    app.source_var = _Var("video.mp4")
    app.pose_var = _Var("heavy")
    app.workers_var = _Var(4)
    app.enable_hands_var = _Var(False)
    app.save_var = _Var(True)
    app.out_var = _Var("out.mp4")
    app._source_state = app_ui.InputSourceState()
    app._source_state.select_video("video.mp4")
    app._camera_entries = []

    state = app_ui.App._collect_state(app)

    assert state.enable_hands is False
    assert state.workers == 4
    assert state.out_path == "out.mp4"


def test_ui_session_pipelines_use_state_enable_hands(monkeypatch):
    """界面的手部开关进入会话配置，推理 stage 按配置构造引擎；站位检测固定 lite、不跑手部（bf8c908）。

    P6 起取代对 app_ui 源码的 AST 检查：引擎在 core.live 的 stage 里构造。
    """
    import numpy as np

    import core.live.stages as stages
    from core.live import LiveControls
    from core.live.stages import FramePacket, OccupancyStage, PoseStage, StageContext
    from core.media import PRIMARY

    specs = []

    class _Engine:
        def __init__(self, spec):
            specs.append(spec)

        def next_timestamp_ms(self, **_kwargs):
            return 0

        def infer(self, frame, *, timestamp_ms=None):
            return None, []

        def close(self):
            pass

    monkeypatch.setattr(stages, "create_engine", lambda spec, models_dir=None: _Engine(spec))
    monkeypatch.setattr(stages, "models_dir", lambda: Path("models"))

    for source, source2 in (("0", None), ("0", "1"), ("video.mp4", None)):
        for hands in (True, False):
            state = app_ui.UiState(source=source, source2=source2, pose_variant="heavy", workers=1,
                                   enable_hands=hands, record_skeleton=True)
            config = app_ui.App._session_config(state)
            assert config.enable_hands is hands
            specs.clear()
            PoseStage(config.roles).open(StageContext(config, LiveControls(), lambda *_: None))
            assert specs and all(spec.enable_hands is hands for spec in specs)
            assert all(spec.pose_variant == "heavy" for spec in specs)

    # 站位检测管线：骨架关的双摄会话里启用后构造，恰一个，lite + 无手部，与界面开关无关。
    state = app_ui.UiState(source="0", source2="1", pose_variant="heavy", workers=1, enable_hands=True)
    config = app_ui.App._session_config(state)
    controls = LiveControls()
    controls.arm_occupancy("exam", True)
    specs.clear()
    stage = OccupancyStage(lambda _pose, _roi: True)
    frame = np.zeros((8, 8, 3), np.uint8)
    stage.process(FramePacket(0, {PRIMARY: frame}, out={PRIMARY: frame.copy()}),
                  StageContext(config, controls, lambda *_: None))
    assert len(specs) == 1
    assert specs[0].pose_variant == "lite" and specs[0].enable_hands is False


class _Var:
    def __init__(self, value: object) -> None:
        self.value = value

    def get(self) -> object:
        return self.value
