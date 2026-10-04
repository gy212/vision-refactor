# -*- coding: utf-8 -*-
"""S5 realtime Hands toggle tests (Issue #41)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apps import app_ui, main as app_main  # noqa: E402


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

    class FakeCap:
        def isOpened(self) -> bool:
            return True

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--source", "sample.mp4", "--workers", "2", "--no-show", "--no-hands"],
    )
    monkeypatch.setattr(app_main, "_open_capture", lambda source: FakeCap())
    monkeypatch.setattr(
        app_main,
        "_process_video_multithread",
        lambda **kwargs: calls.append(kwargs),
    )

    app_main.main()

    assert len(calls) == 1
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

    state = app_ui.App._collect_state(app)

    assert state.enable_hands is False
    assert state.workers == 4
    assert state.out_path == "out.mp4"


def _occupancy_pipeline_configs(tree: ast.AST) -> list[ast.Call]:
    """``occupancy_pipe = MediaPipePipeline(cfg=PipelineConfig(...))`` 中的 PipelineConfig。

    考试 / 学生练习的站位检测只用髋部中点判断是否到位，固定跑 lite pose、不跑手部，
    与界面的手部开关无关（bf8c908 引入）。
    """
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "occupancy_pipe" for t in node.targets):
            continue
        value = node.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "MediaPipePipeline"
        ):
            continue
        found.extend(
            keyword.value
            for keyword in value.keywords
            if keyword.arg == "cfg" and isinstance(keyword.value, ast.Call)
        )
    return found


def test_ui_worker_pipelines_use_state_enable_hands():
    source = Path(app_ui.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    cfg_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PipelineConfig"
        and any(keyword.arg == "enable_hands" for keyword in node.keywords)
    ]

    # 站位检测管线是唯一的例外：恰有一处，且保持 lite、不跑手部。
    occupancy_cfgs = _occupancy_pipeline_configs(tree)
    assert len(occupancy_cfgs) == 1
    occupancy_kwargs = {keyword.arg: keyword.value for keyword in occupancy_cfgs[0].keywords}
    assert isinstance(occupancy_kwargs.get("enable_hands"), ast.Constant)
    assert occupancy_kwargs["enable_hands"].value is False
    assert isinstance(occupancy_kwargs.get("pose_variant"), ast.Constant)
    assert occupancy_kwargs["pose_variant"].value == "lite"

    # 其余实时 / 离线 worker 管线必须跟随界面的手部开关。
    worker_calls = [call for call in cfg_calls if all(call is not occ for occ in occupancy_cfgs)]
    assert len(worker_calls) >= 2
    for call in worker_calls:
        enable_arg = next(keyword.value for keyword in call.keywords if keyword.arg == "enable_hands")
        where = f"apps/app_ui.py:{call.lineno}"
        assert isinstance(enable_arg, ast.Attribute), where
        assert isinstance(enable_arg.value, ast.Name), where
        assert enable_arg.value.id == "state", where
        assert enable_arg.attr == "enable_hands", where


class _Var:
    def __init__(self, value: object) -> None:
        self.value = value

    def get(self) -> object:
        return self.value
