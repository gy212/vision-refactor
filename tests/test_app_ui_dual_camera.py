# -*- coding: utf-8 -*-
"""Tkinter 双摄骨架测试（issue #59 裁剪版）。

不创建真实 Tk root，只验证 source2 收敛点和双 VIDEO pipeline 结构。
"""
from __future__ import annotations

import ast
import sys
import threading
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apps import app_ui  # noqa: E402
from apps.camera_enum import CameraEntry  # noqa: E402


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _GridWidget:
    """Headless ttk widget fake that preserves grid options across grid_remove()."""

    def __init__(self) -> None:
        self.grid_options: dict[str, object] = {}
        self.visible = False

    def grid(self, **kwargs) -> None:
        if kwargs:
            self.grid_options.update(kwargs)
        self.visible = True

    grid_configure = grid

    def grid_remove(self) -> None:
        self.visible = False


class _GridContainer:
    def __init__(self) -> None:
        self.row_weights: dict[int, int] = {}
        self.column_weights: dict[int, int] = {}

    def rowconfigure(self, index: int, *, weight: int) -> None:
        self.row_weights[index] = weight

    def columnconfigure(self, index: int, *, weight: int) -> None:
        self.column_weights[index] = weight


class _ImageWidget:
    def __init__(self) -> None:
        self.images: list[object] = []

    def configure(self, *, image) -> None:
        self.images.append(image)


class _TickRoot:
    def __init__(self) -> None:
        self.after_calls: list[tuple[int, object]] = []

    def after(self, delay: int, callback) -> None:
        self.after_calls.append((delay, callback))


def _app_stub(
    *,
    source: str = "0",
    second_label: str = app_ui.NO_SECOND_CAMERA,
    rotate: str = "0°",
    rotate2: str = "0°",
    record_skeleton: bool = False,
    auto_compare: bool = True,
):
    app = object.__new__(app_ui.App)
    app.source_var = _Var(source)
    app.pose_var = _Var("full")
    app.workers_var = _Var(2)
    app.enable_hands_var = _Var(True)
    app.save_var = _Var(False)
    app.out_var = _Var("")
    app.camera_choice_var_2 = _Var(second_label)
    app.rotate_var = _Var(rotate)
    app.rotate_var_2 = _Var(rotate2)
    app.record_skeleton_var = _Var(record_skeleton)
    app.auto_compare_var = _Var(auto_compare)
    app._source_state = app_ui.InputSourceState()
    if source.isdigit():
        app._source_state.select_camera(int(source))
    else:
        app._source_state.select_video(source)
    app._camera_entries = [
        CameraEntry(label="摄像头 0", index=0),
        CameraEntry(label="摄像头 1", index=1),
    ]
    return app


def _layout_stub():
    app = object.__new__(app_ui.App)
    app._preview_right = _GridContainer()
    app._dual_preview_visible = True
    app._dual_preview_layout = "stacked"
    app.preview = _GridWidget()
    app.preview2 = _GridWidget()
    app.preview.grid(row=0, column=0, sticky="nsew")
    app.preview2.grid(row=1, column=0, sticky="nsew")
    return app


def _dual_preview_stub(*, generation: int = 1, record_skeleton: bool = False):
    app = object.__new__(app_ui.App)
    app._dual_preview_queue = Queue(maxsize=1)
    app._dual_preview_lock = threading.Lock()
    app._dual_render_lock = threading.Lock()
    app._active_dual_generation = generation
    app._dual_preview_stage = "raw"
    app._dual_recording_ready = False
    app._preview_wh = (0, 0)
    app._preview_wh2 = (0, 0)
    app._queue = Queue(maxsize=1)
    app._queue2 = Queue(maxsize=1)
    app._dual_metrics_lock = threading.Lock()
    app._dual_startup_metrics = {
        generation: app_ui._DualStartupMetrics(
            session_generation=generation,
            primary_index=0,
            secondary_index=1,
            record_skeleton=record_skeleton,
            start_click=1.0,
            outcome="starting",
        )
    }
    app._dual_first_render_events = {generation: threading.Event()}
    app.root = _TickRoot()
    app.preview = _ImageWidget()
    app.preview2 = _ImageWidget()
    app.actions_var = _Var("-")
    app.status_var = _Var("就绪")
    app._photo = None
    app._photo2 = None
    app._runtime_rotate_lock = threading.Lock()
    app._runtime_rotate = 0
    app._runtime_rotate2 = 0
    app._dual_auto_compare = True
    app._exam_active = False
    app._exam_pending_row = None
    app._exam_lock = threading.Lock()
    app._exam_occupancy_armed = False
    app._exam_roi = (0.1, 0.1, 0.9, 0.9)
    app._exam_occupancy_stride = 3
    app._exam_occupancy_queue = None
    app._current_auto_compare = lambda prefer_ui=False: True
    app._get_runtime_rotate = lambda which="primary": (
        app._runtime_rotate if which == "primary" else app._runtime_rotate2
    )
    app._drain_recording_compare_updates = lambda: None
    app._drain_dual_preview_layout = lambda: None
    app._drain_camera_enum_results = lambda: None
    app._refresh_recording_status = lambda: None
    return app


def _function_node(name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(app_ui.__file__).read_text(encoding="utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _app_method_node(name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(app_ui.__file__).read_text(encoding="utf-8"))
    app_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "App"
    )
    return next(
        node
        for node in app_class.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _named_calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and (
            (isinstance(item.func, ast.Name) and item.func.id == name)
            or (isinstance(item.func, ast.Attribute) and item.func.attr == name)
        )
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("0°", 0), ("90°", 90), ("180°", 180), ("270°", 270)],
)
def test_parse_rotate_accepts_only_supported_choices(text, expected):
    assert app_ui._parse_rotate(text) == expected


@pytest.mark.parametrize("text", ["", "45°", "360°", "-90°", "abc", None])
def test_parse_rotate_rejects_unsupported_values(text):
    assert app_ui._parse_rotate(text) == 0


@pytest.mark.parametrize(
    ("degrees", "expected"),
    [(0, 90), (90, 180), (180, 270), (270, 0), (45, 90)],
)
def test_next_rotate_cw_steps_quarter_turns(degrees, expected):
    assert app_ui._next_rotate_cw(degrees) == expected


@pytest.mark.parametrize(
    ("degrees", "expected"),
    [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°"), (15, "0°")],
)
def test_degrees_to_rotate_label(degrees, expected):
    assert app_ui._degrees_to_rotate_label(degrees) == expected


def test_fit_frame_to_size_resizes_when_mismatch():
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    fitted = app_ui._fit_frame_to_size(frame, (3, 2))
    assert fitted.shape[:2] == (2, 3)


def test_fit_frame_to_size_returns_same_when_match():
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    assert app_ui._fit_frame_to_size(frame, (6, 4)) is frame


def test_runtime_rotate_primary_and_secondary_are_isolated():
    app = object.__new__(app_ui.App)
    app._runtime_rotate_lock = __import__("threading").Lock()
    app._runtime_rotate = 0
    app._runtime_rotate2 = 0
    app.rotate_var = _Var("0°")
    app.rotate_var_2 = _Var("0°")
    app.status_var = _Var("")
    app._exam_active = False

    app_ui.App._set_runtime_rotate(app, 90, which="primary", announce=False)
    app_ui.App._set_runtime_rotate(app, 180, which="secondary", announce=False)

    assert app_ui.App._get_runtime_rotate(app, "primary") == 90
    assert app_ui.App._get_runtime_rotate(app, "secondary") == 180
    assert app.rotate_var.get() == "90°"
    assert app.rotate_var_2.get() == "180°"

    app_ui.App._bump_runtime_rotate_cw(app, "primary")
    assert app_ui.App._get_runtime_rotate(app, "primary") == 180
    assert app_ui.App._get_runtime_rotate(app, "secondary") == 180


@pytest.mark.parametrize(
    ("degrees", "turns"),
    [(90, -1), (180, 2), (270, 1)],
)
def test_apply_rotation_matches_clockwise_degree_contract(degrees, turns):
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

    rotated = app_ui._apply_rotation(frame, degrees)

    np.testing.assert_array_equal(rotated, np.rot90(frame, k=turns))


@pytest.mark.parametrize("degrees", [0, 45])
def test_apply_rotation_returns_original_frame_when_no_rotation_applies(degrees):
    frame = np.zeros((2, 3, 3), dtype=np.uint8)

    assert app_ui._apply_rotation(frame, degrees) is frame


@pytest.mark.parametrize(
    ("degrees", "expected"),
    [(0, (1280, 720)), (90, (720, 1280)), (180, (1280, 720)), (270, (720, 1280))],
)
def test_rotated_size_matches_rotated_frame_dimensions(degrees, expected):
    assert app_ui._rotated_size(1280, 720, degrees) == expected


@pytest.mark.parametrize(
    ("size1", "size2", "expected"),
    [
        ((720, 1280), (1080, 1920), "side_by_side"),
        ((1280, 720), (1920, 1080), "stacked"),
        ((720, 1280), (1920, 1080), "stacked"),
        ((1280, 720), (1080, 1920), "stacked"),
        ((720, 720), (720, 1280), "stacked"),
    ],
)
def test_choose_dual_preview_layout_handles_portrait_landscape_and_mixed(
    size1, size2, expected
):
    assert app_ui._choose_dual_preview_layout(size1, size2) == expected


def test_set_dual_preview_layout_places_portrait_views_side_by_side():
    app = _layout_stub()

    app_ui.App._set_dual_preview_layout(app, "side_by_side")

    assert app.preview.grid_options == {"row": 0, "column": 0, "sticky": "nsew"}
    assert app.preview2.grid_options == {"row": 0, "column": 1, "sticky": "nsew"}
    assert app._preview_right.row_weights == {0: 1, 1: 0}
    assert app._preview_right.column_weights == {0: 1, 1: 1}


def test_set_dual_preview_layout_places_landscape_and_mixed_views_stacked():
    app = _layout_stub()
    app_ui.App._set_dual_preview_layout(app, "side_by_side")

    app_ui.App._set_dual_preview_layout(app, "stacked")

    assert app.preview.grid_options == {"row": 0, "column": 0, "sticky": "nsew"}
    assert app.preview2.grid_options == {"row": 1, "column": 0, "sticky": "nsew"}
    assert app._preview_right.row_weights == {0: 1, 1: 1}
    assert app._preview_right.column_weights == {0: 1, 1: 0}


def test_hiding_second_preview_collapses_both_optional_grid_tracks():
    app = _layout_stub()
    app_ui.App._set_dual_preview_layout(app, "side_by_side")

    app_ui.App._set_dual_preview_visible(app, False)

    assert app.preview2.visible is False
    assert app._preview_right.row_weights[1] == 0
    assert app._preview_right.column_weights[1] == 0


def test_collect_state_sets_source2_when_two_distinct_cameras_selected():
    app = _app_stub(source="0", second_label="摄像头 1")

    state = app_ui.App._collect_state(app)

    assert state.source == "0"
    assert state.source2 == "1"
    assert state.enable_hands is True


def test_collect_state_captures_each_camera_rotation():
    app = _app_stub(
        source="0",
        second_label="摄像头 1",
        rotate="90°",
        rotate2="270°",
    )

    state = app_ui.App._collect_state(app)

    assert state.rotate == 90
    assert state.rotate2 == 270


def test_collect_state_captures_dual_record_skeleton_choice():
    app = _app_stub(
        source="0",
        second_label="摄像头 1",
        record_skeleton=True,
    )

    state = app_ui.App._collect_state(app)

    assert state.record_skeleton is True


def test_collect_state_captures_dual_auto_compare_choice():
    app = _app_stub(
        source="0",
        second_label="摄像头 1",
        auto_compare=False,
    )

    state = app_ui.App._collect_state(app)

    assert state.auto_compare is False


def test_app_initializes_dual_record_skeleton_toggle_disabled():
    init = _app_method_node("__init__")
    assignments = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "record_skeleton_var"
            for target in node.targets
        )
    ]

    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Call)
    assert any(
        keyword.arg == "value"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in value.keywords
    )


def test_app_initializes_auto_compare_toggle_enabled():
    init = _app_method_node("__init__")
    assignments = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "auto_compare_var"
            for target in node.targets
        )
    ]

    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Call)
    assert any(
        keyword.arg == "value"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in value.keywords
    )


def test_collect_state_captures_record_skeleton_for_single_camera():
    """全局骨架开关：单摄同样透传勾选状态（默认关，勾选后预览/录制均带骨架）。"""
    app = _app_stub(
        source="0",
        second_label=app_ui.NO_SECOND_CAMERA,
        record_skeleton=True,
    )

    state = app_ui.App._collect_state(app)

    assert state.record_skeleton is True


def test_collect_state_defaults_record_skeleton_off_for_single_camera():
    app = _app_stub(
        source="0",
        second_label=app_ui.NO_SECOND_CAMERA,
        record_skeleton=False,
    )

    state = app_ui.App._collect_state(app)

    assert state.record_skeleton is False


def test_collect_state_keeps_auto_compare_true_for_single_camera():
    app = _app_stub(
        source="0",
        second_label=app_ui.NO_SECOND_CAMERA,
        auto_compare=False,
    )

    state = app_ui.App._collect_state(app)

    assert state.auto_compare is True


def test_collect_state_leaves_source2_none_for_single_camera_mode():
    app = _app_stub(source="0", second_label=app_ui.NO_SECOND_CAMERA)

    state = app_ui.App._collect_state(app)

    assert state.source2 is None


def test_collect_state_ignores_second_camera_for_video_file_source():
    app = _app_stub(
        source="input.mp4",
        second_label="摄像头 1",
        record_skeleton=True,
    )

    state = app_ui.App._collect_state(app)

    assert state.source == "input.mp4"
    assert state.source2 is None
    # 全局骨架开关与双摄无关；文件源同样透传勾选状态（离线路径仍始终 annotate）。
    assert state.record_skeleton is True


def test_collect_state_rejects_same_camera_for_both_views():
    app = _app_stub(source="0", second_label="摄像头 0")

    with pytest.raises(ValueError, match="两个摄像头不能选同一个"):
        app_ui.App._collect_state(app)


def test_second_camera_combobox_binds_warmup_selection_handler():
    build_ui = _app_method_node("_build_ui")
    bindings = [
        call
        for call in _named_calls(build_ui, "bind")
        if isinstance(call.func, ast.Attribute)
        and ast.unparse(call.func.value) == "self.camera_combo_2"
    ]

    assert len(bindings) == 1
    assert [ast.unparse(arg) for arg in bindings[0].args] == [
        "'<<ComboboxSelected>>'",
        "self._on_camera_2_selected",
    ]


def test_app_warmup_pool_is_the_only_camera_owner():
    init = _app_method_node("__init__")
    calls = _named_calls(init, "CameraWarmupPool")

    assert len(calls) == 1
    capture_factory = next(
        keyword.value for keyword in calls[0].keywords if keyword.arg == "capture_factory"
    )
    assert ast.unparse(capture_factory) == "self._open_camera_process"
    # 所有摄像头打开都经池；worker 与选择回调不再直接调用 open_camera。
    for name in ("_worker_loop", "_worker_loop_parallel_camera", "_sync_camera_warmup"):
        assert _named_calls(_function_node(name), "open_camera") == []


def test_dual_start_dispatches_state_without_claiming_in_parent_worker(monkeypatch):
    worker_calls: list[object] = []

    class _Pool:
        def __getattr__(self, name):
            pytest.fail(f"parent worker must not call pool.{name}")

    state = SimpleNamespace(source="0", source2="1", workers=4)
    app = SimpleNamespace(
        _camera_warmup_pool=_Pool(),
        _stop_evt=threading.Event(),
        _worker_loop_dual_camera=worker_calls.append,
    )
    monkeypatch.setattr(
        app_ui,
        "open_camera",
        lambda _index: pytest.fail("dual start must reuse CameraWarmupPool"),
    )

    app_ui.App._worker_loop(app, state)

    assert worker_calls == [state]


def test_dual_wait_failure_posts_status_and_done_exactly_once(monkeypatch):
    statuses: list[str] = []
    done_calls: list[bool] = []
    cancel_calls: list[bool] = []

    class _Pool:
        def wait_pair(self, _primary, _secondary, *, timeout, stop_event):
            assert 0.0 <= timeout <= 15.0
            assert not stop_event.is_set()
            raise RuntimeError("camera 1 unavailable")

        def claim_pair(self, _primary: int, _secondary: int, *, timeout: float):
            pytest.fail("claim must not run after wait_pair fails")

    state = SimpleNamespace(
        source="0",
        source2="1",
        workers=1,
        session_generation=3,
        rotate=0,
        rotate2=0,
        record_skeleton=False,
        pose_variant="full",
        enable_hands=True,
    )
    app = SimpleNamespace(
        _camera_warmup_pool=_Pool(),
        _stop_evt=threading.Event(),
        _post_status=statuses.append,
        _post_done=lambda **_kwargs: done_calls.append(True),
        _invalidate_dual_preview=lambda _generation: None,
        _finish_dual_startup_metrics=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        app_ui,
        "open_camera",
        lambda _index: pytest.fail("claim failure must not fall back to serial open"),
    )

    app_ui.App._worker_loop_dual_camera(app, state)

    assert statuses == ["双摄像头预热失败：camera 1 unavailable"]
    assert done_calls == [True]
    # wait_pair 失败由池自行回收观察到的 pair；worker 未取得代次，不再盲目取消角色。
    assert cancel_calls == []


def test_dual_worker_never_opens_camera_after_pair_claim():
    worker = _function_node("_worker_loop_dual_camera")

    assert _named_calls(worker, "open_camera") == []


def test_dual_preview_queue_is_latest_only_and_rejects_old_generation():
    app = _dual_preview_stub()
    first = np.full((2, 3, 3), 10, dtype=np.uint8)
    latest = np.full((2, 3, 3), 20, dtype=np.uint8)

    assert app_ui.App._post_dual_frame_pair(
        app, first, "first", first, "first2", session_generation=1, stage="raw"
    )
    assert app_ui.App._post_dual_frame_pair(
        app, latest, "latest", latest, "latest2", session_generation=1, stage="raw"
    )
    assert not app_ui.App._post_dual_frame_pair(
        app, first, "old", first, "old2", session_generation=0, stage="raw"
    )

    packet = app._dual_preview_queue.get_nowait()
    assert packet.actions == "latest"
    assert int(packet.frame_rgb[0, 0, 0]) == 20


def test_annotated_stage_drops_queued_and_late_raw_packets():
    app = _dual_preview_stub(record_skeleton=True)
    frame = np.zeros((2, 3, 3), dtype=np.uint8)

    assert app_ui.App._post_dual_frame_pair(
        app, frame, "raw", frame, "raw2", session_generation=1, stage="raw"
    )
    assert app_ui.App._advance_dual_preview_stage(app, 1, "annotated")
    assert app._dual_preview_queue.empty()
    assert not app_ui.App._post_dual_frame_pair(
        app, frame, "late", frame, "late2", session_generation=1, stage="raw"
    )
    assert app_ui.App._post_dual_frame_pair(
        app,
        frame,
        "annotated",
        frame,
        "annotated2",
        session_generation=1,
        stage="annotated",
    )
    assert app._dual_preview_queue.get_nowait().stage == "annotated"


def test_tick_renders_dual_packet_atomically_and_emits_terminal_metrics_once(
    monkeypatch, capsys
):
    app = _dual_preview_stub()
    frame = np.full((2, 3, 3), 7, dtype=np.uint8)
    monkeypatch.setattr(app_ui.ImageTk, "PhotoImage", lambda image: image)

    app_ui.App._post_dual_frame_pair(
        app, frame, "front", frame, "side", session_generation=1, stage="raw"
    )
    app_ui.App._tick(app)

    assert len(app.preview.images) == 1
    assert len(app.preview2.images) == 1
    assert app.actions_var.get() == "front"
    assert app._dual_startup_metrics[1].first_pair_rendered is not None
    assert app._dual_first_render_events[1].is_set()
    assert len(app.root.after_calls) == 1
    assert capsys.readouterr().out == ""

    app_ui.App._set_dual_startup_outcome(app, 1, "running")
    app_ui.App._emit_dual_startup_metrics(app, 1)
    terminal_output = capsys.readouterr().out
    assert terminal_output.count("[dual-startup]") == 1
    assert '"outcome": "running"' in terminal_output

    app_ui.App._post_dual_frame_pair(
        app, frame, "front2", frame, "side2", session_generation=1, stage="raw"
    )
    app_ui.App._tick(app)
    assert capsys.readouterr().out == ""


def test_raw_render_then_claim_failure_emits_failed_terminal_metrics_once(capsys):
    app = _dual_preview_stub()
    metrics = app._dual_startup_metrics[1]
    metrics.first_pair_rendered = metrics.start_click + 0.01

    app_ui.App._emit_dual_startup_metrics(app, 1)
    assert capsys.readouterr().out == ""

    app_ui.App._finish_dual_startup_metrics(
        app,
        1,
        outcome="failed",
        error_type="CameraWarmupError",
    )
    output = capsys.readouterr().out
    assert output.count("[dual-startup]") == 1
    assert '"outcome": "failed"' in output
    assert '"error_type": "CameraWarmupError"' in output

    app_ui.App._finish_dual_startup_metrics(
        app,
        1,
        outcome="failed",
        error_type="CameraWarmupError",
    )
    assert capsys.readouterr().out == ""


def test_active_dual_tick_never_consumes_old_single_queue(monkeypatch):
    app = _dual_preview_stub(generation=2)
    app._queue.put((np.zeros((2, 3, 3), dtype=np.uint8), "single"))
    stale = app_ui.DualPreviewPacket(
        session_generation=1,
        frame_rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        actions="stale",
        frame_rgb2=np.zeros((2, 3, 3), dtype=np.uint8),
        actions2="stale2",
        stage="raw",
        enqueued_at=0.0,
    )
    app._dual_preview_queue.put(stale)
    monkeypatch.setattr(app_ui.ImageTk, "PhotoImage", lambda image: image)

    app_ui.App._tick(app)

    assert app._queue.qsize() == 1
    assert app.preview.images == []
    assert app.preview2.images == []


def test_dual_startup_metrics_expose_all_six_timing_points_once(capsys):
    app = _dual_preview_stub(record_skeleton=True)
    metrics = app._dual_startup_metrics[1]
    metrics.start_click = 10.0
    metrics.pair_ready = 10.1
    metrics.first_pair_enqueued = 10.2
    metrics.first_pair_rendered = 10.3
    metrics.pipeline_ready = 10.4
    metrics.first_annotated_enqueued = 10.5
    metrics.first_annotated_rendered = 10.6
    metrics.outcome = "running"

    app_ui.App._emit_dual_startup_metrics(app, 1)
    app_ui.App._emit_dual_startup_metrics(app, 1, force=True)

    output = capsys.readouterr().out.strip()
    assert output.count("[dual-startup]") == 1
    for name in (
        "pair_ready",
        "first_pair_enqueued",
        "first_pair_rendered",
        "pipeline_ready",
        "first_annotated_enqueued",
        "first_annotated_rendered",
    ):
        assert f'"start_to_{name}_ms"' in output


def test_dual_camera_worker_creates_two_mediapipe_pipelines():
    """双摄主预览两路 + 考试占用 lite 一路，源码中共 3 处 MediaPipePipeline 构造点。"""
    worker = _function_node("_worker_loop_dual_camera")

    calls = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "MediaPipePipeline")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "MediaPipePipeline")
        )
    ]

    assert len(calls) == 3


@pytest.mark.parametrize(
    ("worker_name", "expected_args"),
    [
        ("_worker_loop", [("frame", "self._get_runtime_rotate('primary')")]),
        (
            "_worker_loop_dual_camera",
            [
                ("frame", "self._get_runtime_rotate('primary')"),
                ("frame2", "self._get_runtime_rotate('secondary')"),
            ],
        ),
        (
            "_worker_loop_parallel_camera",
            [("frame", "self._get_runtime_rotate('primary')")],
        ),
    ],
)
def test_all_camera_capture_paths_apply_rotation(worker_name, expected_args):
    """运行时旋转：worker 每帧从 runtime 读角度，不再冻结 UiState.rotate。"""
    calls = _named_calls(_function_node(worker_name), "_apply_rotation")
    actual_args = sorted(
        (ast.unparse(call.args[0]), ast.unparse(call.args[1])) for call in calls
    )

    if worker_name == "_worker_loop_dual_camera":
        assert set(expected_args).issubset(set(actual_args))
        assert (
            "pair[0].frame",
            "self._get_runtime_rotate('primary')",
        ) in actual_args
        assert (
            "pair[1].frame",
            "self._get_runtime_rotate('secondary')",
        ) in actual_args
    else:
        assert actual_args == sorted(expected_args)


def test_shared_single_worker_rotation_is_guarded_to_camera_sources():
    worker = _function_node("_worker_loop")
    guarded_calls = []
    for branch in ast.walk(worker):
        if not isinstance(branch, ast.If) or ast.unparse(branch.test) != "not is_file":
            continue
        for statement in branch.body:
            guarded_calls.extend(_named_calls(statement, "_apply_rotation"))

    assert len(guarded_calls) == 1


def test_parallel_video_path_does_not_apply_camera_rotation():
    worker = _function_node("_worker_loop_parallel_video")

    assert _named_calls(worker, "_apply_rotation") == []
    assert _named_calls(worker, "_rotated_size") == []
    assert _named_calls(worker, "update_session_size") == []


def test_dual_recording_frame_selector_defaults_to_raw_frames():
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame2 = np.ones((2, 3, 3), dtype=np.uint8)
    annotated = np.full((2, 3, 3), 2, dtype=np.uint8)
    annotated2 = np.full((2, 3, 3), 3, dtype=np.uint8)

    selected = app_ui._select_dual_recording_frames(
        frame,
        frame2,
        annotated,
        annotated2,
        record_skeleton=False,
    )

    assert selected[0] is frame
    assert selected[1] is frame2


def test_dual_recording_frame_selector_can_write_annotated_frames():
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame2 = np.ones((2, 3, 3), dtype=np.uint8)
    annotated = np.full((2, 3, 3), 2, dtype=np.uint8)
    annotated2 = np.full((2, 3, 3), 3, dtype=np.uint8)

    selected = app_ui._select_dual_recording_frames(
        frame,
        frame2,
        annotated,
        annotated2,
        record_skeleton=True,
    )

    assert selected[0] is annotated
    assert selected[1] is annotated2


def test_dual_worker_writes_selected_frames_but_previews_annotated_frames():
    worker = _function_node("_worker_loop_dual_camera")
    select_calls = _named_calls(worker, "_select_dual_recording_frames")
    write_calls = _named_calls(worker, "_write_recording_pair")
    preview_calls = [
        call
        for call in _named_calls(worker, "_post_dual_frame_pair")
        if call.args and ast.unparse(call.args[0]) == "annotated"
    ]

    assert len(select_calls) == 1
    assert [ast.unparse(arg) for arg in select_calls[0].args] == [
        "frame",
        "frame2",
        "annotated",
        "annotated2",
    ]
    assert [ast.unparse(arg) for arg in write_calls[0].args] == [
        "record_frame",
        "record_frame2",
    ]
    assert len(preview_calls) == 1
    assert [ast.unparse(arg) for arg in preview_calls[0].args[:4]] == [
        "annotated",
        "actions_text",
        "annotated2",
        "actions_text2",
    ]


def test_dual_worker_finally_always_closes_recording_pair():
    worker = _function_node("_worker_loop_dual_camera")
    finally_calls = [
        call
        for node in ast.walk(worker)
        if isinstance(node, ast.Try)
        for statement in node.finalbody
        for call in _named_calls(statement, "_close_recording_pair")
    ]

    assert len(finally_calls) == 1


def _exercise_dual_runtime_failure(monkeypatch, *, record_skeleton: bool, phase: str, student=False):
    frame = np.zeros((4, 6, 3), dtype=np.uint8)

    class _Capture:
        def __init__(self) -> None:
            self.release_calls = 0
            self.settings = []

        def set(self, prop, value):
            self.settings.append((prop, value))
            return False  # 驱动可以拒绝，后续必须仍使用实际帧大小。

        def isOpened(self) -> bool:
            return True

        def get(self, prop) -> int:
            return 6 if prop == app_ui.cv2.CAP_PROP_FRAME_WIDTH else 4

        def read(self):
            return True, frame.copy()

        def release(self) -> None:
            self.release_calls += 1

    class _RecordingSession:
        def __init__(self) -> None:
            self._size = (6, 4)

        def begin_session(self, **kwargs) -> None:
            if "size" in kwargs:
                self._size = kwargs["size"]

        def update_session_size(self, **kwargs) -> None:
            if "size" in kwargs:
                self._size = kwargs["size"]
            return True

        @property
        def session_size(self):
            return self._size

        def write_frame(self, _frame) -> None:
            pass

    class _Pipeline:
        def __init__(self, *, fail_annotate: bool = False) -> None:
            self.fail_annotate = fail_annotate
            self.close_calls = 0

        def next_timestamp_ms(self, **_kwargs) -> int:
            return 1

        def annotate(self, current_frame, **_kwargs):
            if self.fail_annotate:
                raise RuntimeError("annotate failed")
            return current_frame.copy(), []

        def close(self) -> None:
            self.close_calls += 1

    front_cap = _Capture()
    side_cap = _Capture()
    pipelines: list[_Pipeline] = []

    class _Pool:
        def wait_pair(self, primary, secondary, *, timeout, stop_event):
            expected_size = (1920, 1080) if student else (1280, 720)
            assert (primary, secondary) == (
                app_ui.CaptureSpec(0, *expected_size),
                app_ui.CaptureSpec(1, *expected_size),
            )
            return (
                SimpleNamespace(frame=frame.copy(), sequence=1, generation=1),
                SimpleNamespace(frame=frame.copy(), sequence=1, generation=1),
            )

        def snapshot_pair(self, *_args, **_kwargs):
            return None

        def claim_pair(
            self,
            primary,
            secondary,
            *,
            timeout,
            expected_generations,
            stop_event,
        ):
            assert (primary.index, secondary.index) == (0, 1)
            assert expected_generations == (1, 1)
            assert not stop_event.is_set()
            return front_cap, side_cap

        def cancel(self, _role):
            pass

    def pipeline_factory(**_kwargs):
        pipeline = _Pipeline(
            fail_annotate=phase == "annotate" and not pipelines
        )
        pipelines.append(pipeline)
        return pipeline

    close_pair_calls: list[bool] = []
    done_calls: list[bool] = []

    def write_pair(_front, _side) -> None:
        if phase == "write":
            raise RuntimeError("write failed")

    app = SimpleNamespace(
        _camera_warmup_pool=_Pool(),
        _student_practice_active=student,
        _current_session_generation=1,
        _record_pair_lock=threading.Lock(),
        _rec=_RecordingSession(),
        _rec2=_RecordingSession(),
        _stop_evt=threading.Event(),
        _runtime_rotate_lock=threading.Lock(),
        _runtime_rotate=0,
        _runtime_rotate2=0,
        _dual_auto_compare=True,
        _exam_active=False,
        _exam_pending_row=None,
        _get_runtime_rotate=lambda which="primary": 0,
        _current_auto_compare=lambda prefer_ui=False: True,
        _write_rec_frame=lambda rec, frame: rec.write_frame(frame)
        if hasattr(rec, "write_frame")
        else None,
        _post_dual_preview_layout=lambda _layout: None,
        _post_status=lambda _status: None,
        _post_progress=lambda _current, _total: None,
        _post_done=lambda **_kwargs: done_calls.append(True),
        _post_dual_frame_pair=lambda *_args, **_kwargs: True,
        _mark_dual_startup_metric=lambda *_args, **_kwargs: None,
        _advance_dual_preview_stage=lambda *_args, **_kwargs: True,
        _set_dual_startup_outcome=lambda *_args, **_kwargs: None,
        _post_dual_recording_ready=lambda _generation: None,
        _invalidate_dual_preview=lambda _generation: None,
        _finish_dual_startup_metrics=lambda *_args, **_kwargs: None,
        _write_recording_pair=write_pair,
        _close_recording_pair=lambda: close_pair_calls.append(True),
    )
    state = SimpleNamespace(
        source="0",
        source2="1",
        session_generation=1,
        pose_variant="full",
        enable_hands=True,
        rotate=0,
        rotate2=0,
        record_skeleton=record_skeleton,
        capture_size=(1920, 1080) if student else (1280, 720),
    )
    monkeypatch.setattr(app_ui, "models_dir", lambda: Path("models"))
    monkeypatch.setattr(app_ui, "MediaPipePipeline", pipeline_factory)
    monkeypatch.setattr(app_ui.cv2, "putText", lambda image, *_args, **_kwargs: image)
    monkeypatch.setattr(app_ui.cv2, "destroyAllWindows", lambda: None)

    app_ui.App._worker_loop_dual_camera(app, state)

    assert front_cap.release_calls == 1
    assert side_cap.release_calls == 1
    assert close_pair_calls == [True]
    assert done_calls == [True]
    if student:
        assert front_cap.settings == side_cap.settings == []
    return pipelines


def test_student_dual_camera_never_changes_format_after_claim(monkeypatch):
    _exercise_dual_runtime_failure(monkeypatch, record_skeleton=False, phase="write", student=True)


def test_dual_skeleton_annotate_failure_closes_two_pipelines(monkeypatch):
    pipelines = _exercise_dual_runtime_failure(
        monkeypatch, record_skeleton=True, phase="annotate"
    )

    assert len(pipelines) == 2
    assert [pipeline.close_calls for pipeline in pipelines] == [1, 1]


def test_dual_raw_write_failure_creates_no_pipeline(monkeypatch):
    pipelines = _exercise_dual_runtime_failure(
        monkeypatch, record_skeleton=False, phase="write"
    )

    assert pipelines == []


def test_pipeline_loading_keeps_raw_pair_flowing_without_writes(monkeypatch):
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    pipeline_entered = threading.Event()
    allow_pipeline = threading.Event()
    accepted_stages: list[str] = []
    write_calls: list[bool] = []
    ready_calls: list[int] = []
    done_calls: list[int] = []

    class _Capture:
        def __init__(self, *, read_limit: int = 1) -> None:
            self.read_calls = 0
            self.release_calls = 0
            self.read_limit = read_limit

        def get(self, prop) -> int:
            return 6 if prop == app_ui.cv2.CAP_PROP_FRAME_WIDTH else 4

        def read(self):
            self.read_calls += 1
            if self.read_calls <= self.read_limit:
                return True, frame.copy()
            return False, None

        def release(self) -> None:
            self.release_calls += 1

    front_cap = _Capture()
    side_cap = _Capture()

    class _Pool:
        def __init__(self) -> None:
            self.sequence = 1

        def wait_pair(self, *_args, **_kwargs):
            return (
                SimpleNamespace(frame=frame.copy(), sequence=1, generation=1),
                SimpleNamespace(frame=frame.copy(), sequence=1, generation=1),
            )

        def snapshot_pair(self, *_args, **_kwargs):
            self.sequence += 1
            value = self.sequence
            return (
                SimpleNamespace(frame=frame.copy(), sequence=value, generation=1),
                SimpleNamespace(frame=frame.copy(), sequence=value, generation=1),
            )

        def claim_pair(self, *_args, **_kwargs):
            return front_cap, side_cap

        def cancel(self, _role):
            pass

    class _RecordingSession:
        def __init__(self) -> None:
            self._size = (6, 4)

        def begin_session(self, **kwargs) -> None:
            if "size" in kwargs:
                self._size = kwargs["size"]

        def update_session_size(self, **kwargs) -> None:
            if "size" in kwargs:
                self._size = kwargs["size"]
            return True

        @property
        def session_size(self):
            return self._size

        def write_frame(self, _frame) -> None:
            pass

    class _Pipeline:
        def __init__(self) -> None:
            self.close_calls = 0

        def next_timestamp_ms(self, **_kwargs) -> int:
            return 1

        def annotate(self, current_frame, **_kwargs):
            return current_frame.copy(), []

        def close(self) -> None:
            self.close_calls += 1

    pipelines: list[_Pipeline] = []

    def pipeline_factory(**_kwargs):
        if not pipelines:
            pipeline_entered.set()
            assert allow_pipeline.wait(2.0)
        pipeline = _Pipeline()
        pipelines.append(pipeline)
        return pipeline

    app = _dual_preview_stub(record_skeleton=True)
    app._camera_warmup_pool = _Pool()
    app._current_session_generation = 1
    app._stop_evt = threading.Event()
    app._record_pair_lock = threading.Lock()
    app._rec = _RecordingSession()
    app._rec2 = _RecordingSession()
    app._dual_active = False
    app._post_dual_preview_layout = lambda _layout: None
    app._post_status = lambda _status: None
    app._post_progress = lambda _done, _total: None
    app._set_dual_startup_outcome = lambda *_args, **_kwargs: None
    app._emit_dual_startup_metrics = lambda *_args, **_kwargs: None
    app._mark_dual_startup_metric = lambda *_args, **_kwargs: None
    # 必须真正推进 stage：annotated 帧只在 current stage 匹配时被 _post_dual_frame_pair 接受。
    app._advance_dual_preview_stage = (
        lambda generation, stage: app_ui.App._advance_dual_preview_stage(
            app, generation, stage
        )
    )
    app._post_dual_recording_ready = ready_calls.append
    app._write_recording_pair = lambda *_frames: write_calls.append(True)
    app._close_recording_pair = lambda: None
    app._post_done = lambda **kwargs: done_calls.append(
        kwargs.get("session_generation")
    )
    app._finish_dual_startup_metrics = lambda *_args, **_kwargs: None
    app._prepare_preview_rgb = lambda frame, _size: frame
    original_post = app_ui.App._post_dual_frame_pair

    def post_pair(*args, **kwargs):
        accepted = original_post(app, *args, **kwargs)
        if accepted:
            accepted_stages.append(kwargs["stage"])
        return accepted

    app._post_dual_frame_pair = post_pair
    # 放行 pipeline 后主循环需持续可读，否则 claim 后立即 EOF 来不及出 annotated 帧。
    front_cap.read_limit = 8
    side_cap.read_limit = 8
    state = SimpleNamespace(
        source="0",
        source2="1",
        session_generation=1,
        pose_variant="full",
        enable_hands=True,
        rotate=0,
        rotate2=0,
        record_skeleton=True,
    )
    monkeypatch.setattr(app_ui, "models_dir", lambda: Path("models"))
    monkeypatch.setattr(app_ui, "MediaPipePipeline", pipeline_factory)
    monkeypatch.setattr(app_ui.cv2, "putText", lambda image, *_args, **_kwargs: image)
    monkeypatch.setattr(app_ui.cv2, "destroyAllWindows", lambda: None)

    worker = threading.Thread(
        target=lambda: app_ui.App._worker_loop_dual_camera(app, state)
    )
    worker.start()
    assert pipeline_entered.wait(1.0)
    deadline = app_ui.time.monotonic() + 1.0
    while accepted_stages.count("raw") < 2 and app_ui.time.monotonic() < deadline:
        app_ui.time.sleep(0.01)
    assert accepted_stages.count("raw") >= 2
    assert write_calls == []
    assert ready_calls == []

    allow_pipeline.set()
    worker.join(2.0)

    assert not worker.is_alive()
    assert len(pipelines) == 2
    assert [pipeline.close_calls for pipeline in pipelines] == [1, 1]
    assert ready_calls == [1]
    assert write_calls
    assert "annotated" in accepted_stages
    first_annotated = accepted_stages.index("annotated")
    assert "raw" not in accepted_stages[first_annotated + 1 :]
    assert done_calls == [1]
    assert front_cap.release_calls == 1
    assert side_cap.release_calls == 1


def test_second_pipeline_failure_closes_first_without_starting_recording(monkeypatch):
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    cancel_calls: list[str] = []
    begin_calls: list[bool] = []
    done_calls: list[int] = []

    class _Pool:
        def wait_pair(self, *_args, **_kwargs):
            return (
                SimpleNamespace(frame=frame.copy(), sequence=1, generation=1),
                SimpleNamespace(frame=frame.copy(), sequence=1, generation=1),
            )

        def snapshot_pair(self, *_args, **_kwargs):
            return None

        def claim_pair(self, *_args, **_kwargs):
            pytest.fail("claim must not run after pipeline initialization fails")

        def cancel(self, role, *, generation=None):
            cancel_calls.append((role, generation))

    class _RecordingSession:
        def begin_session(self, **_kwargs) -> None:
            begin_calls.append(True)

    class _Pipeline:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    first = _Pipeline()
    calls = 0

    def pipeline_factory(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return first
        raise RuntimeError("second pipeline failed")

    app = _dual_preview_stub(record_skeleton=True)
    app._camera_warmup_pool = _Pool()
    app._current_session_generation = 1
    app._stop_evt = threading.Event()
    app._record_pair_lock = threading.Lock()
    app._rec = _RecordingSession()
    app._rec2 = _RecordingSession()
    app._post_dual_preview_layout = lambda _layout: None
    app._post_status = lambda _status: None
    app._post_progress = lambda *_args: None
    app._post_done = lambda **kwargs: done_calls.append(
        kwargs.get("session_generation")
    )
    app._close_recording_pair = lambda: pytest.fail(
        "recording pair must not start"
    )
    app._finish_dual_startup_metrics = lambda *_args, **_kwargs: None
    state = SimpleNamespace(
        source="0",
        source2="1",
        session_generation=1,
        pose_variant="full",
        enable_hands=True,
        rotate=0,
        rotate2=0,
        record_skeleton=True,
    )
    monkeypatch.setattr(app_ui, "models_dir", lambda: Path("models"))
    monkeypatch.setattr(app_ui, "MediaPipePipeline", pipeline_factory)
    monkeypatch.setattr(app_ui.cv2, "destroyAllWindows", lambda: None)

    app_ui.App._worker_loop_dual_camera(app, state)

    assert first.close_calls == 1
    assert begin_calls == []
    # 失败会话只取消自己 wait_pair 观察到的代次，不误伤之后新声明的预热。
    assert cancel_calls == [(app_ui.PRIMARY, 1), (app_ui.SECONDARY, 1)]
    assert done_calls == [1]


def test_main_stop_does_not_cancel_background_compare_queue():
    stop = _app_method_node("_stop")

    assert _named_calls(stop, "cancel_all") == []


@pytest.mark.parametrize(
    ("worker_name", "expected"),
    [
        (
            "_worker_loop",
            [("self._rec", "(int(frame.shape[1]), int(frame.shape[0]))")],
        ),
        (
            "_worker_loop_dual_camera",
            [("self._rec", "actual_size"), ("self._rec2", "actual_size2")],
        ),
        (
            "_worker_loop_parallel_camera",
            [("self._rec", "(int(frame.shape[1]), int(frame.shape[0]))")],
        ),
    ],
)
def test_camera_paths_correct_writer_size_from_rotated_first_frame(worker_name, expected):
    calls = _named_calls(_function_node(worker_name), "update_session_size")
    actual = [
        (
            ast.unparse(call.func.value),
            ast.unparse(next(kw.value for kw in call.keywords if kw.arg == "size")),
        )
        for call in calls
    ]

    assert sorted(actual) == sorted(expected)

    if worker_name == "_worker_loop_dual_camera":
        source = ast.unparse(_function_node(worker_name))
        assert "actual_size = (int(frame.shape[1]), int(frame.shape[0]))" in source
        assert "actual_size2 = (int(frame2.shape[1]), int(frame2.shape[0]))" in source


@pytest.mark.parametrize(
    ("worker_name", "expected_args"),
    [
        (
            "_worker_loop",
            [("w", "h", "self._get_runtime_rotate('primary')")],
        ),
        (
            "_worker_loop_dual_camera",
            [
                ("w", "h", "self._get_runtime_rotate('primary')"),
                ("w2", "h2", "self._get_runtime_rotate('secondary')"),
            ],
        ),
        (
            "_worker_loop_parallel_camera",
            [("w", "h", "self._get_runtime_rotate('primary')")],
        ),
    ],
)
def test_camera_writer_sizes_follow_rotation_before_session_start(worker_name, expected_args):
    worker = _function_node(worker_name)
    size_calls = _named_calls(worker, "_rotated_size")
    actual_args = sorted(tuple(ast.unparse(arg) for arg in call.args) for call in size_calls)
    begin_calls = _named_calls(worker, "begin_session")

    assert actual_args == sorted(expected_args)
    assert begin_calls
    nested_size_calls = {
        id(call)
        for begin_call in begin_calls
        for call in _named_calls(begin_call, "_rotated_size")
    }
    standalone_size_calls = [
        call for call in size_calls if id(call) not in nested_size_calls
    ]
    assert not standalone_size_calls or max(
        call.lineno for call in standalone_size_calls
    ) < min(call.lineno for call in begin_calls)
