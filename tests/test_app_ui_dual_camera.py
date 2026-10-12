# -*- coding: utf-8 -*-
"""Tkinter 双摄骨架测试（issue #59 裁剪版）。

不创建真实 Tk root，只验证 source2 收敛点、旋转控制、双摄预览布局与会话事件在界面侧的处理。
逐帧采集、推理、录制与失败收尾的行为测试在 test_live_session.py（P6 起取代旧 worker 白盒测试）。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apps import app_ui  # noqa: E402
from core.live import LiveControls, PreviewStore, StartupMetrics  # noqa: E402
from core.live.stages import apply_rotation, rotated_size  # noqa: E402
from core.media import CameraDevice as CameraEntry  # noqa: E402
from core.media import CameraService  # noqa: E402
from core.recording import fit_frame_to_size  # noqa: E402


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


class _FakeLive:
    """会话管理器替身：当前会话的状态、配置与预览槽。"""

    def __init__(self, preview: PreviewStore, *, state: str = "running") -> None:
        self.state = state
        self.preview = preview
        self.token = preview.token
        self.config = SimpleNamespace(dual=True, is_file=False)

    def active(self) -> bool:
        return self.state != "idle"

    def is_current(self, token: int) -> bool:
        return token == self.token


def _dual_preview_stub(*, record_skeleton: bool = False):
    app = object.__new__(app_ui.App)
    metrics = StartupMetrics(
        1, primary_index=0, secondary_index=1, inference=record_skeleton, start_click=1.0,
    )
    app._live = _FakeLive(PreviewStore(1, lambda _role: (0, 0), metrics=metrics))
    app._live_events = Queue()
    app._live_controls = LiveControls()
    app._closing = False
    app._shown_preview_seq = 0
    app._dual_recording_ready = False
    app._dual_active = False
    app._dual_preview_layout = "stacked"
    app._set_dual_preview_layout = lambda layout: setattr(app, "_dual_preview_layout", layout)
    app.root = _TickRoot()
    app.preview = _ImageWidget()
    app.preview2 = _ImageWidget()
    app.actions_var = _Var("-")
    app.status_var = _Var("就绪")
    app._photo = None
    app._photo2 = None
    app._dual_auto_compare = True
    app._current_auto_compare = lambda prefer_ui=False: True
    app._drain_recording_compare_updates = lambda: None
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
    fitted = fit_frame_to_size(frame, (3, 2))
    assert fitted.shape[:2] == (2, 3)


def test_fit_frame_to_size_returns_same_when_match():
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    assert fit_frame_to_size(frame, (6, 4)) is frame


def test_runtime_rotate_primary_and_secondary_are_isolated():
    app = object.__new__(app_ui.App)
    app._live_controls = LiveControls()
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
    assert app._live_controls.rotation_of(app_ui.SECONDARY) == 180


@pytest.mark.parametrize(
    ("degrees", "turns"),
    [(90, -1), (180, 2), (270, 1)],
)
def test_apply_rotation_matches_clockwise_degree_contract(degrees, turns):
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

    rotated = apply_rotation(frame, degrees)

    np.testing.assert_array_equal(rotated, np.rot90(frame, k=turns))


@pytest.mark.parametrize("degrees", [0, 45])
def test_apply_rotation_returns_original_frame_when_no_rotation_applies(degrees):
    frame = np.zeros((2, 3, 3), dtype=np.uint8)

    assert apply_rotation(frame, degrees) is frame


@pytest.mark.parametrize(
    ("degrees", "expected"),
    [(0, (1280, 720)), (90, (720, 1280)), (180, (1280, 720)), (270, (720, 1280))],
)
def test_rotated_size_matches_rotated_frame_dimensions(degrees, expected):
    assert rotated_size(1280, 720, degrees) == expected


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


def test_app_camera_service_is_the_only_camera_owner():
    init = _app_method_node("__init__")
    calls = _named_calls(init, "CameraService")

    assert len(calls) == 1
    capture_factory = next(
        keyword.value for keyword in calls[0].keywords if keyword.arg == "capture_factory"
    )
    assert ast.unparse(capture_factory) == "self._open_camera_process"
    # 所有摄像头打开都经 CameraService 的池；会话输入与选择回调不直接创建采集进程。
    for name in ("_make_session_input", "_sync_camera_warmup"):
        assert _named_calls(_app_method_node(name), "ProcessCamera") == []
        assert _named_calls(_app_method_node(name), "VideoCapture") == []


@pytest.mark.parametrize(
    ("student", "size", "policy"),
    [(False, (1280, 720), "prefer"), (True, (1920, 1080), "require")],
)
def test_dual_session_config_freezes_capture_specs(student, size, policy):
    """开始时冻结的请求尺寸与策略原样进入两路 CaptureSpec（学生练习 1080p、require，D-6）。"""
    state = app_ui.UiState(
        source="0", source2="1", pose_variant="full", workers=4, enable_hands=True,
        capture_size=size, capture_policy=policy, camera_names=((1, "Side Cam"),),
    )
    config = app_ui.App._session_config(state)

    assert config.dual and config.roles == (app_ui.PRIMARY, app_ui.SECONDARY)
    assert config.cameras[app_ui.PRIMARY] == app_ui.CaptureSpec(0, *size, policy=policy)
    assert config.cameras[app_ui.SECONDARY] == app_ui.CaptureSpec(
        1, *size, expected_name="Side Cam", policy=policy
    )
    assert not config.parallel                     # 双摄不走并行推理


def test_session_input_opens_camera_stream_without_touching_pool():
    """开始只登记设备句柄：等待与接管在会话线程的 CameraStream 上进行，Tk 线程不调用池。"""

    class _Pool:
        def __getattr__(self, name):
            pytest.fail(f"Tk thread must not call pool.{name}")

    app = object.__new__(app_ui.App)
    app._camera_service = CameraService(pool=_Pool())
    config = app_ui.App._session_config(
        app_ui.UiState(source="0", source2="1", pose_variant="full", workers=1, enable_hands=True)
    )

    source = app_ui.App._make_session_input(app, config)

    assert isinstance(source, app_ui.CameraInput)


def test_tick_renders_dual_frame_atomically_with_status_once_per_stage(monkeypatch):
    app = _dual_preview_stub(record_skeleton=True)
    store = app._live.preview
    frame = np.full((2, 3, 3), 7, dtype=np.uint8)
    monkeypatch.setattr(app_ui.ImageTk, "PhotoImage", lambda image: image)

    store.publish("raw", {app_ui.PRIMARY: frame, app_ui.SECONDARY: frame}, {app_ui.PRIMARY: "front"})
    app_ui.App._tick(app)

    assert len(app.preview.images) == 1
    assert len(app.preview2.images) == 1
    assert app.actions_var.get() == "front"
    assert app.status_var.get().startswith("双摄画面已显示")
    assert app.status_var.get().endswith("，模型加载中…")
    assert store.wait_rendered(0)
    assert len(app.root.after_calls) == 1

    app.status_var.set("运行中…")
    store.publish("raw", {app_ui.PRIMARY: frame, app_ui.SECONDARY: frame}, {app_ui.PRIMARY: "again"})
    app_ui.App._tick(app)
    assert app.status_var.get() == "运行中…"      # 同一阶段只提示一次
    assert len(app.preview.images) == 2

    store.advance("annotated")
    store.publish("annotated", {app_ui.PRIMARY: frame, app_ui.SECONDARY: frame}, {})
    app_ui.App._tick(app)
    assert app.status_var.get().startswith("运行中…（双摄首个标注帧")
    assert app.actions_var.get() == "-"


def test_tick_shows_nothing_after_stop_requested(monkeypatch):
    app = _dual_preview_stub()
    store = app._live.preview
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    monkeypatch.setattr(app_ui.ImageTk, "PhotoImage", lambda image: image)
    store.publish("raw", {app_ui.PRIMARY: frame, app_ui.SECONDARY: frame}, {})
    app._live.state = "stopping"

    app_ui.App._tick(app)

    assert app.preview.images == []
    assert app.preview2.images == []


def test_session_running_enables_dual_recording_only_for_live_dual_session():
    app = _dual_preview_stub()
    app.record_btn = SimpleNamespace(calls=[], configure=lambda **kw: app.record_btn.calls.append(kw))
    app._student_practice_active = False

    app._live.state = "stopping"
    app_ui.App._on_session_running(app, None)
    assert app._dual_recording_ready is False and app.record_btn.calls == []

    app._live.state = "running"
    app_ui.App._on_session_running(app, None)
    assert app._dual_recording_ready is True and app._dual_active is True
    assert app.record_btn.calls == [{"state": "normal", "text": app_ui.RECORD_BTN_TEXT["idle"]}]


def test_main_stop_does_not_cancel_background_compare_queue():
    stop = _app_method_node("_stop")

    assert _named_calls(stop, "cancel_all") == []
