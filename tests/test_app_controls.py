# -*- coding: utf-8 -*-
"""按钮文本映射与运行态控件联动单元测试（任务 7.4，headless 友好）。

设计来源：``.kiro/specs/ui-layout-redesign/design.md``
（Components 3「运行态控件联动」、Record_Toggle 三态按钮文本映射）。

本文件以**不依赖真实 Tk 窗口**的方式验证：
  1. ``RECORD_BTN_TEXT`` 三态文本映射（纯数据断言）。
  2. ``App._on_record_toggle`` 通过真实 ``RecordingController`` 驱动按钮文本循环演进，
     使用轻量 stub self（真实控制器 + 假 record_btn），无需 Tk。
  3. ``App._set_running_controls`` 运行态 enable/disable 联动，使用 stub self 与捕获
     ``configure`` 调用的假控件，无需 Tk。

设计意图：以 stub self 绑定调用未绑定的 ``App`` 方法（``App._method(stub, ...)``），
从而在无显示（headless / CI）环境中也能稳定运行，不会 skip。

_Requirements: 4.3, 5.2, 5.4, 5.6, 6.4, 6.5_
"""
from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.app_ui import App, RECORD_BTN_TEXT  # noqa: E402
from core.recording_controller import RecordingController  # noqa: E402


# ---------------------------------------------------------------------------
# 测试替身（fakes）
# ---------------------------------------------------------------------------


class FakeWidget:
    """记录 ``configure`` 调用的假控件。

    每次 ``configure(**kwargs)`` 都会合并进 ``self.config`` 并追加到 ``self.calls``，
    便于断言「最后一次文本」「最终状态」等。
    """

    def __init__(self, **initial) -> None:
        self.config = dict(initial)
        self.calls: list[dict] = []

    def configure(self, **kwargs) -> None:
        self.config.update(kwargs)
        self.calls.append(dict(kwargs))

    # 兼容 ttk/tk 的别名
    config_alias = configure

    def cget(self, key):
        return self.config.get(key)


class FakeVar:
    """假 Tk 变量，提供 set/get。"""

    def __init__(self, value: str = "") -> None:
        self._value = value

    def set(self, value) -> None:
        self._value = value

    def get(self):
        return self._value


class FakeWriter:
    """最小假 ``VideoWriterLike``：记录写入帧与释放调用。"""

    def __init__(self) -> None:
        self.frames: list = []
        self.released = False

    def write(self, frame) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


def _make_fake_writer_factory():
    """返回 (factory, created) —— factory 创建 FakeWriter 并记录到 created 列表。"""
    created: list[FakeWriter] = []

    def _factory(path, fps, size):
        w = FakeWriter()
        created.append(w)
        return w, Path(path), "fake"

    return _factory, created


# ---------------------------------------------------------------------------
# 1. RECORD_BTN_TEXT 三态文本映射（纯数据断言） —— 需求 5.2 / 5.4 / 5.6
# ---------------------------------------------------------------------------


def test_record_btn_text_mapping():
    """RECORD_BTN_TEXT 精确映射三态到中文按钮文本。

    idle→开始录制（需求 5.2）、recording→暂停录制（需求 5.4）、
    paused→继续录制（需求 5.6）。
    """
    assert RECORD_BTN_TEXT["idle"] == "开始录制"
    assert RECORD_BTN_TEXT["recording"] == "暂停录制"
    assert RECORD_BTN_TEXT["paused"] == "继续录制"


def test_record_btn_text_has_exactly_three_states():
    """映射恰好覆盖三种状态，不多不少。"""
    assert set(RECORD_BTN_TEXT.keys()) == {"idle", "recording", "paused"}


# ---------------------------------------------------------------------------
# 2. App._on_record_toggle 驱动按钮文本循环演进（真实控制器 + stub self）
#    —— 需求 5.2 / 5.4 / 5.6
# ---------------------------------------------------------------------------


def _make_toggle_stub():
    """构造承载 _on_record_toggle 所需最小属性的 stub self。

    - ``_rec``：真实 RecordingController（注入假 writer 工厂，begin_session 后可 toggle）。
    - ``record_btn``：捕获 configure(text=...) 的假控件。
    """
    factory, _created = _make_fake_writer_factory()
    factory2, _created2 = _make_fake_writer_factory()
    rec = RecordingController(
        writer_factory=factory,
        path_provider=lambda: Path("unused.mp4"),
    )
    rec2 = RecordingController(
        writer_factory=factory2,
        path_provider=lambda: Path("unused-side.mp4"),
    )
    stub = types.SimpleNamespace(
        _rec=rec,
        _rec2=rec2,
        _record_pair_lock=threading.Lock(),
        _record_stamp=None,
        _exam_manual_locked=False,
        _dual_recording_ready=True,
        _closing=False,
        record_btn=FakeWidget(text=RECORD_BTN_TEXT["idle"]),
        _sync_record_stop_enabled=lambda _state: None,
        _current_auto_compare=lambda prefer_ui=False: True,
    )
    return stub


def test_on_record_toggle_drives_button_text_cycle():
    """_on_record_toggle 随控制器状态推进按钮文本：
    开始录制→暂停录制→继续录制→暂停录制（需求 5.2、5.4、5.6）。"""
    stub = _make_toggle_stub()
    stub._rec.begin_session(fps=30.0, size=(640, 480))

    # 初始（会话已启动、懒创建）：idle，文本「开始录制」。
    assert stub._rec.state == "idle"
    assert stub.record_btn.cget("text") == RECORD_BTN_TEXT["idle"]

    # 第一次 toggle：idle→recording → 文本「暂停录制」。
    App._on_record_toggle(stub)
    assert stub._rec.state == "recording"
    assert stub.record_btn.cget("text") == RECORD_BTN_TEXT["recording"] == "暂停录制"

    # 第二次 toggle：recording→paused → 文本「继续录制」。
    App._on_record_toggle(stub)
    assert stub._rec.state == "paused"
    assert stub.record_btn.cget("text") == RECORD_BTN_TEXT["paused"] == "继续录制"

    # 第三次 toggle：paused→recording → 文本「暂停录制」。
    App._on_record_toggle(stub)
    assert stub._rec.state == "recording"
    assert stub.record_btn.cget("text") == RECORD_BTN_TEXT["recording"] == "暂停录制"


def test_on_record_toggle_matches_controller_state_each_step():
    """每次 toggle 后，按钮文本恒等于控制器当前状态对应的 RECORD_BTN_TEXT。"""
    stub = _make_toggle_stub()
    stub._rec.begin_session(fps=30.0, size=(640, 480))

    for _ in range(6):  # 多轮循环，覆盖 recording⇄paused 反复切换
        App._on_record_toggle(stub)
        assert stub.record_btn.cget("text") == RECORD_BTN_TEXT[stub._rec.state]


def test_on_record_toggle_keeps_both_camera_controllers_in_sync():
    stub = _make_toggle_stub()
    stub._rec.begin_session(fps=30.0, size=(640, 480))
    stub._rec2.begin_session(fps=30.0, size=(480, 640))

    for expected in ("recording", "paused", "recording"):
        App._on_record_toggle(stub)
        assert stub._rec.state == expected
        assert stub._rec2.state == expected


def test_on_record_toggle_noop_before_session():
    """会话未运行时 toggle 为 no-op：状态保持 idle、文本保持「开始录制」（需求 5.1）。"""
    stub = _make_toggle_stub()
    # 未调用 begin_session
    App._on_record_toggle(stub)
    assert stub._rec.state == "idle"
    assert stub.record_btn.cget("text") == RECORD_BTN_TEXT["idle"]


def test_on_record_toggle_noop_after_stop_requested():
    stub = _make_toggle_stub()
    stub._rec.begin_session(fps=30.0, size=(640, 480))
    stub._rec2.begin_session(fps=30.0, size=(480, 640))
    stub._stop_evt = threading.Event()
    stub._stop_evt.set()

    App._on_record_toggle(stub)

    assert stub._record_stamp is None
    assert stub._rec.state == "idle"
    assert stub._rec2.state == "idle"
    assert stub.record_btn.cget("text") == RECORD_BTN_TEXT["idle"]


# ---------------------------------------------------------------------------
# 3. App._set_running_controls 运行态联动（stub self + 假控件，无 Tk）
#    —— 需求 4.3 / 5.2 / 6.4 / 6.5（兼带 2.5 / 3.5 / 3.6 / 4.5）
# ---------------------------------------------------------------------------


class _FakeWorker:
    """假 worker 线程：``is_alive`` 返回固定值，供 _set_refresh_enabled 判定运行态。"""

    def __init__(self, alive: bool) -> None:
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


def _make_controls_stub(worker_alive: bool = False):
    """构造 _set_running_controls 所需的 stub self（全部假控件 + 必要属性）。

    提供 refresh_btn / _worker / _enum_busy，使真实 ``App._set_refresh_enabled``
    在 stub 上也能正常工作（运行中禁用、未运行启用）。

    ``worker_alive`` 模拟工作线程是否在跑：真实 App 在启动 worker 后才调用
    ``_set_running_controls(True)``，刷新控件的启停由 worker/_enum_busy 决定。
    """
    stub = types.SimpleNamespace(
        camera_combo=FakeWidget(state="readonly"),
        camera_combo_2=FakeWidget(state="readonly"),
        model_combo=FakeWidget(state="readonly"),
        record_skeleton_check=FakeWidget(state="normal"),
        auto_compare_check=FakeWidget(state="normal"),
        record_btn=FakeWidget(state="disabled", text=RECORD_BTN_TEXT["idle"]),
        compare_btn=FakeWidget(state="normal"),
        start_btn=FakeWidget(state="normal", text="开始"),
        stop_btn=FakeWidget(state="disabled"),
        refresh_btn=FakeWidget(state="normal"),
        status_var=FakeVar("就绪"),
        _camera_entries=["camera 0"],  # 非空，使未运行时 camera_combo 恢复 readonly
        _worker=_FakeWorker(worker_alive),
        _enum_busy=threading.Event(),
    )
    # 绑定真实 _set_refresh_enabled，使 _set_running_controls 内部调用可用。
    stub._set_refresh_enabled = types.MethodType(App._set_refresh_enabled, stub)
    return stub


def test_set_running_controls_when_running():
    """running=True：Camera/Model 禁用、Record_Toggle 启用并置「开始录制」、
    Compare 始终 enabled、Start 切「停止」并禁用（需求 2.5、3.5、4.3、5.2、6.4）。"""
    stub = _make_controls_stub(worker_alive=True)

    App._set_running_controls(stub, True)

    # Camera_Selector 禁用（需求 2.5）。
    assert stub.camera_combo.cget("state") == "disabled"
    assert stub.camera_combo_2.cget("state") == "disabled"
    # Model_Selector 禁用（需求 3.5）。
    assert stub.model_combo.cget("state") == "disabled"
    assert stub.record_skeleton_check.cget("state") == "disabled"
    # 自动比对滑块运行中可快速切换（非考试锁定时）。
    assert stub.auto_compare_check.cget("state") == "normal"
    # Record_Toggle 启用并置「开始录制」（需求 5.2）。
    assert stub.record_btn.cget("state") == "normal"
    assert stub.record_btn.cget("text") == RECORD_BTN_TEXT["idle"] == "开始录制"
    # Compare_Control 始终 enabled（需求 6.4）。
    assert stub.compare_btn.cget("state") == "normal"
    # Start_Control 切「停止」并禁用（需求 4.3）。
    assert stub.start_btn.cget("text") == "停止"
    assert stub.start_btn.cget("state") == "disabled"
    assert stub.stop_btn.cget("state") == "normal"
    # 刷新控件：运行中禁用。
    assert stub.refresh_btn.cget("state") == "disabled"


def test_dual_loading_keeps_record_control_disabled():
    stub = _make_controls_stub(worker_alive=True)
    stub._dual_recording_ready = False

    App._set_running_controls(stub, True)

    assert stub.record_btn.cget("state") == "disabled"


def test_record_toggle_rejects_dual_loading_even_when_called_directly():
    stub = _make_toggle_stub()
    stub._rec.begin_session(fps=30.0, size=(640, 480))
    stub._rec2.begin_session(fps=30.0, size=(480, 640))
    stub._stop_evt = threading.Event()
    stub._active_dual_generation = 7
    stub._dual_recording_ready = False

    App._on_record_toggle(stub)

    assert stub._rec.state == "idle"
    assert stub._rec2.state == "idle"
    assert stub.record_btn.cget("text") == RECORD_BTN_TEXT["idle"]


def test_set_running_controls_when_not_running():
    """running=False：Record_Toggle 禁用、Compare 仍 enabled、Start 恢复「开始」并启用、
    Status_Area 显示「就绪」（需求 4.5、5.1、6.5）。"""
    stub = _make_controls_stub()
    # 先进入运行态，再恢复，验证状态确实被复位。
    App._set_running_controls(stub, True)
    App._set_running_controls(stub, False)

    # Record_Toggle 未运行禁用（需求 5.1）。
    assert stub.record_btn.cget("state") == "disabled"
    # Compare_Control 未运行仍 enabled（需求 6.5）。
    assert stub.compare_btn.cget("state") == "normal"
    # Start_Control 恢复「开始」并启用。
    assert stub.start_btn.cget("text") == "开始"
    assert stub.start_btn.cget("state") == "normal"
    assert stub.stop_btn.cget("state") == "disabled"
    # Model_Selector 恢复 readonly（需求 3.6）。
    assert stub.model_combo.cget("state") == "readonly"
    assert stub.record_skeleton_check.cget("state") == "normal"
    assert stub.auto_compare_check.cget("state") == "normal"
    # Camera_Selector 恢复 readonly（有可选列表时）。
    assert stub.camera_combo.cget("state") == "readonly"
    assert stub.camera_combo_2.cget("state") == "readonly"
    # Status_Area 显示「就绪」（需求 4.5）。
    assert stub.status_var.get() == "就绪"
    # 刷新控件：未运行启用。
    assert stub.refresh_btn.cget("state") == "normal"


def test_compare_control_enabled_in_both_states():
    """Compare_Control 在运行与未运行两态均保持 enabled（需求 6.4、6.5）。"""
    stub = _make_controls_stub()

    App._set_running_controls(stub, True)
    assert stub.compare_btn.cget("state") == "normal"

    App._set_running_controls(stub, False)
    assert stub.compare_btn.cget("state") == "normal"
