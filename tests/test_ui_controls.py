# -*- coding: utf-8 -*-
"""按钮文本映射与运行态控件联动单元测试（任务 7.4）。

设计来源：``.kiro/specs/ui-layout-redesign/design.md``
（Components 3「运行态控件联动」、Record_Toggle 三态按钮文本映射）。

本文件覆盖：
  1. ``RECORD_BTN_TEXT`` 三态文本映射（纯数据断言，安全 import）。
  2. 三态按钮文本随 ``RecordingController.request_toggle()`` 循环演进的序列
     （驱动真实 ``RecordingController``，无需 Tk 窗口）。
  3. ``App._set_running_controls`` 运行态 enable/disable 联动（需真实 Tk；
     无显示环境时 ``pytest.skip``，并对尚未创建的控件做存在性容错）。录制按钮
     另受录制就绪标志把关：单摄会话立即可录，双摄会话等两路首帧就绪。

_Requirements: 4.3, 5.2, 5.4, 5.6, 6.4, 6.5_
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.app_ui import RECORD_BTN_TEXT  # noqa: E402
from core.recording_controller import RecordingController  # noqa: E402


# ---------------------------------------------------------------------------
# 1. RECORD_BTN_TEXT 三态文本映射（纯数据断言）
# ---------------------------------------------------------------------------

def test_record_btn_text_mapping():
    """RECORD_BTN_TEXT 精确映射三态到中文按钮文本（需求 5.2、5.4、5.6）。"""
    assert RECORD_BTN_TEXT["idle"] == "开始录制"
    assert RECORD_BTN_TEXT["recording"] == "暂停录制"
    assert RECORD_BTN_TEXT["paused"] == "继续录制"


def test_record_btn_text_has_exactly_three_states():
    """映射恰好覆盖三种状态，不多不少。"""
    assert set(RECORD_BTN_TEXT.keys()) == {"idle", "recording", "paused"}


# ---------------------------------------------------------------------------
# 2. 三态按钮文本随 toggle 循环演进（驱动真实 RecordingController，无 Tk）
# ---------------------------------------------------------------------------

def _toggle_only_factory():
    """构造一个仅用于状态切换、不会真正写盘的 RecordingController。

    write_frame 不会在本测试中被调用，故 writer_factory 仅需占位。
    """
    def _never_called_factory(path, fps, size):  # pragma: no cover - 不应被调用
        raise AssertionError("writer_factory 不应在 toggle 测试中被调用")

    return RecordingController(
        writer_factory=_never_called_factory,
        path_provider=lambda: Path("unused.mp4"),
    )


def test_button_text_follows_toggle_cycle():
    """从 idle 开始，toggle 循环产生 recording→paused→recording，
    对应按钮文本 开始录制→暂停录制→继续录制→暂停录制（需求 5.2、5.4、5.6）。"""
    rec = _toggle_only_factory()

    # 会话启动前：idle，按钮文本应为「开始录制」。
    assert rec.state == "idle"
    assert RECORD_BTN_TEXT[rec.state] == "开始录制"

    rec.begin_session(fps=30.0, size=(640, 480))
    # 会话刚启动仍为 idle（懒创建语义），文本「开始录制」。
    assert rec.state == "idle"
    assert RECORD_BTN_TEXT[rec.state] == "开始录制"

    # 第一次 toggle: idle→recording → 文本「暂停录制」。
    s1 = rec.request_toggle()
    assert s1 == "recording"
    assert RECORD_BTN_TEXT[s1] == "暂停录制"

    # 第二次 toggle: recording→paused → 文本「继续录制」。
    s2 = rec.request_toggle()
    assert s2 == "paused"
    assert RECORD_BTN_TEXT[s2] == "继续录制"

    # 第三次 toggle: paused→recording → 文本「暂停录制」。
    s3 = rec.request_toggle()
    assert s3 == "recording"
    assert RECORD_BTN_TEXT[s3] == "暂停录制"

    # 完整文本序列断言。
    texts = [RECORD_BTN_TEXT[s] for s in ("idle", s1, s2, s3)]
    assert texts == ["开始录制", "暂停录制", "继续录制", "暂停录制"]


def test_toggle_noop_before_session():
    """会话未运行时 toggle 为 no-op，保持 idle / 文本「开始录制」（需求 5.1）。"""
    rec = _toggle_only_factory()
    assert rec.request_toggle() == "idle"
    assert RECORD_BTN_TEXT[rec.state] == "开始录制"


# ---------------------------------------------------------------------------
# 3. App._set_running_controls 运行态 enable/disable 联动（需真实 Tk）
# ---------------------------------------------------------------------------

def _make_app_or_skip():
    """尝试构造 Tk root 与 App；无显示环境（TclError）时 skip。

    返回 (root, app)。调用方负责在 finally 中 root.destroy()。
    """
    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover - 环境缺 tkinter
        pytest.skip(f"tkinter 不可用：{exc}")

    from apps.app_ui import App

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"无显示环境，无法创建 Tk 窗口：{exc}")

    try:
        root.withdraw()  # 不显示窗口
        app = App(root)
    except tk.TclError as exc:  # pragma: no cover - 构建期窗口错误
        root.destroy()
        pytest.skip(f"构建 App 失败（无显示）：{exc}")

    return root, app


def _state_str(widget) -> str:
    """读取控件状态字符串（兼容 ttk state 标志与 -state 选项）。"""
    try:
        return str(widget.cget("state"))
    except Exception:
        return ""


def test_set_running_controls_enable_disable_linkage():
    """运行态联动：运行中禁用 Camera/Model、启用 Record_Toggle 并置「开始录制」、
    Start 文本切「停止」、Compare 始终 enabled；未运行恢复（需求 2.5、3.5、4.3、5.2、6.4、6.5）。"""
    root, app = _make_app_or_skip()
    try:
        # 真实启动路径 _start 先经 _begin_preview_session 设置录制就绪标志，再调用
        # _set_running_controls(True)：单摄会话为 True，双摄会话为 False（两路首帧
        # 就绪后才置 True，见下一个测试）。这里模拟单摄会话。
        app._dual_recording_ready = True

        # ---- 运行中 ----
        app._set_running_controls(True)

        # Camera_Selector 禁用（需求 2.5）。
        camera_combo = getattr(app, "camera_combo", None)
        if camera_combo is not None:
            assert _state_str(camera_combo) == "disabled"

        # Model_Selector 禁用（需求 3.5）—— 控件可能尚未创建（任务 9.x）。
        model_combo = getattr(app, "model_combo", None)
        if model_combo is not None:
            assert _state_str(model_combo) == "disabled"

        # Record_Toggle 启用并置「开始录制」（需求 5.2）—— 可能尚未创建。
        record_btn = getattr(app, "record_btn", None)
        if record_btn is not None:
            assert _state_str(record_btn) == "normal"
            assert str(record_btn.cget("text")) == RECORD_BTN_TEXT["idle"]

        # Start_Control 文本切「停止」、禁用（需求 4.3）。
        start_btn = getattr(app, "start_btn", None)
        if start_btn is not None:
            assert str(start_btn.cget("text")) == "停止"
            assert _state_str(start_btn) == "disabled"

        # Compare_Control 始终 enabled（需求 6.4）—— 可能尚未创建。
        compare_btn = getattr(app, "compare_btn", None)
        if compare_btn is not None:
            assert _state_str(compare_btn) == "normal"

        # ---- 未运行 ----
        app._set_running_controls(False)

        # Status_Area 显示「就绪」（需求 4.5）。
        assert app.status_var.get() == "就绪"

        # Start_Control 恢复「开始」、启用。
        if start_btn is not None:
            assert str(start_btn.cget("text")) == "开始"
            assert _state_str(start_btn) == "normal"

        # Model_Selector 恢复 readonly。
        if model_combo is not None:
            assert _state_str(model_combo) == "readonly"

        # Record_Toggle 未运行禁用（需求 5.1）。
        if record_btn is not None:
            assert _state_str(record_btn) == "disabled"

        # Compare_Control 未运行仍 enabled（需求 6.5）。
        if compare_btn is not None:
            assert _state_str(compare_btn) == "normal"
    finally:
        root.destroy()


def test_set_running_controls_defers_record_until_dual_ready():
    """双摄会话两路首帧就绪前：其余运行态联动照常，Record_Toggle 保持禁用。

    就绪后由 ``_post_dual_recording_ready`` 单独启用录制按钮，不再经过
    ``_set_running_controls``。
    """
    root, app = _make_app_or_skip()
    try:
        # _begin_preview_session 对双摄会话把录制就绪标志置为 False。
        app._dual_recording_ready = False
        app._set_running_controls(True)

        assert _state_str(app.record_btn) == "disabled"
        assert str(app.start_btn.cget("text")) == "停止"
        assert _state_str(app.start_btn) == "disabled"
        camera_combo = getattr(app, "camera_combo", None)
        if camera_combo is not None:
            assert _state_str(camera_combo) == "disabled"
    finally:
        root.destroy()
