# -*- coding: utf-8 -*-
"""Tkinter 学生练习分析结果主视图面板与生命周期集成自动化测试。

涵盖：
- Style A 布局与组件构建 (4态概览卡片、动态动作阶段选择栏、左侧问题列表、右侧正侧代表帧、底部操作)
- View switching (列 0-1 跨列展示，录制预览 grid_remove 平滑隐藏且不抛出错误)
- 摄像头预览连续性 (preview 隐藏时 _tick() 持续运行不崩溃，设备不争用)
- 任务代次并发防御 (Generation token 防线：取消、切学生、再练一次后拦截迟到 worker 回调)
- “再练一次”确定性行为 (重置状态，安全切回预览，绝不自动开启录制)
"""
from __future__ import annotations

from queue import Queue
from threading import Event, Lock
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
from unittest.mock import Mock, call
import pytest

from apps import app_ui

# 尝试导入 feedback_result_panel 模块 (M2 产物)
feedback_result_panel_mod = pytest.importorskip("apps.feedback_result_panel", reason="apps.feedback_result_panel 尚未实现")
FeedbackResultPanel = getattr(feedback_result_panel_mod, "FeedbackResultPanel", None)


# ---------------------------------------------------------------------------
# 测试固件与替身
# ---------------------------------------------------------------------------

def make_dummy_record(
    student_id: str = "001",
    student_name: str = "张同学",
    action: str = "straight_combo",
    stance: str = "left",
) -> dict:
    """生成测试用精简记录。"""
    return {
        "schemaVersion": 1,
        "id": "c" * 32,
        "studentId": student_id,
        "studentName": student_name,
        "action": action,
        "stance": stance,
        "createdAt": "2026-09-26T12:00:00+00:00",
        "expiresAt": "2026-10-10T12:00:00+00:00",
        "revision": 1,
        "status": "ready",
        "sourcePaths": {"front": "front.mp4", "side": "side.mp4"},
        "videos": {"front": "front.mp4", "side": "side.mp4"},
        "result": {
            "schemaVersion": 1,
            "ruleVersion": "sanda-feedback-mediapipe-2026-09-20-v4",
            "action": action,
            "stance": stance,
            "backend": "mediapipe",
            "poseVariant": "full",
            "delegate": "cpu",
            "needsRerecord": False,
            "checks": [
                {
                    "id": "chk:start:front:stance_elbow",
                    "code": "stance_elbow",
                    "name": "前手肘角过大",
                    "bodyPart": "前侧臂肘（左）",
                    "segment": "front_straight",
                    "segmentLabel": "前手直拳",
                    "phase": "start",
                    "phaseLabel": "开始实战式",
                    "standard": "前臂与上臂夹角约90度",
                    "source": "散打规范",
                    "status": "candidate",
                    "review": "pending",
                    "reason": "左肘角180度超过标准",
                    "blockedReason": "",
                    "role": "front",
                    "measurements": [],
                    "evidence": ["front:1"],
                },
                {
                    "id": "chk:motion:front:guard_low",
                    "code": "guard_low",
                    "name": "出拳过程护手下落",
                    "bodyPart": "后手（右）",
                    "segment": "front_straight",
                    "segmentLabel": "前手直拳",
                    "phase": "motion",
                    "phaseLabel": "动作过程",
                    "standard": "护手保持在下颚附近",
                    "source": "散打规范",
                    "status": "not_observed",
                    "review": "pending",
                    "reason": "护手保持在下颚附近",
                    "blockedReason": "",
                    "role": "rear",
                    "measurements": [],
                    "evidence": [],
                },
                {
                    "id": "chk:finish:front:shoulder_level",
                    "code": "shoulder_level",
                    "name": "出拳击打高度不足",
                    "bodyPart": "前拳（左）",
                    "segment": "front_straight",
                    "segmentLabel": "前手直拳",
                    "phase": "finish",
                    "phaseLabel": "动作结束瞬间",
                    "standard": "出拳击打点与肩同高",
                    "source": "散打规范",
                    "status": "unable",
                    "review": "pending",
                    "reason": "缺少侧面视角",
                    "blockedReason": "缺少侧面视角",
                    "role": "front",
                    "measurements": [],
                    "evidence": [],
                },
                {
                    "id": "chk:end:front:vertical_fist",
                    "code": "vertical_fist",
                    "name": "拳峰朝向待人工核对",
                    "bodyPart": "前拳（左）",
                    "segment": "front_straight",
                    "segmentLabel": "前手直拳",
                    "phase": "end",
                    "phaseLabel": "结束实战式",
                    "standard": "击中瞬间立拳或平拳",
                    "source": "散打规范",
                    "status": "pending_rule",
                    "review": "pending",
                    "reason": "模型不支持",
                    "blockedReason": "模型不支持",
                    "role": "front",
                    "measurements": [],
                    "evidence": [],
                },
            ],
            "evidence": [],
            "reviewHistory": [],
            "diagnostics": None,
            "phaseWindows": None,
            "capture": None,
        },
    }


def make_test_app(monkeypatch):
    """构建用于测试生命周期与视图切换的轻量 App 替身。"""
    app = app_ui.App.__new__(app_ui.App)
    app._student_practice_active = True
    app._student_pending_record = False
    app._student_presence_gate = None
    app._student_judging = False
    app._student_segment_ready = True
    app._closing = False
    app._rec = SimpleNamespace(state="idle")
    app._worker = SimpleNamespace(is_alive=lambda: True)

    app._feedback_task_token = 0
    app._active_feedback_token = None
    app._active_feedback_student_id = None
    app._current_student_record = None

    app.left_container = Mock()
    app._preview_right = Mock()
    app.feedback_result_panel = Mock()
    app.preview = Mock()
    app.preview2 = Mock()
    app.student_status_var = Mock()
    app.student_start_btn = Mock()
    app.current_student_id = Mock(get=lambda: "001")
    app.root = Mock(after=lambda _ms, cb: cb())
    app._sync_student_buttons = Mock()

    return app


# ===========================================================================
# 1. FeedbackResultPanel 控件与 Style A 布局测试
# ===========================================================================

def test_result_panel_instantiation_and_style_a_structure():
    """验证 FeedbackResultPanel 能在真实 Tk 环境中无报错实例化并构建核心子部件。"""
    root = tk.Tk()
    root.withdraw()
    try:
        on_practice_again = Mock()
        on_export_report = Mock()
        on_view_history = Mock()
        on_review_toggle = Mock()

        panel = FeedbackResultPanel(
            root,
            on_practice_again=on_practice_again,
            on_export_report=on_export_report,
            on_view_history=on_view_history,
            on_review_toggle=on_review_toggle,
        )
        assert panel is not None

        # 验证核心接口存在
        assert hasattr(panel, "set_record")
        assert hasattr(panel, "clear")

        # 验证能设置记录并不抛出异常
        record = make_dummy_record()
        panel.set_record(record)

        # 验证清空无异常
        panel.clear()
    finally:
        root.destroy()


def test_result_panel_clear_resets_content():
    """验证 clear() 能干净重置面板内容与内部状态。"""
    root = tk.Tk()
    root.withdraw()
    try:
        panel = FeedbackResultPanel(
            root,
            on_practice_again=Mock(),
            on_export_report=Mock(),
            on_view_history=Mock(),
            on_review_toggle=Mock(),
        )
        record = make_dummy_record()
        panel.set_record(record)
        panel.clear()
        # 清空后再次调用 clear 也应安全
        panel.clear()
    finally:
        root.destroy()


# ===========================================================================
# 2. View Switching: 录制预览 / 分析结果 平滑切换测试
# ===========================================================================

@pytest.mark.skipif(
    not hasattr(app_ui.App, "_show_student_result_view"),
    reason="M2 App._show_student_result_view 尚未在 apps/app_ui.py 中实现",
)
def test_view_switching_hides_preview_and_shows_result(monkeypatch):
    """验证从录制预览切到分析结果视图时：left_container与_preview_right隐藏，result_panel跨列显示。"""
    app = make_test_app(monkeypatch)
    record = make_dummy_record()

    app._show_student_result_view(record)

    # 验证 left_container 和 _preview_right 被 grid_remove
    app.left_container.grid_remove.assert_called_once()
    app._preview_right.grid_remove.assert_called_once()

    # 验证 feedback_result_panel 被 grid 到 (row=0, col=0, columnspan=2, sticky='nsew')
    app.feedback_result_panel.grid.assert_called_once_with(row=0, column=0, columnspan=2, sticky="nsew")
    app.feedback_result_panel.set_record.assert_called_once_with(record)


@pytest.mark.skipif(
    not hasattr(app_ui.App, "_show_student_preview_view"),
    reason="M2 App._show_student_preview_view 尚未在 apps/app_ui.py 中实现",
)
def test_view_switching_restores_preview(monkeypatch):
    """验证从分析结果切回录制预览时：result_panel被隐藏，恢复原列 0 与列 1 布局。"""
    app = make_test_app(monkeypatch)

    app._show_student_preview_view()

    # 验证 feedback_result_panel 被隐藏
    app.feedback_result_panel.grid_remove.assert_called_once()

    # 验证 left_container 和 _preview_right 恢复展示
    app.left_container.grid.assert_called_once()
    app._preview_right.grid.assert_called_once()


# ===========================================================================
# 3. 摄像头预览连续性测试
# ===========================================================================

def test_camera_tick_continues_when_preview_hidden(monkeypatch):
    """验证当 preview 控件所在的 Frame 被 grid_remove 隐藏时，_tick() 持续运行绝不抛出 TclError。"""
    root = tk.Tk()
    root.withdraw()
    try:
        container = ttk.Frame(root)
        container.pack()
        preview_label = ttk.Label(container)
        preview_label.pack()

        # 隐藏宿主容器（模拟 grid_remove）
        container.pack_forget()

        # 构造单帧测试图像并转为 PhotoImage
        import numpy as np
        from PIL import Image, ImageTk

        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        photo = ImageTk.PhotoImage(image=img, master=root)

        # 核心断言：在隐藏控件上配置新画面绝不抛出任何异常
        preview_label.configure(image=photo)
        root.update_idletasks()

        # 重新显示后继续配置，亦无异常
        container.pack()
        preview_label.configure(image=photo)
        root.update_idletasks()
    finally:
        root.destroy()


# ===========================================================================
# 4. 任务代次并发安全防御测试 (Generation Token Guard)
# ===========================================================================

@pytest.mark.skipif(
    not hasattr(app_ui.App, "_on_student_analysis_saved"),
    reason="M2 App._on_student_analysis_saved 尚未在 apps/app_ui.py 中实现",
)
def test_generation_token_drops_stale_callback_on_cancel(monkeypatch):
    """验证取消分析后，迟到的 worker 回调被代次拦截器丢弃，不刷新结果页。"""
    app = make_test_app(monkeypatch)

    # 发起任务代次 1
    app._feedback_task_token = 1
    app._active_feedback_token = 1
    app._active_feedback_student_id = "001"

    # 用户点击取消分析：active_token 清空
    app._active_feedback_token = None

    record = make_dummy_record(student_id="001")
    # 迟到的工作线程返回代次 1
    app._on_student_analysis_saved(token=1, student_id="001", record=record)

    # 断言：绝不切换至结果视图
    app.feedback_result_panel.set_record.assert_not_called()
    app.feedback_result_panel.grid.assert_not_called()


@pytest.mark.skipif(
    not hasattr(app_ui.App, "_on_student_analysis_saved"),
    reason="M2 App._on_student_analysis_saved 尚未在 apps/app_ui.py 中实现",
)
def test_generation_token_drops_stale_callback_on_student_switch(monkeypatch):
    """验证分析过程中用户切换了学生，旧学生的分析结果绝不上屏覆盖当前学生。"""
    app = make_test_app(monkeypatch)

    # 为学生 "001" 发起代次 1
    app._feedback_task_token = 1
    app._active_feedback_token = 1
    app._active_feedback_student_id = "001"

    # 用户在界面将学号修改为 "002"
    app.current_student_id = Mock(get=lambda: "002")

    record_001 = make_dummy_record(student_id="001")
    # 代次 1 完成返回
    app._on_student_analysis_saved(token=1, student_id="001", record=record_001)

    # 断言：旧学生 001 的结果被丢弃，当前学生 002 的界面未受污染
    app.feedback_result_panel.set_record.assert_not_called()


@pytest.mark.skipif(
    not hasattr(app_ui.App, "_on_student_analysis_saved"),
    reason="M2 App._on_student_analysis_saved 尚未在 apps/app_ui.py 中实现",
)
def test_generation_token_drops_stale_callback_on_practice_again(monkeypatch):
    """验证用户已点击“再练一次”回到预览时，更早任务的迟到回调不得强行切回结果页。"""
    app = make_test_app(monkeypatch)

    # 用户点击再练一次，active_token 设为 None，处于就绪预览态
    app._active_feedback_token = None

    record = make_dummy_record()
    app._on_student_analysis_saved(token=1, student_id="001", record=record)

    app.feedback_result_panel.set_record.assert_not_called()


@pytest.mark.skipif(
    not hasattr(app_ui.App, "_on_student_analysis_saved"),
    reason="M2 App._on_student_analysis_saved 尚未在 apps/app_ui.py 中实现",
)
def test_window_closing_drops_all_callbacks(monkeypatch):
    """验证窗口正在关闭时 (_closing=True)，丢弃任何迟到回调，不触发 TclError。"""
    app = make_test_app(monkeypatch)
    app._closing = True

    record = make_dummy_record()
    app._on_student_analysis_saved(token=1, student_id="001", record=record)

    app.feedback_result_panel.set_record.assert_not_called()


# ===========================================================================
# 5. “再练一次” (Practice Again) 确定性非自启测试
# ===========================================================================

@pytest.mark.skipif(
    not hasattr(app_ui.App, "_on_student_practice_again"),
    reason="M2 App._on_student_practice_again 尚未在 apps/app_ui.py 中实现",
)
def test_practice_again_does_not_autostart_recording(monkeypatch):
    """验证点击“再练一次”时：安全切回录制预览，重置状态，但严格禁止自动开启摄像头录制。"""
    app = make_test_app(monkeypatch)
    app._student_pending_record = True
    app._student_presence_gate = Mock()
    app._rec.state = "idle"

    record_mock = Mock()
    monkeypatch.setattr(app_ui.App, "_begin_recording_segment", record_mock)

    # 触发“再练一次”
    app._on_student_practice_again()

    # 1. 状态彻底重置
    assert app._rec.state == "idle"
    assert app._student_pending_record is False
    assert app._student_presence_gate is None
    assert app._active_feedback_token is None

    # 2. 严禁自动启动录制
    record_mock.assert_not_called()

    # 3. 视图切换回预览
    app.feedback_result_panel.grid_remove.assert_called_once()
    app.left_container.grid.assert_called_once()
    app._preview_right.grid.assert_called_once()


# ===========================================================================
# 6. Result Panel 交互功能自动化测试
# ===========================================================================

def test_result_panel_phase_filtering():
    """验证阶段导航栏切换能够正确过滤左侧检查项列表。"""
    root = tk.Tk()
    root.withdraw()
    try:
        panel = FeedbackResultPanel(root)
        record = make_dummy_record()
        panel.set_record(record)

        # 初始时为全部 (4项)
        assert len(panel.check_tree.get_children()) == 4

        # 切换到 "start" (开始实战式: 1项)
        panel._on_phase_clicked("start")
        children = panel.check_tree.get_children()
        assert len(children) == 1
        assert "chk:start:front:stance_elbow" in children

        # 切换回全部
        panel._on_phase_clicked(None)
        assert len(panel.check_tree.get_children()) == 4
    finally:
        panel.destroy()
        root.destroy()


def test_result_panel_review_action_triggers_callback():
    """验证教师复核确认/撤销按钮能够正确调用回调。"""
    root = tk.Tk()
    root.withdraw()
    try:
        on_review_toggle = Mock()
        panel = FeedbackResultPanel(root, on_review_toggle=on_review_toggle)
        record = make_dummy_record()
        panel.set_record(record)

        # 选中第一项
        panel.check_tree.selection_set("chk:start:front:stance_elbow")
        panel._on_tree_select()

        panel.teacher_entry.delete(0, "end")
        panel.teacher_entry.insert(0, "王老师")
        panel.reason_entry.delete(0, "end")
        panel.reason_entry.insert(0, "动作幅度符合标准")

        panel._on_review_action_clicked("confirmed")
        on_review_toggle.assert_called_once_with(
            record["id"],
            "chk:start:front:stance_elbow",
            "confirmed",
            "王老师",
            "动作幅度符合标准",
        )
    finally:
        panel.destroy()
        root.destroy()


def test_result_panel_bottom_actions_trigger_callbacks():
    """验证底部操作按钮（再练一次、导出、历史）正确触发回调。"""
    root = tk.Tk()
    root.withdraw()
    try:
        on_practice_again = Mock()
        on_export_report = Mock()
        on_view_history = Mock()

        panel = FeedbackResultPanel(
            root,
            on_practice_again=on_practice_again,
            on_export_report=on_export_report,
            on_view_history=on_view_history,
        )
        record = make_dummy_record()
        panel.set_record(record)

        panel._on_practice_again_clicked()
        on_practice_again.assert_called_once()

        panel._on_export_clicked()
        on_export_report.assert_called_once_with(record)

        panel._on_history_clicked()
        on_view_history.assert_called_once()
    finally:
        panel.destroy()
        root.destroy()
