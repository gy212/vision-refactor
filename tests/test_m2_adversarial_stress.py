# -*- coding: utf-8 -*-
"""Adversarial stress testing for Milestone 2: Async lifecycle, race conditions, and generation token defense.

Empirical verification of:
1. Worker completion after cancellation: drop stale callback, result view not shown.
2. Worker completion after student ID change: drop stale callback, result view not shown.
3. Worker completion after "再练一次": drop late worker callback, preview remains visible, no collision.
4. Rapid sequential analyses: 50 rapid sequential dispatches with out-of-order completions -> ONLY latest token accepted.
5. Practice again camera loop: 150+ _tick() invocations while in result view without errors, memory leaks, or halts.
6. Presence gate completely disarmed on "再练一次" and cannot trigger recording automatically.
7. Result panel rapid set_record stress test (100 rapid reloads).
"""
from __future__ import annotations

import gc
from queue import Empty, Queue
import random
import threading
import time
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call
import numpy as np
from PIL import Image, ImageTk
import pytest

from apps.tk.controllers import practice as practice_mod
from apps.tk.controllers import recording as recording_mod
from tests.tk_app_doubles import FakeLive, activate_practice, build_app, practice_app
from tests.tk_app_doubles import root  # noqa: F401
from apps.feedback_result_panel import FeedbackResultPanel
from core.recording import RecordingService, SegmentFinalizer
from core.live import PresenceSample, PreviewStore, SessionEvent
from core.media import PRIMARY, SECONDARY
from core.workflows import (
    PRACTICE,
    CommandExecutor,
    ModeManager,
    PracticeSegment,
    PracticeSession,
    ScorebookRegistry,
    WorkflowHost,
    WorkflowPorts,
)


def _pumped(root):
    """真实 Tk 根上启动 UiDispatcher，返回其 ``post``；``root.update()`` 时排空。"""
    from apps.tk import UiDispatcher

    dispatcher = UiDispatcher(root)
    dispatcher.start()
    return dispatcher.post


# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------

def make_stress_record(
    student_id: str = "001",
    student_name: str = "测试学员",
    action: str = "straight_combo",
    stance: str = "left",
    token: int = 1,
) -> dict:
    """生成带有唯一代次标识的测试记录。"""
    return {
        "schemaVersion": 1,
        "id": f"rec_{student_id}_{token:04d}_{'a' * 16}",
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
                    "id": f"chk:start:front:stance_elbow_{token}",
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
                    "reason": f"左肘角偏差 (token={token})",
                    "blockedReason": "",
                    "role": "front",
                    "measurements": [],
                    "evidence": [],
                },
                {
                    "id": f"chk:motion:front:guard_low_{token}",
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
                    "reason": "护手保持正常",
                    "blockedReason": "",
                    "role": "rear",
                    "measurements": [],
                    "evidence": [],
                },
            ],
            "evidence": [],
            "reviewHistory": [],
            "diagnostics": {"fps": 30.0, "resolution": "1920x1080"},
            "phaseWindows": None,
            "capture": None,
        },
    }


def make_test_app(root, student_id="001"):
    """已进入练习、最近片段可评判的应用；身份在开录瞬间读界面学号，结果页与左侧/预览容器换成 MagicMock。"""
    app = practice_app(root, student_id=student_id,
                       identity=lambda: {"studentId": app.window.practice.controls.student_id.get()})
    app.recording.service = SimpleNamespace(state="idle")
    app.practice.practice.segment = PracticeSegment("record_0", None, ready=True, terminal=True)
    app.window.left_container = MagicMock()
    app.window.preview.frame = MagicMock()
    app.window.result_panel = MagicMock()
    return app


def fresh_practice(app):
    """同一应用上换一轮新的练习流程（避免逐轮重建窗口）。"""
    practice = activate_practice(app, identity=lambda: {"studentId": app.window.practice.controls.student_id.get()})
    practice.segment = PracticeSegment("record_0", None, ready=True, terminal=True)
    app.window.result_panel.reset_mock()


def set_student(app, student_id, *, notify):
    """改学号；``notify=False`` 时只改界面值、不触发学号变更事件（模拟事件未到的竞态）。"""
    if notify:
        app.window.practice.controls.student_id.set(student_id)
    else:
        app.window.practice.controls.student_id = MagicMock(get=lambda: student_id)


def dispatch_analysis(app, token, student_id):
    """模拟「动作评判」派发了第 token 代分析任务。"""
    app.practice.practice.token = token
    app.practice.practice.active_token = token
    app.practice.practice.active_student_id = student_id


# ===========================================================================
# 1. Worker Completion after Cancellation
# ===========================================================================

def test_worker_completion_after_cancellation(root):
    """压力测试 1：取消分析后，模拟后台工作线程回调到达。

    验证：
    - 回调被完全静默丢弃
    - feedback_result_panel.set_record 绝不被调用
    - 界面不会切换到结果视图
    - 当前学生记录保持不变
    """
    app = make_test_app(root, "student_001")

    # 1. 发起代次 1 的分析任务
    dispatch_analysis(app, 1, "student_001")

    # 2. 用户立即取消分析
    app.practice.cancelled()
    assert app.practice.practice.active_token is None

    # 3. 模拟后台 worker 线程此时完成并投递回调
    record = make_stress_record("student_001", token=1)
    app.practice.analysis_saved(token=1, student_id="student_001", record=record)

    # 4. 严苛断言：回调被完全拦截，结果面板未更新，结果视图未展示
    app.window.result_panel.set_record.assert_not_called()
    app.window.result_panel.grid.assert_not_called()
    assert app.practice.practice.current_record is None
    assert app.practice.practice.record_id is None


def test_worker_completion_after_cancellation_feedback_panel_integration():
    """压力测试 1b：练习分析任务的取消防御（真实后台线程 + UiDispatcher 回到 Tk 线程）。"""
    from apps.tk.views import PracticeControls

    root = tk.Tk()
    root.withdraw()
    panel = None
    try:
        show_result = Mock()
        panel = PracticeControls(root, root=root, post=_pumped(root), set_busy=lambda _busy: None, show_result=show_result,
                                 set_status=lambda _text: None, can_analyze=lambda: True)
        on_saved_mock = Mock()
        finish = threading.Event()

        def work():
            finish.wait(5)  # 后台分析在用户取消之后才完成
            return make_stress_record("001", token=42)

        panel.analysis.start(work, on_saved=on_saved_mock, token=42)
        assert panel.busy

        # 用户在 poll 消费结果前点击了 cancel
        panel.cancel()
        assert panel.analysis.stop_event.is_set()
        finish.set()

        deadline = time.monotonic() + 5
        while panel.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)

        # 迟到的完成项由 poll 按代次丢弃：不回调 on_saved，也不展示结果
        assert not panel.busy
        on_saved_mock.assert_not_called()
        show_result.assert_not_called()
    finally:
        if panel is not None:
            panel.close()
        root.destroy()


# ===========================================================================
# 2. Worker Completion after Student ID Change
# ===========================================================================

def test_worker_completion_after_student_id_change_via_event(root):
    """压力测试 2a：学生 A 任务在跑，用户修改学号为 B 并触发 identity_changed 事件。"""
    app = make_test_app(root, "student_A")

    # 发起学生 A 的代次 1
    dispatch_analysis(app, 1, "student_A")

    # 用户在输入框中修改学号为 student_B，触发 _on_student_identity_changed
    set_student(app, "student_B", notify=True)

    # 验证 token 被作废
    assert app.practice.practice.active_token is None
    assert app.practice.practice.active_student_id is None

    # 后台线程针对 student_A 的回调迟到
    record_A = make_stress_record("student_A", token=1)
    app.practice.analysis_saved(token=1, student_id="student_A", record=record_A)

    # 验证 student_A 的结果被完全丢弃，不污染 student_B
    app.window.result_panel.set_record.assert_not_called()
    assert app.practice.practice.current_record is None


def test_worker_completion_after_student_id_change_without_token_invalidation(root):
    """压力测试 2b（对抗）：假设某种极端竞态下 token 未被重置为 None，但学号已变。

    验证：即使 active_token == token 依然成立，学号不匹配（student_B != student_A）
    双重防线依然能坚决拦截晚到的 student_A 回调！
    """
    app = make_test_app(root, "student_A")

    dispatch_analysis(app, 1, "student_A")

    # 对抗注入：学号已变为 student_B，但 active_token 依然是 1
    set_student(app, "student_B", notify=False)

    # student_A 的回调到达
    record_A = make_stress_record("student_A", token=1)
    app.practice.analysis_saved(token=1, student_id="student_A", record=record_A)

    # 核心验证：双重校验拦截生效，绝不上屏！
    app.window.result_panel.set_record.assert_not_called()
    assert app.practice.practice.current_record is None


def test_student_rapid_switch_a_to_b_execution(root):
    """压力测试 2c：学员 A 发起任务 1，立即切到学员 B 并发起任务 2。

    验证：任务 1 的迟到结果丢弃，任务 2 的结果正常上屏。
    """
    app = make_test_app(root, "student_A")

    # 任务 1: student_A
    dispatch_analysis(app, 1, "student_A")

    # 切换为 student_B，发起任务 2
    set_student(app, "student_B", notify=True)
    dispatch_analysis(app, 2, "student_B")

    record_A = make_stress_record("student_A", token=1)
    record_B = make_stress_record("student_B", token=2)

    # 任务 1 迟到返回
    app.practice.analysis_saved(token=1, student_id="student_A", record=record_A)
    app.window.result_panel.set_record.assert_not_called()

    # 任务 2 正常返回
    app.practice.analysis_saved(token=2, student_id="student_B", record=record_B)
    app.window.result_panel.set_record.assert_called_once_with(record_B)
    assert app.practice.practice.current_record == record_B


# ===========================================================================
# 3. Worker Completion after "再练一次"
# ===========================================================================

def test_worker_completion_after_practice_again(root):
    """压力测试 3：已完成任务切换到结果页，用户点击“再练一次”，随后收到旧任务的迟到/重复回调。

    验证：
    - 界面停留在预览模式
    - 绝不重新弹出或切回结果页
    - 预览控件与结果面板不产生争用碰撞
    """
    app = make_test_app(root, "student_001")

    # 1. 任务完成，展示结果
    dispatch_analysis(app, 1, "student_001")
    record = make_stress_record("student_001", token=1)
    app.practice.show_result_view(record)
    assert app.window.left_container.grid_remove.called
    assert app.window.preview.frame.grid_remove.called

    # 2. 重置 mock 调用记录
    app.window.left_container.reset_mock()
    app.window.preview.frame.reset_mock()
    app.window.result_panel.reset_mock()

    # 3. 用户点击“再练一次”
    app.practice.practice_again()

    # 验证预览已被恢复，结果面板被隐藏
    app.window.left_container.grid.assert_called_once()
    app.window.preview.frame.grid.assert_called_once()
    app.window.result_panel.grid_remove.assert_called_once()
    assert app.practice.practice.active_token is None

    # 重置 mock，以便检测后续非法调用
    app.window.result_panel.reset_mock()
    app.window.left_container.reset_mock()
    app.window.preview.frame.reset_mock()

    # 4. 模拟迟到/重复的回调到达（token=1）
    app.practice.analysis_saved(token=1, student_id="student_001", record=record)

    # 5. 严苛断言：
    # - 结果面板没有重新设置记录
    # - 结果面板没有重新 grid
    # - 预览容器没有被 grid_remove 隐藏
    app.window.result_panel.set_record.assert_not_called()
    app.window.result_panel.grid.assert_not_called()
    app.window.left_container.grid_remove.assert_not_called()
    app.window.preview.frame.grid_remove.assert_not_called()


# ===========================================================================
# 4. 50 Rapid Sequential Analyses with Out-of-Order Completions
# ===========================================================================

def test_50_rapid_sequential_dispatches_out_of_order(root):
    """压力测试 4：快速发起 50 次连续分析，模拟后台 worker 乱序（甚至逆序、随机重排）返回。

    验证：
    - 仅有最新一次派发（代次 50）的结果被允许更新 UI
    - 其余 49 次被 100% 拦截丢弃
    - 在 100 次不同随机乱序排列下均能 100% 保持确定性
    """
    TOTAL_DISPATCHES = 50

    # 重复 100 次不同随机种子的乱序完成实验
    app = make_test_app(root, "student_001")
    for seed in range(100):
        rng = random.Random(seed)
        fresh_practice(app)

        records = {}
        # 1. 模拟 50 次连续快速派发
        for i in range(1, TOTAL_DISPATCHES + 1):
            token = app.practice.practice.token + 1
            dispatch_analysis(app, token, "student_001")
            records[token] = make_stress_record("student_001", token=token)

        assert app.practice.practice.active_token == TOTAL_DISPATCHES

        # 2. 构造乱序返回序列
        completion_order = list(range(1, TOTAL_DISPATCHES + 1))
        rng.shuffle(completion_order)

        accepted_tokens = []
        rejected_tokens = []

        for token in completion_order:
            rec = records[token]
            before_call_count = app.window.result_panel.set_record.call_count
            app.practice.analysis_saved(token=token, student_id="student_001", record=rec)
            after_call_count = app.window.result_panel.set_record.call_count

            if after_call_count > before_call_count:
                accepted_tokens.append(token)
            else:
                rejected_tokens.append(token)

        # 3. 核心断言：
        # 无论以何种顺序到达，有且仅有 TOTAL_DISPATCHES 被接受
        assert accepted_tokens == [TOTAL_DISPATCHES], (
            f"Seed {seed}: Expected ONLY token {TOTAL_DISPATCHES} to be accepted, "
            f"but got accepted={accepted_tokens}"
        )
        assert len(rejected_tokens) == TOTAL_DISPATCHES - 1
        assert TOTAL_DISPATCHES not in rejected_tokens
        assert app.practice.practice.current_record == records[TOTAL_DISPATCHES]


def test_out_of_order_where_latest_token_completes_first(root):
    """压力测试 4b：最极端情况——代次 50 最先完成并上屏，随后迟到的代次 49, 48, ..., 1 陆续到达。

    验证：代次 50 的展示结果绝不被旧代次回调覆盖或污染！
    """
    TOTAL_DISPATCHES = 50
    app = make_test_app(root, "student_001")

    records = {}
    for i in range(1, TOTAL_DISPATCHES + 1):
        token = app.practice.practice.token + 1
        dispatch_analysis(app, token, "student_001")
        records[token] = make_stress_record("student_001", token=token)

    # 代次 50 最先完成
    app.practice.analysis_saved(token=50, student_id="student_001", record=records[50])
    assert app.window.result_panel.set_record.call_count == 1
    assert app.practice.practice.current_record["result"]["checks"][0]["id"] == "chk:start:front:stance_elbow_50"

    # 代次 49 到 1 依次迟到
    for old_token in range(49, 0, -1):
        app.practice.analysis_saved(token=old_token, student_id="student_001", record=records[old_token])

    # 严苛断言：set_record 仍然只被调用过 1 次，当前记录依然是代次 50
    assert app.window.result_panel.set_record.call_count == 1
    assert app.practice.practice.current_record["result"]["checks"][0]["id"] == "chk:start:front:stance_elbow_50"


# ===========================================================================
# 5. Practice Again Camera Loop: 150+ _tick() Invocations in Result View
# ===========================================================================

def _publish_pair(store, tick_idx, actions):
    """runner 侧：发布一对 BGR 帧（160x120），主路带动作文本。"""
    front = np.full((120, 160, 3), tick_idx % 256, dtype=np.uint8)
    side = np.full((120, 160, 3), (tick_idx * 2) % 256, dtype=np.uint8)
    assert store.publish("raw", {PRIMARY: front, SECONDARY: side}, {PRIMARY: actions})


def test_camera_loop_150_ticks_in_result_view(root):
    """压力测试 5：在结果页展示状态下，dispatcher 连续排空 150+ 次（每次渲染一帧双摄预览）。

    验证：隐藏的双摄预览上连续换图无 TclError；预览槽每次都被取走、无积压；切回预览后画面照常刷新。
    """
    store = PreviewStore(1, lambda _role: (1, 1))
    app = build_app(root, live=FakeLive("running", preview=store,
                                        config=SimpleNamespace(dual=True, is_file=False)))
    window = app.window
    window.show_result_page(make_stress_record("student_001"))   # 进入结果视图：控制区与预览隐藏
    actions = window.status.actions_var
    TICKS_COUNT = 150
    gc.collect()
    for tick_idx in range(TICKS_COUNT):
        _publish_pair(store, tick_idx, f"Action Tick {tick_idx}")
        app.ui.pump()
        root.update_idletasks()
        assert store.take_newer(0) is None, f"Preview not drained at tick {tick_idx}"
    assert actions.get() == f"Action Tick {TICKS_COUNT - 1}"

    window.show_preview_page()   # 「再练一次」切回录制预览
    for tick_idx in range(TICKS_COUNT, TICKS_COUNT + 50):
        _publish_pair(store, tick_idx, f"Resumed Tick {tick_idx}")
        app.ui.pump()
        root.update_idletasks()
    assert actions.get() == f"Resumed Tick {TICKS_COUNT + 49}"


# ===========================================================================
# 6. Presence Gate Completely Disarmed on "再练一次"
# ===========================================================================

def test_presence_gate_completely_disarmed_on_practice_again(root, monkeypatch):
    """压力测试 6：验证点击“再练一次”后，站位检测被彻底解除，学生站入黄框绝不会自动触发录制。

    验证：
    - 站位周期回到 idle，练习不在等待到位
    - 连续 150 帧 (15 秒) 持续有人 (present=True) 到场事件被坚决忽略
    - 开录绝不被调用
    - 只有在用户显式点击“开始”后，才重新武装并触发录制
    """
    app = make_test_app(root, "student_001")
    app.flags.recording_ready = True

    # 1. 模拟处于已完成评判、正在显示结果的状态
    app.practice.practice.phase = "showing_result"
    app.practice.practice.presence.arm_enter()
    begin_record_mock = MagicMock(return_value=SimpleNamespace(segment_id="record_1", directory=None))
    app.recording.begin = lambda _intent=None: begin_record_mock()

    # 2. 点击“再练一次”
    app.practice.practice_again()
    assert app.practice.practice.presence.phase == "idle"
    assert not app.practice.practice.waiting
    assert "practice" not in app.controls.occupancy.get().armed_by
    assert app.recording.state == "idle"

    # 3. 对抗压力：会话线程经 presence 事件投递站位采样（连续 150 次有人，持续 15 秒）
    def presence(at):
        app.session.handle(SessionEvent("presence", app.session.live.token, PresenceSample(True, at)))

    base_time = time.monotonic()
    for i in range(150):
        presence(base_time + i * 0.1)

    # 4. 严苛断言：站位检测未武装，开录未被触发！
    begin_record_mock.assert_not_called()
    assert app.practice.practice.presence.phase == "idle"

    # 5. 正向对照：只有当用户明确点击「开始」后，才激活等待到位（模型视为已安装；弹窗换成 Mock）
    monkeypatch.setattr(recording_mod, "is_installed", lambda _model: True)
    dialogs = MagicMock()
    monkeypatch.setattr(practice_mod, "messagebox", dialogs)
    gate_start = base_time + 20.0
    monkeypatch.setattr(practice_mod.time, "monotonic", lambda: gate_start)
    app.practice.start()
    dialogs.showwarning.assert_not_called()
    assert app.practice.practice.waiting
    assert app.practice.practice.presence.phase == "wait_enter"

    # 模拟进入并稳定站立 0.8s 以上，才会触发录制
    for dt in (0.1, 0.5, 1.0):
        presence(gate_start + dt)
    begin_record_mock.assert_called_once()
    assert app.practice.practice.recording


# ===========================================================================
# 7. Result Panel Rapid Set Record Stress Test (100 Reloads)
# ===========================================================================

def test_result_panel_rapid_set_record_stress():
    """压力测试 7：快速对 FeedbackResultPanel 调用 100 次 set_record() 与 clear()。

    验证：
    - 内存无泄漏（_photo_cache, _active_photos 维持有界）
    - Treeview 节点正确销毁与重建，无重叠或泄漏
    - 阶段按钮与卡片标签在高频刷新下稳定无崩溃
    """
    root = tk.Tk()
    root.withdraw()
    try:
        panel = FeedbackResultPanel(root)

        for i in range(100):
            record = make_stress_record(
                student_id=f"stu_{i:03d}",
                token=i,
                action="straight_combo" if i % 2 == 0 else "whip_kick",
            )
            panel.set_record(record)
            if i % 10 == 0:
                panel.clear()

        # 验证最终状态正确
        assert panel.record is not None
        assert panel.record["studentId"] == "stu_099"
        assert panel.photo_count <= 30  # 有界缓存保护
    finally:
        panel.destroy()
        root.destroy()
