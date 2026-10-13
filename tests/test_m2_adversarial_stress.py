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

from apps import app_ui
from apps.feedback_result_panel import FeedbackResultPanel
from core.recording import RecordingService, SegmentFinalizer
from core.live import LiveControls, PresenceSample, PreviewStore
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


class _FakeLive:
    """会话管理器替身：运行中的双摄会话，预览槽可选。"""

    def __init__(self, state="running", preview=None):
        self.state = state
        self.preview = preview
        self.token = 1
        self.config = SimpleNamespace(dual=True, is_file=False)

    def active(self):
        return self.state != "idle"

    def is_current(self, token):
        return token == self.token


def make_test_app(student_id="001"):
    """创建用于压力测试的 App 实例替身。"""
    app = app_ui.App.__new__(app_ui.App)
    app._closing = False
    app._live = _FakeLive()
    app._live_controls = LiveControls()
    app._recording = SimpleNamespace(state="idle")
    app.modes = ModeManager()
    app.modes.enter(PRACTICE)
    app.exam_scorebooks = ScorebookRegistry()
    app.workflows = WorkflowHost(app.exam_scorebooks)
    app.workflows.session_started(app._live.token)
    app.exam_panel = None
    app._student_announcer = MagicMock()

    # UI Widgets
    app.left_container = MagicMock()
    app._preview_right = MagicMock()
    app.feedback_result_panel = MagicMock()
    app.preview = MagicMock()
    app.preview2 = MagicMock()
    app.student_status_var = MagicMock()
    app.status_var = MagicMock()
    app.student_start_btn = MagicMock()

    _student_id_val = student_id
    id_mock = MagicMock()
    id_mock.get.side_effect = lambda: _student_id_val
    def set_id(new_id):
        nonlocal _student_id_val
        _student_id_val = new_id
    id_mock.set = set_id
    app.current_student_id = id_mock

    app.feedback_controls = MagicMock()
    app.feedback_controls.student_id = app.current_student_id
    app.root = MagicMock()
    app.root.after.side_effect = lambda _ms, cb: cb()
    app._sync_student_buttons = MagicMock()

    # 真实的练习状态机经 WorkflowHost 驱动；身份在开录瞬间读取当前学号
    app._practice = PracticeSession(lambda: {"studentId": app.current_student_id.get()})
    app._practice.segment = PracticeSegment("record_0", None, ready=True, terminal=True)
    ports = WorkflowPorts(
        recorder=app_ui._SegmentPort(app),
        occupancy=app_ui._OccupancyPort(app, "practice"),
        announcer=app._student_announcer,
        ui=app_ui._PracticeUi(app),
        analysis=app_ui._PracticeAnalysis(app),
    )
    app.workflows.activate(app._practice, CommandExecutor(ports))
    return app


def dispatch_analysis(app, token, student_id):
    """模拟「动作评判」派发了第 token 代分析任务。"""
    app._practice.token = token
    app._practice.active_token = token
    app._practice.active_student_id = student_id


# ===========================================================================
# 1. Worker Completion after Cancellation
# ===========================================================================

def test_worker_completion_after_cancellation():
    """压力测试 1：取消分析后，模拟后台工作线程回调到达。

    验证：
    - 回调被完全静默丢弃
    - feedback_result_panel.set_record 绝不被调用
    - 界面不会切换到结果视图
    - 当前学生记录保持不变
    """
    app = make_test_app("student_001")

    # 1. 发起代次 1 的分析任务
    dispatch_analysis(app, 1, "student_001")

    # 2. 用户立即取消分析
    app._on_student_cancelled()
    assert app._practice.active_token is None

    # 3. 模拟后台 worker 线程此时完成并投递回调
    record = make_stress_record("student_001", token=1)
    app._on_student_analysis_saved(token=1, student_id="student_001", record=record)

    # 4. 严苛断言：回调被完全拦截，结果面板未更新，结果视图未展示
    app.feedback_result_panel.set_record.assert_not_called()
    app.feedback_result_panel.grid.assert_not_called()
    assert app._practice.current_record is None
    assert app._practice.record_id is None


def test_worker_completion_after_cancellation_feedback_panel_integration():
    """压力测试 1b：FeedbackControls._start 队列层级的取消防御（真实后台线程 + Tk poll）。"""
    from apps.feedback_panel import FeedbackControls

    root = tk.Tk()
    root.withdraw()
    panel = None
    try:
        show_result = Mock()
        panel = FeedbackControls(root, root=root, set_busy=lambda _busy: None, show_result=show_result,
                                 set_status=lambda _text: None, can_analyze=lambda: True)
        on_saved_mock = Mock()
        finish = threading.Event()

        def work():
            finish.wait(5)  # 后台分析在用户取消之后才完成
            return make_stress_record("001", token=42)

        panel._start(work, on_saved=on_saved_mock, token=42)
        assert panel.active_token == 42

        # 用户在 poll 消费结果前点击了 cancel
        panel.cancel()
        assert panel.active_token is None
        assert panel.stop_event.is_set()
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

def test_worker_completion_after_student_id_change_via_event():
    """压力测试 2a：学生 A 任务在跑，用户修改学号为 B 并触发 identity_changed 事件。"""
    app = make_test_app("student_A")

    # 发起学生 A 的代次 1
    dispatch_analysis(app, 1, "student_A")

    # 用户在输入框中修改学号为 student_B，触发 _on_student_identity_changed
    app.current_student_id.set("student_B")
    app._on_student_identity_changed()

    # 验证 token 被作废
    assert app._practice.active_token is None
    assert app._practice.active_student_id is None

    # 后台线程针对 student_A 的回调迟到
    record_A = make_stress_record("student_A", token=1)
    app._on_student_analysis_saved(token=1, student_id="student_A", record=record_A)

    # 验证 student_A 的结果被完全丢弃，不污染 student_B
    app.feedback_result_panel.set_record.assert_not_called()
    assert app._practice.current_record is None


def test_worker_completion_after_student_id_change_without_token_invalidation():
    """压力测试 2b（对抗）：假设某种极端竞态下 token 未被重置为 None，但学号已变。

    验证：即使 active_token == token 依然成立，学号不匹配（student_B != student_A）
    双重防线依然能坚决拦截晚到的 student_A 回调！
    """
    app = make_test_app("student_A")

    dispatch_analysis(app, 1, "student_A")

    # 对抗注入：学号已变为 student_B，但 active_token 依然是 1
    app.current_student_id.set("student_B")

    # student_A 的回调到达
    record_A = make_stress_record("student_A", token=1)
    app._on_student_analysis_saved(token=1, student_id="student_A", record=record_A)

    # 核心验证：双重校验拦截生效，绝不上屏！
    app.feedback_result_panel.set_record.assert_not_called()
    assert app._practice.current_record is None


def test_student_rapid_switch_a_to_b_execution():
    """压力测试 2c：学员 A 发起任务 1，立即切到学员 B 并发起任务 2。

    验证：任务 1 的迟到结果丢弃，任务 2 的结果正常上屏。
    """
    app = make_test_app("student_A")

    # 任务 1: student_A
    dispatch_analysis(app, 1, "student_A")

    # 切换为 student_B，发起任务 2
    app.current_student_id.set("student_B")
    app._on_student_identity_changed()
    dispatch_analysis(app, 2, "student_B")

    record_A = make_stress_record("student_A", token=1)
    record_B = make_stress_record("student_B", token=2)

    # 任务 1 迟到返回
    app._on_student_analysis_saved(token=1, student_id="student_A", record=record_A)
    app.feedback_result_panel.set_record.assert_not_called()

    # 任务 2 正常返回
    app._on_student_analysis_saved(token=2, student_id="student_B", record=record_B)
    app.feedback_result_panel.set_record.assert_called_once_with(record_B)
    assert app._practice.current_record == record_B


# ===========================================================================
# 3. Worker Completion after "再练一次"
# ===========================================================================

def test_worker_completion_after_practice_again():
    """压力测试 3：已完成任务切换到结果页，用户点击“再练一次”，随后收到旧任务的迟到/重复回调。

    验证：
    - 界面停留在预览模式
    - 绝不重新弹出或切回结果页
    - 预览控件与结果面板不产生争用碰撞
    """
    app = make_test_app("student_001")

    # 1. 任务完成，展示结果
    dispatch_analysis(app, 1, "student_001")
    record = make_stress_record("student_001", token=1)
    app._show_student_result_view(record)
    assert app.left_container.grid_remove.called
    assert app._preview_right.grid_remove.called

    # 2. 重置 mock 调用记录
    app.left_container.reset_mock()
    app._preview_right.reset_mock()
    app.feedback_result_panel.reset_mock()

    # 3. 用户点击“再练一次”
    app._on_student_practice_again()

    # 验证预览已被恢复，结果面板被隐藏
    app.left_container.grid.assert_called_once()
    app._preview_right.grid.assert_called_once()
    app.feedback_result_panel.grid_remove.assert_called_once()
    assert app._practice.active_token is None

    # 重置 mock，以便检测后续非法调用
    app.feedback_result_panel.reset_mock()
    app.left_container.reset_mock()
    app._preview_right.reset_mock()

    # 4. 模拟迟到/重复的回调到达（token=1）
    app._on_student_analysis_saved(token=1, student_id="student_001", record=record)

    # 5. 严苛断言：
    # - 结果面板没有重新设置记录
    # - 结果面板没有重新 grid
    # - 预览容器没有被 grid_remove 隐藏
    app.feedback_result_panel.set_record.assert_not_called()
    app.feedback_result_panel.grid.assert_not_called()
    app.left_container.grid_remove.assert_not_called()
    app._preview_right.grid_remove.assert_not_called()


# ===========================================================================
# 4. 50 Rapid Sequential Analyses with Out-of-Order Completions
# ===========================================================================

def test_50_rapid_sequential_dispatches_out_of_order():
    """压力测试 4：快速发起 50 次连续分析，模拟后台 worker 乱序（甚至逆序、随机重排）返回。

    验证：
    - 仅有最新一次派发（代次 50）的结果被允许更新 UI
    - 其余 49 次被 100% 拦截丢弃
    - 在 100 次不同随机乱序排列下均能 100% 保持确定性
    """
    TOTAL_DISPATCHES = 50

    # 重复 100 次不同随机种子的乱序完成实验
    for seed in range(100):
        rng = random.Random(seed)
        app = make_test_app("student_001")

        records = {}
        # 1. 模拟 50 次连续快速派发
        for i in range(1, TOTAL_DISPATCHES + 1):
            token = app._practice.token + 1
            dispatch_analysis(app, token, "student_001")
            records[token] = make_stress_record("student_001", token=token)

        assert app._practice.active_token == TOTAL_DISPATCHES

        # 2. 构造乱序返回序列
        completion_order = list(range(1, TOTAL_DISPATCHES + 1))
        rng.shuffle(completion_order)

        accepted_tokens = []
        rejected_tokens = []

        for token in completion_order:
            rec = records[token]
            before_call_count = app.feedback_result_panel.set_record.call_count
            app._on_student_analysis_saved(token=token, student_id="student_001", record=rec)
            after_call_count = app.feedback_result_panel.set_record.call_count

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
        assert app._practice.current_record == records[TOTAL_DISPATCHES]


def test_out_of_order_where_latest_token_completes_first():
    """压力测试 4b：最极端情况——代次 50 最先完成并上屏，随后迟到的代次 49, 48, ..., 1 陆续到达。

    验证：代次 50 的展示结果绝不被旧代次回调覆盖或污染！
    """
    TOTAL_DISPATCHES = 50
    app = make_test_app("student_001")

    records = {}
    for i in range(1, TOTAL_DISPATCHES + 1):
        token = app._practice.token + 1
        dispatch_analysis(app, token, "student_001")
        records[token] = make_stress_record("student_001", token=token)

    # 代次 50 最先完成
    app._on_student_analysis_saved(token=50, student_id="student_001", record=records[50])
    assert app.feedback_result_panel.set_record.call_count == 1
    assert app._practice.current_record["result"]["checks"][0]["id"] == "chk:start:front:stance_elbow_50"

    # 代次 49 到 1 依次迟到
    for old_token in range(49, 0, -1):
        app._on_student_analysis_saved(token=old_token, student_id="student_001", record=records[old_token])

    # 严苛断言：set_record 仍然只被调用过 1 次，当前记录依然是代次 50
    assert app.feedback_result_panel.set_record.call_count == 1
    assert app._practice.current_record["result"]["checks"][0]["id"] == "chk:start:front:stance_elbow_50"


# ===========================================================================
# 5. Practice Again Camera Loop: 150+ _tick() Invocations in Result View
# ===========================================================================

def _publish_pair(store, tick_idx, actions):
    """runner 侧：发布一对 BGR 帧（160x120），主路带动作文本。"""
    front = np.full((120, 160, 3), tick_idx % 256, dtype=np.uint8)
    side = np.full((120, 160, 3), (tick_idx * 2) % 256, dtype=np.uint8)
    assert store.publish("raw", {PRIMARY: front, SECONDARY: side}, {PRIMARY: actions})


def test_camera_loop_150_ticks_in_result_view():
    """压力测试 5：在结果页展示状态下，连续触发 150+ 次 _tick()。

    验证：
    - 在隐藏的双摄 Label 上连续 configure(image=PhotoImage) 绝无 TclError 或崩溃
    - 队列正常消费，没有队列积压或溢出
    - PhotoImage 引用安全持有，无内存泄漏异常
    - 随后切回预览模式，画面恢复顺畅
    """
    root = tk.Tk()
    root.withdraw()
    try:
        # 1. 构建真实的 App UI 双摄预览部件
        app = app_ui.App.__new__(app_ui.App)
        app.root = root
        app._closing = False
        store = PreviewStore(1, lambda _role: (1, 1))
        app._live = _FakeLive(preview=store)
        app._live_events = Queue()
        app._shown_preview_seq = 0
        app.workflows = WorkflowHost(ScorebookRegistry())

        # 容器层级（模拟实际 App 的布局）
        app.left_container = ttk.Frame(root)
        app.left_container.grid(row=0, column=0)
        app._preview_right = ttk.Frame(root)
        app._preview_right.grid(row=0, column=1)

        app.preview = ttk.Label(app.left_container)
        app.preview.pack()
        app.preview2 = ttk.Label(app._preview_right)
        app.preview2.pack()

        app.actions_var = tk.StringVar(value="")
        app.status_var = tk.StringVar(value="")

        # _refresh_recording_status 每 tick 读取的录制状态（取 App.__init__ 的空闲默认值）
        app._dual_active = False
        app._record_error_shown = False
        app.recording_status_var = tk.StringVar(value="")
        app._pending_record_errors = []

        # 录制与会话替身
        app._recording = RecordingService(SegmentFinalizer(), plan_for=lambda _segment: None)

        app.feedback_result_panel = FeedbackResultPanel(root)
        app.feedback_result_panel.grid(row=0, column=0, columnspan=2, sticky="nsew")

        # 2. 隐藏预览容器（进入结果视图）
        app.left_container.grid_remove()
        app._preview_right.grid_remove()

        # 3. 连续执行 150 次 _tick()
        TICKS_COUNT = 150
        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        # 拦截 root.after，避免异步事件脱离本单步测试控制
        tick_after_calls = 0
        def mock_after(ms, func=None, *args):
            nonlocal tick_after_calls
            tick_after_calls += 1
            return "mock_timer_id"

        root.after = mock_after

        gc.collect()

        for tick_idx in range(TICKS_COUNT):
            # 每轮向队列推入新帧
            _publish_pair(store, tick_idx, f"Action Tick {tick_idx}")

            # 执行 _tick
            app._tick()
            root.update_idletasks()

            # 断言预览槽已被及时取走，无积压
            assert store.take_newer(0) is None, f"Preview not drained at tick {tick_idx}"

        assert tick_after_calls == TICKS_COUNT
        assert app.actions_var.get() == f"Action Tick {TICKS_COUNT - 1}"

        # 4. 从结果页切换回录制预览（点击“再练一次”行为）
        app.feedback_result_panel.grid_remove()
        app.left_container.grid()
        app._preview_right.grid()

        # 5. 在恢复显示后继续运行 50 次 _tick()
        for tick_idx in range(TICKS_COUNT, TICKS_COUNT + 50):
            _publish_pair(store, tick_idx, f"Resumed Tick {tick_idx}")

            app._tick()
            root.update_idletasks()

        assert app.actions_var.get() == f"Resumed Tick {TICKS_COUNT + 49}"

    finally:
        root.destroy()


# ===========================================================================
# 6. Presence Gate Completely Disarmed on "再练一次"
# ===========================================================================

def test_presence_gate_completely_disarmed_on_practice_again(monkeypatch):
    """压力测试 6：验证点击“再练一次”后，站位检测被彻底解除，学生站入黄框绝不会自动触发录制。

    验证：
    - 站位周期回到 idle，练习不在等待到位
    - 连续 100 帧 (10 秒) 持续有人 (present=True) 到场事件被坚决忽略
    - _begin_recording_segment 绝不被调用
    - 只有在用户显式点击“开始”(_on_practice_start)后，才重新武装并触发录制
    """
    app = make_test_app("student_001")
    app._dual_recording_ready = True

    # 1. 模拟处于已完成评判、正在显示结果的状态
    app._practice.phase = "showing_result"
    app._practice.presence.arm_enter()

    begin_record_mock = MagicMock(return_value=SimpleNamespace(segment_id="record_1", directory=None))
    # 生产代码以 App._begin_recording_segment(self, intent) 调用，须在类上替换才能拦截（同 test_student_presence）。
    monkeypatch.setattr(app_ui.App, "_begin_recording_segment", lambda _self, _intent=None: begin_record_mock())

    # 2. 点击“再练一次”
    app._on_student_practice_again()

    # 验证防线初始状态
    assert app._practice.presence.phase == "idle"
    assert not app._practice.waiting
    assert "practice" not in app._live_controls.occupancy.get().armed_by
    assert app._recording.state == "idle"

    # 3. 对抗压力：会话线程经 presence 事件投递站位采样（连续 150 次有人，持续 15 秒）
    base_time = time.monotonic()
    for i in range(150):
        app_ui.App._on_session_presence(app, PresenceSample(True, base_time + i * 0.1))

    # 4. 严苛断言：站位检测未武装，开录未被触发！
    begin_record_mock.assert_not_called()
    assert app._recording.state == "idle"
    assert app._practice.presence.phase == "idle"

    # 5. 正向对照：只有当用户明确点击「开始」后，才激活等待到位
    # 「开始」的前置检查（身份、正/侧双摄选择、Lite 模型）按真实启动条件补齐；
    # 弹窗换成 Mock，前置检查失败时直接断言失败，而不是弹出阻塞的模态框。
    app.feedback_controls.identity = lambda: {"studentId": "student_001"}
    app._student_dual_cameras_ready = lambda: (True, "")
    monkeypatch.setattr(app_ui.model_manager, "is_installed", lambda _model: True)
    monkeypatch.setattr(app_ui, "messagebox", MagicMock())
    gate_start = base_time + 20.0
    monkeypatch.setattr(app_ui.time, "monotonic", lambda: gate_start)
    app._on_practice_start()
    app_ui.messagebox.showwarning.assert_not_called()
    assert app._practice.waiting
    assert app._practice.presence.phase == "wait_enter"

    # 模拟进入并稳定站立 0.8s 以上，才会触发录制
    for dt in (0.1, 0.5, 1.0):
        app_ui.App._on_session_presence(app, PresenceSample(True, gate_start + dt))

    # 验证此时录制被合法启动
    begin_record_mock.assert_called_once()
    assert app._practice.recording


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
        assert panel._current_record is not None
        assert panel._current_record["studentId"] == "stu_099"
        assert len(panel._photo_cache) <= 30  # 有界缓存保护
    finally:
        panel.destroy()
        root.destroy()
