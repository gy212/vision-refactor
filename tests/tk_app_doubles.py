# -*- coding: utf-8 -*-
"""Tk 桌面测试的替身：会话管理器、摄像头池、录像 writer，以及在真实（隐藏的）Tk 根窗口上组装 ``DesktopApp``。

控制器与视图都是真实的；只替换摄像头设备、会话线程与录像落盘。关窗测试用 ``QueuedRoot`` 接管
``ShutdownCoordinator`` 的 ``after`` 轮询，由测试逐步执行。
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from apps.tk.app import DesktopApp
from apps.tk.shutdown import ShutdownCoordinator
from core.media import CameraService
from core.recording import RecordingService, SegmentFinalizer, SegmentRecorder


class FakeLive:
    """会话管理器替身：记录 start/stop/wait_idle。默认 stop 后会话线程立即结束；
    ``ends_on_stop=False`` 模拟迟迟不结束的会话线程，``on_wait`` 模拟它在等待期间收尾。"""

    def __init__(self, state: str = "idle", *, on_wait=None, ends_on_stop: bool = True, preview=None,
                 config=None) -> None:
        self.state = state
        self.ends_on_stop = ends_on_stop
        self.on_wait = on_wait
        self.stops: list[str] = []
        self.waits: list[float] = []
        self.started: list[object] = []
        self.token = getattr(preview, "token", 1)
        self.config = config
        self.preview = preview

    def active(self) -> bool:
        return self.state != "idle"

    def is_current(self, token: int) -> bool:
        return token == self.token

    def start(self, config, *, start_click=None):
        self.started.append(config)
        self.config = config
        self.state = "starting"
        return self.token

    def stop(self, reason: str = "user_stop") -> bool:
        self.stops.append(reason)
        if self.state in ("idle", "stopping"):
            return False
        self.state = "idle" if self.ends_on_stop else "stopping"
        return True

    def wait_idle(self, timeout: float) -> bool:
        self.waits.append(timeout)
        if self.on_wait is not None:
            self.on_wait()
        return self.state == "idle"


class SyncPool:
    """摄像头池替身：记录声明的预热规格与关闭。"""

    def __init__(self) -> None:
        self.sync_calls: list[tuple[object, object]] = []
        self.release_all_calls = 0
        self.close_calls = 0

    def sync(self, primary, secondary) -> None:
        self.sync_calls.append((primary, secondary))

    def release_all(self) -> None:
        self.release_all_calls += 1

    def cancel(self, role: str, *, generation: int | None = None) -> None:
        pass

    def close(self) -> None:
        self.close_calls += 1

    def wait_released(self, _timeout: float) -> bool:
        return True


class QueuedRoot:
    """关窗协调器的根窗口替身：``after`` 只记录不执行，``destroy`` 只计数。"""

    def __init__(self) -> None:
        self.after_calls: list[tuple[int, object]] = []
        self.destroy_calls = 0

    def after(self, delay: int, callback) -> None:
        self.after_calls.append((delay, callback))

    def destroy(self) -> None:
        self.destroy_calls += 1


class FakeWriter:
    def __init__(self) -> None:
        self.frames: list[object] = []
        self.released = False

    def write(self, frame) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


def writer_factory(failing: frozenset[str] = frozenset()):
    """按视角决定 writer 是否创建失败；单路文件（record_<stamp>.mp4）算 front。"""

    def factory(path, _fps, _size):
        path = Path(path)
        view = "front" if path.stem.startswith("record_") else path.stem
        if view in failing:
            raise RuntimeError(f"{view} writer unavailable")
        return FakeWriter(), path, "fake"

    return factory


class FinalizerDouble:
    """只记录提交的片段与计划，不执行步骤。"""

    def __init__(self) -> None:
        self.jobs: list[SimpleNamespace] = []
        self.cancel_calls = 0
        self.busy = False

    def submit(self, segment, plan) -> bool:
        self.jobs.append(SimpleNamespace(segment=segment, plan=plan))
        return True

    def cancel_all(self) -> None:
        self.cancel_calls += 1

    def close_budget_s(self) -> float:
        return 0.0

    @property
    def idle(self) -> bool:
        return not self.busy


def frame() -> np.ndarray:
    return np.zeros((3, 4, 3), dtype=np.uint8)


def tk_root():
    """隐藏的真实 Tk 根窗口；无显示环境时跳过。"""
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display is unavailable: {exc}")
    root.withdraw()
    return root


def build_app(root, *, pool: SyncPool | None = None, live: FakeLive | None = None) -> DesktopApp:
    """在真实根窗口上组装应用：摄像头池换成替身、启动时不枚举；``live`` 给出时替换会话管理器。"""
    with patch("apps.tk.controllers.cameras.CameraController.start_enumeration"):
        app = DesktopApp(root, camera_service=CameraService(pool=pool or SyncPool()))
    if live is not None:
        app.session.live = live
    return app


def bind_recording(app: DesktopApp, finalizer, *, failing: frozenset[str] = frozenset(),
                   plan_for=None) -> RecordingService:
    """录制服务换成假 writer；计划默认由录制控制器构建。"""
    app.recording.service = RecordingService(
        finalizer, plan_for=plan_for or app.recording.plan, recorder=SegmentRecorder(writer_factory(failing)),
    )
    return app.recording.service


def real_finalizer() -> SegmentFinalizer:
    return SegmentFinalizer()


def attach_queued_close(app: DesktopApp) -> QueuedRoot:
    """关窗协调器的轮询改由测试驱动（不经真实根窗口的 after，也不销毁它）。"""
    queued = QueuedRoot()
    app.shutdown = ShutdownCoordinator(queued, app.ui, app.shutdown_steps())
    return queued


@pytest.fixture
def root():
    """测试模块 ``from tests.tk_app_doubles import root`` 即可使用；用例结束时销毁窗口。"""
    window = tk_root()
    try:
        yield window
    finally:
        try:
            window.destroy()
        except tk.TclError:
            pass


def activate_practice(app: DesktopApp, *, identity=None, clock=None, announcer=None):
    """给已进入练习模式的应用换上一个新的 ``PracticeSession`` 并经 ``WorkflowHost`` 激活。"""
    import time
    from unittest.mock import Mock

    from apps.tk.controllers.session import OccupancyPort
    from core.workflows import CommandExecutor, PracticeSession, WorkflowPorts

    practice = app.practice
    if practice.practice is not None:
        app.workflows.deactivate(practice.practice)
    practice.practice = PracticeSession(identity or app.window.practice.controls.identity)
    practice.announcer = announcer if announcer is not None else Mock()
    ports = WorkflowPorts(recorder=app.recording, occupancy=OccupancyPort(app.controls, "practice"),
                          announcer=practice.announcer, ui=practice, analysis=practice)
    app.workflows.activate(practice.practice, CommandExecutor(ports, clock=clock or (lambda: time.monotonic())))
    return practice.practice


def practice_app(root, *, ready: bool = True, student_id: str = "001", identity=None, clock=None,
                 announcer=None) -> DesktopApp:
    """已进入学生练习、双摄会话运行中的应用：真实的 ``PracticeSession`` 经 ``WorkflowHost`` 驱动。

    练习流程的身份默认读界面学号（``identity`` 可替换）；正/侧摄像头下拉已选好。
    """
    from core.workflows import PRACTICE

    app = build_app(root, live=FakeLive("running", config=SimpleNamespace(dual=True, is_file=False)))
    app.flags.recording_ready = ready
    app.modes.enter(PRACTICE)
    app.workflows.session_started(app.session.live.token)
    app.window.practice.controls.student_id.set(student_id)
    app.window.capture.camera_choice.set("摄像头 0")
    app.window.capture.camera_choice_2.set("摄像头 1")
    activate_practice(app, identity=identity, clock=clock, announcer=announcer)
    return app
