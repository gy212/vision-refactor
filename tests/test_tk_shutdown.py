"""ShutdownCoordinator：步骤顺序、只生效一次、后台释放不阻塞 Tk、按截止时刻有界等待后销毁窗口。"""
from __future__ import annotations

import threading

from apps.tk import ShutdownCoordinator, ShutdownSteps, UiDispatcher, shutdown


class _Root:
    def __init__(self) -> None:
        self.after_calls: list[tuple[int, object]] = []
        self.destroy_calls = 0

    def after(self, delay, callback) -> None:
        self.after_calls.append((delay, callback))

    def destroy(self) -> None:
        self.destroy_calls += 1


def _steps(log: list[str], *, release=None, session_idle=lambda _t: True, finalizer_idle=lambda: True, extra=0.0):
    return ShutdownSteps(
        prepare_workflows=lambda: log.append("prepare"),
        stop_session=lambda: log.append("stop"),
        extra_budget_s=lambda: extra,
        release=release or (lambda: log.append("release")),
        wait_session_idle=session_idle,
        finalizer_idle=finalizer_idle,
        flush=lambda: log.append("flush"),
    )


def test_runs_steps_in_order_once_and_destroys_when_idle():
    root, log = _Root(), []
    dispatcher = UiDispatcher(root)
    coordinator = ShutdownCoordinator(root, dispatcher, _steps(log))

    coordinator.request()
    coordinator.join_release(1.0)
    if root.after_calls:
        root.after_calls.pop(0)[1]()
    coordinator.request()

    assert log == ["prepare", "stop", "release", "flush"]
    assert root.destroy_calls == 1
    assert dispatcher.destroyed


def test_dispatcher_enters_closing_before_session_stops():
    root = _Root()
    dispatcher = UiDispatcher(root)
    observed: list[bool] = []
    steps = _steps([])
    steps = ShutdownSteps(**{**steps.__dict__, "stop_session": lambda: observed.append(dispatcher.closing)})
    ShutdownCoordinator(root, dispatcher, steps).request()
    assert observed == [True]


def test_failing_workflow_prepare_does_not_block_close():
    root, log = _Root(), []

    def boom():
        raise RuntimeError("panel gone")

    steps = _steps(log)
    steps = ShutdownSteps(**{**steps.__dict__, "prepare_workflows": boom})
    coordinator = ShutdownCoordinator(root, UiDispatcher(root), steps)
    coordinator.request()
    coordinator.join_release(1.0)
    assert "stop" in log


def test_blocking_release_polls_without_destroying_until_it_finishes():
    root, log = _Root(), []
    gate = threading.Event()
    coordinator = ShutdownCoordinator(root, UiDispatcher(root), _steps(log, release=lambda: gate.wait(2.0)))

    coordinator.request()
    assert coordinator.releasing
    assert root.destroy_calls == 0
    assert root.after_calls[0][0] == shutdown.POLL_MS

    gate.set()
    assert coordinator.join_release(1.0)
    root.after_calls.pop(0)[1]()
    assert root.destroy_calls == 1 and log[-1] == "flush"


def test_destroys_at_deadline_even_if_session_and_finalizer_stay_busy(monkeypatch):
    root, log = _Root(), []
    clock = [100.0]
    monkeypatch.setattr(shutdown.time, "monotonic", lambda: clock[0])
    waits: list[float] = []

    def session_idle(timeout):
        waits.append(timeout)
        return False

    coordinator = ShutdownCoordinator(
        root, UiDispatcher(root), _steps(log, session_idle=session_idle, finalizer_idle=lambda: False, extra=2.0),
    )
    coordinator.request()
    assert coordinator.deadline == 100.0 + shutdown.JOIN_TIMEOUT_S + 2.0
    coordinator.join_release(1.0)
    assert root.destroy_calls == 0

    clock[0] = coordinator.deadline + 0.1
    root.after_calls.pop(0)[1]()
    assert root.destroy_calls == 1 and log[-1] == "flush"
    assert waits[:2] == [shutdown.JOIN_SLICE_S, 0]
