# -*- coding: utf-8 -*-
"""core.foundation.tasks / epoch：CancelToken 级联与 wait、TaskGroup 的 pending 与 close_deadline、Epoch 过期判断。"""
from __future__ import annotations

import gc
import threading
import time
import weakref

import pytest

from core.foundation.epoch import Epoch
from core.foundation.tasks import (
    CancelToken,
    Cancelled,
    TaskGroup,
    as_stopped,
    translate_interrupted,
)


def test_cancel_is_idempotent_and_first_reason_wins():
    token = CancelToken()
    assert not token.cancelled
    token.raise_if_cancelled()
    token.cancel("user_stop")
    token.cancel("app_close")
    assert token.cancelled and token.reason == "user_stop"
    with pytest.raises(Cancelled) as info:
        token.raise_if_cancelled()
    assert info.value.reason == "user_stop"


def test_child_cascades_down_not_up():
    parent = CancelToken()
    child = parent.child()
    grandchild = child.child()
    sibling = parent.child()

    child.cancel("superseded")
    assert child.cancelled and grandchild.cancelled
    assert not parent.cancelled and not sibling.cancelled

    parent.cancel("app_close")
    assert sibling.cancelled and sibling.reason == "app_close"
    assert grandchild.reason == "superseded"  # 已取消的不被覆盖


def test_child_of_cancelled_parent_is_cancelled_immediately():
    parent = CancelToken()
    parent.cancel("mode_switch")
    child = parent.child()
    assert child.cancelled and child.reason == "mode_switch"


def test_wait_times_out_then_wakes_on_cancel():
    token = CancelToken()
    started = time.monotonic()
    assert token.wait(0.05) is False
    assert time.monotonic() - started >= 0.04

    threading.Timer(0.05, token.cancel, args=("timeout",)).start()
    assert token.wait(5.0) is True


def test_task_group_runs_and_reports_pending_by_owner():
    group = TaskGroup("t")
    release = threading.Event()

    def blocked(token: CancelToken) -> str:
        release.wait(5.0)
        return "ok"

    a = group.spawn("clip", blocked, owner="exam", close_budget_s=30.0)
    b = group.spawn("compare", blocked, owner="postprocess", close_budget_s=12.0)
    assert group.pending() == [a, b]
    assert group.pending("exam") == [a]

    release.set()
    assert a.join(5.0) and b.join(5.0)
    assert a.result == "ok" and group.pending() == []


def test_close_deadline_uses_largest_live_budget():
    group = TaskGroup()
    assert group.close_deadline() == pytest.approx(time.monotonic(), abs=0.05)

    hold = CancelToken()
    group.spawn("roster", lambda t: hold.wait(5.0), owner="exam", close_budget_s=8.0)
    group.spawn("clip", lambda t: hold.wait(5.0), owner="exam", close_budget_s=30.0)
    now = time.monotonic()
    assert group.close_deadline() == pytest.approx(now + 30.0, abs=0.05)
    hold.cancel()
    assert group.join_until(time.monotonic() + 5.0) == []
    assert group.close_deadline() == pytest.approx(time.monotonic(), abs=0.05)


def test_cancel_by_owner_and_whole_group():
    group = TaskGroup()

    def loop(token: CancelToken) -> None:
        while True:
            token.raise_if_cancelled()
            token.wait(0.01)

    exam = group.spawn("a", loop, owner="exam")
    live = group.spawn("b", loop, owner="live")

    group.cancel("exam", reason="user_stop")
    assert exam.join(5.0) and exam.was_cancelled and exam.token.reason == "user_stop"
    assert not live.done

    group.cancel(reason="app_close")
    assert live.join(5.0) and live.was_cancelled
    late = group.spawn("c", loop, owner="live")
    assert late.join(5.0) and late.token.reason == "app_close"


def test_task_error_is_recorded(monkeypatch):
    hooked: list[BaseException] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: hooked.append(args.exc_value))
    group = TaskGroup()

    def boom(token: CancelToken) -> None:
        raise ValueError("坏了")

    handle = group.spawn("x", boom, owner="o")
    assert handle.join(5.0)
    assert isinstance(handle.error, ValueError) and not handle.was_cancelled
    deadline = time.monotonic() + 5.0
    while not hooked and time.monotonic() < deadline:
        time.sleep(0.01)
    assert isinstance(hooked[0], ValueError)


def test_pending_never_drops_a_task_that_has_not_started_running():
    group = TaskGroup()
    gate = threading.Event()
    handles = [group.spawn(str(i), lambda t: gate.wait(5.0), owner="o") for i in range(20)]
    assert group.pending() == handles
    gate.set()
    assert group.join_until(time.monotonic() + 5.0) == []


def test_legacy_adapters():
    token = CancelToken()
    stopped = as_stopped(token)
    assert stopped() is False
    token.cancel("user_stop")
    assert stopped() is True

    with pytest.raises(Cancelled) as info:
        with translate_interrupted(token):
            raise InterruptedError("分析已取消")
    assert info.value.reason == "user_stop"
    assert isinstance(info.value.__cause__, InterruptedError)

    with pytest.raises(Cancelled) as info:
        with translate_interrupted(reason="timeout"):
            raise InterruptedError
    assert info.value.reason == "timeout"


def test_epoch_marks_older_tokens_stale():
    epoch = Epoch()
    assert epoch.current == 0 and epoch.is_current(0)
    first = epoch.advance()
    second = epoch.advance()
    assert (first, second) == (1, 2)
    assert not epoch.is_current(first)
    assert epoch.is_current(second) and epoch.current == second


def test_epoch_advance_is_thread_safe():
    epoch = Epoch()
    seen: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(500):
            token = epoch.advance()
            with lock:
                seen.append(token)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(seen) == list(range(1, 4001))
    assert epoch.current == 4000


def _collected(refs: list[weakref.ref], timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        gc.collect()
        if all(r() is None for r in refs):
            return True
        time.sleep(0.01)
    return False


def test_finished_tasks_do_not_accumulate_in_long_lived_group_token():
    group = TaskGroup("leak")
    handles = [group.spawn(str(i), lambda t: None, owner="o") for i in range(50)]
    refs = [weakref.ref(h.token) for h in handles]
    for h in handles:
        assert h.join(5.0)
    assert group.pending() == []
    del handles, h
    assert _collected(refs)
    group.cancel(reason="app_close")  # 组 token 仍可正常取消


def test_grandchild_still_cascades_when_intermediate_token_is_dropped():
    root = CancelToken()
    grandchild = root.child().child()  # 中间层没有外部引用
    gc.collect()
    root.cancel("mode_switch")
    assert grandchild.cancelled and grandchild.reason == "mode_switch"
