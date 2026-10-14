"""UiDispatcher：线程安全投递、按 token 丢弃过期回调、关闭态与销毁后的拒收、逐帧工作与周期调度。"""
from __future__ import annotations

import threading

from apps.tk import UiDispatcher, epoch_token
from core.foundation import Epoch


class _Root:
    def __init__(self) -> None:
        self.after_calls: list[tuple[int, object]] = []

    def after(self, delay, callback) -> None:
        self.after_calls.append((delay, callback))


def test_posts_from_threads_run_in_order_on_pump():
    dispatcher = UiDispatcher(_Root())
    seen: list[int] = []
    threads = [threading.Thread(target=lambda i=i: dispatcher.post(lambda: seen.append(i))) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == []
    dispatcher.pump()
    assert sorted(seen) == list(range(20))
    dispatcher.pump()
    assert len(seen) == 20


def test_stale_epoch_token_drops_callback():
    dispatcher = UiDispatcher(_Root())
    epoch = Epoch()
    seen: list[str] = []
    dispatcher.post(lambda: seen.append("old"), epoch_token(epoch))
    epoch.advance()
    dispatcher.post(lambda: seen.append("new"), epoch_token(epoch))
    dispatcher.pump()
    assert seen == ["new"]


def test_closing_keeps_only_shutdown_callbacks_and_destroy_rejects_all():
    dispatcher = UiDispatcher(_Root())
    seen: list[str] = []
    dispatcher.post(lambda: seen.append("queued-ui"))
    dispatcher.post(lambda: seen.append("queued-close"), during_close=True)
    dispatcher.close()
    assert dispatcher.post(lambda: seen.append("late-ui")) is False
    assert dispatcher.post(lambda: seen.append("late-close"), during_close=True) is True
    dispatcher.pump()
    assert seen == ["queued-close", "late-close"]

    dispatcher.destroy()
    assert dispatcher.post(lambda: seen.append("x"), during_close=True) is False
    dispatcher.pump()
    assert seen == ["queued-close", "late-close"]


def test_failing_callback_does_not_stop_others_and_ticks_run_after_callbacks(capsys):
    dispatcher = UiDispatcher(_Root())
    order: list[str] = []
    dispatcher.every_tick(lambda: order.append("tick"))
    dispatcher.post(lambda: 1 / 0)
    dispatcher.post(lambda: order.append("cb"))
    dispatcher.pump()
    assert order == ["cb", "tick"]
    assert "ZeroDivisionError" in capsys.readouterr().err


def test_start_schedules_every_interval_until_destroyed():
    root = _Root()
    dispatcher = UiDispatcher(root)
    ticks: list[int] = []
    dispatcher.every_tick(lambda: ticks.append(1))
    dispatcher.start()
    dispatcher.start()
    assert [d for d, _ in root.after_calls] == [16]
    _delay, loop = root.after_calls.pop(0)
    loop()
    assert ticks == [1] and [d for d, _ in root.after_calls] == [16]
    dispatcher.destroy()
    _delay, loop = root.after_calls.pop(0)
    loop()
    assert ticks == [1] and root.after_calls == []
