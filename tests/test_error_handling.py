# -*- coding: utf-8 -*-
"""错误处理场景单元测试（任务 10.2，headless 友好）。

设计来源：``.kiro/specs/ui-layout-redesign/design.md`` 的 Error Handling 章节。

本文件以**不依赖真实 Tk 窗口**的方式验证两类错误处理路径：
  1. 无有效输入源点击「开始」：``_collect_state`` 抛 ``ValueError``，``_start`` 弹框、
     不启动会话（不调用 ``_rec.begin_session``）。（需求 4.2）
  2. 比对窗口创建失败：``_open_compare`` 捕获异常、弹「打开失败」，主窗口状态不变
     （``_compare_win`` 保持原值）。（需求 6.3）

设计意图：以 stub self 绑定调用未绑定的 ``App`` 方法（``App._method(stub, ...)``），
并用 monkeypatch 替换 ``apps.app_ui`` 的 ``messagebox`` /
``CompareWindow``，从而在无显示（headless / CI）环境中也能稳定运行，不会 skip。

_Requirements: 4.2, 6.3_
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import apps.app_ui as app_ui  # noqa: E402
from apps.app_ui import App  # noqa: E402
from apps.camera_enum import InputSourceState  # noqa: E402


# ---------------------------------------------------------------------------
# 测试替身（fakes）
# ---------------------------------------------------------------------------


class FakeVar:
    """假 Tk 变量，提供 set/get。"""

    def __init__(self, value: str = "") -> None:
        self._value = value

    def set(self, value) -> None:
        self._value = value

    def get(self):
        return self._value


class RecordingSpy:
    """录制控制器替身：记录 ``begin_session`` 是否被调用。"""

    def __init__(self) -> None:
        self.begin_session_calls: list = []

    def begin_session(self, *args, **kwargs) -> None:
        self.begin_session_calls.append((args, kwargs))


class ShowErrorRecorder:
    """记录 ``messagebox.showerror`` 调用参数的替身。"""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))

    @property
    def count(self) -> int:
        return len(self.calls)


# ---------------------------------------------------------------------------
# 1. 无有效输入源（需求 4.2）
# ---------------------------------------------------------------------------


def test_collect_state_raises_when_source_state_none():
    """_source_state.kind == "none" 时 _collect_state 抛带统一文案的 ValueError（需求 4.2）。"""
    state = InputSourceState()
    assert state.kind == "none"  # 默认即「无输入源」

    stub = types.SimpleNamespace(
        _source_state=state,
        source_var=FakeVar(""),
    )

    try:
        App._collect_state(stub)
    except ValueError as e:
        assert str(e) == "请先选择摄像头或视频"
    else:  # pragma: no cover - 不应到达
        raise AssertionError("expected ValueError for empty input source")


def test_start_with_no_source_shows_error_and_does_not_begin_session(monkeypatch):
    """点击开始且无有效输入源：弹框一次、不启动会话（begin_session 不被调用）（需求 4.2）。"""
    recorder = ShowErrorRecorder()
    monkeypatch.setattr(app_ui.messagebox, "showerror", recorder)

    rec_spy = RecordingSpy()

    def _raise_no_source(_self=None):
        raise ValueError("请先选择摄像头或视频")

    stub = types.SimpleNamespace(
        _worker=None,
        _collect_state=_raise_no_source,
        _rec=rec_spy,
    )

    App._start(stub)

    # 弹出一次错误提示。
    assert recorder.count == 1
    # 未启动录制会话。
    assert rec_spy.begin_session_calls == []


# ---------------------------------------------------------------------------
# 2. 比对窗口创建失败（需求 6.3）
# ---------------------------------------------------------------------------


def test_open_compare_failure_shows_error_and_keeps_state(monkeypatch):
    """比对窗口创建抛异常：弹「打开失败」一次、_compare_win 保持 None（需求 6.3）。"""

    def _raise_boom(_root):
        raise Exception("boom")

    monkeypatch.setattr(app_ui, "CompareWindow", _raise_boom)
    recorder = ShowErrorRecorder()
    monkeypatch.setattr(app_ui.messagebox, "showerror", recorder)

    stub = types.SimpleNamespace(
        _compare_win=None,
        root=object(),
    )

    App._open_compare(stub)

    # 弹出一次错误提示，且标题为「打开失败」。
    assert recorder.count == 1
    title = recorder.calls[0][0][0]
    assert title == "打开失败"
    # 主窗口状态不变：_compare_win 仍为 None。
    assert stub._compare_win is None
