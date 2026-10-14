# -*- coding: utf-8 -*-
"""错误处理：比对与设置窗口创建失败时弹「打开失败」、主窗口状态不变（需求 6.3）。

无有效输入源点「开始」的弹框与不启动会话（需求 4.2）见 test_tk_session.py 的
``test_collect_state_requires_input_source`` 与 ``test_start_with_bad_config_shows_error_and_does_not_start``。

_Requirements: 4.2, 6.3_
"""
from __future__ import annotations

import pytest

from apps.tk import app as tk_app
from tests.tk_app_doubles import build_app, root  # noqa: F401


@pytest.mark.parametrize(("name", "factory", "opener"), [
    ("compare", "CompareDialog", "open_compare"),
    ("settings", "SettingsDialog", "open_settings"),
])
def test_dialog_failure_shows_error_and_keeps_state(root, monkeypatch, name, factory, opener):
    def boom(*_args, **_kwargs):
        raise Exception("boom")

    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(tk_app, factory, boom)
    monkeypatch.setattr(tk_app.messagebox, "showerror", lambda title, text: errors.append((title, text)))
    app = build_app(root)

    getattr(app, opener)()

    assert len(errors) == 1 and errors[0][0] == "打开失败"
    assert errors[0][1].endswith("创建失败：boom")
    assert name not in app._dialogs   # 主窗口状态不变：不登记失败的窗口


def test_open_dialog_twice_focuses_existing(root, monkeypatch):
    created: list[object] = []

    class _Dialog:
        def __init__(self, *_args, **_kwargs):
            self.focused = 0
            created.append(self)

        def is_open(self):
            return True

        def focus(self):
            self.focused += 1

    monkeypatch.setattr(tk_app, "CompareDialog", _Dialog)
    app = build_app(root)
    app.open_compare()
    app.open_compare()
    assert len(created) == 1 and created[0].focused == 1
