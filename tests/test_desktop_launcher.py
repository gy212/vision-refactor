# -*- coding: utf-8 -*-
"""apps.desktop_launcher：建立日志之前的导入失败也要弹出启动失败提示，而不是绕过报错路径。"""
from __future__ import annotations

import sys
import tkinter.messagebox

import pytest

from apps import desktop_launcher


def test_import_failure_before_log_redirect_reaches_error_dialog(monkeypatch):
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(sys, "argv", ["vision_ui.exe"])
    monkeypatch.setitem(sys.modules, "core.foundation", None)  # 让 `from core.foundation import ...` 抛 ImportError
    monkeypatch.setattr(tkinter.messagebox, "showerror", lambda title, msg: shown.append((title, msg)))

    with pytest.raises(ImportError):
        desktop_launcher.main()

    assert len(shown) == 1
    assert shown[0][0] == "启动失败" and "日志尚未建立" in shown[0][1]
