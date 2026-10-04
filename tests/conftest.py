# -*- coding: utf-8 -*-
"""全局测试隔离。

历史事故：部分测试（如 test_second_dual_segment_...）走真实 App 录制路径，
其中 ``_begin_recording_segment`` 会调 ``save_record_dir`` 把 pytest 临时目录
写进仓库根真实 ``user_prefs.json``，导致用户下次启动录制目录回退到不存在的
临时地址。这里 autouse 把 ``core.paths._prefs_path`` 钉到临时文件，任何测试都
无法再污染真实偏好文件——不必逐个测试补 monkeypatch。
"""
from __future__ import annotations

import pytest

from core import paths


@pytest.fixture(autouse=True)
def _isolate_user_prefs(tmp_path_factory, monkeypatch):
    prefs = tmp_path_factory.mktemp("user_prefs") / "user_prefs.json"
    monkeypatch.setattr(paths, "_prefs_path", lambda: prefs)


@pytest.fixture(scope="session", autouse=True)
def _bind_tcl_std_channels(request):
    """在 pytest 的 fd 捕获之外先建立 Tcl 标准通道，并保持到会话结束。

    Windows 上同一进程反复 ``tk.Tk()`` 时，pytest 默认的 fd 捕获会在用例之间替换并关闭
    stdin/stdout/stderr 句柄，新建解释器偶发 "Can't find a usable init.tcl …: No error"
    （同一组用例在 ``-s`` / ``--capture=sys`` 下不出现）。先用真实句柄建立标准通道即可避免。
    """
    try:
        import tkinter
    except ImportError:  # pragma: no cover - 无 tkinter 的环境
        yield None
        return
    capman = request.config.pluginmanager.getplugin("capturemanager")
    try:
        if capman is None:
            interp = tkinter.Tcl()
            interp.eval("fconfigure stdin; fconfigure stdout; fconfigure stderr")
        else:
            with capman.global_and_fixture_disabled():
                interp = tkinter.Tcl()
                interp.eval("fconfigure stdin; fconfigure stdout; fconfigure stderr")
    except tkinter.TclError:  # pragma: no cover - Tcl 不可用时不影响其余用例
        interp = None
    yield interp
