# -*- coding: utf-8 -*-
"""全局测试隔离。

历史事故：部分测试（如 test_second_dual_segment_...）走真实 App 录制路径，
其中 ``_begin_recording_segment`` 会调 ``save_record_dir`` 把 pytest 临时目录
写进仓库根真实 ``user_prefs.json``，导致用户下次启动录制目录回退到不存在的
临时地址。这里 autouse 把 ``core.foundation.prefs._prefs_path`` 钉到临时文件，任何测试都
无法再污染真实偏好文件——不必逐个测试补 monkeypatch。
"""
from __future__ import annotations

import sys

import cv2
import pytest

from core.foundation import prefs as user_prefs
from core.pose import engine as pose_engine_mod

# 问题 3/8 的回归守护用：会话加载时（任何测试、任何 monkeypatch 之前）记下真值，
# 既用来比对是否被污染，也用来在发现污染时直接恢复，不必依赖"当时还没被二次污染"的猜测。
_REAL_CREATE_ENGINE = pose_engine_mod.create_engine
_REAL_VIDEO_CAPTURE = cv2.VideoCapture


@pytest.fixture(autouse=True)
def _isolate_user_prefs(tmp_path_factory, monkeypatch):
    prefs = tmp_path_factory.mktemp("user_prefs") / "user_prefs.json"
    monkeypatch.setattr(user_prefs, "_prefs_path", lambda: prefs)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """回归守护（问题 3）：某些测试在 ``with H.replay_context():`` 内部用 pytest 的 ``monkeypatch``
    patch ``core.pose.extract.create_engine`` 或 ``cv2.VideoCapture``。``monkeypatch`` 的撤销发生在
    函数级 fixture 全部拆完之后，如果那时 ``replay_context`` 已经退出并恢复了真值，``monkeypatch``
    的撤销会把假值重新写回去，污染同一会话里之后的测试。

    用 ``hookwrapper`` 在 ``yield`` 之后（函数级 fixture 已全部拆完，包括 ``monkeypatch`` 的撤销）
    检查两件事是否已还原，而不是用普通 autouse fixture：普通 fixture 可能在 ``monkeypatch`` 撤销
    之前就拆掉，会误报"已还原"。

    问题 8：发现污染时先用会话加载时记下的真值强制复原，再抛错——否则污染会一直留到下一次被
    这个守护捡到之前，期间所有测试都会连带报错，无法定位真正出问题的那一个。
    """
    outcome = yield
    outcome.get_result()
    violations: list[str] = []
    extract_mod = sys.modules.get("core.pose.extract")
    if extract_mod is not None and extract_mod.create_engine is not _REAL_CREATE_ENGINE:
        extract_mod.create_engine = _REAL_CREATE_ENGINE
        violations.append(
            "core.pose.extract.create_engine 未还原为真函数"
            "（monkeypatch 在 replay_context 退出后才撤销，把假引擎写回去了，已强制复原）"
        )
    if cv2.VideoCapture is not _REAL_VIDEO_CAPTURE:
        cv2.VideoCapture = _REAL_VIDEO_CAPTURE
        violations.append(
            "cv2.VideoCapture 未还原为真实现"
            "（monkeypatch 在 replay_context 退出后才撤销，把假 VideoCapture 写回去了，已强制复原）"
        )
    if violations:
        raise RuntimeError(f"{item.nodeid} 之后发现并已复原：" + "；".join(violations))


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
