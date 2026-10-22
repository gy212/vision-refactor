# -*- coding: utf-8 -*-
"""录制性能测试的桌面入口：控制器的开始条件、后台运行与报告保存（假测试、同步投递），以及窗口的真实 Tk 冒烟。"""
from __future__ import annotations

import json
import threading
import time
import tkinter as tk
from types import SimpleNamespace

import pytest

from apps.tk.controllers import benchmark as bench_mod
from apps.tk.controllers.benchmark import BenchmarkContext, BenchmarkController
from core.media import PRIMARY, SECONDARY, CaptureSpec
from tests.tk_app_doubles import build_app, root  # noqa: F401


class _View:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, *args))

    def named(self, name: str) -> list[tuple]:
        return [call for call in self.calls if call[0] == name]


class _Queue:
    def __init__(self) -> None:
        self.items: list = []
        self.lock = threading.Lock()

    def post(self, cb) -> None:
        with self.lock:
            self.items.append(cb)

    def drain(self) -> None:
        with self.lock:
            items, self.items = self.items, []
        for cb in items:
            cb()


class _Cameras:
    def __init__(self, primary=0, secondary=1) -> None:
        self.indices = (primary, secondary)
        self.opened: list[dict] = []
        self.service = SimpleNamespace(open_stream=lambda specs: self.opened.append(specs) or "stream")

    def selected_indices(self):
        return self.indices

    def spec(self, index):
        return CaptureSpec(index, 1920, 1080)

    def label_for(self, index):
        return {0: "正面相机", 1: "侧面相机"}.get(index)


def _ctx(tmp_path, queue, *, cameras=None, active=False, mode="normal") -> BenchmarkContext:
    return BenchmarkContext(
        cameras=cameras or _Cameras(), session_active=lambda: active, mode_name=lambda: mode,
        rotation_of=lambda role: 90 if role == PRIMARY else 0, pose_variant=lambda: "full",
        enable_hands=lambda: True, record_dir=lambda: tmp_path / "录制", post=queue.post,
    )


def _wait(controller: BenchmarkController, queue: _Queue) -> None:
    deadline = time.monotonic() + 5
    while controller.running and time.monotonic() < deadline:
        for handle in controller._tasks.pending():
            handle.join(0.05)
        queue.drain()
    assert not controller.running


@pytest.mark.parametrize("kwargs, text", [
    ({"active": True}, "停止"),
    ({"mode": "exam"}, "普通模式"),
    ({"cameras": _Cameras(None, None)}, "选择摄像头"),
])
def test_problem_blocks_start_and_tells_why(tmp_path, kwargs, text):
    queue, view = _Queue(), _View()
    controller = BenchmarkController(_ctx(tmp_path, queue, **kwargs), view)
    controller.start()
    assert not controller.running
    assert text in view.named("show_problem")[0][1]


def test_low_disk_space_blocks_start(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_mod.shutil, "disk_usage", lambda _p: SimpleNamespace(free=1 * 2**30))
    controller = BenchmarkController(_ctx(tmp_path, _Queue()), _View())
    assert "1.0 GB" in controller.problem()


def test_run_saves_report_to_desktop_and_shows_full_path(tmp_path, monkeypatch):
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    seen: dict = {}

    def fake_run(setup, cancel, progress):
        seen["setup"] = setup
        progress("第 1/4 段：只采集", 0.5)
        return {"kind": "recording_benchmark", "ok": True, "summary": ["结论：可以"]}

    monkeypatch.setattr(bench_mod, "desktop_dir", lambda: desktop)
    monkeypatch.setattr(bench_mod, "run_recording_benchmark", fake_run)
    queue, view, cameras = _Queue(), _View(), _Cameras()
    controller = BenchmarkController(_ctx(tmp_path, queue, cameras=cameras), view)
    controller.start()
    assert controller.running and view.named("set_running") == [("set_running", True)]
    _wait(controller, queue)

    setup = seen["setup"]
    assert set(cameras.opened[0]) == {PRIMARY, SECONDARY} and setup.stream == "stream"
    assert setup.rotations == {PRIMARY: 90, SECONDARY: 0}
    assert setup.labels == {PRIMARY: "正面相机", SECONDARY: "侧面相机"}
    assert setup.work_dir.parent == tmp_path / "录制" and setup.pose_variant == "full"
    assert ("show_progress", "第 1/4 段：只采集", 0.5) in view.calls
    _name, summary, ok, path, error = view.named("show_result")[0]
    assert ok and error is None and summary == ["结论：可以"]
    assert path.parent == desktop and path.name.startswith("录制性能测试_") and path.suffix == ".json"
    assert json.loads(path.read_text(encoding="utf-8"))["summary"] == ["结论：可以"]
    assert controller.last_report_path == path


def test_report_falls_back_to_data_folder_when_desktop_is_not_writable(tmp_path, monkeypatch):
    blocked = tmp_path / "not_a_dir"
    blocked.write_text("x", encoding="utf-8")                    # 文件挡住了目录，mkdir 失败
    monkeypatch.setattr(bench_mod, "desktop_dir", lambda: blocked)
    monkeypatch.setattr(bench_mod, "repo_root", lambda: tmp_path / "data")
    path, error = BenchmarkController(_ctx(tmp_path, _Queue()), _View())._save({"summary": []})
    assert error is None and path.parent == tmp_path / "data" / "录制性能测试"


def test_cancel_reaches_the_running_benchmark(tmp_path, monkeypatch):
    def fake_run(setup, cancel, progress):
        assert cancel.wait(5)
        return {"ok": False, "error": "已取消", "summary": ["测试中断：已取消"]}

    monkeypatch.setattr(bench_mod, "desktop_dir", lambda: tmp_path)
    monkeypatch.setattr(bench_mod, "run_recording_benchmark", fake_run)
    queue, view = _Queue(), _View()
    controller = BenchmarkController(_ctx(tmp_path, queue), view)
    controller.start()
    controller.cancel()
    _wait(controller, queue)
    _name, summary, ok, path, _error = view.named("show_result")[0]
    assert not ok and summary == ["测试中断：已取消"] and path is not None   # 中断时也保存已测到的部分


def test_dialog_smoke_shows_save_location_and_result_path(tmp_path, monkeypatch):
    from apps.tk.dispatcher import UiDispatcher
    from apps.tk.views import benchmark_dialog as dialog_mod

    monkeypatch.setattr(bench_mod, "desktop_dir", lambda: tmp_path)
    monkeypatch.setattr(dialog_mod, "report_directory", lambda: tmp_path)
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - 无显示环境
        pytest.skip(f"Tk 不可用：{exc}")
    root.withdraw()
    try:
        dispatcher = UiDispatcher(root)
        dialog = dialog_mod.BenchmarkDialog(root, dispatcher.post, _ctx(tmp_path, _Queue()))
        assert dialog.is_open()
        report = tmp_path / "录制性能测试_20261021_093005.json"
        dialog.show_result(["结论：可以"], True, report, None)
        assert dialog.path_var.get() == str(report)
        assert str(dialog.open_btn.cget("state")) == "normal" and dialog.start_btn.cget("text") == "再测一次"
        assert str(report) in dialog.summary.get("1.0", "end")
        dialog.copy_path()
        assert root.clipboard_get() == str(report)
        dialog.set_running(True)
        assert str(dialog.start_btn.cget("state")) == "disabled" and dialog.path_var.get() == ""
        dialog.set_running(False)
        dialog.close()
        assert not dialog.is_open()
    finally:
        root.destroy()


def test_desktop_app_opens_dialog_from_hidden_shortcut_and_closes_it_on_exit(root):   # noqa: F811
    app = build_app(root)
    assert root.bind_all("<Control-Shift-KeyPress-B>") and root.bind_all("<Control-Shift-KeyPress-b>")
    app.open_benchmark()
    dialog = app._dialogs["benchmark"]
    assert dialog.is_open()
    app.open_benchmark()                                  # 再按一次只把已开的窗口提到前面
    assert app._dialogs["benchmark"] is dialog
    app._close_workflows()                                # 主程序退出的第 1 步顺带关掉测试窗口
    assert not dialog.is_open()
