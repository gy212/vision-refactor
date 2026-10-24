# -*- coding: utf-8 -*-
"""录制性能测试的桌面入口：控制器的开始条件、后台运行与报告保存（假测试、同步投递）、窗口的真实 Tk 冒烟（总结论醒目显示），
以及"按当前设置录像"一段的处理循环负载（``BenchmarkSessionLoad``：在已接管的假摄像头上跑正式的会话循环、预览画到主窗口）。"""
from __future__ import annotations

import json
import threading
import time
import tkinter as tk
from types import SimpleNamespace

import pytest

from apps.tk.controllers import benchmark as bench_mod
from apps.tk.controllers.benchmark import BenchmarkContext, BenchmarkController
from apps.tk.controllers.benchmark_load import BenchmarkSessionLoad
from core.foundation import CancelToken
from core.live import camera_session
from core.media import PRIMARY, SECONDARY, CameraPool, CameraService, CaptureSpec
from tests.test_recording_benchmark import FakeEngine, TimedCapture
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


class _Preview:
    """主窗口 ``PreviewSurface`` 的替身：记下显示的帧与单/双路布局的切换。"""

    def __init__(self) -> None:
        self.dual_visible = False
        self.layouts: list[bool] = []
        self.shown: list[dict] = []

    def size_of(self, _role):
        return (320, 180)

    def set_dual_visible(self, visible: bool) -> None:
        self.dual_visible = bool(visible)
        self.layouts.append(self.dual_visible)

    def show(self, images, sizes) -> None:
        self.shown.append({role: image.shape for role, image in images.items()})


def _ctx(tmp_path, queue, *, cameras=None, active=False, mode="normal") -> BenchmarkContext:
    specs = {PRIMARY: CaptureSpec(0, 1920, 1080), SECONDARY: CaptureSpec(1, 1920, 1080)}
    return BenchmarkContext(
        cameras=cameras or _Cameras(), session_active=lambda: active, mode_name=lambda: mode,
        rotation_of=lambda role: 90 if role == PRIMARY else 0, pose_variant=lambda: "full",
        enable_hands=lambda: True, record_dir=lambda: tmp_path / "录制", post=queue.post,
        session_config=lambda: camera_session(specs, skeleton=True), preview=_Preview(),
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
    assert isinstance(setup.session_load, BenchmarkSessionLoad)     # 最后一段跑点「开始」时的会话配置
    _name, conclusion, summary, path, error = view.named("show_result")[0]
    assert conclusion == {} and error is None and summary == ["结论：可以"]
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
        return {"ok": False, "error": "已取消", "summary": ["测试中断：已取消"],
                "conclusion": {"level": "incomplete", "ok": False, "text": "测试没有做完"}}

    monkeypatch.setattr(bench_mod, "desktop_dir", lambda: tmp_path)
    monkeypatch.setattr(bench_mod, "run_recording_benchmark", fake_run)
    queue, view = _Queue(), _View()
    controller = BenchmarkController(_ctx(tmp_path, queue), view)
    controller.start()
    controller.cancel()
    _wait(controller, queue)
    _name, conclusion, summary, path, _error = view.named("show_result")[0]
    assert not conclusion["ok"] and summary == ["测试中断：已取消"] and path is not None   # 中断时也保存已测到的部分


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
        dialog.show_result({"level": "fail", "ok": False, "text": "按当前设置不能稳定录像：正面 写盘跟不上。"},
                           ["结论：按当前设置不能稳定录像：正面 写盘跟不上。", "", "各段明细"], report, None)
        assert dialog.conclusion_var.get().startswith("按当前设置不能稳定录像")       # 总结论单独醒目显示
        assert str(dialog.conclusion_label.cget("foreground")) == "#c62828"
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


def test_session_load_runs_the_production_loop_on_the_claimed_cameras(monkeypatch):
    """双摄骨架关：正式的会话循环在测试接管的摄像头上运行（站位检测开着、在线识别关着），预览按旋转画到主窗口；
    循环结束不关闭摄像头（测试还要用），主窗口恢复原来的单路布局。"""
    import core.live.presence as presence_mod

    monkeypatch.setattr(presence_mod, "create_engine", lambda spec, models_dir: FakeEngine(spec))
    service = CameraService(pool=CameraPool(TimedCapture, join_timeout=0.5))
    stream = service.open_stream({PRIMARY: CaptureSpec(0), SECONDARY: CaptureSpec(1)})
    try:
        stream.wait_ready(CancelToken())
        stream.claim(CancelToken())
        preview = _Preview()
        load = BenchmarkSessionLoad(camera_session(stream.specs, skeleton=False, online_match=True),
                                    rotation_of=lambda role: 90 if role == PRIMARY else 0, preview=preview)
        token, started, result = CancelToken(), threading.Event(), {}
        runner = threading.Thread(target=lambda: result.update(load(stream, token, started.set)))
        runner.start()
        assert started.wait(5)
        deadline = time.monotonic() + 5
        while len(preview.shown) < 5 and time.monotonic() < deadline:
            load.render()
            time.sleep(0.02)
        time.sleep(0.4)
        token.cancel("phase_end")
        runner.join(10)
        load.render()
        assert result["error"] is None and result["frames"] > 0 and result["fps"] > 0
        assert result["settings"]["presence"] is True and result["settings"]["online_match"] is False   # D-7
        assert result["presence_samples"] > 0 and result["rendered"] >= 5
        assert preview.shown[0][PRIMARY][0] > preview.shown[0][PRIMARY][1]          # 主路转正 90°：竖画面
        assert preview.shown[0][SECONDARY][1] > preview.shown[0][SECONDARY][0]
        assert preview.layouts == [True, False]
        assert stream.claimed and stream.latest(PRIMARY) is not None                # 摄像头仍归测试
    finally:
        stream.close()
        service.close(2.0)


def test_missing_models_lists_lite_and_the_skeleton_models_actually_used():
    installed = {"pose_full", "hand"}
    missing = bench_mod.missing_models("full", True, installed=installed.__contains__)
    assert list(missing) == ["lite"] and missing["lite"].startswith("缺少 Lite 模型（pose_landmarker_lite.task）")
    missing = bench_mod.missing_models("heavy", True, installed=set().__contains__)
    assert "pose_landmarker_heavy.task、hand_landmarker.task" in missing["skeleton"]
    assert "hand_landmarker" not in bench_mod.missing_models("heavy", False, installed=set().__contains__)["skeleton"]
    assert bench_mod.missing_models("full", True, installed=lambda _key: True) == {}


def _run_load(load, stream, seconds=0.4):
    token, started, result = CancelToken(), threading.Event(), {}
    runner = threading.Thread(target=lambda: (result.update(load(stream, token, started.set)), started.set()))
    runner.start()
    started.wait(5)
    time.sleep(seconds)
    token.cancel("phase_end")
    runner.join(10)
    return result


def test_session_load_without_the_lite_model_does_not_arm_presence(monkeypatch):
    """缺 Lite 模型时测试不下载：站位检测不开（一次都不建引擎），结果写明跳过；缺骨架模型而要识别骨架时不运行循环。"""
    import core.live.presence as presence_mod

    created = []
    monkeypatch.setattr(presence_mod, "create_engine", lambda spec, models_dir: created.append(spec) or FakeEngine(spec))
    service = CameraService(pool=CameraPool(TimedCapture, join_timeout=0.5))
    stream = service.open_stream({PRIMARY: CaptureSpec(0), SECONDARY: CaptureSpec(1)})
    try:
        stream.wait_ready(CancelToken())
        stream.claim(CancelToken())
        load = BenchmarkSessionLoad(camera_session(stream.specs, skeleton=False), rotation_of=lambda _role: 0,
                                    preview=_Preview(), missing={"lite": "缺少 Lite 模型"})
        result = _run_load(load, stream)
        assert result["error"] is None and result["frames"] > 0
        assert result["presence_skipped"] is True and result["presence_samples"] == 0 and created == []
        refused = BenchmarkSessionLoad(camera_session(stream.specs, skeleton=True), rotation_of=lambda _role: 0,
                                       preview=_Preview(), missing={"skeleton": "缺少骨架识别模型"})
        result = _run_load(refused, stream, seconds=0.0)
        assert result["error"] == "缺少骨架识别模型，没有运行处理循环" and result["frames"] == 0
    finally:
        stream.close()
        service.close(2.0)
