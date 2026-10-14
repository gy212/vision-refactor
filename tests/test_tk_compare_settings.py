"""动作分析与设置窗口的控制器（假服务、同步投递）与两窗口的真实 Tk 冒烟。"""
from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.tk.controllers import compare as compare_mod
from apps.tk.controllers import settings as settings_mod
from apps.tk.controllers.compare import CompareController, compare_report, parse_frame, template_summary
from apps.tk.controllers.settings import SettingsController, active_model_keys, current_summary, model_state


class _View:
    """记录控制器对视图的每次调用。"""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, *args))


class _Queue:
    """替代 UiDispatcher.post：收集回调，测试里显式排空（模拟 Tk 线程）。"""

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


def _result(name: str, score: float, workers_used: int = 2):
    return SimpleNamespace(template_path=Path(f"{name}.npz"), score=score, avg_cost=0.1, cost=1.0,
                           start_frame=0, end_frame=9, fps=30.0, pose_variant="heavy", workers_used=workers_used)


def test_parse_frame_and_template_summary():
    assert parse_frame("  ") is None and parse_frame(" 12 ") == 12
    with pytest.raises(ValueError):
        parse_frame("x")
    text = template_summary("t.npz", {"start_frame": 30, "end_frame": 59, "fps": 30, "selected_retained_ratio": 0.5})
    assert text == "模板：t.npz\n区间：30..59（1.00s，保留 50.0%）"


def test_compare_report_averages_actions_and_notes_downgrade():
    actions, info, payload = compare_report([_result("a", 0.8), _result("b", 0.4, workers_used=1)], "v.mp4", 2)
    assert actions == [("a", 0.8), ("b", 0.4)]
    assert payload["combined_percent"] == 60 and payload["video_path"] == "v.mp4"
    assert info.startswith("共 2 个动作已评分") and "单线程" in info


def test_compare_runs_in_background_and_reports_through_post():
    view, queue = _View(), _Queue()
    gate = threading.Event()

    def fake_compare(templates, video, *, pose_variant, workers, progress_cb, stop_evt):
        assert pose_variant == "heavy" and workers == 3
        progress_cb("比对", 1, 2)
        gate.wait(2.0)
        return [_result("a", 0.9, workers_used=3)]

    controller = CompareController(view, queue.post, compare=fake_compare)
    assert controller.compare(["a.npz"], "v.mp4", 3)
    assert controller.busy and not controller.compare(["a.npz"], "v.mp4", 3)
    assert [c[0] for c in view.calls] == ["set_running", "set_status", "reset_scores", "set_raw", "show_progress"]
    gate.set()
    while controller.busy:
        threading.Event().wait(0.01)
    view.calls.clear()
    queue.drain()
    names = [c[0] for c in view.calls]
    assert names == ["show_progress", "show_scores", "set_raw", "set_status", "set_running"]
    assert view.calls[1][1:] == ([("a", 0.9)], "共 1 个动作已评分")
    assert view.calls[-1] == ("set_running", False)


def test_compare_failure_shows_error_and_generate_adds_template(monkeypatch):
    view, queue = _View(), _Queue()

    def boom(*_a, **_k):
        raise RuntimeError("坏视频")

    monkeypatch.setattr(compare_mod, "load_template", lambda _p, _k: SimpleNamespace(raw_meta={"start_frame": 0}))
    controller = CompareController(view, queue.post, compare=boom, create_template=lambda *_a, **_k: Path("new.npz"))
    controller.compare(["a.npz"], "v.mp4", 1)
    while controller.busy:
        threading.Event().wait(0.01)
    queue.drain()
    assert ("set_result", "坏视频") in view.calls and ("set_status", "分析失败") in view.calls

    view.calls.clear()
    controller.generate("base.mp4", None, None, 1)
    while controller.busy:
        threading.Event().wait(0.01)
    queue.drain()
    assert ("add_template", "new.npz") in view.calls and ("set_status", "模板生成完成") in view.calls


def test_settings_state_texts(monkeypatch):
    spec = SimpleNamespace(key="pose_full", label="Pose Full", approx_mb=9)
    hand = SimpleNamespace(key="hand", label="Hand", approx_mb=None)
    installed = {"pose_full"}
    monkeypatch.setattr(settings_mod, "is_installed", lambda s: s.key in installed)
    monkeypatch.setattr(settings_mod, "installed_size_mb", lambda s: 9.4)
    assert active_model_keys("FULL", True) == {"pose_full", "hand"}
    assert active_model_keys("bogus", False) == {"pose_full"}
    assert model_state(spec, True) == ("● 使用中  已安装 (9.4 MB)", "重新下载")
    assert model_state(hand, False) == ("未安装", "下载")
    summary = current_summary("full", True, {"pose_full", "hand"}, [spec, hand])
    assert summary.startswith("姿态模型：full　手部检测：开启\nPose Full（已就绪）\nHand（缺失）")
    assert summary.endswith("请在下方下载后再使用。")


def test_settings_download_reports_failures_and_reenables_buttons():
    view, queue = _View(), _Queue()
    specs = [SimpleNamespace(key="a", label="A"), SimpleNamespace(key="b", label="B")]

    def fake_download(spec, *, progress_cb, should_stop):
        progress_cb(5, 10)
        if spec.key == "b":
            raise OSError("网络不通")

    controller = SettingsController(view, queue.post, download=fake_download)
    assert controller.download(specs)
    assert view.calls == [("set_buttons_enabled", False)]
    while controller.busy:
        threading.Event().wait(0.01)
    queue.drain()
    names = [c[0] for c in view.calls]
    assert ("set_status", "完成 1 个，1 个失败。") in view.calls
    assert ("show_download_errors", "- B: 网络不通") in view.calls
    assert names[-2:] == ["refresh_states", "set_buttons_enabled"] and view.calls[-1] == ("set_buttons_enabled", True)


def test_dialogs_build_and_close_on_real_tk():
    from apps.tk import UiDispatcher
    from apps.tk.views import CompareDialog, SettingsDialog

    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - 无显示环境
        pytest.skip(f"Tk 不可用：{exc}")
    root.withdraw()
    try:
        dispatcher = UiDispatcher(root)
        compare = CompareDialog(root, dispatcher.post)
        compare.show_scores([("直拳", 0.85), ("鞭腿", 0.45)], "共 2 个动作已评分")
        assert compare.last_score == pytest.approx(0.65) and compare.result_var.get() == "共 2 个动作已评分"
        compare.reset_scores()
        assert compare.last_score is None
        settings = SettingsDialog(root, dispatcher.post, current_pose="lite", hands_enabled=False)
        assert "姿态模型：lite　手部检测：关闭" in settings.current_model_var.get()
        compare._on_close()
        settings._on_close()
        # 窗口关后投递的回调被丢弃，不碰已销毁的控件
        compare.controller._post(lambda: compare.set_status("late"))
        dispatcher.pump()
        assert not compare.is_open() and not settings.is_open()
    finally:
        root.destroy()
