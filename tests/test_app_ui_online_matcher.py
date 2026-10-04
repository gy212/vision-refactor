# -*- coding: utf-8 -*-
"""app_ui 在线 DTW 动作识别接入层测试（无需 Tk root）。

覆盖接入的关键风险，不重测 matcher 内核（见 test_online_matcher.py）：
- 无 templates/online 时不崩预览（_build 返回 None）
- _feed 正确 normalize+push、None 早返回不抛
- _post_match 关窗竞态三层守卫（stop 提前 return / TclError 兜底 / action=None 不刷）
- 两条预览循环都接了 matcher（AST 守卫，防只接一条回归）
"""
from __future__ import annotations

import ast
import sys
import threading
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apps import app_ui  # noqa: E402
from core.online_matcher import MatchResult  # noqa: E402


class _SettableVar:
    """假 StringVar：带 get/set。"""

    def __init__(self, value: object = "") -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class _FakeRoot:
    """假 Tk root：记录 after 调用并同步执行 callback（模拟主线程立即跑）。"""

    def __init__(self) -> None:
        self.after_calls: list[int] = []

    def after(self, delay: int, callback=None):
        self.after_calls.append(delay)
        if callback is not None:
            callback()


class _FakeMatcher:
    def __init__(self) -> None:
        self.pushed: list[tuple] = []

    def push(self, feat, ts, idx) -> None:
        self.pushed.append((feat, ts, idx))


# --- _build_online_matcher -------------------------------------------------

def test_build_online_matcher_no_templates_returns_none(monkeypatch, tmp_path):
    """无 templates/online 目录 → None（预览照常跑，不崩）。"""
    import core.paths as paths_mod

    monkeypatch.setattr(paths_mod, "templates_dir", lambda: tmp_path)  # 下面无 online 子目录
    app = object.__new__(app_ui.App)
    assert app_ui.App._build_online_matcher(app) is None


def test_build_online_matcher_swallows_load_error(monkeypatch, tmp_path):
    """加载/断言异常被吞 → None，不外泄到预览循环。"""
    import core.paths as paths_mod
    import core.online_matcher as om

    monkeypatch.setattr(paths_mod, "templates_dir", lambda: tmp_path)

    def _boom(_dir):
        raise ValueError("模板布局不符")

    monkeypatch.setattr(om, "load_template_library", _boom)
    app = object.__new__(app_ui.App)
    assert app_ui.App._build_online_matcher(app) is None


# --- _feed_online_matcher --------------------------------------------------

def test_feed_online_matcher_normalizes_and_pushes(monkeypatch):
    import core.pose_features as pf

    sentinel = object()
    monkeypatch.setattr(pf, "normalize_pose_xy_v3", lambda lm: sentinel)
    matcher = _FakeMatcher()

    app_ui.App._feed_online_matcher(matcher, object(), 123.0, 7)

    assert matcher.pushed == [(sentinel, 123.0, 7)]


def test_feed_online_matcher_noops_on_none(monkeypatch):
    import core.pose_features as pf

    monkeypatch.setattr(pf, "normalize_pose_xy_v3", lambda lm: object())
    matcher = _FakeMatcher()

    app_ui.App._feed_online_matcher(matcher, None, 1.0, 1)      # pose_landmarks=None
    app_ui.App._feed_online_matcher(None, object(), 1.0, 1)     # matcher=None（不抛）
    assert matcher.pushed == []


def test_feed_online_matcher_skips_when_normalize_returns_none(monkeypatch):
    import core.pose_features as pf

    monkeypatch.setattr(pf, "normalize_pose_xy_v3", lambda lm: None)  # 无效帧
    matcher = _FakeMatcher()

    app_ui.App._feed_online_matcher(matcher, object(), 1.0, 1)
    assert matcher.pushed == []


# --- _post_match（关窗竞态三层守卫）----------------------------------------

def test_post_match_hit_updates_var_and_strips_view_suffix():
    """命中：after(0, _apply) 执行，显示去掉 _正面 后缀的动作名 + 分数。"""
    app = object.__new__(app_ui.App)
    app._stop_evt = threading.Event()  # 未 set
    app.root = _FakeRoot()
    app.match_var = _SettableVar("识别：待机")

    app_ui.App._post_match(app, MatchResult(action="直拳_正面", score=0.87, start_frame=0, end_frame=1))

    assert app.root.after_calls == [0]
    assert app.match_var.value == "识别到：直拳 (0.87)"


def test_post_match_suppressed_after_stop():
    """已停止：提前 return，不调 after、不刷界面。"""
    app = object.__new__(app_ui.App)
    app._stop_evt = threading.Event()
    app._stop_evt.set()
    app.root = _FakeRoot()
    app.match_var = _SettableVar("识别：待机")

    app_ui.App._post_match(app, MatchResult(action="直拳_正面", score=0.9, start_frame=0, end_frame=1))

    assert app.root.after_calls == []
    assert app.match_var.value == "识别：待机"


def test_post_match_none_action_does_not_touch_root():
    """无匹配（action=None）：不刷新，不覆盖上一次成功结果。"""
    app = object.__new__(app_ui.App)
    app._stop_evt = threading.Event()

    class _BoomRoot:
        def after(self, *a):
            raise AssertionError("action=None 不应触碰 root")

    app.root = _BoomRoot()
    app.match_var = _SettableVar("识别到：直拳 (0.87)")

    app_ui.App._post_match(app, MatchResult(action=None, score=0.1, start_frame=0, end_frame=0))
    assert app.match_var.value == "识别到：直拳 (0.87)"  # 保留


def test_post_match_survives_tclerror():
    """root 已 destroy 竞态：after 抛 TclError 被吞，不外泄。"""
    app = object.__new__(app_ui.App)
    app._stop_evt = threading.Event()
    app.match_var = _SettableVar("识别：待机")

    class _RaisingRoot:
        def after(self, *a):
            raise app_ui.TclError("application has been destroyed")

    app.root = _RaisingRoot()
    app_ui.App._post_match(app, MatchResult(action="直拳_正面", score=0.9, start_frame=0, end_frame=1))  # 不抛即通过


# --- 结构守卫：两条预览循环都接了 matcher ----------------------------------

def test_both_preview_loops_wire_matcher():
    """单线程 + 并行两条实时循环都必须 build+feed matcher（默认 workers=2 走并行）。"""
    tree = ast.parse(Path(app_ui.__file__).read_text(encoding="utf-8"))
    build = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_build_online_matcher"
    ]
    feed = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_feed_online_matcher"
    ]
    assert len(build) >= 2, "两条预览循环都应 build matcher"
    assert len(feed) >= 2, "两条预览循环都应 feed matcher"
