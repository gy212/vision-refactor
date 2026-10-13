# -*- coding: utf-8 -*-
"""实时在线 DTW 识别的接入层测试（无需 Tk root）。

覆盖接入的关键风险，不重测 matcher 内核（见 test_online_matcher.py）：
- 无 templates/online 或加载失败时不识别、预览照常（OnlineMatchStage.open 后 matcher 为 None）
- process 把主路原始关键点 normalize 后 push，None 关键点或无效特征不 push
- 命中经 ``match`` 事件交给界面：去掉视角后缀；无匹配不发；会话停止后界面不刷新
- 单摄顺序与并行两种推理都启用在线识别（配置表守卫，替代旧的 AST 守卫）
P6 起取代对 App._build_online_matcher / _feed_online_matcher / _post_match 的测试。
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import core.live.stages as stages  # noqa: E402
from apps import app_ui  # noqa: E402
from core.evaluation import MatchResult  # noqa: E402
from core.live import LiveControls, camera_session  # noqa: E402
from core.live.stages import FramePacket, OnlineMatchStage, StageContext, build_stages, match_text  # noqa: E402
from core.media import PRIMARY, CaptureSpec  # noqa: E402


class _SettableVar:
    def __init__(self, value: object = "") -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class _FakeMatcher:
    def __init__(self, templates, on_result) -> None:
        self.on_result = on_result
        self.pushed: list[tuple] = []
        self.closed = False

    def push(self, feat, ts, idx) -> None:
        self.pushed.append((feat, ts, idx))

    def close(self) -> None:
        self.closed = True


def _ctx(events: list) -> StageContext:
    config = camera_session({PRIMARY: CaptureSpec(0)}, skeleton=True)
    return StageContext(config, LiveControls(), lambda kind, payload: events.append((kind, payload)))


def _opened_stage(monkeypatch, tmp_path, events, *, templates=("t",)) -> OnlineMatchStage:
    monkeypatch.setattr(stages, "templates_dir", lambda: tmp_path)
    monkeypatch.setattr(stages, "load_template_library", lambda _dir: list(templates))
    monkeypatch.setattr(stages, "OnlineActionMatcher", _FakeMatcher)
    stage = OnlineMatchStage()
    stage.open(_ctx(events))
    return stage


# --- open ---------------------------------------------------------------------

def test_no_templates_disables_matching(monkeypatch, tmp_path):
    events: list = []
    stage = _opened_stage(monkeypatch, tmp_path, events, templates=())
    assert stage._matcher is None
    stage.process(FramePacket(0, {}, pose={PRIMARY: object()}), _ctx(events))  # 不抛


def test_load_error_is_swallowed(monkeypatch, tmp_path):
    monkeypatch.setattr(stages, "templates_dir", lambda: tmp_path)

    def _boom(_dir):
        raise ValueError("模板布局不符")

    monkeypatch.setattr(stages, "load_template_library", _boom)
    stage = OnlineMatchStage()
    stage.open(_ctx([]))
    assert stage._matcher is None


# --- process ------------------------------------------------------------------

def test_process_normalizes_raw_primary_pose_and_pushes(monkeypatch, tmp_path):
    events: list = []
    stage = _opened_stage(monkeypatch, tmp_path, events)
    sentinel = object()
    monkeypatch.setattr(stages, "normalize_pose_xy_v3", lambda lm: sentinel)

    stage.process(FramePacket(7, {}, timestamp_ms=123, pose={PRIMARY: object()}), _ctx(events))

    assert stage._matcher.pushed == [(sentinel, 123, 7)]


def test_process_skips_missing_pose_and_invalid_feature(monkeypatch, tmp_path):
    events: list = []
    stage = _opened_stage(monkeypatch, tmp_path, events)
    monkeypatch.setattr(stages, "normalize_pose_xy_v3", lambda lm: None)
    stage.process(FramePacket(1, {}, pose={PRIMARY: None}), _ctx(events))
    stage.process(FramePacket(2, {}, pose={PRIMARY: object()}), _ctx(events))
    assert stage._matcher.pushed == []
    stage.close()


# --- 命中事件 ---------------------------------------------------------------------

def test_hit_emits_match_event_without_view_suffix(monkeypatch, tmp_path):
    events: list = []
    stage = _opened_stage(monkeypatch, tmp_path, events)
    stage._matcher.on_result(MatchResult(action="直拳_正面", score=0.87, start_frame=0, end_frame=1))
    stage._matcher.on_result(MatchResult(action=None, score=0.1, start_frame=0, end_frame=0))
    assert events == [("match", "识别到：直拳 (0.87)")]
    assert match_text(MatchResult(action="摆拳_左侧", score=0.5, start_frame=0, end_frame=1)) == "识别到：摆拳 (0.50)"


class _FakeLive:
    def __init__(self, state: str) -> None:
        self.state = state


def test_app_shows_match_only_while_session_live():
    app = object.__new__(app_ui.App)
    app.match_var = _SettableVar("识别：待机")
    app._live = _FakeLive("running")
    app_ui.App._on_session_match(app, "识别到：直拳 (0.87)")
    assert app.match_var.value == "识别到：直拳 (0.87)"

    app._live = _FakeLive("stopping")
    app_ui.App._on_session_match(app, "识别到：摆拳 (0.50)")
    assert app.match_var.value == "识别到：直拳 (0.87)"


# --- 配置守卫：顺序与并行单摄都接在线识别 ---------------------------------------

def test_sequential_and_parallel_single_camera_both_match():
    for workers in (1, 4):
        config = camera_session({PRIMARY: CaptureSpec(0)}, skeleton=True, workers=workers)
        _pose, after = build_stages(config)
        assert any(isinstance(stage, OnlineMatchStage) for stage in after), workers
    off = camera_session({PRIMARY: CaptureSpec(0)}, skeleton=True, online_match=False)
    assert not any(isinstance(stage, OnlineMatchStage) for stage in build_stages(off)[1])
