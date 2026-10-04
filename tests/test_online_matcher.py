# -*- coding: utf-8 -*-
"""OnlineActionMatcher 单测：合成序列驱动闸门 + 模板库识别。

不依赖摄像头/MediaPipe：直接用 numpy 造归一化特征 (22,2)。
"""
from __future__ import annotations

import threading
import time

import numpy as np

from core import paths
from core.feature_layout import POSE33_V3
from core.online_matcher import (
    ActionTemplate,
    MatcherConfig,
    OnlineActionMatcher,
    load_template_library,
    match_window,
)


def _idle_frame(rng: np.ndarray) -> np.ndarray:
    # 接近静止的姿态 + 微抖（低能量）。
    base = np.zeros((22, 2), dtype=np.float32)
    base[:, 1] = np.linspace(-1.0, 1.0, 22)  # 竖直分布的骨架
    return base + rng * 0.002


# pose33_v3 行 1/3/5 = 右肩/肘/腕（source 12/14/16）。整条右臂一起动 → 能量更接近真实出拳。
_RIGHT_ARM_ROWS = (1, 3, 5)


def _action_traj(amp_axis: int, n: int = 18) -> np.ndarray:
    """造一段动作轨迹 (n,22,2)：右臂沿 amp_axis 三角波恒速来回，其余点静止。

    amp_axis=0 ~ 横向挥（摆拳味），amp_axis=1 ~ 纵向推（直拳味）——只为区分两类模板。
    三角波（非正弦）使动作期速度恒定，避免顶点速度归零被误判闭合。
    """
    seq = np.zeros((n, 22, 2), dtype=np.float32)
    for t in range(n):
        f = np.zeros((22, 2), dtype=np.float32)
        f[:, 1] = np.linspace(-1.0, 1.0, 22)
        u = t / (n - 1)
        phase = 2.0 * u if u <= 0.5 else 2.0 * (1.0 - u)  # 三角波 0→1→0
        for row in _RIGHT_ARM_ROWS:
            f[row, amp_axis] += 1.5 * phase
        seq[t] = f
    return seq


# 合成信号下贴合的阈值：测状态机逻辑，不是默认现场值（默认 arm_thr=0.06 针对真实摄像头）。
_TEST_CFG = MatcherConfig(arm_thr=0.02, disarm_thr=0.008, hold_ms=120.0, min_window_ms=120.0)


def _build_matcher(on_result, cfg=None):
    tpl_a = ActionTemplate(name="punch_straight", features=_action_traj(amp_axis=1))
    tpl_b = ActionTemplate(name="punch_hook", features=_action_traj(amp_axis=0))
    return OnlineActionMatcher([tpl_a, tpl_b], on_result, cfg=cfg or _TEST_CFG)


def _feed(matcher, frames, *, fps: float = 60.0, start_frame: int = 0, drop_every: int | None = None):
    """以固定帧间隔喂入 frames；drop_every 模拟丢帧（跳过该帧但时间戳照常推进）。"""
    dt_ms = 1000.0 / fps
    idx = start_frame
    for i, f in enumerate(frames):
        ts = idx * dt_ms
        if drop_every is None or (i % drop_every) != 0:
            matcher.push(f, ts, idx)
        idx += 1
    return idx


def _wait_results(results, n, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(results) >= n:
            return
        time.sleep(0.02)


def test_bundled_straight_punch_templates_load_from_default_library():
    templates = load_template_library(paths.templates_dir() / "online")
    by_name = {template.name: template for template in templates}

    assert {"直拳_左手", "直拳_右手"} <= by_name.keys()
    for name in ("直拳_左手", "直拳_右手"):
        features = by_name[name].features
        assert features.dtype == np.float32
        assert features.ndim == 3
        assert features.shape[0] > 0
        assert features.shape[1:] == POSE33_V3.shape
        assert np.isfinite(features).all()


def test_match_window_picks_closest_template():
    # 纯函数层：贴近 A 的窗口应识别成 A 而非 B。
    cfg = MatcherConfig()
    tpl_a = ActionTemplate(name="punch_straight", features=_action_traj(amp_axis=1))
    tpl_b = ActionTemplate(name="punch_hook", features=_action_traj(amp_axis=0))
    window = _action_traj(amp_axis=1, n=20)  # 像 A
    res = match_window(window, [tpl_a, tpl_b], cfg)
    assert res.action == "punch_straight"
    assert res.score >= cfg.reject_score_thr
    assert res.score_authorized is False


def test_idle_does_not_trigger():
    results = []
    lock = threading.Lock()
    matcher = _build_matcher(lambda r: (lock.acquire(), results.append(r), lock.release()))
    try:
        rs = np.random.RandomState(0)
        idle = [_idle_frame(rs.randn(22, 2).astype(np.float32)) for _ in range(120)]
        _feed(matcher, idle)
        time.sleep(0.3)
        assert results == [], "纯 idle 不应触发任何识别"
    finally:
        matcher.close()


def test_action_triggers_once_and_recognizes_a():
    results = []
    matcher = _build_matcher(results.append)
    try:
        rs = np.random.RandomState(1)
        idle_pre = [_idle_frame(rs.randn(22, 2).astype(np.float32)) for _ in range(40)]
        action = list(_action_traj(amp_axis=1, n=18))  # A 动作
        idle_post = [_idle_frame(rs.randn(22, 2).astype(np.float32)) for _ in range(40)]
        idx = _feed(matcher, idle_pre)
        idx = _feed(matcher, action, start_frame=idx)
        _feed(matcher, idle_post, start_frame=idx)
        _wait_results(results, 1)
        assert len(results) == 1, f"一次动作应恰好闭合一个窗口，得到 {len(results)}"
        assert results[0].action == "punch_straight"
        assert results[0].end_frame > results[0].start_frame
    finally:
        matcher.close()


def test_time_window_hysteresis_survives_frame_drops():
    # 丢帧（每 3 帧丢 1）下时间戳照常推进，窗口仍应正确闭合并识别。
    results = []
    matcher = _build_matcher(results.append)
    try:
        rs = np.random.RandomState(2)
        idle_pre = [_idle_frame(rs.randn(22, 2).astype(np.float32)) for _ in range(40)]
        action = list(_action_traj(amp_axis=0, n=18))  # B 动作
        idle_post = [_idle_frame(rs.randn(22, 2).astype(np.float32)) for _ in range(40)]
        idx = _feed(matcher, idle_pre, drop_every=3)
        idx = _feed(matcher, action, start_frame=idx, drop_every=3)
        _feed(matcher, idle_post, start_frame=idx, drop_every=3)
        _wait_results(results, 1)
        assert len(results) == 1
        assert results[0].action == "punch_hook"
    finally:
        matcher.close()
