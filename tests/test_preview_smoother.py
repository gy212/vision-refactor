# -*- coding: utf-8 -*-
"""PreviewLandmarkSmoother 回归：减抖、相位滞后上界、不伪造有效点。"""
from __future__ import annotations

import math

import numpy as np

from core.preview_smoother import PreviewLandmarkSmoother, SmoothedLandmark


class _LM:
    """最小 landmark 桩：仅 .x/.y/.z/.visibility，模拟 MediaPipe NormalizedLandmark。"""

    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)


def _median_jitter(seq_xy: np.ndarray) -> float:
    """相邻帧归一化位移中位数（复刻 spike_yolo_baseline.temporal_jitter 的核心公式）。

    seq_xy: (T, 2) 单点轨迹。
    """
    diffs = [
        float(np.linalg.norm(seq_xy[t] - seq_xy[t - 1]))
        for t in range(1, seq_xy.shape[0])
    ]
    return float(np.median(diffs))


def _noisy_track(n=120, fps=30.0, freq=0.5, amp=0.1, noise=0.02):
    """缓慢正弦真实运动 + 确定性白噪（seeded RNG，结果可复现）。"""
    t = np.arange(n) / fps
    clean = 0.5 + amp * np.sin(2 * math.pi * freq * t)
    rng = np.random.default_rng(0)
    jit = noise * rng.standard_normal(n)
    return clean, clean + jit, t


def test_smoother_reduces_jitter():
    """平滑后抖动应明显小于 raw。"""
    clean, noisy, t = _noisy_track()
    sm = PreviewLandmarkSmoother()
    raw_xy = []
    filt_xy = []
    for i in range(len(t)):
        lm = _LM(noisy[i], noisy[i])  # x=y=noisy，单点足够
        out = sm.feed([lm] + [_LM(0.5, 0.5)] * 32, timestamp_ms=t[i] * 1000.0)
        raw_xy.append([noisy[i], noisy[i]])
        filt_xy.append([out[0].x, out[0].y])
    raw_j = _median_jitter(np.array(raw_xy))
    filt_j = _median_jitter(np.array(filt_xy))
    assert filt_j < raw_j, f"平滑未降低抖动: raw={raw_j:.5f} filt={filt_j:.5f}"
    assert filt_j < raw_j * 0.6, "减抖幅度过小"


def test_phase_lag_within_bound():
    """跟踪滞后有界。注意：滤波只进预览绘制，不进评分链，故这里约束的是视觉响应而非评分延迟。

    缓慢干净信号是 One Euro 的**最坏情形**（无快动作，beta 不放松，cutoff≈min_cutoff），
    此时滞后约 4 帧（~130ms@30fps），对预览可接受。"""
    clean, _noisy, t = _noisy_track(noise=0.0)  # 喂干净信号看纯滞后
    sm = PreviewLandmarkSmoother()
    filt = []
    for i in range(len(t)):
        lm = _LM(clean[i], 0.5)
        out = sm.feed([lm] + [_LM(0.5, 0.5)] * 32, timestamp_ms=t[i] * 1000.0)
        filt.append(out[0].x)
    filt = np.array(filt)
    # 用互相关找 filtered 相对 clean 的最佳整数滞后（后半段，避开初始化瞬态）。
    seg = slice(40, 120)
    best_lag, best_err = 0, float("inf")
    for lag in range(0, 9):
        err = float(np.mean((filt[seg] - clean[40 - lag:120 - lag]) ** 2))
        if err < best_err:
            best_err, best_lag = err, lag
    assert best_lag <= 5, f"相位滞后过大: {best_lag} 帧"


def test_fast_motion_lag_not_worse_than_slow():
    """快动作（高速度）下 beta 放松滤波，滞后不应比慢动作更差——出拳峰值不被拖。"""
    n, fps = 120, 30.0
    t = np.arange(n) / fps
    fast = 0.5 + 0.25 * np.sin(2 * math.pi * 2.5 * t)  # 高频大幅 = 快动作
    sm = PreviewLandmarkSmoother()
    filt = []
    for i in range(n):
        out = sm.feed([_LM(fast[i], 0.5)] + [_LM(0.5, 0.5)] * 32, timestamp_ms=t[i] * 1000.0)
        filt.append(out[0].x)
    filt = np.array(filt)
    seg = slice(40, 120)
    best_lag, best_err = 0, float("inf")
    for lag in range(0, 9):
        err = float(np.mean((filt[seg] - fast[40 - lag:120 - lag]) ** 2))
        if err < best_err:
            best_err, best_lag = err, lag
    assert best_lag <= 5, f"快动作滞后过大: {best_lag} 帧"


def test_low_visibility_passthrough_not_faked():
    """低可见点原样透传：不平滑、不补点、不改写 visibility（不伪造有效点）。"""
    sm = PreviewLandmarkSmoother(vis_thr=0.5)
    low = _LM(0.123, 0.456, visibility=0.1)
    high = _LM(0.7, 0.8, visibility=0.9)
    out = sm.feed([low, high] + [_LM(0.5, 0.5)] * 31, timestamp_ms=0.0)
    # 低可见点：返回原对象本身，visibility 不变。
    assert out[0] is low
    assert out[0].visibility == 0.1
    # 高可见点：被平滑成 SmoothedLandmark，visibility 保留。
    assert isinstance(out[1], SmoothedLandmark)
    assert out[1].visibility == 0.9


def test_none_pose_resets_and_returns_none():
    """整帧丢失返回 None 并重置状态。"""
    sm = PreviewLandmarkSmoother()
    sm.feed([_LM(0.5, 0.5)] * 33, timestamp_ms=0.0)
    assert sm.feed(None, timestamp_ms=33.0) is None


def test_reset_on_reappear_no_stale_extrapolation():
    """点从无效→重新有效时重置，首帧返回当前值而非从陈旧状态外推。"""
    sm = PreviewLandmarkSmoother(vis_thr=0.5)
    pad = [_LM(0.5, 0.5)] * 32
    # 先喂一串有效高值，让滤波器收敛到 ~0.9。
    for i in range(10):
        sm.feed([_LM(0.9, 0.9, visibility=0.9)] + pad, timestamp_ms=i * 33.0)
    # 该点短暂无效（重置）。
    sm.feed([_LM(0.9, 0.9, visibility=0.1)] + pad, timestamp_ms=330.0)
    # 重新有效，位置跳到 0.1：首帧应直接返回 0.1（init），不被旧 0.9 拖拽。
    out = sm.feed([_LM(0.1, 0.1, visibility=0.9)] + pad, timestamp_ms=363.0)
    assert abs(out[0].x - 0.1) < 1e-6
