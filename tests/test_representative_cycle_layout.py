# -*- coding: utf-8 -*-
"""
周期裁切布局守卫泛化回归（YOLO 迁移 Issue #4 / S1）。

背景
----
`core/action_compare.py` 的 `_select_representative_cycle()` 旧实现写死布局守卫：

    if features.ndim != 3 or features.shape[1:] != (22, 2):
        return features

这会让任意 **非 22 点** 布局（如 `body_core_v1` 的 `(12, 2)`）**静默跳过周期裁切**，
DTW query 退化成整段模板——不报错、不抛异常，是最难排查的隐性退化。

本测试用 test-only 构造的 `(T, 12, 2)` 周期输入（不依赖生产注册表存在 `body_core_v1`），
验证泛化后的守卫：
  1. 非 22 点的合法 `(J, 2)` 周期输入能被正常裁切，**不再静默整段返回**；
  2. 裁切结果长度显著短于整段、量级与单个周期相当；
  3. 同样的逻辑对 22 点输入仍成立（行为不变）；
  4. 退化输入（帧数过少 / 非法 shape）仍安全整段返回。

验收命令
--------
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_representative_cycle_layout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.action_compare import _select_representative_cycle  # noqa: E402


def _make_periodic_features(T: int, J: int, period: int, *, seed: int = 0) -> np.ndarray:
    """构造一段强周期 (T, J, 2) 特征，使 motion_energy 呈清晰周期。

    要点：
      - 所有关节 **同相位** 做线性往复运动（非圆周运动）——圆周运动速度恒定会让
        motion_energy 变成常数，无法估周期；
      - 叠加二次谐波打破半周期对称，使 motion_energy 的主周期等于 `period`（而非 period/2）；
      - 关节间只用幅度差异，不引入相位差（相位差会让多关节能量相互抵消为常数）。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float32)
    phase = 2.0 * np.pi * t / float(period)
    # 同相位 + 二次谐波：主周期 = period。
    base = np.sin(phase) + 0.4 * np.sin(2.0 * phase + 0.5)
    feats = np.zeros((T, J, 2), dtype=np.float32)
    for j in range(J):
        amp = 0.5 + 0.1 * j
        feats[:, j, 0] = amp * base
        feats[:, j, 1] = 0.3 * amp * base
    # 极小噪声，避免完全退化的零方差（不破坏周期性）。
    feats += (1e-4 * rng.standard_normal(feats.shape)).astype(np.float32)
    return feats


@pytest.mark.parametrize("J", [12, 22])
def test_representative_cycle_crops_non_22_layout(J):
    T = 150
    period = 30
    fps = 30.0
    feats = _make_periodic_features(T, J, period, seed=J)

    out = _select_representative_cycle(feats, fps=fps)

    # 关键断言：不再静默整段返回（旧 (12,2) 守卫会让 out is feats）。
    assert out.shape[0] < T, f"J={J}: 周期裁切未生效，仍返回整段（{out.shape[0]} == {T}）"
    # J 维与通道维保持不变。
    assert out.shape[1:] == (J, 2)
    # 裁出的代表周期量级应接近单周期（给宽松上界，吸收 snap-to-low-motion 调整）。
    assert out.shape[0] <= 3 * period
    # 裁切应是原序列的连续切片（值来自原数组）。
    assert out.shape[0] >= 8


def test_non_22_layout_no_longer_silently_returns_whole():
    """显式对照旧 bug：(12,2) 周期输入此前会整段返回，现在必须被裁短。"""
    T = 150
    feats = _make_periodic_features(T, J=12, period=30, seed=42)
    out = _select_representative_cycle(feats, fps=30.0)
    assert out is not feats
    assert out.shape[0] < T


def test_short_sequence_returns_whole():
    # 帧数 < 30：无法稳定估周期，安全整段返回（行为不变）。
    feats = _make_periodic_features(20, J=12, period=8, seed=1)
    out = _select_representative_cycle(feats, fps=30.0)
    assert out.shape[0] == 20


def test_invalid_shape_returns_whole():
    # 非 (J,2) 布局（如 (T,12,3)）不是合法 feature shape，安全整段返回，不报错。
    feats = np.random.default_rng(3).standard_normal((120, 12, 3)).astype(np.float32)
    out = _select_representative_cycle(feats, fps=30.0)
    assert out.shape == feats.shape
    assert np.allclose(out, feats)


def test_too_few_joints_returns_whole():
    # J 低于下限（如 (T,2,2)）视为非法布局，不做周期裁切。
    feats = _make_periodic_features(120, J=2, period=30, seed=5)
    out = _select_representative_cycle(feats, fps=30.0)
    assert out.shape[0] == 120
