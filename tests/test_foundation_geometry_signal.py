# -*- coding: utf-8 -*-
"""core.foundation.geometry / signal 的边界值：零向量、全无效、偶数窗口、空输入。"""
from __future__ import annotations

import numpy as np
import pytest

from core.foundation import (
    angle_deg,
    contiguous_runs,
    distance,
    find_active_range,
    landmark_xy,
    longest_segment,
    motion_energy,
    smooth_1d,
    torso_length,
    trimmed_mean,
    unit,
)


def test_angle_deg_right_straight_and_zero_vector():
    assert angle_deg(np.array([1.0, 0]), np.zeros(2), np.array([0, 1.0])) == pytest.approx(90.0)
    # 分母带 1e-9，单位长度的平角落在 180° 附近而不是恰好 180°（沿用合并前的口径）
    assert angle_deg(np.array([-1.0, 0]), np.zeros(2), np.array([1.0, 0])) == pytest.approx(180.0, abs=0.01)
    # 零向量：分母取 eps，cos=0 → 90°，不抛异常、不出 NaN
    assert angle_deg(np.zeros(2), np.zeros(2), np.array([1.0, 0])) == 90.0


def test_distance_and_unit():
    assert distance(np.array([0, 0]), np.array([3, 4])) == 5.0
    assert np.allclose(unit(np.array([3.0, 4.0])), [0.6, 0.8])
    z = unit(np.zeros(2))
    assert z.tolist() == [0.0, 0.0]


def test_landmark_xy_is_float32_copy():
    lm = np.arange(33 * 4, dtype=np.float64).reshape(33, 4)
    xy = landmark_xy(lm, 2)
    assert xy.dtype == np.float32 and xy.tolist() == [8.0, 9.0]


def test_torso_length_midpoint_never_zero():
    lm = np.zeros((33, 4), dtype=np.float32)
    assert torso_length(lm) == pytest.approx(1e-9)
    lm[[11, 12], 1] = 0.0
    lm[[23, 24], 1] = 2.0
    assert torso_length(lm) == pytest.approx(2.0)


def test_torso_length_visible_side_chain_all_invalid_is_zero():
    lm = np.random.default_rng(0).normal(size=(5, 33, 3))
    mask = np.zeros((5, 33), dtype=bool)
    assert torso_length(lm, mask, "visible_side_chain").tolist() == [0.0] * 5
    mask[:, [11, 23]] = True  # 只有左链可见
    expected = np.linalg.norm(lm[:, 11, :2] - lm[:, 23, :2], axis=1)
    assert np.allclose(torso_length(lm, mask, "visible_side_chain"), expected)
    with pytest.raises(ValueError):
        torso_length(lm, None, "visible_side_chain")
    with pytest.raises(ValueError):
        torso_length(lm, mask, "nope")  # type: ignore[arg-type]


def test_smooth_1d_even_window_becomes_odd():
    x = np.array([0, 0, 3, 0, 0], dtype=np.float64)
    out = smooth_1d(x, 2)  # 2 → 3
    assert out.dtype == np.float32
    assert np.allclose(out, [0, 1, 1, 1, 0])
    # 不奇数化、不转 dtype：样本不足或窗口 < 2 时返回输入本身
    assert smooth_1d(x, 6, odd=False, dtype=None) is x
    assert smooth_1d(x, 1, odd=False, dtype=None) is x
    assert np.allclose(smooth_1d(x, 2, odd=False, dtype=None), np.convolve(x, [0.5, 0.5], mode="same"))


def test_smooth_1d_empty_short_and_trivial_window():
    assert smooth_1d(np.zeros(0), 5).size == 0
    short = np.array([1.0, 2.0])
    out = smooth_1d(short, 5)
    assert out is not short and out.tolist() == [1.0, 2.0]
    assert smooth_1d(short, 0).dtype == np.float32


def test_longest_segment_boundaries():
    assert longest_segment(np.zeros(0, dtype=bool)) is None
    assert longest_segment(np.zeros(4, dtype=bool)) is None
    assert longest_segment(np.ones(4, dtype=bool)) == (0, 3)
    # 等长取最早的一段
    assert longest_segment(np.array([1, 1, 0, 1, 1], dtype=bool)) == (0, 1)


def test_contiguous_runs_and_trimmed_mean():
    assert contiguous_runs(np.array([], dtype=int)) == []
    runs = contiguous_runs(np.array([1, 2, 3, 7, 9, 10]))
    assert [r.tolist() for r in runs] == [[1, 2, 3], [7], [9, 10]]
    assert trimmed_mean([]) == 0.0
    assert trimmed_mean([1.0, 3.0]) == 2.0
    assert trimmed_mean([100.0, 1.0, 2.0, 3.0, -50.0]) == 2.0


def test_motion_energy_and_active_range():
    seq = np.zeros((30, 2))
    seq[10:20, 0] = np.arange(10)
    seq[20:, 0] = 9.0  # 停在终点，不制造回落尖峰
    e = motion_energy(seq)
    assert e.shape == (29,)
    assert np.allclose(e[10:19], np.sqrt(0.5)) and e[:9].max() == 0
    assert find_active_range(e, pad=0) == (10, 19)
    assert find_active_range(e, pad=3) == (7, 22)
    assert find_active_range(np.zeros(0)) == (0, 0)
    # 平坦能量没有高于 70 分位的帧 → 整段
    assert find_active_range(np.ones(5)) == (0, 5)
