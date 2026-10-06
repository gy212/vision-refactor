# -*- coding: utf-8 -*-
"""一维信号：移动平均、最长连续段、连续段切分、截尾均值、运动能量与活动区间的唯一实现。

数值口径沿用合并前各调用方的实现；调用方之间的差异（窗口奇数化、输出 dtype）写成参数。
"""
from __future__ import annotations

import numpy as np


def smooth_1d(x: np.ndarray, k: int, *, odd: bool = True, dtype=np.float32) -> np.ndarray:
    """``mode="same"`` 的移动平均。

    - ``odd=True``：偶数窗口加 1；``k <= 1``、空输入或样本少于窗口时原样返回副本。
    - ``dtype=np.float32``：输入与输出都转成 float32（比对、技术指标的切分用）。
    - ``dtype=None``：不转换 dtype，样本不足时返回输入本身（考试裁剪的能量平滑用）。
    """
    k = int(k)
    if dtype is None:
        if k < 2 or x.size < k:
            return x
        kernel = np.ones(k, dtype=np.float32) / float(k)
        return np.convolve(x, kernel, mode="same")
    if k <= 1 or x.size == 0:
        return x.astype(dtype, copy=True)
    if odd and k % 2 == 0:
        k += 1
    if x.size < k:
        return x.astype(dtype, copy=True)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x.astype(dtype), kernel, mode="same").astype(dtype)


def longest_segment(mask: np.ndarray) -> tuple[int, int] | None:
    """最长的连续 True 段 ``(start, end)``，两端都含；等长时取最早的一段；没有 True 返回 None。"""
    if mask.size == 0:
        return None

    best_s: int | None = None
    best_e: int | None = None
    i = 0
    n = int(mask.size)
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        s = i
        i += 1
        while i < n and bool(mask[i]):
            i += 1
        e = i - 1
        if best_s is None or (e - s) > (best_e - best_s):  # type: ignore[operator]
            best_s, best_e = s, e
    if best_s is None or best_e is None:
        return None
    return int(best_s), int(best_e)


def contiguous_runs(indices) -> list[np.ndarray]:
    """把升序帧号切成相邻差为 1 的若干段。"""
    if len(indices) == 0:
        return []
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)


def trimmed_mean(scores: list[float]) -> float:
    """均值；不少于 3 个样本时去掉一个最小值和一个最大值。空列表为 0。"""
    if not scores:
        return 0.0
    if len(scores) >= 3:
        xs = sorted(float(s) for s in scores)
        xs = xs[1:-1]  # drop min/max
        return float(sum(xs) / max(1, len(xs)))
    return float(sum(scores) / len(scores))


def motion_energy(seq: np.ndarray) -> np.ndarray:
    """相邻帧的 RMS 位移，``(T, D)`` → ``(T-1,)``；输入须为有限值。"""
    d = np.diff(seq, axis=0)
    return np.sqrt(np.mean(d * d, axis=1))


def find_active_range(energy: np.ndarray, pad: int = 10) -> tuple[int, int]:
    """按运动能量取最活跃的连续段，返回原始帧号 ``(start, end)``。

    能量先做 9 点移动平均（样本不足 9 个时不平滑），高于 70 分位的帧算活跃；
    能量第 i 项对应帧 i→i+1 的变化，结果两端各外扩 ``pad`` 帧。
    """
    if energy.size == 0:
        return 0, 0

    k = 9
    if energy.size >= k:
        kernel = np.ones(k, dtype=np.float32) / k
        e = np.convolve(energy, kernel, mode="same")
    else:
        e = energy

    thr = float(np.percentile(e, 70))
    active = e > thr
    if not active.any():
        return 0, int(energy.size)  # energy is T-1

    best_s = best_e = 0
    cur_s = None
    for i, on in enumerate(active.tolist()):
        if on and cur_s is None:
            cur_s = i
        if (not on) and cur_s is not None:
            cur_e = i - 1
            if (cur_e - cur_s) > (best_e - best_s):
                best_s, best_e = cur_s, cur_e
            cur_s = None
    if cur_s is not None:
        cur_e = int(active.size - 1)
        if (cur_e - cur_s) > (best_e - best_s):
            best_s, best_e = cur_s, cur_e

    start = max(0, best_s - pad)
    end = min(int(active.size), best_e + 1 + pad)
    return start, end
