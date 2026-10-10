# -*- coding: utf-8 -*-
"""子序列 DTW 的唯一实现（含回溯路径版）：模板比对、在线识别与研究脚本都经这里（设计文档 §7.5）。"""
from __future__ import annotations

from collections.abc import Callable
from threading import Event

import numpy as np

_DtwCancelSignal = Event | Callable[[], bool]


def _raise_if_dtw_cancelled(stop_evt: _DtwCancelSignal | None) -> None:
    if stop_evt is None:
        return
    should_stop = stop_evt if callable(stop_evt) else stop_evt.is_set
    if should_stop():
        raise InterruptedError("DTW 评分已取消")


def subsequence_dtw(
    query: np.ndarray,
    seq: np.ndarray,
    *,
    stop_evt: _DtwCancelSignal | None = None,
) -> tuple[float, int, int]:
    """
    Subsequence DTW: find best matching subsequence of `seq` for `query`.
    Returns (cost, start_index, end_index) in seq indices (inclusive).
    """
    _raise_if_dtw_cancelled(stop_evt)
    q = query.astype(np.float32).reshape(query.shape[0], -1)
    s = seq.astype(np.float32).reshape(seq.shape[0], -1)

    n, m = q.shape[0], s.shape[0]
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    prev = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up, 2 left
    dp[0, :] = 0.0

    local_cost = _dtw_local_cost(q, s)
    _raise_if_dtw_cancelled(stop_evt)
    for i in range(1, n + 1):
        _raise_if_dtw_cancelled(stop_evt)
        for j in range(1, m + 1):
            cost = float(local_cost[i - 1, j - 1])
            a = dp[i - 1, j - 1]
            b = dp[i - 1, j]
            c = dp[i, j - 1]
            if a <= b and a <= c:
                dp[i, j] = cost + a
                prev[i, j] = 0
            elif b <= c:
                dp[i, j] = cost + b
                prev[i, j] = 1
            else:
                dp[i, j] = cost + c
                prev[i, j] = 2

    _raise_if_dtw_cancelled(stop_evt)
    end = int(np.argmin(dp[n, 1:]) + 1)  # dp-space
    best_cost = float(dp[n, end])

    i, j = n, end
    while i > 0:
        _raise_if_dtw_cancelled(stop_evt)
        p = int(prev[i, j])
        if p == 0:
            i -= 1
            j -= 1
        elif p == 1:
            i -= 1
        else:
            j -= 1
        if j <= 0:
            break
    start = max(1, j)  # dp-space

    _raise_if_dtw_cancelled(stop_evt)
    return best_cost, start - 1, end - 1


def subsequence_dtw_with_path(
    query: np.ndarray,
    seq: np.ndarray,
    *,
    stop_evt: _DtwCancelSignal | None = None,
) -> tuple[float, int, int, list[tuple[int, int]]]:
    _raise_if_dtw_cancelled(stop_evt)
    q = query.astype(np.float32).reshape(query.shape[0], -1)
    s = seq.astype(np.float32).reshape(seq.shape[0], -1)

    n, m = q.shape[0], s.shape[0]
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    prev = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up, 2 left
    dp[0, :] = 0.0

    local_cost = _dtw_local_cost(q, s)
    _raise_if_dtw_cancelled(stop_evt)
    for i in range(1, n + 1):
        _raise_if_dtw_cancelled(stop_evt)
        for j in range(1, m + 1):
            cost = float(local_cost[i - 1, j - 1])
            a = dp[i - 1, j - 1]
            b = dp[i - 1, j]
            c = dp[i, j - 1]
            if a <= b and a <= c:
                dp[i, j] = cost + a
                prev[i, j] = 0
            elif b <= c:
                dp[i, j] = cost + b
                prev[i, j] = 1
            else:
                dp[i, j] = cost + c
                prev[i, j] = 2

    _raise_if_dtw_cancelled(stop_evt)
    end = int(np.argmin(dp[n, 1:]) + 1)
    best_cost = float(dp[n, end])

    path_rev: list[tuple[int, int]] = []
    i, j = n, end
    while i > 0 and j > 0:
        _raise_if_dtw_cancelled(stop_evt)
        path_rev.append((int(i - 1), int(j - 1)))
        p = int(prev[i, j])
        if p == 0:
            i -= 1
            j -= 1
        elif p == 1:
            i -= 1
        else:
            j -= 1

    start = max(1, int(j))
    path = list(reversed(path_rev))
    _raise_if_dtw_cancelled(stop_evt)
    return best_cost, start - 1, end - 1, path


def _dtw_local_cost(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    diff = q[:, None, :] - s[None, :, :]
    sq = np.sum(diff * diff, axis=2, dtype=np.float32)
    return np.sqrt(sq).astype(np.float32, copy=False)
