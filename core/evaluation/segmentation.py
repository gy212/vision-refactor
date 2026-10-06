# -*- coding: utf-8 -*-
"""时间轴切分的唯一实现：正侧切分、周期估计与代表周期、匹配用的活跃区间、考试裁剪的动作区间（设计文档 §7.3）。

正侧切分有两种现存口径，按 ``mode`` 选用（§3 第 24 项）：
- ``"median_always"``：双模板比对的内联切分，平滑后按中位数二分，总给出两段（可能为 None）；
- ``"conservative"``：技术指标的切分，只有两段都够长、均值差够大且一高一低时才判"mixed"，否则整段当正面。
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from core.foundation.signal import find_active_range, longest_segment, motion_energy, smooth_1d
from core.pose.layouts import is_valid_feature_shape

Segment = tuple[int, int]
SplitMode = Literal["median_always", "conservative"]


def _halves(view_scores: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray, Segment | None, Segment | None]:
    """平滑视角分数，按中位数分成高/低两半，返回 ``(平滑序列, 高半掩码, 最长高段, 最长低段)``。"""
    vs = np.nan_to_num(view_scores.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    k = int(max(5, min(51, round(float(fps) * 0.5))))
    vs_s = smooth_1d(vs, k=k)
    thr = float(np.median(vs_s))
    high_mask = vs_s >= thr
    return vs_s, high_mask, longest_segment(high_mask), longest_segment(~high_mask)


def _shrink(seg: Segment | None, margin: int, min_len: int) -> Segment | None:
    """去掉转身附近的临界帧：段长超过 ``2*margin+10`` 时两端各收 ``margin``；剩余短于 ``min_len`` 判无效。"""
    if seg is None:
        return None
    s, e = int(seg[0]), int(seg[1])
    if e - s + 1 > (2 * margin + 10):
        s += margin
        e -= margin
    if (e - s + 1) < min_len:
        return None
    return s, e


def split_front_side(
    view_scores: np.ndarray,
    fps: float,
    *,
    mode: SplitMode,
    min_len_sec: float = 1.0,
    min_mean_diff: float = 0.12,
    side_thr: float = 0.9,
    front_thr: float = 0.95,
    max_transitions: int = 3,
) -> tuple[Segment | None, Segment | None, str]:
    """按逐帧视角分数切出 ``(正面段, 侧面段, view_mode)``；阈值参数只用于 ``conservative``。

    ``median_always`` 的 view_mode 恒为 ``"median"``；``conservative`` 为 ``"mixed"`` 或 ``"single"``
    （single 时整段当正面、侧面为 None——单一启发式难以可靠识别纯侧面，宁可不给也不误标）。
    """
    margin = int(max(5, min(20, round(float(fps) * 0.3))))
    if mode == "median_always":
        _vs_s, _high, high_seg, low_seg = _halves(view_scores, fps)
        return _shrink(high_seg, margin, 2), _shrink(low_seg, margin, 2), "median"
    if mode != "conservative":
        raise ValueError(f"未知的正侧切分方式：{mode!r}")
    if view_scores.size == 0:
        return None, None, "single"

    vs_s, high_mask, high_seg, low_seg = _halves(view_scores, fps)
    min_len = int(max(1, round(float(fps) * float(min_len_sec))))
    transitions = int(np.sum(high_mask[1:] != high_mask[:-1]))
    high_seg = _shrink(high_seg, margin, min_len)
    low_seg = _shrink(low_seg, margin, min_len)

    if transitions <= int(max_transitions) and high_seg is not None and low_seg is not None:
        hs, he = high_seg
        ls, le = low_seg
        mean_high = float(np.mean(vs_s[hs : he + 1]))
        mean_low = float(np.mean(vs_s[ls : le + 1]))
        if abs(mean_high - mean_low) >= float(min_mean_diff):
            if mean_high >= mean_low:
                if mean_high >= float(front_thr) and mean_low <= float(side_thr):
                    return high_seg, low_seg, "mixed"
            else:
                if mean_low >= float(front_thr) and mean_high <= float(side_thr):
                    return low_seg, high_seg, "mixed"

    return (0, int(vs_s.size - 1)), None, "single"


def classify_single_view(view_scores: np.ndarray, *, side_thr: float, front_thr: float) -> str:
    """整段视角分数的中位数判 ``"side"``/``"front"``/``"unknown"``（技术指标的单视角提示）。"""
    vs = np.nan_to_num(view_scores.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if vs.size == 0:
        return "unknown"
    med = float(np.median(vs))
    if med <= float(side_thr):
        return "side"
    if med >= float(front_thr):
        return "front"
    return "unknown"


def estimate_period_frames(energy: np.ndarray, fps: float) -> int | None:
    """运动能量的归一化自相关估计主重复周期（帧）；不够可信时返回 None。"""
    e = energy.astype(np.float32).ravel()
    if e.size < 30:
        return None

    x = e - float(np.mean(e))
    x_std = float(np.std(x))
    if not np.isfinite(x_std) or x_std < 1e-6:
        return None

    fps = float(fps or 0.0) or 30.0
    min_lag = max(6, int(0.4 * fps))
    max_lag = min(int(3.0 * fps), int(e.size // 2))
    if max_lag <= min_lag:
        return None

    best_lag = None
    best_corr = -1.0
    for lag in range(min_lag, max_lag + 1):
        a = x[:-lag]
        b = x[lag:]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        c = float(np.dot(a, b) / denom)
        if c > best_corr:
            best_corr = c
            best_lag = lag

    if best_lag is None or best_corr < 0.15:  # 周期性弱：不强行切周期
        return None
    return int(best_lag)


def select_representative_cycle(features: np.ndarray, fps: float) -> np.ndarray:
    """重复的标准动作取中间一个周期作 DTW 查询；估不出稳定周期时原样返回。

    只要求 ``features.shape[1:]`` 是合法的 ``(J, 2)`` 布局（Issue #4：不再写死 22 点）。
    """
    if features.ndim != 3 or not is_valid_feature_shape(features.shape[1:]):
        return features
    if features.shape[0] < 30:
        return features

    seq = features.reshape(features.shape[0], -1)
    energy = motion_energy(seq)
    period = estimate_period_frames(energy, fps=float(fps))
    if period is None or period < 8:
        return features

    t = int(features.shape[0])
    mid = t // 2
    start = max(0, mid - (period // 2))

    # 起止点尽量落在低运动帧上，避免切在动作中间。
    e_frame = np.concatenate([energy[:1], energy]).astype(np.float32)  # len T
    low_thr = float(np.percentile(e_frame, 30))
    low = np.where(e_frame <= low_thr)[0].astype(int)

    def nearest_low(idx: int) -> int | None:
        if low.size == 0:
            return None
        radius = max(3, period // 3)
        lo = int(max(0, idx - radius))
        hi = int(min(t - 1, idx + radius))
        cand = low[(low >= lo) & (low <= hi)]
        if cand.size == 0:
            return None
        return int(cand[np.argmin(np.abs(cand - idx))])

    s2 = nearest_low(start)
    if s2 is not None:
        start = int(s2)

    end = min(t, start + period)
    e2 = nearest_low(end)
    if e2 is not None and int(e2) > start + 10:
        end = int(e2)

    if end - start < max(12, period // 2):
        end = min(t, start + period)

    return features[start:end]


def active_window(seq: np.ndarray, seg: Segment | None) -> tuple[np.ndarray, int]:
    """学员序列的匹配窗口：先取分段（None 为整段），再收窄到运动最活跃的区间；返回 ``(窗口, 起始帧)``。"""
    last = int(seq.shape[0] - 1)
    if seg is None:
        seg_s, seg_e = 0, last
    else:
        seg_s = max(0, min(int(seg[0]), last))
        seg_e = max(seg_s, min(int(seg[1]), last))

    window = seq[seg_s : seg_e + 1]
    offset = seg_s
    if window.shape[0] >= 2:  # 静止帧会干扰匹配
        energy = motion_energy(window.reshape(window.shape[0], -1))
        a_s, a_e = find_active_range(energy, pad=10)
        a_s = max(0, min(int(a_s), int(window.shape[0] - 1)))
        a_e = max(a_s, min(int(a_e), int(window.shape[0] - 1)))
        window = window[a_s : a_e + 1]
        offset += int(a_s)
    return window, offset


def estimate_action_range(
    frame_count: int,
    *,
    fps: float = 30.0,
    energy: np.ndarray | None = None,
    smart_crop: bool = False,
    trim_head_s: float = 0.5,
    trim_tail_s: float = 0.8,
    tail_quiet_pct: float = 40.0,
) -> tuple[int, int, list[str]]:
    """考试裁剪的动作区间，返回半开 ``[start, end)`` 与说明；异常或区间过短一律回退整段，不抛错。

    默认（``smart_crop=False``）：进场已由 presence_gate 挡干净，头部固定掐 ``trim_head_s``，中段不裁，
    尾部按能量从末尾回扫到最后一次动作（离场后的空场被切掉，强停时尾部仍是动作则不切）。
    ``smart_crop=True``：``find_active_range`` 活跃段 + 时间裁剪（模拟与单测显式开启）。
    """
    warnings: list[str] = []
    n = int(frame_count)
    if n <= 0:
        return 0, 0, ["empty"]
    fps = float(fps or 30.0)

    if smart_crop:
        if energy is not None and energy.size >= 5:
            # find_active_range 基于 T-1 的 energy，返回含端点；转半开
            s, e_incl = find_active_range(np.asarray(energy, dtype=np.float32), pad=10)
            start = max(0, int(s))
            end = min(n, int(e_incl) + 2)
            if end - start >= max(8, int(0.5 * fps)):
                return start, end, warnings
            warnings.append("motion_range_too_short_fallback_time_trim")

        head = int(round(trim_head_s * fps))
        tail = int(round(trim_tail_s * fps))
        start = min(head, max(0, n // 4))
        end = max(start + 1, n - tail)
        if end <= start:
            start, end = 0, n
            warnings.append("trim_collapsed_use_full")
        else:
            warnings.append(f"time_trim: drop head~{head}f tail~{tail}f -> [{start},{end})")
        return start, end, warnings

    start = min(int(round(trim_head_s * fps)), max(0, n // 4))
    end = n
    if energy is not None and energy.size >= 5:
        e = smooth_1d(np.asarray(energy, dtype=np.float32), max(3, int(round(0.30 * fps))), odd=False, dtype=None)
        thr = float(np.percentile(e, tail_quiet_pct))
        active = np.nonzero(e > thr)[0]
        if active.size:
            # energy idx → 帧 +1；末动作后留 0.3s 收势余量
            end = min(n, int(active[-1]) + 2 + int(round(0.3 * fps)))
            warnings.append(f"tail_scan: end={end} (thr@p{tail_quiet_pct:.0f})")
    if end - start < max(15, int(0.5 * fps)):
        start, end = 0, n
        warnings.append("clip_too_short_use_full")
    return start, end, warnings


def estimate_action_range_from_features(
    features: np.ndarray,
    *,
    fps: float = 30.0,
    trim_head_s: float = 0.5,
    trim_tail_s: float = 0.8,
) -> tuple[int, int, list[str]]:
    """对 ``(T, J, 2)`` 特征序列估活跃区间（``smart_crop=True``；单测与可选路径）。"""
    if features.ndim != 3 or features.shape[0] < 2:
        return 0, int(getattr(features, "shape", [0])[0] or 0), ["bad_features"]
    seq = features.reshape(features.shape[0], -1).astype(np.float32)
    energy = motion_energy(seq)
    return estimate_action_range(
        features.shape[0], fps=fps, energy=energy, smart_crop=True,
        trim_head_s=trim_head_s, trim_tail_s=trim_tail_s,
    )
