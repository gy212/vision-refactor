# -*- coding: utf-8 -*-
"""侧视角低鞭腿逐次动作事件切分。"""

from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.arms import series_angles
from core.foundation.geometry import torso_length

_ENTER_TH = 0.25
_EXIT_TH = 0.14
_PEAK_TH = 0.48
_MIN_DUR_S = 0.35
_MERGE_GAP_S = 0.35
_KNEE_FALLBACK_TH = 0.35
_TOE_FALLBACK_TH = 0.25
_EXT_RATIO = 0.80
_EXT_KNEE_DEG = 155.0


def _median_filter(arr: np.ndarray, k: int = 3) -> np.ndarray:
    """对有限值做局部中值滤波以抑制单帧左右腿标签互换。"""
    n = len(arr)
    out = arr.copy()
    half = k // 2
    for i in range(n):
        win = arr[max(0, i - half) : min(n, i + half + 1)]
        fin = win[np.isfinite(win)]
        if len(fin) >= 2:
            out[i] = float(np.median(fin))
    return out


def _interpolate_gaps(arr: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """仅对不超过 max_gap 帧的短暂缺失做线性插值，长缺口保留 NaN。"""
    out = arr.copy()
    n = len(out)
    fin = np.isfinite(out)
    if fin.sum() < 2:
        return out
    i = 0
    while i < n:
        if fin[i]:
            i += 1
            continue
        j = i
        while j < n and not fin[j]:
            j += 1
        gap = j - i
        if i > 0 and j < n and gap <= max_gap:
            out[i:j] = np.linspace(out[i - 1], out[j], gap + 2)[1:-1]
        i = j
    return out


def _baseline_offset(values: np.ndarray, low_limit: float = 0.28) -> float:
    """在早期低位稳定帧中估计站姿基线高度差，容忍片头缺帧。"""
    low = np.flatnonzero(np.isfinite(values) & (np.abs(values) < low_limit))
    if len(low) < 3:
        return 0.0
    return float(np.median(values[low[:20]]))


def _safe_torso(
    lm: np.ndarray, mask: np.ndarray, raw_torso: np.ndarray | None
) -> np.ndarray:
    torso = (
        raw_torso
        if raw_torso is not None
        else torso_length(lm, mask, "visible_side_chain")
    )
    ok = torso > 1e-4
    vals = torso[ok]
    ref = float(np.median(vals)) if len(vals) else 0.25
    filled = _interpolate_gaps(np.where(ok, torso, np.nan), max_gap=6)
    return np.where(np.isfinite(filled) & (filled > 1e-4), filled, ref)


def _kick_and_support_roles(action: str) -> tuple[str, str]:
    if action in ("step_front_low_kick", "front_low_kick"):
        return "front", "rear"
    if action == "rear_low_kick":
        return "rear", "front"
    raise ValueError(f"不支持的低鞭腿动作: {action!r}")


def _normalized_height_diff(
    lm: np.ndarray,
    mask: np.ndarray,
    torso: np.ndarray,
    sup_idx: int,
    kick_idx: int,
    low_limit: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    valid = mask[:, kick_idx] & mask[:, sup_idx]
    raw = np.where(valid, (lm[:, sup_idx, 1] - lm[:, kick_idx, 1]) / torso, np.nan)
    filled = _interpolate_gaps(_median_filter(raw, 3), max_gap=5)
    base = _baseline_offset(filled, low_limit)
    return raw, filled - base, base


def _prepare_signals(
    series: dict,
    action: str,
    stance: str,
    raw_torso: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    lm = np.asarray(series["landmarks"], dtype=float)
    mask = np.asarray(series["valid_mask"], dtype=bool)
    times = np.asarray(series["times"], dtype=float)
    torso = _safe_torso(lm, mask, raw_torso)

    kick_role, sup_role = _kick_and_support_roles(action)
    _, _, _, kh, kk, ka, _, kt = sides_for_stance(kick_role, stance)
    _, _, _, _, sk, sa, _, _ = sides_for_stance(sup_role, stance)

    sup_raw, sup_rel, base_sup = _normalized_height_diff(lm, mask, torso, sa, ka, 0.28)
    abs_fill = _interpolate_gaps(
        _median_filter(np.abs(sup_raw) - abs(base_sup), 3), max_gap=5
    )
    seg_sig = np.where(
        np.isfinite(sup_rel) & np.isfinite(abs_fill),
        np.maximum(sup_rel, abs_fill * 0.9),
        sup_rel,
    )

    _, knee_rel, _ = _normalized_height_diff(lm, mask, torso, sk, kk, 0.28)
    _, toe_rel, _ = _normalized_height_diff(lm, mask, torso, sa, kt, 0.35)
    knee_lift = np.where(
        np.isfinite(knee_rel) & (knee_rel >= _KNEE_FALLBACK_TH), knee_rel, np.nan
    )
    toe_lift = np.where(
        np.isfinite(toe_rel) & (toe_rel >= _TOE_FALLBACK_TH), toe_rel, np.nan
    )
    fallback = np.fmax(knee_lift, toe_lift)

    ankle_missing = ~np.isfinite(seg_sig)
    hold_sig = np.where(ankle_missing, fallback, seg_sig)
    signed_sig = np.where(np.isfinite(sup_rel), sup_rel, fallback)

    k_ang_raw = series_angles(lm, kh, kk, ka)
    k_ang = np.where(mask[:, kh] & mask[:, kk] & mask[:, ka], k_ang_raw, np.nan)
    return {
        "times": times,
        "sup_rel": sup_rel,
        "signed_sig": signed_sig,
        "hold_sig": hold_sig,
        "knee_angle": k_ang,
    }


def _hysteresis_intervals(
    hold_sig: np.ndarray,
    signed_sig: np.ndarray,
    times: np.ndarray,
) -> list[tuple[int, int, bool]]:
    raw_intervals: list[tuple[int, int, bool]] = []
    n = len(hold_sig)
    in_ev = False
    start = 0
    low_run = 0

    for i in range(n):
        val = hold_sig[i]
        s_val = signed_sig[i]
        if not np.isfinite(val):
            if in_ev:
                low_run += 1
                if low_run >= 6:
                    raw_intervals.append((start, i - low_run, False))
                    in_ev = False
                    low_run = 0
            continue
        if not in_ev:
            trig = s_val if np.isfinite(s_val) else val
            if trig >= _ENTER_TH or val >= _PEAK_TH:
                s0 = i
                while (
                    s0 > 0
                    and np.isfinite(hold_sig[s0 - 1])
                    and hold_sig[s0 - 1] > _EXIT_TH
                ):
                    s0 -= 1
                in_ev = True
                start = s0
                low_run = 0
            continue
        if val <= _EXIT_TH:
            low_run += 1
            if low_run >= 2:
                raw_intervals.append((start, i - low_run, True))
                in_ev = False
                low_run = 0
        else:
            low_run = 0

    if in_ev:
        raw_intervals.append((start, n - 1, False))

    merged: list[tuple[int, int, bool]] = []
    for a, b, closed in raw_intervals:
        end_idx = max(a, b)
        if merged and (times[a] - times[merged[-1][1]]) <= _MERGE_GAP_S:
            merged[-1] = (merged[-1][0], end_idx, closed)
        else:
            merged.append((a, end_idx, closed))
    return merged


def _select_ext_idx(
    start: int,
    peak_idx: int,
    seg_h: np.ndarray,
    ka_seg: np.ndarray,
    peak_val: float,
) -> int:
    ext_mask = np.isfinite(seg_h) & (seg_h >= _EXT_RATIO * peak_val)
    if np.isfinite(ka_seg).any():
        ext_k = ext_mask & np.isfinite(ka_seg) & (ka_seg >= _EXT_KNEE_DEG)
        if ext_k.any():
            ext_indices = np.flatnonzero(ext_k)
            return start + int(ext_indices[len(ext_indices) // 3])
    return peak_idx


def segment_kick_events(
    series: dict,
    action: str,
    stance: str = "left",
    prepared=None,
) -> list[dict]:
    """切分侧视角单段低鞭腿序列中的每一次踢腿事件。"""
    times = np.asarray(series["times"], dtype=float)
    n = len(times)
    if n == 0:
        return []

    raw_torso = prepared.torso if prepared is not None else None
    sig = _prepare_signals(series, action, stance, raw_torso=raw_torso)
    hold_sig = sig["hold_sig"]
    signed_sig = sig["signed_sig"]
    sup_rel = sig["sup_rel"]
    knee_angle = sig["knee_angle"]

    events: list[dict] = []
    for a, b, closed in _hysteresis_intervals(hold_sig, signed_sig, times):
        seg_h = hold_sig[a : b + 1]
        seg_s = signed_sig[a : b + 1]
        fin_h = np.flatnonzero(np.isfinite(seg_h))
        if not len(fin_h):
            continue

        peak_val = float(np.nanmax(seg_h))
        signed_peak = float(np.nanmax(seg_s)) if np.isfinite(seg_s).any() else peak_val
        dur = float(times[b] - times[a])
        after = sup_rel[b : min(n, b + 10)]
        post_min = float(np.nanmin(after)) if np.isfinite(after).any() else 0.0
        if peak_val < _PEAK_TH or dur < _MIN_DUR_S:
            continue
        if signed_peak < 0.42 and post_min < -0.25:
            continue

        peak_idx = a + int(fin_h[int(np.argmax(seg_h[fin_h]))])
        ext_idx = _select_ext_idx(a, peak_idx, seg_h, knee_angle[a : b + 1], peak_val)
        after_hold = hold_sig[b + 1 : min(n, b + 30)]
        landed_after = bool((np.isfinite(after_hold) & (after_hold <= _EXIT_TH)).any())
        events.append(
            {
                "start_idx": int(a),
                "peak_idx": int(peak_idx),
                "ext_idx": int(ext_idx),
                "end_idx": int(b),
                "t_start": float(times[a]),
                "t_peak": float(times[peak_idx]),
                "t_ext": float(times[ext_idx]),
                "t_end": float(times[b]),
                "dur_s": dur,
                "peak_sup": peak_val,
                "signed_peak_sup": signed_peak,
                "closed": bool(closed or (b < n - 2 and landed_after)),
            }
        )
    return events


def kick_repetitions(
    series: dict, action: str, stance: str, cfg=None
) -> list[tuple[float, float]]:
    """返回侧视角低鞭腿每次重复的 ``(起, 止)``（秒）。"""
    del cfg
    return [
        (ev["t_start"], ev["t_end"])
        for ev in segment_kick_events(series, action, stance)
    ]
