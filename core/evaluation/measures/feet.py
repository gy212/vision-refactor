# -*- coding: utf-8 -*-
"""脚部公共测量：实战式后脚外撇、支撑脚碾脚内旋、出拳提踵蹬转、步法推进/越前/并脚与落稳时刻。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.evaluation.body_sides import facing_direction
from core.evaluation.measures.arms import segment_lengths
from core.evaluation.measures.torso import chain_torso, shoulder_width
from core.evaluation.paired_torso import FRONTAL_PROJECTION_RATIO
from core.pose.landmarks import (
    L_FOOT_INDEX,
    L_HEEL,
    L_SHOULDER,
    R_FOOT_INDEX,
    R_HEEL,
    R_SHOULDER,
)

FEET = (L_HEEL, L_FOOT_INDEX, R_HEEL, R_FOOT_INDEX)


@dataclass(frozen=True)
class FootProgression:
    """单只脚沿攻击方向平滑与尺度归一化后的推进序列。"""

    ok: bool
    valid_mask: np.ndarray
    displacement: float
    smoothed_x: np.ndarray
    times: np.ndarray
    torso: np.ndarray
    start_idx: int = 0
    peak_idx: int = 0


def foot_heel_toe_forward_projection(
    lm: np.ndarray,
    mask: np.ndarray,
    torso: np.ndarray,
    heel: int,
    toe: int,
    facing: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """计算单脚脚跟相对脚尖沿攻击方向的前向投影 ``(heel_x - toe_x) * facing / torso`` 及有效掩码。"""
    dir_x = float(facing) if facing else 1.0
    valid = (
        mask[:, [heel, toe]].all(axis=1) & (torso > 1e-4) & bool(facing is not None and facing != 0)
    )
    safe_t = np.where(torso > 1e-4, torso, 1.0)
    proj = (lm[:, heel, 0] - lm[:, toe, 0]) * dir_x / safe_t
    return proj, valid


def measure_rear_foot_splay(
    lm: np.ndarray,
    mask: np.ndarray,
    torso: np.ndarray,
    rear_heel: int,
    rear_toe: int,
    facing: float | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """实战式后脚外撇判定（侧面）：后脚尖落后于后脚跟（投影 < 0）判外撇过度，不查前脚。"""
    heel_proj, valid = foot_heel_toe_forward_projection(
        lm, mask, torso, rear_heel, rear_toe, facing
    )
    toe_proj = -heel_proj + 0.0
    return toe_proj < 0.0, valid, {"后脚尖相对后脚跟前向投影": (toe_proj, "躯干长度")}


def measure_pivot_heel(
    lm: np.ndarray,
    safe_mask: np.ndarray,
    torso: np.ndarray,
    sup_heel: int,
    sup_toe: int,
    facing: float | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """支撑脚内旋碾脚：以前脚掌为轴内旋，脚跟未领先于脚尖（投影 <= 0）判未达成帧。"""
    proj, valid = foot_heel_toe_forward_projection(lm, safe_mask, torso, sup_heel, sup_toe, facing)
    return proj <= 0.0, valid, {"支撑脚跟相对脚尖前向投影": (proj, "躯干长度")}


def measure_punch_heel_pivot(
    lm: np.ndarray,
    mask: np.ndarray,
    torso: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    base_torso: np.ndarray,
    heel: int,
    toe: int,
    *,
    min_lift: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """拳法发力脚蹬转提踵（absence 规则）：相对起始护位提踵增量不足 min_lift 判未达成帧。"""
    base_ok = (
        baseline_mask[:, [heel, toe]].all(axis=1)
        & (base_torso > 1e-4)
        & (segment_lengths(baseline, heel, toe) > 1e-4)
    )
    if not base_ok.any():
        empty = np.zeros(len(lm), dtype=bool)
        return empty, empty.copy(), {}
    base_lift = (baseline[base_ok, toe, 1] - baseline[base_ok, heel, 1]) / base_torso[base_ok]
    base_med = float(np.median(base_lift))
    cur_ok = (
        mask[:, [heel, toe]].all(axis=1) & (torso > 1e-4) & (segment_lengths(lm, heel, toe) > 1e-4)
    )
    safe_t = np.where(torso > 1e-4, torso, 1.0)
    delta = (lm[:, toe, 1] - lm[:, heel, 1]) / safe_t - base_med
    return delta < min_lift, cur_ok, {"发力脚相对基线提踵增量": (delta, "躯干长度")}


def _resolve_times(n_frames: int, times_or_dt: np.ndarray | float | None) -> np.ndarray:
    """将输入时间戳数组或采样步长规整为一维时间序列。"""
    if isinstance(times_or_dt, np.ndarray) and times_or_dt.shape == (n_frames,):
        return np.asarray(times_or_dt, dtype=float)
    dt = float(times_or_dt) if times_or_dt is not None else 1.0 / 30.0
    if not np.isfinite(dt) or dt <= 1e-4:
        dt = 1.0 / 30.0
    return np.arange(n_frames, dtype=float) * dt


def _infer_facing(lm: np.ndarray, mask: np.ndarray) -> float | None:
    """从窗口逐帧统计面朝方向，返回 +1.0（朝右）或 -1.0（朝左）。"""
    dirs = [
        d for row, mask_row in zip(lm, mask) if (d := facing_direction(row, mask_row)) is not None
    ]
    if not dirs:
        return None
    return 1.0 if sum(dirs) >= 0.0 else -1.0


def _median_filter_1d(arr: np.ndarray, k: int = 3) -> np.ndarray:
    """对有限值做局部中值滤波，抑制单帧标签抖动。"""
    out = arr.copy()
    half = k // 2
    n = len(arr)
    for i in range(n):
        win = arr[max(0, i - half) : min(n, i + half + 1)]
        fin = win[np.isfinite(win)]
        if len(fin) >= 2:
            out[i] = float(np.median(fin))
    return out


def _interpolate_short_gaps(arr: np.ndarray, max_gap: int = 4) -> np.ndarray:
    """仅对不超过 max_gap 帧的短暂缺失做线性插值，长缺口保留 NaN。"""
    out = arr.copy()
    fin = np.isfinite(out)
    if fin.sum() < 2:
        return out
    n = len(out)
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


def foot_forward_progression(
    lm: np.ndarray,
    mask: np.ndarray,
    times_or_dt: np.ndarray | float | None,
    ankle: int,
    facing: float | None,
    *,
    min_points: int = 5,
) -> FootProgression:
    """计算指定脚踝向攻击方向平滑后的序列、有效掩码、推进幅度与时间轴。"""
    n = len(lm)
    invalid = np.zeros(n, dtype=bool)
    empty = np.array([], dtype=float)
    dir_x = _infer_facing(lm, mask) if facing is None else facing
    if n == 0 or dir_x is None or dir_x == 0:
        return FootProgression(False, invalid, 0.0, empty, empty, empty)
    torso = chain_torso(lm, mask)
    ok = mask[:, ankle] & (torso > 1e-4)
    if int(ok.sum()) < min_points:
        return FootProgression(False, invalid, 0.0, empty, empty, empty)
    # 先在原始坐标做平滑并求窗内净位移，再用整个窗口统一的躯干中位尺度归一化。
    ref_torso = float(np.median(torso[ok]))
    raw_x = np.where(ok, lm[:, ankle, 0] * float(dir_x), np.nan)
    x_sm = _median_filter_1d(_interpolate_short_gaps(raw_x, max_gap=4), k=3)
    fin_idx = np.flatnonzero(np.isfinite(x_sm))
    if fin_idx.size == 0:
        return FootProgression(False, invalid, 0.0, empty, empty, empty)
    # 按有限值前缀最小值求最大正向净推进，避免尾窗后撤复位拉低起算点导致位移归零。
    best_start = int(fin_idx[0])
    best_end = int(fin_idx[0])
    best_gain = 0.0
    min_idx = int(fin_idx[0])
    min_val = float(x_sm[min_idx])
    for k in fin_idx:
        idx = int(k)
        val = float(x_sm[idx])
        if val < min_val:
            min_val = val
            min_idx = idx
        gain = val - min_val
        if gain > best_gain:
            best_gain = gain
            best_start = min_idx
            best_end = idx
    rx_sm = (x_sm - float(x_sm[best_start])) / ref_torso
    disp = float(best_gain / ref_torso)
    times = _resolve_times(n, times_or_dt)
    return FootProgression(True, ok, disp, rx_sm, times, torso, best_start, best_end)


def _first_planted_plateau(
    fin_idx: np.ndarray,
    rx_sm: np.ndarray,
    vx_eff: np.ndarray,
    times: np.ndarray,
    min_progress: float,
    peak_time: float,
) -> int | None:
    """定位推进后首次跨过出拳峰值且持续静止的落稳平台，避免窗尾下一次预迈步抬高目标位置。"""
    n_fin = len(fin_idx)
    for pos in range(n_fin):
        k = int(fin_idx[pos])
        if float(rx_sm[k]) < min_progress or abs(float(vx_eff[k])) >= 0.08:
            continue
        run_min = run_max = float(rx_sm[k])
        end_pos = pos
        while end_pos + 1 < n_fin:
            nxt = int(fin_idx[end_pos + 1])
            if abs(float(vx_eff[nxt])) >= 0.08:
                break
            val = float(rx_sm[nxt])
            cand_min = min(run_min, val)
            cand_max = max(run_max, val)
            if cand_max - cand_min > 0.015:
                break
            run_min, run_max = cand_min, cand_max
            end_pos += 1
        t_start = float(times[k])
        t_end = float(times[int(fin_idx[end_pos])])
        if end_pos - pos + 1 >= 3 and t_end - t_start >= 0.20 - 1e-9 and t_end >= peak_time - 1e-9:
            return k
    return None


def foot_settle_index(
    progression: FootProgression,
    *,
    peak_time: float | None = None,
) -> int:
    """在单步前推序列中定位速度峰值后首次落稳的有限帧下标。"""
    rx_sm = progression.smoothed_x
    times = progression.times
    fin_idx = np.flatnonzero(np.isfinite(rx_sm))
    if fin_idx.size == 0:
        return 0
    if len(times) < 2 or fin_idx.size < 2:
        return int(fin_idx[-1])
    dt = np.gradient(times)
    pos_dt = dt[dt > 1e-4]
    fallback_dt = float(np.median(pos_dt)) if pos_dt.size else 1.0 / 30.0
    dt_safe = np.where(dt > 1e-4, dt, fallback_dt)
    vx = np.gradient(rx_sm) / dt_safe
    # 长缺口孤立出的有限帧会导致 np.gradient 为 NaN，用有限帧子序列前向差分补齐速度。
    f_rx = rx_sm[fin_idx]
    f_t = times[fin_idx]
    fwd_dt = np.maximum(np.diff(f_t), 1e-4)
    fwd_vx = np.concatenate((np.diff(f_rx) / fwd_dt, [0.0]))
    vx_eff = vx.copy()
    miss_fin = ~np.isfinite(vx_eff[fin_idx])
    if miss_fin.any():
        vx_eff[fin_idx[miss_fin]] = fwd_vx[miss_fin]
    start_idx = progression.start_idx
    peak_idx = progression.peak_idx
    plateau_idx = None
    if peak_time is not None:
        min_progress = max(0.15, 0.5 * progression.displacement)
        plateau_idx = _first_planted_plateau(
            fin_idx[fin_idx >= start_idx], rx_sm, vx_eff, times, min_progress, peak_time
        )
    cap_idx = plateau_idx if plateau_idx is not None else peak_idx
    step_fin = (
        fin_idx[(fin_idx >= start_idx) & (fin_idx <= cap_idx)] if cap_idx > start_idx else fin_idx
    )
    if step_fin.size == 0:
        step_fin = fin_idx
    pk_v_idx = int(step_fin[int(np.argmax(vx_eff[step_fin]))])
    tail_fin = (
        fin_idx[(fin_idx >= pk_v_idx) & (fin_idx <= plateau_idx)]
        if plateau_idx is not None and plateau_idx >= pk_v_idx
        else fin_idx[fin_idx >= pk_v_idx]
    )
    rx_target = float(np.max(rx_sm[tail_fin]))
    for idx in tail_fin:
        k = int(idx)
        if (rx_target - float(rx_sm[k])) <= 0.06 and float(vx_eff[k]) < 0.40:
            return k
    return int(fin_idx[-1])


def measure_foot_forward_displacement(
    lm: np.ndarray,
    mask: np.ndarray,
    times_or_dt: np.ndarray | float | None,
    ankle: int,
    *,
    min_disp: float = 0.45,
    facing: float | None = None,
    label: str = "后脚前推距离",
    min_points: int = 5,
) -> tuple[bool, np.ndarray, dict[str, tuple[float, str]]]:
    """公共步法推进测量（垫步缺失与上步跟步共用）：净推进低于 min_disp 判违规。"""
    prog = foot_forward_progression(lm, mask, times_or_dt, ankle, facing, min_points=min_points)
    if not prog.ok:
        return False, prog.valid_mask, {label: (float("nan"), "躯干长度")}
    return (
        bool(prog.displacement < min_disp),
        prog.valid_mask,
        {label: (prog.displacement, "躯干长度")},
    )


def measure_rear_foot_overtake(
    lm: np.ndarray,
    mask: np.ndarray,
    rear_ankle: int,
    front_ankle: int,
    *,
    overtake_max: float = 0.20,
    facing: float | None = None,
    mode: str = "tail_median",
    label: str = "后脚越前距离",
) -> tuple[bool, np.ndarray, dict[str, tuple[float, str]]]:
    """公共后脚超越前脚测量：tail_median 取窗尾中位（垫步），peak_median 取窗内最高值中位（跟步）。"""
    n = len(lm)
    invalid = np.zeros(n, dtype=bool)
    dir_x = _infer_facing(lm, mask) if facing is None else facing
    if n == 0 or dir_x is None or dir_x == 0:
        return False, invalid, {label: (float("nan"), "躯干长度")}
    full_torso = chain_torso(lm, mask)
    full_ok = mask[:, rear_ankle] & mask[:, front_ankle] & (full_torso > 1e-4)
    if mode == "tail_median":
        lo = max(0, n - 10)
        ok = full_ok[lo:]
        if not ok.any():
            return False, invalid, {label: (float("nan"), "躯干长度")}
        torso = full_torso[lo:]
        diff = (
            (lm[lo:, rear_ankle, 0] - lm[lo:, front_ankle, 0])
            * float(dir_x)
            / np.where(torso > 1e-4, torso, 1.0)
        )
        vals = diff[ok]
        metric = float(np.median(vals[-min(4, len(vals)) :]))
    else:
        if not full_ok.any():
            return False, invalid, {label: (float("nan"), "躯干长度")}
        diff = (
            (lm[:, rear_ankle, 0] - lm[:, front_ankle, 0])
            * float(dir_x)
            / np.where(full_torso > 1e-4, full_torso, 1.0)
        )
        vals = np.sort(diff[full_ok])
        metric = float(np.median(vals[-min(4, len(vals)) :]))
    return bool(metric > overtake_max), full_ok, {label: (metric, "躯干长度")}


def measure_step_feet_together(
    lm: np.ndarray,
    mask: np.ndarray,
    paired_torso: tuple[np.ndarray, np.ndarray],
    indices: np.ndarray,
    *,
    min_ratio: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """双摄协同上步并脚判定：仅在双摄确认的侧身帧（正/侧肩宽比 <= √3）检查两脚跟水平距。"""
    ratios, observed = paired_torso
    obs = observed[indices]
    ratio = ratios[indices]
    side_facing = obs & np.isfinite(ratio) & (ratio <= FRONTAL_PROJECTION_RATIO)
    side_sh = shoulder_width(lm)
    heel_dx = np.abs(lm[:, L_HEEL, 0] - lm[:, R_HEEL, 0])
    feet_ratio = heel_dx / np.maximum(side_sh, 1e-4)
    valid = (
        side_facing
        & mask[:, [L_HEEL, R_HEEL, L_SHOULDER, R_SHOULDER]].all(axis=1)
        & (side_sh > 1e-4)
    )
    return feet_ratio < min_ratio, valid, {"侧身帧两脚跟水平距与侧肩宽比": (feet_ratio, "倍侧肩宽")}


def _single_foot_lag(
    lm: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    ankle: int,
    peak: float,
    *,
    facing: float | None,
    min_step_disp: float,
    cutoff_after_peak: float | None = None,
    min_valid_ratio: float = 0.6,
) -> tuple[bool, float, np.ndarray, np.ndarray]:
    """测量单只脚相对对应出拳峰值的落稳正向滞后，返回 (是否可测, 滞后秒数, 覆盖掩码, 证据掩码)。"""
    n = len(lm)
    empty = np.zeros(n, dtype=bool)
    torso = chain_torso(lm, mask)
    foot_valid = mask[:, ankle] & (torso > 1e-4)
    used_cut = False
    in_cut = np.ones(n, dtype=bool)
    if cutoff_after_peak is not None:
        cut = np.flatnonzero(times <= peak + cutoff_after_peak)
        if cut.size >= 5:
            used_cut = True
            in_cut = np.zeros(n, dtype=bool)
            in_cut[cut] = True
            prog = foot_forward_progression(lm[cut], mask[cut], times[cut], ankle, facing)
        else:
            prog = foot_forward_progression(lm, mask, times, ankle, facing)
    else:
        prog = foot_forward_progression(lm, mask, times, ankle, facing)
    if not prog.ok:
        return False, float("nan"), empty, empty
    if used_cut and float(prog.valid_mask.mean()) < min_valid_ratio:
        return False, float("nan"), empty, empty
    lag = 0.0
    if prog.displacement >= min_step_disp:
        settle_t = float(prog.times[foot_settle_index(prog, peak_time=float(peak))])
        lag = max(0.0, settle_t - float(peak))
    if used_cut:
        evidence_mask = foot_valid & in_cut
        coverage_mask = evidence_mask | ~in_cut
    else:
        evidence_mask = foot_valid
        coverage_mask = foot_valid
    return True, lag, coverage_mask, evidence_mask


def measure_step_landing_frames(
    lm: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    front_ankle: int,
    rear_ankle: int,
    *,
    front_peak: float | None,
    rear_peak: float | None,
    max_lag: float = 0.20,
    min_step_disp: float = 0.15,
    facing: float | None = None,
    min_valid_ratio: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[float, str]]]:
    """脚步落稳同步帧级测量：返回 (违规证据帧掩码, 有效覆盖掩码, 测量明细)，隔离截断尾窗无关遮挡。"""
    n = len(lm)
    invalid = np.zeros(n, dtype=bool)
    label = "脚步落稳滞后出拳峰值"
    if n < 2 or (front_peak is None and rear_peak is None):
        return invalid, invalid.copy(), {label: (float("nan"), "秒")}
    measured: list[tuple[bool, float, np.ndarray, np.ndarray]] = []
    if front_peak is not None:
        measured.append(
            _single_foot_lag(
                lm,
                mask,
                times,
                front_ankle,
                front_peak,
                facing=facing,
                min_step_disp=min_step_disp,
                cutoff_after_peak=0.60,
                min_valid_ratio=min_valid_ratio,
            )
        )
    if rear_peak is not None:
        measured.append(
            _single_foot_lag(
                lm,
                mask,
                times,
                rear_ankle,
                rear_peak,
                facing=facing,
                min_step_disp=min_step_disp,
                min_valid_ratio=min_valid_ratio,
            )
        )
    violated = [(lag, cov, ev) for ok, lag, cov, ev in measured if ok and lag > max_lag]
    if violated:
        valid_mask = np.zeros(n, dtype=bool)
        bad_mask = np.zeros(n, dtype=bool)
        for _, cov, ev in violated:
            valid_mask |= cov
            bad_mask |= ev
        return bad_mask, valid_mask, {label: (max(lag for lag, _, _ in violated), "秒")}
    if any(not ok for ok, _, _, _ in measured):
        return invalid, invalid.copy(), {label: (float("nan"), "秒")}
    pass_mask = np.ones(n, dtype=bool)
    for _, _, cov, _ in measured:
        pass_mask &= cov
    max_observed = max(lag for _, lag, _, _ in measured)
    return invalid, pass_mask, {label: (max_observed, "秒")}


def measure_step_landing_lag(
    lm: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    front_ankle: int,
    rear_ankle: int,
    *,
    front_peak: float | None,
    rear_peak: float | None,
    max_lag: float = 0.20,
    min_step_disp: float = 0.15,
    facing: float | None = None,
    min_valid_ratio: float = 0.6,
) -> tuple[bool, np.ndarray, dict[str, tuple[float, str]]]:
    """脚步落稳同步测量：对比前后脚落稳时刻与对应出拳峰值时刻的最大正向滞后。"""
    bad_mask, valid_mask, details = measure_step_landing_frames(
        lm,
        mask,
        times,
        front_ankle,
        rear_ankle,
        front_peak=front_peak,
        rear_peak=rear_peak,
        max_lag=max_lag,
        min_step_disp=min_step_disp,
        facing=facing,
        min_valid_ratio=min_valid_ratio,
    )
    return bool(bad_mask.any()), valid_mask, details
