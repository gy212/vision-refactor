"""低鞭腿与垫步独有底层几何测量及抗侧视左右腿互换掩码。"""

from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import facing_direction, sides_for_stance
from core.evaluation.measures.posture import joint_angle, relative_height
from core.evaluation.measures.torso import chain_torso


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


def _rear_ankle_progression(
    lm: np.ndarray,
    mask: np.ndarray,
    times_or_dt: np.ndarray | float | None,
    stance: str,
    facing: float | None,
) -> tuple[bool, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """计算后脚踝向攻击方向清理后的序列、有效帧掩码、推进幅度和时域数组。"""
    n = len(lm)
    invalid = np.zeros(n, dtype=bool)
    empty = np.array([], dtype=float)
    dir_x = _infer_facing(lm, mask) if facing is None else facing
    if n == 0 or dir_x is None or dir_x == 0:
        return False, invalid, 0.0, empty, empty, empty
    _, _, _, _, _, rear_ankle, _, _ = sides_for_stance("rear", stance)
    torso = chain_torso(lm, mask)
    ok = mask[:, rear_ankle] & (torso > 1e-4)
    if int(ok.sum()) < 5:
        return False, invalid, 0.0, empty, empty, empty
    # 先在原始坐标做平滑并求窗内净位移，再用整个窗口统一的躯干中位尺度归一化。
    ref_torso = float(np.median(torso[ok]))
    raw_x = np.where(ok, lm[:, rear_ankle, 0] * float(dir_x), np.nan)
    x_sm = _median_filter_1d(_interpolate_short_gaps(raw_x, max_gap=4), k=3)
    if not np.isfinite(x_sm).any():
        return False, invalid, 0.0, empty, empty, empty
    idx_min = int(np.nanargmin(x_sm))
    rx_sm = (x_sm - float(x_sm[idx_min])) / ref_torso
    pad_disp = float(np.nanmax(rx_sm[idx_min:]))
    times = _resolve_times(n, times_or_dt)
    return True, ok, pad_disp, rx_sm, times, torso


def kicking_leg_side_safe(
    lm: np.ndarray,
    mask: np.ndarray,
    kick_ankle: int,
    sup_ankle: int,
    torso: np.ndarray,
    *,
    majority_threshold: float | None = None,
) -> np.ndarray:
    """侧视下肢抗左右腿标签互换：单帧翻转置 False；finish 超半数翻转整窗置 False。"""
    safe_mask = mask.copy()
    if len(lm) == 0:
        return safe_mask
    both = mask[:, kick_ankle] & mask[:, sup_ankle] & (torso > 1e-4)
    safe_t = np.where(torso > 1e-4, torso, 1.0)
    swapped = np.zeros(len(lm), dtype=bool)
    swapped[both] = ((lm[both, sup_ankle, 1] - lm[both, kick_ankle, 1]) / safe_t[both]) < -0.15
    safe_mask[swapped, 23:33] = False
    if (
        majority_threshold is not None
        and bool(both.any())
        and float(np.mean(swapped[both])) > majority_threshold
    ):
        safe_mask[:, 23:33] = False
    return safe_mask


def measure_pad_step_missing(
    lm: np.ndarray,
    mask: np.ndarray,
    times_or_dt: np.ndarray | float | None,
    stance: str,
    min_disp: float = 0.45,
    overtake_max: float = 0.20,
    pause_max: float = 0.60,
    *,
    facing: float | None = None,
) -> tuple[bool, np.ndarray, dict[str, tuple[float, str]]]:
    """垫步缺失判定：起腿前窗口后脚最大正向推进不足 min_disp 判违规，并返回逐帧有效掩码。"""
    del overtake_max, pause_max
    ok, valid_mask, pad_disp, _, _, _ = _rear_ankle_progression(
        lm, mask, times_or_dt, stance, facing
    )
    if not ok:
        return False, valid_mask, {"垫步后脚前推距离": (float("nan"), "躯干长度")}
    return (
        bool(pad_disp < min_disp),
        valid_mask,
        {"垫步后脚前推距离": (pad_disp, "躯干长度")},
    )


def measure_pad_step_overtake(
    lm: np.ndarray,
    mask: np.ndarray,
    times_or_dt: np.ndarray | float | None,
    stance: str,
    min_disp: float = 0.45,
    overtake_max: float = 0.20,
    pause_max: float = 0.60,
    *,
    facing: float | None = None,
) -> tuple[bool, np.ndarray, dict[str, tuple[float, str]]]:
    """垫步后脚超越前脚判定：临起脚末端双脚踝差值中位超过 overtake_max 判违规，并返回逐帧有效掩码。"""
    del times_or_dt, min_disp, pause_max
    n = len(lm)
    invalid = np.zeros(n, dtype=bool)
    dir_x = _infer_facing(lm, mask) if facing is None else facing
    if n == 0 or dir_x is None or dir_x == 0:
        return False, invalid, {"垫步末端后脚越前距离": (float("nan"), "躯干长度")}
    _, _, _, _, _, rear_ankle, _, _ = sides_for_stance("rear", stance)
    _, _, _, _, _, front_ankle, _, _ = sides_for_stance("front", stance)
    full_torso = chain_torso(lm, mask)
    full_ok = mask[:, rear_ankle] & mask[:, front_ankle] & (full_torso > 1e-4)
    lo = max(0, n - 10)
    ok = full_ok[lo:]
    if not ok.any():
        return False, invalid, {"垫步末端后脚越前距离": (float("nan"), "躯干长度")}
    torso = full_torso[lo:]
    diff = (
        (lm[lo:, rear_ankle, 0] - lm[lo:, front_ankle, 0])
        * float(dir_x)
        / np.where(torso > 1e-4, torso, 1.0)
    )
    vals = diff[ok]
    end_med = float(np.median(vals[-min(4, len(vals)) :]))
    return (
        bool(end_med > overtake_max),
        full_ok,
        {"垫步末端后脚越前距离": (end_med, "躯干长度")},
    )


def measure_pad_step_pause(
    lm: np.ndarray,
    mask: np.ndarray,
    times_or_dt: np.ndarray | float | None,
    stance: str,
    min_disp: float = 0.45,
    overtake_max: float = 0.20,
    pause_max: float = 0.60,
    *,
    facing: float | None = None,
) -> tuple[bool, np.ndarray, dict[str, tuple[float, str]]]:
    """垫步后停顿判定：仅当有实质垫步且落稳到起脚超过 pause_max 秒判违规，并返回逐帧有效掩码。"""
    del overtake_max
    ok, valid_mask, pad_disp, rx_sm, times, _ = _rear_ankle_progression(
        lm, mask, times_or_dt, stance, facing
    )
    if not ok or len(times) < 2:
        return (
            False,
            np.zeros(len(lm), dtype=bool),
            {"垫步落稳后停顿": (float("nan"), "秒")},
        )
    if pad_disp < min_disp:
        return False, valid_mask, {"垫步落稳后停顿": (0.0, "秒")}
    dt = np.gradient(times)
    pos_dt = dt[dt > 1e-4]
    fallback_dt = float(np.median(pos_dt)) if pos_dt.size else 1.0 / 30.0
    dt_safe = np.where(dt > 1e-4, dt, fallback_dt)
    vx = np.gradient(rx_sm) / dt_safe
    pk_v_idx = int(np.nanargmax(vx)) if np.isfinite(vx).any() else len(rx_sm) - 1
    rx_target = float(np.nanmax(rx_sm[pk_v_idx:]))
    settle_idx = len(rx_sm) - 1
    for k in range(pk_v_idx, len(rx_sm)):
        if np.isfinite(rx_sm[k]) and np.isfinite(vx[k]):
            if (rx_target - rx_sm[k]) <= 0.06 and vx[k] < 0.40:
                settle_idx = k
                break
    pause_s = max(0.0, float(times[-1] - times[settle_idx]))
    return bool(pause_s > pause_max), valid_mask, {"垫步落稳后停顿": (pause_s, "秒")}


def measure_kick_height(
    lm: np.ndarray,
    safe_mask: np.ndarray,
    torso: np.ndarray,
    kick_toe: int,
    kick_hip: int,
    sup_knee: int,
    hip_margin: float = 0.06,
    knee_margin: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """低鞭腿击打高度双边界：不高过同侧髋，且不明显低于支撑膝。"""
    toe_hip_dy, valid_hip = relative_height(lm, safe_mask, kick_toe, kick_hip, torso)
    toe_knee_dy, valid_knee = relative_height(lm, safe_mask, kick_toe, sup_knee, torso)
    valid = valid_hip & valid_knee
    bad = (toe_hip_dy < -hip_margin) | (toe_knee_dy > knee_margin)
    details = {
        "脚尖相对同侧髋高度": (toe_hip_dy, "躯干长度"),
        "脚尖相对支撑膝高度": (toe_knee_dy, "躯干长度"),
    }
    return bad, valid, details


def measure_kick_retract_segment(
    lm: np.ndarray,
    safe_mask: np.ndarray,
    torso: np.ndarray,
    kick_ankle: int,
    kick_hip: int,
    facing: float | None,
    retract_max_x: float = 0.40,
) -> tuple[bool, np.ndarray, dict[str, tuple[float, str]]]:
    """收腿回屈检查：单次 retract 窗落地前踝相对髋最小前伸仍大于阈值判未收腿，并返回逐帧有效掩码。"""
    ok = safe_mask[:, kick_ankle] & safe_mask[:, kick_hip] & (torso > 1e-4)
    if facing is None or facing == 0 or int(ok.sum()) < 3:
        return (
            False,
            np.zeros(len(lm), dtype=bool),
            {"落地前最小踝髋前伸": (float("nan"), "躯干长度")},
        )
    safe_t = np.where(torso > 1e-4, torso, 1.0)
    xs = (lm[ok, kick_ankle, 0] - lm[ok, kick_hip, 0]) * float(facing) / safe_t[ok]
    min_x = float(np.min(xs))
    return bool(min_x > retract_max_x), ok, {"落地前最小踝髋前伸": (min_x, "躯干长度")}


def measure_foot_flexed(
    lm: np.ndarray,
    safe_mask: np.ndarray,
    torso: np.ndarray,
    kick_knee: int,
    kick_ankle: int,
    kick_toe: int,
    min_angle: float = 165.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """踢腿脚背绷直：踢腿侧膝-踝-脚尖夹角低于阈值判脚背未绷直。"""
    angles, valid = joint_angle(lm, safe_mask, (kick_knee, kick_ankle, kick_toe), torso)
    return angles < min_angle, valid, {"踢腿踝背伸角": (angles, "度")}


def measure_arm_swing_down(
    lm: np.ndarray,
    mask: np.ndarray,
    torso: np.ndarray,
    kick_shoulder: int,
    kick_elbow: int,
    kick_wrist: int,
    min_drop: float = 0.45,
    min_angle: float = 115.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """同侧顺势向下甩臂（absence 规则）：未下伸或肘部明显屈曲判未做到位帧。"""
    drop, valid_drop = relative_height(lm, mask, kick_wrist, kick_shoulder, torso)
    angles, valid_ang = joint_angle(lm, mask, (kick_shoulder, kick_elbow, kick_wrist), torso)
    valid = valid_drop & valid_ang
    bad = (drop < min_drop) | (angles < min_angle)
    details = {
        "踢腿侧腕相对肩下伸": (drop, "躯干长度"),
        "踢腿侧臂伸展角": (angles, "度"),
    }
    return bad, valid, details
