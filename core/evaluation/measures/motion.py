"""按身体尺度测量关节点相对位置与出拳峰值后滞留时长，供姿势、规则与运动事件共用。"""

from __future__ import annotations

import numpy as np


def relative_position(
    landmarks: np.ndarray,
    valid_mask: np.ndarray,
    scale: np.ndarray,
    anchor: int,
    moving: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回移动点相对参照点的二维位置与有效性，由调用方决定如何聚合。"""
    valid = valid_mask[:, [anchor, moving]].all(axis=1) & (scale > 1e-4)
    relative = (landmarks[:, moving, :2] - landmarks[:, anchor, :2]) / np.maximum(
        scale[:, None], 1e-4
    )
    return relative, valid


def normalized_motion(
    landmarks: np.ndarray,
    valid_mask: np.ndarray,
    torso: np.ndarray,
    anchor: int,
    moving: int,
) -> np.ndarray:
    """运动信号保留无效画面为 NaN，禁止跨缺点画面制造运动。"""
    relative, valid = relative_position(landmarks, valid_mask, torso, anchor, moving)
    return np.where(valid[:, None], relative, np.nan)


def measure_punch_post_peak_hold(
    lm: np.ndarray,
    mask: np.ndarray,
    torso: np.ndarray,
    times: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    base_torso: np.ndarray,
    sh: int,
    wr: int,
    facing: float | None,
    *,
    hold_max: float = 0.45,
    return_reach: float = 0.12,
    held_ending: bool = False,
) -> tuple[bool, np.ndarray, dict[str, tuple[float, str]]]:
    """单次出拳峰值后回收滞留测量：未收回（held）或峰值后超过 hold_max 秒仍未回落判违规。"""
    n = len(lm)
    invalid = np.zeros(n, dtype=bool)
    label = "出拳峰值后滞留时长"
    if n == 0 or facing is None or facing == 0:
        return False, invalid, {label: (float("nan"), "秒")}
    base_ok = baseline_mask[:, [sh, wr]].all(axis=1) & (base_torso > 1e-4)
    if not base_ok.any():
        return False, invalid, {label: (float("nan"), "秒")}
    dir_x = float(facing)
    base_rel, _ = relative_position(
        baseline[base_ok], baseline_mask[base_ok], base_torso[base_ok], sh, wr
    )
    base_reach = float(np.median(base_rel[:, 0] * dir_x))
    rel, valid = relative_position(lm, mask, torso, sh, wr)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return False, valid, {label: (float("nan"), "秒")}
    reach = rel[:, 0] * dir_x - base_reach
    peak_local = int(valid_idx[int(np.argmax(reach[valid_idx]))])
    ret_idx = int(valid_idx[-1])
    for k in valid_idx:
        if k > peak_local and reach[k] <= return_reach:
            ret_idx = int(k)
            break
    hold_s = max(0.0, float(times[ret_idx] - times[peak_local]))
    bad = bool(held_ending or hold_s > hold_max)
    return bad, valid, {label: (hold_s, "秒")}
