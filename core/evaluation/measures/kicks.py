# -*- coding: utf-8 -*-
"""低鞭腿与垫步独有底层几何测量及抗侧视左右腿互换掩码。"""

from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.feet import foot_forward_progression, foot_settle_index
from core.evaluation.measures.posture import joint_angle, relative_height


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
    rear_ankle = sides_for_stance("rear", stance)[5]
    prog = foot_forward_progression(lm, mask, times_or_dt, rear_ankle, facing)
    if not prog.ok or len(prog.times) < 2:
        return (
            False,
            np.zeros(len(lm), dtype=bool),
            {"垫步落稳后停顿": (float("nan"), "秒")},
        )
    if prog.displacement < min_disp:
        return False, prog.valid_mask, {"垫步落稳后停顿": (0.0, "秒")}
    settle_idx = foot_settle_index(prog)
    pause_s = max(0.0, float(prog.times[-1] - prog.times[settle_idx]))
    return bool(pause_s > pause_max), prog.valid_mask, {"垫步落稳后停顿": (pause_s, "秒")}


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
