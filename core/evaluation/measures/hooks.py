# -*- coding: utf-8 -*-
"""摆拳独有底层几何测量：大小臂弯曲投影比、正面小臂水平角与中线越界比例。"""

from __future__ import annotations

import numpy as np

from core.evaluation.measures.arms import segment_lengths


def measure_hook_arm_bend(
    lm: np.ndarray,
    mask: np.ndarray,
    torso: np.ndarray,
    sh: int,
    el: int,
    wr: int,
    facing: float | None,
    *,
    tight_max: float = 0.50,
    straight_max: float = 0.70,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """摆拳打到瞬间大小臂弯曲度：侧面小臂投影/大臂长，腕在肘后>tight_max 或腕在肘前>straight_max 判违规。"""
    facing_ok = facing is not None and facing != 0
    dir_x = float(facing) if facing_ok else 1.0
    upper = segment_lengths(lm, sh, el)
    forearm = segment_lengths(lm, el, wr)
    safe_upper = np.where(upper > 1e-4, upper, 1.0)
    safe_torso = np.where(torso > 1e-4, torso, 1.0)
    ratio = forearm / safe_upper
    forward_dx = (lm[:, wr, 0] - lm[:, el, 0]) * dir_x
    wrist_behind = forward_dx < 0.0
    bad = np.where(wrist_behind, ratio > tight_max, ratio > straight_max)
    valid = mask[:, [sh, el, wr]].all(axis=1) & (torso > 1e-4) & (upper > 1e-4) & bool(facing_ok)
    details = {
        "侧面小臂与大臂投影比": (ratio, "倍大臂"),
        "腕相对肘前向偏移": (forward_dx / safe_torso, "躯干长度"),
    }
    return bad, valid, details


def measure_hook_forearm_level(
    lm: np.ndarray,
    mask: np.ndarray,
    el: int,
    wr: int,
    *,
    max_angle: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """摆拳正面小臂水平度：打到瞬间正面小臂（肘→腕）与水平线夹角超过 max_angle 度判违规。"""
    dx = np.abs(lm[:, wr, 0] - lm[:, el, 0])
    dy = np.abs(lm[:, wr, 1] - lm[:, el, 1])
    forearm = segment_lengths(lm, el, wr)
    angles = np.degrees(np.arctan2(dy, dx))
    valid = mask[:, [el, wr]].all(axis=1) & (forearm > 1e-4)
    return angles > max_angle, valid, {"正面小臂与水平线夹角": (angles, "度")}


def measure_hook_over_center(
    lm: np.ndarray,
    mask: np.ndarray,
    wr: int,
    att_sh: int,
    opp_sh: int,
    *,
    max_ratio: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """摆拳单侧防摆过头：正面手腕越过两肩中点到达对侧方向超过 max_ratio 倍正面肩宽判违规。"""
    mid_x = 0.5 * (lm[:, att_sh, 0] + lm[:, opp_sh, 0])
    sh_dx = lm[:, opp_sh, 0] - lm[:, att_sh, 0]
    sh_width = segment_lengths(lm, att_sh, opp_sh)
    inward_sign = np.sign(sh_dx)
    safe_width = np.where(sh_width > 1e-4, sh_width, 1.0)
    ratio = (lm[:, wr, 0] - mid_x) * inward_sign / safe_width
    valid = mask[:, [wr, att_sh, opp_sh]].all(axis=1) & (np.abs(sh_dx) > 1e-4) & (sh_width > 1e-4)
    return ratio > max_ratio, valid, {"正面手腕越过两肩中点比例": (ratio, "倍肩宽")}
