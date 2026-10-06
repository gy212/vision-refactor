# -*- coding: utf-8 -*-
"""单帧"正面程度"视角分数 ``pose_view_score``（正侧切分与模板视角判定用）。输入为 landmark 对象或 ``(33, 4)`` 数组行。"""
from __future__ import annotations

import numpy as np

def pose_view_score(pose_landmarks) -> float | None:
    """
    Heuristic "frontness" score for a single frame.

    Higher => more likely facing the camera (front view).
    Lower  => more likely side view.

    This uses only a few stable BlazePose landmarks (shoulders/hips) and combines:
      - normalized shoulder width (relative to torso length)
      - left/right visibility balance (front tends to be more symmetric)
    """
    if pose_landmarks is None:
        return None

    lm = pose_landmarks

    # BlazePose indices
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24

    if isinstance(lm, np.ndarray):
        def xy(i: int) -> np.ndarray:
            return np.array([lm[i, 0], lm[i, 1]], dtype=np.float32)

        def vis(i: int) -> float:
            return float(lm[i, 3])
    else:
        def xy(i: int) -> np.ndarray:
            return np.array([lm[i].x, lm[i].y], dtype=np.float32)

        def vis(i: int) -> float:
            return float(getattr(lm[i], "visibility", 1.0))

    ls, rs = xy(L_SHOULDER), xy(R_SHOULDER)
    lh, rh = xy(L_HIP), xy(R_HIP)

    if (not np.isfinite(ls).all()) or (not np.isfinite(rs).all()) or (not np.isfinite(lh).all()) or (not np.isfinite(rh).all()):
        return None

    shoulder_w = float(np.linalg.norm(ls - rs))
    torso = float(np.linalg.norm((0.5 * (ls + rs)) - (0.5 * (lh + rh))))
    if (not np.isfinite(shoulder_w)) or (not np.isfinite(torso)) or torso < 1e-6:
        return None

    width_ratio = shoulder_w / (torso + 1e-6)

    # Visibility balance helps when one side is occluded (typical in side view).
    v_ls, v_rs = vis(L_SHOULDER), vis(R_SHOULDER)
    v_lh, v_rh = vis(L_HIP), vis(R_HIP)

    def balance(a: float, b: float) -> float:
        denom = max(a, b, 1e-6)
        return 1.0 - (abs(a - b) / denom)

    vis_balance = 0.5 * (balance(v_ls, v_rs) + balance(v_lh, v_rh))

    s = float(width_ratio * vis_balance)
    if not np.isfinite(s):
        return None
    return s
