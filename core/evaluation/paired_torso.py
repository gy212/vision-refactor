"""同步双机位的肩线投影比，用于上步直拳的实战式朝向检查。"""
from __future__ import annotations

import math

import numpy as np

from core.evaluation.measures.torso import chain_torso, shoulder_width
from core.evaluation.rules.spec import PhaseCheck

# shortcut: 使用正交、弱透视近似的 cot(30°)，非正交机位需先做机位标定。
FRONTAL_PROJECTION_RATIO = math.sqrt(3)


def paired_torso_measurements(views: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """只配对相同采集时刻；缺一路或缺失点不补值，也不作为合格证据。"""
    if set(views) != {"front", "side"}:
        return {}
    front, side = views["front"], views["side"]
    _, fi, si = np.intersect1d(front["times"], side["times"], return_indices=True)
    projections, validity = [], []
    for series, indices in ((front, fi), (side, si)):
        lm = series["landmarks"][indices]
        mask = series["valid_mask"][indices]
        scale = chain_torso(lm, mask)
        projections.append(shoulder_width(lm) / np.maximum(scale, 1e-4))
        validity.append(mask[:, [11, 12]].all(axis=1) & (scale > 1e-4))
    fp, sp = projections
    valid = validity[0] & validity[1] & np.isfinite(fp) & np.isfinite(sp) & (sp > 1e-4)
    ratio = fp / np.maximum(sp, 1e-4)
    result = {}
    for view, indices in (("front", fi), ("side", si)):
        values = np.zeros(len(views[view]["times"]))
        observed = np.zeros(len(values), dtype=bool)
        values[indices], observed[indices] = ratio, valid
        result[view] = values, observed
    return result


def paired_torso_check(scope: str) -> PhaseCheck:
    """把共同证据交给现有相位引擎聚合；单视角入口保持无法判断。"""
    def check(lm, mask, baseline, baseline_mask, ctx):
        if ctx.paired_torso is None:
            return np.zeros(len(lm), bool), np.zeros(len(lm), bool), {}
        values, observed = ctx.paired_torso
        indices = ctx.phase_windows[scope]
        ratio, valid = values[indices], observed[indices]
        details = {"双视角归一肩宽比（正/侧）": (ratio, "倍（正交机位近似）")}
        return ratio > FRONTAL_PROJECTION_RATIO, valid, details

    return check
