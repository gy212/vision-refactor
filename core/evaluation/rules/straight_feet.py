"""公共实战式的脚尖脚跟间距测量。"""

from __future__ import annotations

import math

import numpy as np

from core.evaluation.measures.arms import segment_lengths
from core.evaluation.rules.spec import PhaseCheck

FEET = (29, 30, 31, 32)


def measure_toe_spread(lm: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """比较两脚尖与两脚跟的距离，沿用原来两脚方向相差35°的容许范围。"""
    toes = segment_lengths(lm, 31, 32)
    heels = segment_lengths(lm, 29, 30)
    left, right = segment_lengths(lm, 29, 31), segment_lengths(lm, 30, 32)
    valid = mask[:, FEET].all(axis=1) & (left > 1e-4) & (right > 1e-4)
    spread = (toes - heels) / np.maximum(left + right, 1e-4)
    limit = math.sin(math.radians(35 / 2))
    return np.abs(spread) > limit, valid, {"脚尖间距减脚跟间距": (spread, "两脚长度之和")}


def stance_width_check(scope: str) -> PhaseCheck:
    """量较窄的脚尖或脚跟间距，参照两面中看起来更完整的肩宽。"""

    def check(lm, mask, baseline, baseline_mask, ctx):
        if ctx.paired_torso is None:
            ctx.measurement_issues.append("缺少同步侧面肩宽参照，无法判断站距")
            return np.zeros(len(lm), bool), np.zeros(len(lm), bool), {}
        ratios, observed = ctx.paired_torso
        indices = ctx.phase_windows[scope]
        ratio = ratios[indices]
        if not observed[indices].all():
            ctx.measurement_issues.append(
                f"同步肩宽参照可用帧不足（{observed[indices].sum()}/{len(indices)}帧）"
            )
        shoulder = segment_lengths(lm, 11, 12)
        gap = np.minimum(segment_lengths(lm, 29, 30), segment_lengths(lm, 31, 32))
        # 比值为正面肩宽/侧面肩宽，两路都已除以各自肩到腿根的长度。
        width = gap / np.maximum(shoulder, 1e-4) * np.minimum(1, ratio)
        valid = observed[indices] & mask[:, FEET].all(axis=1) & (shoulder > 1e-4)
        bad = (width < ctx.params.stance_width_min) | (width > ctx.params.stance_width_max)
        return bad, valid, {"脚尖或脚跟间距与较完整肩宽之比": (width, "倍")}

    return check
