"""公共实战式（stance）规则的策略测量与点位契约。"""

from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures import arms, legs
from core.evaluation.measures.feet import FEET, measure_rear_foot_splay
from core.evaluation.measures.posture import (
    elbow_body_distance,
    joint_angle,
    relative_height,
)
from core.evaluation.measures.torso import chain_torso
from core.evaluation.paired_torso import paired_torso_check
from core.evaluation.rules.spec import PhaseCheck, RuleContext
from core.evaluation.rules.window_context import empty_result, resolve_window_facing
from core.pose.landmarks import (
    L_ANKLE,
    L_FOOT_INDEX,
    L_HEEL,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    R_ANKLE,
    R_FOOT_INDEX,
    R_HEEL,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
)

LEGS = (L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE)


def stance_landmarks(strategy: str, role: str, stance: str, view: str = "front") -> tuple[int, ...]:
    """返回公共实战式策略所需的关键点下标。"""
    del view
    sh, el, wr, hip, *_ = sides_for_stance(role, stance)
    _, _, _, _, _, _, rear_hl, rear_toe = sides_for_stance("rear", stance)
    mapping = {
        "elbow_angle_range": (sh, el, wr),
        "wrist_shoulder_floor": (sh, wr),
        "elbow_body_distance": (sh, el, hip),
        "knee_angle_limit": LEGS,
        "feet_paired_shoulder_ratio": (*FEET, L_SHOULDER, R_SHOULDER),
        "rear_foot_toe_projection": (rear_hl, rear_toe),
        "paired_shoulder_projection_ratio": (L_SHOULDER, R_SHOULDER),
    }
    if strategy not in mapping:
        raise KeyError(f"未知的实战式策略: {strategy!r}")
    return mapping[strategy]


def measure_stance_elbow(
    role: str, lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """前手肘角范围判定（下限由参数控制，上限 135°）。"""
    sh, el, wr, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    angles, valid = joint_angle(lm, mask, (sh, el, wr), torso)
    bad = (angles < ctx.params.stance_elbow_min) | (angles > 135)
    return bad, valid, {"前手肘角": (angles, "度")}


def measure_stance_height(
    role: str, lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """实战式前手手腕高度判定：不低于同侧肩膀。"""
    sh, _, wr, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    values, valid = relative_height(lm, mask, wr, sh, torso)
    return values > 0, valid, {"手腕低于肩": (values, "躯干长度")}


def measure_stance_rear_guard(
    role: str, lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """实战式后手肘贴身距离判定。"""
    sh, el, _, hip, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    values, valid = elbow_body_distance(lm, mask, sh, el, hip, torso)
    return values > 0.5, valid, {"肘离身体中部的距离": (values, "躯干长度")}


def measure_stance_knee(
    lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """实战式双膝微屈判定：任一膝角超过上限判伸得过直。"""
    valid = np.asarray(mask[:, LEGS], dtype=bool).all(axis=1)
    bad = np.zeros_like(valid)
    limit = ctx.params.stance_knee_max
    for i, row in enumerate(lm):
        if bool(valid[i]):
            ang_l, ang_r = legs.knee_angles(row)
            bad[i] = (ang_l > limit) or (ang_r > limit)
    details = {
        "左膝角": (arms.series_angles(lm, L_HIP, L_KNEE, L_ANKLE), "度"),
        "右膝角": (arms.series_angles(lm, R_HIP, R_KNEE, R_ANKLE), "度"),
    }
    for a, b in (
        (L_HIP, L_KNEE),
        (L_KNEE, L_ANKLE),
        (R_HIP, R_KNEE),
        (R_KNEE, R_ANKLE),
    ):
        valid &= arms.segment_lengths(lm, a, b) > 1e-4
    return bad, valid, details


def stance_width_check(scope: str) -> PhaseCheck:
    """量较窄的脚尖或脚跟间距，参照两面中看起来更完整的肩宽。"""

    def check(lm, mask, baseline, baseline_mask, ctx):
        del baseline, baseline_mask
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
        shoulder = arms.segment_lengths(lm, L_SHOULDER, R_SHOULDER)
        gap = np.minimum(
            arms.segment_lengths(lm, L_HEEL, R_HEEL),
            arms.segment_lengths(lm, L_FOOT_INDEX, R_FOOT_INDEX),
        )
        # 比值为正面肩宽/侧面肩宽，两路都已除以各自肩到腿根的长度。
        width = gap / np.maximum(shoulder, 1e-4) * np.minimum(1, ratio)
        valid = observed[indices] & mask[:, FEET].all(axis=1) & (shoulder > 1e-4)
        bad = (width < ctx.params.stance_width_min) | (width > ctx.params.stance_width_max)
        return bad, valid, {"脚尖或脚跟间距与较完整肩宽之比": (width, "倍")}

    return check


def measure_stance_toes(
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """实战式后脚外撇判定（侧面）：后脚尖落后于后脚跟即判错，不查前脚。"""
    facing = resolve_window_facing(lm, mask, baseline, baseline_mask)
    if facing is None:
        ctx.measurement_issues.append("实战式窗口关键点不足以确定面朝方向，无法判断后脚外撇")
        return empty_result(len(lm))
    _, _, _, _, _, _, rear_hl, rear_toe = sides_for_stance("rear", ctx.stance)
    torso = chain_torso(lm, mask)
    return measure_rear_foot_splay(lm, mask, torso, rear_hl, rear_toe, facing)


def build_stance_check(strategy: str, role: str, scope: str) -> PhaseCheck:
    """按实战式策略名构造对应的相位判定函数。"""
    if strategy == "paired_shoulder_projection_ratio":
        return paired_torso_check(scope)
    if strategy == "feet_paired_shoulder_ratio":
        return stance_width_check(scope)
    if strategy == "rear_foot_toe_projection":
        return lambda lm, mask, base, base_mask, ctx: measure_stance_toes(
            lm, mask, base, base_mask, ctx
        )
    if strategy == "knee_angle_limit":
        return lambda lm, mask, _b, _bm, ctx: measure_stance_knee(lm, mask, ctx)
    if strategy == "elbow_angle_range":
        return lambda lm, mask, _b, _bm, ctx: measure_stance_elbow(role, lm, mask, ctx)
    if strategy == "wrist_shoulder_floor":
        return lambda lm, mask, _b, _bm, ctx: measure_stance_height(role, lm, mask, ctx)
    if strategy == "elbow_body_distance":
        return lambda lm, mask, _b, _bm, ctx: measure_stance_rear_guard(role, lm, mask, ctx)
    raise KeyError(f"未知的实战式策略: {strategy!r}")
