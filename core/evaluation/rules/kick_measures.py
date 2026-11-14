"""低鞭腿与垫步规则的测量分派与关键点依赖声明（无直接向量范数或反余弦计算）。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.kicks import (
    measure_arm_swing_down,
    measure_body_lean,
    measure_foot_flexed,
    measure_head_down,
    measure_hip_turn_width,
    measure_kick_guard_elbow,
    measure_kick_guard_jaw,
    measure_kick_guard_shoulder,
    measure_kick_height,
    measure_pivot_heel,
)
from core.evaluation.measures.torso import chain_torso
from core.evaluation.rules.kick_step import (
    apply_finish_anti_swap,
    measure_kick_retract_rule,
    measure_pad_step_rule,
    resolve_window_facing,
)
from core.evaluation.rules.spec import PhaseVerdict, RuleContext
from core.evaluation.rules.straight_motion import (
    guard_back_landmarks,
    measure_guard_back,
)
from core.pose.landmarks import (
    L_HIP,
    L_SHOULDER,
    MOUTH_L,
    MOUTH_R,
    NOSE,
    R_HIP,
    R_SHOULDER,
)

KICK_SEGMENTS = frozenset({"pad_step", "front_low_kick", "rear_low_kick"})

# 策略 -> (覆盖 valid_ratio 或 None, 覆盖 trigger_ratio 或 None)
_VERDICT_OVERRIDES: dict[str, tuple[float | None, float | None]] = {
    "pad_step_displacement": (None, 0.01),
    "pad_step_ankle_overtake": (None, 0.01),
    "pad_step_settle_pause": (None, 0.01),
    "retract_ankle_reach": (None, 0.01),
    "kick_peak_height_band": (0.5, 0.5),
    "kick_ankle_extension_angle": (0.5, 0.5),
    "nose_shoulder_drop": (None, 0.01),
    "shoulder_line_tilt_angle": (None, 0.01),
    "kick_elbow_behind_body": (None, 0.01),
}


def kick_phase_verdict(strategy: str, cfg, *, absence: bool) -> PhaseVerdict:
    """按低鞭腿/垫步测量策略生成相位聚合配置。"""
    valid_override, trigger_override = _VERDICT_OVERRIDES.get(strategy, (None, None))
    valid_ratio = cfg.valid_ratio if valid_override is None else valid_override
    trigger_ratio = cfg.trigger_ratio if trigger_override is None else trigger_override
    return PhaseVerdict(cfg.min_valid, valid_ratio, trigger_ratio, absence=absence)


def kick_landmarks(
    strategy: str, role: str, stance: str, view: str = "side"
) -> tuple[int, ...]:
    """返回低鞭腿与垫步规则所需的最小关键点下标元组。"""
    if strategy in {"pad_step_displacement", "pad_step_settle_pause"}:
        return (sides_for_stance("rear", stance)[5],)
    if strategy == "pad_step_ankle_overtake":
        return sides_for_stance("rear", stance)[5], sides_for_stance("front", stance)[5]
    if strategy == "kick_elbow_behind_body":
        return guard_back_landmarks(role, stance, view)
    if strategy == "frontal_hip_width_ratio":
        return L_HIP, R_HIP
    if strategy == "nose_shoulder_drop":
        return NOSE, L_SHOULDER, R_SHOULDER
    if strategy == "shoulder_line_tilt_angle":
        return L_SHOULDER, R_SHOULDER

    sh, el, wr, hip, kn, ank, hl, toe = sides_for_stance(role, stance)
    if strategy == "support_foot_turn_projection":
        return hl, toe
    if strategy == "kick_peak_height_band":
        other = "rear" if role == "front" else "front"
        sup_kn = sides_for_stance(other, stance)[4]
        return toe, hip, sup_kn
    if strategy == "retract_ankle_reach":
        return ank, hip
    if strategy == "kick_ankle_extension_angle":
        return kn, ank, toe
    if strategy == "arm_downward_extension":
        return sh, el, wr
    if strategy == "kick_wrist_shoulder_drop":
        return wr, sh
    if strategy == "upper_arm_torso_flare":
        return el, sh, hip
    if strategy == "guard_wrist_jaw_distance":
        return wr, MOUTH_L, MOUTH_R
    raise KeyError(f"未知的低鞭腿测量策略: {strategy!r}")


def _empty_result(n_frames: int) -> tuple[np.ndarray, np.ndarray, dict]:
    empty = np.zeros(n_frames, dtype=bool)
    return empty, empty.copy(), {}


def _measure_side_leg_rule(
    strategy: str,
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """处理带侧视防左右腿互换掩码的低鞭腿下肢规则。"""
    cfg = ctx.params
    torso = chain_torso(lm, mask)
    other = "rear" if role == "front" else "front"
    if strategy == "support_foot_turn_projection":
        sup_role, kick_role = role, other
    else:
        kick_role, sup_role = role, other
    _, _, _, kick_hip, kick_kn, kick_ank, _, kick_toe = sides_for_stance(
        kick_role, ctx.stance
    )
    _, _, _, _, sup_kn, sup_ank, sup_hl, sup_toe = sides_for_stance(
        sup_role, ctx.stance
    )
    safe_mask = apply_finish_anti_swap(lm, mask, kick_ank, sup_ank, torso, ctx, scope)
    if strategy == "support_foot_turn_projection":
        facing = resolve_window_facing(lm, mask, baseline, baseline_mask)
        if facing is None:
            ctx.measurement_issues.append(
                "起始护位与支撑脚窗口关键点不足以确定面朝方向"
            )
            return _empty_result(len(lm))
        return measure_pivot_heel(lm, safe_mask, torso, sup_hl, sup_toe, facing)
    if strategy == "kick_peak_height_band":
        return measure_kick_height(
            lm,
            safe_mask,
            torso,
            kick_toe,
            kick_hip,
            sup_kn,
            hip_margin=cfg.kick_height_hip_margin,
            knee_margin=cfg.kick_height_knee_margin,
        )
    return measure_foot_flexed(
        lm,
        safe_mask,
        torso,
        kick_kn,
        kick_ank,
        kick_toe,
        min_angle=cfg.foot_flexed_min_angle,
    )


def _measure_baseline_relative_rule(
    strategy: str,
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """处理依赖起始护位基线（正面翻胯、低头）的低鞭腿规则。"""
    cfg = ctx.params
    torso = chain_torso(lm, mask)
    base_torso = chain_torso(baseline, baseline_mask)
    if strategy == "frontal_hip_width_ratio":
        base_ok = (
            baseline_mask[:, L_HIP] & baseline_mask[:, R_HIP] & (base_torso > 1e-4)
        )
        if int(base_ok.sum()) < cfg.min_valid:
            ctx.measurement_issues.append(
                f"起始护位正面两髋参照可用帧不足（{int(base_ok.sum())}/{len(baseline)}帧，至少{cfg.min_valid}帧）"
            )
            return _empty_result(len(lm))
        return measure_hip_turn_width(
            lm,
            mask,
            torso,
            baseline,
            baseline_mask,
            base_torso,
            max_ratio=cfg.hip_turn_width_ratio,
        )
    base_ok = baseline_mask[:, [NOSE, L_SHOULDER, R_SHOULDER]].all(axis=1) & (
        base_torso > 1e-4
    )
    if int(base_ok.sum()) < cfg.min_valid:
        ctx.measurement_issues.append(
            f"起始护位头肩参照可用帧不足（{int(base_ok.sum())}/{len(baseline)}帧，至少{cfg.min_valid}帧）"
        )
        return _empty_result(len(lm))
    return measure_head_down(
        lm,
        mask,
        torso,
        baseline,
        baseline_mask,
        base_torso,
        max_drop=cfg.head_down_drop,
    )


def measure_kick(
    strategy: str,
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """按策略执行低鞭腿或垫步测量。"""
    cfg = ctx.params
    if strategy in {
        "pad_step_displacement",
        "pad_step_ankle_overtake",
        "pad_step_settle_pause",
    }:
        return measure_pad_step_rule(
            strategy, lm, mask, baseline, baseline_mask, ctx, scope
        )
    if strategy == "retract_ankle_reach":
        return measure_kick_retract_rule(
            role, lm, mask, baseline, baseline_mask, ctx, scope
        )
    if strategy == "kick_elbow_behind_body":
        thr = (
            cfg.kick_guard_back_side
            if ctx.view == "side"
            else cfg.kick_guard_back_front
        )
        sub_ctx = replace(ctx, params=replace(cfg, guard_back=thr))
        return measure_guard_back(role, lm, mask, baseline, baseline_mask, sub_ctx)
    if strategy in {
        "support_foot_turn_projection",
        "kick_peak_height_band",
        "kick_ankle_extension_angle",
    }:
        return _measure_side_leg_rule(
            strategy, role, lm, mask, baseline, baseline_mask, ctx, scope
        )
    if strategy in {"frontal_hip_width_ratio", "nose_shoulder_drop"}:
        return _measure_baseline_relative_rule(
            strategy, lm, mask, baseline, baseline_mask, ctx
        )
    if strategy == "shoulder_line_tilt_angle":
        return measure_body_lean(lm, mask, max_angle=cfg.body_lean_angle)

    torso = chain_torso(lm, mask)
    sh, el, wr, hip, *_ = sides_for_stance(role, ctx.stance)
    if strategy == "arm_downward_extension":
        return measure_arm_swing_down(
            lm,
            mask,
            torso,
            sh,
            el,
            wr,
            min_drop=cfg.arm_swing_drop,
            min_angle=cfg.arm_swing_min_angle,
        )
    if strategy == "kick_wrist_shoulder_drop":
        thr = (
            cfg.kick_guard_drop_side
            if ctx.view == "side"
            else cfg.kick_guard_drop_front
        )
        return measure_kick_guard_shoulder(lm, mask, torso, wr, sh, max_drop=thr)
    if strategy == "upper_arm_torso_flare":
        return measure_kick_guard_elbow(
            lm, mask, torso, el, sh, hip, max_flare=cfg.kick_guard_elbow_flare
        )
    if strategy == "guard_wrist_jaw_distance":
        return measure_kick_guard_jaw(
            lm, mask, torso, wr, max_dist=cfg.kick_guard_jaw_dist
        )
    raise KeyError(f"未知的低鞭腿测量策略: {strategy!r}")
