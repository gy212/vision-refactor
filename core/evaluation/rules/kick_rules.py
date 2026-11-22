"""低鞭腿与垫步独有私有规则：垫步停顿、踢高、收腿、绷脚背、同侧甩臂及侧视防互换掩码。"""

from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures.kicks import (
    kicking_leg_side_safe,
    measure_arm_swing_down,
    measure_foot_flexed,
    measure_kick_height,
    measure_kick_retract_segment,
    measure_pad_step_pause,
)
from core.evaluation.measures.torso import chain_torso
from core.evaluation.rules.spec import PhaseCheck, RuleContext
from core.evaluation.rules.window_context import (
    iter_instance_slices,
    resolve_window_facing,
    write_segment_details,
)


def kick_landmarks(strategy: str, role: str, stance: str, view: str = "side") -> tuple[int, ...]:
    """返回低鞭腿与垫步独有策略所需的最小关键点下标元组。"""
    del view
    if strategy == "pad_step_settle_pause":
        return (sides_for_stance("rear", stance)[5],)
    sh, el, wr, hip, kn, ank, _, toe = sides_for_stance(role, stance)
    if strategy == "kick_peak_height_band":
        other = "rear" if role == "front" else "front"
        sup_kn = sides_for_stance(other, stance)[4]
        return toe, hip, sup_kn
    mapping = {
        "retract_ankle_reach": (ank, hip),
        "kick_ankle_extension_angle": (kn, ank, toe),
        "arm_downward_extension": (sh, el, wr),
    }
    if strategy not in mapping:
        raise KeyError(f"未知的低鞭腿独有策略: {strategy!r}")
    return mapping[strategy]


def apply_finish_anti_swap(
    lm: np.ndarray,
    mask: np.ndarray,
    kick_ankle: int,
    sup_ankle: int,
    torso: np.ndarray,
    ctx: RuleContext,
    scope: str,
) -> np.ndarray:
    """过滤侧视左右腿互换帧；finish 阶段按次若超半数互换则整次置无效。"""
    if not scope.endswith(":finish"):
        return kicking_leg_side_safe(lm, mask, kick_ankle, sup_ankle, torso)
    safe_mask = mask.copy()
    for sub, _ in iter_instance_slices(len(lm), ctx, scope):
        safe_mask[sub] = kicking_leg_side_safe(
            lm[sub],
            mask[sub],
            kick_ankle,
            sup_ankle,
            torso[sub],
            majority_threshold=0.5,
        )
    return safe_mask


def support_foot_anti_swap(
    sup_role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    torso: np.ndarray,
    ctx: RuleContext,
    scope: str,
) -> np.ndarray:
    """供低鞭腿碾脚外旋注入的支撑脚侧视防左右腿互换掩码过滤器。"""
    kick_role = "rear" if sup_role == "front" else "front"
    kick_ankle = sides_for_stance(kick_role, ctx.stance)[5]
    sup_ankle = sides_for_stance(sup_role, ctx.stance)[5]
    return apply_finish_anti_swap(lm, mask, kick_ankle, sup_ankle, torso, ctx, scope)


def measure_pad_step_pause_rule(
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """逐次执行垫步落稳后停顿规则并将结果回写为引擎可用的逐帧布尔与数值序列。"""
    n_frames = len(lm)
    bad = np.zeros(n_frames, dtype=bool)
    valid = np.zeros(n_frames, dtype=bool)
    detail_arrays: dict[str, tuple[np.ndarray, str]] = {}
    facing = resolve_window_facing(lm, mask, baseline, baseline_mask)
    if facing is None:
        ctx.measurement_issues.append("起始护位与垫步窗口关键点不足以确定面朝方向")
        return bad, valid, {}
    cfg = ctx.params
    for sub, sub_times in iter_instance_slices(n_frames, ctx, scope):
        bad_seg, valid_seg, seg_details = measure_pad_step_pause(
            lm[sub],
            mask[sub],
            sub_times,
            ctx.stance,
            cfg.pad_step_min_disp,
            cfg.pad_step_overtake_max,
            cfg.pad_step_pause_max,
            facing=facing,
        )
        valid[sub] = valid_seg
        bad[sub] = valid_seg if bad_seg else False
        write_segment_details(detail_arrays, sub, n_frames, seg_details)
    return bad, valid, detail_arrays


def measure_kick_retract_rule(
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    ctx: RuleContext,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """逐次执行前鞭腿收腿回屈规则，按次判落地前未收脚后回写序列结果。"""
    n_frames = len(lm)
    bad = np.zeros(n_frames, dtype=bool)
    valid = np.zeros(n_frames, dtype=bool)
    detail_arrays: dict[str, tuple[np.ndarray, str]] = {}
    facing = resolve_window_facing(lm, mask, baseline, baseline_mask)
    if facing is None:
        ctx.measurement_issues.append("起始护位与收腿窗口关键点不足以确定面朝方向")
        return bad, valid, {}
    _, _, _, kick_hip, _, kick_ankle, _, _ = sides_for_stance(role, ctx.stance)
    sup_role = "rear" if role == "front" else "front"
    _, _, _, _, _, sup_ankle, _, _ = sides_for_stance(sup_role, ctx.stance)
    torso = chain_torso(lm, mask)
    safe_mask = kicking_leg_side_safe(lm, mask, kick_ankle, sup_ankle, torso)
    max_x = ctx.params.kick_retract_max_x
    for sub, _ in iter_instance_slices(n_frames, ctx, scope):
        bad_seg, valid_seg, seg_details = measure_kick_retract_segment(
            lm[sub],
            safe_mask[sub],
            torso[sub],
            kick_ankle,
            kick_hip,
            facing,
            retract_max_x=max_x,
        )
        valid[sub] = valid_seg
        bad[sub] = valid_seg if bad_seg else False
        write_segment_details(detail_arrays, sub, n_frames, seg_details)
    return bad, valid, detail_arrays


def _measure_kick_leg_rule(
    strategy: str,
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    ctx: RuleContext,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """处理踢腿高度带与脚背绷直（带侧视防左右腿互换掩码）。"""
    cfg = ctx.params
    torso = chain_torso(lm, mask)
    sup_role = "rear" if role == "front" else "front"
    _, _, _, kick_hip, kick_kn, kick_ank, _, kick_toe = sides_for_stance(role, ctx.stance)
    _, _, _, _, sup_kn, sup_ank, _, _ = sides_for_stance(sup_role, ctx.stance)
    safe_mask = apply_finish_anti_swap(lm, mask, kick_ank, sup_ank, torso, ctx, scope)
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


def _measure_arm_swing_rule(
    role: str, lm: np.ndarray, mask: np.ndarray, ctx: RuleContext
) -> tuple[np.ndarray, np.ndarray, dict]:
    """踢腿同侧向下甩臂判定。"""
    sh, el, wr, *_ = sides_for_stance(role, ctx.stance)
    torso = chain_torso(lm, mask)
    return measure_arm_swing_down(
        lm,
        mask,
        torso,
        sh,
        el,
        wr,
        min_drop=ctx.params.arm_swing_drop,
        min_angle=ctx.params.arm_swing_min_angle,
    )


def build_kick_check(strategy: str, role: str, scope: str) -> PhaseCheck:
    """按低鞭腿或垫步独有策略名构造对应的相位判定函数。"""
    if strategy == "pad_step_settle_pause":
        return lambda lm, mask, base, base_mask, ctx: measure_pad_step_pause_rule(
            lm, mask, base, base_mask, ctx, scope
        )
    if strategy == "retract_ankle_reach":
        return lambda lm, mask, base, base_mask, ctx: measure_kick_retract_rule(
            role, lm, mask, base, base_mask, ctx, scope
        )
    if strategy in {"kick_peak_height_band", "kick_ankle_extension_angle"}:
        return lambda lm, mask, _b, _bm, ctx: _measure_kick_leg_rule(
            strategy, role, lm, mask, ctx, scope
        )
    if strategy == "arm_downward_extension":
        return lambda lm, mask, _b, _bm, ctx: _measure_arm_swing_rule(role, lm, mask, ctx)
    raise KeyError(f"未知的低鞭腿独有策略: {strategy!r}")
