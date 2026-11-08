"""目录规则的逐帧测量与判定，供所有动作引用。"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures import arms, legs
from core.evaluation.measures import torso as torso_measures
from core.evaluation.measures.motion import relative_position
from core.evaluation.measures.posture import joint_angle, relative_height
from core.evaluation.rules.rule_descriptions import STRAIGHT_RULES
from core.evaluation.rules.straight_measures import straight_landmarks
from core.pose.landmarks import (
    L_ANKLE,
    R_ANKLE,
    L_FOOT_INDEX,
    L_HEEL,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    MOUTH_L,
    MOUTH_R,
    NOSE,
    R_FOOT_INDEX,
    R_HEEL,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
)

if TYPE_CHECKING:
    from core.evaluation.rules.feedback_config import FeedbackConfig


def _relative_to_baseline(
    code, lm, mask, torso, baseline, baseline_mask, required, sh, wr, hip, cfg, details, issues
):
    """相对起始护位的项（向后拉手、甩手、翻胯）的逐帧问题；基线不可用时为 None。"""
    baseline_torso = torso_measures.chain_torso(baseline, baseline_mask)
    base_valid = baseline_mask[:, required].all(axis=1) & (baseline_torso > 1e-4)
    if base_valid.sum() < cfg.min_valid:
        if issues is not None:
            issues.append(
                f"起始护位参照可用帧不足（{base_valid.sum()}/{len(baseline)}帧，至少{cfg.min_valid}帧）"
            )
        return None
    baseline = baseline[base_valid]
    baseline_mask = baseline_mask[base_valid]
    baseline_torso = baseline_torso[base_valid]
    if code == "hip_turn":
        rotation = legs.hip_axis_rotation(lm, baseline)
        details["估计髋轴转角"] = (rotation, "度（估计深度）")
        return rotation < cfg.hip_rotation
    position, _ = relative_position(lm, mask, torso, sh, wr)
    base_position, _ = relative_position(baseline, baseline_mask, baseline_torso, sh, wr)
    if code == "arm_swing":
        initial = np.median(base_position[:, 1])
        swing = position[:, 1] - initial
        details["手腕下摆位移"] = (swing, "躯干长度")
        return swing < cfg.motion_distance
    direction = np.median(baseline[:, 0, 0] - baseline[:, hip, 0])
    if abs(direction) < 0.1 * np.median(baseline_torso):
        if issues is not None:
            issues.append("起始护位的鼻与髋部位置不足以确定面朝方向")
        return None
    initial = np.median(base_position[:, 0])
    change = (position[:, 0] - initial) * np.sign(direction)
    details["护手前后位移"] = (change, "躯干长度")
    return change < -cfg.guard_back


LEGS = (L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE)

FEET = (L_HEEL, L_FOOT_INDEX, R_HEEL, R_FOOT_INDEX)


def required_landmarks(code: str, role: str, stance: str) -> tuple[int, ...]:
    """一个检查项在左右式下所需的关键点（BlazePose33）：声明能力契约，也是测量时的有效帧条件。"""
    if code in STRAIGHT_RULES:
        return straight_landmarks(code, role, stance)
    sh, el, wr, hip, knee, ankle, _heel, toe = sides_for_stance(role, stance)
    return {
        "stance_knee": LEGS,
        "stance_toes": FEET,
        "stance_width": (*FEET, L_SHOULDER, R_SHOULDER),
        "stance_elbow": (sh, el, wr),
        "punch_extension": (sh, el, wr),
        "stance_height": (wr, sh),
        "stance_rear_guard": (el, sh, hip),
        "guard_low": (wr, MOUTH_L, MOUTH_R),
        "guard_recover": (wr, MOUTH_L, MOUTH_R),
        "guard_shoulder": (wr, sh),
        "guard_elbow": (sh, el),
        "shoulder_level": (sh, wr),
        "stance_torso": (L_SHOULDER, R_SHOULDER),
        "foot_flexed": (knee, ankle, toe),
        "guard_back": (wr, sh, hip, NOSE),
        "arm_swing": (wr, sh),
        "hip_turn": (L_HIP, R_HIP),
    }.get(code, ())


def _frames_with(mask: np.ndarray, idxs: tuple[int, ...]) -> np.ndarray:
    return np.asarray(mask[:, idxs], dtype=bool).all(axis=1)


def _per_frame(lm: np.ndarray, valid: np.ndarray, fn) -> np.ndarray:
    bad = np.zeros_like(valid)
    for i, row in enumerate(lm):
        if bool(valid[i]):
            bad[i] = fn(row)
    return bad


def _knee_straight(row: np.ndarray, limit: float = 170.0) -> bool:
    """任一膝角超过本动作的容许上限，视为腿伸得过直。"""
    ang_l, ang_r = legs.knee_angles(row)
    return (ang_l > limit) or (ang_r > limit)


def measure_leg_stance(
    code: str, lm: np.ndarray, mask: np.ndarray, cfg: FeedbackConfig, details: dict
) -> tuple[np.ndarray, np.ndarray]:
    """双膝屈曲的逐帧判定。"""
    valid = _frames_with(mask, LEGS)
    bad = _per_frame(lm, valid, lambda row: _knee_straight(row, cfg.stance_knee_max))
    for label, joints in (("左膝角", (23, 25, 27)), ("右膝角", (24, 26, 28))):
        details[label] = (arms.series_angles(lm, *joints), "度")
    for a, b in ((23, 25), (25, 27), (24, 26), (26, 28)):
        valid &= arms.segment_lengths(lm, a, b) > 1e-4
    return bad, valid


def measure_catalog(
    code: str,
    role: str,
    lm: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    stance: str,
    cfg: FeedbackConfig,
    details: dict,
    issues: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """一个检查项在一段帧上的逐帧 ``(问题帧, 有效帧)``；``details`` 写入供教师核对的测量值序列。"""
    sh, el, wr, hip, knee, ankle, heel, toe = sides_for_stance(role, stance)
    torso = torso_measures.chain_torso(lm, mask)
    required = list(required_landmarks(code, role, stance))
    if code == "stance_knee":
        return measure_leg_stance(code, lm, mask, cfg, details)
    if code == "stance_elbow":
        angles, valid = joint_angle(lm, mask, (sh, el, wr), torso)
        details["前手肘角"] = (angles, "度")
        return (angles < cfg.stance_elbow_min) | (angles > 135), valid
    elif code in {"guard_low", "guard_recover"}:
        mouth = arms.series_mouth(lm)
        if code == "guard_low":
            details["手腕低于口部"] = (
                (lm[:, wr, 1] - mouth[:, 1]) / np.maximum(torso, 1e-4),
                "躯干长度",
            )
            bad = lm[:, wr, 1] - mouth[:, 1] > cfg.guard_drop * torso
        else:
            details["手腕距口部"] = (
                arms.distance_to(lm, wr, mouth) / np.maximum(torso, 1e-4),
                "躯干长度",
            )
            bad = arms.distance_to(lm, wr, mouth) > 0.6 * torso
    elif code == "guard_shoulder":
        drop, _ = relative_height(lm, mask, wr, sh, torso)
        details["手腕低于肩"] = (drop, "躯干长度")
        bad = drop > cfg.guard_shoulder_drop
    elif code in {"guard_elbow", "shoulder_level"}:
        if code == "guard_elbow":
            height, _ = relative_height(lm, mask, sh, el, torso)
            label, threshold = "肘高于肩", cfg.elbow_raise
        else:
            height, _ = relative_height(lm, mask, wr, sh, torso)
            height = np.abs(height)
            label, threshold = "拳肩高度差", cfg.shoulder_height
        bad = height > threshold
        details[label] = (height, "躯干长度")
    elif code == "foot_flexed":
        angles = arms.series_angles(lm, knee, ankle, toe)
        details["膝踝脚尖角"] = (angles, "度")
        bad = angles < cfg.foot_angle
        mask = mask.copy()
        mask[:, toe] &= arms.segment_lengths(lm, toe, ankle) > 1e-4
    elif code in {"guard_back", "arm_swing", "hip_turn"}:
        bad = _relative_to_baseline(
            code,
            lm,
            mask,
            torso,
            baseline,
            baseline_mask,
            required,
            sh,
            wr,
            hip,
            cfg,
            details,
            issues,
        )
        if bad is None:
            return np.zeros(len(lm), bool), np.zeros(len(lm), bool)
    else:
        return np.zeros(len(lm), bool), np.zeros(len(lm), bool)
    return bad, mask[:, required].all(axis=1) & (torso > 1e-4)
