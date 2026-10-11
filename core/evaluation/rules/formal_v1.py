# -*- coding: utf-8 -*-
"""正式扣分规则 ``formal_v1``：8 条散打规则的逐帧判定（原 ``rule_scoring._build_rules``），坐标为归一化图像坐标。

每条规则只组合 ``measures`` 的测量值与阈值；"任一手满足"不区分左右式（D-3 未决，与练习规则的差异见设计文档 §7.2）。
D-2 未决：原始序列按 ``HOLD_LAST_LEGACY`` 补帧，补出的帧仍判有效。
"""
from __future__ import annotations

import numpy as np

from core.evaluation.measures import arms, feet, legs, torso
from core.evaluation.rules.adapters import RuleScore, RuleViolation, to_rule_score
from core.evaluation.rules.engine import evaluate_rules
from core.evaluation.rules.spec import RatioDeduction, RuleProfile, RuleSpec
from core.pose.landmarks import (
    L_ANKLE,
    L_ELBOW,
    L_FOOT_INDEX,
    L_HEEL,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    MOUTH_L,
    MOUTH_R,
    NOSE,
    R_ANKLE,
    R_ELBOW,
    R_FOOT_INDEX,
    R_HEEL,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
)
from core.pose.validity import DEFAULT_VALID_CONF_THR, derive_valid_mask

ARMS = (L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST)
FIST_HEIGHT = (NOSE, L_WRIST, R_WRIST)
BACK_ARM = (MOUTH_L, MOUTH_R, L_ELBOW, L_WRIST, R_ELBOW, R_WRIST, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
LEGS = (L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE)
WIDTH = (L_ANKLE, R_ANKLE, L_SHOULDER, R_SHOULDER)
FEET = (L_HEEL, L_FOOT_INDEX, R_HEEL, R_FOOT_INDEX)
PUNCH = (L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST, L_HIP, R_HIP)
GUARD = (MOUTH_L, MOUTH_R, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP)


def frames_with(valid_mask: np.ndarray, idxs: tuple[int, ...]) -> np.ndarray:
    """逐帧：给定关键点是否全部有效（读 ``derive_valid_mask`` 的结果，不现场比较 visibility）。"""
    return np.asarray(valid_mask[:, idxs], dtype=bool).all(axis=1)


def _per_frame(landmarks, valid, fn) -> np.ndarray:
    viol = np.zeros_like(valid)
    for i, lm in enumerate(landmarks):
        if bool(valid[i]):
            viol[i] = fn(lm)
    return viol


def front_elbow_range(landmarks: np.ndarray, valid_mask: np.ndarray):
    """前手大小臂夹角 90–135°（左右任一手满足即可）。"""
    def bad(lm):
        ang_l, ang_r = arms.elbow_angles(lm)
        return not ((90.0 <= ang_l <= 135.0) or (90.0 <= ang_r <= 135.0))

    valid = frames_with(valid_mask, ARMS)
    return _per_frame(landmarks, valid, bad), valid, "前手肘角需在90-135度"


def fist_height_near_nose(landmarks: np.ndarray, valid_mask: np.ndarray):
    """拳峰与鼻尖同高：任一腕与鼻尖纵向差 ≤ 0.08 倍躯干长（近似 3cm，按画面比例）。"""
    def bad(lm):
        gap_l, gap_r, torso = arms.wrist_nose_height_gaps(lm)
        tol = 0.08 * torso
        return not ((gap_l <= tol) or (gap_r <= tol))

    valid = frames_with(valid_mask, FIST_HEIGHT)
    return _per_frame(landmarks, valid, bad), valid, "拳峰高度需与鼻尖接近"


def back_arm_close(landmarks: np.ndarray, valid_mask: np.ndarray):
    """后手贴近下颌与肋骨：任一手腕–嘴 ≤ 0.45、肘–躯干中心 ≤ 0.5 倍躯干长。"""
    def bad(lm):
        mouth, center = arms.mouth_center(lm), arms.torso_center(lm)
        torso_len = torso.midpoint_torso(lm)
        ok = []
        for side in ("L", "R"):
            wrist_mouth, elbow_center = arms.guard_distances(lm, side, mouth, center)
            ok.append((wrist_mouth <= 0.45 * torso_len) and (elbow_center <= 0.5 * torso_len))
        return not (ok[0] or ok[1])

    valid = frames_with(valid_mask, BACK_ARM)
    return _per_frame(landmarks, valid, bad), valid, "后手应贴近下颌并靠近躯干"


def knee_slight_bend(landmarks: np.ndarray, valid_mask: np.ndarray):
    """双膝微屈：任一膝角 > 170° 视为过直。"""
    def bad(lm):
        ang_l, ang_r = legs.knee_angles(lm)
        return (ang_l > 170.0) or (ang_r > 170.0)

    valid = frames_with(valid_mask, LEGS)
    return _per_frame(landmarks, valid, bad), valid, "膝关节应微曲（过直扣分）"


def stance_width(landmarks: np.ndarray, valid_mask: np.ndarray):
    """两脚间距 ≈ 肩宽：踝距/肩宽在 0.8–1.2；肩宽近零判违规。"""
    def bad(lm):
        foot_w, shoulder_w = legs.stance_widths(lm)
        if shoulder_w < 1e-6:
            return True
        return not (0.8 <= foot_w / shoulder_w <= 1.2)

    valid = frames_with(valid_mask, WIDTH)
    return _per_frame(landmarks, valid, bad), valid, "站距应接近肩宽"


def feet_parallel(landmarks: np.ndarray, valid_mask: np.ndarray):
    """两脚基本平行：跟→脚尖方向夹角 > 35° 违规；任一脚向量近零判违规。"""
    def bad(lm):
        v_l, v_r = feet.foot_vectors(lm)
        n_l, n_r = feet.vector_lengths(v_l, v_r)
        if n_l < 1e-6 or n_r < 1e-6:
            return True
        return feet.vector_angle_deg(v_l, v_r) > 35.0

    valid = frames_with(valid_mask, FEET)
    return _per_frame(landmarks, valid, bad), valid, "双脚方向应基本平行"


def _punch_frames(landmarks: np.ndarray, valid_mask: np.ndarray, idxs, decide):
    """只在出拳帧（伸展比 ≥ 1.05）判定；``decide(lm, 出拳侧, 躯干长)`` 返回是否违规。"""
    visible = frames_with(valid_mask, idxs)
    valid = np.zeros((landmarks.shape[0],), dtype=bool)
    viol = np.zeros_like(valid)
    for i, lm in enumerate(landmarks):
        if not bool(visible[i]):
            continue
        torso_len = torso.midpoint_torso(lm)
        ext_l, ext_r = arms.arm_extensions(lm, torso_len)
        if max(ext_l, ext_r) < 1.05:
            continue
        valid[i] = True
        viol[i] = decide(lm, "L" if ext_l >= ext_r else "R", torso_len)
    return viol, valid


def punch_elbow_straight(landmarks: np.ndarray, valid_mask: np.ndarray):
    """直拳出拳时出拳手肘角 ≥ 160°。"""
    viol, valid = _punch_frames(landmarks, valid_mask, PUNCH,
                                lambda lm, side, _torso: arms.elbow_angle(lm, side) < 160.0)
    return viol, valid, "出拳时肘应接近伸直"


def guard_hand(landmarks: np.ndarray, valid_mask: np.ndarray):
    """直拳出拳时另一手腕–嘴 ≤ 0.6 倍躯干长。"""
    def decide(lm, side, torso_len):
        guard = "R" if side == "L" else "L"
        return arms.wrist_mouth_distance(lm, guard, arms.mouth_center(lm)) > 0.6 * torso_len

    viol, valid = _punch_frames(landmarks, valid_mask, GUARD, decide)
    return viol, valid, "出拳时另一手应靠近下颌护防"


def _rule(code, title, points, view, scope, check, required) -> RuleSpec:
    return RuleSpec(code, title, check, required, view, scope, RatioDeduction(points, 0.3))


FORMAL_V1 = RuleProfile("formal_v1", (
    _rule("stance_elbow", "前手肘角", 2, "front", "stance", front_elbow_range, ARMS),
    _rule("stance_fist_height", "拳峰高度", 2, "front", "stance", fist_height_near_nose, FIST_HEIGHT),
    _rule("stance_back_arm", "后手贴近", 2, "front", "stance", back_arm_close, BACK_ARM),
    _rule("stance_knee", "双膝微曲", 2, "any", "stance", knee_slight_bend, LEGS),
    _rule("stance_width", "站距合理", 2, "front", "stance", stance_width, WIDTH),
    _rule("stance_feet", "脚尖方向", 2, "front", "stance", feet_parallel, FEET),
    _rule("punch_elbow", "出拳伸直", 15, "any", "punch", punch_elbow_straight, PUNCH),
    _rule("punch_guard", "护手位置", 15, "any", "punch", guard_hand, GUARD),
))


def score_rules(
    landmarks: np.ndarray,
    *,
    view: str,
    action_scope: str,
    min_valid: int = 5,
    valid_mask: np.ndarray | None = None,
    supported_capabilities=None,
) -> RuleScore:
    """单段 Pose33 的正式规则评分（100 分减扣分，不低于 0）。

    ``view``：front/side，过滤只适用某视角的规则；``action_scope``：stance/punch/both；``min_valid``：
    单条规则至少需要的有效帧数。``valid_mask=None`` 时由 ``derive_valid_mask`` 现场推导（等价
    ``visibility >= 0.5``）。``supported_capabilities`` 为后端支持的能力集合，None 表示全部支持。
    """
    view = (view or "front").lower()
    action_scope = (action_scope or "both").lower()
    if valid_mask is None:
        valid_mask = derive_valid_mask(landmarks, DEFAULT_VALID_CONF_THR)
    outcomes = evaluate_rules(landmarks, valid_mask, FORMAL_V1, view=view, action_scope=action_scope,
                              min_valid=min_valid, supported_capabilities=supported_capabilities)
    return to_rule_score(outcomes)


__all__ = ["FORMAL_V1", "RuleScore", "RuleViolation", "feet_parallel", "frames_with", "knee_slight_bend",
           "score_rules"]
