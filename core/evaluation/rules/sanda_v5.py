# -*- coding: utf-8 -*-
"""统一规则目录 ``sanda_v5``：检查项目录、检查清单、阈值配置、逐项测量与规则登记（D-3、D-5）。

考试、学生练习和离线处理共用这一套目录（设计文档 §1.3 第 5 条、§7.2）：同一检查项的测量、阈值、取手方式和结论一致，
各模式只在呈现上不同（练习融合正侧证据给出问题说明；考试与 batch 逐视角报告结论，不计入分数）。由 ``practice_v4``
（检查项、相位与测量）并入 ``formal_v1`` 的拳峰高度、站距（相对量）与出拳伸直而来；取手一律按左右式，不再"任一手满足"。

``rule_profile`` 把所选动作与左右式的检查清单登记为 ``RuleSpec``（``PhaseVerdict`` 聚合），经 ``evaluate_rules`` 判定。
坐标为宽高比修正坐标（x、z 以图像高度为单位，D-5）；补帧一律无效（D-2）。现场可用 ``repo_root()`` 下的
``feedback_config.json`` 覆盖 ``FeedbackConfig``，文件名与原有字段不变。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.measures import arms, legs
from core.evaluation.measures import feet
from core.evaluation.measures import torso as torso_measures
from core.evaluation.rules.spec import PhaseCheck, PhaseVerdict, RuleContext, RuleProfile, RuleSpec
from core.evaluation.view_weights import check_front_weights, practice_view_weights
from core.foundation.paths import repo_root
from core.pose.landmarks import (
    L_ANKLE,
    L_FOOT_INDEX,
    L_HEEL,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    MOUTH_L,
    MOUTH_R,
    NOSE,
    R_ANKLE,
    R_FOOT_INDEX,
    R_HEEL,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
)

RULE_VERSION = "sanda-rules-mediapipe-2026-10-14-v5"
PROFILE_NAME = "sanda_v5"
DEFAULT_STANCE = "left"  # 考试与 batch 的左右式：左手、左脚在前，右手护脸（D-3，配置值，以后可改）
ACTIONS = {
    "stance": "实战式",
    "straight_combo": "前后手直拳组合",
    "hook_combo": "前后手摆拳组合",
    "front_straight": "前手直拳", "rear_straight": "后手直拳",
    "front_hook": "前手摆拳", "rear_hook": "后手摆拳",
    "front_low_kick": "前低鞭腿", "rear_low_kick": "后低鞭腿",
}
STANCES = {"left": "左式（左手、左脚在前）", "right": "右式（右手、右脚在前）"}
PHASES = {"start": "开始实战式", "motion": "动作过程", "finish": "动作结束瞬间", "end": "结束实战式"}
STATE_LABELS = {"candidate": "待教师确认", "not_observed": "已检查帧中未发现", "unable": "证据不足", "pending_rule": "待规则/测量确认"}
DOC = "武术散打得分点.docx 表1"
SUMMARY = "扣分点说明功能需求汇总.md"
# code -> 问题、身体部位（相对于前后侧）、标准、不能自动判定的原因。
RULES = {
    "stance_elbow": ("前手肘角不在90°～135°", "front_arm", "前手大小臂夹角90°～135°", ""),
    "stance_height": ("前手拳峰未与鼻尖同高", "front_hand", "拳峰与鼻尖同高：纵向差不超过0.08倍躯干长（暂定，现场微调）", ""),
    "stance_rear_guard": ("后手未夹紧并贴近肋骨与下颚", "rear_arm", "后手大小臂夹紧、紧贴肋骨与下颚", ""),
    "stance_knee": ("单腿或双腿伸直", "knees", "两膝微屈", ""),
    "stance_width": ("两脚站距过宽或过窄", "feet", "两脚略与肩同宽：踝距/肩宽在0.8～1.2（暂定，现场微调）", ""),
    "stance_toes": ("两脚内八或外八", "feet", "两脚大致平行、朝斜前方", ""),
    "stance_torso": ("躯干正向前方", "torso", "躯干侧向前方", ""),
    "stance_head": ("挺胸抬头", "head_torso", "下颚微收、含胸收腹", "Pose33不足以可靠判定含胸收腹，需教师查看"),
    "guard_low": ("向下掉手", "hand", "护手保持在下颚附近", ""),
    "guard_back": ("向后拉手", "hand", "护手保持原护位，不向后拉", ""),
    "guard_elbow": ("向上抬肘", "arm", "护手侧保持护位，不向上抬肘", ""),
    "vertical_fist": ("出拳成立拳", "hand", "前手直拳出拳结束时拳背朝上", "轻量Pose关键点不能可靠判断握拳朝向，需教师查看"),
    "shoulder_level": ("出拳未与肩部齐平", "arm", "出拳与肩部齐平", ""),
    "punch_extension": ("直拳出拳时肘未伸直", "arm", "直拳出拳结束时肘接近伸直（肘角不小于160°）", ""),
    "guard_recover": ("未回收护住下颚", "hand", "前手回收护住下颚", ""),
    "foot_flexed": ("脚背未绷直（勾脚尖）", "foot", "低鞭腿脚背绷直", ""),
    "hip_turn": ("未翻胯", "hip", "低鞭腿翻胯", ""),
    "arm_swing": ("未顺势向下甩手", "arm", "踢腿同侧手顺势向下甩手", ""),
}
STANCE_RULES = tuple(code for code in RULES if code.startswith("stance_"))
PARTS = {"arm": "臂肘", "hand": "手", "foot": "脚背", "hip": "髋", "knees": "双膝", "feet": "双脚", "torso": "躯干", "head_torso": "头部与胸腹"}
# 这些现有公式依赖固定观察方向，不能把另一视角同一公式的数值硬凑进来。
PRIMARY_VIEW = {"guard_back": "side", "stance_toes": "front", "stance_torso": "front", "stance_width": "front"}
FORMAL = "原正式规则 formal_v1（D-3 并入）"


def build_checks(action: str, stance: str) -> list[dict[str, Any]]:
    """所选动作与左右式的检查清单（开始实战式 → 各动作段 → 结束实战式）；不统计重复次数。"""
    if not isinstance(action, str) or not isinstance(stance, str) or action not in ACTIONS or stance not in STANCES:
        raise ValueError("请选择支持的动作和左式/右式")
    checks: list[dict[str, Any]] = []

    def add(code: str, segment: str, phase: str, role: str = "", blocked: str = "", source: str = SUMMARY):
        name, part, standard, reason = RULES[code]
        if part.startswith(("front_", "rear_")):
            role, part = part.split("_", 1)
        side = ("左" if stance == "left" else "右") if role == "front" else ("右" if stance == "left" else "左") if role == "rear" else ""
        prefix = "前侧" if role == "front" else "后侧" if role == "rear" else ""
        checks.append({"id": f"{segment}:{phase}:{role}:{code}", "code": code,
                       "name": name, "bodyPart": f"{prefix}{PARTS[part]}" + (f"（{side}）" if side else ""),
                       "segment": segment, "segmentLabel": ACTIONS[segment], "phase": phase,
                       "phaseLabel": PHASES[phase], "standard": standard,
                       "source": source, "blockedReason": blocked or reason, "role": role})

    for code in STANCE_RULES:
        add(code, "stance", "start", source=f"{DOC} 第3～9行")
    segments = {"straight_combo": ["front_straight", "rear_straight"],
                "hook_combo": ["front_hook", "rear_hook"]}.get(action, [action])
    for segment in segments:
        if segment == "stance":
            continue
        front = segment.startswith("front_")
        role = "front" if front else "rear"
        guard = "rear" if front else "front"
        hook = "hook" in segment
        kick = "kick" in segment
        straight = "straight" in segment
        source = f"{DOC} 第{10 if front else 11}行" if straight else SUMMARY
        for phase in ("motion", "finish"):
            if not kick or front:
                for code in ("guard_low", "guard_back", "guard_elbow"):
                    # 后手摆拳前手回收未包含向后拉手，不扩展问卷清单。
                    if segment == "rear_hook" and code == "guard_back":
                        continue
                    blocked = "原文摆拳阶段误写为直拳，需教师核对" if segment == "front_hook" else ""
                    if hook and code == "guard_elbow":
                        blocked = "摆拳要求抬肘与错误抬肘的幅度/阶段需区分"
                    add(code, segment, phase, guard, blocked, source)
            if segment in {"rear_straight", "rear_hook"}:
                for code in ("guard_low", "guard_back", "guard_elbow"):
                    add(code, segment, phase, "rear", "需确认出拳侧掉手/拉手/抬肘的阶段边界", SUMMARY)
            if segment == "rear_hook" or segment == "rear_low_kick":
                add("guard_recover", segment, phase, "front")
            if kick:
                for code in ("foot_flexed", "hip_turn", "arm_swing"):
                    add(code, segment, phase, role)
            elif phase == "finish" or front:
                add("shoulder_level", segment, phase, role,
                    "原文摆拳阶段误写为直拳，需教师核对" if hook else "")
            if straight and phase == "finish":
                add("punch_extension", segment, phase, role, source=FORMAL)
            if segment == "front_straight" and phase == "finish":
                add("vertical_fist", segment, phase, "front", source=source)
    for code in STANCE_RULES:
        add(code, "stance", "end", source=f"{DOC} 第12～18行")
    # 保持开始→前手→后手→结束，再按各阶段身体部位排列；不统计重复次数。
    order = {name: i + 1 for i, name in enumerate(segments)}
    checks.sort(key=lambda c: (0 if c["phase"] == "start" else 99 if c["phase"] == "end" else order[c["segment"]],
                               list(PHASES).index(c["phase"]), c["bodyPart"], c["id"]))
    return checks


def default_front_weights() -> dict:
    """各动作族、检查项的正面权重初值（工程初值，非准确率/概率；后续按教师标注的双摄样本标定）。"""
    return {
        "stance": {"default": .5, "stance_elbow": .2, "stance_knee": .2,
                   "stance_rear_guard": .6},
        "straight": {"default": .4, "guard_low": .7, "guard_elbow": .7, "shoulder_level": .7},
        "hook": {"default": .6, "guard_low": .75, "guard_elbow": .7, "shoulder_level": .6},
        "kick": {"default": .4, "guard_low": .65, "guard_elbow": .65,
                 "foot_flexed": .15, "hip_turn": .7, "arm_swing": .6},
    }


@dataclass(frozen=True)
class FeedbackConfig:
    """工程初始阈值，保留标定入口；除肘角范围外不冒充教师确认的标准。"""
    sample_fps: float = 0.0  # 0=原帧率逐帧，不主动抽帧
    max_edge: int = 1920
    pose_variant: str = "full"
    min_valid: int = 3
    valid_ratio: float = .6
    trigger_ratio: float = .3
    motion_distance: float = .25  # 相对躯干长度
    guard_drop: float = .25
    guard_back: float = .18
    shoulder_height: float = .20
    elbow_raise: float = .10
    torso_front_ratio: float = .75
    foot_angle: float = 145.0
    hip_rotation: float = 20.0
    fusion_support: float = .7
    phase_tolerance_seconds: float = .4
    stance_window_seconds: float = .2
    min_motion_seconds: float = .06
    return_distance: float = .15
    max_gap_seconds: float = .10
    finish_window_seconds: float = .05
    fist_height: float = .08        # 拳峰与鼻尖纵向差上限（躯干长度，原正式规则，暂定）
    stance_width_min: float = .8    # 踝距/肩宽下限（原正式规则，暂定）
    stance_width_max: float = 1.2   # 踝距/肩宽上限
    punch_elbow: float = 160.0      # 直拳出拳结束时肘角下限（度，原正式规则）
    guard_wrist_height: float = .5  # 实战式护位：手腕不低于肩下 0.5 倍躯干长，排除直立垂手的帧
    front_weights: dict = field(default_factory=default_front_weights)

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for k, v in asdict(self).items()
               if k not in {"front_weights", "pose_variant", "sample_fps"}):
            raise ValueError("规则参数必须是有限正数")
        if not math.isfinite(self.sample_fps) or not 0 <= self.sample_fps <= 60 or self.max_edge < 64 or self.min_valid < 2 or not isinstance(self.min_valid, int):
            raise ValueError("采样频率为0（逐帧）或不超过60，分辨率至少64，有效帧数至少2")
        if self.pose_variant not in {"lite", "full", "heavy"}:
            raise ValueError("模型只能选择lite/full/heavy")
        if self.return_distance >= self.motion_distance:
            raise ValueError("回收阈值必须小于动作起动阈值")
        if self.stance_width_min >= self.stance_width_max:
            raise ValueError("站距下限必须小于上限")
        if self.valid_ratio > 1 or self.trigger_ratio > 1:
            raise ValueError("比例参数不能大于1")
        if not .5 < self.fusion_support <= 1:
            raise ValueError("融合支持阈值必须大于0.5且不超过1")
        check_front_weights(self.front_weights, RULES)


def load_feedback_config() -> FeedbackConfig:
    """练习 profile 的阈值：``repo_root()/feedback_config.json`` 存在时按其字段覆盖默认值。"""
    path = repo_root() / "feedback_config.json"
    if not path.exists():
        return FeedbackConfig()
    try:
        values = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(values, dict):
            raise ValueError("配置应为JSON对象")
        return FeedbackConfig(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"学生练习配置无效：{path}；{exc}") from exc


def check_view_weights(check: dict, cfg: FeedbackConfig) -> dict[str, float]:
    """一个检查项的正侧基础权重（单视角公式的项为 1/0）。"""
    return practice_view_weights(check["code"], check["segment"], cfg.front_weights, PRIMARY_VIEW)


def _relative_to_baseline(code, lm, torso, baseline, baseline_mask, required, sh, wr, hip, cfg, details):
    """相对起始护位的项（向后拉手、甩手、翻胯）的逐帧问题；基线不可用时为 None。"""
    baseline_torso = torso_measures.chain_torso(baseline, baseline_mask)
    if not baseline_mask[:, required].all() or np.any(baseline_torso < 1e-4):
        return None
    if code == "hip_turn":
        rotation = legs.hip_axis_rotation(lm, baseline)
        details["估计髋轴转角"] = (rotation, "度（估计深度）")
        return rotation < cfg.hip_rotation
    if code == "arm_swing":
        initial = np.median((baseline[:, wr, 1] - baseline[:, sh, 1]) / baseline_torso)
        swing = (lm[:, wr, 1] - lm[:, sh, 1]) / np.maximum(torso, 1e-4) - initial
        details["手腕下摆位移"] = (swing, "躯干长度")
        return swing < cfg.motion_distance
    direction = np.median(baseline[:, 0, 0] - baseline[:, hip, 0])
    if abs(direction) < .1 * np.median(baseline_torso):
        return None
    initial = np.median((baseline[:, wr, 0] - baseline[:, sh, 0]) / baseline_torso)
    change = ((lm[:, wr, 0] - lm[:, sh, 0]) / np.maximum(torso, 1e-4) - initial) * np.sign(direction)
    details["护手前后位移"] = (change, "躯干长度")
    return change < -cfg.guard_back


LEGS = (L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE)
FEET = (L_HEEL, L_FOOT_INDEX, R_HEEL, R_FOOT_INDEX)
WIDTH = (L_ANKLE, R_ANKLE, L_SHOULDER, R_SHOULDER)


def required_landmarks(code: str, role: str, stance: str) -> tuple[int, ...]:
    """一个检查项在左右式下所需的关键点（BlazePose33）：声明能力契约，也是测量时的有效帧条件。"""
    sh, el, wr, hip, knee, ankle, _heel, toe = sides_for_stance(role, stance)
    return {
        "stance_knee": LEGS, "stance_toes": FEET, "stance_width": WIDTH,
        "stance_elbow": (sh, el, wr), "punch_extension": (sh, el, wr),
        "stance_height": (wr, NOSE),
        "stance_rear_guard": (wr, MOUTH_L, MOUTH_R, el, sh, hip),
        "guard_low": (wr, MOUTH_L, MOUTH_R), "guard_recover": (wr, MOUTH_L, MOUTH_R),
        "guard_elbow": (sh, el), "shoulder_level": (sh, wr),
        "stance_torso": (L_SHOULDER, R_SHOULDER),
        "foot_flexed": (knee, ankle, toe),
        "guard_back": (wr, sh, hip, NOSE), "arm_swing": (wr, sh), "hip_turn": (L_HIP, R_HIP),
    }.get(code, ())


def _frames_with(mask: np.ndarray, idxs: tuple[int, ...]) -> np.ndarray:
    return np.asarray(mask[:, idxs], dtype=bool).all(axis=1)


def _per_frame(lm: np.ndarray, valid: np.ndarray, fn) -> np.ndarray:
    bad = np.zeros_like(valid)
    for i, row in enumerate(lm):
        if bool(valid[i]):
            bad[i] = fn(row)
    return bad


def _knee_straight(row: np.ndarray) -> bool:
    """任一膝角 > 170° 视为腿伸直（"两膝微屈"）。"""
    ang_l, ang_r = legs.knee_angles(row)
    return (ang_l > 170.0) or (ang_r > 170.0)


def _toes_not_parallel(row: np.ndarray) -> bool:
    """两脚跟→脚尖方向夹角 > 35° 视为内八或外八；任一脚向量近零也算。"""
    v_l, v_r = feet.foot_vectors(row)
    n_l, n_r = feet.vector_lengths(v_l, v_r)
    if n_l < 1e-6 or n_r < 1e-6:
        return True
    return feet.vector_angle_deg(v_l, v_r) > 35.0


def _measure(code: str, role: str, lm: np.ndarray, mask: np.ndarray, baseline: np.ndarray, baseline_mask: np.ndarray,
             stance: str, cfg: FeedbackConfig, details: dict) -> tuple[np.ndarray, np.ndarray]:
    """一个检查项在一段帧上的逐帧 ``(问题帧, 有效帧)``；``details`` 写入供教师核对的测量值序列。"""
    sh, el, wr, hip, knee, ankle, heel, toe = sides_for_stance(role, stance)
    torso = torso_measures.chain_torso(lm, mask)
    required = list(required_landmarks(code, role, stance))
    if code == "stance_knee":
        valid = _frames_with(mask, LEGS)
        bad = _per_frame(lm, valid, _knee_straight)
        for label, joints in (("左膝角", (23, 25, 27)), ("右膝角", (24, 26, 28))):
            details[label] = (arms.series_angles(lm, *joints), "度")
        for a, b in ((23, 25), (25, 27), (24, 26), (26, 28)):
            valid &= arms.segment_lengths(lm, a, b) > 1e-4
        return bad, valid
    if code == "stance_toes":
        valid = _frames_with(mask, FEET)
        bad = _per_frame(lm, valid, _toes_not_parallel)
        valid &= (arms.segment_lengths(lm, 31, 29) > 1e-4)
        valid &= (arms.segment_lengths(lm, 32, 30) > 1e-4)
        return bad, valid
    if code in {"stance_elbow", "punch_extension"}:
        angles = arms.series_angles(lm, sh, el, wr)
        if code == "stance_elbow":
            details["前手肘角"] = (angles, "度")
            bad = (angles < 90) | (angles > 135)
        else:
            details["出拳手肘角"] = (angles, "度")
            bad = angles < cfg.punch_elbow
        mask = mask.copy()
        mask[:, el] &= (arms.segment_lengths(lm, sh, el) > 1e-4)
        mask[:, wr] &= (arms.segment_lengths(lm, wr, el) > 1e-4)
    elif code == "stance_height":
        gap = np.abs(lm[:, wr, 1] - lm[:, NOSE, 1]) / np.maximum(torso, 1e-4)
        details["拳峰与鼻尖纵向差"] = (gap, "躯干长度")
        bad = gap > cfg.fist_height
    elif code == "stance_width":
        shoulder = arms.segment_lengths(lm, L_SHOULDER, R_SHOULDER)
        ratio = arms.segment_lengths(lm, L_ANKLE, R_ANKLE) / np.maximum(shoulder, 1e-6)
        details["踝距/肩宽"] = (ratio, "倍")
        bad = (shoulder < 1e-6) | (ratio < cfg.stance_width_min) | (ratio > cfg.stance_width_max)
    elif code in {"stance_rear_guard", "guard_low", "guard_recover"}:
        mouth = arms.series_mouth(lm)
        if code == "guard_low":
            details["手腕低于口部"] = ((lm[:, wr, 1] - mouth[:, 1]) / np.maximum(torso, 1e-4), "躯干长度")
            bad = lm[:, wr, 1] - mouth[:, 1] > cfg.guard_drop * torso
        else:
            details["手腕距口部"] = (arms.distance_to(lm, wr, mouth) / np.maximum(torso, 1e-4), "躯干长度")
            bad = arms.distance_to(lm, wr, mouth) > (.45 if code == "stance_rear_guard" else .6) * torso
            if code == "stance_rear_guard":
                center = (lm[:, sh, :2] + lm[:, hip, :2]) / 2
                bad |= arms.distance_to(lm, el, center) > .5 * torso
    elif code in {"guard_elbow", "shoulder_level"}:
        bad = (lm[:, sh, 1] - lm[:, el, 1] > cfg.elbow_raise * torso) if code == "guard_elbow" else (np.abs(lm[:, wr, 1] - lm[:, sh, 1]) > cfg.shoulder_height * torso)
        details["肘高于肩" if code == "guard_elbow" else "拳肩高度差"] = (
            (lm[:, sh, 1] - lm[:, el, 1]) / np.maximum(torso, 1e-4) if code == "guard_elbow" else
            np.abs(lm[:, wr, 1] - lm[:, sh, 1]) / np.maximum(torso, 1e-4), "躯干长度")
    elif code == "stance_torso":
        details["投影肩宽"] = (torso_measures.shoulder_width(lm) / np.maximum(torso, 1e-4), "躯干长度")
        bad = torso_measures.shoulder_width(lm) > cfg.torso_front_ratio * torso
    elif code == "foot_flexed":
        angles = arms.series_angles(lm, knee, ankle, toe)
        details["膝踝脚尖角"] = (angles, "度")
        bad = angles < cfg.foot_angle
        mask = mask.copy()
        mask[:, toe] &= arms.segment_lengths(lm, toe, ankle) > 1e-4
    elif code in {"guard_back", "arm_swing", "hip_turn"}:
        bad = _relative_to_baseline(code, lm, torso, baseline, baseline_mask, required, sh, wr, hip, cfg, details)
        if bad is None:
            return np.zeros(len(lm), bool), np.zeros(len(lm), bool)
    else:
        return np.zeros(len(lm), bool), np.zeros(len(lm), bool)
    return bad, mask[:, required].all(axis=1) & (torso > 1e-4)


# "未做某动作"类检查项：只在整个阶段都没出现该动作时才算有问题。
ABSENCE_RULES = frozenset({"hip_turn", "arm_swing"})


def _phase_check(code: str, role: str) -> PhaseCheck:
    def check(lm: np.ndarray, mask: np.ndarray, baseline: np.ndarray, baseline_mask: np.ndarray, ctx: RuleContext):
        details: dict = {}
        bad, valid = _measure(code, role, lm, mask, baseline, baseline_mask, ctx.stance, ctx.params, details)
        return bad, valid, details

    return check


def rule_profile(action: str, stance: str, cfg: FeedbackConfig) -> RuleProfile:
    """所选动作与左右式的检查清单登记为规则目录：规则编码即检查项 id，范围即 ``segment:phase`` 相位窗口。

    只适用单视角公式的项（``PRIMARY_VIEW``）只登记该视角；原文缺失或需教师核对的项登记为 ``blocked``。
    """
    return RuleProfile(name=PROFILE_NAME, rules=tuple(
        RuleSpec(
            code=c["id"], title=c["name"], check=_phase_check(c["code"], c["role"]),
            required=required_landmarks(c["code"], c["role"], stance),
            view=PRIMARY_VIEW.get(c["code"], "any"), scope=f"{c['segment']}:{c['phase']}",
            aggregation=PhaseVerdict(cfg.min_valid, cfg.valid_ratio, cfg.trigger_ratio,
                                     absence=c["code"] in ABSENCE_RULES),
            authorization="blocked" if c["blockedReason"] else "candidate",
            blocked_reason=c["blockedReason"], source=c["source"],
        )
        for c in build_checks(action, stance)
    ))
