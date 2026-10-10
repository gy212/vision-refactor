# -*- coding: utf-8 -*-
"""发力顺序指标：正面脚部旋转门槛 → 侧面逐拳检测蹬地/转腰/送肩/上步 → 时序比对 → 合格判定。

原 ``tech_eval.eval_force_sequence``（514 行）按"事件检测 → 时序 → 判定"拆成各自不超过 100 行的步骤，数值口径不变。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.evaluation.body_sides import facing_direction, infer_front_leg
from core.evaluation.techeval.contract import IndicatorResult, all_valid, resolve_mask
from core.evaluation.techeval.events import detect_retract_events_side
from core.foundation import angle_deg, landmark_xy
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


@dataclass(frozen=True)
class ForceThresholds:
    """``eval_force_sequence`` 的判定阈值（参数名与默认值同旧签名）。"""

    twist_min_deg: float = 10.0
    shoulder_drive_min: float = 0.03
    step_front_min: float = 0.01
    step_back_min: float = 0.01
    step_ok_ratio: float = 0.5
    step_sync_allow: int = 3
    step_post_sec: float = 0.4
    rotation_var_thr: float = 1820.0
    rotation_min_frames: int = 12


# --------------------------------------------------------------------------- #
# 事件检测：逐帧信号
# --------------------------------------------------------------------------- #


def _heel_toe(side: str) -> tuple[int, int]:
    return (L_HEEL, L_FOOT_INDEX) if side.upper() == "L" else (R_HEEL, R_FOOT_INDEX)


def _foot_center_x(lm: np.ndarray, mask_row: np.ndarray, side: str) -> float | None:
    heel, toe = _heel_toe(side)
    if not all_valid(mask_row, (heel, toe)):
        return None
    return float((landmark_xy(lm, heel)[0] + landmark_xy(lm, toe)[0]) * 0.5)


def _foot_angle_deg(lm: np.ndarray, mask_row: np.ndarray, side: str) -> float | None:
    heel, toe = _heel_toe(side)
    if not all_valid(mask_row, (heel, toe)):
        return None
    a = landmark_xy(lm, heel)
    b = landmark_xy(lm, toe)
    return float(np.degrees(np.arctan2(float(b[1] - a[1]), float(b[0] - a[0]))))


def _rotation_check(lms: np.ndarray, lms_mask: np.ndarray, th: ForceThresholds) -> tuple[bool, dict[str, Any]]:
    """正面两脚朝向角（解卷绕后）的方差超过阈值即判脚部旋转异常。"""
    detail: dict[str, Any] = {
        "rotation_fail": False,
        "rotation_var_l": None,
        "rotation_var_r": None,
        "rotation_var_thr": float(th.rotation_var_thr),
        "rotation_frames_l": 0,
        "rotation_frames_r": 0,
    }
    if lms.size == 0:
        return False, detail

    def _series(side: str) -> np.ndarray:
        vals: list[float] = []
        for i in range(int(lms.shape[0])):
            v = _foot_angle_deg(lms[i], lms_mask[i], side)
            if v is None or not np.isfinite(v):
                continue
            vals.append(float(v))
        return np.asarray(vals, dtype=np.float32)

    def _var_deg(angles: np.ndarray) -> float | None:
        if angles.size < int(th.rotation_min_frames):
            return None
        ang = np.deg2rad(angles.astype(np.float32))
        ang = np.unwrap(ang)
        ang_deg = np.rad2deg(ang)
        return float(np.var(ang_deg))

    angles_l = _series("L")
    angles_r = _series("R")
    detail["rotation_frames_l"] = int(angles_l.size)
    detail["rotation_frames_r"] = int(angles_r.size)
    var_l = _var_deg(angles_l)
    var_r = _var_deg(angles_r)
    detail["rotation_var_l"] = None if var_l is None else float(var_l)
    detail["rotation_var_r"] = None if var_r is None else float(var_r)

    fail = False
    if var_l is not None and float(var_l) > float(th.rotation_var_thr):
        fail = True
    if var_r is not None and float(var_r) > float(th.rotation_var_thr):
        fail = True
    detail["rotation_fail"] = bool(fail)
    return bool(fail), detail


def _wrap_deg(x: float) -> float:
    y = (float(x) + 180.0) % 360.0 - 180.0
    return float(y)


def _twist_deg(lm: np.ndarray, mask_row: np.ndarray) -> float | None:
    """肩线与髋线的夹角（转腰幅度，度）。"""
    if not all_valid(mask_row, (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)):
        return None
    ls = landmark_xy(lm, L_SHOULDER)
    rs = landmark_xy(lm, R_SHOULDER)
    lh = landmark_xy(lm, L_HIP)
    rh = landmark_xy(lm, R_HIP)
    sa = float(np.degrees(np.arctan2(float(rs[1] - ls[1]), float(rs[0] - ls[0]))))
    ha = float(np.degrees(np.arctan2(float(rh[1] - lh[1]), float(rh[0] - lh[0]))))
    return abs(_wrap_deg(sa - ha))


def _shoulder_drive(lm: np.ndarray, mask_row: np.ndarray, arm: str) -> float | None:
    """出拳侧肩相对髋中点沿朝向的前送量。"""
    d = facing_direction(lm, mask_row)
    if d is None:
        return None
    sh = L_SHOULDER if arm.upper() == "L" else R_SHOULDER
    if not all_valid(mask_row, (sh, L_HIP, R_HIP)):
        return None
    hip_x = float((landmark_xy(lm, L_HIP)[0] + landmark_xy(lm, R_HIP)[0]) * 0.5)
    return (float(landmark_xy(lm, sh)[0]) - hip_x) * float(d)


def _back_leg_points(back_leg: str) -> tuple[int, int, int, int, int]:
    """后腿的 (脚跟, 脚尖, 髋, 膝, 踝)。"""
    if back_leg == "L":
        return L_HEEL, L_FOOT_INDEX, L_HIP, L_KNEE, L_ANKLE
    return R_HEEL, R_FOOT_INDEX, R_HIP, R_KNEE, R_ANKLE


def _push_off_ok(lm: np.ndarray, mask_row: np.ndarray, back_leg: str) -> bool | None:
    """检测蹬地动作：后脚脚跟明显抬高 + 膝盖参与伸展发力。

    真正的蹬地特征：
    1. 脚跟明显抬高（pitch 在合理范围，且垂直位移足够）
    2. 后腿膝盖有一定伸展（说明腿部在发力推地）
    """
    heel, toe, hip, knee, ankle = _back_leg_points(back_leg)
    if not all_valid(mask_row, (heel, toe, knee, ankle)):
        return None
    # 1. 脚跟明显抬高检测
    heel_xy = landmark_xy(lm, heel)
    toe_xy = landmark_xy(lm, toe)
    dy = float(toe_xy[1] - heel_xy[1])  # 脚尖y - 脚跟y（y向下为正）
    if dy <= 0.0:
        return False  # 脚跟必须高于脚尖（抬脚跟）
    dx = float(toe_xy[0] - heel_xy[0])
    pitch = float(np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-9)))
    # 基于标准动作NPZ数据分析优化的阈值（侧面视角）
    # - Pitch P80 范围: 31-38°, 取 20-45° 作为有效范围
    # - dy_ratio P80 范围: 0.6-0.8, 取 ≥0.5 作为阈值
    # - 膝盖角度 P20-P90: 125-165°, 取 120-165° 作为有效范围
    pitch_ok = 20.0 <= pitch <= 45.0
    dy_ratio = dy / (abs(dx) + 1e-6)
    heel_lift_ok = pitch_ok and (dy >= 0.03) and (dy_ratio >= 0.5)
    # 2. 后腿膝盖角度检测（基于标准动作数据：P20=125°, P90=165°）；蹬地时通常在 120°-165° 之间
    knee_ang = angle_deg(landmark_xy(lm, hip), landmark_xy(lm, knee), landmark_xy(lm, ankle))
    knee_ok = 120.0 <= knee_ang <= 165.0
    return heel_lift_ok and knee_ok


def _heel_lift(lm: np.ndarray, mask_row: np.ndarray, back_leg: str) -> float | None:
    """后脚脚跟抬高程度（dy / |dx| 的比例，范围 0~1+）。"""
    heel, toe, _hip, _knee, _ankle = _back_leg_points(back_leg)
    if not all_valid(mask_row, (heel, toe)):
        return None
    heel_xy = landmark_xy(lm, heel)
    toe_xy = landmark_xy(lm, toe)
    dy = float(toe_xy[1] - heel_xy[1])  # y向下为正，脚尖在脚跟下方时dy>0
    dx = float(toe_xy[0] - heel_xy[0])
    if dy <= 0:
        return 0.0
    return dy / (abs(dx) + 1e-6)


def _back_knee_angle(lm: np.ndarray, mask_row: np.ndarray, back_leg: str) -> float | None:
    """后腿膝盖角度。"""
    _heel, _toe, hip, knee, ankle = _back_leg_points(back_leg)
    if not all_valid(mask_row, (hip, knee, ankle)):
        return None
    return angle_deg(landmark_xy(lm, hip), landmark_xy(lm, knee), landmark_xy(lm, ankle))


# --------------------------------------------------------------------------- #
# 事件检测：出拳前窗口里的蹬地、转腰、送肩，以及前后的上步
# --------------------------------------------------------------------------- #


def _find_push(lm_pre: np.ndarray, mask_pre: np.ndarray, s: int, back_leg: str) -> tuple[int | None, float]:
    """蹬地时刻：先找单帧蹬地姿态，找不到再看脚跟抬高趋势；返回 ``(时刻, 动态得分)``。"""
    for j in range(lm_pre.shape[0]):
        ok_push = _push_off_ok(lm_pre[j], mask_pre[j], back_leg)
        if ok_push is None:
            continue
        if bool(ok_push):
            return int(s + j), 0.0
    if lm_pre.shape[0] < 5:
        return None, 0.0
    heel_lifts = [_heel_lift(lm_pre[j], mask_pre[j], back_leg) for j in range(lm_pre.shape[0])]
    knee_angles = [_back_knee_angle(lm_pre[j], mask_pre[j], back_leg) for j in range(lm_pre.shape[0])]
    # 寻找脚跟抬高的趋势：早期低 -> 晚期高
    valid_lifts = [(j, v) for j, v in enumerate(heel_lifts) if v is not None]
    valid_knees = [(j, v) for j, v in enumerate(knee_angles) if v is not None]
    if len(valid_lifts) < 4 or len(valid_knees) < 4:
        return None, 0.0
    # 分前后两半比较
    mid = len(valid_lifts) // 2
    early_lifts = [v for _, v in valid_lifts[:mid]]
    late_lifts = [v for _, v in valid_lifts[mid:]]
    early_knees = [v for _, v in valid_knees[:mid]]
    late_knees = [v for _, v in valid_knees[mid:]]
    lift_increase = float(np.median(late_lifts)) - float(np.median(early_lifts))
    knee_change = float(np.median(late_knees)) - float(np.median(early_knees))
    # 动态蹬地特征（基于标准动作NPZ数据分析优化）：
    # - 脚跟明显抬高（后半段比前半段高至少 0.10，且后半段中位数≥0.40）
    # - 膝盖角度变化不大或略微伸展（变化在 -15°~+20° 之间）
    lift_ok = lift_increase >= 0.10 and float(np.median(late_lifts)) >= 0.40
    knee_stable = -15.0 <= knee_change <= 20.0
    if lift_ok and knee_stable:
        # 找到后半段第一个lift达标的位置作为t_push
        for j, lift in valid_lifts[mid:]:
            if lift >= 0.40:
                return int(s + j), float(lift_increase)
    return None, 0.0


def _find_twist(lm_pre: np.ndarray, mask_pre: np.ndarray, s: int, twist_min_deg: float) -> int | None:
    """转腰时刻：窗口内转腰幅度最大的帧，幅度不足阈值时为 None。"""
    twists: list[tuple[int, float]] = []
    for j in range(lm_pre.shape[0]):
        v = _twist_deg(lm_pre[j], mask_pre[j])
        if v is None:
            continue
        twists.append((int(s + j), float(v)))
    if not twists:
        return None
    t_twist, twist_v = max(twists, key=lambda t: t[1])
    return None if float(twist_v) < float(twist_min_deg) else t_twist


def _find_drive(lm_pre: np.ndarray, mask_pre: np.ndarray, s: int, arm: str, shoulder_drive_min: float) -> int | None:
    """送肩时刻：前送量相对窗口前半段中位数首次超过阈值的帧。"""
    drives: list[tuple[int, float]] = []
    for j in range(lm_pre.shape[0]):
        v = _shoulder_drive(lm_pre[j], mask_pre[j], arm)
        if v is None:
            continue
        drives.append((int(s + j), float(v)))
    if not drives:
        return None
    base = float(np.median(np.asarray([v for _t, v in drives[: max(1, len(drives) // 2)]], dtype=np.float32)))
    for t, v in drives:
        if float(v - base) >= float(shoulder_drive_min):
            return int(t)
    return None


def _foot_series(lms: np.ndarray, lms_mask: np.ndarray, side: str) -> np.ndarray:
    """逐帧脚中心沿朝向的 x（朝向或脚点无效时为 NaN）。"""
    xs = np.full((lms.shape[0],), np.nan, dtype=np.float32)
    for i in range(lms.shape[0]):
        lm = lms[i]
        mask_row = lms_mask[i]
        d = facing_direction(lm, mask_row)
        if d is None:
            continue
        cx = _foot_center_x(lm, mask_row, side)
        if cx is None:
            continue
        xs[i] = float(cx) * float(d)
    return xs


def _first_idx(vals: np.ndarray, thr: float) -> int | None:
    for i in range(int(vals.size)):
        v = float(vals[i])
        if np.isfinite(v) and v >= float(thr):
            return int(i)
    return None


def _baseline(vals: np.ndarray) -> float | None:
    """前 20% 有限值的中位数。"""
    if vals.size == 0:
        return None
    n0 = max(1, int(round(float(vals.size) * 0.2)))
    base = vals[:n0]
    base = base[np.isfinite(base)]
    if base.size == 0:
        return None
    return float(np.median(base))


def _find_steps(side_lm, side_mask, lm_pre, mask_pre, s: int, t_punch: int, post_step: int,
                front_leg: str, back_leg: str, th: ForceThresholds) -> dict[str, Any]:
    """上步检测（前脚上步 -> 后脚跟进）：前脚位移取出拳前窗口，后脚位移取出拳后窗口；时刻为窗口内下标。"""
    post_end = min(int(side_lm.shape[0]), int(t_punch + post_step + 1))
    lm_post = side_lm[t_punch:post_end]
    mask_post = side_mask[t_punch:post_end]
    front_x_pre = _foot_series(lm_pre, mask_pre, front_leg)
    back_x_post = _foot_series(lm_post, mask_post, back_leg) if lm_post.size > 0 else np.zeros((0,), dtype=np.float32)
    steps: dict[str, Any] = {"front_detected": None, "back_detected": None,
                             "front_50": None, "front_100": None, "back_start": None}
    base_front = _baseline(front_x_pre)
    if base_front is not None:
        front_disp = front_x_pre - float(base_front)
        steps["front_100"] = _first_idx(front_disp, float(th.step_front_min))
        steps["front_50"] = _first_idx(front_disp, float(th.step_front_min) * 0.5)
        steps["front_detected"] = steps["front_100"] is not None
    base_back = _baseline(back_x_post) if back_x_post.size > 0 else None
    if base_back is not None:
        back_disp = back_x_post - float(base_back)
        steps["back_start"] = _first_idx(back_disp, float(th.step_back_min))
        steps["back_detected"] = steps["back_start"] is not None
    return steps


# --------------------------------------------------------------------------- #
# 时序：每一拳的各信号先后
# --------------------------------------------------------------------------- #


def _step_order_ok(steps: dict[str, Any], s: int, t_punch: int, step_sync_allow: int) -> bool | None:
    """前脚半程上步不晚于出拳、全程上步与出拳同步、后脚在出拳后跟进；信号不足为 None。"""
    if steps["front_detected"] is None or steps["back_detected"] is None:
        return None
    if not steps["front_detected"] or not steps["back_detected"]:
        return False
    front_50 = None if steps["front_50"] is None else int(s + steps["front_50"])
    front_100 = None if steps["front_100"] is None else int(s + steps["front_100"])
    back = None if steps["back_start"] is None else int(t_punch + steps["back_start"])
    front_before_punch = (front_50 is not None) and (front_50 <= int(t_punch))
    front_sync = (front_100 is not None) and (abs(int(front_100) - int(t_punch)) <= int(step_sync_allow))
    back_after_punch = (back is not None) and (int(back) >= int(t_punch - 1))
    return bool(front_before_punch and front_sync and back_after_punch)


def _signal_order_ok(t_push: int, t_twist: int, t_drive: int, t_punch: int) -> bool:
    """蹬地 → 转腰 → 送肩 → 出拳。

    - 蹬地应在转腰之前或同时（允许蹬地略晚于转腰3帧，约0.1秒@30fps，考虑到侧面视角检测误差）
    - 转腰应在送肩之前或同时（允许3帧重叠）
    - 送肩应在出拳之前或同时
    """
    allow_overlap = 3
    push_before_twist = (t_push <= t_twist + 3)
    twist_before_drive = (t_twist <= t_drive + allow_overlap)
    drive_before_punch = (t_drive <= t_punch)
    return bool(push_before_twist and twist_before_drive and drive_before_punch)


def _event_record(e: dict, side_lm, side_mask, *, pre: int, post_step: int, front_leg: str, back_leg: str,
                  th: ForceThresholds) -> dict[str, Any] | None:
    """一拳的检测与时序记录；出拳前窗口为空时为 None（不计入）。``ok`` 为 None 表示信号不全未评估。"""
    arm = str(e.get("arm") or "")
    t_punch = int(e.get("start") or 0)
    s = max(0, t_punch - pre)
    lm_pre = side_lm[s : t_punch + 1]
    mask_pre = side_mask[s : t_punch + 1]
    if lm_pre.size == 0:
        return None
    t_push, push_dynamic_score = _find_push(lm_pre, mask_pre, s, back_leg)
    t_twist = _find_twist(lm_pre, mask_pre, s, th.twist_min_deg)
    t_drive = _find_drive(lm_pre, mask_pre, s, arm, th.shoulder_drive_min)
    steps = _find_steps(side_lm, side_mask, lm_pre, mask_pre, s, t_punch, post_step, front_leg, back_leg, th)
    step_fields = {
        "push_dynamic_score": float(push_dynamic_score) if push_dynamic_score > 0 else None,
        "front_step_detected": steps["front_detected"],
        "back_step_detected": steps["back_detected"],
        "t_front_step_50": None if steps["front_50"] is None else int(s + steps["front_50"]),
        "t_front_step_100": None if steps["front_100"] is None else int(s + steps["front_100"]),
        "t_back_step": None if steps["back_start"] is None else int(t_punch + steps["back_start"]),
        "step_ok": _step_order_ok(steps, s, t_punch, th.step_sync_allow),
    }
    if t_push is None or t_twist is None or t_drive is None:
        return {"arm": arm, "t_push": t_push, "t_twist": t_twist, "t_drive": t_drive,
                "t_punch": int(t_punch), "ok": None,
                "push_detected": t_push is not None,
                "twist_detected": t_twist is not None,
                "drive_detected": t_drive is not None,
                **step_fields}
    return {"arm": arm, "t_push": int(t_push), "t_twist": int(t_twist), "t_drive": int(t_drive),
            "t_punch": int(t_punch), "ok": _signal_order_ok(int(t_push), int(t_twist), int(t_drive), int(t_punch)),
            **step_fields}


# --------------------------------------------------------------------------- #
# 判定
# --------------------------------------------------------------------------- #


def _summary(per_event: list[dict], n_events: int, front_leg: str, back_leg: str, rotation_detail: dict,
             th: ForceThresholds) -> dict[str, Any]:
    used = len(per_event)
    ok = sum(1 for e in per_event if e["ok"] is True)
    bad = sum(1 for e in per_event if e["ok"] is False)
    step_eval = [e for e in per_event if e.get("step_ok") is not None]
    step_ok = sum(1 for e in step_eval if e.get("step_ok") is True)
    return {
        "events_total": int(n_events),
        "events_used": int(used),
        "events_ok": int(ok),
        "events_bad": int(bad),
        "push_detected": int(sum(1 for e in per_event if e.get("t_push") is not None)),
        "twist_detected": int(sum(1 for e in per_event if e.get("t_twist") is not None)),
        "drive_detected": int(sum(1 for e in per_event if e.get("t_drive") is not None)),
        "step_front_detected": int(sum(1 for e in per_event if e.get("front_step_detected") is True)),
        "step_back_detected": int(sum(1 for e in per_event if e.get("back_step_detected") is True)),
        "step_ok": int(step_ok),
        "step_ok_ratio": None if len(step_eval) == 0 else float(step_ok / max(1, len(step_eval))),
        "front_leg": str(front_leg),
        "back_leg": str(back_leg),
        "twist_min_deg": float(th.twist_min_deg),
        "shoulder_drive_min": float(th.shoulder_drive_min),
        "step_front_min": float(th.step_front_min),
        "step_back_min": float(th.step_back_min),
        "step_ok_ratio_thr": float(th.step_ok_ratio),
        "rotation": rotation_detail,
        "events": per_event,
    }


def _verdict(detail: dict[str, Any], step_ok_ratio: float) -> IndicatorResult:
    """出拳次数 → 蹬地检测率 → 上步 → 顺序正确率，逐级判定。

    蹬地检测率过低说明大量出拳没有标准蹬地动作：< 40% 判不合格（侧面视角下虽然后腿可能部分遮挡，
    但 40% 以上的检测率是基本要求）；检测率达标后，上步规范且顺序正确率 ≥ 70% 才合格。
    """
    used, ok, bad = detail["events_used"], detail["events_ok"], detail["events_bad"]
    if used < 2:
        detail["primary_cause"] = "出拳次数不足"
        return IndicatorResult(status="无法判定", reason="发力顺序：出拳次数不足", detail=detail)
    push_detection_rate = float(detail["push_detected"] / max(1, used))
    evaluated = ok + bad
    accuracy = float(ok / max(1, evaluated)) if evaluated > 0 else 0.0
    if push_detection_rate < 0.40:
        detail["primary_cause"] = "蹬地检测率低"
        detail["failed_stage"] = "push_off"
        return IndicatorResult(status="不合格", reason="发力顺序：蹬地动作检测率偏低，多数出拳蹬地动作不标准", detail=detail)
    if detail["step_ok_ratio"] is None:
        detail["primary_cause"] = "上步信号不足"
        detail["failed_stage"] = "step"
        return IndicatorResult(status="无法判定", reason="发力顺序：上步信号不足", detail=detail)
    if detail["step_ok_ratio"] < float(step_ok_ratio):
        detail["primary_cause"] = "上步不规范"
        detail["failed_stage"] = "step"
        return IndicatorResult(status="不合格", reason="发力顺序：上步不规范", detail=detail)
    if evaluated < 2:
        detail["primary_cause"] = "有效评估事件不足"
        return IndicatorResult(status="无法判定", reason="发力顺序：有效评估事件不足", detail=detail)
    if accuracy >= 0.7:  # 至少70%的正确率才算合格
        detail["primary_cause"] = "达标"
        return IndicatorResult(status="合格", reason="发力顺序：顺序一致性良好", detail=detail)
    detail["primary_cause"] = "顺序一致性不足"
    detail["failed_stage"] = "sequence"
    return IndicatorResult(status="不合格", reason="发力顺序：顺序一致性不足", detail=detail)


def _rotation_failed(rotation_detail: dict[str, Any]) -> IndicatorResult:
    detail = {
        "events_total": 0, "events_used": 0, "events_ok": 0, "events_bad": 0,
        "push_detected": 0, "twist_detected": 0, "drive_detected": 0,
        "step_front_detected": 0, "step_back_detected": 0, "step_ok": 0, "step_ok_ratio": None,
        "rotation_fail": True, "rotation": rotation_detail, "events": [],
        "primary_cause": "脚部旋转异常", "failed_stage": "rotation",
    }
    return IndicatorResult(status="不合格", reason="发力顺序：正面脚部旋转异常，判定蹬地与顺序不合格", detail=detail)


def eval_force_sequence(
    front_landmarks: np.ndarray,
    side_landmarks: np.ndarray,
    *,
    fps: float,
    stance: str = "left",
    vis_thr: float = 0.5,
    twist_min_deg: float = 10.0,
    shoulder_drive_min: float = 0.03,
    step_front_min: float = 0.01,
    step_back_min: float = 0.01,
    step_ok_ratio: float = 0.5,
    step_sync_allow: int = 3,
    step_post_sec: float = 0.4,
    rotation_var_thr: float = 1820.0,
    rotation_min_frames: int = 12,
    front_valid_mask: np.ndarray | None = None,
    side_valid_mask: np.ndarray | None = None,
) -> IndicatorResult:
    th = ForceThresholds(twist_min_deg, shoulder_drive_min, step_front_min, step_back_min, step_ok_ratio,
                         step_sync_allow, step_post_sec, rotation_var_thr, rotation_min_frames)
    front_mask = resolve_mask(front_landmarks, front_valid_mask, vis_thr)
    side_mask = resolve_mask(side_landmarks, side_valid_mask, vis_thr)
    rotation_fail, rotation_detail = _rotation_check(front_landmarks, front_mask, th)
    if rotation_fail:
        return _rotation_failed(rotation_detail)
    if side_landmarks.size == 0:
        return IndicatorResult(status="无法判定", reason="发力顺序：侧面段缺失", detail={"primary_cause": "侧面段缺失"})
    events = detect_retract_events_side(
        side_landmarks, fps=fps, vis_thr=vis_thr, ext_angle_thr=160.0, retract_angle_thr=45.0,
        ext_forward_dx=0.02, retract_sec_thr=0.2, max_window_sec=1.0, valid_mask=side_mask,
    )
    if not events:
        return IndicatorResult(status="无法判定", reason="发力顺序：未检测到出拳",
                               detail={"primary_cause": "未检测到出拳", "events": []})
    front_leg = infer_front_leg(side_landmarks, side_mask, fallback=("L" if (stance or "left").lower() == "left" else "R"))
    back_leg = "R" if front_leg == "L" else "L"
    pre = int(round(0.3 * float(fps)))
    post_step = int(round(float(step_post_sec) * float(fps)))
    per_event = []
    for e in events:
        record = _event_record(e, side_landmarks, side_mask, pre=pre, post_step=post_step,
                               front_leg=front_leg, back_leg=back_leg, th=th)
        if record is not None:
            per_event.append(record)
    detail = _summary(per_event, len(events), front_leg, back_leg, rotation_detail, th)
    return _verdict(detail, step_ok_ratio)
