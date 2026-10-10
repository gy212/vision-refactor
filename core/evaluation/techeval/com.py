# -*- coding: utf-8 -*-
"""分段质心（方案3）重心指标：按人体节段质量比合成单帧质心，再判定出拳过程的质心稳定性。"""
from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import facing_direction, foot_edges_x
from core.evaluation.techeval.contract import IndicatorResult, all_valid, resolve_mask
from core.foundation import landmark_xy
from core.pose.landmarks import (
    L_ANKLE,
    L_EAR,
    L_ELBOW,
    L_FOOT_INDEX,
    L_HEEL,
    L_HIP,
    L_INDEX,
    L_KNEE,
    L_PINKY,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_ANKLE,
    R_EAR,
    R_ELBOW,
    R_FOOT_INDEX,
    R_HEEL,
    R_HIP,
    R_INDEX,
    R_KNEE,
    R_PINKY,
    R_SHOULDER,
    R_WRIST,
)


BODY_SEGMENT_MASS_RATIOS = {
    "head": 0.081,           # 头部
    "trunk": 0.497,          # 躯干
    "upper_arm_l": 0.028,    # 左上臂
    "upper_arm_r": 0.028,    # 右上臂
    "forearm_l": 0.015,      # 左前臂
    "forearm_r": 0.015,      # 右前臂
    "hand_l": 0.006,         # 左手
    "hand_r": 0.006,         # 右手
    "thigh_l": 0.100,        # 左大腿
    "thigh_r": 0.100,        # 右大腿
    "shank_l": 0.047,        # 左小腿
    "shank_r": 0.047,        # 右小腿
    "foot_l": 0.014,         # 左脚
    "foot_r": 0.014,         # 右脚
}


def _compute_segment_center(lm: np.ndarray, mask_row: np.ndarray, indices: list[int]) -> np.ndarray | None:
    """计算身体段的中心点坐标（基于可见关键点）"""
    points = []
    for idx in indices:
        if bool(mask_row[idx]):
            points.append(landmark_xy(lm, idx))
    if len(points) == 0:
        return None
    return np.mean(points, axis=0)


def _compute_body_com_single(lm: np.ndarray, mask_row: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """
    计算单帧人体质心（CoM）
    返回: (com_coords, debug_info)
    """
    weighted_sum = np.zeros(2)
    total_mass = 0.0
    segment_status = {}

    # 头部：使用鼻子和双耳
    head_center = _compute_segment_center(lm, mask_row, [NOSE, L_EAR, R_EAR])
    if head_center is not None:
        mass = BODY_SEGMENT_MASS_RATIOS["head"]
        weighted_sum += head_center * mass
        total_mass += mass
        segment_status["head"] = {"mass": mass, "center": head_center.tolist()}
    else:
        segment_status["head"] = None

    # 躯干：质心约在剑突-肚脐之间（偏下约40%，即肩:髋 = 0.6:0.4）
    if all_valid(mask_row, [L_SHOULDER, R_SHOULDER]) and all_valid(mask_row, [L_HIP, R_HIP]):
        shoulder_center = (landmark_xy(lm, L_SHOULDER) + landmark_xy(lm, R_SHOULDER)) / 2
        hip_center = (landmark_xy(lm, L_HIP) + landmark_xy(lm, R_HIP)) / 2
        trunk_com = shoulder_center * 0.6 + hip_center * 0.4
        mass = BODY_SEGMENT_MASS_RATIOS["trunk"]
        weighted_sum += trunk_com * mass
        total_mass += mass
        segment_status["trunk"] = {"mass": mass, "center": trunk_com.tolist()}
    else:
        segment_status["trunk"] = None

    # 左上臂
    if all_valid(mask_row, [L_SHOULDER, L_ELBOW]):
        shoulder = landmark_xy(lm, L_SHOULDER)
        elbow = landmark_xy(lm, L_ELBOW)
        com = (shoulder + elbow) / 2
        mass = BODY_SEGMENT_MASS_RATIOS["upper_arm_l"]
        weighted_sum += com * mass
        total_mass += mass
        segment_status["upper_arm_l"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["upper_arm_l"] = None

    # 右上臂
    if all_valid(mask_row, [R_SHOULDER, R_ELBOW]):
        shoulder = landmark_xy(lm, R_SHOULDER)
        elbow = landmark_xy(lm, R_ELBOW)
        com = (shoulder + elbow) / 2
        mass = BODY_SEGMENT_MASS_RATIOS["upper_arm_r"]
        weighted_sum += com * mass
        total_mass += mass
        segment_status["upper_arm_r"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["upper_arm_r"] = None

    # 左前臂
    if all_valid(mask_row, [L_ELBOW, L_WRIST]):
        elbow = landmark_xy(lm, L_ELBOW)
        wrist = landmark_xy(lm, L_WRIST)
        com = (elbow + wrist) / 2
        mass = BODY_SEGMENT_MASS_RATIOS["forearm_l"]
        weighted_sum += com * mass
        total_mass += mass
        segment_status["forearm_l"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["forearm_l"] = None

    # 右前臂
    if all_valid(mask_row, [R_ELBOW, R_WRIST]):
        elbow = landmark_xy(lm, R_ELBOW)
        wrist = landmark_xy(lm, R_WRIST)
        com = (elbow + wrist) / 2
        mass = BODY_SEGMENT_MASS_RATIOS["forearm_r"]
        weighted_sum += com * mass
        total_mass += mass
        segment_status["forearm_r"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["forearm_r"] = None

    # 左手
    if all_valid(mask_row, [L_WRIST, L_INDEX, L_PINKY]):
        com = _compute_segment_center(lm, mask_row, [L_WRIST, L_INDEX, L_PINKY])
        if com is not None:
            mass = BODY_SEGMENT_MASS_RATIOS["hand_l"]
            weighted_sum += com * mass
            total_mass += mass
            segment_status["hand_l"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["hand_l"] = None

    # 右手
    if all_valid(mask_row, [R_WRIST, R_INDEX, R_PINKY]):
        com = _compute_segment_center(lm, mask_row, [R_WRIST, R_INDEX, R_PINKY])
        if com is not None:
            mass = BODY_SEGMENT_MASS_RATIOS["hand_r"]
            weighted_sum += com * mass
            total_mass += mass
            segment_status["hand_r"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["hand_r"] = None

    # 左大腿
    if all_valid(mask_row, [L_HIP, L_KNEE]):
        hip = landmark_xy(lm, L_HIP)
        knee = landmark_xy(lm, L_KNEE)
        com = (hip + knee) / 2
        mass = BODY_SEGMENT_MASS_RATIOS["thigh_l"]
        weighted_sum += com * mass
        total_mass += mass
        segment_status["thigh_l"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["thigh_l"] = None

    # 右大腿
    if all_valid(mask_row, [R_HIP, R_KNEE]):
        hip = landmark_xy(lm, R_HIP)
        knee = landmark_xy(lm, R_KNEE)
        com = (hip + knee) / 2
        mass = BODY_SEGMENT_MASS_RATIOS["thigh_r"]
        weighted_sum += com * mass
        total_mass += mass
        segment_status["thigh_r"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["thigh_r"] = None

    # 左小腿
    if all_valid(mask_row, [L_KNEE, L_ANKLE]):
        knee = landmark_xy(lm, L_KNEE)
        ankle = landmark_xy(lm, L_ANKLE)
        com = (knee + ankle) / 2
        mass = BODY_SEGMENT_MASS_RATIOS["shank_l"]
        weighted_sum += com * mass
        total_mass += mass
        segment_status["shank_l"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["shank_l"] = None

    # 右小腿
    if all_valid(mask_row, [R_KNEE, R_ANKLE]):
        knee = landmark_xy(lm, R_KNEE)
        ankle = landmark_xy(lm, R_ANKLE)
        com = (knee + ankle) / 2
        mass = BODY_SEGMENT_MASS_RATIOS["shank_r"]
        weighted_sum += com * mass
        total_mass += mass
        segment_status["shank_r"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["shank_r"] = None

    # 左脚
    if all_valid(mask_row, [L_ANKLE, L_HEEL, L_FOOT_INDEX]):
        com = _compute_segment_center(lm, mask_row, [L_ANKLE, L_HEEL, L_FOOT_INDEX])
        if com is not None:
            mass = BODY_SEGMENT_MASS_RATIOS["foot_l"]
            weighted_sum += com * mass
            total_mass += mass
            segment_status["foot_l"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["foot_l"] = None

    # 右脚
    if all_valid(mask_row, [R_ANKLE, R_HEEL, R_FOOT_INDEX]):
        com = _compute_segment_center(lm, mask_row, [R_ANKLE, R_HEEL, R_FOOT_INDEX])
        if com is not None:
            mass = BODY_SEGMENT_MASS_RATIOS["foot_r"]
            weighted_sum += com * mass
            total_mass += mass
            segment_status["foot_r"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["foot_r"] = None

    if total_mass < 0.5:  # 至少50%的身体质量可见
        return None, {"error": "insufficient_visible_mass", "total_mass": total_mass, "segments": segment_status}

    com = weighted_sum / total_mass
    return com, {"total_mass": total_mass, "segments": segment_status}


def eval_cog_com(
    landmarks: np.ndarray,
    *,
    fps: float,
    vis_thr: float = 0.5,
    com_r_back: float = 0.38,
    com_r_front: float = 0.62,
    trigger_ratio: float = 0.3,
    min_valid_frames: int = 15,
    valid_mask: np.ndarray | None = None,
) -> IndicatorResult:
    """
    基于分段质心(CoM)的重心判定（方案3）

    优势：
    - 更准确反映全身重心分布
    - 对"上身前倾/后仰"场景更鲁棒
    - 减少单点（髋部）代理带来的偏差

    参数说明：
    - com_r_back/com_r_front: CoM投影比例阈值（需重新标定，比髋部阈值更宽松）
    """
    if landmarks.size == 0:
        detail = {
            "primary_cause": "无有效帧",
            "valid_frames": 0,
            "total_frames": 0,
        }
        return IndicatorResult(status="无法判定", reason="无有效帧", detail=detail)

    mask = resolve_mask(landmarks, valid_mask, vis_thr)
    forward = backward = center = unknown = 0
    com_ratios = []

    for i, lm in enumerate(landmarks):
        mask_row = mask[i]
        # 计算该帧CoM
        com, _info = _compute_body_com_single(lm, mask_row)

        if com is None:
            unknown += 1
            continue

        d = facing_direction(lm, mask_row)
        if d is None:
            unknown += 1
            continue

        com_x = float(com[0]) * float(d)

        # 计算支撑面边界（两脚heel/toe）
        left_edges = foot_edges_x(lm, mask_row, side="L", dir_x=d)
        right_edges = foot_edges_x(lm, mask_row, side="R", dir_x=d)

        if left_edges is None or right_edges is None:
            unknown += 1
            continue

        support_back = float(min(left_edges[0], right_edges[0]))
        support_front = float(max(left_edges[1], right_edges[1]))
        denom = float(support_front - support_back)

        if denom < 0.02:
            unknown += 1
            continue

        # CoM投影比例
        r = (float(com_x) - support_back) / (denom + 1e-9)
        com_ratios.append(r)

        if float(r) <= float(com_r_back):
            backward += 1
        elif float(r) >= float(com_r_front):
            forward += 1
        else:
            center += 1

    valid = forward + backward + center

    # 计算com_ratio统计信息
    com_stats = {}
    if com_ratios:
        com_arr = np.array(com_ratios)
        com_stats = {
            "mean": float(np.mean(com_arr)),
            "std": float(np.std(com_arr)),
            "median": float(np.median(com_arr)),
            "min": float(np.min(com_arr)),
            "max": float(np.max(com_arr)),
        }

    detail = {
        "method": "CoM_segment_based",
        "valid_frames": int(valid),
        "total_frames": int(landmarks.shape[0]),
        "forward_frames": int(forward),
        "backward_frames": int(backward),
        "center_frames": int(center),
        "unknown_frames": int(unknown),
        "com_r_back": float(com_r_back),
        "com_r_front": float(com_r_front),
        "com_ratio_stats": com_stats,
        "trigger_ratio": float(trigger_ratio),
    }

    f_ratio = float(forward / max(1, valid))
    b_ratio = float(backward / max(1, valid))
    c_ratio = float(center / max(1, valid))
    detail["forward_ratio"] = float(f_ratio)
    detail["backward_ratio"] = float(b_ratio)
    detail["center_ratio"] = float(c_ratio)
    detail["unknown_ratio"] = float(unknown / max(1, int(landmarks.shape[0])))

    if valid < int(min_valid_frames):
        detail["primary_cause"] = "有效帧不足"
        return IndicatorResult(status="无法判定", reason="CoM：有效帧不足/遮挡", detail=detail)

    if f_ratio >= float(trigger_ratio) and b_ratio >= float(trigger_ratio):
        detail["primary_cause"] = "冲突"
        return IndicatorResult(status="无法判定", reason="CoM：前后判断冲突（可能斜拍/遮挡）", detail=detail)
    if f_ratio >= float(trigger_ratio):
        detail["primary_cause"] = "偏前"
        return IndicatorResult(status="不合格", reason="重心偏前（CoM）", detail=detail)
    if b_ratio >= float(trigger_ratio):
        detail["primary_cause"] = "偏后"
        return IndicatorResult(status="不合格", reason="重心偏后（CoM）", detail=detail)
    detail["primary_cause"] = "居中"
    return IndicatorResult(status="合格", reason="重心居中（CoM）", detail=detail)
