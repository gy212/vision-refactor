# -*- coding: utf-8 -*-
"""重心指标：侧面重心、正面重心，以及侧面优先取出拳窗口的重心评估（可附带分段质心结果）。"""
from __future__ import annotations

import numpy as np

from core.evaluation.body_sides import center_x, facing_direction, foot_edges_x, infer_front_leg
from core.evaluation.techeval.com import eval_cog_com
from core.evaluation.techeval.contract import IndicatorResult, all_valid, resolve_mask
from core.evaluation.techeval.events import detect_retract_events_side, subset_by_intervals
from core.foundation import angle_deg, landmark_xy
from core.pose import DEFAULT_VALID_CONF_THR
from core.pose.landmarks import L_ANKLE, L_FOOT_INDEX, L_HIP, L_KNEE, R_ANKLE, R_FOOT_INDEX, R_HIP, R_KNEE


def eval_cog_side(
    landmarks: np.ndarray,
    *,
    fps: float,
    vis_thr: float = 0.5,
    knee_straight_thr: float = 175.0,
    hip_r_back: float = 0.35,
    hip_r_front: float = 0.65,
    trigger_ratio: float = 0.3,
    min_valid_frames: int = 15,
    stance_fallback: str = "L",
    valid_mask: np.ndarray | None = None,
) -> IndicatorResult:
    if landmarks.size == 0:
        detail = {
            "primary_cause": "无有效帧",
            "valid_frames": 0,
            "total_frames": 0,
        }
        return IndicatorResult(status="无法判定", reason="无有效帧", detail=detail)

    mask = resolve_mask(landmarks, valid_mask, vis_thr)
    front_leg = infer_front_leg(landmarks, mask, fallback=stance_fallback)
    back_leg = "R" if front_leg == "L" else "L"

    forward = backward = center = unknown = 0

    for i, lm in enumerate(landmarks):
        mask_row = mask[i]
        d = facing_direction(lm, mask_row)
        if d is None:
            unknown += 1
            continue

        def knee_angle(leg: str) -> float | None:
            if leg == "L":
                idxs = (L_HIP, L_KNEE, L_ANKLE)
                hip, knee, ankle = L_HIP, L_KNEE, L_ANKLE
            else:
                idxs = (R_HIP, R_KNEE, R_ANKLE)
                hip, knee, ankle = R_HIP, R_KNEE, R_ANKLE
            if not all_valid(mask_row, idxs):
                return None
            return angle_deg(landmark_xy(lm, hip), landmark_xy(lm, knee), landmark_xy(lm, ankle))

        ang_front = knee_angle(front_leg)
        ang_back = knee_angle(back_leg)

        # Scheme 2 (side): straight back-leg => forward; straight front-leg => backward.
        if ang_back is not None and float(ang_back) >= float(knee_straight_thr):
            forward += 1
            continue
        if ang_front is not None and float(ang_front) >= float(knee_straight_thr):
            backward += 1
            continue

        # Center by knee projection on the front foot.
        if front_leg == "L":
            knee_idx = L_KNEE
            foot_edges = foot_edges_x(lm, mask_row, side="L", dir_x=d)
        else:
            knee_idx = R_KNEE
            foot_edges = foot_edges_x(lm, mask_row, side="R", dir_x=d)

        if foot_edges is not None and bool(mask_row[knee_idx]):
            fb, ff = foot_edges
            foot_len = float(ff - fb)
            if foot_len >= 0.02:
                kx = float(landmark_xy(lm, knee_idx)[0]) * float(d)
                t = (kx - fb) / (foot_len + 1e-9)
                if 0.25 <= float(t) <= 0.75:
                    center += 1
                    continue

        # Scheme 1: hip projection ratio in the overall support area.
        hip_x = center_x(lm, mask_row, L_HIP, R_HIP)
        if hip_x is None:
            unknown += 1
            continue

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

        r = ((float(hip_x) * float(d)) - support_back) / (denom + 1e-9)
        if float(r) <= float(hip_r_back):
            backward += 1
        elif float(r) >= float(hip_r_front):
            forward += 1
        else:
            center += 1

    valid = forward + backward + center
    detail = {
        "valid_frames": int(valid),
        "total_frames": int(landmarks.shape[0]),
        "forward_frames": int(forward),
        "backward_frames": int(backward),
        "center_frames": int(center),
        "unknown_frames": int(unknown),
        "front_leg": str(front_leg),
        "knee_straight_thr": float(knee_straight_thr),
        "hip_r_back": float(hip_r_back),
        "hip_r_front": float(hip_r_front),
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
        return IndicatorResult(status="无法判定", reason="侧面：有效帧不足/遮挡", detail=detail)

    if f_ratio >= float(trigger_ratio) and b_ratio >= float(trigger_ratio):
        detail["primary_cause"] = "冲突"
        return IndicatorResult(status="无法判定", reason="侧面：前后判断冲突（可能斜拍/遮挡）", detail=detail)
    if f_ratio >= float(trigger_ratio):
        detail["primary_cause"] = "偏前"
        return IndicatorResult(status="不合格", reason="侧面：重心偏前", detail=detail)
    if b_ratio >= float(trigger_ratio):
        detail["primary_cause"] = "偏后"
        return IndicatorResult(status="不合格", reason="侧面：重心偏后", detail=detail)
    detail["primary_cause"] = "居中"
    return IndicatorResult(status="合格", reason="侧面：重心居中", detail=detail)


def eval_cog_front(
    landmarks: np.ndarray,
    *,
    fps: float,
    stance: str = "left",
    vis_thr: float = 0.5,
    knee_straight_thr: float = 175.0,
    ankle_acute_thr: float = 90.0,
    trigger_ratio: float = 0.3,
    min_valid_frames: int = 15,
    valid_mask: np.ndarray | None = None,
) -> IndicatorResult:
    if landmarks.size == 0:
        detail = {
            "primary_cause": "无有效帧",
            "valid_frames": 0,
            "total_frames": 0,
        }
        return IndicatorResult(status="无法判定", reason="无有效帧", detail=detail)

    mask = resolve_mask(landmarks, valid_mask, vis_thr)
    stance = (stance or "left").lower()
    front_leg = "L" if stance == "left" else "R"

    forward = backward = center = unknown = 0

    for i, lm in enumerate(landmarks):
        mask_row = mask[i]
        if front_leg == "L":
            knee_idxs = (L_HIP, L_KNEE, L_ANKLE)
            ankle_idxs = (L_KNEE, L_ANKLE, L_FOOT_INDEX)
            hip, knee, ankle, toe = L_HIP, L_KNEE, L_ANKLE, L_FOOT_INDEX
        else:
            knee_idxs = (R_HIP, R_KNEE, R_ANKLE)
            ankle_idxs = (R_KNEE, R_ANKLE, R_FOOT_INDEX)
            hip, knee, ankle, toe = R_HIP, R_KNEE, R_ANKLE, R_FOOT_INDEX

        knee_ang = None
        ankle_ang = None
        if all_valid(mask_row, knee_idxs):
            knee_ang = angle_deg(landmark_xy(lm, hip), landmark_xy(lm, knee), landmark_xy(lm, ankle))
        if all_valid(mask_row, ankle_idxs):
            ankle_ang = angle_deg(landmark_xy(lm, knee), landmark_xy(lm, ankle), landmark_xy(lm, toe))

        if knee_ang is None and ankle_ang is None:
            unknown += 1
            continue

        # Front rules (from requirements):
        # - front knee ~180 => backward
        # - front ankle acute (<90) => forward
        k_back = (knee_ang is not None) and (float(knee_ang) >= float(knee_straight_thr))
        a_fwd = (ankle_ang is not None) and (float(ankle_ang) < float(ankle_acute_thr))

        if k_back and a_fwd:
            unknown += 1
            continue
        if k_back:
            backward += 1
        elif a_fwd:
            forward += 1
        else:
            center += 1

    valid = forward + backward + center
    detail = {
        "valid_frames": int(valid),
        "total_frames": int(landmarks.shape[0]),
        "forward_frames": int(forward),
        "backward_frames": int(backward),
        "center_frames": int(center),
        "unknown_frames": int(unknown),
        "front_leg": str(front_leg),
        "knee_straight_thr": float(knee_straight_thr),
        "ankle_acute_thr": float(ankle_acute_thr),
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
        return IndicatorResult(status="无法判定", reason="正面：有效帧不足/遮挡", detail=detail)

    if f_ratio >= float(trigger_ratio) and b_ratio >= float(trigger_ratio):
        detail["primary_cause"] = "冲突"
        return IndicatorResult(status="无法判定", reason="正面：前后判断冲突（可能斜拍/遮挡）", detail=detail)
    if f_ratio >= float(trigger_ratio):
        detail["primary_cause"] = "偏前"
        return IndicatorResult(status="不合格", reason="正面：重心偏前", detail=detail)
    if b_ratio >= float(trigger_ratio):
        detail["primary_cause"] = "偏后"
        return IndicatorResult(status="不合格", reason="正面：重心偏后", detail=detail)
    detail["primary_cause"] = "居中"
    return IndicatorResult(status="合格", reason="正面：重心居中", detail=detail)


def eval_cog_side_prefer_punch_windows(
    landmarks: np.ndarray,
    *,
    fps: float,
    use_com: bool = False,
    valid_mask: np.ndarray | None = None,
) -> tuple[IndicatorResult, IndicatorResult | None]:
    """
    侧面重心评估（优先出拳窗口）

    Args:
        use_com: 是否同时返回CoM（方案3）评估结果
        valid_mask: ``(T,33)`` 有效性掩码，``None`` 时现场推导（行为不变）。

    Returns:
        (主评估结果, CoM评估结果或None)
    """
    mask = resolve_mask(landmarks, valid_mask, DEFAULT_VALID_CONF_THR)
    seg_res = eval_cog_side(landmarks, fps=fps, stance_fallback="L", valid_mask=mask)
    events = detect_retract_events_side(
        landmarks,
        fps=fps,
        vis_thr=0.5,
        ext_angle_thr=160.0,
        retract_angle_thr=45.0,
        ext_forward_dx=0.02,
        retract_sec_thr=0.2,
        max_window_sec=1.0,
        valid_mask=mask,
    )

    com_res: IndicatorResult | None = None

    if not events:
        if seg_res.detail is not None:
            seg_res.detail["eval_scope"] = "segment"
        # 无出拳事件时，全段使用CoM评估
        if use_com:
            com_res = eval_cog_com(landmarks, fps=fps, valid_mask=mask)
        return seg_res, com_res

    pre = int(round(0.3 * float(fps)))
    post = int(round(0.1 * float(fps)))
    n = int(landmarks.shape[0])

    intervals: list[tuple[int, int]] = []
    for e in events:
        s = int(e.get("start") or 0)
        end = e.get("end")
        if end is None:
            end = min(n - 1, s + int(round(1.0 * float(fps))))
        intervals.append((s - pre, int(end) + post))

    subset, used, subset_mask = subset_by_intervals(landmarks, intervals, mask)

    # 方案1+2：基于髋部/膝角的评估
    win_res = eval_cog_side(subset, fps=fps, stance_fallback="L", valid_mask=subset_mask)
    if win_res.detail is not None:
        win_res.detail["eval_scope"] = "punch_windows"
        win_res.detail["punch_windows"] = used
        win_res.detail["punch_events"] = int(len(events))

    # 方案3：基于分段质心的评估（出拳窗口）
    if use_com:
        com_res = eval_cog_com(subset, fps=fps, valid_mask=subset_mask)
        if com_res.detail is not None:
            com_res.detail["eval_scope"] = "punch_windows"
            com_res.detail["punch_windows"] = used
            com_res.detail["punch_events"] = int(len(events))

    if win_res.status != "无法判定":
        return win_res, com_res

    # 出拳窗口无法判定时，回退到全段评估
    if seg_res.detail is not None:
        seg_res.detail["eval_scope"] = "segment_fallback"
        seg_res.detail["punch_windows"] = used
        seg_res.detail["punch_events"] = int(len(events))

    return seg_res, com_res
