# -*- coding: utf-8 -*-
"""侧别的唯一实现：左右式→前后手的关键点、前腿推断、踢腿腿推断（设计文档 §7.1、§3 第 22 项）。

三者回答的是不同问题，算法各自保留：
- ``sides_for_stance``：学生练习按左右式取前/后侧肢体（原 ``feedback_phases.limb_indices``）；
- ``infer_front_leg``：技术指标按逐帧朝向推断哪只脚在前（原 ``tech_eval._infer_front_leg_side``）；
- ``infer_kick_leg``：考试踢腿质量按踝的帧间位移判定踢腿腿（原 ``kick_quality._kick_stand_sides``）。
"""
from __future__ import annotations

import numpy as np

from core.foundation.geometry import landmark_xy
from core.pose.landmarks import (
    L_ANKLE,
    L_EAR,
    L_FOOT_INDEX,
    L_HEEL,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    MOUTH_L,
    MOUTH_R,
    NOSE,
    R_ANKLE,
    R_EAR,
    R_FOOT_INDEX,
    R_HEEL,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
)

_LEFT_LIMB = (11, 13, 15, 23, 25, 27, 29, 31)
_RIGHT_LIMB = (12, 14, 16, 24, 26, 28, 30, 32)


def sides_for_stance(role: str, stance: str) -> tuple[int, ...]:
    """``role``（front/rear）在 ``stance``（left/right）下的肢体关键点：肩、肘、腕、髋、膝、踝、跟、脚尖。"""
    left = (stance == "left") == (role != "rear")
    return _LEFT_LIMB if left else _RIGHT_LIMB


def side_label(role: str, stance: str) -> str:
    """前/后侧在左右式下对应的"左/右"文案；不分前后的项目返回空串（练习检查项的身体部位）。"""
    if role == "front":
        return "左" if stance == "left" else "右"
    if role == "rear":
        return "右" if stance == "left" else "左"
    return ""


def _valid(mask_row: np.ndarray, idxs) -> bool:
    return all(bool(mask_row[i]) for i in idxs)


def center_x(lm: np.ndarray, mask_row: np.ndarray, idx_a: int, idx_b: int) -> float | None:
    """两点 x 的中点；只有一点有效时取该点，都无效为 None。"""
    va, vb = bool(mask_row[idx_a]), bool(mask_row[idx_b])
    xa, xb = float(landmark_xy(lm, idx_a)[0]), float(landmark_xy(lm, idx_b)[0])
    if va and vb:
        return 0.5 * (xa + xb)
    if va:
        return xa
    if vb:
        return xb
    return None


def facing_direction(lm: np.ndarray, mask_row: np.ndarray) -> float | None:
    """单帧面朝方向：脸（鼻，缺则口角/耳的均值）在髋（缺则肩）中点的右侧为 +1，左侧为 -1；无法判断为 None。"""
    ref = center_x(lm, mask_row, L_HIP, R_HIP)
    if ref is None:
        ref = center_x(lm, mask_row, L_SHOULDER, R_SHOULDER)
    if ref is None:
        return None

    face_x = None
    if bool(mask_row[NOSE]):
        face_x = float(landmark_xy(lm, NOSE)[0])
    else:
        cand: list[float] = []
        for idx in (MOUTH_L, MOUTH_R, L_EAR, R_EAR):
            if bool(mask_row[idx]):
                cand.append(float(landmark_xy(lm, idx)[0]))
        if cand:
            face_x = float(np.mean(np.asarray(cand, dtype=np.float32)))
    if face_x is None:
        return None

    dx = float(face_x) - float(ref)
    if not np.isfinite(dx):
        return None
    return 1.0 if dx >= 0.0 else -1.0


def foot_edges_x(lm: np.ndarray, mask_row: np.ndarray, *, side: str, dir_x: float) -> tuple[float, float] | None:
    """一只脚沿面朝方向的前后缘 x（跟、脚尖都有效时取二者，否则退到踝）；乘 ``dir_x`` 后"越大越靠前"。"""
    if side.upper() == "L":
        heel, toe, ankle = L_HEEL, L_FOOT_INDEX, L_ANKLE
    else:
        heel, toe, ankle = R_HEEL, R_FOOT_INDEX, R_ANKLE

    if _valid(mask_row, (heel, toe)):
        hx = float(landmark_xy(lm, heel)[0]) * float(dir_x)
        tx = float(landmark_xy(lm, toe)[0]) * float(dir_x)
        return (min(hx, tx), max(hx, tx))

    if bool(mask_row[ankle]):
        ax = float(landmark_xy(lm, ankle)[0]) * float(dir_x)
        return (ax, ax)

    return None


def infer_front_leg(landmarks: np.ndarray, valid_mask: np.ndarray, *, fallback: str = "L") -> str:
    """按逐帧面朝方向统计哪只踝更靠前，返回 ``"L"``/``"R"``；可判断的帧不足 10 帧时返回 ``fallback``。"""
    cnt_l = 0
    cnt_r = 0
    for i, lm in enumerate(landmarks):
        mask_row = valid_mask[i]
        d = facing_direction(lm, mask_row)
        if d is None:
            continue
        if not _valid(mask_row, (L_ANKLE, R_ANKLE)):
            continue
        xl = float(landmark_xy(lm, L_ANKLE)[0]) * float(d)
        xr = float(landmark_xy(lm, R_ANKLE)[0]) * float(d)
        if xl >= xr:
            cnt_l += 1
        else:
            cnt_r += 1

    if (cnt_l + cnt_r) < 10:
        return fallback.upper()
    return "L" if cnt_l >= cnt_r else "R"


def infer_kick_leg(landmarks: np.ndarray, valid: np.ndarray) -> tuple[int, int, int, int]:
    """窗口内踝的帧间位移大的一侧是踢腿腿；返回 ``(踢腿踝, 踢腿膝, 支撑髋, 支撑膝)``。"""
    def _travel(ankle_idx: int) -> float:
        ys = np.array(
            [float(landmarks[i, ankle_idx, 1]) for i in range(len(landmarks)) if valid[i]],
            dtype=np.float32,
        )
        xs = np.array(
            [float(landmarks[i, ankle_idx, 0]) for i in range(len(landmarks)) if valid[i]],
            dtype=np.float32,
        )
        if ys.size < 2:
            return 0.0
        return float(np.abs(np.diff(ys)).sum() + np.abs(np.diff(xs)).sum())

    left_travel = _travel(L_ANKLE)
    right_travel = _travel(R_ANKLE)
    if left_travel >= right_travel:
        return L_ANKLE, L_KNEE, R_HIP, R_KNEE
    return R_ANKLE, R_KNEE, L_HIP, L_KNEE
