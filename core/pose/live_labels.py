# -*- coding: utf-8 -*-
"""实时动作标签：V 手势、举手、下蹲的单帧判定与中文标签表（原 ``_classify_pose_actions``、``ACTION_LABELS_ZH``）。

放在姿态层而不是评估层：``render.render_pose`` 要调用它，且它只做姿态分类、不参与评分（设计文档 §6.4）。
"""
from __future__ import annotations

import numpy as np

from core.foundation.geometry import angle_deg
from core.pose.landmarks import visibility as _visibility

ACTION_LABELS_ZH = {
    "V_SIGN": "✌（V 手势）",
    "HANDS_UP": "双手举起",
    "LEFT_HAND_UP": "左手举起",
    "RIGHT_HAND_UP": "右手举起",
    "SQUAT": "下蹲",
}


def actions_text(actions: list[str]) -> str:
    """把标签列表转成界面上显示的中文文本；没有标签时为 ``-``。"""
    return ", ".join(ACTION_LABELS_ZH.get(a, a) for a in actions) if actions else "-"


def _pt2d(lm, w: int, h: int) -> np.ndarray:
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def _finger_extended(lm, w: int, h: int, mcp: int, pip: int, dip: int) -> bool:
    # Use joint angle at PIP: closer to 180deg => straighter finger.
    return angle_deg(_pt2d(lm[mcp], w, h), _pt2d(lm[pip], w, h), _pt2d(lm[dip], w, h)) > 150.0


def _finger_curled(lm, w: int, h: int, mcp: int, pip: int, dip: int) -> bool:
    return angle_deg(_pt2d(lm[mcp], w, h), _pt2d(lm[pip], w, h), _pt2d(lm[dip], w, h)) < 150.0


def is_v_sign(hand_lm, w: int, h: int) -> bool:
    """
    Very lightweight "V sign" rule using only hand landmarks:
      - index & middle extended
      - ring & pinky curled
      - index tip and middle tip reasonably separated
    """
    if hand_lm is None:
        return False

    lm = hand_lm
    # Indices: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
    idx_ext = _finger_extended(lm, w, h, 5, 6, 7)
    mid_ext = _finger_extended(lm, w, h, 9, 10, 11)
    ring_cur = _finger_curled(lm, w, h, 13, 14, 15)
    pinky_cur = _finger_curled(lm, w, h, 17, 18, 19)

    if not (idx_ext and mid_ext and ring_cur and pinky_cur):
        return False

    index_tip = _pt2d(lm[8], w, h)
    middle_tip = _pt2d(lm[12], w, h)
    wrist = _pt2d(lm[0], w, h)
    middle_mcp = _pt2d(lm[9], w, h)
    hand_scale = float(np.linalg.norm(middle_mcp - wrist) + 1e-6)

    return float(np.linalg.norm(index_tip - middle_tip)) > 0.25 * hand_scale


def classify_pose_actions(landmarks, w: int, h: int) -> list[str]:
    """举手（左/右/双手）与下蹲的单帧判定；``landmarks`` 为 33 个 landmark 对象。"""
    if landmarks is None:
        return []

    lm = landmarks

    # MediaPipe pose landmark indices (BlazePose, 33 points)
    L_SHOULDER = 11
    R_SHOULDER = 12
    L_WRIST = 15
    R_WRIST = 16
    L_HIP = 23
    R_HIP = 24
    L_KNEE = 25
    R_KNEE = 26
    L_ANKLE = 27
    R_ANKLE = 28

    # Normalized coordinates: x,y in [0,1], y smaller => higher on screen.
    l_wrist = lm[L_WRIST]
    r_wrist = lm[R_WRIST]
    l_sh = lm[L_SHOULDER]
    r_sh = lm[R_SHOULDER]

    actions: list[str] = []

    left_hand_up = _visibility(l_wrist) > 0.5 and _visibility(l_sh) > 0.5 and (l_wrist.y < l_sh.y - 0.05)
    right_hand_up = _visibility(r_wrist) > 0.5 and _visibility(r_sh) > 0.5 and (r_wrist.y < r_sh.y - 0.05)
    if left_hand_up and right_hand_up:
        actions.append("HANDS_UP")
    elif left_hand_up:
        actions.append("LEFT_HAND_UP")
    elif right_hand_up:
        actions.append("RIGHT_HAND_UP")

    # Squat (very rough): both knee angles small + hip close to knee vertically.
    def knee_ok(side: str) -> bool:
        if side == "L":
            hip_lm = lm[L_HIP]
            knee_lm = lm[L_KNEE]
            ankle_lm = lm[L_ANKLE]
        else:
            hip_lm = lm[R_HIP]
            knee_lm = lm[R_KNEE]
            ankle_lm = lm[R_ANKLE]

        v_ok = _visibility(hip_lm) > 0.5 and _visibility(knee_lm) > 0.5 and _visibility(ankle_lm) > 0.5
        if not v_ok:
            return False

        hip = _pt2d(hip_lm, w, h)
        knee = _pt2d(knee_lm, w, h)
        ankle = _pt2d(ankle_lm, w, h)

        ang = angle_deg(hip, knee, ankle)  # 180 ~= straight
        hip_knee_dist = abs((hip[1] - knee[1]) / max(h, 1))  # normalized vertical distance
        return ang < 125.0 and hip_knee_dist < 0.18

    if knee_ok("L") and knee_ok("R"):
        actions.append("SQUAT")

    return actions
