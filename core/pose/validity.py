# -*- coding: utf-8 -*-
"""点位有效性的唯一判据 ``derive_valid_mask``，以及 BlazePose33 关键点名称与后端能力分组。

能力分组描述"后端结构上能否提供这个点"，是有效性的另一面：COCO17 缺嘴角、脚跟脚尖、手指，
这些点在 YOLO 路径下不得伪造成有效点。
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# valid_mask 契约（YOLO 迁移 S1 / Issue #5）
# ---------------------------------------------------------------------------
# “此点是否可用”的唯一判据集中在此，不再散落 `lm[idx, 3] >= thr` / `_lm_vis(...) >= thr`。
# - MediaPipe 路径：第 4 通道存 visibility，按 `visibility >= 0.5` 灌入，
#   故 validity_policy=visibility_thr、valid_conf_thr=0.5（视为已标定）。
# - 阈值只允许出现在 `derive_valid_mask`，避免再次散落。
DEFAULT_VALID_CONF_THR: float = 0.5
# MediaPipe 路径的有效性策略名（写入 meta/config，便于下游区分可信/待标定）。
MEDIAPIPE_VALIDITY_POLICY: str = "visibility_thr"


def derive_valid_mask(landmarks: np.ndarray, thr: float = DEFAULT_VALID_CONF_THR) -> np.ndarray:
    """集中式有效性判据：``landmarks[..., 3] >= thr``。

    这是把散落在 ``analysis/tech_eval.py`` 与 ``core/rule_scoring.py`` 的
    ``lm[idx, 3] >= thr`` 风格判据统一收口的唯一入口（YOLO 迁移 Issue #5）。
    下游一律读返回的 mask，不再各自比较第 4 通道。

    参数
    ----
    landmarks:
        ``(33, 4)`` 单帧或 ``(T, 33, 4)`` 序列。第 4 通道按 ``confidence_kind``
        语义存储（MediaPipe 为 visibility）。
    thr:
        有效阈值。MediaPipe 侧默认 ``0.5``（已标定）。

    返回
    ----
    与输入对应的 bool mask：单帧返回 ``(33,)``，序列返回 ``(T, 33)``。
    其逐元素结果与旧式 ``float(lm[idx, 3]) >= thr`` 完全一致（float32→float64
    拓宽无精度损失，``0.5`` 在两种精度下均可精确表示）。
    """
    arr = np.asarray(landmarks)
    if arr.ndim not in (2, 3) or arr.shape[-1] < 4:
        raise ValueError(
            f"derive_valid_mask 期望 (33,4) 或 (T,33,4) 的关键点数组，实际 shape={arr.shape}"
        )
    return arr[..., 3] >= float(thr)


# ---------------------------------------------------------------------------
# 关键点命名与能力分组（YOLO 迁移 S4 / Issue #11）
# ---------------------------------------------------------------------------
# 评估链路要能诚实声明「依赖哪些关键点」「运行时缺了哪些点」。本节提供集中的
# BlazePose33 名称表与「后端可提供性」能力分组，供 `rule_scoring`（规则）与
# `tech_eval`（技术指标）统一声明 ``required_landmarks`` / ``required_capabilities``
# 与运行时 ``missing_landmarks``，下游用稳定字段判断而非靠中文字符串。
#
# 能力分组按 COCO17（YOLO）可映射性划分：COCO17 缺眼细分(1,3,4,6)/
# 嘴角(9,10)/手指(17-22)/脚跟脚尖(29-32)，故这些分组在 YOLO-only 下
# 结构性不可用（缺点指标）。本期 #10 标定结论为「仅预览」，YOLO partial eval
# 未启用，这些声明先服务 MediaPipe 侧的结构化状态，并为后续 YOLO 路径预留判据。
BLAZE33_LANDMARK_NAMES: tuple[str, ...] = (
    "nose",            # 0
    "left_eye_inner",  # 1
    "left_eye",        # 2
    "left_eye_outer",  # 3
    "right_eye_inner",  # 4
    "right_eye",       # 5
    "right_eye_outer",  # 6
    "left_ear",        # 7
    "right_ear",       # 8
    "mouth_left",      # 9
    "mouth_right",     # 10
    "left_shoulder",   # 11
    "right_shoulder",  # 12
    "left_elbow",      # 13
    "right_elbow",     # 14
    "left_wrist",      # 15
    "right_wrist",     # 16
    "left_pinky",      # 17
    "right_pinky",     # 18
    "left_index",      # 19
    "right_index",     # 20
    "left_thumb",      # 21
    "right_thumb",     # 22
    "left_hip",        # 23
    "right_hip",       # 24
    "left_knee",       # 25
    "right_knee",      # 26
    "left_ankle",      # 27
    "right_ankle",     # 28
    "left_heel",       # 29
    "right_heel",      # 30
    "left_foot_index",  # 31
    "right_foot_index",  # 32
)

# 能力分组标签。
CAP_FACE_CENTER = "face_center"  # 鼻 0（COCO17 有）
CAP_EYES = "eyes"                # 左/右眼中心 2,5（COCO17 有近似映射）
CAP_EYE_DETAILS = "eye_details"  # 眼细分 1,3,4,6（COCO17 无）
CAP_EARS = "ears"                # 耳 7,8（COCO17 有）
CAP_MOUTH = "mouth"              # 嘴角 9,10（COCO17 无）
CAP_ARMS = "arms"                # 肩/肘/腕 11-16（COCO17 有）
CAP_HANDS = "hands"              # 手指 17-22（COCO17 无）
CAP_LEGS = "legs"                # 髋/膝/踝 23-28（COCO17 有）
CAP_FEET = "feet"                # 脚跟/脚尖 29-32（COCO17 无）

# COCO17（YOLO）结构性支持的能力集合；缺 eye_details/mouth/hands/feet。
# 仅作声明用途，本期不接 YOLO partial eval。
COCO17_SUPPORTED_CAPABILITIES: frozenset[str] = frozenset(
    {CAP_FACE_CENTER, CAP_EYES, CAP_EARS, CAP_ARMS, CAP_LEGS}
)

_LANDMARK_CAPABILITY: dict[int, str] = {
    0: CAP_FACE_CENTER,
    1: CAP_EYE_DETAILS, 2: CAP_EYES, 3: CAP_EYE_DETAILS,
    4: CAP_EYE_DETAILS, 5: CAP_EYES, 6: CAP_EYE_DETAILS,
    7: CAP_EARS, 8: CAP_EARS,
    9: CAP_MOUTH, 10: CAP_MOUTH,
    11: CAP_ARMS, 12: CAP_ARMS, 13: CAP_ARMS, 14: CAP_ARMS, 15: CAP_ARMS, 16: CAP_ARMS,
    17: CAP_HANDS, 18: CAP_HANDS, 19: CAP_HANDS, 20: CAP_HANDS, 21: CAP_HANDS, 22: CAP_HANDS,
    23: CAP_LEGS, 24: CAP_LEGS, 25: CAP_LEGS, 26: CAP_LEGS, 27: CAP_LEGS, 28: CAP_LEGS,
    29: CAP_FEET, 30: CAP_FEET, 31: CAP_FEET, 32: CAP_FEET,
}


def landmark_names(indices) -> tuple[str, ...]:
    """把 BlazePose33 索引序列映射为稳定的英文名称元组（顺序保持输入顺序）。"""
    return tuple(BLAZE33_LANDMARK_NAMES[int(i)] for i in indices)


def landmark_capabilities(indices) -> tuple[str, ...]:
    """把索引序列归并为去重后的能力分组元组（按首次出现顺序）。

    供规则 / 指标声明 ``required_capabilities``。例如依赖嘴角的规则会带 ``mouth``，
    依赖脚跟脚尖的指标会带 ``feet``——这些在 COCO17 下结构性缺失。
    """
    caps: list[str] = []
    for i in indices:
        c = _LANDMARK_CAPABILITY.get(int(i))
        if c is not None and c not in caps:
            caps.append(c)
    return tuple(caps)


def landmarks_missing_for_capabilities(indices, supported_capabilities) -> tuple[str, ...]:
    """返回 ``indices`` 中其能力分组不在 ``supported_capabilities`` 内的关键点名称。

    用于「后端结构性缺点」判定：例如 YOLO（COCO17）不支持 ``feet`` 能力时，依赖
    脚跟/脚尖的规则会被判为 ``skipped`` + ``skip_reason=missing_landmarks``，并由本
    函数列出具体缺失的关键点名（``left_heel`` 等）。``supported_capabilities=None``
    视为「全部支持」（MediaPipe full），返回空元组。
    """
    if supported_capabilities is None:
        return ()
    supported = frozenset(supported_capabilities)
    return tuple(
        BLAZE33_LANDMARK_NAMES[int(i)]
        for i in indices
        if _LANDMARK_CAPABILITY.get(int(i)) not in supported
    )
