# -*- coding: utf-8 -*-
"""技术指标的结果结构与关键点契约：指标结果、整段结果、有效掩码解析，补齐 required/missing landmarks 与 backend（原 ``_attach_contract``）。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from core.pose import derive_valid_mask
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
from core.pose.validity import landmark_names, landmarks_missing_for_capabilities


Status = Literal["合格", "不合格", "无法判定"]

# 后端标识（YOLO 迁移 S4 / Issue #11）。本期 #10 结论为「仅预览」，tech_eval 主链路
# 仍是 MediaPipe；backend 字段让结果诚实声明数据来源，为后续 YOLO 路径预留。
BACKEND_MEDIAPIPE = "mediapipe"


@dataclass(frozen=True)
class IndicatorResult:
    status: Status
    reason: str
    detail: dict[str, Any] | None = None
    # YOLO 迁移 S4 / Issue #11：结构化契约字段。
    # - required_landmarks：该指标依赖的 BlazePose33 关键点名（声明）。
    # - missing_landmarks：运行时缺失/后端不支持的关键点名。
    # - backend：数据来源后端（本期主链路恒为 mediapipe）。
    # 这些字段在指标计算内默认空，由 _evaluate_from_arrays 装配层统一补齐，
    # 既不改动各 eval_* 内部判定逻辑（golden 不漂移），又让结果可被下游稳定解析。
    required_landmarks: tuple[str, ...] = ()
    missing_landmarks: tuple[str, ...] = ()
    backend: str = BACKEND_MEDIAPIPE


@dataclass(frozen=True)
class TechEvalResult:
    video_path: str
    pose_variant: str
    fps: float
    view_mode: str  # mixed|single
    front_segment: tuple[int, int] | None
    side_segment: tuple[int, int] | None

    cog_final: IndicatorResult
    cog_side: IndicatorResult
    cog_front: IndicatorResult
    cog_com: IndicatorResult | None  # 方案3：分段质心评估结果
    retract_speed: IndicatorResult
    force_sequence: IndicatorResult
    wrist_angle: IndicatorResult


def all_valid(mask_row: np.ndarray, idxs: tuple[int, ...] | list[int]) -> bool:
    """该帧给定关键点是否全部可用。

    读取由 ``derive_valid_mask`` 产出的单帧 ``(33,)`` bool 切片，不再现场比较
    ``lm[idx, 3] >= thr``（阈值只存在于 ``derive_valid_mask``，YOLO 迁移 Issue #5）。
    """
    return all(bool(mask_row[i]) for i in idxs)


def resolve_mask(landmarks: np.ndarray, valid_mask: np.ndarray | None, vis_thr: float) -> np.ndarray:
    """统一解析 ``valid_mask``。

    ``None`` 时由集中式 ``derive_valid_mask`` 按 ``vis_thr`` 现场推导，等价旧式
    ``visibility >= vis_thr``，保证旧调用方（不传 mask）行为逐位不变。
    """
    if valid_mask is None:
        return derive_valid_mask(landmarks, vis_thr)
    return np.asarray(valid_mask, dtype=bool)


# ---------------------------------------------------------------------------
# 指标关键点契约声明（YOLO 迁移 S4 / Issue #11）
# ---------------------------------------------------------------------------
# 每个技术指标声明其依赖的 BlazePose33 关键点索引。装配层据此补齐
# required_landmarks（声明）与 missing_landmarks（运行时/后端缺失），让结果诚实表达
# “依赖什么、缺了什么”，下游无需解析中文 reason。
COG_REQUIRED_INDICES: tuple[int, ...] = (
    NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP,
    L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
    L_HEEL, R_HEEL, L_FOOT_INDEX, R_FOOT_INDEX,
)
COG_COM_REQUIRED_INDICES: tuple[int, ...] = (
    NOSE, L_EAR, R_EAR,
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
    L_PINKY, R_PINKY, L_INDEX, R_INDEX,
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
    L_HEEL, R_HEEL, L_FOOT_INDEX, R_FOOT_INDEX,
)
RETRACT_REQUIRED_INDICES: tuple[int, ...] = (
    NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP,
)
FORCE_REQUIRED_INDICES: tuple[int, ...] = (
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP,
    L_HEEL, R_HEEL, L_FOOT_INDEX, R_FOOT_INDEX,
)
WRIST_REQUIRED_INDICES: tuple[int, ...] = (
    L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_INDEX, R_INDEX, L_PINKY, R_PINKY,
)


def runtime_missing_landmarks(indices: tuple[int, ...], mask: np.ndarray | None) -> tuple[str, ...]:
    """返回 ``indices`` 中在整段 ``mask`` 内从未有效的关键点名（运行时缺失）。

    “从未有效”= 该点在所有帧的 valid_mask 都为 False（被遮挡/低置信/段缺失），据此
    诚实声明该指标本次缺了哪些点。空段（无帧）视为全部缺失。
    """
    if mask is None:
        return landmark_names(indices)
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0 or arr.shape[0] == 0:
        return landmark_names(indices)
    ever_valid = np.any(arr, axis=0)  # (33,)
    missing_idx = [int(i) for i in indices if not bool(ever_valid[int(i)])]
    return landmark_names(missing_idx)


def attach_contract(
    result: "IndicatorResult",
    required_indices: tuple[int, ...],
    *,
    mask: np.ndarray | None,
    supported_capabilities=None,
    backend: str = BACKEND_MEDIAPIPE,
) -> "IndicatorResult":
    """给指标结果补齐 Issue #11 结构化契约字段（不改 status/reason/detail 判定）。

    missing_landmarks = 运行时从未有效的点 ∪ 后端结构性不支持的点（按能力分组）。
    本期 MediaPipe 路径 ``supported_capabilities=None``（全部支持），故只反映运行时缺失。
    """
    runtime_missing = runtime_missing_landmarks(required_indices, mask)
    backend_missing = landmarks_missing_for_capabilities(required_indices, supported_capabilities)
    # 合并去重，保持声明顺序。
    missing: list[str] = list(runtime_missing)
    for n in backend_missing:
        if n not in missing:
            missing.append(n)
    return replace(
        result,
        required_landmarks=landmark_names(required_indices),
        missing_landmarks=tuple(missing),
        backend=str(backend),
    )
