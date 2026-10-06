# -*- coding: utf-8 -*-
"""``body_core_v1`` 特征提取：MediaPipe 与 YOLO 两后端共用同一 normalizer 产出 ``(T, 12, 2)`` 特征。

匹配与模板生成留在 ``core.body_core_compare``（P3 迁入评估层）。YOLO 只允许 ``body_core_v1``，分数
未标定（``calibration_status=unvalidated``），不进入正式评分。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.pose.layouts import BODY_CORE_V1
from core.pose.normalize import normalize_pose_body_core_v1
from core.pose.raw_series import extract_pose_raw

# 支持的后端。YOLO 只允许 body_core_v1（在评分入口强制）。
BACKEND_MEDIAPIPE = "mediapipe"
BACKEND_YOLO = "yolo"


# --------------------------------------------------------------------------- #
# 特征提取（两后端共享 body_core_v1 normalizer）
# --------------------------------------------------------------------------- #
def _extract_body_core_mediapipe(
    video_path: Path,
    *,
    pose_variant: str = "full",
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> tuple[np.ndarray, float]:
    """MediaPipe 路径：抽取 ``body_core_v1`` ``(T,12,2)`` 特征。

    复用既有 ``rule_scoring.extract_pose_raw``（与旧链路同款 VIDEO-mode 管线）拿到
    ``(T,33,4)`` 原始关键点与 ``valid_mask``，再逐帧套用共享
    ``normalize_pose_body_core_v1``。缺帧/核心点无效帧按 layout shape 补零或沿用上一帧，
    与旧路径风格一致。

    复用 ``extract_pose_raw`` 而非自建解码循环，既减少重复，也让 golden harness 能确定性回放本路径。
    """
    landmarks, meta = extract_pose_raw(
        video_path,
        pose_variant=pose_variant,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    fps = float(meta.get("fps") or 30.0)

    feats, _valid_core_frames = _normalize_body_core_sequence(
        landmarks,
        valid_mask=meta.get("valid_mask"),
    )

    if feats.shape[0] == 0:
        raise RuntimeError(f"未能从视频提取 body_core_v1 特征：{video_path}")
    return feats, float(fps)


def _normalize_body_core_sequence(
    landmarks: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Normalize BlazePose33-like rows into ``body_core_v1`` features.

    When ``valid_mask`` is present, a frame is usable only if all 12 body-core source
    points are valid. Invalid frames are gap-filled with the previous usable feature
    (or zeros for a leading gap), so low-confidence coordinates never enter DTW.
    """
    arr = np.asarray(landmarks)
    if arr.ndim != 3 or arr.shape[1] < 33 or arr.shape[2] < 2:
        raise ValueError(f"body_core_v1 期望 landmarks shape (T,33,>=2)，实际 {arr.shape}")

    mask = None if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if mask is not None and (mask.ndim != 2 or mask.shape[0] != arr.shape[0] or mask.shape[1] < 33):
        raise ValueError(f"body_core_v1 期望 valid_mask shape (T,33)，实际 {mask.shape}")

    zero = np.zeros(tuple(int(x) for x in BODY_CORE_V1.shape), dtype=np.float32)
    core_idx = np.asarray(BODY_CORE_V1.source_indices, dtype=int)
    feats: list[np.ndarray] = []
    valid_core_frames = 0

    for t in range(int(arr.shape[0])):
        core_valid = True if mask is None else bool(np.all(mask[t, core_idx]))
        f = normalize_pose_body_core_v1(arr[t]) if core_valid else None
        if f is None:
            f = feats[-1].copy() if feats else zero.copy()
        else:
            valid_core_frames += 1
        feats.append(f)

    if not feats:
        return np.zeros((0, *BODY_CORE_V1.shape), dtype=np.float32), 0
    return np.stack(feats, axis=0).astype(np.float32), int(valid_core_frames)


def _extract_body_core_yolo(
    video_path: Path,
    *,
    yolo_model=None,
    valid_conf_thr: float | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    should_stop=None,
) -> tuple[np.ndarray, float, dict]:
    """YOLO 路径：抽取 ``body_core_v1`` ``(T,12,2)`` 特征。

    先用 ``extract_yolo_landmark_series`` 拿到序列层 ``(T,33,4)`` + ``valid_mask[T,33]``
    （只返回 numpy，遵守序列层契约），先按 ``valid_mask`` 判定核心 12 点是否可用，
    再逐帧套用共享 ``normalize_pose_body_core_v1``。

    返回 ``(features[T,12,2], fps, yolo_meta)``；``yolo_meta`` 含
    ``valid_conf_thr`` / ``calibration_status`` / 多人信息，供闭环 meta 透传。

    缺帧（无人帧零行 / 核心点无效 / normalizer 返回 None）按 layout shape 补零或沿用上一帧。
    body_core_v1 的 12 点全部落在 COCO17 可映射点，因此该布局在 YOLO 路径下是真实点。
    """
    from core.pose.backends.yolo import (
        DEFAULT_YOLO_VALID_CONF_THR,
        extract_yolo_landmark_series,
    )

    thr = float(DEFAULT_YOLO_VALID_CONF_THR if valid_conf_thr is None else valid_conf_thr)
    landmarks, valid_mask, yolo_meta = extract_yolo_landmark_series(
        video_path,
        yolo_model=yolo_model,
        valid_conf_thr=thr,
        start_frame=start_frame,
        end_frame=end_frame,
        should_stop=should_stop,
    )
    if should_stop is not None and should_stop():
        raise InterruptedError("YOLO body_core analysis cancelled")
    fps = float(yolo_meta.get("fps") or 30.0)

    features, valid_core_frames = _normalize_body_core_sequence(landmarks, valid_mask=valid_mask)
    yolo_meta = dict(yolo_meta)
    yolo_meta["body_core_valid_frame_ratio"] = (
        float(valid_core_frames) / float(features.shape[0]) if features.shape[0] else 0.0
    )
    # 多人闸门字段（multi_person_detected / review_required / gate_status /
    # max_persons / multi_person_frames）已由 extract_yolo_landmark_series 落进 yolo_meta，
    # 此处随 backend_meta 透传给评分入口（match_body_core_template）做拒绝/降级。
    return features, fps, yolo_meta


def extract_body_core_features(
    video_path: str | Path,
    *,
    backend: str,
    pose_variant: str = "full",
    yolo_model=None,
    valid_conf_thr: float | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    should_stop=None,
) -> tuple[np.ndarray, float, dict]:
    """按后端抽取 ``body_core_v1`` ``(T,12,2)`` 特征，返回 ``(features, fps, backend_meta)``。"""
    backend = str(backend).lower()
    video_path = Path(video_path)
    if backend == BACKEND_MEDIAPIPE:
        features, fps = _extract_body_core_mediapipe(
            video_path,
            pose_variant=pose_variant,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        return features, fps, {"backend": BACKEND_MEDIAPIPE}
    if backend == BACKEND_YOLO:
        return _extract_body_core_yolo(
            video_path,
            yolo_model=yolo_model,
            valid_conf_thr=valid_conf_thr,
            start_frame=start_frame,
            end_frame=end_frame,
            should_stop=should_stop,
        )
    raise ValueError(
        f"未知 backend：{backend!r}（body_core_v1 闭环仅支持 "
        f"{BACKEND_MEDIAPIPE!r} / {BACKEND_YOLO!r}）"
    )
