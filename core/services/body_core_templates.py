# -*- coding: utf-8 -*-
"""``body_core_v1`` 离线模板生成（YOLO 或 MediaPipe）：提取特征、按运动能量定活动区间、写模板与未标定 meta（原 ``core/body_core_compare.py``）。

只供 batch/analysis 显式选用，不动 MediaPipe 默认 ``pose33_v3`` 路径。baseline 1.2826、YOLO ``valid_conf_thr``=0.6
已落库，但跨视频判据未通过：模板 meta 标 ``calibration_status=unvalidated``、``score_authorized=False``，
分数不得进入用户报告或正式评分。匹配在 ``core.evaluation.matching.body_core``。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.evaluation.matching.body_core import (
    BODY_CORE_CALIBRATION_NOTE,
    BODY_CORE_V1_CALIBRATED_BASELINE,
    CALIBRATION_STATUS_UNVALIDATED,
)
from core.foundation.paths import templates_dir
from core.foundation.signal import find_active_range, motion_energy
from core.pose.backends import BACKEND_MEDIAPIPE, BACKEND_YOLO, extract_body_core_features
from core.pose.layouts import BODY_CORE_V1


def create_body_core_template(
    video_path: str | Path,
    *,
    backend: str,
    pose_variant: str = "full",
    start: int | None = None,
    end: int | None = None,
    out_path: str | Path | None = None,
    yolo_model=None,
    valid_conf_thr: float | None = None,
) -> Path:
    """从视频生成 ``body_core_v1`` 模板（YOLO 或 MediaPipe）。

    模板 metadata 写 ``feature_layout=body_core_v1`` 与
    ``calibration_status=unvalidated``——S3 #10 已落 baseline / conf 阈值，但跨视频
    go/no-go 未通过，该模板分数仍不得对外评分。
    """
    backend = str(backend).lower()
    video_path = Path(video_path)
    features, fps, backend_meta = extract_body_core_features(
        video_path,
        backend=backend,
        pose_variant=pose_variant,
        yolo_model=yolo_model,
        valid_conf_thr=valid_conf_thr,
    )
    if features.shape[0] == 0:
        raise RuntimeError(f"视频未产出任何 body_core_v1 帧：{video_path}")

    seq = features.reshape(features.shape[0], -1)
    if features.shape[0] >= 2:
        energy = motion_energy(seq)
        auto_start, auto_end = find_active_range(energy, pad=10)
    else:
        auto_start, auto_end = 0, int(features.shape[0] - 1)

    start_i = int(start) if start is not None else int(auto_start)
    end_i = int(end) if end is not None else int(auto_end)
    start_i = max(0, min(start_i, features.shape[0] - 1))
    end_i = max(start_i, min(end_i, features.shape[0] - 1))

    out_dir = templates_dir()
    out_path = (
        Path(out_path)
        if out_path
        else (out_dir / f"{video_path.stem}_{backend}_body_core_v1.npz")
    )

    meta: dict = {
        "video": str(video_path),
        "fps": float(fps),
        "frame_count": int(features.shape[0]),
        "start_frame": int(start_i),
        "end_frame": int(end_i),
        "auto_start_frame": int(auto_start),
        "auto_end_frame": int(auto_end),
        "backend": backend,
        "raw_layout": str(backend_meta.get("raw_layout") or ("pose33_like_coco17" if backend == BACKEND_YOLO else "pose33_v3")),
        "feature_layout": BODY_CORE_V1.name,
        "capability": str(backend_meta.get("capability") or "body_only"),
        "normalizer_version": "body_core_v1",
        "running_mode": "video",
        # S3（#10）已落参数，但结论为仅预览 / 内部标定参考，不授权评分。
        "calibration_status": CALIBRATION_STATUS_UNVALIDATED,
        "calibration_note": BODY_CORE_CALIBRATION_NOTE,
        "baseline": float(BODY_CORE_V1_CALIBRATED_BASELINE),
        "baseline_calibrated": True,
        "score_authorized": False,
        "display_scope": str(backend_meta.get("display_scope") or "internal"),
    }
    if backend == BACKEND_MEDIAPIPE:
        meta["pose_variant"] = pose_variant
        meta["model_name"] = f"pose_landmarker_{pose_variant}"
        meta["confidence_kind"] = "visibility"
    else:  # YOLO
        meta["model_name"] = str(backend_meta.get("model_name", "yolo"))
        meta["confidence_kind"] = str(backend_meta.get("confidence_kind", "yolo_conf"))
        meta["validity_policy"] = str(backend_meta.get("validity_policy", "confidence_thr"))
        meta["valid_conf_thr"] = float(backend_meta.get("valid_conf_thr", 0.5))
        # 透传 YOLO 多人闸门 / track 信息，供 #9 拒绝/降级与 #10 标定参考。
        for k in (
            "max_persons",
            "multi_person_frames",
            "multi_person_detected",
            "review_required",
            "gate_status",
            "track_reset_note",
        ):
            if k in backend_meta:
                meta[k] = backend_meta[k]
        if "body_core_valid_frame_ratio" in backend_meta:
            meta["body_core_valid_frame_ratio"] = float(backend_meta["body_core_valid_frame_ratio"])

    np.savez_compressed(
        out_path,
        features=features[start_i : end_i + 1],
        meta=np.array(meta, dtype=object),
    )
    return Path(out_path)
