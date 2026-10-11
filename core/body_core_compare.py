# -*- coding: utf-8 -*-
"""
body_core_v1 离线模板生成（YOLO 迁移 Issue #8 / S2）；匹配已迁入 ``core.evaluation.matching.body_core``。

定位
----
本模块打通**一条** YOLO ``body_core_v1`` 离线模板闭环（生成 → 匹配），并让
MediaPipe 也能生成同一布局模板（供 #10 三方对比）。它是 ``body_core_v1`` 这个共享
布局（12 点躯干四肢核心）的唯一消费者入口，故 normalizer / baseline 的「真正用途」
都在此体现（见 docs/yolo_migration_plan_optimized.md 三审拆层说明）。

硬约束（来自 AGENTS.md / 迁移计划，必须守住）
--------------------------------------------
- **不动 MediaPipe 默认 ``pose33_v3`` 路径**：``apps/make_template.py`` /
  ``compare_video_to_template`` 等旧默认行为完全不变；本模块是显式 opt-in 的独立入口。
- **YOLO 只允许 ``body_core_v1``**：本模块的 YOLO 路径只产出 / 匹配该布局。
- **S3（#10）只落参数，不授权出分**：``body_core_v1`` baseline=1.2826（取自
  ``feature_layout.BODY_CORE_V1.default_baseline``），YOLO 侧 ``valid_conf_thr``=0.6。
  但跨视频预注册判据未通过，结论为「仅预览 / 内部标定参考」，因此本模块分数 meta
  仍标 ``calibration_status="unvalidated"``，不得进入用户报告 / 正式评分。
- **不同 layout 比对报清晰错误**：复用 ``matching.engine.assert_feature_layout_match``，
  绝不静默把 ``body_core_v1`` 模板与 ``pose33_v3`` 序列比对。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .evaluation.matching.body_core import (  # noqa: F401  （匹配已迁入评估层，旧导入路径保留）
    BODY_CORE_CALIBRATION_NOTE,
    BODY_CORE_V1_CALIBRATED_BASELINE,
    CALIBRATION_STATUS_UNVALIDATED,
    BodyCoreMatchResult,
    MultiPersonReviewRequiredError,
    match_body_core_template,
)
from .pose.layouts import BODY_CORE_V1
from .paths import templates_dir
from .foundation.signal import find_active_range, motion_energy
from .pose.backends.body_core import (  # noqa: F401  （提取已迁入姿态层，旧导入路径保留）
    BACKEND_MEDIAPIPE,
    BACKEND_YOLO,
    extract_body_core_features,
)

DEFAULT_YOLO26L_MODEL_NAME = "yolo26l-pose.pt"


# --------------------------------------------------------------------------- #
# 模板生成
# --------------------------------------------------------------------------- #
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
