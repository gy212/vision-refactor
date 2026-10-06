# -*- coding: utf-8 -*-
"""
body_core_v1 离线模板闭环（YOLO 迁移 Issue #8 / S2）。

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
- **不同 layout 比对报清晰错误**：复用 ``action_compare._assert_feature_layout_match``，
  绝不静默把 ``body_core_v1`` 模板与 ``pose33_v3`` 序列比对。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .action_compare import _assert_feature_layout_match
from .pose.layouts import BODY_CORE_V1, get_layout
from .paths import templates_dir
from .foundation.signal import find_active_range, motion_energy
from .pose_features import subsequence_dtw
from .pose.backends.body_core import (  # noqa: F401  （提取已迁入姿态层，旧导入路径保留）
    BACKEND_MEDIAPIPE,
    BACKEND_YOLO,
    extract_body_core_features,
)

# body_core_v1 的 DTW baseline 已在 **S3（#10）标定**：取自
# ``feature_layout.BODY_CORE_V1.default_baseline``（1.2826，按尺度对齐 pose33_v3 反推）。
# 标定依据见 docs/yolo_body_core_calibration.md。占位 2.0 已移除；但 S3 结论未授权评分。
BODY_CORE_V1_CALIBRATED_BASELINE: float = float(BODY_CORE_V1.default_baseline or 2.0)

# 标定状态标记（S3 #10）：baseline / YOLO conf 阈值已落库，但跨视频 go/no-go 未通过；
# 分数只可预览 / 内部标定参考，不得对外评分。
CALIBRATION_STATUS_UNVALIDATED: str = "unvalidated"
BODY_CORE_CALIBRATION_NOTE: str = (
    "body_core_v1 baseline=1.2826、YOLO valid_conf_thr=0.6 已在 S3（#10）落库，"
    "但跨视频预注册判据未通过，结论为「仅预览 / 内部标定参考」。"
    "本分数不得进入用户报告或正式评分；YOLO-only 也不进 full tech_eval。"
)

DEFAULT_YOLO26L_MODEL_NAME = "yolo26l-pose.pt"


class MultiPersonReviewRequiredError(RuntimeError):
    """多人场景闸门拒绝出分（YOLO 迁移 Issue #9 / S2）。

    YOLO 模板或目标视频检出 ``num_persons>1`` 时「最大框取单人」可能稳定选错实例——
    这是正确性风险。默认（``reject_multi_person=True``）下，
    ``match_body_core_template`` 直接抛本异常，确保多人来源**不混入正常评分结果**，
    必须人工复核。
    """

    def __init__(
        self,
        message: str,
        *,
        max_persons: int = 0,
        multi_person_frames: int = 0,
        gate_source: str = "video",
    ) -> None:
        super().__init__(message)
        self.max_persons = int(max_persons)
        self.multi_person_frames = int(multi_person_frames)
        self.gate_source = str(gate_source)


@dataclass(frozen=True)
class BodyCoreMatchResult:
    """body_core_v1 单模板匹配结果（仅供预览 / 内部标定参考）。

    多人场景下（``review_required=True``）``score`` 为 ``None``——降级模式不产出分数，
    避免多人视频混入正常评分结果。默认拒绝模式则根本不返回结果（抛
    ``MultiPersonReviewRequiredError``）。
    """

    template_path: Path
    video_path: Path
    backend: str
    feature_layout: str
    fps: float
    start_frame: int
    end_frame: int
    cost: float
    avg_cost: float
    score: float | None
    baseline: float
    calibration_status: str
    valid_frame_ratio: float | None = None
    # 多人闸门信息（Issue #9）。
    review_required: bool = False
    multi_person_detected: bool = False
    max_persons: int = 0
    multi_person_frames: int = 0
    multi_person_gate_source: str = ""


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


# --------------------------------------------------------------------------- #
# 模板匹配
# --------------------------------------------------------------------------- #
def match_body_core_template(
    template_path: str | Path,
    video_path: str | Path,
    *,
    backend: str | None = None,
    pose_variant: str | None = None,
    yolo_model=None,
    valid_conf_thr: float | None = None,
    precomputed_features: tuple[np.ndarray, float, dict] | None = None,
    reject_multi_person: bool = True,
) -> BodyCoreMatchResult:
    """用 ``body_core_v1`` 模板匹配视频，产出仅供预览 / 内部标定参考的分数。

    - 校验模板 ``feature_layout`` 必须是 ``body_core_v1``；非该布局直接报清晰错误
      （不静默套用 pose33_v3 路径）。
    - 用模板自身 ``feature_layout`` 与视频提取序列做 ``_assert_feature_layout_match``，
      确保 shape / layout 一致。
    - 返回结果标 ``calibration_status=unvalidated``，不得进入用户报告 / 正式评分。

    多人场景闸门（Issue #9 / S2）
    -----------------------------
    YOLO 模板或目标视频检出 ``num_persons>1`` 时「最大框取单人」可能稳定选错实例
    （正确性风险）：
      - ``reject_multi_person=True``（默认）：抛 ``MultiPersonReviewRequiredError``，
        多人来源**不混入正常评分结果**，必须人工复核。
      - ``reject_multi_person=False``：降级——返回 ``score=None`` 且
        ``review_required=True`` 的结果，仍不产出分数，供 batch 标「需人工复核」。
    单人 / MediaPipe 路径不受影响。
    """
    template_path = Path(template_path)
    video_path = Path(video_path)

    tpl = np.load(template_path, allow_pickle=True)
    query = tpl["features"]
    meta = dict(tpl["meta"].item() or {})

    tpl_layout = str(meta.get("feature_layout", ""))
    if tpl_layout != BODY_CORE_V1.name:
        raise ValueError(
            f"body_core_v1 闭环只接受 feature_layout={BODY_CORE_V1.name!r} 的模板，"
            f"实际模板 {template_path} 的 feature_layout={tpl_layout!r}；"
            "请勿与 pose33_v3 模板混用。"
        )
    # 校验布局存在（防御性：拼写错误的布局名应尽早报错）。
    get_layout(BODY_CORE_V1.name)

    eff_backend = str(backend or meta.get("backend") or BACKEND_MEDIAPIPE).lower()
    pv = pose_variant or meta.get("pose_variant", "full")

    # 模板来源本身也受多人闸门约束：如果 YOLO 模板由多人视频生成，后续匹配同样
    # 不能产出正常分数。默认拒绝时可在跑目标视频推理前快速失败。
    template_review_required = bool(meta.get("review_required", False))
    template_multi_person_detected = bool(meta.get("multi_person_detected", False))
    template_max_persons = int(meta.get("max_persons", 0))
    template_multi_person_frames = int(meta.get("multi_person_frames", 0))
    if template_review_required and reject_multi_person:
        raise MultiPersonReviewRequiredError(
            f"多人场景闸门拒绝出分：模板 {template_path} 来源视频检出多人"
            f"（max_persons={template_max_persons}，"
            f"multi_person_frames={template_multi_person_frames}）。"
            "YOLO「最大框取单人」可能稳定选错实例，该模板需人工复核，不得进入正常评分。",
            max_persons=template_max_persons,
            multi_person_frames=template_multi_person_frames,
            gate_source="template",
        )

    if precomputed_features is None:
        seq, fps, backend_meta = extract_body_core_features(
            video_path,
            backend=eff_backend,
            pose_variant=pv,
            yolo_model=yolo_model,
            valid_conf_thr=valid_conf_thr,
        )
    else:
        seq, fps, backend_meta = precomputed_features
        seq = np.asarray(seq, dtype=np.float32)
        backend_meta = dict(backend_meta or {})
        pre_backend = str(backend_meta.get("backend") or eff_backend).lower()
        if pre_backend != eff_backend:
            raise ValueError(
                f"precomputed_features backend={pre_backend!r} 与请求 backend={eff_backend!r} 不一致"
            )

    # 多人闸门（Issue #9）：YOLO 模板 meta 与目标视频 meta 都参与判定。
    # MediaPipe 路径无该字段，默认单人，不受影响。
    video_review_required = bool(backend_meta.get("review_required", False))
    video_multi_person_detected = bool(backend_meta.get("multi_person_detected", False))
    video_max_persons = int(backend_meta.get("max_persons", 0))
    video_multi_person_frames = int(backend_meta.get("multi_person_frames", 0))

    review_required = bool(template_review_required or video_review_required)
    multi_person_detected = bool(template_multi_person_detected or video_multi_person_detected)
    max_persons = max(template_max_persons, video_max_persons)
    multi_person_frames = int(template_multi_person_frames + video_multi_person_frames)
    gate_sources: list[str] = []
    if template_review_required:
        gate_sources.append("template")
    if video_review_required:
        gate_sources.append("video")
    multi_person_gate_source = "+".join(gate_sources)

    if review_required and reject_multi_person:
        raise MultiPersonReviewRequiredError(
            f"多人场景闸门拒绝出分：{video_path} 检出多人"
            f"（max_persons={video_max_persons}，multi_person_frames={video_multi_person_frames}）。"
            "YOLO「最大框取单人」可能稳定选错实例，该视频需人工复核，不得进入正常评分。",
            max_persons=max_persons,
            multi_person_frames=multi_person_frames,
            gate_source=multi_person_gate_source or "video",
        )

    # 不同 layout / shape 不允许静默比对。
    _assert_feature_layout_match(
        query,
        seq,
        left_label=f"template {template_path}",
        right_label=f"video {video_path}",
        left_layout=tpl_layout,
        right_layout=BODY_CORE_V1.name,
    )

    if seq.shape[0] == 0:
        raise RuntimeError(f"视频未产出任何 body_core_v1 帧：{video_path}")

    baseline = float(meta.get("baseline", BODY_CORE_V1_CALIBRATED_BASELINE))
    cost, start, end = subsequence_dtw(query, seq)
    avg_cost = float(cost) / max(1, int(query.shape[0]))

    # 降级模式（reject_multi_person=False）：多人时不产出分数（score=None），
    # 仍计算 cost/avg_cost 供调试，但绝不把多人分数当对外评分。
    score: float | None
    if review_required:
        score = None
    else:
        score = float(baseline / (baseline + avg_cost))

    return BodyCoreMatchResult(
        template_path=template_path,
        video_path=video_path,
        backend=eff_backend,
        feature_layout=BODY_CORE_V1.name,
        fps=float(fps),
        start_frame=int(start),
        end_frame=int(end),
        cost=float(cost),
        avg_cost=float(avg_cost),
        score=score,
        baseline=float(baseline),
        # S3（#10）已落参数，但结论为仅预览 / 内部标定参考，不授权评分。
        calibration_status=CALIBRATION_STATUS_UNVALIDATED,
        valid_frame_ratio=(
            float(backend_meta["body_core_valid_frame_ratio"])
            if "body_core_valid_frame_ratio" in backend_meta
            else None
        ),
        review_required=review_required,
        multi_person_detected=multi_person_detected,
        max_persons=max_persons,
        multi_person_frames=multi_person_frames,
        multi_person_gate_source=multi_person_gate_source,
    )
