# -*- coding: utf-8 -*-
"""``body_core_v1`` 模板匹配（YOLO 迁移 Issue #8 / S2）：布局校验、多人场景闸门、子序列 DTW 与未标定分数。

- 只接受 ``feature_layout=body_core_v1`` 的模板，不同布局报清晰错误，不静默套用 pose33_v3 路径。
- baseline 取自布局注册表（``BODY_CORE_V1.default_baseline``，1.2826，S3 #10 标定），但跨视频判据未通过：
  结果标 ``calibration_status=unvalidated``，只供预览 / 内部标定参考，不得进入用户报告或正式评分。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.evaluation.matching.engine import assert_feature_layout_match, match_templates
from core.evaluation.scoring.cost import baseline_for
from core.evaluation.templates import COMPARE, load_template
from core.pose.backends.body_core import BACKEND_MEDIAPIPE, extract_body_core_features
from core.pose.layouts import BODY_CORE_V1, get_layout

BODY_CORE_V1_CALIBRATED_BASELINE: float = baseline_for(BODY_CORE_V1)

# 标定状态标记（S3 #10）：baseline / YOLO conf 阈值已落库，但跨视频 go/no-go 未通过；
# 分数只可预览 / 内部标定参考，不得对外评分。
CALIBRATION_STATUS_UNVALIDATED: str = "unvalidated"
BODY_CORE_CALIBRATION_NOTE: str = (
    "body_core_v1 baseline=1.2826、YOLO valid_conf_thr=0.6 已在 S3（#10）落库，"
    "但跨视频预注册判据未通过，结论为「仅预览 / 内部标定参考」。"
    "本分数不得进入用户报告或正式评分；YOLO-only 也不进 full tech_eval。"
)


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


@dataclass(frozen=True)
class _PersonGate:
    """模板与视频合并后的多人闸门状态。"""

    review_required: bool
    multi_person_detected: bool
    max_persons: int
    multi_person_frames: int
    gate_source: str


def _check_template(template_path: Path, meta: dict, reject_multi_person: bool) -> str:
    """模板必须是 body_core_v1；模板来源本身多人且默认拒绝时，在跑目标视频推理前快速失败。"""
    tpl_layout = str(meta.get("feature_layout", ""))
    if tpl_layout != BODY_CORE_V1.name:
        raise ValueError(
            f"body_core_v1 闭环只接受 feature_layout={BODY_CORE_V1.name!r} 的模板，"
            f"实际模板 {template_path} 的 feature_layout={tpl_layout!r}；"
            "请勿与 pose33_v3 模板混用。"
        )
    # 校验布局存在（防御性：拼写错误的布局名应尽早报错）。
    get_layout(BODY_CORE_V1.name)
    if bool(meta.get("review_required", False)) and reject_multi_person:
        max_persons = int(meta.get("max_persons", 0))
        frames = int(meta.get("multi_person_frames", 0))
        raise MultiPersonReviewRequiredError(
            f"多人场景闸门拒绝出分：模板 {template_path} 来源视频检出多人"
            f"（max_persons={max_persons}，"
            f"multi_person_frames={frames}）。"
            "YOLO「最大框取单人」可能稳定选错实例，该模板需人工复核，不得进入正常评分。",
            max_persons=max_persons,
            multi_person_frames=frames,
            gate_source="template",
        )
    return tpl_layout


def _video_features(video_path: Path, eff_backend: str, pv: str, yolo_model, valid_conf_thr,
                    precomputed_features) -> tuple[np.ndarray, float, dict]:
    if precomputed_features is None:
        return extract_body_core_features(
            video_path,
            backend=eff_backend,
            pose_variant=pv,
            yolo_model=yolo_model,
            valid_conf_thr=valid_conf_thr,
        )
    seq, fps, backend_meta = precomputed_features
    seq = np.asarray(seq, dtype=np.float32)
    backend_meta = dict(backend_meta or {})
    pre_backend = str(backend_meta.get("backend") or eff_backend).lower()
    if pre_backend != eff_backend:
        raise ValueError(
            f"precomputed_features backend={pre_backend!r} 与请求 backend={eff_backend!r} 不一致"
        )
    return seq, fps, backend_meta


def _person_gate(meta: dict, backend_meta: dict, video_path: Path, reject_multi_person: bool) -> _PersonGate:
    """多人闸门（Issue #9）：YOLO 模板 meta 与目标视频 meta 都参与判定；MediaPipe 路径无该字段，默认单人。"""
    template_review = bool(meta.get("review_required", False))
    video_review = bool(backend_meta.get("review_required", False))
    video_max_persons = int(backend_meta.get("max_persons", 0))
    video_frames = int(backend_meta.get("multi_person_frames", 0))
    gate = _PersonGate(
        review_required=bool(template_review or video_review),
        multi_person_detected=bool(bool(meta.get("multi_person_detected", False))
                                   or bool(backend_meta.get("multi_person_detected", False))),
        max_persons=max(int(meta.get("max_persons", 0)), video_max_persons),
        multi_person_frames=int(int(meta.get("multi_person_frames", 0)) + video_frames),
        gate_source="+".join(s for s, on in (("template", template_review), ("video", video_review)) if on),
    )
    if gate.review_required and reject_multi_person:
        raise MultiPersonReviewRequiredError(
            f"多人场景闸门拒绝出分：{video_path} 检出多人"
            f"（max_persons={video_max_persons}，multi_person_frames={video_frames}）。"
            "YOLO「最大框取单人」可能稳定选错实例，该视频需人工复核，不得进入正常评分。",
            max_persons=gate.max_persons,
            multi_person_frames=gate.multi_person_frames,
            gate_source=gate.gate_source or "video",
        )
    return gate


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

    ``reject_multi_person=True``（默认）时多人来源抛 ``MultiPersonReviewRequiredError``；``False`` 时降级为
    ``score=None``、``review_required=True`` 的结果，供 batch 标「需人工复核」。单人 / MediaPipe 路径不受影响。
    """
    template_path = Path(template_path)
    video_path = Path(video_path)
    template = load_template(template_path, COMPARE)
    meta = template.raw_meta
    tpl_layout = _check_template(template_path, meta, reject_multi_person)
    eff_backend = str(backend or meta.get("backend") or BACKEND_MEDIAPIPE).lower()
    pv = pose_variant or meta.get("pose_variant", "full")
    seq, fps, backend_meta = _video_features(video_path, eff_backend, pv, yolo_model, valid_conf_thr,
                                             precomputed_features)
    gate = _person_gate(meta, backend_meta, video_path, reject_multi_person)
    # 不同 layout / shape 不允许静默比对。
    assert_feature_layout_match(template.features, seq, left_label=f"template {template_path}",
                                right_label=f"video {video_path}", left_layout=tpl_layout,
                                right_layout=BODY_CORE_V1.name)
    if seq.shape[0] == 0:
        raise RuntimeError(f"视频未产出任何 body_core_v1 帧：{video_path}")
    baseline = float(meta.get("baseline", BODY_CORE_V1_CALIBRATED_BASELINE))
    (m,) = match_templates(seq, [template], layout=BODY_CORE_V1.name, baseline=baseline,
                           video_label=f"video {video_path}")
    return BodyCoreMatchResult(
        template_path=template_path,
        video_path=video_path,
        backend=eff_backend,
        feature_layout=BODY_CORE_V1.name,
        fps=float(fps),
        start_frame=int(m.start_frame),
        end_frame=int(m.end_frame),
        cost=float(m.cost),
        avg_cost=float(m.avg_cost),
        # 降级模式（reject_multi_person=False）：多人时仍算 cost/avg_cost 供调试，但不产出分数。
        score=None if gate.review_required else m.score,
        baseline=float(baseline),
        # S3（#10）已落参数，但结论为仅预览 / 内部标定参考，不授权评分。
        calibration_status=CALIBRATION_STATUS_UNVALIDATED,
        valid_frame_ratio=(
            float(backend_meta["body_core_valid_frame_ratio"])
            if "body_core_valid_frame_ratio" in backend_meta
            else None
        ),
        review_required=gate.review_required,
        multi_person_detected=gate.multi_person_detected,
        max_persons=gate.max_persons,
        multi_person_frames=gate.multi_person_frames,
        multi_person_gate_source=gate.gate_source,
    )
