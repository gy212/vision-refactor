# -*- coding: utf-8 -*-
"""Shared backend routing decisions for UI and batch entry points.

This module is intentionally pure and dependency-light. It decides what a
caller is allowed to run; it does not load MediaPipe, YOLO, Rust/Tauri, or UI
code. Keep it as the single place for backend capability and authorization
rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .feature_layout import BODY_CORE_V1, POSE33_V3


BACKEND_MEDIAPIPE = "mediapipe"
BACKEND_YOLO = "yolo"

FEATURE_LAYOUT_POSE33 = POSE33_V3.name
FEATURE_LAYOUT_BODY_CORE = BODY_CORE_V1.name
RAW_LAYOUT_POSE33 = POSE33_V3.name
RAW_LAYOUT_YOLO_COCO17 = "pose33_like_coco17"

CAPABILITY_BODY = "body"
CAPABILITY_BODY_ONLY = "body_only"
CAPABILITY_FINGERS = "fingers"
CAPABILITY_HANDS = "hands"
CAPABILITY_MOUTH_CORNERS = "mouth_corners"
CAPABILITY_HEELS = "heels"
CAPABILITY_FOOT_TIPS = "foot_tips"
CAPABILITY_FULL_TECH_EVAL = "full_tech_eval"
CAPABILITY_FORMAL_SCORE = "formal_score"

DISPLAY_SCOPE_STANDARD = "standard"
DISPLAY_SCOPE_LIMITED = "limited"
DISPLAY_SCOPE_INTERNAL = "internal"

EVAL_COMPLETENESS_FULL = "full"
EVAL_COMPLETENESS_PARTIAL = "partial"
EVAL_COMPLETENESS_BODY_ONLY = "body_only"
EVAL_COMPLETENESS_UNAVAILABLE = "unavailable"

DEFAULT_YOLO_MODEL_NAME = "yolo11n-pose.pt"
DEFAULT_YOLO_VALID_CONF_THR = 0.6
YOLO_CALIBRATION_STATUS = "unvalidated"
YOLO_CONFIDENCE_KIND = "yolo_conf"
YOLO_VALIDITY_POLICY = "confidence_thr"


class TaskType(str, Enum):
    REALTIME_PREVIEW = "realtime_preview"
    OFFLINE_BODY_ANALYSIS = "offline_body_analysis"
    FORMAL_SCORE = "formal_score"
    FULL_TECH_EVAL = "full_tech_eval"
    BATCH_BODY_CORE = "batch_body_core"


class QualityProfile(str, Enum):
    DEFAULT = "default"
    REALTIME = "realtime"
    HIGH_QUALITY = "high_quality"
    EXTREME = "extreme"


class ScoreMode(str, Enum):
    NONE = "none"
    PREVIEW = "preview"
    INTERNAL = "internal"
    FORMAL = "formal"
    FULL_TECH_EVAL = "full_tech_eval"


@dataclass(frozen=True)
class ModelAvailability:
    """Known model/packaging availability for routing decisions."""

    yolo_realtime: bool = False
    yolo_realtime_model: str = ""
    yolo26l: bool = False
    yolo_supported: bool = True
    reason: str = ""


@dataclass(frozen=True)
class BackendRouteRequest:
    task_type: str
    enable_hands: bool = True
    quality_profile: str = QualityProfile.DEFAULT.value
    score_mode: str = ScoreMode.NONE.value
    requires_capabilities: tuple[str, ...] = ()
    model_availability: ModelAvailability = field(default_factory=ModelAvailability)
    multi_person_detected: bool = False
    person_count: int = 0
    requested_backend: str | None = None
    installed_app: bool = False


@dataclass(frozen=True)
class BackendRouteDecision:
    backend: str
    model_profile: str
    raw_layout: str
    feature_layout: str
    capability: str
    capabilities: tuple[str, ...]
    requires_capabilities: tuple[str, ...]
    calibration_status: str
    score_authorized: bool
    display_scope: str
    eval_completeness: str
    skipped_capabilities: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()
    fallback_reason: str | None = None
    requested_backend: str | None = None
    reason: str = ""
    multi_person_detected: bool = False
    person_count: int = 0
    review_required: bool = False
    target_policy: str = ""
    error_code: str | None = None
    user_message: str | None = None

    @property
    def ok(self) -> bool:
        return self.error_code is None

    def to_snake_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "backend": self.backend,
            "model_profile": self.model_profile,
            "raw_layout": self.raw_layout,
            "feature_layout": self.feature_layout,
            "capability": self.capability,
            "capabilities": list(self.capabilities),
            "requires_capabilities": list(self.requires_capabilities),
            "calibration_status": self.calibration_status,
            "score_authorized": self.score_authorized,
            "display_scope": self.display_scope,
            "eval_completeness": self.eval_completeness,
            "skipped_capabilities": list(self.skipped_capabilities),
            "missing_capabilities": list(self.missing_capabilities),
            "fallback_reason": self.fallback_reason,
            "requested_backend": self.requested_backend,
            "reason": self.reason,
            "multi_person_detected": self.multi_person_detected,
            "person_count": self.person_count,
            "review_required": self.review_required,
            "target_policy": self.target_policy,
            "error_code": self.error_code,
            "user_message": self.user_message,
        }

    def to_camel_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "backend": self.backend,
            "modelProfile": self.model_profile,
            "rawLayout": self.raw_layout,
            "featureLayout": self.feature_layout,
            "capability": self.capability,
            "capabilities": list(self.capabilities),
            "requiresCapabilities": list(self.requires_capabilities),
            "calibrationStatus": self.calibration_status,
            "scoreAuthorized": self.score_authorized,
            "displayScope": self.display_scope,
            "evalCompleteness": self.eval_completeness,
            "skippedCapabilities": list(self.skipped_capabilities),
            "missingCapabilities": list(self.missing_capabilities),
            "fallbackReason": self.fallback_reason,
            "requestedBackend": self.requested_backend,
            "reason": self.reason,
            "multiPersonDetected": self.multi_person_detected,
            "personCount": self.person_count,
            "reviewRequired": self.review_required,
            "targetPolicy": self.target_policy,
            "errorCode": self.error_code,
            "userMessage": self.user_message,
        }


def model_availability_from_mapping(values: dict[str, Any] | None) -> ModelAvailability:
    values = dict(values or {})
    yolo_supported = bool(values.get("yoloSupported", values.get("yolo_supported", True)))
    return ModelAvailability(
        yolo_realtime=bool(
            values.get("yoloRealtime", values.get("yolo_realtime", values.get("yolo26n", values.get("yolo26s", False))))
        ),
        yolo_realtime_model=str(
            values.get("yoloRealtimeModel", values.get("yolo_realtime_model", ""))
        ).strip().lower(),
        yolo26l=bool(values.get("yolo26L", values.get("yolo26l", values.get("yolo_l", False)))),
        yolo_supported=yolo_supported,
        reason=str(values.get("reason") or values.get("message") or ""),
    )


def route_backend(request: BackendRouteRequest) -> BackendRouteDecision:
    task_type = _normalize_task_type(request.task_type)
    quality = _normalize_quality(request.quality_profile)
    score_mode = _normalize_score_mode(request.score_mode, task_type)
    required = _normalize_capabilities(request.requires_capabilities)
    availability = request.model_availability

    if request.enable_hands:
        return _mediapipe_full(
            required=required,
            reason="enableHands=true requires MediaPipe full with hand landmarker",
            score_authorized=score_mode in {ScoreMode.FORMAL.value, ScoreMode.FULL_TECH_EVAL.value},
        )

    if task_type == TaskType.REALTIME_PREVIEW.value:
        if _requires_non_yolo_capability(required):
            return _mediapipe_pose_only_partial(
                required=required,
                reason="requested capabilities exceed YOLO body-only preview; enableHands=false keeps hand landmarker disabled",
            )
        if availability.yolo_supported and availability.yolo_realtime:
            realtime_model = availability.yolo_realtime_model if availability.yolo_realtime_model in {"yolo26n", "yolo26s"} else "yolo26n"
            return _yolo_body_only(
                required=required,
                model_profile=realtime_model,
                display_scope=DISPLAY_SCOPE_LIMITED,
                reason=f"enableHands=false realtime preview with {realtime_model} model available",
                multi_person_detected=request.multi_person_detected,
                person_count=request.person_count,
            )
        fallback = availability.reason or "YOLO realtime model unavailable"
        return _mediapipe_pose_only_partial(
            required=required,
            reason="YOLO realtime unavailable; fallback to MediaPipe pose-only preview",
            fallback_reason=fallback,
            requested_backend=BACKEND_YOLO,
        )

    if task_type in {TaskType.OFFLINE_BODY_ANALYSIS.value, TaskType.BATCH_BODY_CORE.value}:
        if _requires_non_yolo_capability(required):
            return _mediapipe_pose_only_partial(
                required=required,
                reason="body-only analysis request includes non-body YOLO capabilities; enableHands=false keeps hand landmarker disabled",
            )
        if availability.yolo_supported and availability.yolo26l:
            return _yolo_body_only(
                required=required,
                model_profile="yolo26l",
                display_scope=DISPLAY_SCOPE_INTERNAL,
                reason="offline high-quality body-only analysis with YOLO26L available",
                multi_person_detected=request.multi_person_detected,
                person_count=request.person_count,
            )
        return _structured_error(
            required=required,
            reason="YOLO26L is required for offline high-quality body-only analysis",
            user_message="请先下载或安装 YOLO26L 模型后再运行离线高质量 body-only 分析。",
            detail=availability.reason or "YOLO26L model unavailable",
        )

    if task_type in {TaskType.FORMAL_SCORE.value, TaskType.FULL_TECH_EVAL.value}:
        return _mediapipe_pose_only_partial(
            required=required,
            reason="formal scoring/full tech_eval remain on MediaPipe pose-only when enableHands=false; hand indicators are skip-aware partial",
            score_authorized=True,
        )

    return _mediapipe_pose_only_partial(
        required=required,
        reason="default route uses MediaPipe pose-only with enableHands=false",
    )


def route_for_preview(
    *,
    enable_hands: bool,
    model_availability: ModelAvailability | dict[str, Any] | None = None,
    requires_capabilities: tuple[str, ...] = (),
) -> BackendRouteDecision:
    return route_backend(
        BackendRouteRequest(
            task_type=TaskType.REALTIME_PREVIEW.value,
            enable_hands=enable_hands,
            quality_profile=QualityProfile.REALTIME.value,
            score_mode=ScoreMode.PREVIEW.value,
            requires_capabilities=requires_capabilities,
            model_availability=_coerce_availability(model_availability),
        )
    )


def route_for_analysis(
    *,
    enable_hands: bool,
    do_compare: bool,
    do_tech_eval: bool,
    quality_profile: str = QualityProfile.DEFAULT.value,
    model_availability: ModelAvailability | dict[str, Any] | None = None,
    requires_capabilities: tuple[str, ...] = (),
) -> BackendRouteDecision:
    if do_tech_eval:
        task_type = TaskType.FULL_TECH_EVAL.value
        score_mode = ScoreMode.FULL_TECH_EVAL.value
    elif do_compare:
        task_type = TaskType.FORMAL_SCORE.value
        score_mode = ScoreMode.FORMAL.value
    elif quality_profile == QualityProfile.HIGH_QUALITY.value:
        task_type = TaskType.OFFLINE_BODY_ANALYSIS.value
        score_mode = ScoreMode.INTERNAL.value
    else:
        task_type = TaskType.OFFLINE_BODY_ANALYSIS.value
        score_mode = ScoreMode.INTERNAL.value
    return route_backend(
        BackendRouteRequest(
            task_type=task_type,
            enable_hands=enable_hands,
            quality_profile=quality_profile,
            score_mode=score_mode,
            requires_capabilities=requires_capabilities,
            model_availability=_coerce_availability(model_availability),
        )
    )


def validate_backend_layout(backend: str, feature_layout: str) -> BackendRouteDecision:
    backend = str(backend or BACKEND_MEDIAPIPE).lower()
    feature_layout = str(feature_layout or FEATURE_LAYOUT_POSE33)
    if backend == BACKEND_YOLO:
        if feature_layout != FEATURE_LAYOUT_BODY_CORE:
            raise ValueError("YOLO 后端仅支持 --feature-layout body_core_v1；不得使用 YOLO + pose33_v3")
        return _yolo_body_only(
            required=(CAPABILITY_BODY_ONLY,),
            model_profile=DEFAULT_YOLO_MODEL_NAME,
            display_scope=DISPLAY_SCOPE_INTERNAL,
            reason="explicit batch YOLO body_core_v1 route",
        )
    if backend != BACKEND_MEDIAPIPE:
        raise ValueError(f"未知 backend：{backend!r}")
    if feature_layout not in {FEATURE_LAYOUT_POSE33, FEATURE_LAYOUT_BODY_CORE}:
        raise ValueError(f"未知 feature_layout：{feature_layout!r}")
    if feature_layout == FEATURE_LAYOUT_BODY_CORE:
        return _mediapipe_body_core("explicit batch MediaPipe body_core_v1 route")
    return _mediapipe_full(
        required=(CAPABILITY_BODY,),
        reason="default MediaPipe pose33_v3 route",
        score_authorized=True,
    )


def yolo_metadata(source_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    source_meta = dict(source_meta or {})
    decision = _yolo_body_only(
        required=(CAPABILITY_BODY_ONLY,),
        model_profile=str(source_meta.get("model_name") or DEFAULT_YOLO_MODEL_NAME),
        display_scope=str(source_meta.get("display_scope") or DISPLAY_SCOPE_INTERNAL),
        reason="YOLO body_core_v1 metadata",
        multi_person_detected=bool(source_meta.get("multi_person_detected", False)),
        person_count=int(source_meta.get("max_persons", source_meta.get("person_count", 0)) or 0),
    )
    meta = decision.to_snake_dict()
    meta.update(
        {
            "model_name": decision.model_profile,
            "feature_layout": FEATURE_LAYOUT_BODY_CORE,
            "confidence_kind": str(source_meta.get("confidence_kind") or YOLO_CONFIDENCE_KIND),
            "validity_policy": str(source_meta.get("validity_policy") or YOLO_VALIDITY_POLICY),
            "valid_conf_thr": float(source_meta.get("valid_conf_thr", DEFAULT_YOLO_VALID_CONF_THR)),
            "calibration_status": YOLO_CALIBRATION_STATUS,
            "score_authorized": False,
            "review_required": bool(source_meta.get("review_required", decision.review_required)),
        }
    )
    return meta


def mediapipe_body_core_metadata(
    source_meta: dict[str, Any] | None = None,
    *,
    pose_variant: str = "full",
) -> dict[str, Any]:
    source_meta = dict(source_meta or {})
    decision = _mediapipe_body_core("MediaPipe body_core_v1 metadata")
    meta = decision.to_snake_dict()
    meta.update(
        {
            "model_name": str(source_meta.get("model_name") or f"pose_landmarker_{pose_variant}"),
            "feature_layout": FEATURE_LAYOUT_BODY_CORE,
            "confidence_kind": str(source_meta.get("confidence_kind") or "visibility"),
            "validity_policy": str(source_meta.get("validity_policy") or "valid_mask"),
            "valid_conf_thr": source_meta.get("valid_conf_thr", ""),
            "calibration_status": str(
                source_meta.get("calibration_status") or YOLO_CALIBRATION_STATUS
            ),
            "score_authorized": False,
            "review_required": bool(source_meta.get("review_required", False)),
        }
    )
    return meta


def _coerce_availability(value: ModelAvailability | dict[str, Any] | None) -> ModelAvailability:
    if isinstance(value, ModelAvailability):
        return value
    return model_availability_from_mapping(value)


def _normalize_task_type(value: str) -> str:
    normalized = str(value or TaskType.REALTIME_PREVIEW.value).strip().lower()
    aliases = {
        "preview": TaskType.REALTIME_PREVIEW.value,
        "realtime": TaskType.REALTIME_PREVIEW.value,
        "session.start": TaskType.REALTIME_PREVIEW.value,
        "offline_high_quality": TaskType.OFFLINE_BODY_ANALYSIS.value,
        "offline_body": TaskType.OFFLINE_BODY_ANALYSIS.value,
        "body_core": TaskType.BATCH_BODY_CORE.value,
        "tech_eval": TaskType.FULL_TECH_EVAL.value,
        "score": TaskType.FORMAL_SCORE.value,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {item.value for item in TaskType}:
        return TaskType.REALTIME_PREVIEW.value
    return normalized


def _normalize_quality(value: str) -> str:
    normalized = str(value or QualityProfile.DEFAULT.value).strip().lower()
    if normalized in {item.value for item in QualityProfile}:
        return normalized
    return QualityProfile.DEFAULT.value


def _normalize_score_mode(value: str, task_type: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {item.value for item in ScoreMode}:
        return normalized
    if task_type == TaskType.FULL_TECH_EVAL.value:
        return ScoreMode.FULL_TECH_EVAL.value
    if task_type == TaskType.FORMAL_SCORE.value:
        return ScoreMode.FORMAL.value
    if task_type == TaskType.REALTIME_PREVIEW.value:
        return ScoreMode.PREVIEW.value
    return ScoreMode.INTERNAL.value


def _normalize_capabilities(values: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
    normalized = []
    for value in values or ():
        text = str(value).strip().lower()
        aliases = {
            "finger": CAPABILITY_FINGERS,
            "hand": CAPABILITY_HANDS,
            "mouth": CAPABILITY_MOUTH_CORNERS,
            "mouth_corner": CAPABILITY_MOUTH_CORNERS,
            "heel": CAPABILITY_HEELS,
            "foot_tip": CAPABILITY_FOOT_TIPS,
        }
        text = aliases.get(text, text)
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _requires_non_yolo_capability(required: tuple[str, ...]) -> bool:
    blockers = {
        CAPABILITY_FINGERS,
        CAPABILITY_HANDS,
        CAPABILITY_MOUTH_CORNERS,
        CAPABILITY_HEELS,
        CAPABILITY_FOOT_TIPS,
        CAPABILITY_FULL_TECH_EVAL,
        CAPABILITY_FORMAL_SCORE,
    }
    return any(item in blockers for item in required)


def _missing_for_pose_only(required: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(item for item in required if item in {CAPABILITY_FINGERS, CAPABILITY_HANDS})


def _mediapipe_full(
    *,
    required: tuple[str, ...],
    reason: str,
    score_authorized: bool,
) -> BackendRouteDecision:
    return BackendRouteDecision(
        backend=BACKEND_MEDIAPIPE,
        model_profile="pose_full_with_hands",
        raw_layout=RAW_LAYOUT_POSE33,
        feature_layout=FEATURE_LAYOUT_POSE33,
        capability="full",
        capabilities=(CAPABILITY_BODY, CAPABILITY_HANDS, CAPABILITY_FINGERS),
        requires_capabilities=required,
        calibration_status="validated",
        score_authorized=bool(score_authorized),
        display_scope=DISPLAY_SCOPE_STANDARD,
        eval_completeness=EVAL_COMPLETENESS_FULL,
        reason=reason,
    )


def _mediapipe_pose_only_partial(
    *,
    required: tuple[str, ...],
    reason: str,
    fallback_reason: str | None = None,
    requested_backend: str | None = None,
    score_authorized: bool = False,
) -> BackendRouteDecision:
    skipped = _missing_for_pose_only(required)
    completeness = EVAL_COMPLETENESS_PARTIAL if skipped else EVAL_COMPLETENESS_FULL
    return BackendRouteDecision(
        backend=BACKEND_MEDIAPIPE,
        model_profile="pose_only",
        raw_layout=RAW_LAYOUT_POSE33,
        feature_layout=FEATURE_LAYOUT_POSE33,
        capability="pose_only",
        capabilities=(CAPABILITY_BODY,),
        requires_capabilities=required,
        calibration_status="validated",
        score_authorized=bool(score_authorized),
        display_scope=DISPLAY_SCOPE_STANDARD,
        eval_completeness=completeness,
        skipped_capabilities=skipped,
        missing_capabilities=skipped,
        fallback_reason=fallback_reason,
        requested_backend=requested_backend,
        reason=reason,
    )


def _mediapipe_body_core(reason: str, *, model_profile: str = "pose_body_core") -> BackendRouteDecision:
    return BackendRouteDecision(
        backend=BACKEND_MEDIAPIPE,
        model_profile=model_profile,
        raw_layout=RAW_LAYOUT_POSE33,
        feature_layout=FEATURE_LAYOUT_BODY_CORE,
        capability=CAPABILITY_BODY_ONLY,
        capabilities=(CAPABILITY_BODY_ONLY,),
        requires_capabilities=(CAPABILITY_BODY_ONLY,),
        calibration_status=YOLO_CALIBRATION_STATUS,
        score_authorized=False,
        display_scope=DISPLAY_SCOPE_INTERNAL,
        eval_completeness=EVAL_COMPLETENESS_BODY_ONLY,
        reason=reason,
    )


def _yolo_body_only(
    *,
    required: tuple[str, ...],
    model_profile: str,
    display_scope: str,
    reason: str,
    multi_person_detected: bool = False,
    person_count: int = 0,
) -> BackendRouteDecision:
    review_required = bool(multi_person_detected)
    return BackendRouteDecision(
        backend=BACKEND_YOLO,
        model_profile=model_profile,
        raw_layout=RAW_LAYOUT_YOLO_COCO17,
        feature_layout=FEATURE_LAYOUT_BODY_CORE,
        capability=CAPABILITY_BODY_ONLY,
        capabilities=(CAPABILITY_BODY_ONLY,),
        requires_capabilities=required,
        calibration_status=YOLO_CALIBRATION_STATUS,
        score_authorized=False,
        display_scope=display_scope if display_scope in {DISPLAY_SCOPE_LIMITED, DISPLAY_SCOPE_INTERNAL} else DISPLAY_SCOPE_LIMITED,
        eval_completeness=EVAL_COMPLETENESS_BODY_ONLY,
        reason=reason,
        multi_person_detected=bool(multi_person_detected),
        person_count=int(person_count or 0),
        review_required=review_required,
        target_policy="select_main_person_largest_box_highest_score",
    )


def _structured_error(
    *,
    required: tuple[str, ...],
    reason: str,
    user_message: str,
    detail: str,
) -> BackendRouteDecision:
    return BackendRouteDecision(
        backend="unavailable",
        model_profile="yolo26l",
        raw_layout="n/a",
        feature_layout=FEATURE_LAYOUT_BODY_CORE,
        capability=CAPABILITY_BODY_ONLY,
        capabilities=(),
        requires_capabilities=required,
        calibration_status=YOLO_CALIBRATION_STATUS,
        score_authorized=False,
        display_scope=DISPLAY_SCOPE_INTERNAL,
        eval_completeness=EVAL_COMPLETENESS_UNAVAILABLE,
        missing_capabilities=(CAPABILITY_BODY_ONLY,),
        requested_backend=BACKEND_YOLO,
        reason=f"{reason}: {detail}",
        error_code="yolo26l_unavailable",
        user_message=user_message,
    )
