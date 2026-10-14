# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.pose.backends import (  # noqa: E402
    BACKEND_MEDIAPIPE,
    BACKEND_YOLO,
    BackendRouteRequest,
    CAPABILITY_FINGERS,
    CAPABILITY_FOOT_TIPS,
    CAPABILITY_HEELS,
    CAPABILITY_MOUTH_CORNERS,
    FEATURE_LAYOUT_BODY_CORE,
    FEATURE_LAYOUT_POSE33,
    ModelAvailability,
    QualityProfile,
    TaskType,
    route_backend,
    route_for_analysis,
    route_for_preview,
    validate_backend_layout,
    yolo_metadata,
)


def test_enable_hands_true_routes_to_mediapipe_full_with_hand_landmarker():
    decision = route_for_preview(
        enable_hands=True,
        model_availability=ModelAvailability(yolo_realtime=True, yolo26l=True),
    )

    assert decision.backend == BACKEND_MEDIAPIPE
    assert decision.feature_layout == FEATURE_LAYOUT_POSE33
    assert decision.model_profile == "pose_full_with_hands"
    assert decision.eval_completeness == "full"
    assert "hands" in decision.capabilities


def test_realtime_hands_off_uses_yolo_when_model_available():
    decision = route_for_preview(
        enable_hands=False,
        model_availability=ModelAvailability(yolo_supported=True, yolo_realtime=True),
    )

    assert decision.backend == BACKEND_YOLO
    assert decision.model_profile == "yolo26n"
    assert decision.raw_layout == "pose33_like_coco17"
    assert decision.feature_layout == FEATURE_LAYOUT_BODY_CORE
    assert decision.capability == "body_only"
    assert decision.score_authorized is False
    assert decision.calibration_status == "unvalidated"
    assert decision.display_scope == "limited"
    assert decision.reason


def test_realtime_hands_off_falls_back_to_mediapipe_pose_only_when_yolo_missing():
    decision = route_for_preview(
        enable_hands=False,
        model_availability=ModelAvailability(
            yolo_supported=True,
            yolo_realtime=False,
            reason="missing yolo26n/s",
        ),
    )

    assert decision.backend == BACKEND_MEDIAPIPE
    assert decision.model_profile == "pose_only"
    assert decision.requested_backend == BACKEND_YOLO
    assert decision.fallback_reason == "missing yolo26n/s"
    assert decision.score_authorized is False


def test_offline_high_quality_yolo26l_missing_returns_structured_error_not_fallback():
    decision = route_for_analysis(
        enable_hands=False,
        do_compare=False,
        do_tech_eval=False,
        quality_profile=QualityProfile.HIGH_QUALITY.value,
        model_availability=ModelAvailability(yolo_supported=True, yolo26l=False),
    )

    assert decision.ok is False
    assert decision.backend == "unavailable"
    assert decision.requested_backend == BACKEND_YOLO
    assert decision.error_code == "yolo26l_unavailable"
    assert "YOLO26L" in (decision.user_message or "")


def test_offline_high_quality_yolo26l_available_routes_internal_body_only():
    decision = route_for_analysis(
        enable_hands=False,
        do_compare=False,
        do_tech_eval=False,
        quality_profile=QualityProfile.HIGH_QUALITY.value,
        model_availability=ModelAvailability(yolo_supported=True, yolo26l=True),
    )

    assert decision.backend == BACKEND_YOLO
    assert decision.model_profile == "yolo26l"
    assert decision.feature_layout == FEATURE_LAYOUT_BODY_CORE
    assert decision.display_scope == "internal"
    assert decision.score_authorized is False


def test_high_quality_template_compare_stays_mediapipe_formal_compare():
    decision = route_for_analysis(
        enable_hands=False,
        do_compare=True,
        do_tech_eval=False,
        quality_profile=QualityProfile.HIGH_QUALITY.value,
        model_availability=ModelAvailability(yolo_supported=True, yolo26l=True, yolo_realtime=True),
    )

    assert decision.backend == BACKEND_MEDIAPIPE
    assert decision.feature_layout == FEATURE_LAYOUT_POSE33
    assert decision.model_profile == "pose_only"
    assert decision.score_authorized is True
    assert decision.requested_backend is None


def test_yolo_metadata_never_inherits_validated_calibration_status():
    meta = yolo_metadata({"calibration_status": "validated", "score_authorized": True})

    assert meta["calibration_status"] == "unvalidated"
    assert meta["score_authorized"] is False


def test_formal_tech_eval_hands_off_stays_mediapipe_pose_only_and_skips_fingers():
    decision = route_for_analysis(
        enable_hands=False,
        do_compare=False,
        do_tech_eval=True,
        requires_capabilities=(CAPABILITY_FINGERS,),
        model_availability=ModelAvailability(yolo_supported=True, yolo26l=True, yolo_realtime=True),
    )

    assert decision.backend == BACKEND_MEDIAPIPE
    assert decision.model_profile == "pose_only"
    assert decision.score_authorized is True
    assert decision.eval_completeness == "partial"
    assert decision.skipped_capabilities == (CAPABILITY_FINGERS,)
    assert "hand" in decision.reason.lower()


def test_pose_landmark_capabilities_do_not_force_hands_when_hands_disabled():
    decision = route_backend(
        BackendRouteRequest(
            task_type=TaskType.FORMAL_SCORE.value,
            enable_hands=False,
            requires_capabilities=(
                CAPABILITY_MOUTH_CORNERS,
                CAPABILITY_HEELS,
                CAPABILITY_FOOT_TIPS,
            ),
        )
    )

    assert decision.backend == BACKEND_MEDIAPIPE
    assert decision.model_profile == "pose_only"
    assert decision.skipped_capabilities == ()


def test_yolo_multi_person_preview_marks_review_without_authorizing_score():
    decision = route_backend(
        BackendRouteRequest(
            task_type=TaskType.REALTIME_PREVIEW.value,
            enable_hands=False,
            model_availability=ModelAvailability(yolo_supported=True, yolo_realtime=True),
            multi_person_detected=True,
            person_count=3,
        )
    )

    assert decision.backend == BACKEND_YOLO
    assert decision.review_required is True
    assert decision.multi_person_detected is True
    assert decision.person_count == 3
    assert decision.target_policy == "select_main_person_largest_box_highest_score"
    assert decision.score_authorized is False


def test_validate_backend_layout_keeps_batch_yolo_body_core_constraint():
    decision = validate_backend_layout("yolo", "body_core_v1")

    assert decision.backend == BACKEND_YOLO
    assert decision.feature_layout == FEATURE_LAYOUT_BODY_CORE
    with pytest.raises(ValueError, match="YOLO 后端仅支持"):
        validate_backend_layout("yolo", "pose33_v3")
