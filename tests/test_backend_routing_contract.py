# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.backend_router import (  # noqa: E402
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
from apps import ui_backend  # noqa: E402


def _wait_frame_events(events: list[dict], count: int = 1, timeout: float = 1.0) -> list[dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame_events = _wait_frame_events(events)
        if len(frame_events) >= count:
            return frame_events
        time.sleep(0.01)
    return [event for event in events if event["event"] == "session.frame"]


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


def test_ui_session_start_uses_yolo_preview_pipeline_and_surfaces_frame_meta():
    events: list[dict] = []
    manager = ui_backend.BridgeJobManager(events.append)

    class FakeCapture:
        def __init__(self) -> None:
            self.frames = [np.zeros((24, 32, 3), dtype=np.uint8)]
            self.released = False

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            values = {
                ui_backend.CAP_PROP_FPS: 30.0,
                ui_backend.CAP_PROP_FRAME_COUNT: 0.0,
                ui_backend.CAP_PROP_FRAME_WIDTH: 32.0,
                ui_backend.CAP_PROP_FRAME_HEIGHT: 24.0,
            }
            return values.get(prop, 0.0)

        def read(self):
            if self.frames:
                return True, self.frames.pop(0)
            return False, None

        def release(self) -> None:
            self.released = True

    class FakeRecorder:
        def begin_session(self, *, fps, size) -> None:
            self.fps = fps
            self.size = size

        def write_frame(self, frame) -> None:
            self.frame = frame

        def close_session(self):
            return None

        def snapshot(self):
            return type(
                "Snapshot",
                (),
                {
                    "state": "idle",
                    "current_path": None,
                    "frames_written": 0,
                    "last_error": None,
                },
            )()

    class FakeYoloPreview:
        def __init__(self, options: ui_backend.PreviewSessionOptions) -> None:
            self.options = options
            self.closed = False

        def next_timestamp_ms(self, *, is_file: bool, fps_for_ts: float) -> int:
            return 0

        def annotate(self, frame, *, timestamp_ms=None):
            assert self.options.backend_route["backend"] == "yolo"
            return frame, ["LEFT_HAND_UP"], {
                "backend": "yolo",
                "raw_layout": "pose33_like_coco17",
                "feature_layout": "body_core_v1",
                "score_authorized": False,
                "calibration_status": "unvalidated",
                "display_scope": "limited",
                "multi_person_detected": True,
                "person_count": 2,
                "review_required": True,
                "target_policy": "select_main_person_largest_box_highest_score",
            }

        def close(self) -> None:
            self.closed = True

    cap = FakeCapture()
    service = ui_backend.PreviewSessionService(
        job_manager=manager,
        capture_factory=lambda source: cap,
        pipeline_factory=lambda options: FakeYoloPreview(options),
        frame_encoder=lambda frame: b"yolo-preview-frame",
        recording_factory=lambda options: FakeRecorder(),
    )
    previous = ui_backend.DEFAULT_PREVIEW_SERVICE
    ui_backend.DEFAULT_PREVIEW_SERVICE = service
    try:
        response = ui_backend.handle_command(
            ui_backend.CommandRequest(
                command="session.start",
                request_id="req-yolo-preview",
                session_id="session-yolo-preview",
                payload={
                    "source": "0",
                    "enableHands": False,
                    "frameLimit": 1,
                    "modelAvailability": {
                        "yoloSupported": True,
                        "yoloRealtime": True,
                    },
                },
            )
        )
        final = manager.wait(response["jobId"], 2.0)

        assert response["payload"]["backendRoute"]["backend"] == BACKEND_YOLO
        assert final is not None
        assert final.status == "succeeded"
        assert final.result["backendRoute"]["backend"] == BACKEND_YOLO
        frame_events = [event for event in events if event["event"] == "session.frame"]
        assert frame_events
        payload = frame_events[0]["payload"]
        assert payload["backend"] == BACKEND_YOLO
        assert payload["backendMeta"]["scoreAuthorized"] is False
        assert payload["backendMeta"]["calibrationStatus"] == "unvalidated"
        assert payload["backendMeta"]["displayScope"] == "limited"
        assert payload["multiPersonDetected"] is True
        assert payload["personCount"] == 2
        assert payload["reviewRequired"] is True
        assert payload["targetPolicy"] == "select_main_person_largest_box_highest_score"
    finally:
        ui_backend.DEFAULT_PREVIEW_SERVICE = previous


def test_ui_session_start_requires_capabilities_prevents_yolo_preview_route():
    options = ui_backend.normalize_session_options(
        {
            "source": "0",
            "enableHands": False,
            "requiresCapabilities": [CAPABILITY_FINGERS],
            "modelAvailability": {
                "yoloSupported": True,
                "yoloRealtime": True,
                "yoloRealtimeModel": "yolo26n",
            },
        }
    )

    assert options.requires_capabilities == (CAPABILITY_FINGERS,)
    assert options.backend_route["backend"] == BACKEND_MEDIAPIPE
    assert options.backend_route["modelProfile"] == "pose_only"
    assert options.backend_route["evalCompleteness"] == "partial"
    assert options.backend_route["skippedCapabilities"] == [CAPABILITY_FINGERS]
    assert options.backend_route["requestedBackend"] is None


def test_default_pipeline_factory_maps_yolo_realtime_profile_to_yolo26n_without_loading_ultralytics():
    options = ui_backend.normalize_session_options(
        {
            "source": "0",
            "enableHands": False,
            "modelAvailability": {
                "yoloSupported": True,
                "yoloRealtime": True,
            },
        }
    )
    pipe = ui_backend._default_pipeline_factory(options)

    try:
        assert pipe.__class__.__name__ == "YoloPoseAdapter"
        assert pipe.model_path.name == "yolo26n-pose.pt"
        assert pipe.warmup is True
        assert "ultralytics" not in sys.modules
    finally:
        close = getattr(pipe, "close", None)
        if close is not None:
            close()


def test_default_pipeline_factory_maps_yolo26s_profile_to_yolo26s_without_loading_ultralytics():
    options = ui_backend.normalize_session_options(
        {
            "source": "0",
            "enableHands": False,
            "modelAvailability": {
                "yoloSupported": True,
                "yoloRealtime": True,
                "yoloRealtimeModel": "yolo26s",
            },
        }
    )
    pipe = ui_backend._default_pipeline_factory(options)

    try:
        assert pipe.__class__.__name__ == "YoloPoseAdapter"
        assert pipe.model_path.name == "yolo26s-pose.pt"
        assert pipe.warmup is True
        assert "ultralytics" not in sys.modules
    finally:
        close = getattr(pipe, "close", None)
        if close is not None:
            close()


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
