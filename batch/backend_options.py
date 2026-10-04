# -*- coding: utf-8 -*-
"""Shared backend/layout CLI helpers for batch entry points."""

from __future__ import annotations

import argparse
from typing import Any

from core.backend_router import (
    BACKEND_MEDIAPIPE,
    BACKEND_YOLO,
    FEATURE_LAYOUT_BODY_CORE,
    FEATURE_LAYOUT_POSE33,
    mediapipe_body_core_metadata,
    validate_backend_layout,
    yolo_metadata,
)
from core.body_core_compare import DEFAULT_YOLO26L_MODEL_NAME

BATCH_META_FIELDS: tuple[str, ...] = (
    "backend",
    "model_name",
    "raw_layout",
    "feature_layout",
    "capability",
    "confidence_kind",
    "validity_policy",
    "valid_conf_thr",
    "calibration_status",
    "score_authorized",
    "display_scope",
    "review_required",
)


def add_backend_layout_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--backend",
        default=BACKEND_MEDIAPIPE,
        choices=[BACKEND_MEDIAPIPE, BACKEND_YOLO],
        help="Pose backend (default: mediapipe). YOLO is debug/calibration only.",
    )
    ap.add_argument(
        "--feature-layout",
        dest="feature_layout",
        default=FEATURE_LAYOUT_POSE33,
        choices=[FEATURE_LAYOUT_POSE33, FEATURE_LAYOUT_BODY_CORE],
        help="Feature layout (default: pose33_v3).",
    )


def normalize_backend_layout(backend: str, feature_layout: str) -> tuple[str, str]:
    backend = str(backend or BACKEND_MEDIAPIPE).lower()
    feature_layout = str(feature_layout or FEATURE_LAYOUT_POSE33)
    decision = validate_backend_layout(backend, feature_layout)
    return decision.backend, decision.feature_layout


def is_default_pose33_path(backend: str, feature_layout: str) -> bool:
    backend, feature_layout = normalize_backend_layout(backend, feature_layout)
    return backend == BACKEND_MEDIAPIPE and feature_layout == FEATURE_LAYOUT_POSE33


def yolo_batch_meta(source_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    source_meta = dict(source_meta or {})
    source_meta.setdefault("model_name", DEFAULT_YOLO26L_MODEL_NAME)
    return yolo_metadata(source_meta)


def mediapipe_body_core_meta(source_meta: dict[str, Any] | None = None, *, pose_variant: str = "full") -> dict[str, Any]:
    return mediapipe_body_core_metadata(source_meta, pose_variant=pose_variant)


def meta_for_backend(backend: str, source_meta: dict[str, Any] | None = None, *, pose_variant: str = "full") -> dict[str, Any]:
    if backend == BACKEND_YOLO:
        return yolo_batch_meta(source_meta)
    return mediapipe_body_core_meta(source_meta, pose_variant=pose_variant)


def csv_meta_fields(meta: dict[str, Any]) -> dict[str, Any]:
    return {field: meta.get(field, "") for field in BATCH_META_FIELDS}
