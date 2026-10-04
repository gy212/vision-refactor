# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core import action_compare as ac  # noqa: E402
from core import body_core_compare  # noqa: E402
from core.body_core_compare import (  # noqa: E402
    CALIBRATION_STATUS_UNVALIDATED,
    create_body_core_template,
    extract_body_core_features,
    match_body_core_template,
)
from core.rule_scoring import extract_pose_raw, extract_pose_raw_series, slice_pose_raw_series  # noqa: E402
from tests import golden_harness as H  # noqa: E402


FIX_DIR = Path(__file__).resolve().parent / "fixtures" / "pose33_v3"
FRONT_TPL = FIX_DIR / "front_template.npz"
SIDE_TPL = FIX_DIR / "side_template.npz"
STUDENT = "golden://student.mp4"
FRONT_SRC = "golden://front_src.mp4"
FPS = 30.0


def _load_raw(name: str) -> np.ndarray:
    d = np.load(FIX_DIR / name, allow_pickle=True)
    return np.asarray(d["landmarks"], dtype=np.float32)


@pytest.fixture()
def registered_videos():
    H.clear_registry()
    H.register_video(STUDENT, _load_raw("student_raw.npz"), fps=FPS)
    H.register_video(FRONT_SRC, _load_raw("front_src_raw.npz"), fps=FPS)
    yield
    H.clear_registry()


def test_pose33_raw_series_slice_matches_legacy_extract_pose_raw(registered_videos):
    with H.replay_context():
        series = extract_pose_raw_series(STUDENT, pose_variant="full")
        raw, meta = slice_pose_raw_series(series, start_frame=7, end_frame=25)
        legacy_raw, legacy_meta = extract_pose_raw(
            STUDENT,
            pose_variant="full",
            start_frame=7,
            end_frame=25,
        )

    np.testing.assert_array_equal(raw, legacy_raw)
    np.testing.assert_array_equal(meta["valid_mask"], legacy_meta["valid_mask"])
    assert meta["start_frame"] == legacy_meta["start_frame"] == 7
    assert meta["end_frame"] == legacy_meta["end_frame"] == 25
    assert meta["frame_count"] == legacy_meta["frame_count"]
    assert meta["validity_policy"] == legacy_meta["validity_policy"]


def test_dual_compare_reuses_one_raw_series_for_rules_and_joint_errors(monkeypatch, registered_videos):
    real_extract_series = ac.extract_pose_raw_series
    calls: list[str] = []

    def counted_extract(video_path, *, pose_variant):
        calls.append(str(video_path))
        return real_extract_series(video_path, pose_variant=pose_variant)

    monkeypatch.setattr(ac, "extract_pose_raw_series", counted_extract)

    with H.replay_context():
        res = ac.compare_video_to_dual_templates(
            FRONT_TPL,
            SIDE_TPL,
            STUDENT,
            pose_variant="full",
            enable_rules=True,
            enable_error_analysis=True,
        )

    assert calls == [str(Path(STUDENT))]
    assert res.front_rule_score is not None
    assert res.side_rule_score is not None
    assert res.front_joint_errors is not None
    assert res.side_joint_errors is not None


def test_body_core_match_accepts_precomputed_features_without_reextract(monkeypatch, tmp_path, registered_videos):
    out_tpl = tmp_path / "mp_body_core_v1.npz"
    with H.replay_context():
        tpl_path = create_body_core_template(
            FRONT_SRC,
            backend="mediapipe",
            pose_variant="full",
            out_path=out_tpl,
        )
        precomputed = extract_body_core_features(
            STUDENT,
            backend="mediapipe",
            pose_variant="full",
        )

    def fail_extract(*args, **kwargs):
        raise AssertionError("precomputed body_core features should avoid re-extraction")

    monkeypatch.setattr(body_core_compare, "extract_body_core_features", fail_extract)
    res = match_body_core_template(
        tpl_path,
        STUDENT,
        backend="mediapipe",
        pose_variant="full",
        precomputed_features=precomputed,
    )

    assert res.calibration_status == CALIBRATION_STATUS_UNVALIDATED
    assert res.score is not None
    assert 0 <= float(res.score) <= 1


def test_body_core_precomputed_backend_mismatch_is_rejected(tmp_path):
    tpl = tmp_path / "body_core.npz"
    np.savez_compressed(
        tpl,
        features=np.zeros((2, 12, 2), dtype=np.float32),
        meta=np.array(
            {
                "feature_layout": "body_core_v1",
                "backend": "mediapipe",
                "pose_variant": "full",
                "baseline": 1.2826,
            },
            dtype=object,
        ),
    )

    precomputed = (
        np.zeros((2, 12, 2), dtype=np.float32),
        30.0,
        {"backend": "yolo"},
    )

    with pytest.raises(ValueError, match="backend"):
        match_body_core_template(
            tpl,
            STUDENT,
            backend="mediapipe",
            pose_variant="full",
            precomputed_features=precomputed,
        )
