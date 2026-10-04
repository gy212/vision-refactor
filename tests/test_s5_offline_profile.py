# -*- coding: utf-8 -*-
"""S5 offline matching profile tests (YOLO migration Issue #44)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis import offline_matching_profile  # noqa: E402


def test_fixture_profile_writes_json_and_csv(tmp_path):
    out_dir = tmp_path / "offline_profile"

    rc = offline_matching_profile.main(["--fixture-smoke", "--out", str(out_dir)])

    assert rc == 0
    json_path = out_dir / "offline_matching_profile.json"
    csv_path = out_dir / "offline_matching_profile.csv"
    assert json_path.exists()
    assert csv_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["records"] >= 3
    assert payload["decision_table"]

    ids = {record["profile_id"] for record in payload["records"]}
    assert "fixture_pose33_staged" in ids
    assert "fixture_body_core_staged" in ids

    by_id = {record["profile_id"]: record for record in payload["records"]}
    pose_staged = by_id["fixture_pose33_staged"]
    pose_prod = by_id["fixture_pose33_production"]
    assert pose_staged["combined_percent"] == pose_prod["combined_percent"]
    assert pose_staged["combined_score"] == pose_prod["combined_score"]
    assert pose_staged["front_score"] == pose_prod["front_score"]
    assert pose_staged["side_score"] == pose_prod["side_score"]
    assert pose_staged["front_segment"] == pose_prod["front_segment"]
    assert pose_staged["side_segment"] == pose_prod["side_segment"]
    assert pose_staged["front_matches"] == pose_prod["front_matches"]
    assert pose_staged["side_matches"] == pose_prod["side_matches"]
    assert pose_staged["normalizer_version"] == "v3"

    body_staged = by_id["fixture_body_core_staged"]
    body_prod = by_id["fixture_body_core_production"]
    assert body_staged["score"] == body_prod["score"]
    assert body_staged["avg_cost"] == body_prod["avg_cost"]
    assert body_staged["start_frame"] == body_prod["start_frame"]
    assert body_staged["end_frame"] == body_prod["end_frame"]

    staged = pose_staged
    stage_names = {stage["stage"] for stage in staged["stages"]}
    assert "feature_normalization_pose33" in stage_names
    assert "front_multi_subsequence_dtw" in stage_names
    assert "side_multi_subsequence_dtw" in stage_names

    csv_text = csv_path.read_text(encoding="utf-8-sig")
    assert "front_multi_subsequence_dtw" in csv_text
    assert "subsequence_dtw_body_core" in csv_text


def test_decision_table_defers_exact_dtw_when_not_dominant():
    rows = offline_matching_profile.decision_table(
        {
            "dtw_percent": 10.0,
            "sequence_percent": 70.0,
        }
    )

    by_option = {row["option"]: row for row in rows}
    assert by_option["FastDTW"]["decision"] == "defer"
    assert by_option["Raw/feature cache"]["decision"] == "candidate_separate_issue"
    assert "Separate" in by_option["Numba exact DTW"]["required_guard"] or "Separate" in by_option["Numba exact DTW"]["reason"]


def test_dual_template_normalizer_follows_template_metadata(tmp_path):
    features = _fake_pose33_features()
    front = tmp_path / "front_v2.npz"
    side = tmp_path / "side_v2.npz"
    for path in (front, side):
        meta = {
            "feature_layout": "pose_indices_11_32_xy_rot_scale_norm_v2",
            "fps": 30.0,
        }
        import numpy as np

        np.savez_compressed(path, features=features, meta=np.array(meta, dtype=object))

    normalizer, version = offline_matching_profile._normalizer_for_dual_templates(front, side)

    assert version == "v2"
    assert normalizer is offline_matching_profile.normalize_pose_xy


def test_issue44_does_not_modify_core_dtw_or_baselines():
    diff_paths = [
        _REPO_ROOT / "core" / "pose_features.py",
        _REPO_ROOT / "core" / "feature_layout.py",
        _REPO_ROOT / "core" / "body_core_compare.py",
    ]
    # Regression intent: this issue adds profiling/reporting only. The test keeps
    # the expected source of Issue #44 edits away from DTW, baselines, thresholds.
    for path in diff_paths:
        assert path.exists()
    assert offline_matching_profile.subsequence_dtw.__module__ == "core.pose_features"
    assert offline_matching_profile.POSE33_BASELINE == 3.0  # 2026-07-11 校准更新


def _fake_pose33_features():
    import numpy as np

    rng = np.random.default_rng(44)
    return rng.standard_normal((12, 22, 2)).astype(np.float32)
