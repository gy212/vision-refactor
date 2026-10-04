from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import calibrate_body_core, spike_yolo_baseline  # noqa: E402


REQUIRED_YOLO_METADATA = {
    "backend": "yolo",
    "raw_layout": "pose33_like_coco17",
    "feature_layout": "body_core_v1",
    "capability": "body_only",
    "score_authorized": False,
    "calibration_status": "unvalidated",
    "display_scope": "internal",
}


def test_spike_yolo_records_and_keypoint_export_include_authorization_metadata(tmp_path: Path) -> None:
    mp_run = spike_yolo_baseline.BackendRun(
        backend="mediapipe",
        model_name="pose_full",
        n_frames=1,
        landmarks=np.zeros((1, 33, 4), dtype=np.float32),
        valid_mask=np.ones((1, 33), dtype=bool),
        person_counts=[1],
        selected_indices=[0],
        track_ids=[None],
        boxes_xywh_norm=[None],
    )
    yolo_run = spike_yolo_baseline.BackendRun(
        backend="yolo",
        model_name="yolo26n-pose.pt",
        n_frames=1,
        landmarks=np.zeros((1, 33, 4), dtype=np.float32),
        valid_mask=np.ones((1, 33), dtype=bool),
        person_counts=[1],
        selected_indices=[0],
        track_ids=[None],
        boxes_xywh_norm=[None],
        extra={"calibration_status": "validated"},
    )

    frame_export = spike_yolo_baseline._frame_export(yolo_run, 0)
    for key, value in REQUIRED_YOLO_METADATA.items():
        assert frame_export[key] == value

    jsonl_path = tmp_path / "spike_keypoints.jsonl"
    spike_yolo_baseline._append_keypoints_jsonl(jsonl_path, tmp_path / "sample.mp4", mp_run, yolo_run)
    assert '"score_authorized":false' in jsonl_path.read_text(encoding="utf-8")
    assert '"calibration_status":"unvalidated"' in jsonl_path.read_text(encoding="utf-8")

    rec = {
        "video": "sample.mp4",
        "frames": 1,
        "mediapipe": {"fps_infer": 1, "miss_rate": 0, "jitter_body_core": None},
        "yolo": {
            **REQUIRED_YOLO_METADATA,
            "fps_infer": 2,
            "miss_rate": 0,
            "jitter_body_core": None,
            "multi_person_frames": 0,
            "max_persons": 1,
        },
        "fps_speedup_infer": 2,
        "cross_backend_body_core_diff": {"median_diff": 0, "p90_diff": 0},
    }
    csv_path = tmp_path / "spike.csv"
    spike_yolo_baseline._write_csv(csv_path, [rec])
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    assert rows[0]["yolo_backend"] == "yolo"
    assert rows[0]["yolo_score_authorized"] == "False"
    assert rows[0]["yolo_calibration_status"] == "unvalidated"
    assert rows[0]["yolo_display_scope"] == "internal"


def test_calibration_outputs_include_yolo_authorization_metadata(tmp_path: Path) -> None:
    meta = calibrate_body_core.YOLO_ARTIFACT_METADATA
    assert meta == REQUIRED_YOLO_METADATA

    cache_path = tmp_path / "raw_cache.npz"
    np.savez_compressed(
        cache_path,
        mp_landmarks=np.zeros((1, 33, 4), dtype=np.float32),
        mp_valid_mask=np.ones((1, 33), dtype=bool),
        yolo_landmarks=np.zeros((1, 33, 4), dtype=np.float32),
        meta=np.array(
            {
                "yolo_artifact_metadata": dict(meta),
                **{f"yolo_{key}": value for key, value in meta.items()},
            },
            dtype=object,
        ),
    )
    loaded = dict(np.load(cache_path, allow_pickle=True)["meta"].item())
    assert loaded["yolo_artifact_metadata"] == REQUIRED_YOLO_METADATA
    assert loaded["yolo_score_authorized"] is False
    assert loaded["yolo_calibration_status"] == "unvalidated"

    rows = [
        {
            "template": "a",
            "video": "b",
            "yolo_bodycore_score": 0.5,
            **{f"yolo_{key}": value for key, value in meta.items()},
        }
    ]
    csv_path = tmp_path / "calibration_pairwise.csv"
    calibrate_body_core._write_csv(csv_path, rows)
    exported = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    assert exported[0]["yolo_backend"] == "yolo"
    assert exported[0]["yolo_score_authorized"] == "False"
    assert exported[0]["yolo_calibration_status"] == "unvalidated"

    gate_rows = [
        {
            "sample_id": "sample-a",
            "declared_multi_person": "unknown_or_false",
            "max_persons": 1,
            "multi_person_frames": 0,
            "review_required": False,
            "gate_status": "pass",
            "in_calibration_set": True,
            **{f"yolo_{key}": value for key, value in meta.items()},
        }
    ]
    gate_csv_path = tmp_path / "calibration_multi_person_gate.csv"
    calibrate_body_core._write_csv(gate_csv_path, gate_rows)
    gate_exported = list(csv.DictReader(gate_csv_path.open(encoding="utf-8-sig")))
    assert gate_exported[0]["yolo_backend"] == "yolo"
    assert gate_exported[0]["yolo_score_authorized"] == "False"
    assert gate_exported[0]["yolo_calibration_status"] == "unvalidated"
    assert gate_exported[0]["yolo_display_scope"] == "internal"
