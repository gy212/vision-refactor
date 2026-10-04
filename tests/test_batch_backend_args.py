# -*- coding: utf-8 -*-
"""Batch backend/layout argument contract tests (YOLO migration Issue #24)."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from batch import batch_dual_compare, batch_export_skeleton, batch_tech_eval  # noqa: E402
from batch.backend_options import meta_for_backend, normalize_backend_layout, yolo_batch_meta  # noqa: E402
from apps import ui_backend  # noqa: E402


def test_backend_layout_defaults_preserve_pose33_path():
    assert normalize_backend_layout("mediapipe", "pose33_v3") == ("mediapipe", "pose33_v3")


def test_yolo_pose33_combination_rejected():
    with pytest.raises(ValueError, match="YOLO 后端仅支持"):
        normalize_backend_layout("yolo", "pose33_v3")


def test_yolo_meta_marks_unvalidated_and_unauthorized():
    meta = yolo_batch_meta(
        {
            "model_name": "fake-yolo.pt",
            "valid_conf_thr": 0.6,
            "review_required": True,
        }
    )
    assert meta["backend"] == "yolo"
    assert meta["raw_layout"] == "pose33_like_coco17"
    assert meta["feature_layout"] == "body_core_v1"
    assert meta["capability"] == "body_only"
    assert meta["calibration_status"] == "unvalidated"
    assert meta["score_authorized"] is False
    assert meta["display_scope"] in {"limited", "internal"}
    assert meta["review_required"] is True


def test_yolo_body_core_batch_meta_defaults_to_yolo26l():
    meta = yolo_batch_meta()

    assert meta["backend"] == "yolo"
    assert meta["model_name"] == "yolo26l-pose.pt"
    assert meta["model_profile"] == "yolo26l-pose.pt"
    assert meta["display_scope"] == "internal"
    assert meta["score_authorized"] is False


def test_ui_high_quality_analysis_runs_yolo26l_internal_body_core():
    manager = ui_backend.BridgeJobManager()
    calls: list[dict[str, object]] = []

    def fake_body_core_analysis(video_path, **kwargs):
        calls.append({"video_path": video_path, **kwargs})
        return (
            np.zeros((4, 12, 2), dtype=np.float32),
            24.0,
            {
                "backend": "yolo",
                "model_name": "yolo26l-pose.pt",
                "raw_layout": "pose33_like_coco17",
                "feature_layout": "body_core_v1",
                "calibration_status": "unvalidated",
                "score_authorized": False,
                "display_scope": "internal",
                "body_core_valid_frame_ratio": 0.75,
                "multi_person_detected": False,
                "review_required": False,
                "max_persons": 1,
                "calibration_note": "internal only",
            },
        )

    service = ui_backend.TemplateAnalysisService(
        job_manager=manager,
        body_core_analysis=fake_body_core_analysis,
    )
    response = service.start_analysis_run(
        ui_backend.CommandRequest(
            command="analysis.run",
            request_id="req-body-core",
            payload={
                "videoPath": "student.mp4",
                "doCompare": False,
                "doTechEval": False,
                "qualityProfile": "high_quality",
                "enableHands": False,
                "modelAvailability": {"yoloSupported": True, "yolo26L": True},
            },
        )
    )

    assert response["ok"] is True
    route = response["payload"]["backendRoute"]
    assert route["backend"] == "yolo"
    assert route["modelProfile"] == "yolo26l"
    assert route["displayScope"] == "internal"
    assert route["scoreAuthorized"] is False

    final = manager.wait(response["jobId"], 2.0)

    assert final is not None
    assert final.status == "succeeded"
    assert calls == [
        {
            "video_path": "student.mp4",
            "backend": "yolo",
            "pose_variant": "full",
            "model_profile": "yolo26l",
            "should_stop": calls[0]["should_stop"],
        }
    ]
    assert callable(calls[0]["should_stop"])
    result = final.result["bodyCoreAnalysis"]
    assert "compare" not in final.result
    assert "techEval" not in final.result
    assert result["backend"] == "yolo"
    assert result["requestedBackend"] == "yolo"
    assert result["modelProfile"] == "yolo26l"
    assert result["modelName"] == "yolo26l-pose.pt"
    assert result["rawLayout"] == "pose33_like_coco17"
    assert result["featureLayout"] == "body_core_v1"
    assert result["scoreAuthorized"] is False
    assert result["calibrationStatus"] == "unvalidated"
    assert result["displayScope"] == "internal"
    assert result["score"] is None
    assert result["validFrameRatio"] == pytest.approx(0.75)


def test_ui_high_quality_missing_yolo26l_is_structured_error_without_job():
    service = ui_backend.TemplateAnalysisService(job_manager=ui_backend.BridgeJobManager())
    response = service.start_analysis_run(
        ui_backend.CommandRequest(
            command="analysis.run",
            request_id="req-missing-yolo26l",
            payload={
                "videoPath": "student.mp4",
                "doCompare": False,
                "doTechEval": False,
                "qualityProfile": "high_quality",
                "enableHands": False,
                "modelAvailability": {"yoloSupported": True, "yolo26L": False},
            },
        )
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "yolo26l_unavailable"
    route = response["payload"]["backendRoute"]
    assert route["ok"] is False
    assert route["requestedBackend"] == "yolo"
    assert route["modelProfile"] == "yolo26l"
    assert route["featureLayout"] == "body_core_v1"
    assert route["displayScope"] == "internal"


def test_meta_for_backend_uses_shared_router_contract_for_mediapipe_body_core():
    meta = meta_for_backend("mediapipe", {"review_required": True}, pose_variant="heavy")

    assert meta["backend"] == "mediapipe"
    assert meta["raw_layout"] == "pose33_v3"
    assert meta["feature_layout"] == "body_core_v1"
    assert meta["capability"] == "body_only"
    assert meta["calibration_status"] == "unvalidated"
    assert meta["score_authorized"] is False
    assert meta["display_scope"] == "internal"
    assert meta["review_required"] is True


def test_dual_compare_body_core_row_keeps_debug_scores_out_of_outward_columns(tmp_path):
    meta = yolo_batch_meta({"review_required": False})
    front = SimpleNamespace(
        template_path=tmp_path / "front.npz",
        video_path=tmp_path / "student.mp4",
        backend="yolo",
        feature_layout="body_core_v1",
        start_frame=1,
        end_frame=2,
        cost=1.3,
        calibration_status="unvalidated",
        multi_person_detected=False,
        multi_person_gate_source="",
        score=0.42,
        avg_cost=1.2,
        valid_frame_ratio=0.9,
        review_required=False,
        max_persons=1,
        multi_person_frames=0,
    )
    side = SimpleNamespace(
        template_path=tmp_path / "side.npz",
        video_path=tmp_path / "student.mp4",
        backend="yolo",
        feature_layout="body_core_v1",
        start_frame=3,
        end_frame=4,
        cost=1.4,
        calibration_status="unvalidated",
        multi_person_detected=False,
        multi_person_gate_source="",
        score=0.52,
        avg_cost=1.1,
        valid_frame_ratio=0.8,
        review_required=False,
        max_persons=1,
        multi_person_frames=0,
    )

    row = batch_dual_compare._body_core_row(tmp_path / "student.mp4", front, side, meta=meta)

    assert row["backend"] == "yolo"
    assert row["feature_layout"] == "body_core_v1"
    assert row["calibration_status"] == "unvalidated"
    assert row["score_authorized"] is False
    assert row["front_debug_score"] == pytest.approx(0.42)
    assert "front_score" not in row
    assert "combined_percent" not in row
    debug = batch_dual_compare._body_core_result_dict(front, meta=meta)
    assert debug["raw_layout"] == "pose33_like_coco17"
    assert debug["capability"] == "body_only"
    assert debug["display_scope"] == "internal"


def test_dual_compare_body_core_row_marks_multi_person_review(tmp_path):
    meta = yolo_batch_meta({"review_required": False})
    front = SimpleNamespace(
        score=None,
        avg_cost=1.2,
        valid_frame_ratio=0.9,
        review_required=True,
        max_persons=3,
        multi_person_frames=4,
    )
    side = SimpleNamespace(
        score=0.52,
        avg_cost=1.1,
        valid_frame_ratio=0.8,
        review_required=False,
        max_persons=1,
        multi_person_frames=0,
    )

    row = batch_dual_compare._body_core_row(tmp_path / "student.mp4", front, side, meta=meta)

    assert row["review_required"] is True
    assert row["review_status"] == "需人工复核"
    assert row["front_debug_score"] == ""
    assert row["max_persons"] == 3
    assert row["multi_person_frames"] == 4


def test_dual_compare_parallel_video_workers_keep_order_and_inner_workers_one(tmp_path, monkeypatch):
    front = tmp_path / "front.mp4"
    side = tmp_path / "side.mp4"
    front.write_bytes(b"fake")
    side.write_bytes(b"fake")
    student_dir = tmp_path / "students"
    student_dir.mkdir()
    slow = student_dir / "a_slow.mp4"
    fast = student_dir / "b_fast.mp4"
    slow.write_bytes(b"fake")
    fast.write_bytes(b"fake")
    out_dir = tmp_path / "out"
    calls: list[tuple[str, int]] = []

    def fake_template(video_path, *, pose_variant: str, out_path: Path):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            features=np.zeros((2, 22, 2), dtype=np.float32),
            meta=np.array({"pose_variant": pose_variant, "feature_layout": "pose33_v3"}, dtype=object),
        )
        return Path(out_path)

    def fake_compare(front_tpl, side_tpl, video_path, **kwargs):
        video_path = Path(video_path)
        calls.append((video_path.name, int(kwargs["workers"])))
        if video_path.name.startswith("a_"):
            time.sleep(0.05)
        return batch_dual_compare.ac.DualCompareResult(
            front_template_path=Path(front_tpl),
            side_template_path=Path(side_tpl),
            video_path=video_path,
            pose_variant="full",
            fps=30.0,
            front_score=0.7,
            side_score=0.8,
            combined_score=0.75,
            combined_percent=75 if video_path.name.startswith("a_") else 85,
            front_matches=(batch_dual_compare.ac.RepetitionMatch(1, 2, 0.1, 0.9),),
            side_matches=(batch_dual_compare.ac.RepetitionMatch(3, 4, 0.2, 0.8),),
            front_segment=(1, 2),
            side_segment=(3, 4),
        )

    monkeypatch.setattr(batch_dual_compare.ac, "create_template_from_video", fake_template)
    monkeypatch.setattr(batch_dual_compare.ac, "compare_video_to_dual_templates", fake_compare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_dual_compare.py",
            "--front",
            str(front),
            "--side",
            str(side),
            "--student_dir",
            str(student_dir),
            "--out_dir",
            str(out_dir),
            "--workers",
            "2",
        ],
    )

    batch_dual_compare.main()

    rows = list(csv.DictReader((out_dir / "compare_results.csv").open(encoding="utf-8-sig")))
    assert [Path(row["video"]).name for row in rows] == ["a_slow.mp4", "b_fast.mp4"]
    payloads = [json.loads(line) for line in (out_dir / "compare_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [Path(payload["video_path"]).name for payload in payloads] == ["a_slow.mp4", "b_fast.mp4"]
    assert sorted(calls) == [("a_slow.mp4", 1), ("b_fast.mp4", 1)]


def test_dual_compare_pairs_students_by_front_side_keywords(tmp_path, capsys):
    videos = [
        tmp_path / "张三_正面.mp4",
        tmp_path / "张三_侧面.mp4",
        tmp_path / "alice-front.mp4",
        tmp_path / "alice-side.mp4",
        tmp_path / "bob_front.mp4",
        tmp_path / "notes.mp4",
    ]
    for video in videos:
        video.write_bytes(b"fake")

    pairs = batch_dual_compare._pair_students(videos)

    assert [(student_id, front.name, side.name) for student_id, front, side in pairs] == [
        ("alice", "alice-front.mp4", "alice-side.mp4"),
        ("张三", "张三_正面.mp4", "张三_侧面.mp4"),
    ]
    out = capsys.readouterr().out
    assert "Skipping unclassified student videos" in out
    assert "notes.mp4" in out
    assert "Skipping unpaired student videos" in out
    assert "bob" in out


def test_dual_compare_pair_students_keeps_directories_isolated(tmp_path, capsys):
    dir_a = tmp_path / "class_a"
    dir_b = tmp_path / "class_b"
    dir_a.mkdir()
    dir_b.mkdir()
    front = dir_a / "alice_front.mp4"
    side = dir_b / "alice_side.mp4"
    front.write_bytes(b"fake")
    side.write_bytes(b"fake")

    pairs = batch_dual_compare._pair_students([front, side])

    assert pairs == []
    out = capsys.readouterr().out
    assert "Skipping unpaired student videos" in out
    assert "class_a" in out
    assert "class_b" in out


def test_dual_compare_paired_mode_uses_dual_streams_and_skips_raw_export(tmp_path, monkeypatch, capsys):
    front = tmp_path / "front.mp4"
    side = tmp_path / "side.mp4"
    front.write_bytes(b"fake")
    side.write_bytes(b"fake")
    student_dir = tmp_path / "students"
    student_dir.mkdir()
    for name in ("a_front.mp4", "a_side.mp4", "b_front.mp4", "b_side.mp4"):
        (student_dir / name).write_bytes(b"fake")
    out_dir = tmp_path / "out"
    calls: list[tuple[str, str, int]] = []

    def fake_template(video_path, *, pose_variant: str, out_path: Path):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            features=np.zeros((2, 22, 2), dtype=np.float32),
            meta=np.array({"pose_variant": pose_variant, "feature_layout": "pose33_v3"}, dtype=object),
        )
        return Path(out_path)

    def fake_compare_dual_streams(front_tpl, side_tpl, front_video, side_video, **kwargs):
        front_video = Path(front_video)
        side_video = Path(side_video)
        calls.append((front_video.name, side_video.name, int(kwargs["workers"])))
        if front_video.name.startswith("a_"):
            time.sleep(0.05)
        pct = 75 if front_video.name.startswith("a_") else 85
        return batch_dual_compare.ac.DualCompareResult(
            front_template_path=Path(front_tpl),
            side_template_path=Path(side_tpl),
            video_path=front_video,
            pose_variant="full",
            fps=30.0,
            front_score=0.7,
            side_score=0.8,
            combined_score=float(pct) / 100.0,
            combined_percent=pct,
            front_matches=(batch_dual_compare.ac.RepetitionMatch(1, 2, 0.1, 0.9),),
            side_matches=(batch_dual_compare.ac.RepetitionMatch(3, 4, 0.2, 0.8),),
            front_segment=(0, 9),
            side_segment=(0, 11),
        )

    monkeypatch.setattr(batch_dual_compare.ac, "create_template_from_video", fake_template)
    monkeypatch.setattr(batch_dual_compare.ac, "compare_dual_streams", fake_compare_dual_streams)
    monkeypatch.setattr(
        batch_dual_compare.ac,
        "compare_video_to_dual_templates",
        lambda *args, **kwargs: pytest.fail("paired mode must use compare_dual_streams"),
    )
    monkeypatch.setattr(
        batch_dual_compare,
        "extract_pose_raw",
        lambda *args, **kwargs: pytest.fail("--export_raw must be skipped in paired mode"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_dual_compare.py",
            "--front",
            str(front),
            "--side",
            str(side),
            "--student_dir",
            str(student_dir),
            "--out_dir",
            str(out_dir),
            "--workers",
            "2",
            "--paired",
            "--export_raw",
        ],
    )

    batch_dual_compare.main()

    rows = list(csv.DictReader((out_dir / "compare_results.csv").open(encoding="utf-8-sig")))
    assert [row["video"] for row in rows] == ["a", "b"]
    assert [int(row["combined_percent"]) for row in rows] == [75, 85]
    payloads = [json.loads(line) for line in (out_dir / "compare_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [payload["student_id"] for payload in payloads] == ["a", "b"]
    assert [Path(payload["front_video_path"]).name for payload in payloads] == ["a_front.mp4", "b_front.mp4"]
    assert sorted(calls) == [
        ("a_front.mp4", "a_side.mp4", 1),
        ("b_front.mp4", "b_side.mp4", 1),
    ]
    assert "--export_raw is not supported in --paired mode" in capsys.readouterr().out
    assert not (out_dir / "skeleton" / "a_front_raw_full.npz").exists()


def test_dual_compare_body_core_parallel_keeps_order_and_yolo_metadata(tmp_path, monkeypatch):
    from core import body_core_compare

    student_dir = tmp_path / "students"
    student_dir.mkdir()
    slow = student_dir / "a_slow.mp4"
    fast = student_dir / "b_fast.mp4"
    slow.write_bytes(b"fake")
    fast.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    def fake_create_body_core_template(video_path, *, backend, pose_variant, out_path, yolo_model):
        meta = yolo_batch_meta({"model_name": "fake-yolo.pt"})
        meta.update({"backend": backend, "pose_variant": pose_variant})
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            features=np.zeros((2, 12, 2), dtype=np.float32),
            meta=np.array(meta, dtype=object),
        )
        return Path(out_path)

    def fake_match_body_core_template(template_path, video_path, **kwargs):
        video_path = Path(video_path)
        if video_path.name.startswith("a_"):
            time.sleep(0.05)
        assert "precomputed_features" in kwargs
        return SimpleNamespace(
            template_path=Path(template_path),
            video_path=video_path,
            backend="yolo",
            feature_layout="body_core_v1",
            start_frame=1,
            end_frame=2,
            cost=1.0,
            avg_cost=0.5,
            score=0.4,
            calibration_status="unvalidated",
            valid_frame_ratio=0.9,
            review_required=False,
            multi_person_detected=False,
            max_persons=1,
            multi_person_frames=0,
            multi_person_gate_source="",
        )

    def fake_extract_body_core_features(video_path, **kwargs):
        video_path = Path(video_path)
        return (
            np.zeros((3, 12, 2), dtype=np.float32),
            30.0,
            {
                "backend": str(kwargs["backend"]),
                "display_scope": "internal",
                "body_core_valid_frame_ratio": 1.0,
            },
        )

    monkeypatch.setattr(body_core_compare, "create_body_core_template", fake_create_body_core_template)
    monkeypatch.setattr(body_core_compare, "extract_body_core_features", fake_extract_body_core_features)
    monkeypatch.setattr(body_core_compare, "match_body_core_template", fake_match_body_core_template)

    batch_dual_compare._run_body_core_batch(
        args=SimpleNamespace(pose="full", rules=False, workers=2),
        backend="yolo",
        front_video=tmp_path / "front.mp4",
        side_video=tmp_path / "side.mp4",
        student_dir=student_dir,
        out_dir=out_dir,
    )

    rows = list(csv.DictReader((out_dir / "compare_results.csv").open(encoding="utf-8-sig")))
    assert [Path(row["video"]).name for row in rows] == ["a_slow.mp4", "b_fast.mp4"]
    assert all(row["score_authorized"] == "False" for row in rows)

    payloads = [json.loads(line) for line in (out_dir / "compare_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [Path(payload["video_path"]).name for payload in payloads] == ["a_slow.mp4", "b_fast.mp4"]
    assert all(payload["front_debug_match"]["score_authorized"] is False for payload in payloads)
    assert all(payload["front_debug_match"]["display_scope"] == "internal" for payload in payloads)


def test_export_skeleton_body_core_npz_meta_contains_issue24_fields(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    video = src / "demo.mp4"
    video.write_bytes(b"fake")

    def fake_iter(_root: Path, *, skip_keywords: tuple[str, ...]):
        return [video]

    def fake_extract(video_path: Path, *, backend: str, pose_variant: str):
        assert backend == "yolo"
        assert pose_variant == "full"
        features = np.zeros((2, 12, 2), dtype=np.float32)
        meta = yolo_batch_meta({"model_name": "fake-yolo.pt"})
        meta.update(
            {
                "video": str(video_path),
                "name": video_path.stem,
                "fps": 30.0,
                "frame_count": 2,
                "width": 1280,
                "height": 720,
                "landmark_layout": "body_core_v1_normalized_xy",
                "skeleton_video": None,
            }
        )
        return features, meta

    monkeypatch.setattr(batch_export_skeleton, "_iter_videos", fake_iter)
    monkeypatch.setattr(batch_export_skeleton, "_extract_body_core_features", fake_extract)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_export_skeleton.py",
            "--source_dir",
            str(src),
            "--out_dir",
            str(tmp_path / "out"),
            "--backend",
            "yolo",
            "--feature-layout",
            "body_core_v1",
        ],
    )

    batch_export_skeleton.main()

    npz_path = next((tmp_path / "out").rglob("*_yolo_body_core_v1.npz"))
    data = np.load(npz_path, allow_pickle=True)
    meta = dict(data["meta"].item())
    assert data["features"].shape == (2, 12, 2)
    assert meta["backend"] == "yolo"
    assert meta["feature_layout"] == "body_core_v1"
    assert meta["calibration_status"] == "unvalidated"
    assert meta["score_authorized"] is False

    manifest = next((tmp_path / "out").rglob("manifest.csv"))
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig")))
    assert rows[0]["backend"] == "yolo"
    assert rows[0]["score_authorized"] == "False"


def test_export_skeleton_skip_existing_body_core_keeps_manifest_meta(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    old_video = src / "a_old.mp4"
    new_video = src / "b_new.mp4"
    old_video.write_bytes(b"fake")
    new_video.write_bytes(b"fake")
    out_dir = tmp_path / "out"
    existing_npz = out_dir / "a_old_yolo_body_core_v1.npz"
    existing_npz.parent.mkdir(parents=True)
    existing_meta = yolo_batch_meta({"model_name": "existing-yolo.pt"})
    existing_meta.update({"fps": 24.0, "frame_count": 3, "width": 640, "height": 480})
    np.savez_compressed(
        existing_npz,
        features=np.zeros((3, 12, 2), dtype=np.float32),
        meta=np.array(existing_meta, dtype=object),
    )

    def fake_iter(_root: Path, *, skip_keywords: tuple[str, ...]):
        return [old_video, new_video]

    def fake_extract(video_path: Path, *, backend: str, pose_variant: str):
        features = np.zeros((2, 12, 2), dtype=np.float32)
        meta = yolo_batch_meta({"model_name": "new-yolo.pt"})
        meta.update(
            {
                "video": str(video_path),
                "name": video_path.stem,
                "fps": 30.0,
                "frame_count": 2,
                "width": 1280,
                "height": 720,
                "landmark_layout": "body_core_v1_normalized_xy",
                "skeleton_video": None,
            }
        )
        return features, meta

    monkeypatch.setattr(batch_export_skeleton, "_iter_videos", fake_iter)
    monkeypatch.setattr(batch_export_skeleton, "_extract_body_core_features", fake_extract)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_export_skeleton.py",
            "--source_dir",
            str(src),
            "--out_dir",
            str(out_dir),
            "--backend",
            "yolo",
            "--feature-layout",
            "body_core_v1",
        ],
    )

    batch_export_skeleton.main()

    manifest = out_dir / "manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig")))
    assert len(rows) == 2
    assert rows[0]["video_name"] == "a_old.mp4"
    assert rows[0]["backend"] == "yolo"
    assert rows[0]["model_name"] == "existing-yolo.pt"
    assert rows[0]["score_authorized"] == "False"
    assert rows[1]["video_name"] == "b_new.mp4"
    assert rows[1]["backend"] == "yolo"
    assert rows[1]["model_name"] == "new-yolo.pt"


def test_export_skeleton_parallel_workers_keep_manifest_order(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    slow = src / "a_slow.mp4"
    fast = src / "b_fast.mp4"
    slow.write_bytes(b"fake")
    fast.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    def fake_iter(_root: Path, *, skip_keywords: tuple[str, ...]):
        return [slow, fast]

    def fake_extract(video_path: Path, *, backend: str, pose_variant: str):
        if video_path.name.startswith("a_"):
            time.sleep(0.05)
        features = np.zeros((2, 12, 2), dtype=np.float32)
        meta = yolo_batch_meta({"model_name": f"{video_path.stem}.pt"})
        meta.update(
            {
                "video": str(video_path),
                "name": video_path.stem,
                "fps": 30.0,
                "frame_count": 2,
                "width": 1280,
                "height": 720,
                "landmark_layout": "body_core_v1_normalized_xy",
                "skeleton_video": None,
            }
        )
        return features, meta

    monkeypatch.setattr(batch_export_skeleton, "_iter_videos", fake_iter)
    monkeypatch.setattr(batch_export_skeleton, "_extract_body_core_features", fake_extract)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_export_skeleton.py",
            "--source_dir",
            str(src),
            "--out_dir",
            str(out_dir),
            "--backend",
            "yolo",
            "--feature-layout",
            "body_core_v1",
            "--workers",
            "2",
        ],
    )

    batch_export_skeleton.main()

    rows = list(csv.DictReader((out_dir / "manifest.csv").open(encoding="utf-8-sig")))
    assert [row["video_name"] for row in rows] == ["a_slow.mp4", "b_fast.mp4"]
    assert [row["model_name"] for row in rows] == ["a_slow.pt", "b_fast.pt"]
    assert all(row["score_authorized"] == "False" for row in rows)


def test_tech_eval_yolo_body_core_skips_outward_scoring(tmp_path):
    meta = yolo_batch_meta({"review_required": True})
    meta.update({"max_persons": 2, "multi_person_frames": 5})

    row = batch_tech_eval._body_core_unvalidated_row(tmp_path / "multi.mp4", meta)

    assert row["backend"] == "yolo"
    assert row["feature_layout"] == "body_core_v1"
    assert row["calibration_status"] == "unvalidated"
    assert row["score_authorized"] is False
    assert row["review_required"] is True
    assert row["review_status"] == "需人工复核"
    assert row["重心(侧面优先)"] == "无法判定"
    assert row["发力顺序"] == "无法判定"


def test_tech_eval_full_jsonl_meta_marks_unvalidated(tmp_path, monkeypatch):
    video = tmp_path / "student.mp4"
    video.write_bytes(b"fake")

    monkeypatch.setattr(
        batch_tech_eval,
        "_body_core_meta_for_video",
        lambda v, *, backend, pose_variant: yolo_batch_meta({"model_name": "fake-yolo.pt"}),
    )

    args = SimpleNamespace(full=True, pose="full")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    batch_tech_eval._run_unvalidated_body_core_batch(
        videos=[video],
        out_dir=out_dir,
        args=args,
        backend="yolo",
    )

    line = (out_dir / "tech_report_full.jsonl").read_text(encoding="utf-8").splitlines()[0]
    payload = json.loads(line)
    assert payload["tech_eval_status"] == "skipped"
    assert payload["skip_reason"] == "unvalidated_body_core_v1"
    assert payload["score_authorized"] is False
    assert payload["meta"]["calibration_status"] == "unvalidated"
