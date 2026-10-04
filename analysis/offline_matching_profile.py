# -*- coding: utf-8 -*-
"""Offline matching profile harness for S5 Issue #44.

This module profiles the existing offline matching algorithms without changing
their scoring semantics. It intentionally lives under ``analysis/`` and imports
private helpers from the production modules so the profile can mirror the hot
path while keeping timers out of core scoring code.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.action_compare import (  # noqa: E402
    _longest_segment,
    _multi_subsequence_matches,
    _normalizer_version_from_layout,
    _select_representative_cycle,
    _smooth_1d,
    _trimmed_mean,
    _runtime_feature_layout,
    compare_video_to_dual_templates,
    normalize_template_meta,
)
from core.body_core_compare import (  # noqa: E402
    BODY_CORE_V1_CALIBRATED_BASELINE,
    CALIBRATION_STATUS_UNVALIDATED,
    BODY_CORE_CALIBRATION_NOTE,
    match_body_core_template,
)
from core.feature_layout import BODY_CORE_V1, POSE33_V3  # noqa: E402
from core.paths import models_dir  # noqa: E402
from core.pose_features import (  # noqa: E402
    DEFAULT_VALID_CONF_THR,
    derive_valid_mask,
    find_active_range,
    motion_energy,
    normalize_pose_body_core_v1,
    normalize_pose_xy,
    normalize_pose_xy_v1,
    normalize_pose_xy_v3,
    pose_view_score,
    subsequence_dtw,
)
from core.vision_pipeline import MediaPipePipeline, PipelineConfig  # noqa: E402


POSE33_BASELINE = 3.0  # 2026-07-11 校准：2.0 → 3.0（同人自复现 avg_cost≈0.333 锚到 ~0.90 分）


@dataclass
class StageTimer:
    name: str
    seconds: float = 0.0
    count: int = 0


@dataclass
class StageSet:
    timers: dict[str, StageTimer] = field(default_factory=dict)

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - start)

    def add(self, name: str, seconds: float, *, count: int = 1) -> None:
        timer = self.timers.setdefault(name, StageTimer(name=name))
        timer.seconds += float(seconds)
        timer.count += int(count)

    def total(self) -> float:
        return float(sum(t.seconds for t in self.timers.values()))

    def records(self) -> list[dict[str, Any]]:
        total = self.total()
        rows: list[dict[str, Any]] = []
        for t in self.timers.values():
            rows.append(
                {
                    "stage": t.name,
                    "seconds": round(float(t.seconds), 6),
                    "percent": round(100.0 * float(t.seconds) / total, 2) if total > 0 else 0.0,
                    "count": int(t.count),
                }
            )
        return rows


class _Pt:
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, row: np.ndarray) -> None:
        self.x = float(row[0])
        self.y = float(row[1])
        self.z = float(row[2]) if row.shape[0] > 2 else 0.0
        self.visibility = float(row[3]) if row.shape[0] > 3 else 0.0


class _RowView:
    __slots__ = ("_row",)

    def __init__(self, row: np.ndarray) -> None:
        self._row = np.asarray(row, dtype=np.float32)

    def __getitem__(self, idx: int) -> _Pt:
        return _Pt(self._row[idx])


def _score(avg_cost: float | None, baseline: float) -> float:
    if avg_cost is None:
        return 0.0
    return float(float(baseline) / (float(baseline) + float(avg_cost)))


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) > 0 else 0.0


def _profile_env() -> dict[str, Any]:
    env = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
    }
    try:
        import mediapipe as mp

        env["mediapipe"] = mp.__version__
    except Exception as exc:  # noqa: BLE001
        env["mediapipe_error"] = repr(exc)
    return env


def _normalizer_for_dual_templates(front_template: Path, side_template: Path) -> tuple[Callable[[Any], np.ndarray | None], str]:
    with np.load(front_template, allow_pickle=True) as tpl_f:
        raw_meta_f = dict(tpl_f["meta"].item() or {})
    with np.load(side_template, allow_pickle=True) as tpl_s:
        raw_meta_s = dict(tpl_s["meta"].item() or {})
    meta_f = normalize_template_meta(dict(raw_meta_f))
    meta_s = normalize_template_meta(dict(raw_meta_s))
    layout_f = _runtime_feature_layout(raw_meta_f, meta_f)
    layout_s = _runtime_feature_layout(raw_meta_s, meta_s)
    version_f = _normalizer_version_from_layout(layout_f)
    version_s = _normalizer_version_from_layout(layout_s)
    if version_f != version_s:
        raise ValueError("Front/side templates use different normalizer versions; cannot stage-profile as one production path.")
    if version_f == "v3":
        return normalize_pose_xy_v3, "v3"
    if version_f == "v2":
        return normalize_pose_xy, "v2"
    return normalize_pose_xy_v1, "v1"


def _normalize_pose33_from_raw(
    landmarks: np.ndarray,
    *,
    fps: float,
    stages: StageSet,
    normalizer: Callable[[Any], np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray]:
    feats: list[np.ndarray] = []
    views: list[float] = []
    last_view = 0.0
    zero = np.zeros(tuple(int(x) for x in POSE33_V3.shape), dtype=np.float32)
    for row in np.asarray(landmarks, dtype=np.float32):
        rv = _RowView(row)
        with stages.time("feature_normalization_pose33"):
            f = normalizer(rv)
        if f is None:
            f = feats[-1].copy() if feats else zero.copy()
        feats.append(f)
        with stages.time("view_score"):
            v = pose_view_score(rv)
        if v is None:
            v = last_view
        else:
            last_view = float(v)
        views.append(float(v))
    if not feats:
        return np.zeros((0, *POSE33_V3.shape), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(feats).astype(np.float32), np.asarray(views, dtype=np.float32)


def _extract_pose33_from_video(
    video_path: Path,
    *,
    pose_variant: str,
    limit_frames: int | None,
    stages: StageSet,
    normalizer: Callable[[Any], np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    feats: list[np.ndarray] = []
    views: list[float] = []
    last_view = 0.0
    zero = np.zeros(tuple(int(x) for x in POSE33_V3.shape), dtype=np.float32)
    i = 0
    pipe: MediaPipePipeline | None = None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        pipe = MediaPipePipeline(
            models_dir=models_dir(),
            cfg=PipelineConfig(pose_variant=pose_variant, running_mode="video", enable_hands=False),
        )
        while True:
            if limit_frames is not None and i >= int(limit_frames):
                break
            with stages.time("video_io_read"):
                ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(i * 1000.0 / max(1e-6, fps))
            with stages.time("model_inference_mediapipe_pose"):
                pose_landmarks, _hands = pipe.infer(frame, timestamp_ms=timestamp_ms)
            with stages.time("feature_normalization_pose33"):
                f = normalizer(pose_landmarks)
            if f is None:
                f = feats[-1].copy() if feats else zero.copy()
            feats.append(f)
            with stages.time("view_score"):
                v = pose_view_score(pose_landmarks)
            if v is None:
                v = last_view
            else:
                last_view = float(v)
            views.append(float(v))
            i += 1
    finally:
        cap.release()
        if pipe is not None:
            pipe.close()
    if not feats:
        raise RuntimeError(f"视频未产出 pose33_v3 特征：{video_path}")
    return np.stack(feats).astype(np.float32), np.asarray(views, dtype=np.float32), fps


def _split_views(seq: np.ndarray, view_scores: np.ndarray, fps: float, stages: StageSet) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    with stages.time("view_split"):
        front_seg: tuple[int, int] | None = None
        side_seg: tuple[int, int] | None = None
        if view_scores.size == seq.shape[0] and seq.shape[0] >= 10:
            vs = np.nan_to_num(view_scores.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            k = int(max(5, min(51, round(float(fps) * 0.5))))
            vs_s = _smooth_1d(vs, k=k)
            thr = float(np.median(vs_s))
            front_mask = vs_s >= thr
            front_seg = _longest_segment(front_mask)
            side_seg = _longest_segment(~front_mask)
            margin = int(max(5, min(20, round(float(fps) * 0.3))))

            def shrink(seg: tuple[int, int] | None) -> tuple[int, int] | None:
                if seg is None:
                    return None
                s, e = int(seg[0]), int(seg[1])
                if e - s + 1 > (2 * margin + 10):
                    s += margin
                    e -= margin
                if e <= s:
                    return None
                return s, e

            front_seg = shrink(front_seg)
            side_seg = shrink(side_seg)
    return front_seg, side_seg


def _profile_multi_view(
    *,
    label: str,
    front_template: Path,
    side_template: Path,
    seq: np.ndarray,
    view_scores: np.ndarray,
    fps: float,
    stages: StageSet,
    normalizer_version: str,
    baseline: float = POSE33_BASELINE,
) -> dict[str, Any]:
    with stages.time("template_load_pose33"):
        tpl_f = np.load(front_template, allow_pickle=True)
        tpl_s = np.load(side_template, allow_pickle=True)
        feat_f = np.asarray(tpl_f["features"], dtype=np.float32)
        feat_s = np.asarray(tpl_s["features"], dtype=np.float32)
        meta_f = dict(tpl_f["meta"].item() or {})
        meta_s = dict(tpl_s["meta"].item() or {})

    front_seg, side_seg = _split_views(seq, view_scores, fps, stages)

    def score_view(name: str, tpl_features: np.ndarray, tpl_meta: dict, seg: tuple[int, int] | None) -> tuple[float, list[Any], dict[str, Any]]:
        with stages.time(f"{name}_query_select"):
            tpl_fps = float(tpl_meta.get("fps") or fps)
            query = _select_representative_cycle(tpl_features, fps=tpl_fps)
        if seg is None:
            seg_s, seg_e = 0, int(seq.shape[0] - 1)
        else:
            seg_s, seg_e = int(seg[0]), int(seg[1])
            seg_s = max(0, min(seg_s, int(seq.shape[0] - 1)))
            seg_e = max(seg_s, min(seg_e, int(seq.shape[0] - 1)))

        seg_seq = seq[seg_s : seg_e + 1]
        seg_offset = seg_s
        with stages.time(f"{name}_active_range"):
            if seg_seq.shape[0] >= 2:
                energy = motion_energy(seg_seq.reshape(seg_seq.shape[0], -1))
                a_s, a_e = find_active_range(energy, pad=10)
                a_s = max(0, min(int(a_s), int(seg_seq.shape[0] - 1)))
                a_e = max(a_s, min(int(a_e), int(seg_seq.shape[0] - 1)))
                seg_seq = seg_seq[a_s : a_e + 1]
                seg_offset += int(a_s)

        exclusion = max(3, int(round(0.2 * float(query.shape[0]))))
        max_matches = int(min(30, max(1, round(float(seg_seq.shape[0]) / max(1.0, float(query.shape[0]))))))
        with stages.time(f"{name}_multi_subsequence_dtw"):
            matches = _multi_subsequence_matches(
                query,
                seg_seq,
                baseline=float(baseline),
                max_matches=max_matches,
                exclusion=exclusion,
                offset=seg_offset,
            )
        if not matches and seg_seq.size > 0:
            with stages.time(f"{name}_fallback_subsequence_dtw"):
                cost, s, e = subsequence_dtw(query, seg_seq)
            avg = float(cost / max(1, int(query.shape[0])))
            score = _score(avg, baseline)
            return score, [], {
                "query_frames": int(query.shape[0]),
                "segment_frames": int(seg_seq.shape[0]),
                "matches": 1,
                "fallback": True,
                "start_frame": int(seg_offset + s),
                "end_frame": int(seg_offset + e),
            }
        scores = [m.score for m in matches]
        return _trimmed_mean(scores), matches, {
            "query_frames": int(query.shape[0]),
            "segment_frames": int(seg_seq.shape[0]),
            "matches": int(len(matches)),
            "fallback": False,
        }

    front_score, front_matches, front_meta = score_view("front", feat_f, meta_f, front_seg)
    side_score, side_matches, side_meta = score_view("side", feat_s, meta_s, side_seg)
    combined = float(np.clip((0.4 * float(front_score)) + (0.6 * float(side_score)), 0.0, 1.0))
    return {
        "profile_id": label,
        "target": "compare_video_to_dual_templates",
        "mode": "staged",
        "normalizer_version": str(normalizer_version),
        "frames": int(seq.shape[0]),
        "fps": float(fps),
        "front_score": float(front_score),
        "side_score": float(side_score),
        "combined_score": float(combined),
        "combined_percent": int(np.clip(int(round(combined * 100.0)), 0, 100)),
        "front_segment": None if front_seg is None else [int(front_seg[0]), int(front_seg[1])],
        "side_segment": None if side_seg is None else [int(side_seg[0]), int(side_seg[1])],
        "front_query_frames": front_meta["query_frames"],
        "side_query_frames": side_meta["query_frames"],
        "front_segment_frames": front_meta["segment_frames"],
        "side_segment_frames": side_meta["segment_frames"],
        "front_matches": int(len(front_matches)) if front_matches else int(front_meta["matches"]),
        "side_matches": int(len(side_matches)) if side_matches else int(side_meta["matches"]),
        "stages": stages.records(),
    }


def _body_core_from_raw(landmarks: np.ndarray, valid_mask: np.ndarray, stages: StageSet) -> tuple[np.ndarray, int]:
    feats: list[np.ndarray] = []
    zero = np.zeros(tuple(int(x) for x in BODY_CORE_V1.shape), dtype=np.float32)
    core_idx = np.asarray(BODY_CORE_V1.source_indices, dtype=int)
    valid_core_frames = 0
    for t in range(int(landmarks.shape[0])):
        core_valid = bool(np.all(valid_mask[t, core_idx]))
        with stages.time("feature_normalization_body_core"):
            f = normalize_pose_body_core_v1(landmarks[t]) if core_valid else None
        if f is None:
            f = feats[-1].copy() if feats else zero.copy()
        else:
            valid_core_frames += 1
        feats.append(f)
    if not feats:
        return np.zeros((0, *BODY_CORE_V1.shape), dtype=np.float32), 0
    return np.stack(feats).astype(np.float32), int(valid_core_frames)


def _write_fixture_body_core_template(path: Path, raw: np.ndarray, *, fps: float = 30.0) -> Path:
    stages = StageSet()
    mask = derive_valid_mask(raw, DEFAULT_VALID_CONF_THR)
    feats, valid_frames = _body_core_from_raw(raw, mask, stages)
    seq = feats.reshape(feats.shape[0], -1)
    energy = motion_energy(seq)
    start, end = find_active_range(energy, pad=10)
    start = max(0, min(int(start), int(feats.shape[0] - 1)))
    end = max(start, min(int(end), int(feats.shape[0] - 1)))
    meta = {
        "video": "fixture://body_core_template.mp4",
        "fps": float(fps),
        "frame_count": int(feats.shape[0]),
        "start_frame": int(start),
        "end_frame": int(end),
        "backend": "mediapipe",
        "feature_layout": BODY_CORE_V1.name,
        "normalizer_version": "body_core_v1",
        "running_mode": "video",
        "calibration_status": CALIBRATION_STATUS_UNVALIDATED,
        "calibration_note": BODY_CORE_CALIBRATION_NOTE,
        "baseline": float(BODY_CORE_V1_CALIBRATED_BASELINE),
        "baseline_calibrated": True,
        "score_authorized": False,
        "body_core_valid_frame_ratio": _safe_ratio(valid_frames, feats.shape[0]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=feats[start : end + 1], meta=np.array(meta, dtype=object))
    return path


def _profile_body_core_from_raw(
    *,
    label: str,
    template_path: Path,
    landmarks: np.ndarray,
    stages: StageSet,
) -> dict[str, Any]:
    with stages.time("template_load_body_core"):
        tpl = np.load(template_path, allow_pickle=True)
        query = np.asarray(tpl["features"], dtype=np.float32)
        meta = dict(tpl["meta"].item() or {})
    with stages.time("valid_mask_derivation"):
        valid_mask = derive_valid_mask(landmarks, DEFAULT_VALID_CONF_THR)
    seq, valid_frames = _body_core_from_raw(landmarks, valid_mask, stages)
    baseline = float(meta.get("baseline", BODY_CORE_V1_CALIBRATED_BASELINE))
    with stages.time("subsequence_dtw_body_core"):
        cost, start, end = subsequence_dtw(query, seq)
    avg = float(cost / max(1, int(query.shape[0])))
    return {
        "profile_id": label,
        "target": "match_body_core_template",
        "mode": "staged",
        "frames": int(seq.shape[0]),
        "query_frames": int(query.shape[0]),
        "valid_frame_ratio": _safe_ratio(valid_frames, seq.shape[0]),
        "start_frame": int(start),
        "end_frame": int(end),
        "avg_cost": float(avg),
        "score": _score(avg, baseline),
        "calibration_status": CALIBRATION_STATUS_UNVALIDATED,
        "score_authorized": False,
        "stages": stages.records(),
    }


def _load_fixture_raw(name: str) -> np.ndarray:
    path = _REPO_ROOT / "tests" / "fixtures" / "pose33_v3" / name
    d = np.load(path, allow_pickle=True)
    return np.asarray(d["landmarks"], dtype=np.float32)


def run_fixture_profile(out_dir: Path) -> dict[str, Any]:
    fixture_dir = _REPO_ROOT / "tests" / "fixtures" / "pose33_v3"
    front_tpl = fixture_dir / "front_template.npz"
    side_tpl = fixture_dir / "side_template.npz"
    student_raw = _load_fixture_raw("student_raw.npz")
    front_raw = _load_fixture_raw("front_src_raw.npz")
    fps = 30.0
    records: list[dict[str, Any]] = []

    prod_stages = StageSet()
    try:
        from tests import golden_harness as H

        H.clear_registry()
        H.register_video("golden://front_src.mp4", front_raw, fps=fps)
        H.register_video("golden://side_src.mp4", _load_fixture_raw("side_src_raw.npz"), fps=fps)
        H.register_video("golden://student.mp4", student_raw, fps=fps)
        with H.replay_context(), prod_stages.time("production_compare_video_to_dual_templates"):
            prod_res = compare_video_to_dual_templates(
                front_tpl,
                side_tpl,
                "golden://student.mp4",
                pose_variant="full",
                enable_rules=False,
                enable_error_analysis=False,
            )
        records.append(
            {
                "profile_id": "fixture_pose33_production",
                "target": "compare_video_to_dual_templates",
                "mode": "production_call",
                "frames": int(student_raw.shape[0]),
                "combined_percent": int(prod_res.combined_percent),
                "combined_score": float(prod_res.combined_score),
                "front_score": float(prod_res.front_score),
                "side_score": float(prod_res.side_score),
                "front_segment": None if prod_res.front_segment is None else [int(prod_res.front_segment[0]), int(prod_res.front_segment[1])],
                "side_segment": None if prod_res.side_segment is None else [int(prod_res.side_segment[0]), int(prod_res.side_segment[1])],
                "front_matches": int(len(prod_res.front_matches)),
                "side_matches": int(len(prod_res.side_matches)),
                "stages": prod_stages.records(),
            }
        )
    finally:
        try:
            H.clear_registry()  # type: ignore[name-defined]
        except Exception:
            pass

    pose_stages = StageSet()
    normalizer, normalizer_version = _normalizer_for_dual_templates(front_tpl, side_tpl)
    seq, views = _normalize_pose33_from_raw(
        student_raw,
        fps=fps,
        stages=pose_stages,
        normalizer=normalizer,
    )
    records.append(
        _profile_multi_view(
            label="fixture_pose33_staged",
            front_template=front_tpl,
            side_template=side_tpl,
            seq=seq,
            view_scores=views,
            fps=fps,
            stages=pose_stages,
            normalizer_version=normalizer_version,
        )
    )

    body_tpl = _write_fixture_body_core_template(out_dir / "fixture_body_core_template.npz", front_raw, fps=fps)
    body_stages = StageSet()
    records.append(
        _profile_body_core_from_raw(
            label="fixture_body_core_staged",
            template_path=body_tpl,
            landmarks=student_raw,
            stages=body_stages,
        )
    )

    body_prod_stages = StageSet()
    try:
        from tests import golden_harness as H

        H.clear_registry()
        H.register_video("golden://student.mp4", student_raw, fps=fps)
        with H.replay_context(), body_prod_stages.time("production_match_body_core_template"):
            body_res = match_body_core_template(body_tpl, "golden://student.mp4", backend="mediapipe")
        records.append(
            {
                "profile_id": "fixture_body_core_production",
                "target": "match_body_core_template",
                "mode": "production_call",
                "frames": int(student_raw.shape[0]),
                "score": None if body_res.score is None else float(body_res.score),
                "avg_cost": float(body_res.avg_cost),
                "start_frame": int(body_res.start_frame),
                "end_frame": int(body_res.end_frame),
                "stages": body_prod_stages.records(),
            }
        )
    finally:
        try:
            H.clear_registry()  # type: ignore[name-defined]
        except Exception:
            pass

    return {
        "profile_kind": "fixture_smoke",
        "records": records,
    }


def run_video_profile(
    *,
    front_template: Path,
    side_template: Path,
    video_path: Path,
    pose_variant: str,
    limit_frames: int | None,
) -> dict[str, Any]:
    normalizer, normalizer_version = _normalizer_for_dual_templates(front_template, side_template)
    stages = StageSet()
    seq, views, fps = _extract_pose33_from_video(
        video_path,
        pose_variant=pose_variant,
        limit_frames=limit_frames,
        stages=stages,
        normalizer=normalizer,
    )
    record = _profile_multi_view(
        label="local_video_pose33_staged",
        front_template=front_template,
        side_template=side_template,
        seq=seq,
        view_scores=views,
        fps=fps,
        stages=stages,
        normalizer_version=normalizer_version,
    )
    record["video"] = str(video_path)
    record["limit_frames"] = None if limit_frames is None else int(limit_frames)
    return {
        "profile_kind": "local_video",
        "records": [record],
    }


def _stage_seconds(record: dict[str, Any], names: set[str]) -> float:
    total = 0.0
    for stage in record.get("stages", []):
        if str(stage.get("stage")) in names:
            total += float(stage.get("seconds") or 0.0)
    return total


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    sequence_names = {
        "video_io_read",
        "model_inference_mediapipe_pose",
        "feature_normalization_pose33",
        "feature_normalization_body_core",
        "view_score",
        "valid_mask_derivation",
    }
    dtw_names = {
        "front_multi_subsequence_dtw",
        "side_multi_subsequence_dtw",
        "front_fallback_subsequence_dtw",
        "side_fallback_subsequence_dtw",
        "subsequence_dtw_body_core",
    }
    out: dict[str, Any] = {
        "records": len(records),
        "targets": sorted({str(r.get("target")) for r in records}),
    }
    total_stage_sec = 0.0
    sequence_sec = 0.0
    dtw_sec = 0.0
    model_sec = 0.0
    video_io_sec = 0.0
    for rec in records:
        total = sum(float(s.get("seconds") or 0.0) for s in rec.get("stages", []))
        total_stage_sec += total
        sequence_sec += _stage_seconds(rec, sequence_names)
        dtw_sec += _stage_seconds(rec, dtw_names)
        model_sec += _stage_seconds(rec, {"model_inference_mediapipe_pose"})
        video_io_sec += _stage_seconds(rec, {"video_io_read"})
    out.update(
        {
            "total_stage_sec": round(total_stage_sec, 6),
            "sequence_sec": round(sequence_sec, 6),
            "sequence_percent": round(100.0 * sequence_sec / total_stage_sec, 2) if total_stage_sec > 0 else 0.0,
            "dtw_sec": round(dtw_sec, 6),
            "dtw_percent": round(100.0 * dtw_sec / total_stage_sec, 2) if total_stage_sec > 0 else 0.0,
            "model_inference_sec": round(model_sec, 6),
            "video_io_sec": round(video_io_sec, 6),
        }
    )
    return out


def decision_table(summary: dict[str, Any]) -> list[dict[str, str]]:
    dtw_percent = float(summary.get("dtw_percent") or 0.0)
    sequence_percent = float(summary.get("sequence_percent") or 0.0)
    rows: list[dict[str, str]] = []
    if dtw_percent >= 50.0:
        fastdtw_decision = "candidate_separate_issue"
        fastdtw_reason = "DTW dominates the observed profile, but approximate alignment can change scores."
    else:
        fastdtw_decision = "defer"
        fastdtw_reason = "DTW is not the dominant bottleneck in this profile; no evidence to change exact DTW."
    rows.append(
        {
            "option": "FastDTW",
            "decision": fastdtw_decision,
            "reason": fastdtw_reason,
            "required_guard": "Open a separate implementation issue; require pose33_v3 golden and body_core fixture score equality.",
        }
    )
    rows.append(
        {
            "option": "Numba exact DTW",
            "decision": "candidate_only_if_dtw_dominates" if dtw_percent >= 35.0 else "defer",
            "reason": "Exact acceleration is safer than approximation, but adds packaging/runtime complexity.",
            "required_guard": "Separate issue with optional dependency strategy and exact score/path regression tests.",
        }
    )
    rows.append(
        {
            "option": "DTW window constraint",
            "decision": "defer",
            "reason": "Windowing can alter subsequence start/end and user-visible scores; current PR must not change thresholds or baseline.",
            "required_guard": "Only revisit with per-fixture score and interval equality, or explicit product decision to change scoring.",
        }
    )
    rows.append(
        {
            "option": "Raw/feature cache",
            "decision": "candidate_separate_issue" if sequence_percent >= 40.0 else "low_priority",
            "reason": "Caching preserves scoring semantics and targets sequence extraction cost, which is high in real-video runs." if sequence_percent >= 40.0 else "Sequence extraction is not dominant in the current fixture profile.",
            "required_guard": "Cache key must include backend, model, pose_variant, feature_layout, normalizer, valid_conf_thr, and source file metadata.",
        }
    )
    rows.append(
        {
            "option": "Template query cache",
            "decision": "low_priority",
            "reason": "Template load/query selection is small and already uses compact npz templates.",
            "required_guard": "Can be folded into raw/feature cache if future batch profiles prove repeated query selection cost.",
        }
    )
    return rows


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    cols = [
        "profile_id",
        "target",
        "mode",
        "stage",
        "seconds",
        "percent",
        "count",
        "frames",
        "fps",
        "combined_percent",
        "score",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            base = {
                "profile_id": rec.get("profile_id"),
                "target": rec.get("target"),
                "mode": rec.get("mode"),
                "frames": rec.get("frames"),
                "fps": rec.get("fps"),
                "combined_percent": rec.get("combined_percent"),
                "score": rec.get("score"),
            }
            for stage in rec.get("stages", []):
                row = dict(base)
                row.update(stage)
                writer.writerow(row)


def write_profile(out_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "offline_matching_profile.json"
    csv_path = out_dir / "offline_matching_profile.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, payload["records"])
    return json_path, csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S5 offline DTW / sequence extraction profiler (Issue #44)")
    parser.add_argument("--fixture-smoke", action="store_true", help="Run deterministic fixture profile without models/videos")
    parser.add_argument("--front-template", default=None)
    parser.add_argument("--side-template", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--pose-variant", default="full", choices=["lite", "full", "heavy"])
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--out", default="outputs/offline_matching_profile_issue44")
    args = parser.parse_args(argv)

    out_dir = (_REPO_ROOT / args.out).resolve()
    runs: list[dict[str, Any]] = []
    if args.fixture_smoke or not (args.front_template and args.side_template and args.video):
        runs.append(run_fixture_profile(out_dir))
    if args.front_template and args.side_template and args.video:
        runs.append(
            run_video_profile(
                front_template=Path(args.front_template).resolve(),
                side_template=Path(args.side_template).resolve(),
                video_path=Path(args.video).resolve(),
                pose_variant=str(args.pose_variant),
                limit_frames=args.limit_frames,
            )
        )

    records: list[dict[str, Any]] = []
    for run in runs:
        records.extend(run["records"])
    summary = summarize(records)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": _profile_env(),
        "runs": [{"profile_kind": r["profile_kind"], "records": len(r["records"])} for r in runs],
        "summary": summary,
        "decision_table": decision_table(summary),
        "records": records,
    }
    json_path, csv_path = write_profile(out_dir, payload)
    print(f"[offline-profile] written: {json_path}")
    print(f"[offline-profile] written: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
