from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from core import action_compare as ac
from core.rule_scoring import extract_pose_raw
from core.paths import outputs_dir
from batch.backend_options import (
    BACKEND_YOLO,
    FEATURE_LAYOUT_BODY_CORE,
    add_backend_layout_args,
    csv_meta_fields,
    is_default_pose33_path,
    meta_for_backend,
    normalize_backend_layout,
)


_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi"}
_FRONT_KEYS = ("正面", "front", "face", "正", "前")
_SIDE_KEYS = ("侧面", "side", "ce", "侧")
_VIEW_SEPARATORS = " _-.()（）[]【】"


def _matches_view_key(name: str, key: str) -> re.Match[str] | None:
    stem = Path(name).stem
    if any(ord(ch) > 127 for ch in key):
        return re.search(re.escape(key), stem, flags=re.IGNORECASE)
    return re.search(rf"(?<![0-9a-zA-Z]){re.escape(key)}(?![0-9a-zA-Z])", stem, flags=re.IGNORECASE)


def _find_view_match(name: str) -> tuple[str, str, tuple[int, int]] | None:
    for view, keys in (("front", _FRONT_KEYS), ("side", _SIDE_KEYS)):
        for key in keys:
            match = _matches_view_key(name, key)
            if match:
                return view, key, match.span()
    return None


def _classify_view(name: str) -> str | None:
    match = _find_view_match(name)
    return None if match is None else match[0]


def _student_id_from_view_name(name: str) -> str:
    stem = Path(name).stem
    match = _find_view_match(name)
    if match is None:
        return stem.strip(_VIEW_SEPARATORS) or stem
    _view, _key, (start, end) = match
    cleaned = f"{stem[:start]}{stem[end:]}".strip(_VIEW_SEPARATORS)
    cleaned = re.sub(r"[\s_.-]+", "_", cleaned).strip("_")
    return cleaned or stem


def _pair_students(videos) -> list[tuple[str, Path, Path]]:
    buckets: dict[tuple[Path, str], dict[str, list[Path]]] = {}
    skipped_unknown: list[str] = []
    for video in sorted((Path(v) for v in videos), key=lambda p: p.name.lower()):
        view = _classify_view(video.name)
        if view is None:
            skipped_unknown.append(video.name)
            continue
        student_id = _student_id_from_view_name(video.name)
        buckets.setdefault((video.parent, student_id), {"front": [], "side": []})[view].append(video)

    if skipped_unknown:
        print(f"Skipping unclassified student videos: {skipped_unknown}")

    pairs: list[tuple[str, Path, Path]] = []
    skipped_pairs: list[str] = []
    for parent, student_id in sorted(buckets):
        front = buckets[(parent, student_id)]["front"]
        side = buckets[(parent, student_id)]["side"]
        if len(front) == 1 and len(side) == 1:
            pairs.append((student_id, front[0], side[0]))
            continue
        skipped_pairs.append(
            f"{parent / student_id}: front={[p.name for p in front]}, side={[p.name for p in side]}"
        )

    if skipped_pairs:
        print(f"Skipping unpaired student videos: {skipped_pairs}")
    return pairs


def _find_standard_video(std_dir: Path, *, kind: str) -> Path:
    """
    Find a front/side standard video in `std_dir` using filename heuristics.
    `kind` is "front" or "side".
    """
    if kind not in ("front", "side"):
        raise ValueError("kind must be 'front' or 'side'")

    vids = sorted([p for p in std_dir.iterdir() if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES])
    if not vids:
        raise FileNotFoundError(f"No videos found in: {std_dir}")

    keys = _FRONT_KEYS if kind == "front" else _SIDE_KEYS
    for k in keys:
        for p in vids:
            if k.lower() in p.name.lower():
                return p

    raise FileNotFoundError(
        f"Cannot auto-detect {kind} standard video in {std_dir}. "
        f"Found: {[p.name for p in vids]}. Use --front/--side to specify explicitly."
    )


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description="Dual-template compare (front+side) + raw skeleton export.")
    ap.add_argument("--standard_dir", type=str, default=None, help="Folder containing standard front/side videos")
    ap.add_argument("--front", type=str, default=None, help="Explicit path to standard FRONT video")
    ap.add_argument("--side", type=str, default=None, help="Explicit path to standard SIDE video")
    ap.add_argument("--student_dir", type=str, required=True, help="Folder containing student videos (*.mp4)")
    ap.add_argument("--pose", default="full", choices=["lite", "full", "heavy"], help="Pose model variant")
    ap.add_argument("--out_dir", type=str, default=None, help="Output folder (default: ./outputs/dual_compare_<ts>)")
    ap.add_argument("--export_raw", action="store_true", help="Export raw pose landmarks (.npz) for std + student segments")
    ap.add_argument("--rules", action="store_true", help="Enable rule-based scoring")
    ap.add_argument("--action", default="both", choices=["stance", "punch", "both"], help="Rule action scope")
    ap.add_argument("--error-analysis", action="store_true", help="Export error analysis CSVs (rules + joints)")
    ap.add_argument("--workers", type=int, default=1, help="Number of videos to process concurrently (default: 1)")
    ap.add_argument("--paired", action="store_true", help="Pair student front/side files by filename and compare two streams")
    add_backend_layout_args(ap)
    args = ap.parse_args()
    backend, feature_layout = normalize_backend_layout(args.backend, args.feature_layout)
    enable_error_analysis = bool(getattr(args, "error_analysis", False))

    std_dir = Path(args.standard_dir) if args.standard_dir else None
    student_dir = Path(args.student_dir)
    if not student_dir.exists():
        raise FileNotFoundError(f"student_dir not found: {student_dir}")

    if args.front:
        front_video = Path(args.front)
    else:
        if std_dir is None:
            raise ValueError("Need --standard_dir or --front")
        front_video = _find_standard_video(std_dir, kind="front")

    if args.side:
        side_video = Path(args.side)
    else:
        if std_dir is None:
            raise ValueError("Need --standard_dir or --side")
        side_video = _find_standard_video(std_dir, kind="side")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (outputs_dir() / f"dual_compare_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "templates").mkdir(parents=True, exist_ok=True)
    (out_dir / "skeleton").mkdir(parents=True, exist_ok=True)

    if bool(getattr(args, "paired", False)) and not is_default_pose33_path(backend, feature_layout):
        raise ValueError("--paired 仅支持默认 MediaPipe pose33_v3 路径；body_core_v1 双文件批处理不在本期范围内。")

    if not is_default_pose33_path(backend, feature_layout):
        _run_body_core_batch(
            args=args,
            backend=backend,
            front_video=front_video,
            side_video=side_video,
            student_dir=student_dir,
            out_dir=out_dir,
        )
        return

    # 1) Create templates from the two standard videos.
    front_tpl = out_dir / "templates" / f"standard_front_{args.pose}.npz"
    side_tpl = out_dir / "templates" / f"standard_side_{args.pose}.npz"
    front_tpl = ac.create_template_from_video(front_video, pose_variant=args.pose, out_path=front_tpl)
    side_tpl = ac.create_template_from_video(side_video, pose_variant=args.pose, out_path=side_tpl)

    # 2) Batch compare all student videos.
    student_videos = sorted([p for p in student_dir.iterdir() if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES])
    if not student_videos:
        raise FileNotFoundError(f"No student videos found in: {student_dir}")

    paired_mode = bool(getattr(args, "paired", False))
    if paired_mode:
        paired_students = _pair_students(student_videos)
        if not paired_students:
            raise FileNotFoundError(f"No paired front/side student videos found in: {student_dir}")
        work_items = paired_students
    else:
        work_items = student_videos

    export_raw_enabled = bool(args.export_raw) and not paired_mode
    if paired_mode and bool(args.export_raw):
        print("--export_raw is not supported in --paired mode; skipping raw skeleton export.")

    rows: list[dict[str, Any]] = []
    error_rules_rows: list[dict[str, Any]] = []
    error_joints_rows: list[dict[str, Any]] = []
    jsonl_path = out_dir / "compare_results.jsonl"
    video_workers = max(1, int(args.workers))

    def _process_pose33_video(idx: int, item) -> tuple[int, Any, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
        if paired_mode:
            student_id, front_student_video, side_student_video = item
            res = ac.compare_dual_streams(
                front_tpl,
                side_tpl,
                front_student_video,
                side_student_video,
                workers=1,
                enable_rules=bool(args.rules),
                action_scope=str(args.action),
                enable_error_analysis=enable_error_analysis,
            )
            video_label = str(student_id)
            error_video_label = str(student_id)
            payload_extra = {
                "student_id": str(student_id),
                "front_video_path": str(front_student_video),
                "side_video_path": str(side_student_video),
            }
        else:
            v = item
            res = ac.compare_video_to_dual_templates(
                front_tpl,
                side_tpl,
                v,
                workers=1,
                enable_rules=bool(args.rules),
                action_scope=str(args.action),
                enable_error_analysis=enable_error_analysis,
            )
            video_label = str(v)
            error_video_label = str(v.name)
            payload_extra = None

        per_video_rule_rows: list[dict[str, Any]] = []
        per_video_joint_rows: list[dict[str, Any]] = []
        if enable_error_analysis and bool(args.rules):
            for view, violations in (
                ("front", res.front_rule_violations),
                ("side", res.side_rule_violations),
            ):
                if not violations:
                    continue
                for rv in violations:
                    per_video_rule_rows.append(
                        {
                            "video": error_video_label,
                            "view": str(view),
                            "rule_id": str(rv.rule_id),
                            "rule_name": str(rv.name),
                            "violation_ratio": float(rv.violation_ratio),
                            "penalty": int(rv.penalty),
                            "valid_frames": int(rv.valid_frames),
                            "total_frames": int(rv.total_frames),
                            # YOLO 迁移 S4 / Issue #11：结构化三态写入报告，
                            # 未评估原因清晰，不把不可评估当合格/不合格。
                            "state": str(getattr(rv, "state", "") or ""),
                            "skip_reason": str(getattr(rv, "skip_reason", "") or ""),
                            "missing_landmarks": ",".join(getattr(rv, "missing_landmarks", ()) or ()),
                        }
                    )
        if enable_error_analysis:
            for view, joints in (
                ("front", res.front_joint_errors),
                ("side", res.side_joint_errors),
            ):
                if not joints:
                    continue
                for je in joints:
                    per_video_joint_rows.append(
                        {
                            "video": error_video_label,
                            "view": str(view),
                            "joint": str(je.joint),
                            "mean_dist": "" if je.mean_dist is None else float(je.mean_dist),
                            "p90_dist": "" if je.p90_dist is None else float(je.p90_dist),
                            "max_dist": "" if je.max_dist is None else float(je.max_dist),
                            "valid_frames": int(je.valid_frames),
                        }
                    )
        row = {
            "video": video_label,
            "front_score": float(res.front_score),
            "side_score": float(res.side_score),
            "combined_percent": int(res.combined_percent),
            "front_rule_score": "" if res.front_rule_score is None else int(res.front_rule_score),
            "side_rule_score": "" if res.side_rule_score is None else int(res.side_rule_score),
            "front_rule_deduction": "" if res.front_rule_deduction is None else int(res.front_rule_deduction),
            "side_rule_deduction": "" if res.side_rule_deduction is None else int(res.side_rule_deduction),
            "front_segment": "" if res.front_segment is None else f"{res.front_segment[0]}..{res.front_segment[1]}",
            "side_segment": "" if res.side_segment is None else f"{res.side_segment[0]}..{res.side_segment[1]}",
            "front_matches": int(len(res.front_matches)),
            "side_matches": int(len(res.side_matches)),
            "pose_variant": str(res.pose_variant),
            "fps": float(res.fps),
        }
        return idx, res, row, per_video_rule_rows, per_video_joint_rows, payload_extra

    compare_results: list[tuple[int, Any, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]] = []
    if video_workers <= 1:
        for i, item in enumerate(work_items):
            compare_results.append(_process_pose33_video(i, item))
    else:
        with ThreadPoolExecutor(max_workers=video_workers) as ex:
            futures = [ex.submit(_process_pose33_video, i, item) for i, item in enumerate(work_items)]
            for fut in as_completed(futures):
                compare_results.append(fut.result())
    compare_results.sort(key=lambda item: item[0])

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for _idx, res, row, per_video_rule_rows, per_video_joint_rows, payload_extra in compare_results:
            payload = _jsonable(res)
            if payload_extra:
                payload.update(payload_extra)
            jf.write(json.dumps(payload, ensure_ascii=False) + "\n")
            rows.append(row)
            error_rules_rows.extend(per_video_rule_rows)
            error_joints_rows.extend(per_video_joint_rows)

    # Export standard raw (front/side) if requested.
    if export_raw_enabled:
        std_front_raw, meta_f = extract_pose_raw(front_video, pose_variant=args.pose)
        std_side_raw, meta_s = extract_pose_raw(side_video, pose_variant=args.pose)
        np.savez_compressed(out_dir / "skeleton" / f"standard_front_raw_{args.pose}.npz", landmarks=std_front_raw, meta=np.array(meta_f, dtype=object))
        np.savez_compressed(out_dir / "skeleton" / f"standard_side_raw_{args.pose}.npz", landmarks=std_side_raw, meta=np.array(meta_s, dtype=object))

        # Export student segments raw.
        # Re-read the jsonl for segments to avoid keeping all results in memory as dataclasses.
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            v = Path(d["video_path"])
            stem = v.stem

            fseg = d.get("front_segment")
            sseg = d.get("side_segment")
            if fseg:
                s, e = int(fseg[0]), int(fseg[1])
                arr, meta = extract_pose_raw(v, pose_variant=args.pose, start_frame=s, end_frame=e)
                meta["segment_kind"] = "front"
                np.savez_compressed(out_dir / "skeleton" / f"{stem}_front_raw_{args.pose}.npz", landmarks=arr, meta=np.array(meta, dtype=object))
            if sseg:
                s, e = int(sseg[0]), int(sseg[1])
                arr, meta = extract_pose_raw(v, pose_variant=args.pose, start_frame=s, end_frame=e)
                meta["segment_kind"] = "side"
                np.savez_compressed(out_dir / "skeleton" / f"{stem}_side_raw_{args.pose}.npz", landmarks=arr, meta=np.array(meta, dtype=object))

    # 3) CSV summary (UTF-8 BOM for Excel).
    csv_path = out_dir / "compare_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if enable_error_analysis and bool(args.rules):
        error_rules_path = out_dir / "error_rules.csv"
        with error_rules_path.open("w", encoding="utf-8-sig", newline="") as f:
            fieldnames = ["video", "view", "rule_id", "rule_name", "violation_ratio", "penalty", "valid_frames", "total_frames", "state", "skip_reason", "missing_landmarks"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(error_rules_rows)

    if enable_error_analysis:
        error_joints_path = out_dir / "error_joints.csv"
        with error_joints_path.open("w", encoding="utf-8-sig", newline="") as f:
            fieldnames = ["video", "view", "joint", "mean_dist", "p90_dist", "max_dist", "valid_frames"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(error_joints_rows)

    print(f"Saved templates: {front_tpl} , {side_tpl}")
    print(f"Saved results:   {csv_path}")
    print(f"Saved jsonl:     {jsonl_path}")
    if export_raw_enabled:
        print(f"Saved skeleton:  {out_dir / 'skeleton'}")
    if enable_error_analysis and bool(args.rules):
        print(f"Saved error rules: {out_dir / 'error_rules.csv'}")
    if enable_error_analysis:
        print(f"Saved error joints: {out_dir / 'error_joints.csv'}")


def _load_npz_meta(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return dict(data["meta"].item() or {})


def _body_core_result_dict(res: Any, *, meta: dict[str, Any]) -> dict[str, Any]:
    score_authorized = bool(meta.get("score_authorized", False))
    return {
        "template_path": str(res.template_path),
        "video_path": str(res.video_path),
        "backend": str(res.backend),
        "raw_layout": str(meta.get("raw_layout", "")),
        "feature_layout": str(res.feature_layout),
        "capability": str(meta.get("capability", "")),
        "start_frame": int(res.start_frame),
        "end_frame": int(res.end_frame),
        "cost": float(res.cost),
        "avg_cost": float(res.avg_cost),
        "debug_score": None if res.score is None else float(res.score),
        "calibration_status": str(res.calibration_status),
        "score_authorized": score_authorized,
        "display_scope": str(meta.get("display_scope", "")),
        "valid_frame_ratio": res.valid_frame_ratio,
        "review_required": bool(res.review_required),
        "multi_person_detected": bool(res.multi_person_detected),
        "max_persons": int(res.max_persons),
        "multi_person_frames": int(res.multi_person_frames),
        "multi_person_gate_source": str(res.multi_person_gate_source or ""),
    }


def _body_core_row(v: Path, front_res: Any, side_res: Any, *, meta: dict[str, Any]) -> dict[str, Any]:
    review_required = bool(front_res.review_required or side_res.review_required or meta.get("review_required", False))
    review_status = "需人工复核" if review_required else "调试/标定"
    row: dict[str, Any] = {
        "video": str(v),
        **csv_meta_fields({**meta, "review_required": review_required}),
        "review_status": review_status,
        "front_debug_score": "" if front_res.score is None else float(front_res.score),
        "side_debug_score": "" if side_res.score is None else float(side_res.score),
        "front_avg_cost": float(front_res.avg_cost),
        "side_avg_cost": float(side_res.avg_cost),
        "front_valid_frame_ratio": "" if front_res.valid_frame_ratio is None else float(front_res.valid_frame_ratio),
        "side_valid_frame_ratio": "" if side_res.valid_frame_ratio is None else float(side_res.valid_frame_ratio),
        "front_review_required": bool(front_res.review_required),
        "side_review_required": bool(side_res.review_required),
        "max_persons": max(int(front_res.max_persons), int(side_res.max_persons)),
        "multi_person_frames": int(front_res.multi_person_frames) + int(side_res.multi_person_frames),
    }
    return row


def _run_body_core_batch(*, args, backend: str, front_video: Path, side_video: Path, student_dir: Path, out_dir: Path) -> None:
    """body_core_v1 debug/calibration batch path.

    Scores are written only as ``*_debug_score`` with ``score_authorized=False``.
    They never populate the pose33 outward-facing score columns.
    """
    from core.body_core_compare import create_body_core_template, extract_body_core_features, match_body_core_template

    if backend == BACKEND_YOLO and args.rules:
        print("YOLO body_core_v1 不支持规则评分；本批次仅输出调试/标定分数。")

    yolo_model_name = None
    if backend == BACKEND_YOLO:
        from core.body_core_compare import DEFAULT_YOLO26L_MODEL_NAME
        from core.paths import models_dir

        yolo_model_name = DEFAULT_YOLO26L_MODEL_NAME
        yolo_model = models_dir() / DEFAULT_YOLO26L_MODEL_NAME
    else:
        yolo_model = None

    front_tpl = create_body_core_template(
        front_video,
        backend=backend,
        pose_variant=args.pose,
        out_path=out_dir / "templates" / f"standard_front_{backend}_{FEATURE_LAYOUT_BODY_CORE}.npz",
        yolo_model=yolo_model,
    )
    side_tpl = create_body_core_template(
        side_video,
        backend=backend,
        pose_variant=args.pose,
        out_path=out_dir / "templates" / f"standard_side_{backend}_{FEATURE_LAYOUT_BODY_CORE}.npz",
        yolo_model=yolo_model,
    )
    template_meta = _load_npz_meta(front_tpl)
    if yolo_model_name:
        template_meta.setdefault("model_name", yolo_model_name)
    batch_meta = meta_for_backend(backend, template_meta, pose_variant=args.pose)

    student_videos = sorted([p for p in student_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi"}])
    if not student_videos:
        raise FileNotFoundError(f"No student videos found in: {student_dir}")

    rows: list[dict[str, Any]] = []
    jsonl_path = out_dir / "compare_results.jsonl"
    video_workers = max(1, int(getattr(args, "workers", 1)))

    def _process_body_core_video(idx: int, v: Path) -> tuple[int, dict[str, Any], dict[str, Any]]:
        precomputed = extract_body_core_features(
            v,
            backend=backend,
            pose_variant=args.pose,
            yolo_model=yolo_model,
        )
        front_res = match_body_core_template(
            front_tpl,
            v,
            backend=backend,
            pose_variant=args.pose,
            reject_multi_person=False,
            yolo_model=yolo_model,
            precomputed_features=precomputed,
        )
        side_res = match_body_core_template(
            side_tpl,
            v,
            backend=backend,
            pose_variant=args.pose,
            reject_multi_person=False,
            yolo_model=yolo_model,
            precomputed_features=precomputed,
        )
        row = _body_core_row(v, front_res, side_res, meta=batch_meta)
        payload = {
            "video_path": str(v),
            "backend": backend,
            "feature_layout": FEATURE_LAYOUT_BODY_CORE,
            "meta": {**batch_meta, "review_required": bool(row["review_required"])},
            "front_debug_match": _body_core_result_dict(front_res, meta=batch_meta),
            "side_debug_match": _body_core_result_dict(side_res, meta=batch_meta),
        }
        return idx, row, payload

    body_core_results: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    if video_workers <= 1:
        for i, v in enumerate(student_videos):
            body_core_results.append(_process_body_core_video(i, v))
    else:
        with ThreadPoolExecutor(max_workers=video_workers) as ex:
            futures = [ex.submit(_process_body_core_video, i, v) for i, v in enumerate(student_videos)]
            for fut in as_completed(futures):
                body_core_results.append(fut.result())
    body_core_results.sort(key=lambda item: item[0])

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for _idx, row, payload in body_core_results:
            rows.append(row)
            jf.write(
                json.dumps(
                    _jsonable(payload),
                    ensure_ascii=False,
                )
                + "\n"
            )

    csv_path = out_dir / "compare_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Saved templates: {front_tpl} , {side_tpl}")
    print(f"Saved debug/calibration results: {csv_path}")
    print(f"Saved jsonl: {jsonl_path}")


if __name__ == "__main__":
    main()
