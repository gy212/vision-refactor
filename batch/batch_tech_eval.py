# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from analysis.tech_eval import evaluate_video_assets, evaluate_video_detail, export_debug_video, to_jsonable
from core.paths import outputs_dir
from batch.backend_options import (
    FEATURE_LAYOUT_BODY_CORE,
    add_backend_layout_args,
    csv_meta_fields,
    is_default_pose33_path,
    meta_for_backend,
    normalize_backend_layout,
)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def _iter_videos(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        out.append(p)
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch evaluate tech indicators for straight punches (重心/回收速度/发力顺序/拳面角度).")
    ap.add_argument("--video_dir", "--student_dir", dest="video_dir", required=True, help="Folder containing videos")
    ap.add_argument("--pose", default="heavy", choices=["lite", "full", "heavy"], help="Pose model variant")
    ap.add_argument("--stance", default="left", choices=["left", "right"], help="Stance (default: left)")
    ap.add_argument("--view", default="auto", choices=["auto", "front", "side"], help="View hint (auto/front/side)")
    ap.add_argument("--out_dir", default=None, help="Output folder (default: outputs/tech_eval_<ts>)")
    ap.add_argument("--full", action="store_true", help="Also export full jsonl with numeric details")
    ap.add_argument("--debug-video", action="store_true", help="Export per-video debug skeleton mp4 (overlay skeleton + statuses)")
    ap.add_argument("--workers", type=int, default=10, help="Number of worker threads (default: 10)")
    add_backend_layout_args(ap)
    args = ap.parse_args()
    backend, feature_layout = normalize_backend_layout(args.backend, args.feature_layout)

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"video_dir not found: {video_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (outputs_dir() / f"tech_eval_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = _iter_videos(video_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found in: {video_dir}")

    if not is_default_pose33_path(backend, feature_layout):
        _run_unvalidated_body_core_batch(
            videos=videos,
            out_dir=out_dir,
            args=args,
            backend=backend,
        )
        return

    rows: list[dict[str, Any]] = []
    full_jsonl = out_dir / "tech_report_full.jsonl"
    full_f = full_jsonl.open("w", encoding="utf-8") if args.full else None
    debug_dir = out_dir / "debug_videos"
    if args.debug_video:
        debug_dir.mkdir(parents=True, exist_ok=True)

    def _build_row(v: Path, res: Any) -> dict[str, Any]:
        def _cause(ind: Any) -> str:
            if ind is None or getattr(ind, "detail", None) is None:
                return ""
            return str(ind.detail.get("primary_cause") or "")

        def _missing(ind: Any) -> str:
            # YOLO 迁移 S4 / Issue #11：把缺失/不可评估关键点如实写入报告，
            # 不把不可评估当合格/不合格。
            if ind is None:
                return ""
            return ",".join(getattr(ind, "missing_landmarks", ()) or ())

        def _backend(ind: Any) -> str:
            if ind is None:
                return ""
            return str(getattr(ind, "backend", "") or "")

        row = {
            "video": str(v.name),
            "view_mode": str(res.view_mode),
            "重心(侧面优先)": str(res.cog_final.status),
            "重心说明": str(res.cog_final.reason),
            "重心原因类型": _cause(res.cog_final),
            "重心缺失关键点": _missing(res.cog_final),
            "重心_侧面": str(res.cog_side.status),
            "重心_侧面说明": str(res.cog_side.reason),
            "重心_侧面原因类型": _cause(res.cog_side),
            "重心_侧面缺失关键点": _missing(res.cog_side),
            "重心_正面": str(res.cog_front.status),
            "重心_正面说明": str(res.cog_front.reason),
            "重心_正面原因类型": _cause(res.cog_front),
            "重心_正面缺失关键点": _missing(res.cog_front),
            "回收速度": str(res.retract_speed.status),
            "回收速度说明": str(res.retract_speed.reason),
            "回收速度原因类型": _cause(res.retract_speed),
            "回收速度缺失关键点": _missing(res.retract_speed),
            "发力顺序": str(res.force_sequence.status),
            "发力顺序说明": str(res.force_sequence.reason),
            "发力顺序原因类型": _cause(res.force_sequence),
            "发力顺序缺失关键点": _missing(res.force_sequence),
            "拳面角度": str(res.wrist_angle.status),
            "拳面角度说明": str(res.wrist_angle.reason),
            "拳面角度原因类型": _cause(res.wrist_angle),
            "拳面角度缺失关键点": _missing(res.wrist_angle),
            "backend": _backend(res.cog_final),
        }
        # 方案3：分段质心评估结果（CoM）
        if res.cog_com is not None:
            row["重心_CoM(方案3)"] = str(res.cog_com.status)
            row["重心_CoM说明"] = str(res.cog_com.reason)
            row["重心_CoM原因类型"] = _cause(res.cog_com)
            row["重心_CoM缺失关键点"] = _missing(res.cog_com)
        else:
            row["重心_CoM(方案3)"] = "未评估"
            row["重心_CoM说明"] = ""
            row["重心_CoM原因类型"] = ""
            row["重心_CoM缺失关键点"] = ""
        return row

    def _process_video(idx: int, v: Path) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
        res_detail = None
        lm = None
        vs = None
        meta = None
        if args.full or args.debug_video:
            res_detail, lm, vs, meta = evaluate_video_assets(v, pose_variant=args.pose, stance=args.stance, view_hint=args.view)
            res = res_detail
        else:
            res = evaluate_video_detail(v, pose_variant=args.pose, stance=args.stance, view_hint=args.view)

        row = _build_row(v, res)
        full_data = None
        if args.full:
            if res_detail is None or lm is None or vs is None or meta is None:
                res_detail, lm, vs, meta = evaluate_video_assets(v, pose_variant=args.pose, stance=args.stance, view_hint=args.view)
            full_data = {
                "video_path": str(v),
                "pose_variant": str(args.pose),
                "fps": float(res_detail.fps),
                "view_mode": str(res_detail.view_mode),
                "front_segment": None if res_detail.front_segment is None else [int(res_detail.front_segment[0]), int(res_detail.front_segment[1])],
                "side_segment": None if res_detail.side_segment is None else [int(res_detail.side_segment[0]), int(res_detail.side_segment[1])],
                "meta": meta,
                "cog_side": to_jsonable(res_detail.cog_side),
                "cog_front": to_jsonable(res_detail.cog_front),
                "cog_final": to_jsonable(res_detail.cog_final),
                "cog_com": to_jsonable(res_detail.cog_com),
                "retract_speed": to_jsonable(res_detail.retract_speed),
                "force_sequence": to_jsonable(res_detail.force_sequence),
                "wrist_angle": to_jsonable(res_detail.wrist_angle),
            }

        if args.debug_video:
            if res_detail is None or lm is None or vs is None or meta is None:
                res_detail, lm, vs, meta = evaluate_video_assets(v, pose_variant=args.pose, stance=args.stance, view_hint=args.view)
            out_mp4 = debug_dir / f"{v.stem}_debug.mp4"
            export_debug_video(
                v,
                out_mp4,
                pose_variant=args.pose,
                stance=args.stance,
                view_hint=args.view,
                res=res_detail,
                landmarks=lm,
                view_scores=vs,
                meta=meta,
            )

        return idx, row, full_data

    try:
        if args.workers <= 1:
            full_lines: list[dict[str, Any]] = []
            for i, v in enumerate(videos):
                _, row, full_data = _process_video(i, v)
                rows.append(row)
                if full_data is not None:
                    full_lines.append(full_data)
        else:
            full_lines = []
            with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
                futures = [ex.submit(_process_video, i, v) for i, v in enumerate(videos)]
                results: list[tuple[int, dict[str, Any], dict[str, Any] | None]] = []
                for fut in as_completed(futures):
                    results.append(fut.result())
            results.sort(key=lambda item: item[0])
            for _idx, row, full_data in results:
                rows.append(row)
                if full_data is not None:
                    full_lines.append(full_data)

        if full_f is not None:
            for d in full_lines:
                full_f.write(json.dumps(d, ensure_ascii=False) + "\n")
    finally:
        if full_f is not None:
            full_f.close()

    csv_path = out_dir / "tech_report.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Saved report: {csv_path}")
    if args.full:
        print(f"Saved full:   {full_jsonl}")


def _unvalidated_reason(meta: dict[str, Any]) -> str:
    if bool(meta.get("review_required", False)):
        return "YOLO body_core_v1 检出多人，需人工复核，不进入对外评分"
    return "body_core_v1 未授权 full tech_eval；仅用于离线调试/标定，不进入对外评分"


def _body_core_unvalidated_row(v: Path, meta: dict[str, Any]) -> dict[str, Any]:
    reason = _unvalidated_reason(meta)
    row = {
        "video": str(v.name),
        "view_mode": "skipped",
        "重心(侧面优先)": "无法判定",
        "重心说明": reason,
        "重心原因类型": "unvalidated_backend",
        "重心缺失关键点": "",
        "重心_侧面": "无法判定",
        "重心_侧面说明": reason,
        "重心_侧面原因类型": "unvalidated_backend",
        "重心_侧面缺失关键点": "",
        "重心_正面": "无法判定",
        "重心_正面说明": reason,
        "重心_正面原因类型": "unvalidated_backend",
        "重心_正面缺失关键点": "",
        "回收速度": "无法判定",
        "回收速度说明": reason,
        "回收速度原因类型": "unvalidated_backend",
        "回收速度缺失关键点": "",
        "发力顺序": "无法判定",
        "发力顺序说明": reason,
        "发力顺序原因类型": "unvalidated_backend",
        "发力顺序缺失关键点": "",
        "拳面角度": "无法判定",
        "拳面角度说明": reason,
        "拳面角度原因类型": "unvalidated_backend",
        "拳面角度缺失关键点": "",
        "backend": str(meta.get("backend", "")),
        "重心_CoM(方案3)": "无法判定",
        "重心_CoM说明": reason,
        "重心_CoM原因类型": "unvalidated_backend",
        "重心_CoM缺失关键点": "",
        **csv_meta_fields(meta),
    }
    row["review_status"] = "需人工复核" if bool(meta.get("review_required", False)) else "未标定跳过"
    return row


def _body_core_meta_for_video(v: Path, *, backend: str, pose_variant: str) -> dict[str, Any]:
    if backend == "yolo":
        from core.yolo_adapter import extract_yolo_landmark_series

        _landmarks, _valid_mask, source_meta = extract_yolo_landmark_series(v)
        return meta_for_backend(backend, source_meta, pose_variant=pose_variant) | {
            key: source_meta[key]
            for key in ("multi_person_detected", "max_persons", "multi_person_frames", "gate_status", "gate_note")
            if key in source_meta
        }

    from core.body_core_compare import extract_body_core_features

    _features, _fps, source_meta = extract_body_core_features(
        v,
        backend=backend,
        pose_variant=pose_variant,
    )
    return meta_for_backend(backend, source_meta, pose_variant=pose_variant)


def _run_unvalidated_body_core_batch(*, videos: list[Path], out_dir: Path, args, backend: str) -> None:
    rows: list[dict[str, Any]] = []
    full_jsonl = out_dir / "tech_report_full.jsonl"
    full_f = full_jsonl.open("w", encoding="utf-8") if args.full else None
    try:
        for v in videos:
            meta = _body_core_meta_for_video(v, backend=backend, pose_variant=args.pose)
            meta["feature_layout"] = FEATURE_LAYOUT_BODY_CORE
            meta["score_authorized"] = False
            row = _body_core_unvalidated_row(v, meta)
            rows.append(row)
            if full_f is not None:
                full_f.write(
                    json.dumps(
                        to_jsonable(
                            {
                                "video_path": str(v),
                                "backend": backend,
                                "feature_layout": FEATURE_LAYOUT_BODY_CORE,
                                "tech_eval_status": "skipped",
                                "skip_reason": "unvalidated_body_core_v1",
                                "score_authorized": False,
                                "meta": meta,
                            }
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if full_f is not None:
            full_f.close()

    csv_path = out_dir / "tech_report.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Saved report: {csv_path}")
    if args.full:
        print(f"Saved full:   {full_jsonl}")


if __name__ == "__main__":
    main()
