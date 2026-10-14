# -*- coding: utf-8 -*-
"""批处理入口：批量计算直拳技术指标（重心、回收速度、发力顺序、拳面角度）并导出。"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from core.evaluation import evaluate_video_assets, evaluate_video_detail, export_debug_video
from core.foundation import to_jsonable
from core.foundation import outputs_dir
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


# (状态列, 说明/原因/缺失列的前缀, 结果字段)：CSV 列顺序即此顺序，CoM 列排在 backend 之后（见 _tech_row）。
_INDICATOR_COLUMNS = (
    ("重心(侧面优先)", "重心", "cog_final"),
    ("重心_侧面", "重心_侧面", "cog_side"),
    ("重心_正面", "重心_正面", "cog_front"),
    ("回收速度", "回收速度", "retract_speed"),
    ("发力顺序", "发力顺序", "force_sequence"),
    ("拳面角度", "拳面角度", "wrist_angle"),
)


def _parse_args() -> argparse.Namespace:
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
    return ap.parse_args()


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


def _indicator_cells(status_col: str, prefix: str, ind: Any) -> dict[str, str]:
    return {
        status_col: str(ind.status),
        f"{prefix}说明": str(ind.reason),
        f"{prefix}原因类型": _cause(ind),
        f"{prefix}缺失关键点": _missing(ind),
    }


def _tech_row(v: Path, res: Any) -> dict[str, Any]:
    """一个视频的 CSV 行：六项指标各四列、backend，再加方案3 分段质心（CoM）四列。"""
    row: dict[str, Any] = {"video": str(v.name), "view_mode": str(res.view_mode)}
    for status_col, prefix, field in _INDICATOR_COLUMNS:
        row.update(_indicator_cells(status_col, prefix, getattr(res, field)))
    row["backend"] = str(getattr(res.cog_final, "backend", "") or "") if res.cog_final is not None else ""
    if res.cog_com is not None:
        row.update(_indicator_cells("重心_CoM(方案3)", "重心_CoM", res.cog_com))
    else:
        row.update({"重心_CoM(方案3)": "未评估", "重心_CoM说明": "", "重心_CoM原因类型": "", "重心_CoM缺失关键点": ""})
    return row


def _full_record(v: Path, pose: str, res: Any, meta: Any) -> dict[str, Any]:
    """``--full`` 的 JSONL 记录：区间、元数据与每项指标的完整数值细节。"""
    def segment(seg):
        return None if seg is None else [int(seg[0]), int(seg[1])]

    return {
        "video_path": str(v),
        "pose_variant": str(pose),
        "fps": float(res.fps),
        "view_mode": str(res.view_mode),
        "front_segment": segment(res.front_segment),
        "side_segment": segment(res.side_segment),
        "meta": meta,
        **{name: to_jsonable(getattr(res, name)) for name in (
            "cog_side", "cog_front", "cog_final", "cog_com", "retract_speed", "force_sequence", "wrist_angle")},
    }


def _tech_video(v: Path, args, debug_dir: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """评估一个视频：返回 CSV 行与（``--full`` 时的）JSONL 记录；``--debug-video`` 时另写骨架调试视频。"""
    kw = dict(pose_variant=args.pose, stance=args.stance, view_hint=args.view)
    if not (args.full or args.debug_video):
        return _tech_row(v, evaluate_video_detail(v, **kw)), None
    res, lm, vs, meta = evaluate_video_assets(v, **kw)
    full_data = _full_record(v, args.pose, res, meta) if args.full else None
    if args.debug_video:
        export_debug_video(v, debug_dir / f"{v.stem}_debug.mp4", **kw, res=res, landmarks=lm, view_scores=vs, meta=meta)
    return _tech_row(v, res), full_data


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _run_pose33_tech_eval(videos: list[Path], out_dir: Path, args) -> None:
    """默认 MediaPipe 路径：逐个（或多线程按输入顺序）评估，写 ``tech_report.csv`` 与可选完整 JSONL。"""
    debug_dir = out_dir / "debug_videos"
    if args.debug_video:
        debug_dir.mkdir(parents=True, exist_ok=True)
    if args.workers <= 1:
        results = [_tech_video(v, args, debug_dir) for v in videos]
    else:
        with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
            results = list(ex.map(lambda v: _tech_video(v, args, debug_dir), videos))

    full_jsonl = out_dir / "tech_report_full.jsonl"
    if args.full:
        with full_jsonl.open("w", encoding="utf-8") as full_f:
            for _row, full_data in results:
                if full_data is not None:
                    full_f.write(json.dumps(full_data, ensure_ascii=False) + "\n")
    csv_path = out_dir / "tech_report.csv"
    _write_rows_csv(csv_path, [row for row, _full in results])
    print(f"Saved report: {csv_path}")
    if args.full:
        print(f"Saved full:   {full_jsonl}")


def main() -> None:
    args = _parse_args()
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
        _run_unvalidated_body_core_batch(videos=videos, out_dir=out_dir, args=args, backend=backend)
        return
    _run_pose33_tech_eval(videos, out_dir, args)


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
        from core.pose.backends.yolo import extract_yolo_landmark_series

        _landmarks, _valid_mask, source_meta = extract_yolo_landmark_series(v)
        return meta_for_backend(backend, source_meta, pose_variant=pose_variant) | {
            key: source_meta[key]
            for key in ("multi_person_detected", "max_persons", "multi_person_frames", "gate_status", "gate_note")
            if key in source_meta
        }

    from core.pose.backends import extract_body_core_features

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
    _write_rows_csv(csv_path, rows)

    print(f"Saved report: {csv_path}")
    if args.full:
        print(f"Saved full:   {full_jsonl}")


if __name__ == "__main__":
    main()
