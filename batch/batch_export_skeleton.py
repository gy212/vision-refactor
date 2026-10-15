"""批处理入口：批量导出 Pose33 骨架数据与骨架叠加视频。"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Iterable

import cv2
import numpy as np

# 允许以 ``python apps/xxx.py``、``python batch/xxx.py`` 方式运行（AGENTS.md 记载的命令）。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.media import probe_video
from core.foundation import load_npz, npz_meta, outputs_dir
from core.pose import FORMAL, GapPolicy, draw_skeleton, extract_pose_sequence, formal_engine
from batch.backend_options import (
    FEATURE_LAYOUT_BODY_CORE,
    add_backend_layout_args,
    csv_meta_fields,
    is_default_pose33_path,
    meta_for_backend,
    normalize_backend_layout,
)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def _iter_videos(root: Path, *, skip_keywords: tuple[str, ...]) -> list[Path]:
    videos: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        if _should_skip(p, skip_keywords=skip_keywords):
            continue
        videos.append(p)
    return sorted(videos)


def _should_skip(path: Path, *, skip_keywords: tuple[str, ...]) -> bool:
    if not skip_keywords:
        return False
    for part in path.parts:
        for kw in skip_keywords:
            if kw and kw in part:
                return True
    return False


def _sanitize_name(name: str) -> str:
    cleaned = name.strip().rstrip(".")
    for ch in "\\/:*?\"<>|":
        cleaned = cleaned.replace(ch, "_")
    return cleaned or "video"


def _unique_name(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    idx = 2
    while True:
        cand = f"{base}_{idx}"
        if cand not in used:
            used.add(cand)
            return cand
        idx += 1


class _SkeletonVideoWriter:
    """边提取边写骨架视频（mp4v）：每帧推理后在原帧上画骨架，未检出帧不画；只解码一遍。"""

    def __init__(self, video_path: Path, out_video: Path, *, draw_face: bool) -> None:
        meta = probe_video(video_path)
        out_video.parent.mkdir(parents=True, exist_ok=True)
        self._draw_face = draw_face
        self._writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), meta.fps,
                                       (meta.width, meta.height))

    def __call__(self, frame, pose_landmarks) -> None:
        annotated = frame.bgr.copy()
        if pose_landmarks is not None:
            draw_skeleton(annotated, pose_landmarks, draw_face=self._draw_face)
        self._writer.write(annotated)

    def release(self) -> None:
        self._writer.release()


def _extract_pose_and_video(
    video_path: Path,
    *,
    out_video: Path | None,
    pose_variant: str,
    draw_face: bool,
) -> tuple[np.ndarray, dict]:
    """``FORMAL`` 顺序提取，原始序列按 ``HOLD_LAST_INVALID`` 补未检出帧（补帧判为无效）；可选同时导出骨架视频。"""
    writer = None if out_video is None else _SkeletonVideoWriter(video_path, out_video, draw_face=draw_face)
    try:
        seq = extract_pose_sequence(video_path, formal_engine(pose_variant), FORMAL, on_frame=writer)
    finally:
        if writer is not None:
            writer.release()
    if len(seq) == 0:
        raise RuntimeError(f"未能从视频提取姿态：{video_path}")

    m = seq.meta
    meta = {
        "video": str(video_path),
        "name": video_path.stem.strip(),
        "fps": float(m.fps),
        "frame_count": int(m.frame_count),
        "width": int(m.width),
        "height": int(m.height),
        "pose_variant": str(pose_variant),
        "landmark_layout": "pose33_normalized_xyzw(visibility)",
        "skeleton_video": None if out_video is None else str(out_video),
    }
    return seq.raw(GapPolicy.HOLD_LAST_INVALID).copy(), meta


def _video_basic_meta(video_path: Path) -> dict:
    meta = probe_video(video_path)
    return {
        "video": str(video_path),
        "name": video_path.stem.strip(),
        "fps": meta.fps,
        "frame_count": meta.frame_count,
        "width": meta.width,
        "height": meta.height,
    }


def _extract_body_core_features(video_path: Path, *, backend: str, pose_variant: str) -> tuple[np.ndarray, dict]:
    from core.pose.backends import extract_body_core_features

    features, fps, backend_meta = extract_body_core_features(
        video_path,
        backend=backend,
        pose_variant=pose_variant,
    )
    meta = {
        **_video_basic_meta(video_path),
        **meta_for_backend(backend, backend_meta, pose_variant=pose_variant),
        "fps": float(fps),
        "landmark_layout": "body_core_v1_normalized_xy",
        "skeleton_video": None,
        "body_core_valid_frame_ratio": backend_meta.get("body_core_valid_frame_ratio", ""),
    }
    for key in ("multi_person_detected", "max_persons", "multi_person_frames", "gate_status", "gate_note"):
        if key in backend_meta:
            meta[key] = backend_meta[key]
    return features, meta


def _write_manifest(rows: Iterable[dict], out_path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _manifest_row(
    *,
    rel_parent: Path,
    video_path: Path,
    npz_path: Path,
    skel_video_path: Path | None,
    pose_variant: str,
    meta: dict | None = None,
) -> dict:
    meta = dict(meta or {})
    row = {
        "action": str(rel_parent),
        "video_name": video_path.name,
        "source_video": str(video_path),
        "skeleton_npz": str(npz_path),
        "skeleton_video": "" if skel_video_path is None else str(skel_video_path),
        "pose_variant": str(pose_variant),
        "fps": meta.get("fps", ""),
        "frame_count": meta.get("frame_count", ""),
        "width": meta.get("width", ""),
        "height": meta.get("height", ""),
    }
    row.update(csv_meta_fields(meta))
    return row


def _read_npz_meta(path: Path) -> dict:
    try:
        return npz_meta(load_npz(path))
    except Exception:  # noqa: BLE001 - stale/corrupt skipped files still get a manifest row.
        return {}


@dataclass(frozen=True)
class _ExportItem:
    idx: int
    video_path: Path
    rel_parent: Path
    out_subdir: Path
    npz_path: Path
    skel_video_path: Path | None


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch export Pose33 skeleton data + skeleton videos.")
    ap.add_argument("--source_dir", default="标准动作视频--分解版", help="Input root folder")
    ap.add_argument("--out_dir", default=None, help="Output root folder (default: outputs/<name>_<ts>)")
    ap.add_argument("--pose", default="full", choices=["lite", "full", "heavy"], help="Pose model variant")
    ap.add_argument(
        "--skip_keywords",
        default="汇总",
        help="Comma-separated keywords to skip folders (default: 汇总)",
    )
    ap.add_argument("--no_video", action="store_true", help="Only export .npz (skip skeleton video)")
    ap.add_argument("--draw_face", action="store_true", help="Draw face landmarks on skeleton video")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--workers", type=int, default=1, help="Number of videos to process concurrently (default: 1)")
    add_backend_layout_args(ap)
    return ap.parse_args()


def _export_items(videos: list[Path], source_dir: Path, out_dir: Path, args, *, pose33: bool, backend: str) -> list[_ExportItem]:
    """为每个视频定好输出子目录与不重名的 npz、骨架视频路径（序号从 1 开始）。"""
    used_names_by_dir: dict[Path, set[str]] = {}
    items: list[_ExportItem] = []
    for idx, video_path in enumerate(videos, start=1):
        rel_parent = video_path.parent.relative_to(source_dir)
        out_subdir = out_dir / rel_parent
        safe_name = _unique_name(_sanitize_name(video_path.stem), used_names_by_dir.setdefault(out_subdir, set()))
        if pose33:
            npz_path = out_subdir / f"{safe_name}_pose33_{args.pose}.npz"
            skel_video_path = None if args.no_video else (out_subdir / f"{safe_name}_skeleton_{args.pose}.mp4")
        else:
            npz_path = out_subdir / f"{safe_name}_{backend}_{FEATURE_LAYOUT_BODY_CORE}.npz"
            # body_core_v1 features are normalized coordinates, not overlay-ready pixel landmarks.
            skel_video_path = None
        items.append(_ExportItem(idx, video_path, rel_parent, out_subdir, npz_path, skel_video_path))
    return items


def _export_one(item: _ExportItem, args, *, pose33: bool, backend: str, total: int) -> dict:
    """导出一个视频的 npz（及骨架视频），返回清单行；已存在且未要求覆盖时只读回元数据。"""
    def row(meta: dict) -> dict:
        return _manifest_row(rel_parent=item.rel_parent, video_path=item.video_path, npz_path=item.npz_path,
                             skel_video_path=item.skel_video_path, pose_variant=str(args.pose), meta=meta)

    if not args.overwrite and item.npz_path.exists() and (item.skel_video_path is None or item.skel_video_path.exists()):
        print(f"[{item.idx}/{total}] Skip (exists): {item.video_path}")
        return row(_read_npz_meta(item.npz_path))

    item.out_subdir.mkdir(parents=True, exist_ok=True)
    print(f"[{item.idx}/{total}] Processing: {item.video_path}")
    if pose33:
        arrays, meta = _extract_pose_and_video(item.video_path, out_video=item.skel_video_path,
                                               pose_variant=str(args.pose), draw_face=bool(args.draw_face))
        key = "landmarks"
    else:
        arrays, meta = _extract_body_core_features(item.video_path, backend=backend, pose_variant=str(args.pose))
        key = "features"
    meta["output_npz"] = str(item.npz_path)
    meta["action"] = str(item.rel_parent)
    np.savez_compressed(item.npz_path, **{key: arrays}, meta=np.array(meta, dtype=object))
    return row(meta)


def main() -> None:
    args = _parse_args()
    backend, feature_layout = normalize_backend_layout(args.backend, args.feature_layout)
    pose33 = is_default_pose33_path(backend, feature_layout)
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir not found: {source_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (outputs_dir() / f"标准动作视频--分解版_骨架_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    skip_keywords = tuple(k.strip() for k in str(args.skip_keywords).split(",") if k.strip())
    videos = _iter_videos(source_dir, skip_keywords=skip_keywords)
    if not videos:
        raise FileNotFoundError(f"No videos found in: {source_dir}")
    items = _export_items(videos, source_dir, out_dir, args, pose33=pose33, backend=backend)

    def process(item: _ExportItem) -> dict:
        return _export_one(item, args, pose33=pose33, backend=backend, total=len(videos))

    video_workers = max(1, int(args.workers))
    if video_workers <= 1:
        manifest_rows = [process(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=video_workers) as ex:
            manifest_rows = list(ex.map(process, items))

    manifest_path = out_dir / "manifest.csv"
    _write_manifest(manifest_rows, manifest_path)
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved skeleton data: {out_dir}")


if __name__ == "__main__":
    main()
