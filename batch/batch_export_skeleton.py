from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from core.vision_pipeline import MediaPipePipeline, PipelineConfig
from core.paths import models_dir, outputs_dir
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


def _extract_pose_and_video(
    video_path: Path,
    *,
    out_video: Path | None,
    pose_variant: str,
    draw_face: bool,
) -> tuple[np.ndarray, dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    models_dir_path = models_dir()
    pipe = MediaPipePipeline(
        models_dir=models_dir_path,
        cfg=PipelineConfig(
            pose_variant=pose_variant,
            running_mode="video",
            enable_hands=False,
            draw_pose_face=draw_face,
        ),
    )

    writer: cv2.VideoWriter | None = None
    if out_video is not None:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))

    out: list[np.ndarray] = []
    last: np.ndarray | None = None

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(i * 1000.0 / fps)
        pose_landmarks, _hands = pipe.infer(frame, timestamp_ms=ts)
        if pose_landmarks is None:
            arr = last.copy() if last is not None else np.zeros((33, 4), dtype=np.float32)
        else:
            arr = np.zeros((33, 4), dtype=np.float32)
            for j in range(min(33, len(pose_landmarks))):
                lm = pose_landmarks[j]
                arr[j, 0] = float(getattr(lm, "x", 0.0))
                arr[j, 1] = float(getattr(lm, "y", 0.0))
                arr[j, 2] = float(getattr(lm, "z", 0.0))
                arr[j, 3] = float(getattr(lm, "visibility", 0.0))
            last = arr

        out.append(arr)

        if writer is not None:
            annotated = frame.copy()
            MediaPipePipeline._draw_pose(annotated, pose_landmarks, w, h, draw_face=draw_face)
            writer.write(annotated)

        i += 1

    cap.release()
    if writer is not None:
        writer.release()

    if not out:
        raise RuntimeError(f"未能从视频提取姿态：{video_path}")

    meta = {
        "video": str(video_path),
        "name": video_path.stem.strip(),
        "fps": float(fps),
        "frame_count": int(n_frames),
        "width": int(w),
        "height": int(h),
        "pose_variant": str(pose_variant),
        "landmark_layout": "pose33_normalized_xyzw(visibility)",
        "skeleton_video": None if out_video is None else str(out_video),
    }
    return np.stack(out, axis=0), meta


def _video_basic_meta(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    try:
        return {
            "video": str(video_path),
            "name": video_path.stem.strip(),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
    finally:
        cap.release()


def _extract_body_core_features(video_path: Path, *, backend: str, pose_variant: str) -> tuple[np.ndarray, dict]:
    from core.body_core_compare import extract_body_core_features

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
        data = np.load(path, allow_pickle=True)
        return dict(data["meta"].item() or {})
    except Exception:  # noqa: BLE001 - stale/corrupt skipped files still get a manifest row.
        return {}


def main() -> None:
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
    args = ap.parse_args()
    backend, feature_layout = normalize_backend_layout(args.backend, args.feature_layout)

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

    used_names_by_dir: dict[Path, set[str]] = {}
    work_items: list[tuple[int, Path, Path, Path, Path, Path | None]] = []
    for idx, video_path in enumerate(videos, start=1):
        rel_parent = video_path.parent.relative_to(source_dir)
        out_subdir = out_dir / rel_parent
        used = used_names_by_dir.setdefault(out_subdir, set())

        base_name = _sanitize_name(video_path.stem)
        safe_name = _unique_name(base_name, used)

        if is_default_pose33_path(backend, feature_layout):
            npz_path = out_subdir / f"{safe_name}_pose33_{args.pose}.npz"
            skel_video_path = None if args.no_video else (out_subdir / f"{safe_name}_skeleton_{args.pose}.mp4")
        else:
            npz_path = out_subdir / f"{safe_name}_{backend}_{FEATURE_LAYOUT_BODY_CORE}.npz"
            # body_core_v1 features are normalized coordinates, not overlay-ready pixel landmarks.
            skel_video_path = None

        work_items.append((idx, video_path, rel_parent, out_subdir, npz_path, skel_video_path))

    def _process_video(item: tuple[int, Path, Path, Path, Path, Path | None]) -> tuple[int, dict]:
        idx, video_path, rel_parent, out_subdir, npz_path, skel_video_path = item
        if not args.overwrite:
            if npz_path.exists() and (skel_video_path is None or skel_video_path.exists()):
                print(f"[{idx}/{len(videos)}] Skip (exists): {video_path}")
                return (
                    idx,
                    _manifest_row(
                        rel_parent=rel_parent,
                        video_path=video_path,
                        npz_path=npz_path,
                        skel_video_path=skel_video_path,
                        pose_variant=str(args.pose),
                        meta=_read_npz_meta(npz_path),
                    ),
                )

        out_subdir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(videos)}] Processing: {video_path}")
        if is_default_pose33_path(backend, feature_layout):
            landmarks, meta = _extract_pose_and_video(
                video_path,
                out_video=skel_video_path,
                pose_variant=str(args.pose),
                draw_face=bool(args.draw_face),
            )
        else:
            features, meta = _extract_body_core_features(
                video_path,
                backend=backend,
                pose_variant=str(args.pose),
            )
        meta["output_npz"] = str(npz_path)
        meta["action"] = str(rel_parent)
        if is_default_pose33_path(backend, feature_layout):
            np.savez_compressed(npz_path, landmarks=landmarks, meta=np.array(meta, dtype=object))
        else:
            np.savez_compressed(npz_path, features=features, meta=np.array(meta, dtype=object))

        return (
            idx,
            _manifest_row(
                rel_parent=rel_parent,
                video_path=video_path,
                npz_path=npz_path,
                skel_video_path=skel_video_path,
                pose_variant=str(args.pose),
                meta=meta,
            ),
        )

    video_workers = max(1, int(args.workers))
    results: list[tuple[int, dict]] = []
    if video_workers <= 1:
        for item in work_items:
            results.append(_process_video(item))
    else:
        with ThreadPoolExecutor(max_workers=video_workers) as ex:
            futures = [ex.submit(_process_video, item) for item in work_items]
            for fut in as_completed(futures):
                results.append(fut.result())

    results.sort(key=lambda item: item[0])
    manifest_rows = [row for _idx, row in results]

    manifest_path = out_dir / "manifest.csv"
    _write_manifest(manifest_rows, manifest_path)
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved skeleton data: {out_dir}")


if __name__ == "__main__":
    main()
