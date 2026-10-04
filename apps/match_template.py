from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from core.vision_pipeline import MediaPipePipeline, PipelineConfig
from core.pose_features import normalize_pose_xy, normalize_pose_xy_v1, normalize_pose_xy_v3, subsequence_dtw
from core.feature_layout import POSE33_V3
from core.paths import models_dir, templates_dir
from core.video_writer import open_video_writer


def _extract_features(
    video_path: Path,
    *,
    pose_variant: str,
    normalizer=normalize_pose_xy,
) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0

    models_dir_path = models_dir()
    pipe = MediaPipePipeline(
        models_dir=models_dir_path,
        cfg=PipelineConfig(pose_variant=pose_variant, running_mode="video", enable_hands=False),
    )

    feats: list[np.ndarray] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(i * 1000.0 / fps)
        pose_landmarks, _hands = pipe.infer(frame, timestamp_ms=ts)
        f = normalizer(pose_landmarks)
        if f is None:
            if feats:
                f = feats[-1].copy()
            else:
                f = np.zeros((22, 2), dtype=np.float32)
        feats.append(f)
        i += 1
    cap.release()

    if not feats:
        raise RuntimeError("No frames read")
    return np.stack(feats, axis=0), fps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="Template .npz produced by make_template.py")
    ap.add_argument("--video", required=True, help="Target video to search")
    ap.add_argument("--pose", default=None, choices=["lite", "full", "heavy"], help="Override pose variant")
    ap.add_argument("--preview", action="store_true", help="Export best-match preview video")
    # YOLO 迁移 Issue #8 / S2：body_core_v1 离线闭环匹配（显式 opt-in）。
    ap.add_argument(
        "--backend",
        default=None,
        choices=["mediapipe", "yolo"],
        help="Pose backend override（默认按模板 meta.backend）。body_core_v1 模板才支持 yolo。",
    )
    ap.add_argument(
        "--feature-layout",
        dest="feature_layout",
        default=None,
        choices=["pose33_v3", "body_core_v1"],
        help="Feature layout override（默认按模板 meta.feature_layout 推断）。",
    )
    ap.add_argument(
        "--allow-multi-person",
        dest="allow_multi_person",
        action="store_true",
        help=(
            "多人场景降级而非拒绝（Issue #9）：检出多人时不报错，但仍不产出分数"
            "（标记『需人工复核』）。默认拒绝多人视频出分。"
        ),
    )
    args = ap.parse_args()

    # 按模板 meta 或显式参数判定是否走 body_core_v1 闭环，绝不改动 pose33_v3 默认路径。
    tpl = np.load(args.template, allow_pickle=True)
    tpl_meta = dict(tpl["meta"].item() or {})
    tpl_layout = str(tpl_meta.get("feature_layout", ""))
    want_body_core = (
        args.feature_layout == "body_core_v1"
        or args.backend == "yolo"
        or tpl_layout == "body_core_v1"
    )
    if want_body_core:
        _match_body_core(args, tpl_meta)
        return
    _match_pose33_v3(args)


def _match_body_core(args, tpl_meta: dict) -> None:
    """body_core_v1 模板匹配（未标定调试分数，不得对外评分）。"""
    from core.body_core_compare import (
        MultiPersonReviewRequiredError,
        match_body_core_template,
    )

    try:
        res = match_body_core_template(
            args.template,
            Path(args.video),
            backend=args.backend,
            pose_variant=args.pose,
            reject_multi_person=not args.allow_multi_person,
        )
    except MultiPersonReviewRequiredError as e:
        source_label = {
            "template": "模板来源视频",
            "video": "待匹配视频",
            "template+video": "模板来源视频与待匹配视频",
        }.get(getattr(e, "gate_source", "video"), "多人来源")
        print(f"Template: {args.template}")
        print(f"Video:    {args.video}")
        print(
            f"多人场景闸门：{source_label}检出多人（max_persons="
            f"{e.max_persons}，multi_person_frames={e.multi_person_frames}），"
            "拒绝出分 → 需人工复核（不混入正常评分结果）。"
        )
        print("  （如需查看降级结果，可加 --allow-multi-person，但分数仍不产出。）")
        return

    print(f"Template: {args.template}")
    print(f"Video:    {res.video_path}")
    print(f"Backend:  {res.backend}  feature_layout={res.feature_layout}")
    print(
        f"Match:    frames {res.start_frame}..{res.end_frame}  "
        f"(t={res.start_frame / res.fps:.2f}s..{res.end_frame / res.fps:.2f}s)"
    )
    if res.review_required or res.score is None:
        source_label = {
            "template": "模板来源视频",
            "video": "待匹配视频",
            "template+video": "模板来源视频与待匹配视频",
        }.get(res.multi_person_gate_source or "video", "多人来源")
        print(
            f"Cost:     total={res.cost:.2f}  avg/frame={res.avg_cost:.3f}  "
            f"score=N/A  baseline={res.baseline:.3f}"
        )
        print(
            f"多人场景闸门：{source_label}检出多人（max_persons="
            f"{res.max_persons}，multi_person_frames={res.multi_person_frames}），"
            "降级为『需人工复核』，不产出对外分数。"
        )
    else:
        print(
            f"Cost:     total={res.cost:.2f}  avg/frame={res.avg_cost:.3f}  "
            f"score={res.score:.3f}  baseline={res.baseline:.3f}"
        )
    print(f"Calibration: {res.calibration_status}（仅预览 / 内部标定参考，不得对外评分）")
    if res.valid_frame_ratio is not None:
        print(f"Valid body_core frames: {res.valid_frame_ratio * 100:.1f}%")


def _match_pose33_v3(args) -> None:
    tpl = np.load(args.template, allow_pickle=True)
    query = tpl["features"]
    meta = tpl["meta"].item()
    pose_variant = args.pose or meta.get("pose_variant", "full")
    layout = str(meta.get("feature_layout", "pose_indices_11_32_xy_rot_scale_norm"))
    if layout == POSE33_V3.name or layout.endswith("_v3"):
        normalizer = normalize_pose_xy_v3
    elif layout.endswith("_v2"):
        normalizer = normalize_pose_xy
    else:
        normalizer = normalize_pose_xy_v1

    video_path = Path(args.video)
    seq, fps = _extract_features(video_path, pose_variant=pose_variant, normalizer=normalizer)

    cost, start, end = subsequence_dtw(query, seq)
    avg_cost = cost / max(1, query.shape[0])
    # Baseline normalization: score=1.0 when avg_cost=0, score=0.5 when avg_cost=baseline
    # 2026-07-11 校准：2.0 → 3.0（同人自复现 avg_cost≈0.333 锚到 ~0.90 分）。
    baseline = 3.0
    score = float(baseline / (baseline + avg_cost))

    print(f"Template: {args.template}")
    print(f"Video:    {video_path}")
    print(f"Match:    frames {start}..{end}  (t={start/fps:.2f}s..{end/fps:.2f}s)")
    print(f"Cost:     total={cost:.2f}  avg/frame={avg_cost:.3f}  score={score:.3f}")

    if args.preview:
        out_dir = templates_dir()
        out_path = out_dir / f"{video_path.stem}.match_{Path(args.template).stem}.mp4"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot reopen video: {video_path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        vw, actual_path, codec = open_video_writer(out_path, fps=fps, size=(w, h))

        models_dir_path = models_dir()
        pipe = MediaPipePipeline(
            models_dir=models_dir_path,
            cfg=PipelineConfig(pose_variant=pose_variant, running_mode="video", enable_hands=False),
        )
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i < start:
                i += 1
                continue
            if i > end:
                break
            ts = int(i * 1000.0 / fps)
            annotated, _actions = pipe.annotate(frame, timestamp_ms=ts)
            vw.write(annotated)
            i += 1
        cap.release()
        vw.release()
        print(f"Saved match preview: {actual_path} (codec={codec})")


if __name__ == "__main__":
    main()
