from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from core.vision_pipeline import MediaPipePipeline, PipelineConfig
from core.pose_features import (
    DEFAULT_VALID_CONF_THR,
    MEDIAPIPE_VALIDITY_POLICY,
    motion_energy,
    normalize_pose_xy_v3,
)
from core.action_compare import (
    TemplateAutoCropReviewRequired,
    analyze_template_auto_crop,
)
from core.feature_layout import POSE33_V3
from core.paths import models_dir, templates_dir
from core.video_writer import open_video_writer


#
# NOTE: normalization/segment heuristics live in `pose_features.py` so they can be reused
# by other backends (template matching, UI actions, etc.).
#


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input video file path")
    ap.add_argument("--pose", default="full", choices=["lite", "full", "heavy"], help="Pose model variant")
    ap.add_argument("--out", default=None, help="Output template path (.npz). Default: templates/<stem>_<pose>.npz")
    ap.add_argument("--start", type=int, default=None, help="Override start frame (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="Override end frame (inclusive)")
    ap.add_argument("--preview", action="store_true", help="Export a preview video of the extracted segment")
    # YOLO 迁移 Issue #8 / S2：body_core_v1 离线闭环（显式 opt-in，默认不变）。
    ap.add_argument(
        "--backend",
        default="mediapipe",
        choices=["mediapipe", "yolo"],
        help="Pose backend. 'yolo' 仅支持 --feature-layout body_core_v1（未标定，仅供调试/标定）。",
    )
    ap.add_argument(
        "--feature-layout",
        dest="feature_layout",
        default="pose33_v3",
        choices=["pose33_v3", "body_core_v1"],
        help="Feature layout. 默认 pose33_v3（MediaPipe 旧默认，行为不变）；body_core_v1 为 S2 共享布局。",
    )
    args = ap.parse_args()

    # body_core_v1 闭环（YOLO 或 MediaPipe）走独立模块，绝不改动 pose33_v3 默认路径。
    if args.feature_layout == "body_core_v1" or args.backend == "yolo":
        _make_body_core_template(args)
        return
    try:
        _make_pose33_v3_template(args)
    except TemplateAutoCropReviewRequired as exc:
        raise SystemExit(str(exc)) from exc


def _make_body_core_template(args) -> None:
    """body_core_v1 模板生成（YOLO / MediaPipe 共享布局，仅供预览 / 内部标定参考）。"""
    from core.body_core_compare import create_body_core_template

    if args.backend == "yolo" and args.feature_layout != "body_core_v1":
        raise SystemExit("YOLO 后端仅支持 --feature-layout body_core_v1")

    out_path = create_body_core_template(
        Path(args.video),
        backend=args.backend,
        pose_variant=args.pose,
        start=args.start,
        end=args.end,
        out_path=args.out,
    )
    print(f"Saved template: {out_path}")
    print(f"Backend={args.backend} feature_layout=body_core_v1 (calibration_status=unvalidated)")
    print("说明：S3（#10）已落 baseline/阈值，但结论为仅预览 / 内部标定参考，不得对外评分。")


def _make_pose33_v3_template(args) -> None:
    video_path = Path(args.video)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    models_dir_path = models_dir()
    pipe = MediaPipePipeline(
        models_dir=models_dir_path,
        cfg=PipelineConfig(pose_variant=args.pose, running_mode="video", enable_hands=False),
    )

    feats: list[np.ndarray] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(frame_idx * 1000.0 / fps)
        pose_landmarks, _hands = pipe.infer(frame, timestamp_ms=ts)
        f = normalize_pose_xy_v3(pose_landmarks)
        if f is None:
            # Keep alignment: repeat last valid if available, else zeros.
            if feats:
                f = feats[-1].copy()
            else:
                f = np.zeros((22, 2), dtype=np.float32)
        feats.append(f)
        frame_idx += 1

    cap.release()

    if not feats:
        raise RuntimeError("No frames read from video")

    feat_arr = np.stack(feats, axis=0)  # (T, 22, 2)
    seq = feat_arr.reshape(feat_arr.shape[0], -1)  # (T, 44)
    energy = motion_energy(seq)
    auto_crop = analyze_template_auto_crop(energy, fps=fps)
    auto_start, auto_end = auto_crop.start_frame, auto_crop.end_frame
    if args.start is None and args.end is None and auto_crop.requires_review:
        raise TemplateAutoCropReviewRequired(auto_crop)
    start = int(args.start) if args.start is not None else int(auto_start)
    end = int(args.end) if args.end is not None else int(auto_end)
    start = max(0, min(start, feat_arr.shape[0] - 1))
    end = max(start, min(end, feat_arr.shape[0] - 1))

    out_dir = templates_dir()
    out_path = Path(args.out) if args.out else (out_dir / f"{video_path.stem}_{args.pose}.npz")

    meta = {
        "video": str(video_path),
        "fps": fps,
        "frame_count": n_frames,
        "start_frame": int(start),
        "end_frame": int(end),
        "auto_start_frame": int(auto_start),
        "auto_end_frame": int(auto_end),
        "pose_variant": args.pose,
        "feature_layout": POSE33_V3.name,
        "running_mode": "video",
        "cfg": asdict(PipelineConfig(pose_variant=args.pose, running_mode="video", enable_hands=False)),
        # 模板 metadata 扩展（YOLO 迁移 S1 / Issue #6）：增量字段，向后兼容。
        "backend": "mediapipe",
        "model_name": f"pose_landmarker_{args.pose}",
        "normalizer_version": "v3",
        "confidence_kind": "visibility",
        "validity_policy": MEDIAPIPE_VALIDITY_POLICY,
        "valid_conf_thr": float(DEFAULT_VALID_CONF_THR),
        "crop_selection": "manual" if args.start is not None or args.end is not None else "auto",
        "selected_retained_ratio": float((end - start + 1) / feat_arr.shape[0]),
        "template_scope": "single_action",
        **auto_crop.to_meta(),
    }

    np.savez_compressed(
        out_path,
        features=feat_arr[start : end + 1],
        meta=np.array(meta, dtype=object),
    )
    print(f"Saved template: {out_path}")
    print(f"Segment frames: {start}..{end} (len={end - start + 1}, fps={fps:.3f})")
    print(f"Auto segment:   {auto_start}..{auto_end}")

    if args.preview:
        preview_path = out_path.with_suffix(".preview.mp4")
        cap2 = cv2.VideoCapture(str(video_path))
        if not cap2.isOpened():
            raise RuntimeError(f"Cannot reopen video: {video_path}")
        w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        vw, actual_path, codec = open_video_writer(preview_path, fps=fps, size=(w, h))

        pipe2 = MediaPipePipeline(
            models_dir=models_dir_path,
            cfg=PipelineConfig(pose_variant=args.pose, running_mode="video", enable_hands=False),
        )
        i = 0
        while True:
            ok, frame = cap2.read()
            if not ok:
                break
            if i < start:
                i += 1
                continue
            if i > end:
                break
            ts = int(i * 1000.0 / fps)
            annotated, _actions = pipe2.annotate(frame, timestamp_ms=ts)
            vw.write(annotated)
            i += 1

        cap2.release()
        vw.release()
        print(f"Saved preview: {actual_path} (codec={codec})")


if __name__ == "__main__":
    main()
