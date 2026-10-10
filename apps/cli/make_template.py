"""``make_template`` CLI：从标准动作视频生成比对模板（pose33_v3 与 body_core 调试路径）。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from core.foundation import motion_energy, templates_dir
from core.media import VideoOpenError
from core.pose import (
    DEFAULT_VALID_CONF_THR,
    FORMAL,
    MEDIAPIPE_VALIDITY_POLICY,
    POSE33_V3,
    extract_pose_sequence,
    formal_engine,
    normalize_pose_xy_v3,
)
from core.services import TemplateAutoCropReviewRequired, analyze_template_auto_crop, write_pose_preview


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
    try:
        pose_seq = extract_pose_sequence(video_path, formal_engine(args.pose), FORMAL)
    except VideoOpenError:
        raise RuntimeError(f"Cannot open video: {video_path}") from None
    if len(pose_seq) == 0:
        raise RuntimeError("No frames read from video")
    fps, n_frames = pose_seq.meta.fps, pose_seq.meta.frame_count
    # 未检出帧沿用上一帧特征、首帧前补零，保持与视频帧对齐。
    feats = pose_seq.features(normalize_pose_xy_v3, tuple(POSE33_V3.shape))

    feat_arr = feats  # (T, 22, 2)
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
        "cfg": asdict(formal_engine(args.pose)),
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
        actual_path, codec = write_pose_preview(
            video_path, out_path=preview_path, start_frame=start, end_frame=end, pose_variant=args.pose,
        )
        print(f"Saved preview: {actual_path} (codec={codec})")
