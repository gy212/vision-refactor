"""``make_template`` CLI：从标准动作视频生成比对模板（pose33_v3 与 body_core 调试路径）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.evaluation import COMPARE, load_template
from core.foundation import templates_dir
from core.media import VideoOpenError
from core.services import (
    TemplateAutoCropReviewRequired,
    TemplateQualityError,
    create_template_from_video,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input video file path")
    ap.add_argument(
        "--pose",
        default="heavy",
        choices=["lite", "full", "heavy"],
        help="Pose model variant",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output template path (.npz). Default: templates/<stem>_<pose>.npz",
    )
    ap.add_argument(
        "--start", type=int, default=None, help="Override start frame (inclusive)"
    )
    ap.add_argument(
        "--end", type=int, default=None, help="Override end frame (inclusive)"
    )
    ap.add_argument(
        "--no-auto-crop",
        action="store_true",
        help="Disable auto crop and keep the whole video when --start/--end are omitted",
    )
    ap.add_argument(
        "--preview",
        action="store_true",
        help="Export a preview video of the extracted segment",
    )
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
    except (TemplateAutoCropReviewRequired, TemplateQualityError) as exc:
        raise SystemExit(str(exc)) from exc


def _make_body_core_template(args) -> None:
    """body_core_v1 模板生成（YOLO / MediaPipe 共享布局，仅供预览 / 内部标定参考）。"""
    from core.services import create_body_core_template

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
    print(
        f"Backend={args.backend} feature_layout=body_core_v1 (calibration_status=unvalidated)"
    )
    print(
        "说明：S3（#10）已落 baseline/阈值，但结论为仅预览 / 内部标定参考，不得对外评分。"
    )


def _make_pose33_v3_template(args) -> None:
    video_path = Path(args.video)
    out_path = (
        Path(args.out)
        if args.out
        else (templates_dir() / f"{video_path.stem}_{args.pose}.npz")
    )
    try:
        saved_path = create_template_from_video(
            video_path,
            pose_variant=args.pose,
            start=args.start,
            end=args.end,
            out_path=out_path,
            preview=args.preview,
            auto_crop=not args.no_auto_crop,
        )
    except VideoOpenError:
        raise RuntimeError(f"Cannot open video: {video_path}") from None

    meta = load_template(saved_path, COMPARE).raw_meta
    start = int(meta.get("start_frame") or 0)
    end = int(meta.get("end_frame") or start)
    auto_start = int(meta.get("auto_start_frame") or start)
    auto_end = int(meta.get("auto_end_frame") or end)
    fps = float(meta.get("fps") or 30.0)
    print(f"Saved template: {saved_path}")
    print(f"Segment frames: {start}..{end} (len={end - start + 1}, fps={fps:.3f})")
    print(f"Auto segment:   {auto_start}..{auto_end}")
    if args.preview:
        print(f"Saved preview: {saved_path.with_suffix('.preview.mp4')}")
