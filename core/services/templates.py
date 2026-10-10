# -*- coding: utf-8 -*-
"""从标准动作视频生成 pose33_v3 比对模板，以及长视频自动裁剪的复核（CompareWindow、make_template CLI、batch 共用）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event

import numpy as np

from core.foundation.paths import templates_dir
from core.foundation.signal import find_active_range, motion_energy
from core.pose.engine import EngineSpec
from core.pose.features import ProgressCb, extract_features
from core.pose.layouts import POSE33_V3
from core.pose.normalize import normalize_pose_xy_v3
from core.pose.validity import DEFAULT_VALID_CONF_THR, MEDIAPIPE_VALIDITY_POLICY
from core.services.preview import export_match_preview


@dataclass(frozen=True)
class TemplateAutoCropAnalysis:
    """Diagnostics for the legacy single-action template auto crop."""

    start_frame: int
    end_frame: int
    frame_count: int
    fps: float
    activity_groups: tuple[tuple[int, int], ...]

    @property
    def selected_frames(self) -> int:
        return max(0, int(self.end_frame) - int(self.start_frame) + 1)

    @property
    def retained_ratio(self) -> float:
        return float(self.selected_frames) / float(max(1, int(self.frame_count)))

    @property
    def duration_s(self) -> float:
        return float(self.frame_count) / float(self.fps)

    @property
    def selected_duration_s(self) -> float:
        return float(self.selected_frames) / float(self.fps)

    @property
    def requires_review(self) -> bool:
        if self.duration_s < 30.0 or len(self.activity_groups) < 2:
            return False
        suspicious_limit_s = max(5.0, self.duration_s * 0.10)
        return self.selected_duration_s <= suspicious_limit_s

    def to_meta(self) -> dict:
        return {
            "auto_crop_strategy": "primary_activity_segment_v1",
            "auto_activity_group_count": len(self.activity_groups),
            "auto_activity_groups": [
                {"start_frame": int(start), "end_frame": int(end)}
                for start, end in self.activity_groups
            ],
            "auto_retained_ratio": float(self.retained_ratio),
            "auto_review_required": bool(self.requires_review),
        }


class TemplateAutoCropReviewRequired(ValueError):
    def __init__(self, analysis: TemplateAutoCropAnalysis):
        self.analysis = analysis
        ranges = ", ".join(
            f"{start}..{end} ({start / analysis.fps:.1f}-{end / analysis.fps:.1f}s)"
            for start, end in analysis.activity_groups[:12]
        )
        if len(analysis.activity_groups) > 12:
            ranges += f"，另有 {len(analysis.activity_groups) - 12} 段"
        super().__init__(
            "长视频自动裁剪需要人工确认："
            f"输入 {analysis.duration_s:.2f}s，当前算法只会保留 "
            f"{analysis.selected_duration_s:.2f}s "
            f"({analysis.retained_ratio * 100.0:.1f}%)，"
            f"并检测到 {len(analysis.activity_groups)} 个活动组。"
            "请填写 startFrame/endFrame（CLI 使用 --start/--end），"
            "或先按动作拆分视频后分别生成模板。"
            f"候选区间：{ranges}"
        )


def analyze_template_auto_crop(
    energy: np.ndarray,
    *,
    fps: float,
) -> TemplateAutoCropAnalysis:
    """Analyze whether the single-action auto crop is unsafe for a long video.

    The selected range intentionally remains identical to ``find_active_range``.
    Activity groups are diagnostics only: they prevent a long multi-activity video
    from silently becoming a tiny template without changing scoring/exam callers.
    """
    values = np.asarray(energy, dtype=np.float32).reshape(-1)
    fps_value = float(fps or 0.0) or 30.0
    frame_count = int(values.size + 1)
    auto_start, auto_end = find_active_range(values, pad=10)

    if values.size == 0:
        groups: tuple[tuple[int, int], ...] = ()
    else:
        smooth_frames = max(3, int(round(0.30 * fps_value)))
        kernel = np.ones(smooth_frames, dtype=np.float32) / float(smooth_frames)
        smoothed = (
            np.convolve(values, kernel, mode="same")
            if values.size >= smooth_frames
            else values
        )
        active = smoothed > float(np.percentile(smoothed, 70))
        runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for index, is_active in enumerate(active.tolist()):
            if is_active and run_start is None:
                run_start = index
            elif (not is_active) and run_start is not None:
                runs.append((run_start, index - 1))
                run_start = None
        if run_start is not None:
            runs.append((run_start, int(active.size - 1)))

        min_active_frames = max(2, int(round(0.15 * fps_value)))
        runs = [
            (start, end)
            for start, end in runs
            if end - start + 1 >= min_active_frames
        ]
        merge_gap_frames = max(1, int(round(2.0 * fps_value)))
        merged: list[tuple[int, int]] = []
        for start, end in runs:
            if merged and start - merged[-1][1] - 1 <= merge_gap_frames:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))

        group_pad_frames = max(1, int(round(0.50 * fps_value)))
        groups = tuple(
            (
                max(0, int(start - group_pad_frames)),
                min(frame_count - 1, int(end + 1 + group_pad_frames)),
            )
            for start, end in merged
        )

    return TemplateAutoCropAnalysis(
        start_frame=int(auto_start),
        end_frame=int(auto_end),
        frame_count=frame_count,
        fps=fps_value,
        activity_groups=groups,
    )


def create_template_from_video(
    video_path: str | Path,
    *,
    pose_variant: str = "heavy",
    start: int | None = None,
    end: int | None = None,
    out_path: str | Path | None = None,
    workers: int = 1,
    preview: bool = False,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> Path:
    video_path = Path(video_path)
    features, fps, _view = extract_features(
        video_path,
        pose_variant=pose_variant,
        workers=workers,
        normalizer=normalize_pose_xy_v3,
        progress_cb=progress_cb,
        stop_evt=stop_evt,
    )

    seq = features.reshape(features.shape[0], -1)
    energy = motion_energy(seq)
    auto_crop = analyze_template_auto_crop(energy, fps=fps)
    auto_start, auto_end = auto_crop.start_frame, auto_crop.end_frame
    if start is None and end is None and auto_crop.requires_review:
        raise TemplateAutoCropReviewRequired(auto_crop)

    start_i = int(start) if start is not None else int(auto_start)
    end_i = int(end) if end is not None else int(auto_end)
    start_i = max(0, min(start_i, features.shape[0] - 1))
    end_i = max(start_i, min(end_i, features.shape[0] - 1))

    out_dir = templates_dir()
    out_path = Path(out_path) if out_path else (out_dir / f"{video_path.stem}_{pose_variant}.npz")

    meta = {
        "video": str(video_path),
        "fps": float(fps),
        "frame_count": int(features.shape[0]),
        "start_frame": int(start_i),
        "end_frame": int(end_i),
        "auto_start_frame": int(auto_start),
        "auto_end_frame": int(auto_end),
        "pose_variant": pose_variant,
        "feature_layout": POSE33_V3.name,
        "running_mode": "video",
        "cfg": asdict(EngineSpec(pose_variant=pose_variant, running_mode="video", enable_hands=False)),
        # 模板 metadata 扩展（YOLO 迁移 S1 / Issue #6）：增量字段，向后兼容。
        "backend": "mediapipe",
        "model_name": f"pose_landmarker_{pose_variant}",
        "normalizer_version": "v3",
        "confidence_kind": "visibility",
        "validity_policy": MEDIAPIPE_VALIDITY_POLICY,
        "valid_conf_thr": float(DEFAULT_VALID_CONF_THR),
        "crop_selection": "manual" if start is not None or end is not None else "auto",
        "selected_retained_ratio": float((end_i - start_i + 1) / features.shape[0]),
        "template_scope": "single_action",
        **auto_crop.to_meta(),
    }

    np.savez_compressed(
        out_path,
        features=features[start_i : end_i + 1],
        meta=np.array(meta, dtype=object),
    )

    if preview and progress_cb is not None:
        progress_cb("导出模板预览", 0, 1)
    if preview:
        preview_path = out_path.with_suffix(".preview.mp4")
        export_match_preview(
            video_path,
            out_path=preview_path,
            start_frame=start_i,
            end_frame=end_i,
            pose_variant=pose_variant,
            progress_cb=progress_cb,
            stop_evt=stop_evt,
        )

    return out_path
