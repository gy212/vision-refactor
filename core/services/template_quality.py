# -*- coding: utf-8 -*-
"""模板建库的录制目录发现、动作名校验、落盘路径规划与骨架质量体检（L4 公共服务）。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.evaluation.templates import VIEW_SUFFIXES, split_view_suffix
from core.foundation.paths import templates_dir
from core.foundation.signal import find_active_range, motion_energy
from core.media.segment_layout import VIDEO_SUFFIXES, SegmentLayout
from core.pose.validity import DEFAULT_VALID_CONF_THR, derive_valid_mask

MIN_TEMPLATE_FRAMES = 8
MIN_DETECTED_RATIO = 0.6
MIN_CORE_VALID_RATIO = 0.6

_CORE_INDICES = (11, 12, 23, 24)
_LIMB_INDICES = (13, 14, 15, 16, 25, 26, 27, 28)
_ILLEGAL_NAME_CHARS = set('/\\:*?"<>|')
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)


@dataclass(frozen=True)
class RecordingPair:
    """录制目录或单视频文件发现到的正/侧视频路径。"""

    front: Path | None
    side: Path | None


@dataclass(frozen=True)
class TemplateQualityReport:
    """模板截取片段的骨架检出与关键点质量体检报告。"""

    frame_count: int
    detected_frames: int
    detected_ratio: float
    core_valid_frames: int
    core_valid_ratio: float
    limb_valid_ratio: float
    warnings: tuple[str, ...] = ()

    def to_meta(self) -> dict:
        return {
            "frame_count": int(self.frame_count),
            "detected_frames": int(self.detected_frames),
            "detected_ratio": float(self.detected_ratio),
            "core_valid_frames": int(self.core_valid_frames),
            "core_valid_ratio": float(self.core_valid_ratio),
            "limb_valid_ratio": float(self.limb_valid_ratio),
            "warnings": list(self.warnings),
        }


class TemplateQualityError(ValueError):
    """截取区间帧数不足、全为空帧或骨架有效率不达标时抛出的拒绝异常。"""

    def __init__(
        self, message: str, *, report: TemplateQualityReport | None = None
    ) -> None:
        super().__init__(message)
        self.report: TemplateQualityReport | None = report


def discover_recording_videos(path: str | Path) -> RecordingPair:
    """从录制片段目录（委托 ``SegmentLayout.resolve_videos``）或单视频文件解析待建库视频。"""
    p = Path(path)
    if p.is_dir():
        front, side = SegmentLayout.resolve_videos(p)
        if front is None and side is None:
            exts = ", ".join(sorted(VIDEO_SUFFIXES))
            raise ValueError(
                f"录制目录下未找到 front.mp4/.avi 或 side.mp4/.avi（支持 {exts}）：{p}"
            )
        return RecordingPair(front=front, side=side)
    if p.is_file():
        if p.suffix.lower() not in VIDEO_SUFFIXES:
            exts = ", ".join(sorted(VIDEO_SUFFIXES))
            raise ValueError(f"不支持的视频文件类型：{p.name}（仅支持 {exts}）")
        return RecordingPair(front=p, side=None)
    raise FileNotFoundError(f"视频文件或目录不存在：{p}")


def sanitize_action_name(action_name: str) -> str:
    """校验并规范化动作名，拒绝空白、非法路径字符、Windows 保留设备名及自带视角后缀。"""
    name = str(action_name or "").strip()
    if not name:
        raise ValueError("动作名称不能为空")
    if (
        name in {".", ".."}
        or ".." in name
        or name.endswith(".")
        or any(ch in _ILLEGAL_NAME_CHARS or ord(ch) < 32 for ch in name)
    ):
        raise ValueError(f"动作名称包含非法路径字符：{action_name!r}")
    if name.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"动作名称不能使用系统保留设备名：{action_name!r}")
    _base, view_suffix = split_view_suffix(name)
    if view_suffix is not None:
        raise ValueError(f"动作名称无需自带视角后缀（_正面/_侧面）：{action_name!r}")
    return name


def validate_frame_bounds(
    n_frames: int, start: int | None, end: int | None
) -> tuple[int, int]:
    """校验并解析闭区间起止帧 ``(start_i, end_i)``，越界或 ``start > end`` 时抛 ``ValueError``。"""
    total = int(n_frames)
    if total <= 0:
        raise ValueError(f"视频总帧数无效：{total}")
    start_i = 0 if start is None else int(start)
    end_i = (total - 1) if end is None else int(end)
    if start_i < 0 or end_i < 0 or start_i >= total or end_i >= total:
        raise ValueError(
            f"起止帧越界：{start_i}..{end_i}（总帧数 {total}，有效范围 0..{total - 1}）"
        )
    if start_i > end_i:
        raise ValueError(f"起始帧 {start_i} 不能大于结束帧 {end_i}")
    return start_i, end_i


def planned_template_paths(
    action_name: str,
    *,
    views: Sequence[str],
    out_dir: str | Path | None = None,
) -> dict[str, Path]:
    """按动作名与视角列表生成落盘目标路径（front→_正面.npz，side→_侧面.npz，single→.npz）。"""
    name = sanitize_action_name(action_name)
    target_dir = Path(out_dir) if out_dir is not None else templates_dir()
    out: dict[str, Path] = {}
    for view in views:
        if view not in VIEW_SUFFIXES:
            raise ValueError(
                f"不支持的模板视角：{view!r}（可选：{sorted(VIEW_SUFFIXES)}）"
            )
        out[view] = target_dir / f"{name}{VIEW_SUFFIXES[view]}.npz"
    return out


def assess_template_quality(
    pose_seq: object | None,
    seg_feats: np.ndarray,
    start_i: int,
    end_i: int,
    *,
    validate_quality: bool = True,
) -> TemplateQualityReport:
    """评估截取区间内的骨架检出率、躯干核心点/四肢点有效率与静止告警。"""
    frame_count = int(seg_feats.shape[0])
    if pose_seq is not None:
        seg_detected = pose_seq.detected[start_i : end_i + 1]
        detected_frames = int(np.count_nonzero(seg_detected))
        detected_ratio = float(detected_frames / max(1, frame_count))
        seg_lm = pose_seq.landmarks[start_i : end_i + 1]
        valid_mask = (
            derive_valid_mask(seg_lm, DEFAULT_VALID_CONF_THR) & seg_detected[:, None]
        )
        core_valid_frames = int(
            np.count_nonzero(np.all(valid_mask[:, _CORE_INDICES], axis=1))
        )
        core_valid_ratio = float(core_valid_frames / max(1, frame_count))
        limb_valid_ratio = (
            float(np.mean(valid_mask[:, _LIMB_INDICES])) if frame_count > 0 else 0.0
        )
    else:
        nonzero = (
            int(np.count_nonzero(np.max(np.abs(seg_feats), axis=(1, 2)) >= 1e-6))
            if frame_count > 0
            else 0
        )
        detected_frames = nonzero
        detected_ratio = float(nonzero / max(1, frame_count))
        core_valid_frames = nonzero
        core_valid_ratio = detected_ratio
        limb_valid_ratio = detected_ratio

    warnings: list[str] = []
    if limb_valid_ratio < 0.6:
        warnings.append(
            f"四肢关键点有效率偏低（{limb_valid_ratio * 100:.1f}% < 60%），可能存在遮挡或模糊"
        )
    if detected_ratio < 0.9 or core_valid_ratio < 0.85:
        warnings.append(
            f"局部遮挡或未检出（检出率 {detected_ratio * 100:.1f}%，躯干核心有效率 {core_valid_ratio * 100:.1f}%）"
        )
    if frame_count >= 2:
        seg_energy = motion_energy(seg_feats.reshape(frame_count, -1))
        if seg_energy.size > 0 and float(np.percentile(seg_energy, 90)) < 0.01:
            warnings.append("截取片段几乎静止，动作幅度和运动能量极低")

    report = TemplateQualityReport(
        frame_count=frame_count,
        detected_frames=detected_frames,
        detected_ratio=detected_ratio,
        core_valid_frames=core_valid_frames,
        core_valid_ratio=core_valid_ratio,
        limb_valid_ratio=limb_valid_ratio,
        warnings=tuple(warnings),
    )
    if validate_quality:
        if frame_count < MIN_TEMPLATE_FRAMES:
            raise TemplateQualityError(
                f"截取片段过短（{frame_count} 帧 < {MIN_TEMPLATE_FRAMES} 帧），无法生成可靠模板",
                report=report,
            )
        if float(np.max(np.abs(seg_feats))) < 1e-6:
            raise TemplateQualityError(
                "截取区间未提取到有效人体骨架（全为空帧）",
                report=report,
            )
        if pose_seq is not None and (
            detected_ratio < MIN_DETECTED_RATIO
            or core_valid_ratio < MIN_CORE_VALID_RATIO
        ):
            raise TemplateQualityError(
                "截取区间骨架质量不达标："
                f"检出率 {detected_ratio * 100:.1f}%（需 ≥ {MIN_DETECTED_RATIO * 100:.0f}%），"
                f"躯干核心有效率 {core_valid_ratio * 100:.1f}%（需 ≥ {MIN_CORE_VALID_RATIO * 100:.0f}%）",
                report=report,
            )
    return report


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
    """Analyze whether the single-action auto crop is unsafe for a long video."""
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
            (start, end) for start, end in runs if end - start + 1 >= min_active_frames
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
