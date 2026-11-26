# -*- coding: utf-8 -*-
"""模板建库的录制目录发现、动作名校验、落盘路径规划与骨架质量体检（L4 公共服务）。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.foundation.paths import templates_dir
from core.foundation.signal import motion_energy
from core.pose.validity import DEFAULT_VALID_CONF_THR, derive_valid_mask

MIN_TEMPLATE_FRAMES = 8
MIN_DETECTED_RATIO = 0.6
MIN_CORE_VALID_RATIO = 0.6
VIEW_SUFFIXES: dict[str, str] = {"front": "_正面", "side": "_侧面", "single": ""}

_CORE_INDICES = (11, 12, 23, 24)
_LIMB_INDICES = (13, 14, 15, 16, 25, 26, 27, 28)
_ILLEGAL_NAME_CHARS = set('/\\:*?"<>|')


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


def _first_existing(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def discover_recording_videos(path: str | Path) -> RecordingPair:
    """从录制片段目录（查找 front/side .mp4/.avi）或单视频文件解析待建库视频。"""
    p = Path(path)
    if p.is_dir():
        front = _first_existing(p, ("front.mp4", "front.avi"))
        side = _first_existing(p, ("side.mp4", "side.avi"))
        if front is None and side is None:
            raise ValueError(f"录制目录下未找到 front.mp4/.avi 或 side.mp4/.avi：{p}")
        return RecordingPair(front=front, side=side)
    if p.is_file():
        return RecordingPair(front=p, side=None)
    raise FileNotFoundError(f"视频文件或目录不存在：{p}")


def sanitize_action_name(action_name: str) -> str:
    """校验并规范化动作名，拒绝空白及非法路径字符。"""
    name = str(action_name or "").strip()
    if not name:
        raise ValueError("动作名称不能为空")
    if (
        name in {".", ".."}
        or ".." in name
        or any(ch in _ILLEGAL_NAME_CHARS or ord(ch) < 32 for ch in name)
    ):
        raise ValueError(f"动作名称包含非法路径字符：{action_name!r}")
    return name


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
