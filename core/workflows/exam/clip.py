# -*- coding: utf-8 -*-
"""考试片段的裁剪步骤：正侧帧数截齐与动作段裁剪，写出 ``front_exam`` / ``side_exam``，作为考试处理计划的第一步。

``ExamClipStep`` 实现 ``core.recording`` 的 ``Step`` 协议，由考试计划的构建方放进计划（``recording`` 不导入本模块）。
裁剪失败不中断片段：记一条 ``exam_clip_failed`` 说明，后续步骤继续用原始录像（与迁移前一致）。

内存约束：禁止把整段双路帧装进 list（1080p×30fps×60s 双路可达数十 GiB）。
一律流式：先扫 front 估能量/区间，再二次流式写出裁剪结果。P4 从 ``core/exam_clip.py`` 迁入。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.evaluation.segmentation import estimate_action_range  # 动作区间迁入 segmentation，考试裁剪与旧导入继续从这里取
from core.evaluation.segmentation import estimate_action_range_from_features as estimate_action_range_from_feature_seq  # noqa: F401
from core.foundation.tasks import CancelToken
from core.media.video_source import DecodeOptions, open_video
from core.recording import StepContext


# 关窗时为进行中的考试裁剪预留的等待时间（迁移前关窗对裁剪线程的追加预算）
EXAM_CLOSE_BUDGET_S = 30.0


@dataclass
class ClipResult:
    front_path: Path
    side_path: Path
    front_frames: int
    side_frames: int
    warnings: list[str] = field(default_factory=list)
    trim_start: int = 0
    trim_end: int = 0  # exclusive
    meta: dict[str, Any] = field(default_factory=dict)


def align_frame_counts(front_n: int, side_n: int) -> tuple[int, list[str]]:
    """返回截齐后的目标帧数 n = min(front, side)。"""
    warnings: list[str] = []
    front_n = int(front_n)
    side_n = int(side_n)
    if front_n <= 0 or side_n <= 0:
        return 0, ["empty_recording"]
    n = min(front_n, side_n)
    if front_n != side_n:
        warnings.append(
            f"frame_align: front={front_n} side={side_n} -> n={n} (trim longer tail)"
        )
    return n, warnings



def energy_from_gray_frames(frames: list[np.ndarray]) -> np.ndarray:
    """简易帧差能量，供无 pose 特征时裁剪（仅小样本/测试路径）。"""
    if len(frames) < 2:
        return np.zeros(0, dtype=np.float32)
    diffs: list[float] = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    for fr in frames[1:]:
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        d = float(np.mean((g - prev) ** 2) ** 0.5)
        diffs.append(d)
        prev = g
    return np.asarray(diffs, dtype=np.float32)


def _stream_energy_first_n(
    path: Path, n: int
) -> tuple[np.ndarray, int, float, tuple[int, int]]:
    """流式读取前 n 帧，只保留上一灰度帧算能量；返回 energy、实际帧数、fps、尺寸。"""
    diffs: list[float] = []
    prev: np.ndarray | None = None
    count = 0
    size = (0, 0)
    with open_video(path, DecodeOptions(end=n - 1)) as source:
        fps = source.meta.fps
        for frame in source:
            fr = frame.bgr
            if fr is None:
                break
            if count == 0:
                h, w = fr.shape[:2]
                size = (int(w), int(h))
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                diffs.append(float(np.mean((g - prev) ** 2) ** 0.5))
            prev = g
            count += 1
            del fr
    return np.asarray(diffs, dtype=np.float32), count, fps, size


def _count_frames_up_to(path: Path, n: int) -> tuple[int, float, tuple[int, int]]:
    """流式计数，不保留像素。"""
    count = 0
    size = (0, 0)
    with open_video(path, DecodeOptions(end=n - 1)) as source:
        fps = source.meta.fps
        for frame in source:
            if frame.bgr is None:
                break
            if count == 0:
                h, w = frame.bgr.shape[:2]
                size = (int(w), int(h))
            count += 1
    return count, fps, size


def _stream_write_range(
    path: Path,
    out_path: Path,
    *,
    start: int,
    end: int,
    fps: float,
    size_hint: tuple[int, int] | None = None,
) -> tuple[Path, int]:
    """流式写出 [start, end) 帧到 out_path；返回实际路径与写出帧数。"""
    if end <= start:
        raise RuntimeError("裁剪区间为空")
    writer = None
    out = out_path
    written = 0
    # 尽量 seek 并核对位置；不可靠时重开顺序跳过
    source = open_video(path, DecodeOptions(start=start, end=end - 1, seek="verified"))
    try:
        for frame in source:
            fr = frame.bgr
            if fr is None:
                break
            if writer is None:
                h, w = fr.shape[:2]
                if size_hint and size_hint[0] > 0 and size_hint[1] > 0:
                    w, h = size_hint
                out.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out), fourcc, float(fps), (w, h))
                if not writer.isOpened():
                    out = out_path.with_suffix(".avi")
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    writer = cv2.VideoWriter(str(out), fourcc, float(fps), (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建视频写出：{out_path}")
            writer.write(fr)
            written += 1
            del fr
    finally:
        if writer is not None:
            writer.release()
        source.close()
    if written <= 0:
        raise RuntimeError(f"裁剪写出 0 帧：{path}")
    return out, written


def _aligned_range(
    front_source: Path,
    side_source: Path,
    n: int,
    warnings: list[str],
    *,
    trim_head_s: float,
    trim_tail_s: float,
    use_pixel_energy: bool,
) -> tuple[int, int, int, float, tuple[int, int], tuple[int, int]]:
    """读出两路实际可读帧数并估计动作区间；返回 (start, end, 截齐帧数, fps, 正面尺寸, 侧面尺寸)。"""
    if use_pixel_energy:
        energy, front_n, fps_f, size_f = _stream_energy_first_n(front_source, n)
    else:
        front_n, fps_f, size_f = _count_frames_up_to(front_source, n)
        energy = None
    side_n, fps_s, size_s = _count_frames_up_to(side_source, n)
    if front_n <= 0 or side_n <= 0:
        raise RuntimeError("无法读取录制帧")

    n2 = min(front_n, side_n)
    if n2 < n:
        warnings.append(
            f"read_short: expected {n} got front={front_n} side={side_n}"
        )
    if energy is not None and energy.size > max(0, n2 - 1):
        energy = energy[: max(0, n2 - 1)]

    fps = float(fps_f or fps_s or 30.0)
    start, end, trim_warnings = estimate_action_range(
        n2,
        fps=fps,
        energy=energy,
        trim_head_s=trim_head_s,
        trim_tail_s=trim_tail_s,
    )
    warnings.extend(trim_warnings)
    if end <= start:
        start, end = 0, n2
        warnings.append("clip_empty_fallback_full")
    return start, end, n2, fps, size_f, size_s


def _write_aligned_pair(
    sources: tuple[Path, Path],
    out_dir: Path,
    sizes: tuple[tuple[int, int], tuple[int, int]],
    *,
    start: int,
    end: int,
    fps: float,
    warnings: list[str],
) -> tuple[Path, Path, int]:
    """写出两路 [start, end) 并保证写出帧数一致（不一致时按较短一路重写），返回 (正面, 侧面, 帧数)。"""
    names = ("front_exam.mp4", "side_exam.mp4")

    def write(i: int, stop: int) -> tuple[Path, int]:
        size = sizes[i]
        return _stream_write_range(
            sources[i], out_dir / names[i], start=start, end=stop, fps=fps,
            size_hint=size if size[0] > 0 else None,
        )

    (front_out, front_written), (side_out, side_written) = write(0, end), write(1, end)
    # 再次截齐：元数据与文件必须一致，禁止「假帧数 + 完整长文件」绕过 postprocess 校验
    written = min(front_written, side_written)
    if front_written != side_written:
        warnings.append(
            f"clip_write_mismatch: front={front_written} side={side_written} -> retrim {written}"
        )
        if front_written > written:
            front_out, front_written = write(0, start + written)
        if side_written > written:
            side_out, side_written = write(1, start + written)
        written = min(front_written, side_written)
        if front_written != side_written:
            # 仍不一致则失败，避免错误评分
            raise RuntimeError(
                f"clip re-trim still mismatched: front={front_written} side={side_written}"
            )
    return front_out, side_out, written


def prepare_exam_pair(
    front_source: Path,
    side_source: Path,
    *,
    front_frames: int,
    side_frames: int,
    out_dir: Path,
    trim_head_s: float = 0.5,
    trim_tail_s: float = 0.8,
    use_pixel_energy: bool = True,
) -> ClipResult:
    """截齐 + 裁剪，写出 front_exam / side_exam（流式，峰值内存 O(1 帧)）。"""
    front_source = Path(front_source)
    side_source = Path(side_source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n, warnings = align_frame_counts(front_frames, side_frames)
    if n <= 0:
        raise RuntimeError("录制帧数为 0，无法派发")
    start, end, n2, fps, size_f, size_s = _aligned_range(
        front_source, side_source, n, warnings,
        trim_head_s=trim_head_s, trim_tail_s=trim_tail_s, use_pixel_energy=use_pixel_energy,
    )
    front_out, side_out, written = _write_aligned_pair(
        (front_source, side_source), out_dir, (size_f, size_s),
        start=start, end=end, fps=fps, warnings=warnings,
    )
    return ClipResult(
        front_path=front_out,
        side_path=side_out,
        front_frames=written,
        side_frames=written,
        warnings=warnings,
        trim_start=start,
        trim_end=start + written,
        meta={
            "aligned_n": n2,
            "fps": fps,
            "source_front_frames": front_frames,
            "source_side_frames": side_frames,
            "streamed": True,
        },
    )


@dataclass(frozen=True)
class ExamClipStep:
    """考试计划的第一步（io 线）：截齐并裁剪正侧录像，后续步骤改用裁剪产物。

    裁剪不响应取消：关窗时也先做完裁剪，再由后续步骤发布取消终态（与迁移前关窗先等裁剪线程、再取消队列一致）。
    考试计划的关窗预算 ``EXAM_CLOSE_BUDGET_S`` 覆盖这段时间。
    """

    clip: Callable[..., ClipResult] = prepare_exam_pair
    lane: str = "io"

    def run(self, ctx: StepContext, cancel: CancelToken) -> None:
        if ctx.segment_dir is None:
            return None
        try:
            result = self.clip(
                ctx.sources["front"],
                ctx.sources["side"],
                front_frames=ctx.frames.get("front", 0),
                side_frames=ctx.frames.get("side", 0),
                out_dir=ctx.segment_dir,
            )
        except Exception as exc:  # noqa: BLE001 - 裁剪失败时退回原始录像，由后续步骤校验
            ctx.notes.append(f"exam_clip_failed: {exc}")
            return None
        ctx.sources["front"] = Path(result.front_path)
        ctx.sources["side"] = Path(result.side_path)
        ctx.frames["front"] = int(result.front_frames)
        ctx.frames["side"] = int(result.side_frames)
        ctx.notes.extend(result.warnings)
        return None
