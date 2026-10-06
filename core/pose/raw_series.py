# -*- coding: utf-8 -*-
"""正式评估的 Pose33 原始序列：``extract_pose_raw``/``extract_pose_raw_series``/``slice_pose_raw_series``。

是 ``extract_pose_sequence(FORMAL)`` 加 ``raw(HOLD_LAST_LEGACY)`` 视图的薄适配，输出结构与迁移前
``rule_scoring`` 的同名函数完全相同（D-2 有结论前维持 ``HOLD_LAST_LEGACY``）。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from core.media.video_source import NO_END, VideoOpenError, probe_video
from core.pose.extract import FORMAL, extract_pose_sequence, formal_engine
from core.pose.sequence import GapPolicy, PoseSequence
from core.pose.validity import DEFAULT_VALID_CONF_THR, MEDIAPIPE_VALIDITY_POLICY, derive_valid_mask

LANDMARK_LAYOUT = "pose33_normalized_xyzw(visibility)"


@dataclass(frozen=True)
class Pose33RawSeries:
    landmarks: np.ndarray
    meta: dict


def formal_end_frame(video: Path, end_frame: int | None) -> int:
    """正式路径的读取终点（含）：显式给出就用它，否则用容器帧数 - 1，帧数未知时读到结尾。"""
    if end_frame is not None:
        return max(0, int(end_frame))
    try:
        n_frames = probe_video(video).frame_count
    except VideoOpenError:
        n_frames = 0
    return n_frames - 1 if n_frames > 0 else NO_END


def formal_sequence(video: Path, pose_variant: str, end_frame: int) -> PoseSequence:
    """从第 0 帧顺序跟踪读到 ``end_frame``（含）。"""
    profile = replace(FORMAL, decode=replace(FORMAL.decode, end=int(end_frame)))
    return extract_pose_sequence(video, formal_engine(pose_variant), profile)


def raw_meta(seq: PoseSequence, *, video: Path, pose_variant: str, start_frame: int,
             landmarks: np.ndarray) -> dict:
    """``extract_pose_raw`` 的 meta（含按 0.5 阈值推导的 ``valid_mask``）。"""
    m = seq.meta
    return {
        "video": str(video),
        "fps": float(m.fps),
        "frame_count": int(m.frame_count),
        "width": int(m.width),
        "height": int(m.height),
        "pose_variant": str(pose_variant),
        "start_frame": int(start_frame),
        "end_frame": int(start_frame + landmarks.shape[0] - 1),
        "landmark_layout": LANDMARK_LAYOUT,
        # valid_mask 契约（YOLO 迁移 Issue #5）：MediaPipe 侧 0.5 视为已标定。
        "validity_policy": MEDIAPIPE_VALIDITY_POLICY,
        "valid_conf_thr": float(DEFAULT_VALID_CONF_THR),
        "valid_mask": derive_valid_mask(landmarks, DEFAULT_VALID_CONF_THR),
    }


def _prefix_series(video: Path, *, pose_variant: str, end_frame: int) -> Pose33RawSeries:
    seq = formal_sequence(video, pose_variant, end_frame)
    if len(seq) == 0:
        raise RuntimeError(f"未能从视频提取姿态：{video}")
    landmarks = seq.raw(GapPolicy.HOLD_LAST_LEGACY).copy()
    meta = raw_meta(seq, video=video, pose_variant=pose_variant, start_frame=0, landmarks=landmarks)
    return Pose33RawSeries(landmarks=landmarks, meta=meta)


def slice_pose_raw_series(
    series: Pose33RawSeries,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Return an ``extract_pose_raw``-compatible slice from a pre-extracted series."""
    base = np.asarray(series.landmarks, dtype=np.float32)
    if base.ndim != 3 or base.shape[1:] != (33, 4):
        raise ValueError(f"Pose33RawSeries.landmarks 形状必须为 (T,33,4)，实际 {base.shape}")
    if base.shape[0] == 0:
        raise RuntimeError(f"未能从视频提取姿态：{series.meta.get('video', '')}")

    start_i = int(start_frame) if start_frame is not None else 0
    end_i = int(end_frame) if end_frame is not None else int(base.shape[0] - 1)
    start_i = max(0, start_i)
    end_i = max(start_i, end_i)
    end_i = min(end_i, int(base.shape[0] - 1))
    if start_i >= int(base.shape[0]):
        raise RuntimeError(f"未能从视频提取姿态：{series.meta.get('video', '')}")

    landmarks = base[start_i : end_i + 1].astype(np.float32, copy=False)
    meta = dict(series.meta)
    meta["start_frame"] = int(start_i)
    meta["end_frame"] = int(start_i + landmarks.shape[0] - 1)
    meta["valid_mask"] = derive_valid_mask(landmarks, float(meta.get("valid_conf_thr") or DEFAULT_VALID_CONF_THR))
    return landmarks, meta


def extract_pose_raw_series(video_path: Path | str, *, pose_variant: str) -> Pose33RawSeries:
    """Extract full-video raw Pose33 once for downstream read-only slicing."""
    video = Path(video_path)
    return _prefix_series(video, pose_variant=pose_variant, end_frame=formal_end_frame(video, None))


def extract_pose_raw(
    video_path: Path | str,
    *,
    pose_variant: str,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    提取原始 Pose33 关键点（标准化坐标）。
    返回 (landmarks[T,33,4], meta)。
    """
    video = Path(video_path)
    start_i = max(0, int(start_frame) if start_frame is not None else 0)
    end_i = max(start_i, formal_end_frame(video, end_frame))
    series = _prefix_series(video, pose_variant=pose_variant, end_frame=end_i)
    return slice_pose_raw_series(series, start_frame=start_i, end_frame=end_i)
