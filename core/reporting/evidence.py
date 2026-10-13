# -*- coding: utf-8 -*-
"""练习证据帧：从历史录像取出指定帧并叠加相关关节，供结果页与图文报告共用（原 ``feedback_report`` 的截帧部分）。

解码只经 ``media.video_source.read_frame_at``；录像路径只经 ``SegmentLayout.resolve_contained`` 定位。
"""
from __future__ import annotations

import base64
import io
import math
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image

from core.media.segment_layout import SegmentLayout
from core.media.video_source import read_frame_at

# 规则代码 → 证据帧上高亮的关节
RULE_JOINTS: dict[str, set[int]] = {
    "stance_elbow": {11, 13, 15, 12, 14, 16},
    "stance_rear_guard": {11, 12, 13, 14, 15, 16, 9, 10, 23, 24},
    "guard_low": {11, 12, 13, 14, 15, 16, 9, 10},
    "guard_back": {11, 12, 13, 14, 15, 16, 23, 24, 0},
    "guard_elbow": {11, 12, 13, 14},
    "shoulder_level": {11, 12, 13, 14, 15, 16},
    "guard_recover": {11, 12, 13, 14, 15, 16, 9, 10},
    "stance_knee": {23, 24, 25, 26, 27, 28},
    "stance_toes": {27, 28, 29, 30, 31, 32},
    "stance_torso": {11, 12, 23, 24},
    "foot_flexed": {25, 26, 27, 28, 29, 30, 31, 32},
    "hip_turn": {11, 12, 23, 24, 25, 26},
    "arm_swing": {11, 12, 13, 14, 15, 16},
}

SKELETON_BONES = [
    (11, 13), (13, 15), (12, 14), (14, 16), (11, 12),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    (9, 10), (0, 9), (0, 10)
]


def is_valid_point(pt: Any) -> bool:
    """Validate that pt is a sequence of at least 2 finite numeric coordinates."""
    if not isinstance(pt, (list, tuple, np.ndarray)):
        return False
    if len(pt) < 2:
        return False
    try:
        px = float(pt[0])
        py = float(pt[1])
        return math.isfinite(px) and math.isfinite(py)
    except (ValueError, TypeError, IndexError):
        return False


def resolve_record_video_path(history_root: Path, record_id: str, view: str, videos: dict) -> Path | None:
    """历史记录里某一视角的受管视频：记录 id 合法，且逐级不越出历史目录、不经符号链接。"""
    if not isinstance(record_id, str) or not re.fullmatch(r"[0-9a-f]{32}", record_id):
        return None
    filename = videos.get(view)
    if not filename or not isinstance(filename, str):
        return None
    return SegmentLayout.resolve_contained(history_root, record_id, filename)


def _pixel(pt: Any, h: int) -> tuple[int, int]:
    """高度归一坐标 → 像素：x、y 都乘画面高度。"""
    return int(round(float(pt[0]) * h)), int(round(float(pt[1]) * h))


def _draw_bones(frame: np.ndarray, pts: np.ndarray, vmask: list, highlight_joints: set[int]) -> None:
    h, w = frame.shape[:2]
    for j1, j2 in SKELETON_BONES:
        if (j1 in highlight_joints and j2 in highlight_joints and
                j1 < len(vmask) and j2 < len(vmask) and vmask[j1] and vmask[j2] and
                j1 < len(pts) and j2 < len(pts)):
            p1 = pts[j1]
            p2 = pts[j2]
            if is_valid_point(p1) and is_valid_point(p2):
                x1, y1 = _pixel(p1, h)
                x2, y2 = _pixel(p2, h)
                if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                    cv2.line(frame, (x1, y1), (x2, y2), (235, 99, 37), max(2, h // 250))


def _draw_joints(frame: np.ndarray, pts: np.ndarray, vmask: list, highlight_joints: set[int] | None) -> None:
    h, w = frame.shape[:2]
    for idx in range(min(len(pts), len(vmask))):
        if not vmask[idx] or not is_valid_point(pts[idx]):
            continue
        px, py = _pixel(pts[idx], h)
        if not (0 <= px < w and 0 <= py < h):
            continue
        if highlight_joints and idx in highlight_joints:
            rad = max(4, h // 140)
            cv2.circle(frame, (px, py), rad + 2, (255, 255, 255), 2)
            cv2.circle(frame, (px, py), rad, (14, 64, 235), -1)
        else:
            rad = max(2, h // 300)
            cv2.circle(frame, (px, py), rad, (0, 220, 100), -1)


def extract_evidence_frame(
    video_path: Path,
    frame_idx: int,
    landmarks: list | None = None,
    valid_mask: list | None = None,
    highlight_joints: set[int] | None = None
) -> np.ndarray | None:
    """Extract single frame from OpenCV video, paint height-normalized landmarks and skeleton.

    Coordinates scaling strictly enforces height-normalization:
    pixel_x = round(point[0] * h)
    pixel_y = round(point[1] * h)
    """
    if not isinstance(video_path, Path):
        video_path = Path(video_path)
    if not video_path.is_file() or video_path.is_symlink():
        return None
    try:
        frame = read_frame_at(video_path, int(frame_idx))
    except Exception:
        return None
    if frame is None:
        return None
    if landmarks is not None and len(landmarks) > 0:
        pts = np.asarray(landmarks)
        vmask = valid_mask if valid_mask is not None else [True] * len(pts)
        if highlight_joints:
            _draw_bones(frame, pts, vmask, highlight_joints)
        _draw_joints(frame, pts, vmask, highlight_joints)
    return frame


def frame_to_data_uri(frame: np.ndarray, max_edge: int = 480, quality: int = 80) -> str:
    """Convert OpenCV BGR image to JPEG Base64 Data URI."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    if max(pil_img.size) > max_edge:
        pil_img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64_str}"
