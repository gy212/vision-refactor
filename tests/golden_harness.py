# -*- coding: utf-8 -*-
"""
确定性回放 harness（YOLO 迁移 Issue #3 / S1 pose33_v3 golden 安全网）。

目的
----
S1 后续（#4 / #5）会改动以下热路径：
  - core.action_compare._extract_pose_features() 缺帧补零形状
  - core.pose_features.mirror_pose_features()
  - core.action_compare._select_representative_cycle()
  - 双模板关节误差统计
  - analysis.tech_eval / core.rule_scoring 的 visibility 有效性判据

为了把“当前行为”冻结成 golden，又要避免 MediaPipe 模型推理带来的不确定性
（同一视频多次推理结果可能有浮点漂移、且依赖模型权重），本 harness 把：
  1) cv2.VideoCapture  替换为读取“已保存的 (T,33,4) landmark 序列”的假对象；
  2) MediaPipePipeline 替换为直接把序列里的一帧 (33,4) 还原成 landmark 对象的假管线。

这样所有上层入口（compare_video_to_template / compare_video_to_dual_templates /
evaluate_video_full / extract_pose_raw / score_rules / extract_pose_and_view_scores）
都能在“真实代码路径 + 确定性输入”下运行，golden 只反映我们自己代码的行为，
不受模型版本影响。

注意：本 harness 仅用于测试与 golden 重生成，不会进入主链路。
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _norm_key(path) -> str:
    """统一 key：上层一律 cv2.VideoCapture(str(Path(video_path)))，这里同样归一化，
    避免 Windows 下 'golden://x' 被 Path 规整成 'golden:\\x' 后对不上。"""
    return str(Path(str(path)))

# 上层模块在各自命名空间里绑定了 MediaPipePipeline，需要逐个 patch。
import core.action_compare as _ac
import core.rule_scoring as _rs
import analysis.tech_eval as _te


# 路径 -> (landmarks[T,33,4], fps, width, height) 注册表。
# 假 VideoCapture 按 str(video_path) 查表回放。
_VIDEO_REGISTRY: dict[str, tuple[np.ndarray, float, int, int]] = {}


def register_video(path, landmarks: np.ndarray, *, fps: float = 30.0, width: int = 1280, height: int = 720) -> str:
    """把一段 (T,33,4) 序列注册为“可被 VideoCapture 打开的视频”，返回其 key。"""
    key = _norm_key(path)
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (33, 4):
        raise ValueError(f"landmarks 形状必须为 (T,33,4)，实际 {arr.shape}")
    _VIDEO_REGISTRY[key] = (arr, float(fps), int(width), int(height))
    return key


def clear_registry() -> None:
    _VIDEO_REGISTRY.clear()


@dataclass
class _FakeLandmark:
    x: float
    y: float
    z: float
    visibility: float


class _FakeFrame:
    """假帧：只携带该帧的 (33,4) landmark 行，供假管线直接还原。"""

    __slots__ = ("landmarks",)

    def __init__(self, landmarks_row: np.ndarray) -> None:
        self.landmarks = landmarks_row


class _FakeVideoCapture:
    def __init__(self, path) -> None:
        key = _norm_key(path)
        if key not in _VIDEO_REGISTRY:
            raise RuntimeError(f"golden harness 未注册的视频路径：{key}")
        self._seq, self._fps, self._w, self._h = _VIDEO_REGISTRY[key]
        self._idx = 0

    def isOpened(self) -> bool:  # noqa: N802 (匹配 cv2 接口)
        return True

    def read(self):
        if self._idx >= int(self._seq.shape[0]):
            return False, None
        frame = _FakeFrame(self._seq[self._idx])
        self._idx += 1
        return True, frame

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return float(self._fps)
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._seq.shape[0])
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        return 0.0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._idx = int(value)
        return True

    def release(self) -> None:
        return None


class _FakePipeline:
    """假 MediaPipe 管线：把 _FakeFrame 里的 (33,4) 行还原成 landmark 对象。"""

    def __init__(self, *, models_dir=None, cfg=None) -> None:
        self.cfg = cfg
        self.models_dir = models_dir

    def infer(self, frame, *, timestamp_ms=None):
        if not isinstance(frame, _FakeFrame):
            raise TypeError("golden harness 下收到非 _FakeFrame 帧，patch 未生效")
        row = frame.landmarks
        landmarks = [
            _FakeLandmark(float(row[i, 0]), float(row[i, 1]), float(row[i, 2]), float(row[i, 3]))
            for i in range(int(row.shape[0]))
        ]
        return landmarks, []


@contextlib.contextmanager
def replay_context():
    """在上下文内，所有上层入口的视频读取/姿态推理都走确定性回放。"""
    orig_capture = cv2.VideoCapture
    orig_ac_pipe = _ac.MediaPipePipeline
    orig_rs_pipe = _rs.MediaPipePipeline
    orig_te_pipe = _te.MediaPipePipeline
    try:
        cv2.VideoCapture = _FakeVideoCapture  # type: ignore[assignment]
        _ac.MediaPipePipeline = _FakePipeline  # type: ignore[assignment]
        _rs.MediaPipePipeline = _FakePipeline  # type: ignore[assignment]
        _te.MediaPipePipeline = _FakePipeline  # type: ignore[assignment]
        yield
    finally:
        cv2.VideoCapture = orig_capture  # type: ignore[assignment]
        _ac.MediaPipePipeline = orig_ac_pipe  # type: ignore[assignment]
        _rs.MediaPipePipeline = orig_rs_pipe  # type: ignore[assignment]
        _te.MediaPipePipeline = orig_te_pipe  # type: ignore[assignment]
