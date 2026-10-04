# -*- coding: utf-8 -*-
"""
YOLO 测试用 fake 工具（YOLO 迁移 Issue #7 / S2）。

为什么需要它
------------
Issue #7 的映射 / 契约逻辑测试**不得**下载真实模型或读真实视频、不得联网。
本模块提供：
  - ``make_coco17``：合成一份 ``(17,2)`` 归一化坐标 + ``(17,)`` 置信度。
  - ``FakeYoloResult``：模拟 ultralytics 单帧结果对象（``keypoints.xyn``/``conf``、
    ``boxes.xywh``/``conf``），供 ``extract_persons`` / ``yolo_result_to_arrays`` 解析。
  - ``FakeYoloAdapter``：替身 adapter（不加载 ultralytics），按预置的逐帧结果驱动
    ``extract_yolo_landmark_series``，从而在无模型/无视频下测试序列层契约。
  - ``FakeCapture`` + ``patch_cv2_capture``：把 ``cv2.VideoCapture`` 换成回放固定帧数的
    假对象，让序列提取层在无真实视频下也能跑完。
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def make_coco17(
    *,
    conf_value: float = 0.9,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成一份合成 COCO17：``xy[17,2]`` 归一化坐标 + ``conf[17]`` 置信度。"""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.1, 0.9, size=(17, 2)).astype(np.float32)
    conf = np.full((17,), float(conf_value), dtype=np.float32)
    return xy, conf


class _FakeKeypoints:
    def __init__(self, xyn: np.ndarray, conf: np.ndarray | None) -> None:
        # (N,17,2) 与 (N,17)
        self.xyn = np.asarray(xyn, dtype=np.float32)
        self.conf = None if conf is None else np.asarray(conf, dtype=np.float32)
        self.data = self.xyn


class _FakeBoxes:
    def __init__(self, xywh: np.ndarray | None, conf: np.ndarray | None) -> None:
        self.xywh = None if xywh is None else np.asarray(xywh, dtype=np.float32)
        self.conf = None if conf is None else np.asarray(conf, dtype=np.float32)


@dataclass
class FakeYoloResult:
    """模拟 ultralytics 单帧 result（仅暴露解析所需字段）。"""

    keypoints: Any = None
    boxes: Any = None

    @classmethod
    def single(
        cls,
        xy: np.ndarray,
        conf: np.ndarray | None,
        *,
        box_xywh: tuple[float, float, float, float] | None = (0.5, 0.5, 0.4, 0.8),
        box_conf: float = 0.9,
    ) -> "FakeYoloResult":
        xyn = np.asarray(xy, dtype=np.float32)[None, :, :]
        kconf = None if conf is None else np.asarray(conf, dtype=np.float32)[None, :]
        bxywh = None if box_xywh is None else np.asarray([box_xywh], dtype=np.float32)
        bconf = None if box_conf is None else np.asarray([box_conf], dtype=np.float32)
        return cls(keypoints=_FakeKeypoints(xyn, kconf), boxes=_FakeBoxes(bxywh, bconf))

    @classmethod
    def multi(
        cls,
        persons: list[tuple[np.ndarray, np.ndarray | None, tuple[float, float, float, float]]],
    ) -> "FakeYoloResult":
        """多人结果：persons = [(xy[17,2], conf[17], box_xywh), ...]。"""
        xyn = np.stack([np.asarray(p[0], dtype=np.float32) for p in persons], axis=0)
        if all(p[1] is not None for p in persons):
            kconf = np.stack([np.asarray(p[1], dtype=np.float32) for p in persons], axis=0)
        else:
            kconf = None
        bxywh = np.asarray([p[2] for p in persons], dtype=np.float32)
        bconf = np.asarray([0.9] * len(persons), dtype=np.float32)
        return cls(keypoints=_FakeKeypoints(xyn, kconf), boxes=_FakeBoxes(bxywh, bconf))

    @classmethod
    def empty(cls) -> "FakeYoloResult":
        """无人帧：keypoints 为空。"""
        return cls(keypoints=_FakeKeypoints(np.zeros((0, 17, 2), np.float32), None), boxes=None)


@dataclass
class FakeYoloAdapter:
    """替身 adapter：按预置逐帧结果驱动序列层，不加载 ultralytics。

    ``frames`` 每项是一个 ``FakeYoloResult``；``infer_arrays`` 依次消费。
    复用真实解析 / tracker 逻辑（从 core.yolo_adapter 导入），只替换「模型推理」。
    """

    frames: list[Any]
    valid_conf_thr: float = 0.5
    confidence_kind: str = "yolo_conf"
    model_name: str = "fake-yolo.pt"
    _cursor: int = 0
    _tracker: Any = None
    last_num_persons: int = 0

    def __post_init__(self) -> None:
        from core.yolo_adapter import SingleTargetTracker

        self._tracker = SingleTargetTracker()

    def reset_tracker(self) -> None:
        self._cursor = 0
        self._tracker.reset()
        self.last_num_persons = 0

    def infer_arrays(self, _frame: np.ndarray):
        from core.yolo_adapter import yolo_result_to_arrays

        res = self.frames[self._cursor] if self._cursor < len(self.frames) else None
        self._cursor += 1
        row, valid, num_persons, _idx = yolo_result_to_arrays(
            res, valid_conf_thr=self.valid_conf_thr
        )
        self.last_num_persons = num_persons
        track_id = self._tracker.update(num_persons >= 1)
        return row, valid, num_persons, track_id


@dataclass
class FakeCapture:
    """模拟 cv2.VideoCapture：回放固定帧数的占位帧。"""

    n_frames: int
    fps: float = 30.0
    width: int = 1280
    height: int = 720
    _i: int = 0

    def isOpened(self) -> bool:  # noqa: N802 (cv2 API 名)
        return True

    def get(self, prop: int) -> float:
        import cv2

        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.n_frames)
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def read(self):
        if self._i >= self.n_frames:
            return False, None
        self._i += 1
        return True, np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


@contextlib.contextmanager
def patch_cv2_capture(n_frames: int, *, fps: float = 30.0):
    """临时把 core.yolo_adapter 内的 cv2.VideoCapture 换成 FakeCapture。

    extract_yolo_landmark_series 内部 ``import cv2`` 后调用 ``cv2.VideoCapture``，
    故 patch 全局 cv2 模块的该符号即可。
    """
    import cv2

    orig = cv2.VideoCapture

    def _factory(_path):
        return FakeCapture(n_frames=n_frames, fps=fps)

    cv2.VideoCapture = _factory  # type: ignore[assignment]
    try:
        yield
    finally:
        cv2.VideoCapture = orig  # type: ignore[assignment]
