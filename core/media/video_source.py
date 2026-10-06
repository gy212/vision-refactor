# -*- coding: utf-8 -*-
"""离线视频解码的唯一入口：``open_video`` 按 ``DecodeOptions`` 逐帧产出 ``Frame``（设计文档 §5.1）。

默认值复现正式评估（action_compare、rule_scoring、tech_eval）的读法：逐帧 ``read()``、不缩放、
``floor`` 时间戳、读取失败即视为结束。练习分析 v4 的长边缩放、``round_monotonic`` 时间戳、
抽帧与严格解码都是同一函数的参数。``cv2.VideoCapture(`` 只允许出现在 ``core/media/``。
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from core.foundation.tasks import CancelToken

DEFAULT_FPS = 30.0
NO_END = 10**9  # 帧数未知时的"读到结尾"


class VideoError(RuntimeError):
    """解码失败的基类。"""


class VideoOpenError(VideoError):
    """视频无法打开。"""

    def __init__(self, path: Path | str) -> None:
        super().__init__(f"无法打开视频：{path}")
        self.path = Path(str(path))


class VideoFpsUnavailable(VideoError):
    """严格模式下容器没有给出有效帧率。"""


class UndecodableFrame(VideoError):
    """严格模式下某一帧 grab 成功但解码失败。"""


@dataclass(frozen=True)
class DecodeOptions:
    start: int = 0
    end: int | None = None              # 含端点；None 表示读到结尾
    sample_fps: float | None = None     # None/0：逐帧；否则步长 ceil(fps/sample_fps)，末帧总会保留
    max_edge: int | None = None         # 长边超过时按比例缩小（练习分析 v4：1920）
    timestamps: Literal["floor", "round_monotonic"] = "floor"
    strict: bool = False                # True：帧率无效或帧不可解码时报错；False：读取失败即结束
    seek: bool | Literal["verified"] = False
    # True：先定位到 start（分段并行）；"verified"：定位后核对位置，不准就重开并顺序跳过（考试裁剪）；
    # False：从 0 读并跳过 start 之前的帧


@dataclass(frozen=True)
class VideoMeta:
    path: Path
    fps: float                # 正式路径的帧率：容器值为 0 时取 30
    fps_reported: float       # 容器报告的原值（可能为 0 或 NaN）
    frame_count: int          # 容器报告的帧数，未知为 0
    width: int
    height: int


@dataclass(frozen=True)
class Frame:
    index: int
    t_ms: int
    bgr: np.ndarray
    source_size: tuple[int, int]   # 缩放前的 (width, height)


def _probe(capture, path: Path) -> VideoMeta:
    reported = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    count = int(count) if math.isfinite(count) else 0
    return VideoMeta(
        path=path,
        fps=reported or DEFAULT_FPS,
        fps_reported=reported,
        frame_count=count,
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    )


def _resize(frame: np.ndarray, max_edge: int | None) -> np.ndarray:
    if max_edge is None:
        return frame
    h, w = frame.shape[:2]
    if max(h, w) <= max_edge:
        return frame
    scale = max_edge / max(h, w)
    return cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))))


class VideoSource:
    """一次打开的离线视频。迭代产出 ``Frame``；用完 ``close()``，或用 ``with``。"""

    def __init__(self, path: Path, options: DecodeOptions, cancel: CancelToken | None) -> None:
        self.path = Path(path)
        self.options = options
        self._cancel = cancel
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            raise VideoOpenError(self.path)
        try:
            self.meta = _probe(self._capture, self.path)
            if options.strict and (not math.isfinite(self.meta.fps_reported) or self.meta.fps_reported <= 0):
                raise VideoFpsUnavailable("视频帧率不可用")
        except BaseException:
            self._capture.release()
            raise
        self.frames_scanned = 0   # 实际 grab/read 到的帧数（含抽帧跳过的）

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._capture.release()

    def stride(self) -> int:
        sample = self.options.sample_fps
        return max(1, math.ceil(self.meta.fps / sample)) if sample else 1

    def _timestamp(self, index: int, previous: int) -> int:
        fps = self.meta.fps
        if self.options.timestamps == "round_monotonic":
            return max(previous + 1, round(index * 1000 / fps))
        return int(index * 1000.0 / fps)

    def __iter__(self) -> Iterator[Frame]:
        opts = self.options
        index = 0
        if opts.seek == "verified":
            if opts.start > 0:
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, float(opts.start))
                if int(self._capture.get(cv2.CAP_PROP_POS_FRAMES) or 0) == opts.start:
                    index = opts.start
                else:  # 定位不可靠：重开后从 0 顺序跳过
                    self._capture.release()
                    self._capture = cv2.VideoCapture(str(self.path))
                    if not self._capture.isOpened():
                        raise VideoOpenError(self.path)
        elif opts.seek:
            # 定位失败时 OpenCV 通常从 0 继续读；帧序号仍按 start 计（与旧分段实现一致）。
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, float(opts.start))
            index = opts.start
        end = NO_END if opts.end is None else opts.end
        stride = self.stride()
        total = self.meta.frame_count
        use_grab = opts.strict or stride > 1
        previous = -1
        while index <= end:
            if use_grab:
                if not self._capture.grab():
                    return
            else:
                ok, frame = self._capture.read()
                if not ok:
                    return
            self.frames_scanned += 1
            if self._cancel is not None:
                self._cancel.raise_if_cancelled()
            if use_grab:
                if index % stride != 0 and index != total - 1:
                    index += 1
                    continue
                ok, frame = self._capture.retrieve()
                if not ok or frame is None:
                    if opts.strict:
                        raise UndecodableFrame("视频存在不可解码的帧")
                    return
            if index >= opts.start:
                shape = getattr(frame, "shape", None)
                source_size = (int(shape[1]), int(shape[0])) if shape is not None else (0, 0)
                previous = self._timestamp(index, previous)
                yield Frame(index, previous, _resize(frame, opts.max_edge), source_size)
            index += 1

    def read_at(self, index: int) -> np.ndarray | None:
        """定位到第 ``index`` 帧读一帧；失败返回 None（报告证据帧、结果页取帧）。"""
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, int(index))))
        ok, frame = self._capture.read()
        return frame if ok and frame is not None else None


def open_video(path: Path | str, options: DecodeOptions = DecodeOptions(),
               cancel: CancelToken | None = None) -> VideoSource:
    """打开离线视频；打不开抛 ``VideoOpenError``。"""
    return VideoSource(Path(str(path)), options, cancel)


def probe_video(path: Path | str) -> VideoMeta:
    """只读容器元数据（帧率、帧数、尺寸），不解码。"""
    with open_video(path) as source:
        return source.meta


def read_frame_at(path: Path | str, index: int) -> np.ndarray | None:
    """读取指定帧；视频打不开或该帧读不到时返回 None。"""
    try:
        source = open_video(path)
    except VideoOpenError:
        return None
    with source:
        return source.read_at(index)


def first_frame_ok(path: Path | str) -> bool:
    """能打开且第一帧可解码（录后处理的视频校验）。"""
    try:
        source = open_video(path)
    except VideoOpenError:
        return False
    with source:
        for frame in source:
            return getattr(frame.bgr, "size", 0) > 0
    return False
