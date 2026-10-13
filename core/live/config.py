# -*- coding: utf-8 -*-
"""实时会话的参数：开始时冻结的 ``SessionConfig`` 与运行中可调的 ``LiveControls``（设计文档 §8.1）。

- ``SessionConfig`` 由入口层在点击"开始"时一次算好，``camera_session``/``video_session`` 按 §8.2 的表
  把四个旧循环的差异变成字段默认值；runner 和 stages 只读它，不再读 Tk 变量或模式标志。
- ``LiveControls`` 的每一项都是带版本号的线程安全值（``Versioned``）：UI 线程写，runner 每帧读。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generic, Literal, Mapping, TypeVar

from core.media import PRIMARY, ROLES, SECONDARY, CaptureSpec

T = TypeVar("T")

Roi = tuple[float, float, float, float]
PresenceProbe = Callable[[object, Roi], bool]   # (pose landmarks, ROI) -> 人是否在 ROI 内

DEFAULT_ROI: Roi = (0.2, 0.1, 0.8, 0.95)
DEFAULT_OCCUPANCY_STRIDE = 3
NOMINAL_CAMERA_FPS = 30.0


@dataclass(frozen=True)
class SessionConfig:
    """一次会话的冻结参数。``cameras`` 与 ``video_path`` 二选一。"""

    cameras: Mapping[str, CaptureSpec] | None = None
    video_path: Path | None = None
    pose_variant: str = "full"
    enable_hands: bool = True
    delegate: str = "cpu"
    workers: int = 1
    parallel: bool = False            # True：ParallelPoseEngine（IMAGE 模式）；摄像头满则丢帧，视频保序
    inference: bool = True            # False：不构造引擎，预览与录制都是裸帧（摄像头骨架关）
    record_overlay: bool = True       # 录制写叠加后的帧（骨架、HUD）还是转正后的原始帧
    record_hud: bool = True           # HUD 在写录像之前画上（并行视频的旧行为是写完再画）
    smoothing: bool = False           # 预览平滑（PreviewLandmarkSmoother）
    online_match: bool = False        # 在线动作识别（OnlineActionMatcher）
    loading_raw_preview: bool = False  # 构造模型期间由裸帧泵发布预览（D-8）
    occupancy: bool = False           # 是否具备站位检测能力（骨架关的双摄会话）
    hud: Literal["tk", "cli"] = "tk"
    hud_hint: bool = True             # HUD 底部的"点击停止"/"按 q 退出"提示
    hud_title: str = "FPS"
    nominal_fps: float = NOMINAL_CAMERA_FPS
    latest_frame: bool = False        # 采集与推理分线程、推理只取最新帧（CLI 的 latest-frame 冒烟）
    limit_frames: int | None = None   # 最多采集这么多帧（冒烟用）

    def __post_init__(self) -> None:
        if (self.cameras is None) == (self.video_path is None):
            raise ValueError("SessionConfig needs exactly one of cameras / video_path")
        if self.cameras is not None and (PRIMARY not in self.cameras or set(self.cameras) - set(ROLES)):
            raise ValueError("cameras needs a primary role and only primary/secondary")

    @property
    def is_file(self) -> bool:
        return self.video_path is not None

    @property
    def roles(self) -> tuple[str, ...]:
        if self.cameras is None:
            return (PRIMARY,)
        return tuple(role for role in ROLES if role in self.cameras)

    @property
    def dual(self) -> bool:
        return self.cameras is not None and SECONDARY in self.cameras


def camera_session(
    cameras: Mapping[str, CaptureSpec],
    *,
    pose_variant: str = "full",
    enable_hands: bool = True,
    workers: int = 1,
    skeleton: bool = False,
    online_match: bool = True,
    loading_raw_preview: bool = True,
    hud: Literal["tk", "cli"] = "tk",
    delegate: str = "cpu",
    smoothing: bool = True,
    dual_smoothing: bool = False,
    dual_online_match: bool = False,
) -> SessionConfig:
    """摄像头会话（§8.2 表）：骨架关时不推理；单摄骨架开时平滑与在线识别按参数（Tk 默认开，CLI 关）；
    多 worker 且骨架开时并行推理。

    双摄的平滑与在线识别保持关闭（D-7），``dual_smoothing``/``dual_online_match`` 是保留的配置项。
    并行单摄的引擎与摄像头同时启动，没有加载期裸帧。站位检测只在骨架关的双摄会话里可用。
    """
    dual = SECONDARY in cameras
    parallel = (not dual) and workers > 1 and skeleton
    return SessionConfig(
        cameras=dict(cameras),
        pose_variant=pose_variant,
        enable_hands=enable_hands,
        delegate=delegate,
        workers=max(1, int(workers)),
        parallel=parallel,
        inference=skeleton,
        record_overlay=skeleton or not dual,   # 单摄的录像一直带 HUD；双摄骨架关时录原始帧
        smoothing=skeleton and (dual_smoothing if dual else smoothing),
        online_match=skeleton and (dual_online_match if dual else online_match),
        loading_raw_preview=skeleton and loading_raw_preview and not parallel,
        occupancy=dual and not skeleton,
        hud=hud,
        hud_hint=hud == "cli" or not parallel,
    )


def video_session(
    path: Path | str,
    *,
    pose_variant: str = "full",
    enable_hands: bool = True,
    workers: int = 1,
    hud: Literal["tk", "cli"] = "tk",
    delegate: str = "cpu",
) -> SessionConfig:
    """视频文件会话：始终推理（逐帧 ``annotate``），不平滑、不做在线识别；多 worker 时并行保序。"""
    return SessionConfig(
        video_path=Path(path),
        pose_variant=pose_variant,
        enable_hands=enable_hands,
        delegate=delegate,
        workers=max(1, int(workers)),
        parallel=workers > 1,
        record_hud=hud == "cli" or workers <= 1,   # Tk 并行视频录原始帧；CLI 导出一直带 HUD
        hud=hud,
        hud_hint=hud == "cli" or workers <= 1,
    )


class Versioned(Generic[T]):
    """线程安全的值，每次 ``set`` 版本号加一。"""

    def __init__(self, value: T) -> None:
        self._lock = threading.Lock()
        self._value = value
        self._version = 0

    def get(self) -> T:
        with self._lock:
            return self._value

    def set(self, value: T) -> int:
        with self._lock:
            self._value = value
            self._version += 1
            return self._version

    @property
    def version(self) -> int:
        with self._lock:
            return self._version


@dataclass(frozen=True)
class OccupancySettings:
    armed_by: frozenset[str] = frozenset()   # 谁在用站位检测（考试、学生练习）；非空即启用
    roi: Roi = DEFAULT_ROI
    stride: int = DEFAULT_OCCUPANCY_STRIDE

    @property
    def armed(self) -> bool:
        return bool(self.armed_by)


@dataclass
class LiveControls:
    """会话运行中可调的参数（§8.1）：每路旋转每帧读取，站位检测的启用与 ROI 每帧读取。"""

    rotation: dict[str, Versioned[int]] = field(
        default_factory=lambda: {role: Versioned(0) for role in ROLES}
    )
    occupancy: Versioned[OccupancySettings] = field(default_factory=lambda: Versioned(OccupancySettings()))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def rotation_of(self, role: str) -> int:
        return self.rotation[role].get()

    def arm_occupancy(self, owner: str, armed: bool) -> None:
        """``owner`` 是启用方的名字（如 ``exam``、``practice``）；任一方启用即在推理关的会话里采样。"""
        with self._lock:
            current = self.occupancy.get()
            owners = current.armed_by | {owner} if armed else current.armed_by - {owner}
            if owners != current.armed_by:
                self.occupancy.set(OccupancySettings(frozenset(owners), current.roi, current.stride))

    def set_roi(self, roi: Roi) -> None:
        with self._lock:
            current = self.occupancy.get()
            value = tuple(float(x) for x in roi)
            self.occupancy.set(OccupancySettings(current.armed_by, value, current.stride))  # type: ignore[arg-type]


__all__ = [
    "DEFAULT_OCCUPANCY_STRIDE",
    "DEFAULT_ROI",
    "LiveControls",
    "NOMINAL_CAMERA_FPS",
    "OccupancySettings",
    "PresenceProbe",
    "Roi",
    "SessionConfig",
    "Versioned",
    "camera_session",
    "video_session",
]
