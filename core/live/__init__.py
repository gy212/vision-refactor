# -*- coding: utf-8 -*-
"""实时会话（L4）的公共 API：会话配置与运行时控制、帧来源、逐帧 stages、``SessionRunner``、预览槽与会话管理。"""
from core.live.config import (
    DEFAULT_OCCUPANCY_STRIDE,
    DEFAULT_ROI,
    LiveControls,
    OccupancySettings,
    PresenceProbe,
    SessionConfig,
    Versioned,
    camera_session,
    video_session,
)
from core.live.inputs import CameraInput, FrameInput, VideoFileInput
from core.live.manager import IDLE, RUNNING, STARTING, STOPPING, LiveSessionManager
from core.live.metrics import StartupMetrics
from core.live.preview import PreviewFrame, PreviewStore, fit_preview_rgb
from core.live.runner import APP_CLOSE, FrameSink, RunningInfo, SessionEvent, SessionOutcome, SessionRunner
from core.live.stages import (
    FramePacket,
    PresenceSample,
    apply_rotation,
    build_stages,
    match_text,
    rotated_size,
)

__all__ = [
    "APP_CLOSE",
    "CameraInput",
    "DEFAULT_OCCUPANCY_STRIDE",
    "DEFAULT_ROI",
    "FrameInput",
    "FramePacket",
    "FrameSink",
    "IDLE",
    "LiveControls",
    "LiveSessionManager",
    "OccupancySettings",
    "PresenceProbe",
    "PresenceSample",
    "PreviewFrame",
    "PreviewStore",
    "RUNNING",
    "RunningInfo",
    "STARTING",
    "STOPPING",
    "SessionConfig",
    "SessionEvent",
    "SessionOutcome",
    "SessionRunner",
    "StartupMetrics",
    "Versioned",
    "VideoFileInput",
    "apply_rotation",
    "build_stages",
    "camera_session",
    "fit_preview_rgb",
    "match_text",
    "rotated_size",
    "video_session",
]
