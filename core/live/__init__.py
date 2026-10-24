# -*- coding: utf-8 -*-
"""实时会话（L4）的公共 API：会话配置与运行时控制、帧来源、逐帧 stages、``SessionRunner``、预览槽与会话管理。"""
from core.live.config import (
    DEFAULT_ROI,
    LiveControls,
    OccupancySettings,
    PresenceProbe,
    SessionConfig,
    Versioned,
    camera_session,
    video_session,
)
from core.live.inputs import CameraInput, ClaimedCameraInput, FrameInput, InputFrames, VideoFileInput
from core.live.manager import IDLE, RUNNING, STARTING, STOPPING, LiveSessionManager
from core.live.metrics import StartupMetrics
from core.live.presence import PRESENCE_INTERVAL_S, PresenceSample, PresenceSampler
from core.live.preview import PreviewFrame, PreviewStore, fit_preview_rgb
from core.live.runner import APP_CLOSE, FrameSink, RunningInfo, SessionEvent, SessionOutcome, SessionRunner
from core.live.stages import (
    FramePacket,
    build_stages,
    match_text,
)

__all__ = [
    "APP_CLOSE",
    "CameraInput",
    "ClaimedCameraInput",
    "DEFAULT_ROI",
    "FrameInput",
    "InputFrames",
    "FramePacket",
    "FrameSink",
    "IDLE",
    "LiveControls",
    "LiveSessionManager",
    "OccupancySettings",
    "PRESENCE_INTERVAL_S",
    "PresenceProbe",
    "PresenceSample",
    "PresenceSampler",
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
    "build_stages",
    "camera_session",
    "fit_preview_rgb",
    "match_text",
    "video_session",
]
