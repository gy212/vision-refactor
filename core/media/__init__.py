# -*- coding: utf-8 -*-
"""媒体层（L1）的公共 API：离线视频解码与视频写出/转码（摄像头子系统 P5 迁入）。"""
from core.media.video_source import (
    DEFAULT_FPS,
    NO_END,
    DecodeOptions,
    Frame,
    UndecodableFrame,
    VideoError,
    VideoFpsUnavailable,
    VideoMeta,
    VideoOpenError,
    VideoSource,
    first_frame_ok,
    open_video,
    probe_video,
    read_frame_at,
)
from core.media.writer import open_video_writer, transcode_to_h264

__all__ = [
    "DEFAULT_FPS",
    "DecodeOptions",
    "Frame",
    "NO_END",
    "UndecodableFrame",
    "VideoError",
    "VideoFpsUnavailable",
    "VideoMeta",
    "VideoOpenError",
    "VideoSource",
    "first_frame_ok",
    "open_video",
    "open_video_writer",
    "probe_video",
    "read_frame_at",
    "transcode_to_h264",
]
