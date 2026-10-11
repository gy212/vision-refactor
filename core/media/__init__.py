# -*- coding: utf-8 -*-
"""媒体层（L1）的公共 API：离线视频解码、视频写出/转码与片段路径布局（摄像头子系统 P5 迁入）。"""
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
from core.media.segment_layout import VIDEO_SUFFIXES, SegmentLayout, view_for_role
from core.media.writer import open_video_writer, transcode_to_h264

__all__ = [
    "DEFAULT_FPS",
    "DecodeOptions",
    "Frame",
    "NO_END",
    "SegmentLayout",
    "UndecodableFrame",
    "VIDEO_SUFFIXES",
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
    "view_for_role",
]
