"""视频写出门面：转发到 ``core.media.writer``（P9 删除）。"""
from core.media.writer import (  # noqa: F401
    FFMPEG_TRANSCODE_TIMEOUT_S,
    open_video_writer,
    transcode_to_h264,
)

__all__ = [
    "FFMPEG_TRANSCODE_TIMEOUT_S",
    "open_video_writer",
    "transcode_to_h264",
]
