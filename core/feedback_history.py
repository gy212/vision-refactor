"""练习历史门面：实现已迁入 ``core.services.practice_history``，片段视频解析迁入 ``core.media.segment_layout``（P9 删除）。"""
from core.media.segment_layout import VIDEO_SUFFIXES  # noqa: F401
from core.media.segment_layout import SegmentLayout
from core.services.practice_history import HistoryStore as FeedbackHistory  # noqa: F401
from core.services.practice_history import validate_identity  # noqa: F401

resolve_segment_videos = SegmentLayout.resolve_videos
