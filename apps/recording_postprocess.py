"""双摄录后处理门面：实现已迁入 ``core.recording``（finalizer、steps）（P9 删除）。"""
from __future__ import annotations

from core.recording import SegmentStatus as PostprocessStatus  # noqa: F401
from core.recording import SegmentUpdate as PostprocessUpdate  # noqa: F401
from core.recording import (  # noqa: F401
    PostprocessError,
    default_template_paths,
    template_action_name,
    try_load_heavy_template,
    utc_timestamp,
    validate_auto_compare_template_lists,
    validate_auto_compare_templates,
    validate_template_pair,
)
