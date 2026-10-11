# -*- coding: utf-8 -*-
"""考试裁剪门面：实现已迁入 ``core.workflows.exam.clip``（P9 删除）。"""
from core.workflows.exam.clip import (  # noqa: F401
    ClipResult,
    align_frame_counts,
    energy_from_gray_frames,
    estimate_action_range,
    estimate_action_range_from_feature_seq,
    prepare_exam_pair,
)
