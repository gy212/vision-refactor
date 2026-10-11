# -*- coding: utf-8 -*-
"""考试流程的公共 API：考试片段裁剪步骤与成绩台账（P4）；叫号会话与名单 P7 迁入。"""
from core.workflows.exam.clip import EXAM_CLOSE_BUDGET_S, ClipResult, ExamClipStep, prepare_exam_pair
from core.workflows.exam.scorebook import (
    ExamCandidate,
    ExamResultRow,
    ExamScorebook,
    ScorebookRegistry,
    apply_segment_update,
    write_scorebook_xlsx,
)

__all__ = [
    "ClipResult",
    "EXAM_CLOSE_BUDGET_S",
    "ExamCandidate",
    "ExamClipStep",
    "ExamResultRow",
    "ExamScorebook",
    "ScorebookRegistry",
    "apply_segment_update",
    "prepare_exam_pair",
    "write_scorebook_xlsx",
]
