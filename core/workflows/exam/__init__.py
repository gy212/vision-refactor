# -*- coding: utf-8 -*-
"""考试流程的公共 API：叫号状态机、名单导入、成绩台账与考试片段裁剪步骤。"""
from core.workflows.exam.clip import (
    EXAM_CLOSE_BUDGET_S,
    ClipResult,
    ExamClipStep,
    align_frame_counts,
    prepare_exam_pair,
)
from core.workflows.exam.roster import IMPORT_HEADERS, RosterError, import_roster_xlsx, write_import_template
from core.workflows.exam.scorebook import (
    ExamCandidate,
    ExamResultRow,
    ExamScorebook,
    ScorebookRegistry,
    apply_segment_update,
    local_timestamp,
    write_scorebook_xlsx,
)
from core.workflows.exam.session import (
    BEGIN_FAILED_MESSAGE,
    ExamCommand,
    ExamPhase,
    ExamSession,
    ExamSessionConfig,
)

__all__ = [
    "BEGIN_FAILED_MESSAGE",
    "ClipResult",
    "EXAM_CLOSE_BUDGET_S",
    "ExamCandidate",
    "ExamClipStep",
    "ExamCommand",
    "ExamPhase",
    "ExamResultRow",
    "ExamScorebook",
    "ExamSession",
    "ExamSessionConfig",
    "IMPORT_HEADERS",
    "RosterError",
    "ScorebookRegistry",
    "align_frame_counts",
    "apply_segment_update",
    "import_roster_xlsx",
    "local_timestamp",
    "prepare_exam_pair",
    "write_import_template",
    "write_scorebook_xlsx",
]
