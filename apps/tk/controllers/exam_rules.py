# -*- coding: utf-8 -*-
"""考试的开考、重考与 ROI 判定规则：纯函数，只读会话与成绩行，返回拒绝理由（可以执行时返回 None）及会话计时配置。"""
from __future__ import annotations

from collections.abc import Sequence

from core.foundation import (
    EXAM_EMPTY_HOLD_S,
    EXAM_ENTER_STABLE_S,
    EXAM_INTER_STUDENT_GAP_S,
    EXAM_MIN_RECORD_S,
    EXAM_POST_CALL_GUARD_S,
)
from core.workflows import ExamResultRow, ExamSession, ExamSessionConfig

RETESTABLE = frozenset({"completed", "failed", "skipped", "superseded", "cancelled"})

NO_SCOREBOOK_TEXT = (
    "尚无本场成绩台账，无法重考。\n\n"
    "正确顺序：\n"
    "1. 双摄主界面点「开始」\n"
    "2. 本面板点「开始考试」产生场次与成绩表\n"
    "3. 考生考完（或失败/跳过）后，选中该考生点「重考」\n"
    "4. 若整场已结束/中止，再点一次「开始考试」才会叫到重考行\n\n"
    "说明：仅恢复名单不会恢复旧成绩；关闭面板后重考记录不自动续接。"
)
NO_SELECTION_TEXT = (
    "请先在列表中点选一名考生（已完成 / 失败 / 跳过），再点重考。\n"
    "考试进行中未选中时，默认对当前叫号考生操作。"
)


def session_config(prefs: dict) -> ExamSessionConfig:
    return ExamSessionConfig(
        post_call_guard_s=EXAM_POST_CALL_GUARD_S.read(prefs),
        inter_student_gap_s=EXAM_INTER_STUDENT_GAP_S.read(prefs),
        enter_stable_s=EXAM_ENTER_STABLE_S.read(prefs),
        empty_hold_s=EXAM_EMPTY_HOLD_S.read(prefs),
        min_record_s=EXAM_MIN_RECORD_S.read(prefs),
    )


def roi_problem(roi: Sequence[float]) -> str | None:
    """ROI 须满足 0≤x0<x1≤1 且 0≤y0<y1≤1；不满足时返回提示。"""
    if not (0.0 <= roi[0] < roi[2] <= 1.0 and 0.0 <= roi[1] < roi[3] <= 1.0):
        return "须满足 0≤x0<x1≤1 且 0≤y0<y1≤1"
    return None


def start_refusal(session: ExamSession) -> str | None:
    """开考前的阶段与名单检查；可以开考时返回 None。"""
    if not session.rows:
        return "请先导入名单"
    if session.phase != "ready":
        if session.phase == "completed":
            return "名单已全部完成，请先选择考生并点击重考"
        if session.phase == "aborted":
            return "考试已中止，请先选择考生并点击重考"
        return f"当前阶段 {session.phase}，无法重复开始"
    if not any(row.status == "pending" for row in session.rows):
        return "名单中没有待考考生，请先选择考生并点击重考"
    return None


def retest_refusal(row: ExamResultRow, same_student: Sequence[ExamResultRow]) -> str | None:
    """设计 §7.6：failed / skipped / completed（不满意）才补考；纯 pending 无需重考。可以重考时返回 None。"""
    name = row.candidate.name
    if row.status == "processing":
        return f"{name} 正在后台比对（processing），请待出分后再重考。"
    if row.status == "recording":
        return (f"{name} 正在录制。请先「强制结束」或「跳过当前」，"
                "待本段结束后再重考；或等其正常离场完成后再补考。")
    if not any(r.status in RETESTABLE for r in same_student):
        return (f"{name} 尚未完成首次作答（仍为 pending），无需重考。\n"
                "请等待叫号上场，或用「跳过当前」后再安排补考。")
    if any(r.status == "pending" and r.attempt_index > 1 for r in same_student):
        return f"{name} 已有待考的重考行在队尾，请先考完该次再追加。"
    return None
