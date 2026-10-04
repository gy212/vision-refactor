# -*- coding: utf-8 -*-
from __future__ import annotations

from core.exam_roster import ExamCandidate
from core.exam_session import ExamSession, ExamSessionConfig


def _cands(n: int = 2) -> list[ExamCandidate]:
    return [
        ExamCandidate(i + 1, f"S{i+1}", f"名{i+1}")
        for i in range(n)
    ]


def _kinds(cmds) -> list[str]:
    return [c.kind for c in cmds]


def test_happy_path_two_students() -> None:
    s = ExamSession(ExamSessionConfig(post_call_guard_s=1.0, inter_student_gap_s=0.5))
    s.load_roster(_cands(2))
    assert s.phase == "ready"
    cmds = s.handle("start_exam", now=0.0, run_id="r1")
    assert s.phase == "calling"
    assert "ANNOUNCE" in _kinds(cmds)
    assert "SCHEDULE_CALL_GUARD" in _kinds(cmds)

    cmds = s.handle("tick", now=1.0)
    assert s.phase == "wait_enter"
    assert "ARM_OCCUPANCY" in _kinds(cmds)

    cmds = s.handle("enter_stable", now=2.0)
    assert s.phase == "recording"
    assert "BEGIN_SEGMENT" in _kinds(cmds)
    assert not any(
        c.kind == "UPDATE_ROW" and c.payload.get("status") == "recording" for c in cmds
    )

    cmds = s.handle("empty_stable", now=10.0)
    assert s.phase == "finishing"
    assert "END_SEGMENT" in _kinds(cmds)

    cmds = s.handle("record_stopped", now=10.1)
    assert s.pointer == 1
    # gap then call
    cmds = s.handle("tick", now=10.1 + 0.5)
    assert s.phase == "calling"
    assert any("名2" in (c.payload.get("text") or "") for c in cmds if c.kind == "ANNOUNCE")


def test_skip_before_record() -> None:
    s = ExamSession(ExamSessionConfig(inter_student_gap_s=0.0, post_call_guard_s=0.0))
    s.load_roster(_cands(2))
    s.handle("start_exam", now=0.0)
    s.handle("tick", now=0.0)  # guard 0 → wait_enter
    cmds = s.handle("skip_current", now=1.0)
    assert s.rows[0].status == "skipped"
    assert s.pointer == 1
    # gap 0 → immediate next on tick
    cmds = s.handle("tick", now=1.0)
    assert s.phase == "calling"
    assert s.current.candidate.student_id == "S2"


def test_skip_while_recording_discards() -> None:
    s = ExamSession()
    s.load_roster(_cands(1))
    s.handle("start_exam", now=0.0)
    s.handle("call_guard_elapsed", now=2.0)
    s.handle("enter_stable", now=3.0)
    cmds = s.handle("skip_current", now=4.0)
    assert "DISCARD_SEGMENT" in _kinds(cmds)
    assert s.phase == "finishing"


def test_force_finish_from_recording() -> None:
    s = ExamSession()
    s.load_roster(_cands(1))
    s.handle("start_exam", now=0.0)
    s.handle("call_guard_elapsed", now=1.5)
    s.handle("enter_stable", now=2.0)
    cmds = s.handle("force_finish_current", now=2.1)  # < min_record 也允许
    assert "END_SEGMENT" in _kinds(cmds)
    assert s.phase == "finishing"


def test_pause_resume_wait_enter() -> None:
    s = ExamSession()
    s.load_roster(_cands(1))
    s.handle("start_exam", now=0.0)
    s.handle("call_guard_elapsed", now=2.0)
    assert s.phase == "wait_enter"
    s.handle("pause", now=3.0)
    assert s.phase == "paused"
    cmds = s.handle("resume", now=4.0)
    assert s.phase == "wait_enter"
    assert "ARM_OCCUPANCY" in _kinds(cmds)


def test_begin_failed_does_not_advance_allows_retry() -> None:
    """开录失败：记 failed，不 advance；回到 wait_enter 供重试/skip。"""
    s = ExamSession(ExamSessionConfig(inter_student_gap_s=0.0, post_call_guard_s=0.0))
    s.load_roster(_cands(2))
    s.handle("start_exam", now=0.0)
    s.handle("tick", now=0.0)
    s.handle("enter_stable", now=1.0)
    assert s.phase == "recording"
    cmds = s.handle("begin_failed", now=1.1, message="writer busy")
    assert s.rows[0].status == "failed"
    assert s.rows[0].error_code == "begin_failed"
    assert s.pointer == 0  # 不得静默跳过
    assert s.phase == "wait_enter"
    assert "ARM_OCCUPANCY" in _kinds(cmds)
    assert any(c.kind == "UPDATE_ROW" and c.payload.get("status") == "failed" for c in cmds)
    # 重试：再次 enter_stable
    s.handle("enter_stable", now=2.0)
    assert s.phase == "recording"
    assert s.pointer == 0
    assert s.rows[0].status == "recording"
    # 或 skip 离开
    s2 = ExamSession(ExamSessionConfig(inter_student_gap_s=0.0, post_call_guard_s=0.0))
    s2.load_roster(_cands(2))
    s2.handle("start_exam", now=0.0)
    s2.handle("tick", now=0.0)
    s2.handle("enter_stable", now=1.0)
    s2.handle("begin_failed", now=1.1, message="x")
    s2.handle("skip_current", now=2.0)
    assert s2.pointer == 1
    assert s2.rows[0].status == "failed"


def test_abort_while_paused_from_recording_discards() -> None:
    s = ExamSession()
    s.load_roster(_cands(1))
    s.handle("start_exam", now=0.0)
    s.handle("call_guard_elapsed", now=2.0)
    s.handle("enter_stable", now=3.0)
    s.handle("pause", now=4.0)
    assert s.phase == "paused"
    assert s.paused_from == "recording"
    cmds = s.handle("abort_exam", now=5.0)
    assert s.phase == "aborted"
    assert "DISCARD_SEGMENT" in _kinds(cmds)
    assert s.rows[0].status == "skipped"
    assert s.rows[0].error_code == "exam_aborted"
    assert s.locked is False


def test_retest_after_complete_does_not_reset_pointer_to_zero() -> None:
    """整场完成后追加重考再 start，应从 pending 重考行叫号，不回到 0 号。"""
    s = ExamSession(ExamSessionConfig(inter_student_gap_s=0.0, post_call_guard_s=0.0))
    s.load_roster(_cands(2))
    s.handle("start_exam", now=0.0)
    # 跳过两人至 completed
    s.handle("tick", now=0.0)
    s.handle("skip_current", now=1.0)
    s.handle("tick", now=1.0)
    s.handle("tick", now=1.0)
    s.handle("skip_current", now=2.0)
    assert s.phase == "completed"
    assert s.rows[0].status == "skipped"
    assert s.rows[1].status == "skipped"

    # 给第一人追加重考
    row = s.append_retest_row(s.rows[0].candidate)
    assert s.phase == "ready"
    assert row.status == "pending"
    assert row.attempt_index == 2

    cmds = s.handle("start_exam", now=3.0, run_id="r2")
    assert s.phase == "calling"
    assert s.pointer == 2  # 队尾重考行
    assert s.current is not None
    assert s.current.attempt_index == 2
    assert s.rows[0].status == "skipped"  # 历史不丢
    assert any("名1" in (c.payload.get("text") or "") for c in cmds if c.kind == "ANNOUNCE")


def test_retest_after_abort_resumes_remaining_queue_before_tail_retest() -> None:
    """中止后补考应恢复 ready，但不能跳过原名单中仍 pending 的考生。"""
    s = ExamSession(ExamSessionConfig(inter_student_gap_s=0.0, post_call_guard_s=0.0))
    s.load_roster(_cands(2))
    s.handle("start_exam", now=0.0)
    s.handle("tick", now=0.0)
    s.handle("enter_stable", now=1.0)
    s.handle("abort_exam", now=2.0)
    assert s.phase == "aborted"
    assert s.rows[0].status == "skipped"
    assert s.rows[1].status == "pending"

    retest = s.append_retest_row(s.rows[0].candidate)

    assert s.phase == "ready"
    assert s.pointer == 0
    assert retest.attempt_index == 2
    assert s.rows[-1] is retest

    cmds = s.handle("start_exam", now=3.0, run_id="resumed")
    assert s.phase == "calling"
    assert s.pointer == 1
    assert s.current is s.rows[1]
    assert any("名2" in (c.payload.get("text") or "") for c in cmds if c.kind == "ANNOUNCE")


def test_complete_exports() -> None:
    s = ExamSession(ExamSessionConfig(inter_student_gap_s=0.0, post_call_guard_s=0.0))
    s.load_roster(_cands(1))
    s.handle("start_exam", now=0.0)
    s.handle("tick", now=0.0)
    s.handle("enter_stable", now=1.0)
    s.handle("empty_stable", now=5.0)
    cmds = s.handle("record_stopped", now=5.1)
    assert s.phase == "completed"
    assert "EXPORT" in _kinds(cmds)
