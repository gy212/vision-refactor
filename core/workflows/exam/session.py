# -*- coding: utf-8 -*-
"""考试编排纯状态机（无 Tk / 无 OpenCV）：叫号、到位开录、离场结束、跳过、暂停、恢复、重考与中止。

到位/离场由 ``PresenceRecordingCycle`` 判定（``presence`` 事件），会话停止、片段开录结果等由
``WorkflowHost`` 与 ``CommandExecutor`` 回送。契约原见旧仓库 docs/exam_system_design.md §6.6；
P7 从 ``core/exam_session.py`` 迁入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from core.workflows.commands import Command, CommandKind
from core.workflows.exam.scorebook import ExamCandidate, ExamResultRow, local_timestamp
from core.workflows.presence import PresenceGateConfig, PresenceRecordingCycle

ExamPhase = Literal[
    "idle",
    "ready",
    "calling",
    "wait_enter",
    "recording",
    "finishing",
    "completed",
    "paused",
    "aborted",
]

# 考试命令即统一命令词表（``core.workflows.commands``）；保留旧名。
ExamCommandKind = CommandKind
ExamCommand = Command

BEGIN_FAILED_MESSAGE = "开录失败（请确认双摄已运行且空闲）"
_IDLE_PHASES = frozenset({"idle", "ready", "completed", "aborted"})


@dataclass
class ExamSessionConfig:
    post_call_guard_s: float = 1.5
    inter_student_gap_s: float = 2.0
    enter_stable_s: float = 0.8
    empty_hold_s: float = 2.0
    min_record_s: float = 3.0


@dataclass
class ExamSession:
    config: ExamSessionConfig = field(default_factory=ExamSessionConfig)
    phase: ExamPhase = "idle"
    rows: list[ExamResultRow] = field(default_factory=list)
    pointer: int = 0
    paused_from: ExamPhase | None = None
    # 定时：单调时钟由外部驱动
    call_guard_deadline: float | None = None
    inter_gap_deadline: float | None = None
    record_started_mono: float | None = None
    run_id: str = ""
    locked: bool = False
    # 从开考到主会话停止（或换名单）为 True：主会话异常结束时据此中止考试
    run_active: bool = False
    # 最近一次站位样本，供界面显示
    present: bool | None = None
    presence: PresenceRecordingCycle = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.presence = PresenceRecordingCycle(
            PresenceGateConfig(
                enter_stable_s=self.config.enter_stable_s,
                empty_hold_s=self.config.empty_hold_s,
                min_record_s=self.config.min_record_s,
            )
        )

    def load_roster(self, candidates: list[ExamCandidate]) -> list[ExamCommand]:
        self.rows = [
            ExamResultRow(candidate=c, attempt_index=1, is_scoring_row=True)
            for c in candidates
        ]
        self.pointer = 0
        self.phase = "ready" if self.rows else "idle"
        self.paused_from = None
        self.run_active = False
        self.presence.disarm()
        self.call_guard_deadline = None
        self.inter_gap_deadline = None
        self.record_started_mono = None
        return []

    @property
    def current(self) -> ExamResultRow | None:
        if 0 <= self.pointer < len(self.rows):
            return self.rows[self.pointer]
        return None

    def row(self, row_id: str) -> ExamResultRow | None:
        return next((r for r in self.rows if r.row_id == row_id), None)

    def handle(self, event: str, **kw: Any) -> list[ExamCommand]:
        handler = {
            "start_exam": self._on_start_exam,
            "call_guard_elapsed": self._on_call_guard_elapsed,
            "enter_stable": self._on_enter_stable,
            "empty_stable": self._on_empty_stable,
            "record_started": self._on_record_started,
            "record_stopped": self._on_record_stopped,
            "skip_current": self._on_skip,
            "force_finish_current": self._on_force_finish,
            "pause": self._on_pause,
            "resume": self._on_resume,
            "abort_exam": self._on_abort,
            "tick": self._on_tick,
            "begin_failed": self._on_begin_failed,
            "presence": self._on_presence,
            "session_stopped": self._on_session_stopped,
            "session_ended": self._on_session_ended,
        }.get(event)
        if handler is None:
            return []
        return handler(**kw)

    # --- events ---

    def _first_pending_index(self) -> int | None:
        """下一个可叫号行：优先已有 pointer 上的 pending，否则扫描首个 pending。"""
        if 0 <= self.pointer < len(self.rows) and self.rows[self.pointer].status == "pending":
            return self.pointer
        for i, row in enumerate(self.rows):
            if row.status == "pending":
                return i
        return None

    def _on_start_exam(self, *, now: float, run_id: str = "", **_: Any) -> list[ExamCommand]:
        if self.phase != "ready" or not self.rows:
            return []
        start_at = self._first_pending_index()
        if start_at is None:
            return []
        if run_id:
            self.run_id = run_id
        elif not self.run_id:
            self.run_id = local_timestamp().replace(":", "")
        # 不得无条件 pointer=0：整场结束后追加重考时 pointer 已指向队尾 pending 行
        self.pointer = start_at
        self.locked = True
        self.run_active = True
        cmds = [ExamCommand("UI_LOCK_MANUAL_RECORD"), *self._start_call(now)]
        return cmds

    def _start_call(self, now: float) -> list[ExamCommand]:
        row = self.current
        self.presence.disarm()
        if row is None:
            self.phase = "completed"
            return [
                ExamCommand("DISARM_OCCUPANCY"),
                ExamCommand("ANNOUNCE", {"text": "本场考试已全部结束"}),
                ExamCommand("EXPORT"),
                ExamCommand("UI_UNLOCK_MANUAL_RECORD"),
            ]
        self.phase = "calling"
        self.call_guard_deadline = now + self.config.post_call_guard_s
        self.inter_gap_deadline = None
        # 仅刷新叫号时间；已 completed 的行不应被 _start_call 碰到
        if row.status not in {"completed", "superseded", "skipped"}:
            row.status = "pending"
        row.called_at = local_timestamp()
        text = f"请{row.candidate.order}号 {row.candidate.name} 上场考试"
        return [
            ExamCommand("DISARM_OCCUPANCY"),
            ExamCommand("ANNOUNCE", {"text": text}),
            ExamCommand("SCHEDULE_CALL_GUARD", {"deadline": self.call_guard_deadline}),
            ExamCommand("UPDATE_ROW", {"row_id": row.row_id, "status": row.status}),
        ]

    def _on_call_guard_elapsed(self, *, now: float | None = None, **_: Any) -> list[ExamCommand]:
        if self.phase != "calling":
            return []
        self.phase = "wait_enter"
        self.call_guard_deadline = None
        # 站位样本带源帧采集时刻：叫号守护期间采到的帧不算进场（D-30）
        self.presence.arm_enter(ignore_before=now)
        return [ExamCommand("ARM_OCCUPANCY")]

    def _on_enter_stable(self, *, now: float, **_: Any) -> list[ExamCommand]:
        if self.phase != "wait_enter":
            return []
        row = self.current
        if row is None:
            return []
        self.phase = "recording"
        self.record_started_mono = now
        # 允许 begin_failed 后同人重试：清掉失败标记
        row.status = "recording"
        row.error_code = None
        row.error_message = None
        row.record_started_at = local_timestamp()
        # 注意：UPDATE_ROW(recording) 必须在 BEGIN 成功后由执行层发出。
        # 若与 BEGIN_SEGMENT 同批且 BEGIN 失败触发 begin_failed，外层后续
        # UPDATE_ROW(recording) 会把 failed 盖回 recording。
        return [
            ExamCommand("ANNOUNCE", {"text": "考试开始"}),
            ExamCommand("BEGIN_SEGMENT", {"row_id": row.row_id, "exam": self._segment_meta(row)}),
        ]

    def _segment_meta(self, row: ExamResultRow) -> dict[str, Any]:
        """写入片段结果 JSON 的考生信息（``ResultSpec.exam``），录后结果按其中的 run_id 回填台账。"""
        candidate = row.candidate
        return {
            "exam_run_id": str(self.run_id) if self.run_id else None,
            "student_id": candidate.student_id,
            "student_name": candidate.name,
            "student_order": candidate.order,
            "attempt_index": row.attempt_index,
        }

    def _on_empty_stable(self, **_: Any) -> list[ExamCommand]:
        if self.phase != "recording":
            return []
        return self._finish_recording(discard=False)

    def _on_force_finish(self, **_: Any) -> list[ExamCommand]:
        if self.phase != "recording":
            return []
        # bypass min_record_s — PresenceGate 不参与
        return self._finish_recording(discard=False)

    def _finish_recording(self, *, discard: bool) -> list[ExamCommand]:
        row = self.current
        if row is None:
            return []
        self.phase = "finishing"
        self.presence.disarm()
        cmds: list[ExamCommand] = [
            ExamCommand("ANNOUNCE", {"text": "考试结束"}),
            ExamCommand("DISARM_OCCUPANCY"),
        ]
        if discard:
            cmds.append(ExamCommand("DISCARD_SEGMENT", {"row_id": row.row_id}))
            row.status = "skipped"
            row.record_ended_at = local_timestamp()
            cmds.append(
                ExamCommand(
                    "UPDATE_ROW",
                    {"row_id": row.row_id, "status": "skipped"},
                )
            )
        else:
            cmds.append(ExamCommand("END_SEGMENT", {"row_id": row.row_id}))
            row.status = "processing"
            row.record_ended_at = local_timestamp()
            cmds.append(
                ExamCommand(
                    "UPDATE_ROW",
                    {"row_id": row.row_id, "status": "processing"},
                )
            )
        return cmds

    def _on_record_started(self, *, now: float, segment: Any = None, **_: Any) -> list[ExamCommand]:
        """开录成功（执行层回送）：从此刻起等待离场，绑定片段并把行状态改为 recording（清掉上次失败标记）。"""
        if self.phase != "recording":
            return []
        row = self.current
        self.presence.recording_started(now)
        if row is None:
            return []
        cmds: list[ExamCommand] = []
        if segment is not None:
            directory = getattr(segment, "directory", None)
            cmds.append(
                ExamCommand(
                    "BIND_SEGMENT_PENDING",
                    {
                        "row_id": row.row_id,
                        "segment_id": segment.segment_id,
                        "segment_dir": str(directory) if directory is not None else None,
                    },
                )
            )
        cmds.append(
            ExamCommand(
                "UPDATE_ROW",
                {"row_id": row.row_id, "status": "recording", "error_code": None, "error_message": None},
            )
        )
        return cmds

    def _on_presence(self, *, present: bool, now: float, **_: Any) -> list[ExamCommand]:
        """站位样本：只在等待进入与录制中交给到位周期判定。"""
        self.present = bool(present)
        if self.phase not in {"wait_enter", "recording"}:
            return []
        cmds: list[ExamCommand] = []
        for ev in self.presence.update(bool(present), now):
            cmds.extend(self.handle(ev.kind, now=now))
        return cmds

    def _on_session_stopped(self, *, now: float, **_: Any) -> list[ExamCommand]:
        """主会话停止：中止进行中的考试并解除占用与手动录制锁。"""
        cmds: list[ExamCommand] = []
        if self.phase not in _IDLE_PHASES:
            cmds.extend(self._on_abort())
        self.run_active = False
        self.presence.disarm()
        return [
            *cmds,
            ExamCommand("DISARM_OCCUPANCY"),
            ExamCommand("UI_UNLOCK_MANUAL_RECORD"),
            ExamCommand("STATUS", {"text": "主会话已停止，考试已中止"}),
        ]

    def _on_session_ended(self, *, now: float, **_: Any) -> list[ExamCommand]:
        """主会话结束（含异常结束）：开考后才按停止处理；用户点停止时已处理过，这里不再重复。"""
        if not self.run_active:
            return []
        return self._on_session_stopped(now=now)

    def _on_record_stopped(
        self, *, now: float, discarded: bool = False, **_: Any
    ) -> list[ExamCommand]:
        if self.phase != "finishing":
            return []
        self.record_started_mono = None
        self.pointer += 1
        if self.pointer >= len(self.rows):
            self.phase = "completed"
            self.locked = False
            return [
                ExamCommand("ANNOUNCE", {"text": "本场考试已全部结束"}),
                ExamCommand("EXPORT"),
                ExamCommand("UI_UNLOCK_MANUAL_RECORD"),
            ]
        # 下一位前 inter-gap
        self.phase = "calling"  # 先进入 calling 前的 gap：用 inter_gap_deadline 挡住 call_guard
        # 设计：gap 结束后再 _start_call。这里用 phase 临时保持 finishing 已结束。
        # 用 dedicated：phase 回到特殊 — 简化：直接 schedule gap，phase=ready-like internal
        self.inter_gap_deadline = now + self.config.inter_student_gap_s
        self.phase = "calling"
        # 在 gap 期间 phase 用 "calling" 但 call_guard 未设；tick 到 gap 后再真正 start_call
        # 为避免重复叫号，用 flag：call_guard_deadline is None and inter_gap set 表示 gap 中
        self.call_guard_deadline = None
        return [
            ExamCommand(
                "SCHEDULE_INTER_GAP",
                {"deadline": self.inter_gap_deadline},
            )
        ]

    def _on_tick(self, *, now: float, **_: Any) -> list[ExamCommand]:
        cmds: list[ExamCommand] = []
        if (
            self.phase == "calling"
            and self.inter_gap_deadline is not None
            and now >= self.inter_gap_deadline
        ):
            self.inter_gap_deadline = None
            # 真正叫下一位
            cmds.extend(self._start_call(now))
            return cmds
        if (
            self.phase == "calling"
            and self.call_guard_deadline is not None
            and now >= self.call_guard_deadline
        ):
            cmds.extend(self._on_call_guard_elapsed(now=now))
        return cmds

    def _on_skip(self, *, now: float, **_: Any) -> list[ExamCommand]:
        row = self.current
        if row is None:
            return []
        if self.phase in {"calling", "wait_enter"}:
            # begin_failed 后停在 wait_enter 且 status=failed：skip 保留 failed，否则标 skipped
            self.presence.disarm()
            if row.status != "failed":
                row.status = "skipped"
            row.record_ended_at = local_timestamp()
            cmds = [
                ExamCommand("DISARM_OCCUPANCY"),
                ExamCommand(
                    "UPDATE_ROW",
                    {
                        "row_id": row.row_id,
                        "status": row.status,
                        "error_code": row.error_code,
                        "error_message": row.error_message,
                    },
                ),
            ]
            self.pointer += 1
            return cmds + self._after_skip_or_done(now)
        if self.phase == "recording":
            cmds = self._finish_recording(discard=True)
            # record_stopped 会 advance；这里标记 discarded 路径由执行层在 stop 后发 record_stopped(discarded=True)
            return cmds
        return []

    def _after_skip_or_done(self, now: float) -> list[ExamCommand]:
        if self.pointer >= len(self.rows):
            self.phase = "completed"
            self.locked = False
            return [
                ExamCommand("ANNOUNCE", {"text": "本场考试已全部结束"}),
                ExamCommand("EXPORT"),
                ExamCommand("UI_UNLOCK_MANUAL_RECORD"),
            ]
        self.inter_gap_deadline = now + self.config.inter_student_gap_s
        self.phase = "calling"
        self.call_guard_deadline = None
        return [
            ExamCommand(
                "SCHEDULE_INTER_GAP",
                {"deadline": self.inter_gap_deadline},
            )
        ]

    def _on_begin_failed(
        self, *, now: float, message: str = BEGIN_FAILED_MESSAGE, **_: Any
    ) -> list[ExamCommand]:
        """开录失败：标记 failed，回到 wait_enter 供考官重试或 skip；不静默 advance。

        设计契约 docs/exam_system_design.md §6.6：BEGIN_SEGMENT 失败可 skip/重试，
        不得自动 pointer+=1。
        """
        if self.phase != "recording":
            return []
        row = self.current
        if row is None:
            return []
        row.status = "failed"
        row.error_code = "begin_failed"
        row.error_message = message or "开录失败"
        # 不写 record_ended_at：允许同一次尝试重试开录
        self.record_started_mono = None
        self.phase = "wait_enter"
        self.call_guard_deadline = None
        self.inter_gap_deadline = None
        self.presence.arm_enter(ignore_before=now)
        return [
            ExamCommand("DISARM_OCCUPANCY"),
            ExamCommand(
                "ANNOUNCE",
                {
                    "text": (
                        f"{row.candidate.name} 开录失败，请重试上场或跳过"
                    )
                },
            ),
            ExamCommand(
                "UPDATE_ROW",
                {
                    "row_id": row.row_id,
                    "status": "failed",
                    "error_code": row.error_code,
                    "error_message": row.error_message,
                },
            ),
            # 重新武装占用，便于进场稳定后再次 BEGIN
            ExamCommand("ARM_OCCUPANCY"),
        ]

    def _on_pause(self, **_: Any) -> list[ExamCommand]:
        if self.phase in {"idle", "ready", "completed", "aborted", "paused", "finishing"}:
            return []
        self.paused_from = self.phase
        self.phase = "paused"
        self.presence.disarm()
        return [ExamCommand("DISARM_OCCUPANCY")]

    def _on_resume(self, *, now: float, **_: Any) -> list[ExamCommand]:
        if self.phase != "paused" or self.paused_from is None:
            return []
        prev = self.paused_from
        self.paused_from = None
        self.phase = prev
        cmds: list[ExamCommand] = []
        if prev == "wait_enter":
            self.presence.arm_enter(ignore_before=now)   # 暂停前采到的样本不算（D-30）
        elif prev == "recording":
            # 恢复录制：最短录制时长从恢复时刻重新计
            self.presence.recording_started(now)
        if prev in {"wait_enter", "recording"}:
            cmds.append(ExamCommand("ARM_OCCUPANCY"))
        return cmds

    def _needs_discard_on_abort(self) -> bool:
        """录制中、收尾中，或从 recording 暂停时，中止须停 writer。"""
        if self.phase in {"recording", "finishing"}:
            return True
        if self.phase == "paused" and self.paused_from == "recording":
            return True
        return False

    def _on_abort(self, **_: Any) -> list[ExamCommand]:
        if self.phase in {"idle", "aborted", "completed"}:
            return []
        cmds: list[ExamCommand] = [ExamCommand("DISARM_OCCUPANCY")]
        row = self.current
        if self._needs_discard_on_abort() and row is not None:
            cmds.append(
                ExamCommand("DISCARD_SEGMENT", {"row_id": row.row_id})
            )
            # 中止时不再走 record_stopped 推进；直接标记当前行
            if row.status in {"recording", "pending", "processing"}:
                row.status = "skipped"
                row.error_code = row.error_code or "exam_aborted"
                row.error_message = row.error_message or "考试中止"
                row.record_ended_at = local_timestamp()
                cmds.append(
                    ExamCommand(
                        "UPDATE_ROW",
                        {
                            "row_id": row.row_id,
                            "status": "skipped",
                            "error_code": row.error_code,
                            "error_message": row.error_message,
                        },
                    )
                )
        self.phase = "aborted"
        self.paused_from = None
        self.presence.disarm()
        self.locked = False
        self.call_guard_deadline = None
        self.inter_gap_deadline = None
        self.record_started_mono = None
        cmds.append(ExamCommand("UI_UNLOCK_MANUAL_RECORD"))
        return cmds

    def append_retest_row(self, candidate: ExamCandidate) -> ExamResultRow:
        attempts = [
            r.attempt_index
            for r in self.rows
            if r.candidate.student_id == candidate.student_id
        ]
        next_attempt = (max(attempts) if attempts else 0) + 1
        row = ExamResultRow(
            candidate=candidate,
            attempt_index=next_attempt,
            is_scoring_row=True,
        )
        return self.queue_retest_row(row)

    def queue_retest_row(self, row: ExamResultRow) -> ExamResultRow:
        """登记补考行，并让已结束或中止的场次重新进入可开考状态。"""
        if not any(existing.row_id == row.row_id for existing in self.rows):
            self.rows.append(row)

        previous_phase = self.phase
        if previous_phase not in {"completed", "aborted"}:
            return row

        self.phase = "ready"
        self.paused_from = None
        self.call_guard_deadline = None
        self.inter_gap_deadline = None
        self.record_started_mono = None
        self.locked = False
        if previous_phase == "completed":
            self.pointer = self.rows.index(row)
        # aborted 可能仍有未考名单，保留 pointer，由 _first_pending_index 按原队列续考。
        return row
