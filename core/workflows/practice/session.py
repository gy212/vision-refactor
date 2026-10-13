# -*- coding: utf-8 -*-
"""学生练习的纯状态机：等待会话 → 等待进入 → 录制中 → 等待后处理终态 → 可评判 → 分析中 → 显示结果。

承接原 App 的 ``_student_start``、``_student_wait_for_position``、``_student_on_occupancy``、
``_student_try_begin_recording``、``_student_end``、``_on_student_postprocess_update``、``_student_judge``
的流程逻辑（设计文档 §8.6）。不依赖 Tk：界面、录制、播报与分析都经命令交给 ``CommandExecutor``。
到位/离场用 ``PresenceRecordingCycle``，离场 2 s 即结束，不设最短录制时长。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from core.media import SegmentLayout
from core.workflows.commands import Command
from core.workflows.presence import PresenceGateConfig, PresenceRecordingCycle

PracticePhase = Literal[
    "idle",
    "waiting_session",
    "waiting_enter",
    "recording",
    "waiting_postprocess",
    "ready_to_judge",
    "analyzing",
    "showing_result",
]

_WAITING = frozenset({"waiting_session", "waiting_enter"})
_TERMINAL = frozenset({"completed", "failed", "skipped", "cancelled"})
_IN_PROGRESS = frozenset({"queued", "transcoding", "validating"})

STATUS_WAIT_ENTER = "请进入正面预览黄框，站稳0.8秒后播报开始并自动录制"


@dataclass
class PracticeSegment:
    """本次练习录下的片段：开录时登记 id 与目录，后处理终态后记下可读的视频。"""

    segment_id: str
    directory: Path | None
    front: Path | None = None
    side: Path | None = None
    ready: bool = False
    terminal: bool = False


def _status(text: str = "", *, alert: str = "", title: str = "", error: bool = False) -> Command:
    """状态文字；``alert`` 另弹提示框（``error`` 为错误框）。"""
    payload: dict[str, Any] = {}
    if text:
        payload["text"] = text
    if alert:
        payload.update(alert=alert, title=title, error=error)
    return Command("STATUS", payload)


def _readable(path: Any) -> Path | None:
    return Path(path) if path is not None and Path(path).is_file() else None


class PracticeSession:
    """一次学生练习模式内的流程状态。``identity`` 在开录瞬间读取学生、动作与左右式，无效时抛 ``ValueError``。"""

    def __init__(
        self,
        identity: Callable[[], dict[str, Any]],
        *,
        presence: PresenceGateConfig | None = None,
    ) -> None:
        self._identity = identity
        self.presence = PresenceRecordingCycle(presence or PresenceGateConfig(min_record_s=0.0))
        self.phase: PracticePhase = "idle"
        self.session_ready = False
        self.busy = False  # 有分析任务在跑（含导入旧视频）
        self.segment: PracticeSegment | None = None
        self.identity: dict[str, Any] | None = None  # 开录时绑定的身份，评判按它进行
        self.record_id: str | None = None
        self.token = 0
        self.active_token: int | None = None
        self.active_student_id: str | None = None
        self.current_record: dict[str, Any] | None = None
        self._pending_identity: dict[str, Any] | None = None

    # ------------------------------------------------------------------ 查询
    @property
    def waiting(self) -> bool:
        return self.phase in _WAITING

    @property
    def recording(self) -> bool:
        return self.phase == "recording"

    @property
    def can_judge(self) -> bool:
        return bool(self.segment and self.segment.ready) and not (self.busy or self.waiting or self.recording)

    def row(self, _row_id: str) -> None:
        return None

    def handle(self, event: str, **kw: Any) -> list[Command]:
        handler = {
            "start": self._on_start,
            "session_ready": self._on_session_ready,
            "session_failed": self._on_session_failed,
            "presence": self._on_presence,
            "enter_stable": self._on_enter_stable,
            "record_started": self._on_record_started,
            "begin_failed": self._on_begin_failed,
            "end": self._on_end,
            "empty_stable": self._on_end,
            "segment_update": self._on_segment_update,
            "judge": self._on_judge,
            "analysis_busy": self._on_analysis_busy,
            "analysis_start_failed": self._on_analysis_start_failed,
            "record_saved": self._on_record_saved,
            "analysis_saved": self._on_analysis_saved,
            "identity_changed": self._on_identity_changed,
            "analysis_cancelled": self._on_analysis_cancelled,
            "practice_again": self._on_practice_again,
            "session_stopped": self._on_session_stopped,
            "session_ended": self._on_session_ended,
            "exit": self._on_exit,
        }.get(event)
        if handler is None:
            return []
        return handler(**kw)

    # ------------------------------------------------------------------ 开始与到位
    def _rest_phase(self) -> PracticePhase:
        """没有进行中的录制时应处的阶段：由最近片段的后处理状态决定。"""
        if self.segment is None:
            return "idle"
        if self.segment.ready:
            return "ready_to_judge"
        return "idle" if self.segment.terminal else "waiting_postprocess"

    def _arm(self, now: float) -> list[Command]:
        self.phase = "waiting_enter"
        # 武装前排队的旧样本不能让新一轮立即开始
        self.presence.arm_enter(ignore_before=now)
        return [Command("ARM_OCCUPANCY"), _status(STATUS_WAIT_ENTER)]

    def _on_start(self, *, now: float, session_ready: bool, launching: bool = False, **_: Any) -> list[Command]:
        """「开始」：会话已就绪则等待进入，否则等会话就绪（``launching`` 表示调用方正在启动会话）。"""
        if self.busy or self.phase in _WAITING or self.phase == "recording":
            return []
        self.session_ready = bool(session_ready)
        if self.session_ready:
            return self._arm(now)
        self.phase = "waiting_session"
        return [_status("正在启动双摄预览…")] if launching else []

    def _on_session_ready(self, *, now: float, **_: Any) -> list[Command]:
        """双摄已接管设备并可录制：只启用站位检测，到位事件才开录。"""
        self.session_ready = True
        if self.phase != "waiting_session":
            return []
        return self._arm(now)

    def _on_session_failed(self, *, message: str = "", **_: Any) -> list[Command]:
        if self.phase != "waiting_session":
            return []
        self.phase = self._rest_phase()
        return [_status(message)] if message else []

    def _on_presence(self, *, present: bool, now: float, **_: Any) -> list[Command]:
        if self.busy or self.phase not in {"waiting_enter", "recording"}:
            return []
        cmds: list[Command] = []
        for ev in self.presence.update(bool(present), now):
            cmds.extend(self.handle(ev.kind, now=now))
        return cmds

    def _on_enter_stable(self, **_: Any) -> list[Command]:
        if self.busy or self.phase != "waiting_enter":
            return []
        if not self.session_ready:
            self.presence.disarm()
            self.phase = self._rest_phase()
            return [Command("DISARM_OCCUPANCY"), _status("双摄尚未就绪，请重试「开始」")]
        try:
            self._pending_identity = self._identity()
        except ValueError as exc:
            self.presence.disarm()
            self.phase = self._rest_phase()
            return [Command("DISARM_OCCUPANCY"), _status(str(exc))]
        return [Command("BEGIN_SEGMENT", {})]

    def _on_record_started(self, *, now: float, segment: Any = None, **_: Any) -> list[Command]:
        if self.phase != "waiting_enter":
            return []
        self.phase = "recording"
        self.presence.recording_started(now)
        self.identity = self._pending_identity
        self._pending_identity = None
        self.record_id = None
        if segment is not None:
            directory = getattr(segment, "directory", None)
            self.segment = PracticeSegment(segment.segment_id, Path(directory) if directory is not None else None)
        return [
            Command("ANNOUNCE", {"text": "开始"}),
            _status("开始，录制中…离开黄框连续2秒自动结束，也可点「结束」"),
        ]

    def _on_begin_failed(self, **_: Any) -> list[Command]:
        if self.phase != "waiting_enter":
            return []
        self._pending_identity = None
        self.presence.disarm()
        self.phase = self._rest_phase()
        return [
            Command("DISARM_OCCUPANCY"),
            _status("开录失败，请重试「开始」", alert="无法开始录制，请确认双摄已就绪。", title="学生练习", error=True),
        ]

    # ------------------------------------------------------------------ 结束与后处理
    def _on_end(self, **_: Any) -> list[Command]:
        """结束 = 停录落盘，不自动评判；须等后处理终态才可评判。"""
        if self.busy:
            return []
        waiting = self.phase in _WAITING
        self.presence.disarm()
        cmds: list[Command] = [Command("DISARM_OCCUPANCY")]
        if self.phase != "recording":
            if waiting:
                self.phase = self._rest_phase()
            return [*cmds, _status("已取消等待到位" if waiting else "当前没有进行中的录制")]
        self.phase = "waiting_postprocess"
        segment = self.segment
        if segment is not None:
            segment.ready, segment.front, segment.side = False, None, None
        if segment is not None and segment.directory is not None:
            text = f"已停录：{segment.directory.name}，正在转码落盘，完成后可「动作评判」…"
        else:
            text = "录制已结束（未解析到片段目录，请重试）"
        return [*cmds, Command("END_SEGMENT", {}), Command("ANNOUNCE", {"text": "结束"}), _status(text)]

    def _on_segment_update(self, *, update: Any, **_: Any) -> list[Command]:
        """后处理终态：记录最终视频路径，再允许动作评判。"""
        segment = self.segment
        if segment is None or update.segment_id != segment.segment_id:
            return []
        if update.status not in _TERMINAL:
            if update.status in _IN_PROGRESS:
                return [_status(f"处理中：{update.message or update.status}…")]
            return []
        segment.terminal = True
        front = _readable(getattr(update, "front_video", None))
        side = _readable(getattr(update, "side_video", None))
        # failed/cancelled 时仍有可读源文件也允许评判（降级），否则禁用
        if update.status in {"failed", "cancelled"} and front is None and side is None:
            segment.ready, segment.front, segment.side = False, None, None
            self._settle()
            return [_status(f"片段处理失败，无法评判：{update.error_code or update.message}")]
        if (front is None or side is None) and segment.directory is not None:
            try:
                f2, s2 = SegmentLayout.resolve_videos(segment.directory)
                front, side = front or f2, side or s2
            except Exception:
                pass
        segment.front, segment.side = front, side
        # 单路仍可检查可见项目；缺视角的规则明确标记无法判断
        segment.ready = _readable(front) is not None or _readable(side) is not None
        self._settle()
        if segment.ready:
            return [_status(f"已就绪：{segment.segment_id}，可点「动作评判」")]
        return [_status(f"处理结束但未找到可读视频：{update.error_code or update.status}")]

    def _settle(self) -> None:
        if self.phase == "waiting_postprocess":
            self.phase = self._rest_phase()

    # ------------------------------------------------------------------ 评判与结果
    def _on_judge(self, **_: Any) -> list[Command]:
        """按开录时绑定的学生与动作发起后台分析。"""
        if self.busy or self.phase == "analyzing" or self.phase in _WAITING:
            return []
        title = "动作评判"
        if not (self.segment and self.segment.ready):
            return [_status(alert="请先结束练习，并等待录像转码落盘。", title=title)]
        if self.phase == "recording":
            return [_status(alert="请先结束录制再评判。", title=title)]
        if self.identity is None:
            return [_status(alert="这段录像未绑定学生，请使用「导入旧视频」分析。", title=title)]
        self.token += 1
        self.active_token = self.token
        self.active_student_id = self.identity.get("studentId") if isinstance(self.identity, dict) else None
        self.phase = "analyzing"
        return [
            Command(
                "START_ANALYSIS",
                {
                    "identity": self.identity,
                    "front": self.segment.front,
                    "side": self.segment.side,
                    "record_id": self.record_id,
                    "token": self.token,
                    "student_id": self.active_student_id,
                },
            )
        ]

    def _on_analysis_busy(self, *, busy: bool, **_: Any) -> list[Command]:
        self.busy = bool(busy)
        if not self.busy and self.phase == "analyzing":
            self.phase = self._rest_phase()
        return []

    def _on_analysis_start_failed(self, *, token: int | None = None, **_: Any) -> list[Command]:
        """分析没能启动（如配置错误）：作废本次任务代次，回到可重试的阶段。"""
        if token is not None and token != self.active_token:
            return []
        self.active_token = None
        self.active_student_id = None
        if self.phase == "analyzing":
            self.phase = self._rest_phase()
        return []

    def _on_record_saved(self, *, record_id: str | None, **_: Any) -> list[Command]:
        """评判任务的记录已落盘：之后对同一片段重评时覆盖这条记录（不论结果是否上屏）。"""
        self.record_id = record_id
        return []

    def _on_analysis_saved(
        self, *, token: int, student_id: str | None, record: dict[str, Any],
        current_student_id: str | None = None, **_: Any,
    ) -> list[Command]:
        """分析完成且已保存：按任务代次与当前学号双重拦截晚到或作废的结果。"""
        if self.active_token is None or token != self.active_token:
            return []
        if current_student_id is not None and current_student_id != student_id:
            return []
        self.current_record = record
        self.record_id = record.get("id")
        self.phase = "showing_result"
        return [Command("SHOW_RESULT", {"record": record})]

    def _on_identity_changed(self, **_: Any) -> list[Command]:
        """修改了学号：作废当前任务代次，清空旧记录。"""
        self.active_token = None
        self.active_student_id = None
        self.current_record = None
        return []

    def _on_analysis_cancelled(self, **_: Any) -> list[Command]:
        self.active_token = None
        return []

    def _on_practice_again(self, **_: Any) -> list[Command]:
        """从结果页回到预览：解除站位检测、清空就绪态并作废代次；不会自动开录。"""
        if self.busy:
            return []
        self.presence.disarm()
        if self.segment is not None:
            self.segment.ready, self.segment.front, self.segment.side = False, None, None
        self.phase = "idle"
        self.active_token = None
        return [Command("DISARM_OCCUPANCY"), _status("已返回录制预览。进入黄框就位后请点击「开始」")]

    # ------------------------------------------------------------------ 会话与退出
    def _stop_waiting(self) -> list[Command]:
        self.presence.disarm()
        if self.phase in _WAITING:
            self.phase = self._rest_phase()
        elif self.phase == "recording":
            # 会话收尾时片段照常提交，等待后处理终态
            self.phase = "waiting_postprocess"
        return [Command("DISARM_OCCUPANCY")]

    def _on_session_stopped(self, **_: Any) -> list[Command]:
        return self._stop_waiting()

    def _on_session_ended(self, *, status: str = "", **_: Any) -> list[Command]:
        self.session_ready = False
        return [*self._stop_waiting(), _status(status or "双摄会话已结束，可点「开始」重试")]

    def _on_exit(self, **_: Any) -> list[Command]:
        """退出练习：仍在录制则先结束片段，再复位全部状态。"""
        if self.busy:
            return []
        recording = self.phase == "recording"
        self.presence.disarm()
        self.phase = "idle"
        self.segment = None
        self.identity = None
        self._pending_identity = None
        self.active_token = None
        self.active_student_id = None
        self.current_record = None
        cmds = [Command("DISARM_OCCUPANCY")]
        return [*cmds, Command("END_SEGMENT", {})] if recording else cmds
