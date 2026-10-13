# -*- coding: utf-8 -*-
"""业务流程的统一命令词表：工作流状态机只产出 ``Command``，由 ``CommandExecutor`` 映射到端口执行。

名称沿用考试状态机原有的 ``ExamCommandKind``；学生练习只新增 ``START_ANALYSIS``、``SHOW_RESULT``、``STATUS``
（设计文档 §8.6）。``SCHEDULE_*``、``ADVANCE_POINTER`` 只是记录，执行层不做任何事。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

CommandKind = Literal[
    "ANNOUNCE",
    "ARM_OCCUPANCY",
    "DISARM_OCCUPANCY",
    "BEGIN_SEGMENT",
    "END_SEGMENT",
    "DISCARD_SEGMENT",
    "UPDATE_ROW",
    "SCHEDULE_CALL_GUARD",
    "SCHEDULE_INTER_GAP",
    "ADVANCE_POINTER",
    "EXPORT",
    "UI_LOCK_MANUAL_RECORD",
    "UI_UNLOCK_MANUAL_RECORD",
    "BIND_SEGMENT_PENDING",
    # 学生练习新增
    "START_ANALYSIS",
    "SHOW_RESULT",
    "STATUS",
]


@dataclass(frozen=True)
class Command:
    kind: CommandKind
    payload: dict[str, Any] = field(default_factory=dict)
