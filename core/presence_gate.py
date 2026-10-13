# -*- coding: utf-8 -*-
"""旧导入路径的转发门面：站位去抖闸门在 ``core.workflows.presence``（P9 删除）。"""
from core.workflows.presence import (
    PresenceEvent,
    PresenceEventKind,
    PresenceGate,
    PresenceGateConfig,
    hip_midpoint_in_roi,
)

__all__ = ["PresenceEvent", "PresenceEventKind", "PresenceGate", "PresenceGateConfig", "hip_midpoint_in_roi"]
