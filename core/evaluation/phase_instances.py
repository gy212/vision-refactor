"""记录检查阶段所属的动作次数与原始时间范围。"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class PhaseInstance(TypedDict):
    eventId: str
    segment: str
    phase: str
    repetition: int
    start: float
    end: float


def phase_instance(
    segment: str, phase: str, repetition: int, indices: np.ndarray, times: np.ndarray
) -> PhaseInstance:
    """为已找到的非空阶段保存归属，避免报告根据散落事件重新猜次数。"""
    return {
        "eventId": f"{segment}:{repetition}",
        "segment": segment,
        "phase": phase,
        "repetition": repetition,
        "start": float(times[indices[0]]),
        "end": float(times[indices[-1]]),
    }
