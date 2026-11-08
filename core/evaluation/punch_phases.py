"""按已确认的直拳事件生成预摆与收回护位窗口，不包含收拳路径。"""

from __future__ import annotations

import numpy as np

from core.evaluation.motion_events import MotionEvent
from core.evaluation.phase_instances import PhaseInstance, phase_instance

PREPARE_SECONDS = 0.4
RECOVERY_SETTLE_SECONDS = 0.6


def punch_detail_windows(
    times: np.ndarray,
    punches: list[MotionEvent],
    stable: list[np.ndarray],
    *,
    segment: str,
    phase_instances: list[PhaseInstance],
) -> dict[tuple[str, str], np.ndarray]:
    """每次直拳出拳前0.4秒查预摆，收回后0.6秒内取首个稳定护位。

    起点由该手前伸事件给出；有下一次出拳时不得跨入下一次准备或动作。
    没有及时稳定的护位就不生成窗口，由规则引擎报告证据不足。
    """
    prepare, recovered = [], []
    previous_end = -1
    for index, punch in enumerate(punches):
        start, end = punch.start, punch.end
        floor = max(previous_end + 1, int(np.searchsorted(times, times[start] - PREPARE_SECONDS)))
        prepare.append(np.arange(floor, start))
        if len(prepare[-1]):
            phase_instances.append(
                phase_instance(segment, "prepare", index + 1, prepare[-1], times)
            )

        if punch.ending != "returned":
            previous_end = end
            continue
        deadline = times[end] + RECOVERY_SETTLE_SECONDS
        if index + 1 < len(punches):
            deadline = min(deadline, times[punches[index + 1].start] - PREPARE_SECONDS)
        candidates = [
            window for window in stable if window[0] >= end and times[window[-1]] <= deadline
        ]
        if candidates:
            recovered.append(candidates[0])
            phase_instances.append(
                phase_instance(segment, "recovered", index + 1, candidates[0], times)
            )
        previous_end = end

    windows = {}
    for phase, parts in (("prepare", prepare), ("recovered", recovered)):
        if parts:
            windows[(segment, phase)] = np.unique(np.concatenate(parts))
    return windows
