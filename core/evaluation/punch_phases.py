"""按已确认的直拳事件生成预摆与收回护位窗口，不包含收拳路径。"""
from __future__ import annotations

import numpy as np

PREPARE_SECONDS = .4
RECOVERY_SETTLE_SECONDS = .6


def punch_detail_windows(
    times: np.ndarray,
    punches: list[tuple[int, int, int]],
    stable: list[np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    """每次后手出拳前0.4秒查预摆，收回后0.6秒内取首个稳定护位。

    起点由后手前伸事件给出；有下一次出拳时不得跨入下一次准备或动作。
    没有及时稳定的护位就不生成窗口，由规则引擎报告证据不足。
    """
    prepare, recovered = [], []
    previous_end = -1
    for index, (start, _peak, end) in enumerate(punches):
        floor = max(previous_end + 1, int(np.searchsorted(times, times[start] - PREPARE_SECONDS)))
        prepare.append(np.arange(floor, start))

        deadline = times[end] + RECOVERY_SETTLE_SECONDS
        if index + 1 < len(punches):
            deadline = min(deadline, times[punches[index + 1][0]] - PREPARE_SECONDS)
        candidates = [
            window for window in stable
            if window[0] >= end and times[window[-1]] <= deadline
        ]
        if candidates:
            recovered.append(candidates[0])
        previous_end = end

    windows = {}
    for phase, parts in (("prepare", prepare), ("recovered", recovered)):
        if parts:
            windows[("rear_straight", phase)] = np.unique(np.concatenate(parts))
    return windows
