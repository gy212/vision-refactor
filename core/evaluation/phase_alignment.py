"""将一份动作阶段投影到同步时间轴，并核对各机位的数据覆盖。"""

from __future__ import annotations

import numpy as np

from core.foundation.signal import contiguous_runs


def phase_correspondence(views: dict, windows: dict, tolerance: float) -> dict:
    """检查共用阶段是否有两路时间覆盖；不识别动作或比较次数。"""
    keys = set().union(*(set(part) for part in windows.values()))
    result = {}
    for segment, phase in sorted(keys):
        key = (segment, phase)
        ranges = {
            view: [
                (float(series["times"][run[0]]), float(series["times"][run[-1]]))
                for run in contiguous_runs(windows[view][key])
            ]
            for view, series in views.items()
            if key in windows[view]
        }
        entry = {"reliable": False, "ranges": ranges, "reason": "同步机位缺少该阶段的时间覆盖"}
        if set(ranges) == {"front", "side"}:
            front, side = ranges["front"], ranges["side"]
            # 所有区间均来自同一份识别结果；这里核对录像是否覆盖该时间，
            # 不把采样造成的连续帧段数量当作动作次数。
            covers = all(
                any(lo <= a + tolerance and hi >= b - tolerance for lo, hi in target)
                for source, target in ((front, side), (side, front))
                for a, b in source
            )
            if covers:
                entry.update(reliable=True, reason="同步机位覆盖同一动作阶段")
            else:
                entry["reason"] = "同步机位缺少该阶段的完整时间覆盖"
        result[f"{segment}:{phase}"] = entry
    return result


def project_windows(windows: dict, source_times, target_times) -> dict:
    """把一路的相位窗口按时间投到同步录制的另一路：每段连续帧取另一路同一时间范围内的帧，超出另一路时长的段丢弃。"""
    source_times, target_times = np.asarray(source_times), np.asarray(target_times)
    projected = {}
    for key, indices in windows.items():
        runs = []
        for run in contiguous_runs(indices):
            lo, hi = source_times[run[[0, -1]]]
            if lo < target_times[0] - 1e-6 or hi > target_times[-1] + 1e-6:
                continue
            runs.append(np.flatnonzero((target_times >= lo - 1e-6) & (target_times <= hi + 1e-6)))
        if runs:
            projected[key] = np.unique(np.concatenate(runs))
    return projected
