"""将一份动作阶段投影到同步时间轴，并逐实例核对各机位的实际覆盖。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from core.evaluation.phase_instances import PhaseInstance
from core.foundation.signal import contiguous_runs

_VIEW_LABELS = {"front": "正面", "side": "侧面"}
_PHASE_LABELS = {
    "start": "起始护位",
    "end": "结束护位",
    "motion": "动作过程",
    "finish": "击出",
    "prepare": "准备",
    "recovered": "收回",
}


def _expected_instances(windows: dict, source_times, instances: Sequence[PhaseInstance]) -> dict:
    """保留识别来源的动作身份；无动作次数的站姿窗口按自己的时间范围登记。"""
    grouped = {key: [] for key in windows}
    for instance in instances:
        key = (instance["segment"], instance["phase"])
        grouped.setdefault(key, []).append(instance)
    times = np.asarray(source_times)
    for key, indices in windows.items():
        if not grouped[key]:
            grouped[key] = [
                {
                    "segment": key[0],
                    "phase": key[1],
                    "repetition": None,
                    "start": float(times[run[0]]),
                    "end": float(times[run[-1]]),
                }
                for run in contiguous_runs(indices)
            ]
    return grouped


def _instance_coverage(instance: dict, times: np.ndarray, interval: float) -> dict:
    """只使用本实例时间范围内的样本，采样误差不能代替缺失的一帧或另一整次。"""
    start, end = instance["start"], instance["end"]
    samples = times[(times >= start - 1e-6) & (times <= end + 1e-6)]
    covered = bool(len(samples))
    if covered:
        # 差一个完整采样周期，意味着该机位本应有的帧缺失；不足一个周期才是边界取样误差。
        edge = max(interval - 1e-6, 1e-6)
        covered = bool(
            samples[0] - start <= edge
            and end - samples[-1] <= edge
            and np.all(np.diff(samples) <= interval + 1e-6)
        )
    return {
        **instance,
        "covered": covered,
        "observedFrames": len(samples),
        "observedStart": float(samples[0]) if len(samples) else None,
        "observedEnd": float(samples[-1]) if len(samples) else None,
    }


def _view_coverage(view: str, times: np.ndarray, instances: list[dict]) -> dict:
    interval = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    records = [_instance_coverage(instance, times, interval) for instance in instances]
    missing = [record for record in records if not record["covered"]]
    descriptions = []
    for record in missing:
        repetition = record["repetition"]
        identity = f"第 {repetition} 次动作的" if repetition is not None else ""
        phase = _PHASE_LABELS[record["phase"]]
        descriptions.append(
            f"{identity}{phase}阶段（{record['start']:.3f}–{record['end']:.3f} 秒）"
        )
    return {
        "complete": not missing,
        "samplingIntervalSeconds": interval,
        "instances": records,
        "reason": _VIEW_LABELS[view] + "录像未完整覆盖" + "、".join(descriptions)
        if missing
        else "",
    }


def phase_correspondence(
    views: dict,
    windows: dict,
    source: str,
    source_windows: dict,
    instances: Sequence[PhaseInstance],
) -> dict:
    """按唯一来源的同一次阶段检查录像覆盖；所有规则入口共用这些事实。"""
    expected = _expected_instances(source_windows, views[source]["times"], instances)
    result = {}
    for (segment, phase), phases in sorted(expected.items()):
        coverage = {
            view: _view_coverage(view, np.asarray(series["times"]), phases)
            for view, series in views.items()
        }
        ranges = {
            view: [
                (float(series["times"][run[0]]), float(series["times"][run[-1]]))
                for run in contiguous_runs(windows[view][(segment, phase)])
            ]
            for view, series in views.items()
            if (segment, phase) in windows[view]
        }
        reasons = [item["reason"] for item in coverage.values() if not item["complete"]]
        reliable = set(coverage) == {"front", "side"} and not reasons
        result[f"{segment}:{phase}"] = {
            "reliable": reliable,
            "ranges": ranges,
            "views": coverage,
            "reason": "；".join(reasons)
            or ("同步机位覆盖同一动作阶段" if reliable else "同步机位缺少该阶段的时间覆盖"),
        }
    return result


def missing_phase_coverage(coverage: dict, scope: str, views: Sequence[str]) -> str:
    """返回规则实际依赖机位的缺失阶段；未识别出的阶段仍由规则引擎说明。"""
    if scope not in coverage:
        return ""
    per_view = coverage[scope]["views"]
    return "；".join(
        per_view[view]["reason"] if view in per_view else f"缺少{_VIEW_LABELS[view]}视频"
        for view in views
        if view not in per_view or not per_view[view]["complete"]
    )


def project_windows(
    windows: dict, source_times, target_times, instances: Sequence[PhaseInstance] = ()
) -> dict:
    """逐来源阶段读取目标机位同一时段的帧；保留已有画面，完整性另按同次阶段检查。"""
    target_times = np.asarray(target_times)
    projected = {}
    for key, phases in _expected_instances(windows, source_times, instances).items():
        chunks = [
            np.flatnonzero(
                (target_times >= instance["start"] - 1e-6)
                & (target_times <= instance["end"] + 1e-6)
            )
            for instance in phases
        ]
        indices = np.unique(np.concatenate(chunks))
        if len(indices):
            projected[key] = indices
    return projected
