"""把保存的动作计数与时间对应记录转换为各报告入口共用的说明。"""

from __future__ import annotations

from collections import Counter

from core.evaluation.action_catalog import ACTIONS, PHASES

_VIEWS = {"front": "正面", "side": "侧面"}


def _count_text(counts: dict) -> str:
    return "、".join(f"{ACTIONS.get(action, action)}{count}次" for action, count in counts.items())


def _event_text(event: dict) -> str:
    name = ACTIONS.get(event.get("segment", ""), event.get("segment", "动作未记录"))
    times = []
    for key, label in (("start", "开始"), ("peak", "峰值"), ("end", "结束")):
        value = event.get(key)
        times.append(
            f"{label}{value:.3f}秒" if isinstance(value, (int, float)) else f"{label}时刻未记录"
        )
    ending = "；未观察到收回" if event.get("ending") == "held" else ""
    return f"{name}；{'，'.join(times)}{ending}"


def _view_lines(view: str, diagnostic: dict, counts: dict) -> list[str]:
    label = _VIEWS[view]
    source = diagnostic.get("timingSource")
    borrowed = source in _VIEWS and source != view
    events = diagnostic.get("eventDetails")
    projected = diagnostic.get("projectedEventDetails")
    lines = []
    if borrowed:
        lines.append(f"{label}阶段时间来自{_VIEWS[source]}，本路未独立计次。")
        for number, event in enumerate(projected or [], 1):
            lines.append(f"{label}共用事件{number}：{_event_text(event)}")
    elif events is not None or view in counts:
        saved_counts = counts.get(view)
        if saved_counts is None:
            saved_counts = dict(Counter(event["segment"] for event in events))
        summary = _count_text(saved_counts) or "0次"
        lines.append(f"{label}独立检测：{summary}。")
        for number, event in enumerate(events or [], 1):
            lines.append(f"{label}事件{number}：{_event_text(event)}")
        if events is None and saved_counts:
            lines.append(f"{label}未保存逐次动作时间。")
    return lines


def _correspondence_lines(diagnostics: dict, alignment: dict) -> list[str]:
    lines = []
    for pair in alignment.get("eventPairs", []):
        action = ACTIONS.get(pair["segment"], pair["segment"])
        lines.append(
            f"已对应：{action}，正面事件{pair['frontIndex'] + 1}与侧面事件{pair['sideIndex'] + 1}。"
        )
    for event in alignment.get("unmatchedEvents", []):
        view = event["view"]
        borrowing = [
            _VIEWS[name]
            for name, diagnostic in diagnostics.items()
            if name in _VIEWS and diagnostic.get("timingSource") == view and name != view
        ]
        reason = (
            f"供{'、'.join(borrowing)}共用阶段时间，未作独立事件配对"
            if borrowing
            else event.get("reason") or alignment.get("reason") or "未找到唯一对应事件"
        )
        lines.append(
            f"未独立对应：{_VIEWS.get(view, view)}事件{event['index'] + 1}，"
            f"{_event_text(event)}；原因：{reason}。"
        )
    for scope, phase in alignment.get("phases", {}).items():
        if phase.get("reliable") or not phase.get("reason"):
            continue
        segment, _, moment = scope.partition(":")
        lines.append(
            f"阶段对应受限：{ACTIONS.get(segment, segment)}/{PHASES.get(moment, moment)}；"
            f"{phase['reason']}。"
        )
    return lines


def format_timing_diagnostics(diagnostics: dict, alignment: dict) -> list[str]:
    """只解释保存的数据；未记录事件的旧结果不会被补写为零次。"""
    lines = []
    if alignment:
        method = {
            "video_timestamps": "录像时间",
            "shared_clock": "共同采集时钟",
            "action_peaks": "出拳峰值",
        }.get(alignment.get("method"), "对应方法未记录")
        parts = [f"时间对应：{method}"]
        offset = alignment.get("sideOffsetSeconds")
        if isinstance(offset, (int, float)):
            parts.append(f"侧面时间偏移{offset:.3f}秒")
        if "reliable" in alignment:
            parts.append("可对应" if alignment["reliable"] else "存在对应限制")
        if alignment.get("reason"):
            parts.append(str(alignment["reason"]))
        lines.append("；".join(parts) + "。")
    counts = alignment.get("eventCounts", {})
    for view in _VIEWS:
        lines.extend(_view_lines(view, diagnostics.get(view, {}), counts))
    lines.extend(_correspondence_lines(diagnostics, alignment))
    return lines
