"""保存一次公共动作识别的来源、完整事件、阶段及各机位覆盖事实。"""

from __future__ import annotations

from copy import deepcopy


def shared_action_record(
    action: str, stance: str, diagnostics: dict, alignment: dict, phase_sources: dict
) -> dict:
    """从公共时间轴直接生成可持久化记录；不从检查结论或少量证据图反推动作。

    事件和阶段仅保存来源机位的完整记录，各机位保留自己的覆盖与缺失原因。
    零事件、没有可用计时机位时仍保存选择依据和空列表，不能省略这次识别。
    """
    source = alignment.get("timingSource")
    source_diagnostics = diagnostics.get(source, {})
    # 投影机位中的事件副本属于同一次识别，不再作为另一份动作记录保存。
    shared_fields = {
        "events",
        "eventDetails",
        "phaseInstances",
        "projectedEventDetails",
        "projectedPhaseInstances",
    }
    return deepcopy(
        {
            "schemaVersion": 1,
            "action": action,
            "stance": stance,
            "inputViews": sorted(diagnostics),
            "timingSource": source,
            "eventCounts": {},
            **alignment,
            "events": source_diagnostics.get("events", []),
            "eventDetails": source_diagnostics.get("eventDetails", []),
            "phaseInstances": source_diagnostics.get("phaseInstances", []),
            "views": {
                view: {
                    **{key: value for key, value in data.items() if key not in shared_fields},
                    "phaseSource": phase_sources[view],
                }
                for view, data in diagnostics.items()
            },
        }
    )
