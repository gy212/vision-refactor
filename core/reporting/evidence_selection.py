"""从保存的检查依据中选择报告与结果页共用的代表画面。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.reporting.practice_model import NormalizedCheck


@dataclass(frozen=True)
class EvidenceSelection:
    """某条原始检查所使用的画面及其展示依据。"""

    ref: str
    check_id: str
    phase_label: str
    view: str
    highlight_joints: tuple[int, ...]
    purpose: str
    weight: float
    metadata_recorded: bool
    repetition: int | None = None
    timing_source: str = ""
    reference_for: str = ""


def _saved_selection(check: NormalizedCheck, detail: dict, catalog: dict) -> EvidenceSelection:
    ref = detail["ref"]
    return EvidenceSelection(
        ref=ref,
        check_id=str(detail.get("checkId", check.id)),
        phase_label=str(detail.get("phaseLabel", check.phase_label)),
        view=str(catalog.get(ref, {}).get("view", detail.get("view", ""))),
        highlight_joints=tuple(detail.get("highlightJoints", [])),
        purpose=str(detail.get("purpose", "unknown")),
        weight=float(detail.get("weight", 0.0)),
        metadata_recorded="highlightJoints" in detail,
        repetition=detail.get("repetition"),
        timing_source=str(detail.get("timingSource", "")),
        reference_for=str(detail.get("referenceFor", "")),
    )


def _legacy_selection(check: NormalizedCheck, ref: str, catalog: dict) -> EvidenceSelection:
    """旧记录仅使用当时保存的观测与权重，不调用当前规则补造部位。"""
    view = str(catalog.get(ref, {}).get("view", ref.partition(":")[0]))
    measurement = next((m for m in check.measurements if m.get("view") == view), {})
    observed = measurement.get("issueObserved")
    purpose = "violation" if observed is True else "observation" if observed is False else "unknown"
    fusion = check.fusion or {}
    weights = fusion.get("effectiveWeights") or fusion.get("baseWeights") or {}
    return EvidenceSelection(
        ref,
        check.id,
        check.phase_label,
        view,
        (),
        purpose,
        float(weights.get(view, measurement.get("baseWeight", 0.0))),
        False,
    )


def _member_candidates(check: NormalizedCheck, catalog: dict) -> list[EvidenceSelection]:
    refs = set(check.evidence_refs)
    saved = [d for d in check.evidence_details if isinstance(d, dict) and d.get("ref") in refs]
    result = [_saved_selection(check, detail, catalog) for detail in saved]
    saved_refs = {detail["ref"] for detail in saved}
    result.extend(
        _legacy_selection(check, ref, catalog)
        for ref in dict.fromkeys(check.evidence_refs)
        if ref not in saved_refs
    )
    # 支持结论的画面优先；共同测量的参照面保留；权重只使用记录保存值。
    priority = {"violation": 0, "diagnostic": 0, "reference": 1, "observation": 2, "unknown": 3}
    return sorted(result, key=lambda item: (priority.get(item.purpose, 3), -item.weight))


def _reference_matches(
    reference: EvidenceSelection, source: EvidenceSelection, catalog: dict[str, dict]
) -> bool:
    """新证据按已保存的主图引用匹配，旧证据只接受已保存的相同采样时刻。"""
    if reference.reference_for:
        return reference.reference_for == source.ref
    source_frame = catalog.get(source.ref, {})
    return (
        "timeSeconds" in source_frame
        and catalog.get(reference.ref, {}).get("timeSeconds") == source_frame["timeSeconds"]
    )


def _instance_representatives(
    group: list[EvidenceSelection], catalog: dict[str, dict]
) -> list[EvidenceSelection]:
    """每个机位取一张，双面测量参照必须属于所选主图的同一采样。"""
    primary = [candidate for candidate in group if candidate.purpose != "reference"]
    views = dict.fromkeys(candidate.view for candidate in primary)
    selected = [next(c for c in primary if c.view == name) for name in views]
    for candidate in group:
        if candidate.purpose != "reference" or candidate.view in views:
            continue
        source = next(
            (
                item
                for item in selected
                if item.purpose != "reference" and _reference_matches(candidate, item, catalog)
            ),
            None,
        )
        if source is not None:
            selected.append(replace(candidate, reference_for=source.ref))
            views[candidate.view] = None
    return selected


def select_evidence(
    check: NormalizedCheck, catalog: dict[str, dict], *, view: str | None = None
) -> list[EvidenceSelection]:
    """各成员和次数先保留问题依据及必要参照，再追加对照或同阶段第二张图。"""
    selected = []
    for member in check.members or (check,):
        candidates = _member_candidates(member, catalog)
        if view is not None:
            candidates = [candidate for candidate in candidates if candidate.view == view]
        for repetition in dict.fromkeys(candidate.repetition for candidate in candidates):
            group = [candidate for candidate in candidates if candidate.repetition == repetition]
            picks = _instance_representatives(group, catalog) if view is None else group[:1]
            if view is None and len(picks) == 1:
                extras = [c for c in group if c.view == picks[0].view and c not in picks]
                picks.extend(extras[:1])
            selected.extend(picks)
    return prioritize_evidence(selected)


def prioritize_evidence(selections: Sequence[EvidenceSelection]) -> list[EvidenceSelection]:
    """先覆盖各阶段的问题或诊断主图，再放其余主图、参照、对照和补图。"""
    representatives = set()
    supported_instances = set()
    covered_stages = set()
    purposes = {
        (item.check_id, item.repetition, item.ref): item.purpose
        for item in selections
        if item.purpose != "reference"
    }
    ranked = []
    for selected in selections:
        identity = (selected.check_id, selected.repetition, selected.view)
        instance = (selected.check_id, selected.repetition)
        if identity in representatives:
            priority = 4
        else:
            representatives.add(identity)
            if selected.purpose == "reference":
                source = (*instance, selected.reference_for)
                priority = 2 if purposes.get(source) in {"violation", "diagnostic"} else 3
            elif selected.purpose in {"violation", "diagnostic"}:
                stage = (selected.phase_label, selected.repetition, selected.purpose)
                if instance in supported_instances:
                    priority = 2
                else:
                    priority = 1 if stage in covered_stages else 0
                    covered_stages.add(stage)
                supported_instances.add(instance)
            else:
                priority = 3
        ranked.append((priority, selected))
    return [selected for _, selected in sorted(ranked, key=lambda item: item[0])]


def evidence_caption(selection: EvidenceSelection, evidence: dict) -> str:
    """解释画面对应的原始阶段和用途，不把参照面描述成独立发现问题。"""
    view = {"front": "正面", "side": "侧面"}.get(selection.view, selection.view)
    purpose = {
        "violation": "问题依据",
        "diagnostic": "未能判定的取帧",
        "reference": "测量参照",
        "observation": "观测画面",
        "unknown": "历史画面",
    }.get(selection.purpose, "历史画面")
    phase = selection.phase_label
    if selection.repetition is not None:
        phase = f"第{selection.repetition}次 · {phase}"
    caption = (
        f"{phase} · {view}{purpose} · "
        f"{evidence.get('timeSeconds', 0.0):.3f}s · 第{evidence.get('frame', 0)}帧"
    )
    if not selection.metadata_recorded:
        caption += "（旧记录未保存测量部位，未加问题标记）"
    if selection.timing_source and selection.timing_source != "own_view":
        caption += "（阶段时间来自同步机位）"
    return caption
