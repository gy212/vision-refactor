# -*- coding: utf-8 -*-
"""学生练习报告的数据模型：把历史记录归一成结果页与图文报告共用的只读模型（原 ``feedback_report.build_report_model``）。

纯函数、无 Tk 依赖：检查项去重与守恒计数、七级标题摘要、同原因分组、相位时间线数据。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.evaluation.rules.sanda_v5 import ACTIONS, PHASES, STANCES, build_checks
from core.reporting.text import format_fusion

STATUS_LABELS = {
    "candidate": "发现疑似问题",
    "not_observed": "已检查未发现问题",
    "unable": "暂无法判断",
    "pending_rule": "需教师判断",
}
_REVIEW_LABELS = {"confirmed": "教师已确认", "revoked": "教师已撤销"}
_RERECORD_HINT = " 请重新录制：全身及拳脚入画、光线充足、减少遮挡，保留开始和结束实战式。"
_COMBO_SEGMENTS = {
    "straight_combo": ["front_straight", "rear_straight"],
    "hook_combo": ["front_hook", "rear_hook"],
}


@dataclass(frozen=True)
class StatusCounts:
    """Strictly conserved check item counters."""
    total_checks: int         # 原始去重检查项总数
    candidate_count: int      # 当前未撤销候选数 (unconfirmed + confirmed)
    confirmed_count: int      # 教师已确认数
    unconfirmed_count: int    # 待教师确认数
    revoked_count: int        # 教师已撤销数
    not_observed_count: int   # 已检查未发现问题数
    unable_count: int         # 暂无法判断数
    pending_rule_count: int   # 需教师判断数

    def verify_conservation(self) -> bool:
        """Verify strict item conservation."""
        cat_sum = (self.candidate_count + self.not_observed_count +
                   self.unable_count + self.pending_rule_count + self.revoked_count)
        sub_sum = self.unconfirmed_count + self.confirmed_count
        return cat_sum == self.total_checks and sub_sum == self.candidate_count


@dataclass
class ReportHeader:
    """Header metadata desensitized for presentation."""
    student_id: str
    student_name: str
    action: str
    action_label: str
    stance: str
    stance_label: str
    created_at_utc: str
    created_at_local: str
    record_id: str
    revision: int
    rule_version: str
    backend_description: str


@dataclass
class HeadlineSummary:
    """Headline and priority status tone."""
    title: str               # 主标题文案
    subtitle: str            # 辅标题/引导说明
    needs_rerecord: bool     # 是否需要重新录制
    status_tone: str         # "warning" | "caution" | "notice" | "info" | "success"
    level: str               # "error" | "warning" | "info" | "success"
    reason: str              # 决策原因


@dataclass
class NormalizedCheck:
    """Clean, normalized check entry with diagnostics and review state."""
    id: str
    code: str
    name: str
    body_part: str
    segment: str
    segment_label: str
    phase: str
    phase_label: str
    standard: str
    source: str
    status: str              # "candidate" | "not_observed" | "unable" | "pending_rule"
    status_display: str      # "发现疑似问题" | "已检查未发现问题" | "暂无法判断" | "需教师判断"
    review: str              # "pending" | "confirmed" | "revoked"
    review_display: str      # "" | "教师已确认" | "教师已撤销"
    reason: str
    blocked_reason: str
    evidence_refs: list[str] = field(default_factory=list)
    measurements_summary: str = ""
    measurements: list[dict] = field(default_factory=list)
    fusion: dict | None = None
    audit_logs: list[dict] = field(default_factory=list)


@dataclass
class PhaseTimelineItem:
    """Timeline stage representation."""
    segment: str
    phase: str
    title: str
    time_range_text: str
    start_seconds: float
    end_seconds: float
    candidate_count: int
    unable_count: int
    pending_count: int
    not_observed_count: int
    checks: list[NormalizedCheck] = field(default_factory=list)


@dataclass
class GroupedReason:
    """Cluster of checks sharing the identical reason."""
    reason: str
    phase: str
    view: str
    count: int
    checks: list[NormalizedCheck] = field(default_factory=list)


@dataclass
class NormalizedReportModel:
    """Root model for UI result panel and offline HTML export."""
    header: ReportHeader
    headline: HeadlineSummary
    counts: StatusCounts
    phases: list[PhaseTimelineItem]
    candidate_groups: list[GroupedReason]
    candidate_checks: list[NormalizedCheck]   # Active (unconfirmed + confirmed)
    revoked_checks: list[NormalizedCheck]     # Excluded from candidate list
    not_observed_checks: list[NormalizedCheck]
    unable_checks: list[NormalizedCheck]
    pending_rule_checks: list[NormalizedCheck]
    all_checks: list[NormalizedCheck]         # Deduplicated by id
    evidence_catalog: dict[str, dict]         # Keyed by evidence ID
    diagnostics: dict[str, Any]
    capture: dict[str, Any]
    alignment: dict[str, Any]
    needs_rerecord: bool

    @property
    def candidate_items(self) -> list[NormalizedCheck]:
        return self.candidate_checks

    @property
    def revoked_items(self) -> list[NormalizedCheck]:
        return self.revoked_checks

    @property
    def not_observed_items(self) -> list[NormalizedCheck]:
        return self.not_observed_checks

    @property
    def unable_items(self) -> list[NormalizedCheck]:
        return self.unable_checks

    @property
    def pending_items(self) -> list[NormalizedCheck]:
        return self.pending_rule_checks


def format_local_datetime(iso_str: str) -> str:
    """Convert UTC ISO 8601 string to local time formatted string."""
    if not iso_str:
        return "时间未记录"
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local_dt = dt.astimezone()
        offset = local_dt.utcoffset()
        hours = int(offset.total_seconds() // 3600) if offset else 0
        tz_str = f"UTC{hours:+d}" if hours != 0 else "UTC"
        return local_dt.strftime("%Y-%m-%d %H:%M:%S") + f" ({tz_str})"
    except Exception:
        return str(iso_str)


def _headline(title: str, subtitle: str, needs_rerecord: bool, tone: str, level: str, reason: str,
              *, rerecord_hint: bool = True) -> HeadlineSummary:
    """组装标题摘要；需要重录时在副标题后追加统一的重录提示。"""
    if rerecord_hint and needs_rerecord:
        subtitle += _RERECORD_HINT
    return HeadlineSummary(title=title, subtitle=subtitle, needs_rerecord=needs_rerecord,
                           status_tone=tone, level=level, reason=reason)


def _is_unjudgeable(counts: StatusCounts) -> bool:
    """所有可测项（不含已撤销）都无法判断。"""
    measurable = counts.candidate_count + counts.not_observed_count + counts.unable_count
    return ((counts.candidate_count == 0 and counts.not_observed_count == 0 and counts.unable_count > 0)
            or (measurable > 0 and counts.unable_count == measurable)
            or (counts.total_checks > 0 and counts.unable_count == counts.total_checks))


def _candidate_headline(counts: StatusCounts, needs_rerecord: bool) -> HeadlineSummary:
    """存在未撤销候选问题时的标题（第 5、6 级）。"""
    confirmed_note = f"（含教师已确认 {counts.confirmed_count} 项）" if counts.confirmed_count > 0 else ""
    if counts.unable_count > 0:
        return _headline(
            "分析完成，部分项目暂无法判断",
            f"优先展示发现的 {counts.candidate_count} 项疑似问题{confirmed_note}；另有 {counts.unable_count} 项因视角或标定不足未判定。",
            needs_rerecord, "caution", "warning",
            f"发现 {counts.candidate_count} 项疑似问题，且有 {counts.unable_count} 项无法判断")
    return _headline(
        "发现疑似动作问题，请核对依据",
        f"总结共发现 {counts.candidate_count} 项疑似动作问题{confirmed_note}，请结合正侧代表帧与测量依据查看。",
        needs_rerecord, "caution", "warning", f"发现 {counts.candidate_count} 项疑似问题")


def get_headline_summary(counts: StatusCounts, checks: list, needs_rerecord: bool) -> HeadlineSummary:
    """7-Level priority headline hierarchy (no grades, no penalty points)."""
    c = counts
    if c.total_checks == 0:
        return _headline("暂无检查记录", "当前练习记录中未包含动作检查项数据。", needs_rerecord,
                         "info", "info", "记录中无检查项", rerecord_hint=False)
    if _is_unjudgeable(c):
        return HeadlineSummary(
            title="本次未能完成有效判断",
            subtitle="画面有效关键点不足或遮挡严重，无法判定动作对错。建议调整机位与光线后重新录制。",
            needs_rerecord=True, status_tone="unable", level="error",
            reason="所有可自动检查项目均为证据不足或无法判定")
    if c.candidate_count == 0 and c.not_observed_count == 0 and c.pending_rule_count > 0 and c.unable_count == 0:
        return _headline("本次需教师判断",
                         "本次动作包含需要人工复核的标准（如拳峰朝向或含胸收腹），暂无自动判断项。",
                         needs_rerecord, "info", "info", "本次动作仅包含需人工复核项，无自动判断项")
    if c.candidate_count == 0 and c.pending_rule_count > 0:
        return _headline("已检查项目未发现问题，仍有待判断项目",
                         f"已完成检查的项未见异常，但尚有 {c.pending_rule_count} 项需教师人工确认，请教师核对。",
                         needs_rerecord, "notice", "info", "自动检查项目未见异常，尚有需教师复核项目")
    if c.candidate_count == 0 and c.unable_count > 0 and c.not_observed_count > 0:
        return _headline("分析完成，部分项目暂无法判断",
                         f"已检查项目未见异常，另有 {c.unable_count} 项因缺少真实长度标定或有效帧不足无法判断。",
                         needs_rerecord, "notice", "warning", "已检查项目未发现异常，部分项目因证据不足无法判定")
    if c.candidate_count > 0:
        return _candidate_headline(c, needs_rerecord)
    if c.unable_count == 0 and c.pending_rule_count == 0 and c.not_observed_count > 0:
        return _headline("已检查项目未发现问题",
                         "已按现有规则检查完毕，未发现明显动作问题（二维规则仅供参考，不代表全部实战标准合格）。",
                         needs_rerecord, "success", "success", "所有检查项目均未发现异常")
    if c.revoked_count > 0:
        return _headline("候选问题已全部撤销", "教师已撤销所有候选问题，暂无其他判定项。",
                         needs_rerecord, "info", "info", "所有候选问题均已被教师撤销", rerecord_hint=False)
    return _headline("暂无有效判定结果", "检查项目未能匹配已知判定规则。",
                     needs_rerecord, "info", "info", "未匹配到具体判定分类", rerecord_hint=False)


def group_checks_by_reason(checks: list[NormalizedCheck]) -> list[GroupedReason]:
    """Group checks sharing identical reason within the same phase/view into collapsible clusters."""
    groups_dict: dict[tuple[str, str], list[NormalizedCheck]] = {}
    for c in checks:
        key_reason = (c.reason or c.blocked_reason or "未知原因").strip()
        key_phase = c.phase_label or c.phase
        groups_dict.setdefault((key_reason, key_phase), []).append(c)

    grouped_list: list[GroupedReason] = []
    for (reason_text, phase_text), item_list in groups_dict.items():
        views = set()
        for item in item_list:
            for ref in item.evidence_refs:
                if ":" in ref:
                    views.add(ref.split(":", 1)[0])
        view_label = "正面/侧面" if len(views) > 1 else ({"front": "正面", "side": "侧面"}.get(next(iter(views)), "双视角") if views else "全部")
        grouped_list.append(GroupedReason(reason=reason_text, phase=phase_text, view=view_label,
                                          count=len(item_list), checks=item_list))

    grouped_list.sort(key=lambda g: -g.count)
    return grouped_list


def _text_or(value: Any, default: str) -> str:
    return default if value is None else str(value)


def _build_header(record: dict, result: dict) -> ReportHeader:
    """记录头：学员、动作、时间、版本与分析后端说明。"""
    action = str(record.get("action") or "stance")
    stance = str(record.get("stance") or "left")
    created_at_utc = str(record.get("createdAt") or "")
    try:
        revision = int(record.get("revision", 1))
    except (ValueError, TypeError):
        revision = 1
    backend = result.get("backend", "mediapipe")
    backend_desc = ("MediaPipe关键点规则检查；二维投影与工程阈值待标定，不作为正式成绩。"
                    if backend == "mediapipe" else "旧版规则检查记录；可重新分析生成MediaPipe规则结果。")
    return ReportHeader(
        student_id=_text_or(record.get("studentId"), "未登记学号"),
        student_name=_text_or(record.get("studentName"), ""),
        action=action,
        action_label=ACTIONS.get(action, action),
        stance=stance,
        stance_label=STANCES.get(stance, stance),
        created_at_utc=created_at_utc,
        created_at_local=format_local_datetime(created_at_utc),
        record_id=_text_or(record.get("id"), "0" * 32),
        revision=revision,
        rule_version=_text_or(result.get("ruleVersion"), "sanda-feedback-mediapipe-legacy"),
        backend_description=backend_desc,
    )


def _raw_checks(result: dict, action: str, stance: str) -> list[dict]:
    """记录里的检查项；旧记录没有时按动作目录生成全部"无法判断"的占位项。"""
    raw_checks = result.get("checks")
    if isinstance(raw_checks, list) and raw_checks:
        return raw_checks
    placeholders = []
    for bc in build_checks(action, stance):
        item = dict(bc)
        item["status"] = "unable"
        item["reason"] = item.get("blockedReason") or "旧版记录未包含详细规则检查清单"
        item["review"] = "pending"
        placeholders.append(item)
    return placeholders


def _review_map(result: dict) -> dict[str, list[dict]]:
    review_map: dict[str, list[dict]] = {}
    review_history = result.get("reviewHistory", [])
    if isinstance(review_history, list):
        for entry in review_history:
            if isinstance(entry, dict) and "checkId" in entry:
                review_map.setdefault(entry["checkId"], []).append(entry)
    return review_map


def _normalize_check(cid: str, c: dict, audit_logs: list[dict]) -> NormalizedCheck:
    """单个检查项归一：状态与复核取值兜底、原因回退、测量摘要。"""
    status = c.get("status")
    if status not in STATUS_LABELS:
        status = "unable"
    review = c.get("review", "pending")
    if review not in {"pending", "confirmed", "revoked"}:
        review = "pending"
    blocked_reason = c.get("blockedReason", "")
    reason = c.get("reason", "") or blocked_reason or "旧版记录未提供此信息"
    measurements = c.get("measurements", [])
    if not isinstance(measurements, list):
        measurements = []
    fusion = c.get("fusion")
    if not isinstance(fusion, dict):
        fusion = None
    return NormalizedCheck(
        id=cid,
        code=str(c.get("code") or ""),
        name=_text_or(c.get("name"), "未命名检查项"),
        body_part=_text_or(c.get("bodyPart"), "未指定部位"),
        segment=str(c.get("segment") or "stance"),
        segment_label=str(c.get("segmentLabel") or ACTIONS.get(c.get("segment", ""), c.get("segment", ""))),
        phase=str(c.get("phase") or "motion"),
        phase_label=str(c.get("phaseLabel") or PHASES.get(c.get("phase", ""), c.get("phase", ""))),
        standard=str(c.get("standard") or ""),
        source=str(c.get("source") or ""),
        status=status,
        status_display=STATUS_LABELS[status],
        review=review,
        review_display=_REVIEW_LABELS.get(review, ""),
        reason=reason,
        blocked_reason=blocked_reason,
        evidence_refs=c.get("evidence", []) if isinstance(c.get("evidence"), list) else [],
        measurements_summary=format_fusion(c) if fusion or measurements else "",
        measurements=measurements,
        fusion=fusion,
        audit_logs=audit_logs,
    )


@dataclass
class _Buckets:
    """按守恒律分桶的检查项。"""
    candidate: list[NormalizedCheck] = field(default_factory=list)
    revoked: list[NormalizedCheck] = field(default_factory=list)
    not_observed: list[NormalizedCheck] = field(default_factory=list)
    unable: list[NormalizedCheck] = field(default_factory=list)
    pending_rule: list[NormalizedCheck] = field(default_factory=list)

    def add(self, check: NormalizedCheck) -> None:
        if check.status == "candidate":
            (self.revoked if check.review == "revoked" else self.candidate).append(check)
        else:
            getattr(self, check.status).append(check)

    def counts(self, total: int) -> StatusCounts:
        confirmed = sum(1 for c in self.candidate if c.review == "confirmed")
        return StatusCounts(
            total_checks=total,
            candidate_count=len(self.candidate),
            confirmed_count=confirmed,
            unconfirmed_count=len(self.candidate) - confirmed,
            revoked_count=len(self.revoked),
            not_observed_count=len(self.not_observed),
            unable_count=len(self.unable),
            pending_rule_count=len(self.pending_rule),
        )


def _ordered_stages(action: str) -> list[tuple[str, str, str]]:
    """时间线的阶段顺序：开始实战式 →（各动作段）→ 结束实战式。"""
    if action == "stance":
        return [("stance", "start", "开始实战式"), ("stance", "end", "结束实战式")]
    stages = [("stance", "start", "开始实战式")]
    for seg in _COMBO_SEGMENTS.get(action, [action]):
        stages.append((seg, "segment", ACTIONS.get(seg, seg)))
    stages.append(("stance", "end", "结束实战式"))
    return stages


def _phase_times(phase_windows: dict) -> dict[tuple[str, str], list[float]]:
    phase_times: dict[tuple[str, str], list[float]] = {}
    for view_phases in phase_windows.values():
        if isinstance(view_phases, list):
            for pw in view_phases:
                key = (pw.get("segment", ""), pw.get("phase", ""))
                for r in pw.get("ranges", []):
                    if len(r) >= 2:
                        phase_times.setdefault(key, []).extend([float(r[0]), float(r[1])])
    return phase_times


def build_phase_timeline(action: str, all_checks: list[NormalizedCheck], phase_windows: dict) -> list[PhaseTimelineItem]:
    """相位时间线数据：各阶段的起止时间与按状态计数的检查项。"""
    phase_times = _phase_times(phase_windows)
    items: list[PhaseTimelineItem] = []
    for seg, p_kind, stage_title in _ordered_stages(action):
        if p_kind == "segment":
            stage_checks = [c for c in all_checks if c.segment == seg and c.phase in {"motion", "finish"}]
            times_list = phase_times.get((seg, "motion"), []) + phase_times.get((seg, "finish"), [])
        else:
            stage_checks = [c for c in all_checks if c.segment == seg and c.phase == p_kind]
            times_list = phase_times.get((seg, p_kind), [])
        if times_list:
            t_min, t_max = min(times_list), max(times_list)
            time_text = f"{t_min:.2f}s ～ {t_max:.2f}s"
        else:
            t_min, t_max = 0.0, 0.0
            time_text = "时间段未记录"
        items.append(PhaseTimelineItem(
            segment=seg, phase=p_kind, title=stage_title, time_range_text=time_text,
            start_seconds=t_min, end_seconds=t_max,
            candidate_count=sum(1 for c in stage_checks if c.status == "candidate" and c.review != "revoked"),
            unable_count=sum(1 for c in stage_checks if c.status == "unable"),
            pending_count=sum(1 for c in stage_checks if c.status == "pending_rule"),
            not_observed_count=sum(1 for c in stage_checks if c.status == "not_observed"),
            checks=stage_checks,
        ))
    return items


def _dict_field(result: dict, key: str) -> dict:
    value = result.get(key)
    return value if isinstance(value, dict) else {}


def build_report_model(record: dict) -> NormalizedReportModel:
    """Pure reduction pipeline: normalizes record into immutable presentation model."""
    result = record.get("result")
    if not isinstance(result, dict):
        result = {}
    header = _build_header(record, result)
    needs_rerecord = bool(result.get("needsRerecord", False))

    dedup: dict[str, dict] = {}
    for c in _raw_checks(result, header.action, header.stance):  # 按 id 去重，后出现者覆盖
        cid = c.get("id")
        if cid:
            dedup[cid] = c
    review_map = _review_map(result)
    all_checks = [_normalize_check(cid, c, review_map.get(cid, [])) for cid, c in dedup.items()]
    buckets = _Buckets()
    for check in all_checks:
        buckets.add(check)
    counts = buckets.counts(len(all_checks))
    assert counts.verify_conservation(), f"Conservation invariant violated: {counts}"

    evidence_catalog: dict[str, dict] = {}
    for ev in result.get("evidence", []):
        if isinstance(ev, dict) and "id" in ev:
            evidence_catalog[ev["id"]] = ev

    return NormalizedReportModel(
        header=header,
        headline=get_headline_summary(counts, all_checks, needs_rerecord),
        counts=counts,
        phases=build_phase_timeline(header.action, all_checks, _dict_field(result, "phaseWindows")),
        candidate_groups=group_checks_by_reason(buckets.candidate),
        candidate_checks=buckets.candidate,
        revoked_checks=buckets.revoked,
        not_observed_checks=buckets.not_observed,
        unable_checks=buckets.unable,
        pending_rule_checks=buckets.pending_rule,
        all_checks=all_checks,
        evidence_catalog=evidence_catalog,
        diagnostics=_dict_field(result, "diagnostics"),
        capture=_dict_field(result, "capture"),
        alignment=_dict_field(result, "alignment"),
        needs_rerecord=needs_rerecord,
    )
