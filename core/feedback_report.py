"""Student practice feedback report normalization, SVG timeline and self-contained HTML export.

Pure function module: zero Tkinter imports, deterministic reductions, strict item conservation,
7-level headline priority hierarchy, diagram-design compliant inline SVG timeline, and
100% self-contained offline HTML report generation.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import html
import io
import math
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

from core.action_feedback import ACTIONS, STANCES, PHASES, build_checks, format_fusion

# Color tokens adhering to Style A and diagram-design editorial palette
COLOR_CANDIDATE = "#92400E"       # Foreground alert
COLOR_CANDIDATE_BG = "#FFF7ED"    # Light bg
COLOR_CANDIDATE_BORDER = "#FED7AA"# Border
COLOR_NOT_OBSERVED = "#166534"    # Success green
COLOR_NOT_OBSERVED_BG = "#F0FDF4"
COLOR_NOT_OBSERVED_BORDER = "#BBF7D0"
COLOR_UNABLE = "#475569"          # Muted slate
COLOR_UNABLE_BG = "#F1F5F9"
COLOR_UNABLE_BORDER = "#E2E8F0"
COLOR_PENDING = "#1D4ED8"         # Brand blue
COLOR_PENDING_BG = "#EFF6FF"
COLOR_PENDING_BORDER = "#BFDBFE"
COLOR_REVOKED = "#6B7280"         # Gray
COLOR_REVOKED_BG = "#F9FAFB"
COLOR_REVOKED_BORDER = "#E5E7EB"

# Mapping of rule codes to joints for focused skeleton overlay
RULE_JOINTS: dict[str, set[int]] = {
    "stance_elbow": {11, 13, 15, 12, 14, 16},
    "stance_rear_guard": {11, 12, 13, 14, 15, 16, 9, 10, 23, 24},
    "guard_low": {11, 12, 13, 14, 15, 16, 9, 10},
    "guard_back": {11, 12, 13, 14, 15, 16, 23, 24, 0},
    "guard_elbow": {11, 12, 13, 14},
    "shoulder_level": {11, 12, 13, 14, 15, 16},
    "guard_recover": {11, 12, 13, 14, 15, 16, 9, 10},
    "stance_knee": {23, 24, 25, 26, 27, 28},
    "stance_toes": {27, 28, 29, 30, 31, 32},
    "stance_torso": {11, 12, 23, 24},
    "foot_flexed": {25, 26, 27, 28, 29, 30, 31, 32},
    "hip_turn": {11, 12, 23, 24, 25, 26},
    "arm_swing": {11, 12, 13, 14, 15, 16},
}

SKELETON_BONES = [
    (11, 13), (13, 15), (12, 14), (14, 16), (11, 12),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    (9, 10), (0, 9), (0, 10)
]


WIN_UNC_PATH_RE = re.compile(
    r'(?:[A-Za-z]:[\\/]|\\\\)(?:[^\\/\r\n\"\'<>,;，。]+[\\/])*(?:[^\\/\r\n\"\'<>,;，。]*\.[a-zA-Z0-9_]+|[^ \t\\/\r\n\"\'<>,;，。]+)'
)
UNIX_PATH_RE = re.compile(
    r'/(?:home|Users)/(?:[^\\/\r\n\"\'<>,;，。]+[\\/])*(?:[^\\/\r\n\"\'<>,;，。]*\.[a-zA-Z0-9_]+|[^ \t\\/\r\n\"\'<>,;，。]+)'
)


def desensitize_paths(text: str) -> str:
    """Strip or replace Windows and Unix absolute filesystem paths to avoid machine leaks."""
    if not isinstance(text, str):
        return text
    cleaned = WIN_UNC_PATH_RE.sub("[已脱敏路径]", text)
    cleaned = UNIX_PATH_RE.sub("[已脱敏路径]", cleaned)
    return cleaned


def safe_clean(val: Any, default: str = "") -> str:
    """Safely handle None, desensitize filesystem paths, and HTML escape."""
    if val is None:
        val = default
    text = str(val)
    return html.escape(desensitize_paths(text))


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely coerce any value to float, falling back to default on error."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _is_valid_point(pt: Any) -> bool:
    """Validate that pt is a sequence of at least 2 finite numeric coordinates."""
    if not isinstance(pt, (list, tuple, np.ndarray)):
        return False
    if len(pt) < 2:
        return False
    try:
        px = float(pt[0])
        py = float(pt[1])
        return math.isfinite(px) and math.isfinite(py)
    except (ValueError, TypeError, IndexError):
        return False


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


def get_headline_summary(counts: StatusCounts, checks: list[dict], needs_rerecord: bool) -> HeadlineSummary:
    """7-Level priority headline hierarchy (no grades, no penalty points)."""
    # 0. 记录中无任何检查项
    if counts.total_checks == 0:
        return HeadlineSummary(
            title="暂无检查记录",
            subtitle="当前练习记录中未包含动作检查项数据。",
            needs_rerecord=needs_rerecord,
            status_tone="info",
            level="info",
            reason="记录中无检查项"
        )

    # 1. 全部可测项为 unable 或 所有自动项无法判断 (measurable_count does NOT include revoked_count)
    measurable_count = counts.candidate_count + counts.not_observed_count + counts.unable_count
    if (counts.candidate_count == 0 and counts.not_observed_count == 0 and counts.unable_count > 0) or (measurable_count > 0 and counts.unable_count == measurable_count) or (counts.total_checks > 0 and counts.unable_count == counts.total_checks):
        return HeadlineSummary(
            title="本次未能完成有效判断",
            subtitle="画面有效关键点不足或遮挡严重，无法判定动作对错。建议调整机位与光线后重新录制。",
            needs_rerecord=True,
            status_tone="unable",
            level="error",
            reason="所有可自动检查项目均为证据不足或无法判定"
        )

    # 2. 只有待人工确认项目（零活动候选，零未发现，全为 pending_rule，零 unable）
    if counts.candidate_count == 0 and counts.not_observed_count == 0 and counts.pending_rule_count > 0 and counts.unable_count == 0:
        sub = "本次动作包含需要人工复核的标准（如拳峰朝向或含胸收腹），暂无自动判断项。"
        if needs_rerecord:
            sub += " 请重新录制：全身及拳脚入画、光线充足、减少遮挡，保留开始和结束实战式。"
        return HeadlineSummary(
            title="本次需教师判断",
            subtitle=sub,
            needs_rerecord=needs_rerecord,
            status_tone="info",
            level="info",
            reason="本次动作仅包含需人工复核项，无自动判断项"
        )

    # 3. 零活动候选，存在 pending_rule
    if counts.candidate_count == 0 and counts.pending_rule_count > 0:
        sub = f"已完成检查的项未见异常，但尚有 {counts.pending_rule_count} 项需教师人工确认，请教师核对。"
        if needs_rerecord:
            sub += " 请重新录制：全身及拳脚入画、光线充足、减少遮挡，保留开始和结束实战式。"
        return HeadlineSummary(
            title="已检查项目未发现问题，仍有待判断项目",
            subtitle=sub,
            needs_rerecord=needs_rerecord,
            status_tone="notice",
            level="info",
            reason="自动检查项目未见异常，尚有需教师复核项目"
        )

    # 4. 零活动候选，存在 unable 且存在 not_observed
    if counts.candidate_count == 0 and counts.unable_count > 0 and counts.not_observed_count > 0:
        sub = f"已检查项目未见异常，另有 {counts.unable_count} 项因缺少真实长度标定或有效帧不足无法判断。"
        if needs_rerecord:
            sub += " 请重新录制：全身及拳脚入画、光线充足、减少遮挡，保留开始和结束实战式。"
        return HeadlineSummary(
            title="分析完成，部分项目暂无法判断",
            subtitle=sub,
            needs_rerecord=needs_rerecord,
            status_tone="notice",
            level="warning",
            reason="已检查项目未发现异常，部分项目因证据不足无法判定"
        )

    # 5. 存在活动候选问题，且存在 unable
    if counts.candidate_count > 0 and counts.unable_count > 0:
        confirmed_note = f"（含教师已确认 {counts.confirmed_count} 项）" if counts.confirmed_count > 0 else ""
        sub = f"优先展示发现的 {counts.candidate_count} 项疑似问题{confirmed_note}；另有 {counts.unable_count} 项因视角或标定不足未判定。"
        if needs_rerecord:
            sub += " 请重新录制：全身及拳脚入画、光线充足、减少遮挡，保留开始和结束实战式。"
        return HeadlineSummary(
            title="分析完成，部分项目暂无法判断",
            subtitle=sub,
            needs_rerecord=needs_rerecord,
            status_tone="caution",
            level="warning",
            reason=f"发现 {counts.candidate_count} 项疑似问题，且有 {counts.unable_count} 项无法判断"
        )

    # 6. 存在活动候选问题，且 unable == 0
    if counts.candidate_count > 0 and counts.unable_count == 0:
        confirmed_note = f"（含教师已确认 {counts.confirmed_count} 项）" if counts.confirmed_count > 0 else ""
        sub = f"总结共发现 {counts.candidate_count} 项疑似动作问题{confirmed_note}，请结合正侧代表帧与测量依据查看。"
        if needs_rerecord:
            sub += " 请重新录制：全身及拳脚入画、光线充足、减少遮挡，保留开始和结束实战式。"
        return HeadlineSummary(
            title="发现疑似动作问题，请核对依据",
            subtitle=sub,
            needs_rerecord=needs_rerecord,
            status_tone="caution",
            level="warning",
            reason=f"发现 {counts.candidate_count} 项疑似问题"
        )

    # 7. 全部已检查项目均为 not_observed (无活动候选，无 unable，无 pending_rule，且 not_observed > 0)
    if counts.candidate_count == 0 and counts.unable_count == 0 and counts.pending_rule_count == 0 and counts.not_observed_count > 0:
        sub = "已按现有规则检查完毕，未发现明显动作问题（二维规则仅供参考，不代表全部实战标准合格）。"
        if needs_rerecord:
            sub += " 请重新录制：全身及拳脚入画、光线充足、减少遮挡，保留开始和结束实战式。"
        return HeadlineSummary(
            title="已检查项目未发现问题",
            subtitle=sub,
            needs_rerecord=needs_rerecord,
            status_tone="success",
            level="success",
            reason="所有检查项目均未发现异常"
        )

    # 8. 教师撤销兜底（所有候选均已撤销，无未发现/无法判断/待教师判断项）
    if counts.revoked_count > 0 and counts.candidate_count == 0:
        return HeadlineSummary(
            title="候选问题已全部撤销",
            subtitle="教师已撤销所有候选问题，暂无其他判定项。",
            needs_rerecord=needs_rerecord,
            status_tone="info",
            level="info",
            reason="所有候选问题均已被教师撤销"
        )

    # 9. 极端兜底
    return HeadlineSummary(
        title="暂无有效判定结果",
        subtitle="检查项目未能匹配已知判定规则。",
        needs_rerecord=needs_rerecord,
        status_tone="info",
        level="info",
        reason="未匹配到具体判定分类"
    )


def group_checks_by_reason(checks: list[NormalizedCheck]) -> list[GroupedReason]:
    """Group checks sharing identical reason within the same phase/view into collapsible clusters."""
    groups_dict: dict[tuple[str, str], list[NormalizedCheck]] = {}
    for c in checks:
        key_reason = (c.reason or c.blocked_reason or "未知原因").strip()
        key_phase = c.phase_label or c.phase
        key = (key_reason, key_phase)
        if key not in groups_dict:
            groups_dict[key] = []
        groups_dict[key].append(c)

    grouped_list: list[GroupedReason] = []
    for (reason_text, phase_text), item_list in groups_dict.items():
        views = set()
        for item in item_list:
            for ref in item.evidence_refs:
                if ":" in ref:
                    views.add(ref.split(":", 1)[0])
        view_label = "正面/侧面" if len(views) > 1 else ({"front": "正面", "side": "侧面"}.get(next(iter(views)), "双视角") if views else "全部")
        grouped_list.append(GroupedReason(
            reason=reason_text,
            phase=phase_text,
            view=view_label,
            count=len(item_list),
            checks=item_list
        ))

    grouped_list.sort(key=lambda g: -g.count)
    return grouped_list


def build_report_model(record: dict) -> NormalizedReportModel:
    """Pure reduction pipeline: normalizes record into immutable presentation model."""
    record_id = record.get("id")
    record_id = "0" * 32 if record_id is None else str(record_id)
    student_id = record.get("studentId")
    student_id = "未登记学号" if student_id is None else str(student_id)
    student_name = record.get("studentName")
    student_name = "" if student_name is None else str(student_name)
    action = str(record.get("action") or "stance")
    action_label = ACTIONS.get(action, action)
    stance = str(record.get("stance") or "left")
    stance_label = STANCES.get(stance, stance)
    created_at_utc = str(record.get("createdAt") or "")
    created_at_local = format_local_datetime(created_at_utc)
    try:
        revision = int(record.get("revision", 1))
    except (ValueError, TypeError):
        revision = 1

    result = record.get("result")
    if not isinstance(result, dict):
        result = {}

    rule_version = result.get("ruleVersion")
    rule_version = "sanda-feedback-mediapipe-legacy" if rule_version is None else str(rule_version)
    backend = result.get("backend", "mediapipe")
    backend_desc = ("MediaPipe关键点规则检查；二维投影与工程阈值待标定，不作为正式成绩。"
                    if backend == "mediapipe" else "旧版规则检查记录；可重新分析生成MediaPipe规则结果。")

    needs_rerecord = bool(result.get("needsRerecord", False))
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    capture = result.get("capture") if isinstance(result.get("capture"), dict) else {}
    alignment = result.get("alignment") if isinstance(result.get("alignment"), dict) else {}
    phase_windows = result.get("phaseWindows") if isinstance(result.get("phaseWindows"), dict) else {}

    raw_checks = result.get("checks")
    if not isinstance(raw_checks, list) or len(raw_checks) == 0:
        base_checklist = build_checks(action, stance)
        raw_checks = []
        for bc in base_checklist:
            item = dict(bc)
            item["status"] = "unable"
            item["reason"] = item.get("blockedReason") or "旧版记录未包含详细规则检查清单"
            item["review"] = "pending"
            raw_checks.append(item)

    # Deduplicate checks by check.id (last occurrence wins)
    dedup_dict: dict[str, dict] = {}
    for c in raw_checks:
        cid = c.get("id")
        if cid:
            dedup_dict[cid] = c

    # Review history mapping
    review_history = result.get("reviewHistory", [])
    review_map: dict[str, list[dict]] = {}
    if isinstance(review_history, list):
        for entry in review_history:
            if isinstance(entry, dict) and "checkId" in entry:
                review_map.setdefault(entry["checkId"], []).append(entry)

    # Evidence catalog
    evidence_catalog: dict[str, dict] = {}
    for ev in result.get("evidence", []):
        if isinstance(ev, dict) and "id" in ev:
            evidence_catalog[ev["id"]] = ev

    # Normalize individual checks
    all_checks: list[NormalizedCheck] = []
    candidate_checks: list[NormalizedCheck] = []
    revoked_checks: list[NormalizedCheck] = []
    not_observed_checks: list[NormalizedCheck] = []
    unable_checks: list[NormalizedCheck] = []
    pending_rule_checks: list[NormalizedCheck] = []

    unconfirmed_count = 0
    confirmed_count = 0
    revoked_count = 0
    not_observed_count = 0
    unable_count = 0
    pending_rule_count = 0

    status_labels_map = {
        "candidate": "发现疑似问题",
        "not_observed": "已检查未发现问题",
        "unable": "暂无法判断",
        "pending_rule": "需教师判断"
    }

    for cid, c in dedup_dict.items():
        status = c.get("status")
        if status not in {"candidate", "not_observed", "unable", "pending_rule"}:
            status = "unable"

        review = c.get("review", "pending")
        if review not in {"pending", "confirmed", "revoked"}:
            review = "pending"

        reason = c.get("reason", "")
        blocked_reason = c.get("blockedReason", "")
        if not reason:
            reason = blocked_reason or "旧版记录未提供此信息"

        review_display = ""
        if review == "confirmed":
            review_display = "教师已确认"
        elif review == "revoked":
            review_display = "教师已撤销"

        measurements = c.get("measurements", [])
        if not isinstance(measurements, list):
            measurements = []

        fusion = c.get("fusion")
        if not isinstance(fusion, dict):
            fusion = None

        meas_summary = format_fusion(c) if fusion or measurements else ""

        chk_name = c.get("name")
        chk_name = "未命名检查项" if chk_name is None else str(chk_name)
        chk_body_part = c.get("bodyPart")
        chk_body_part = "未指定部位" if chk_body_part is None else str(chk_body_part)

        norm_check = NormalizedCheck(
            id=cid,
            code=str(c.get("code") or ""),
            name=chk_name,
            body_part=chk_body_part,
            segment=str(c.get("segment") or "stance"),
            segment_label=str(c.get("segmentLabel") or ACTIONS.get(c.get("segment", ""), c.get("segment", ""))),
            phase=str(c.get("phase") or "motion"),
            phase_label=str(c.get("phaseLabel") or PHASES.get(c.get("phase", ""), c.get("phase", ""))),
            standard=str(c.get("standard") or ""),
            source=str(c.get("source") or ""),
            status=status,
            status_display=status_labels_map[status],
            review=review,
            review_display=review_display,
            reason=reason,
            blocked_reason=blocked_reason,
            evidence_refs=c.get("evidence", []) if isinstance(c.get("evidence"), list) else [],
            measurements_summary=meas_summary,
            measurements=measurements,
            fusion=fusion,
            audit_logs=review_map.get(cid, [])
        )
        all_checks.append(norm_check)

        # Bucketing according to conservation law
        if status == "candidate":
            if review == "revoked":
                revoked_checks.append(norm_check)
                revoked_count += 1
            else:
                candidate_checks.append(norm_check)
                if review == "confirmed":
                    confirmed_count += 1
                else:
                    unconfirmed_count += 1
        elif status == "not_observed":
            not_observed_checks.append(norm_check)
            not_observed_count += 1
        elif status == "unable":
            unable_checks.append(norm_check)
            unable_count += 1
        elif status == "pending_rule":
            pending_rule_checks.append(norm_check)
            pending_rule_count += 1

    candidate_count = unconfirmed_count + confirmed_count
    total_checks = len(all_checks)

    counts = StatusCounts(
        total_checks=total_checks,
        candidate_count=candidate_count,
        confirmed_count=confirmed_count,
        unconfirmed_count=unconfirmed_count,
        revoked_count=revoked_count,
        not_observed_count=not_observed_count,
        unable_count=unable_count,
        pending_rule_count=pending_rule_count
    )
    assert counts.verify_conservation(), f"Conservation invariant violated: {counts}"

    # Headline Summary
    headline = get_headline_summary(counts, all_checks, needs_rerecord)

    # Reason grouping for active candidate checks
    candidate_groups = group_checks_by_reason(candidate_checks)

    segments_sequence = {"straight_combo": ["front_straight", "rear_straight"],
                         "hook_combo": ["front_hook", "rear_hook"]}.get(action, [action])
    if action == "stance":
        ordered_stages = [("stance", "start", "开始实战式"), ("stance", "end", "结束实战式")]
    else:
        ordered_stages = [("stance", "start", "开始实战式")]
        for seg in segments_sequence:
            ordered_stages.append((seg, "segment", ACTIONS.get(seg, seg)))
        ordered_stages.append(("stance", "end", "结束实战式"))

    # Extract time ranges from phase_windows
    phase_times: dict[tuple[str, str], list[float]] = {}
    for view_phases in phase_windows.values():
        if isinstance(view_phases, list):
            for pw in view_phases:
                s_key = pw.get("segment", "")
                p_key = pw.get("phase", "")
                for r in pw.get("ranges", []):
                    if len(r) >= 2:
                        phase_times.setdefault((s_key, p_key), []).extend([float(r[0]), float(r[1])])

    phase_items: list[PhaseTimelineItem] = []
    for seg, p_kind, stage_title in ordered_stages:
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

        p_cand = sum(1 for c in stage_checks if c.status == "candidate" and c.review != "revoked")
        p_unab = sum(1 for c in stage_checks if c.status == "unable")
        p_pend = sum(1 for c in stage_checks if c.status == "pending_rule")
        p_noto = sum(1 for c in stage_checks if c.status == "not_observed")

        phase_items.append(PhaseTimelineItem(
            segment=seg,
            phase=p_kind,
            title=stage_title,
            time_range_text=time_text,
            start_seconds=t_min,
            end_seconds=t_max,
            candidate_count=p_cand,
            unable_count=p_unab,
            pending_count=p_pend,
            not_observed_count=p_noto,
            checks=stage_checks
        ))

    header = ReportHeader(
        student_id=student_id,
        student_name=student_name,
        action=action,
        action_label=action_label,
        stance=stance,
        stance_label=stance_label,
        created_at_utc=created_at_utc,
        created_at_local=created_at_local,
        record_id=record_id,
        revision=revision,
        rule_version=rule_version,
        backend_description=backend_desc
    )

    return NormalizedReportModel(
        header=header,
        headline=headline,
        counts=counts,
        phases=phase_items,
        candidate_groups=candidate_groups,
        candidate_checks=candidate_checks,
        revoked_checks=revoked_checks,
        not_observed_checks=not_observed_checks,
        unable_checks=unable_checks,
        pending_rule_checks=pending_rule_checks,
        all_checks=all_checks,
        evidence_catalog=evidence_catalog,
        diagnostics=diagnostics,
        capture=capture,
        alignment=alignment,
        needs_rerecord=needs_rerecord
    )


def generate_phase_timeline_svg(phases: list[PhaseTimelineItem], counts: StatusCounts) -> str:
    """Generate self-contained inline SVG timeline compliant with diagram-design Process/Timeline standard.

    Adheres strictly to:
    - 2px stroke, round caps/joins (stroke-linecap="round" stroke-linejoin="round")
    - 4px grid alignment for structural bounds
    - Accessible contract (role="img", aria-labelledby, <title>, <desc>)
    - Zero external URLs or fonts (no http:// or https://)
    """
    n = len(phases)
    vb_w = 800
    vb_h = 132

    if n == 0:
        return (
            f'<svg viewBox="0 0 {vb_w} 80" width="100%" height="auto" '
            'role="img" aria-labelledby="flow-title flow-desc">\n'
            '  <title id="flow-title">动作阶段时序流图</title>\n'
            '  <desc id="flow-desc">暂无记录的动作阶段时序信息。</desc>\n'
            '  <rect width="100%" height="100%" rx="6" fill="#F8FAFC" stroke="#D8E0EA" stroke-width="1"/>\n'
            '  <text x="400" y="45" fill="#475569" font-size="12" text-anchor="middle" '
            'font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', \'Microsoft YaHei UI\', sans-serif">未记录动作阶段时间</text>\n'
            '</svg>'
        )

    margin_x = 24
    gap = 20
    avail_w = vb_w - 2 * margin_x - (n - 1) * gap
    raw_node_w = avail_w / n
    node_w = int(max(96, math.floor(raw_node_w / 4.0) * 4))
    node_h = 68
    node_y = 36
    center_y = node_y + node_h // 2

    total_needed_w = 2 * margin_x + n * node_w + (n - 1) * gap
    if total_needed_w > vb_w:
        vb_w = total_needed_w

    svg_parts = [
        f'<svg viewBox="0 0 {vb_w} {vb_h}" width="100%" height="auto" '
        'role="img" aria-labelledby="flow-title flow-desc">',
        '  <title id="flow-title">动作阶段时序流图</title>',
        '  <desc id="flow-desc">展示散打动作各阶段起止时间、检查项与疑似问题分布时序图。</desc>',
        '  <defs>',
        '    <marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">',
        '      <polygon points="0 0, 8 3, 0 6" fill="#475569"/>',
        '    </marker>',
        '    <marker id="arrow-warn" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">',
        '      <polygon points="0 0, 8 3, 0 6" fill="#92400E"/>',
        '    </marker>',
        '  </defs>',
        f'  <rect width="100%" height="100%" rx="6" fill="#F8FAFC" stroke="#D8E0EA" stroke-width="1"/>'
    ]

    node_coords = []
    for i in range(n):
        nx = margin_x + i * (node_w + gap)
        node_coords.append((nx, node_y))

    # Connectors drawn before nodes (z-order)
    for i in range(n - 1):
        x1 = node_coords[i][0] + node_w
        x2 = node_coords[i + 1][0]
        warn = (phases[i].candidate_count > 0 or phases[i + 1].candidate_count > 0)
        marker_id = "arrow-warn" if warn else "arrow"
        stroke_color = "#92400E" if warn else "#475569"
        svg_parts.append(
            f'  <line x1="{x1}" y1="{center_y}" x2="{x2}" y2="{center_y}" '
            f'stroke="{stroke_color}" stroke-width="2" stroke-linecap="round" marker-end="url(#{marker_id})"/>'
        )

    # Render Phase Nodes
    for i, phase in enumerate(phases):
        nx, ny = node_coords[i]
        step_num = f"{i + 1:02d}"
        cand_count = phase.candidate_count
        unable_count = phase.unable_count

        if cand_count > 0:
            fill_color = COLOR_CANDIDATE_BG
            stroke_color = COLOR_CANDIDATE_BORDER
            accent_line = COLOR_CANDIDATE
            badge_text = f"⚠ {cand_count}项"
            badge_fill = "#FEF3C7"
            badge_stroke = COLOR_CANDIDATE
        elif unable_count > 0 and cand_count == 0:
            fill_color = COLOR_UNABLE_BG
            stroke_color = COLOR_UNABLE_BORDER
            accent_line = COLOR_UNABLE
            badge_text = f"? {unable_count}项"
            badge_fill = "#E2E8F0"
            badge_stroke = COLOR_UNABLE
        else:
            fill_color = "#FFFFFF"
            stroke_color = COLOR_PENDING_BORDER if i == 0 else "#D8E0EA"
            accent_line = COLOR_PENDING if i == 0 else "#10B981"
            badge_text = "✓"
            badge_fill = COLOR_NOT_OBSERVED_BG
            badge_stroke = COLOR_NOT_OBSERVED

        svg_parts.append(f'  <!-- Stage {step_num}: {html.escape(phase.title)} -->')
        svg_parts.append(f'  <rect x="{nx}" y="{ny}" width="{node_w}" height="{node_h}" rx="6" fill="#FFFFFF"/>')
        svg_parts.append(
            f'  <rect x="{nx}" y="{ny}" width="{node_w}" height="{node_h}" rx="6" '
            f'fill="{fill_color}" stroke="{stroke_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        )

        svg_parts.append(
            f'  <line x1="{nx + 4}" y1="{ny + 8}" x2="{nx + 4}" y2="{ny + node_h - 8}" '
            f'stroke="{accent_line}" stroke-width="2" stroke-linecap="round"/>'
        )

        svg_parts.append(
            f'  <text x="{nx + 12}" y="{ny + 20}" fill="#475569" font-size="10" font-weight="600" '
            f'font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', \'Microsoft YaHei UI\', sans-serif">{step_num}</text>'
        )

        badge_w = 40 if len(badge_text) > 2 else 22
        badge_x = nx + node_w - badge_w - 6
        svg_parts.append(
            f'  <rect x="{badge_x}" y="{ny + 8}" width="{badge_w}" height="16" rx="4" '
            f'fill="{badge_fill}" stroke="{badge_stroke}" stroke-width="1"/>'
        )
        svg_parts.append(
            f'  <text x="{badge_x + badge_w // 2}" y="{ny + 20}" fill="{badge_stroke}" font-size="9" font-weight="600" '
            f'text-anchor="middle" font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', \'Microsoft YaHei UI\', sans-serif">{html.escape(badge_text)}</text>'
        )

        escaped_title = html.escape(phase.title)
        svg_parts.append(
            f'  <text x="{nx + 12}" y="{ny + 40}" fill="#1F2937" font-size="12" font-weight="600" '
            f'font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', \'Microsoft YaHei UI\', sans-serif">{escaped_title}</text>'
        )

        escaped_time = html.escape(phase.time_range_text)
        svg_parts.append(
            f'  <text x="{nx + 12}" y="{ny + 58}" fill="#64748B" font-size="10" '
            f'font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', \'Microsoft YaHei UI\', sans-serif">{escaped_time}</text>'
        )

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def resolve_record_video_path(history_root: Path, record_id: str, view: str, videos: dict) -> Path | None:
    """Safely resolve managed video path with strict directory traversal guards."""
    if not isinstance(record_id, str) or not re.fullmatch(r"[0-9a-f]{32}", record_id):
        return None
    directory = (history_root / record_id).resolve()
    try:
        if directory.parent != history_root.resolve() or directory.is_symlink():
            return None
    except Exception:
        return None

    filename = videos.get(view)
    if not filename or not isinstance(filename, str):
        return None

    video_path = (directory / filename).resolve()
    try:
        if video_path.parent != directory or not video_path.is_file() or video_path.is_symlink():
            return None
    except Exception:
        return None

    return video_path


def extract_evidence_frame(
    video_path: Path,
    frame_idx: int,
    landmarks: list | None = None,
    valid_mask: list | None = None,
    highlight_joints: set[int] | None = None
) -> np.ndarray | None:
    """Extract single frame from OpenCV video, paint height-normalized landmarks and skeleton.

    Coordinates scaling strictly enforces height-normalization:
    pixel_x = round(point[0] * h)
    pixel_y = round(point[1] * h)
    """
    if not isinstance(video_path, Path):
        video_path = Path(video_path)
    if not video_path.is_file() or video_path.is_symlink():
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
    except Exception:
        return None
    finally:
        cap.release()

    h, w = frame.shape[:2]
    if landmarks is not None and len(landmarks) > 0:
        pts = np.asarray(landmarks)
        vmask = valid_mask if valid_mask is not None else [True] * len(pts)

        # 1. Connect skeleton lines for highlighted joints
        if highlight_joints:
            for j1, j2 in SKELETON_BONES:
                if (j1 in highlight_joints and j2 in highlight_joints and
                        j1 < len(vmask) and j2 < len(vmask) and vmask[j1] and vmask[j2] and
                        j1 < len(pts) and j2 < len(pts)):
                    p1 = pts[j1]
                    p2 = pts[j2]
                    if _is_valid_point(p1) and _is_valid_point(p2):
                        x1 = int(round(float(p1[0]) * h))
                        y1 = int(round(float(p1[1]) * h))
                        x2 = int(round(float(p2[0]) * h))
                        y2 = int(round(float(p2[1]) * h))
                        if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                            cv2.line(frame, (x1, y1), (x2, y2), (235, 99, 37), max(2, h // 250))

        # 2. Draw circles for valid joints
        for idx in range(min(len(pts), len(vmask))):
            if vmask[idx]:
                pt = pts[idx]
                if _is_valid_point(pt):
                    px = int(round(float(pt[0]) * h))
                    py = int(round(float(pt[1]) * h))
                    if 0 <= px < w and 0 <= py < h:
                        is_highlighted = bool(highlight_joints and idx in highlight_joints)
                        if is_highlighted:
                            rad = max(4, h // 140)
                            cv2.circle(frame, (px, py), rad + 2, (255, 255, 255), 2)
                            cv2.circle(frame, (px, py), rad, (14, 64, 235), -1)
                        else:
                            rad = max(2, h // 300)
                            cv2.circle(frame, (px, py), rad, (0, 220, 100), -1)

    return frame


def frame_to_data_uri(frame: np.ndarray, max_edge: int = 480, quality: int = 80) -> str:
    """Convert OpenCV BGR image to JPEG Base64 Data URI."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    if max(pil_img.size) > max_edge:
        pil_img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64_str}"


def _get_svg_status_icon(status: str, size: int = 20) -> str:
    """Inline SVG icons for report states adhering to Style A geometry (no http://)."""
    if status == "candidate":
        return (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" role="img" aria-label="疑似问题">'
            f'<path d="M12 4L4 19.5H20L12 4Z" stroke="{COLOR_CANDIDATE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="{COLOR_CANDIDATE_BG}"/>'
            f'<line x1="12" y1="9" x2="12" y2="13.5" stroke="{COLOR_CANDIDATE}" stroke-width="2" stroke-linecap="round"/>'
            f'<circle cx="12" cy="16.5" r="1" fill="{COLOR_CANDIDATE}"/>'
            '</svg>'
        )
    elif status == "not_observed":
        return (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" role="img" aria-label="未发现问题">'
            f'<circle cx="12" cy="12" r="9.5" stroke="{COLOR_NOT_OBSERVED}" stroke-width="2" fill="{COLOR_NOT_OBSERVED_BG}"/>'
            f'<path d="M7 12L10.5 15.5L17 9" stroke="{COLOR_NOT_OBSERVED}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
            '</svg>'
        )
    elif status == "unable":
        return (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" role="img" aria-label="暂无法判断">'
            f'<circle cx="12" cy="12" r="9.5" stroke="{COLOR_UNABLE}" stroke-width="2" fill="{COLOR_UNABLE_BG}"/>'
            f'<path d="M9.5 9.5C9.5 7.8 10.6 6.5 12 6.5C13.5 6.5 14.5 7.6 14.5 9C14.5 10.5 12 11.5 12 13" stroke="{COLOR_UNABLE}" stroke-width="2" stroke-linecap="round"/>'
            f'<circle cx="12" cy="16.5" r="1" fill="{COLOR_UNABLE}"/>'
            '</svg>'
        )
    elif status == "pending_rule":
        return (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" role="img" aria-label="需教师判断">'
            f'<rect x="4.5" y="5.5" width="15" height="15" rx="2" stroke="{COLOR_PENDING}" stroke-width="2" fill="{COLOR_PENDING_BG}"/>'
            f'<path d="M8.5 3.5H15.5V6.5H8.5V3.5Z" stroke="{COLOR_PENDING}" stroke-width="1.8" fill="#FFFFFF"/>'
            f'<path d="M7.5 12.5L10.5 15.5L16.5 9.5" stroke="{COLOR_PENDING}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
            '</svg>'
        )
    elif status == "revoked":
        return (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" role="img" aria-label="已撤销">'
            f'<circle cx="12" cy="12" r="9.5" stroke="{COLOR_REVOKED}" stroke-width="2" fill="{COLOR_REVOKED_BG}"/>'
            f'<line x1="7" y1="12" x2="17" y2="12" stroke="{COLOR_REVOKED}" stroke-width="2" stroke-linecap="round"/>'
            '</svg>'
        )
    return ""


def render_html_report(
    record: dict,
    *,
    history_root: Path | None = None,
    max_image_edge: int = 480,
    max_images: int = 12
) -> str:
    """Render 100% offline, self-contained single-file HTML graphic report.

    Security & Offline Guarantees:
    - Zero external HTTP/HTTPS requests (no CDN, no web fonts, no external scripts).
    - Strict html.escape(quote=True) on all user and dynamic text.
    - Strips local absolute machine paths.
    - Embeds Base64 JPEG evidence thumbnails with memory and count limits.
    - Built-in @media print A4 printable stylesheet.
    """
    model = build_report_model(record)
    header = model.header
    counts = model.counts
    headline = model.headline

    # Timeline SVG
    timeline_svg = generate_phase_timeline_svg(model.phases, counts)

    # Resolve evidence images safely
    images_embedded_count = 0
    evidence_image_cache: dict[tuple[str, int, tuple[int, ...]], str] = {}

    def get_evidence_data_uri(ev_id: str, rule_code: str) -> tuple[str | None, str]:
        nonlocal images_embedded_count
        ev_data = model.evidence_catalog.get(ev_id)
        if not ev_data:
            return None, "未找到关联证据数据"
        if history_root is None:
            return None, "未指定历史目录，跳过截帧"

        view = ev_data.get("view", "front")
        frame_idx = ev_data.get("frame", 0)
        time_sec = ev_data.get("timeSeconds", 0.0)
        view_label = "正面" if view == "front" else "侧面"
        caption = f"{view_label}证据 · {time_sec:.3f}s · 第{frame_idx}帧"

        if images_embedded_count >= max_images:
            return None, f"{caption}（已省略次要截图以精简报告体积）"

        highlight_joints = tuple(sorted(RULE_JOINTS.get(rule_code, set())))
        cache_key = (view, frame_idx, highlight_joints)
        if cache_key in evidence_image_cache:
            return evidence_image_cache[cache_key], caption

        video_path = resolve_record_video_path(history_root, header.record_id, view, record.get("videos", {}))
        if not video_path:
            return None, f"{caption}（原录像文件不可用）"

        landmarks = ev_data.get("landmarks")
        vmask = ev_data.get("validMask")
        try:
            frame = extract_evidence_frame(
                video_path,
                frame_idx,
                landmarks=landmarks,
                valid_mask=vmask,
                highlight_joints=set(highlight_joints) if highlight_joints else None
            )
            if frame is None:
                return None, f"{caption}（截帧解码失败）"

            data_uri = frame_to_data_uri(frame, max_edge=max_image_edge)
            evidence_image_cache[cache_key] = data_uri
            images_embedded_count += 1
            return data_uri, caption
        except Exception:
            return None, f"{caption}（截帧解码失败）"

    safe_student_id = safe_clean(header.student_id, "未登记学号")
    safe_student_name = safe_clean(header.student_name)
    student_display = f"{safe_student_id} · {safe_student_name}" if header.student_name else safe_student_id
    action_display = f"{safe_clean(header.action_label)} · {safe_clean(header.stance_label)}"
    time_display = safe_clean(header.created_at_local)

    headline_class = f"banner-{headline.status_tone}"
    headline_title_escaped = safe_clean(headline.title)
    headline_subtitle_escaped = safe_clean(headline.subtitle)

    # HTML document generation
    doc = [
        '<!DOCTYPE html>',
        '<html lang="zh-CN">',
        '<head>',
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f'  <title>学生练习分析报告 - {safe_student_id}</title>',
        '  <style>',
        '    /* Style A 简洁教学风 - 离线专用打印友好样式表 */',
        '    :root {',
        '      --bg-page: #F5F7FB;',
        '      --bg-card: #FFFFFF;',
        '      --color-ink: #1F2937;',
        '      --color-muted: #475569;',
        '      --color-subtle: #64748B;',
        '      --color-border: #D8E0EA;',
        '      --color-primary: #2563EB;',
        '      --color-primary-bg: #EFF6FF;',
        '      --color-warn: #92400E;',
        '      --color-warn-bg: #FFF7ED;',
        '      --color-warn-border: #FED7AA;',
        '      --color-ok: #166534;',
        '      --color-ok-bg: #F0FDF4;',
        '      --color-ok-border: #BBF7D0;',
        '      --color-unable: #475569;',
        '      --color-unable-bg: #F1F5F9;',
        '    }',
        '    * { box-sizing: border-box; margin: 0; padding: 0; }',
        '    body {',
        '      background-color: var(--bg-page);',
        '      color: var(--color-ink);',
        '      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;',
        '      font-size: 14px;',
        '      line-height: 1.6;',
        '      padding: 24px;',
        '    }',
        '    .report-container {',
        '      max-width: 980px;',
        '      margin: 0 auto;',
        '      background: var(--bg-card);',
        '      border: 1px solid var(--color-border);',
        '      border-radius: 8px;',
        '      padding: 32px;',
        '    }',
        '    .header-bar {',
        '      display: flex;',
        '      justify-content: space-between;',
        '      align-items: flex-start;',
        '      padding-bottom: 20px;',
        '      border-bottom: 1px solid var(--color-border);',
        '      margin-bottom: 24px;',
        '    }',
        '    .header-meta h1 { font-size: 20px; font-weight: 700; color: var(--color-ink); margin-bottom: 6px; }',
        '    .header-meta .student-info { font-size: 14px; font-weight: 600; color: var(--color-ink); margin-bottom: 4px; }',
        '    .header-meta .action-info { font-size: 13px; color: var(--color-muted); }',
        '    .header-date { text-align: right; font-size: 12px; color: var(--color-subtle); line-height: 1.8; }',
        '    .headline-banner {',
        '      padding: 16px 20px;',
        '      border-radius: 6px;',
        '      margin-bottom: 24px;',
        '      border: 1px solid transparent;',
        '    }',
        '    .banner-warning { background: var(--color-warn-bg); border-color: var(--color-warn-border); }',
        '    .banner-warning h2 { color: var(--color-warn); font-size: 16px; margin-bottom: 4px; }',
        '    .banner-caution { background: var(--color-warn-bg); border-color: var(--color-warn-border); }',
        '    .banner-caution h2 { color: var(--color-warn); font-size: 16px; margin-bottom: 4px; }',
        '    .banner-notice { background: var(--color-primary-bg); border-color: #BFDBFE; }',
        '    .banner-notice h2 { color: var(--color-primary); font-size: 16px; margin-bottom: 4px; }',
        '    .banner-info { background: var(--color-primary-bg); border-color: #BFDBFE; }',
        '    .banner-info h2 { color: var(--color-primary); font-size: 16px; margin-bottom: 4px; }',
        '    .banner-success { background: var(--color-ok-bg); border-color: var(--color-ok-border); }',
        '    .banner-success h2 { color: var(--color-ok); font-size: 16px; margin-bottom: 4px; }',
        '    .banner-unable { background: var(--color-unable-bg); border-color: #E2E8F0; }',
        '    .banner-unable h2 { color: var(--color-unable); font-size: 16px; margin-bottom: 4px; }',
        '    .headline-banner p { font-size: 13px; color: var(--color-muted); }',
        '    .counts-bar {',
        '      display: flex;',
        '      gap: 12px;',
        '      margin-bottom: 24px;',
        '      flex-wrap: wrap;',
        '    }',
        '    .count-pill {',
        '      display: flex;',
        '      align-items: center;',
        '      gap: 8px;',
        '      padding: 8px 14px;',
        '      border-radius: 6px;',
        '      border: 1px solid var(--color-border);',
        '      background: #FFFFFF;',
        '      font-size: 13px;',
        '      font-weight: 500;',
        '    }',
        '    .count-pill strong { font-weight: 700; }',
        '    .pill-candidate { background: var(--color-warn-bg); border-color: var(--color-warn-border); color: var(--color-warn); }',
        '    .pill-not-observed { background: var(--color-ok-bg); border-color: var(--color-ok-border); color: var(--color-ok); }',
        '    .pill-unable { background: var(--color-unable-bg); border-color: #E2E8F0; color: var(--color-unable); }',
        '    .pill-pending { background: var(--color-primary-bg); border-color: #BFDBFE; color: var(--color-primary); }',
        '    .pill-revoked { background: #F9FAFB; border-color: #E5E7EB; color: #6B7280; }',
        '    .timeline-section { margin-bottom: 28px; }',
        '    .section-title {',
        '      font-size: 15px;',
        '      font-weight: 700;',
        '      color: var(--color-ink);',
        '      margin-bottom: 14px;',
        '      display: flex;',
        '      align-items: center;',
        '      gap: 8px;',
        '    }',
        '    .issue-card {',
        '      background: #FFFFFF;',
        '      border: 1px solid var(--color-border);',
        '      border-radius: 6px;',
        '      padding: 18px;',
        '      margin-bottom: 16px;',
        '    }',
        '    .issue-header {',
        '      display: flex;',
        '      justify-content: space-between;',
        '      align-items: center;',
        '      margin-bottom: 10px;',
        '    }',
        '    .issue-name {',
        '      font-size: 15px;',
        '      font-weight: 700;',
        '      color: var(--color-ink);',
        '      display: flex;',
        '      align-items: center;',
        '      gap: 8px;',
        '    }',
        '    .issue-tag {',
        '      font-size: 12px;',
        '      font-weight: 600;',
        '      padding: 2px 8px;',
        '      border-radius: 4px;',
        '    }',
        '    .tag-confirmed { background: #DCFCE7; color: #166534; border: 1px solid #BBF7D0; }',
        '    .tag-pending { background: #FEF3C7; color: #92400E; border: 1px solid #FED7AA; }',
        '    .issue-standard { font-size: 13px; color: var(--color-muted); margin-bottom: 6px; }',
        '    .issue-reason { font-size: 13px; color: var(--color-warn); font-weight: 500; margin-bottom: 12px; }',
        '    .evidence-grid {',
        '      display: grid;',
        '      grid-template-columns: 1fr 1fr;',
        '      gap: 16px;',
        '      margin-bottom: 12px;',
        '    }',
        '    @media (max-width: 680px) { .evidence-grid { grid-template-columns: 1fr; } }',
        '    .evidence-box {',
        '      background: #F8FAFC;',
        '      border: 1px solid var(--color-border);',
        '      border-radius: 6px;',
        '      overflow: hidden;',
        '      text-align: center;',
        '    }',
        '    .evidence-box img {',
        '      width: 100%;',
        '      height: auto;',
        '      display: block;',
        '      border-bottom: 1px solid var(--color-border);',
        '    }',
        '    .evidence-caption {',
        '      padding: 6px 10px;',
        '      font-size: 12px;',
        '      color: var(--color-muted);',
        '      background: #FFFFFF;',
        '    }',
        '    .evidence-placeholder {',
        '      height: 180px;',
        '      display: flex;',
        '      align-items: center;',
        '      justify-content: center;',
        '      color: var(--color-subtle);',
        '      font-size: 12px;',
        '      background: #F1F5F9;',
        '      padding: 16px;',
        '    }',
        '    details {',
        '      background: #F8FAFC;',
        '      border: 1px solid var(--color-border);',
        '      border-radius: 6px;',
        '      margin-bottom: 12px;',
        '      overflow: hidden;',
        '    }',
        '    summary {',
        '      padding: 10px 14px;',
        '      font-weight: 600;',
        '      font-size: 13px;',
        '      color: var(--color-ink);',
        '      cursor: pointer;',
        '      background: #FFFFFF;',
        '      border-bottom: 1px solid transparent;',
        '    }',
        '    details[open] summary { border-bottom-color: var(--color-border); }',
        '    .details-content { padding: 12px 14px; font-size: 12px; color: var(--color-muted); line-height: 1.7; }',
        '    .item-list { list-style: none; }',
        '    .item-row {',
        '      display: flex;',
        '      justify-content: space-between;',
        '      padding: 8px 0;',
        '      border-bottom: 1px solid #EDF2F7;',
        '      font-size: 13px;',
        '    }',
        '    .item-row:last-child { border-bottom: none; }',
        '    .footer-note {',
        '      margin-top: 32px;',
        '      padding-top: 16px;',
        '      border-top: 1px solid var(--color-border);',
        '      font-size: 11px;',
        '      color: var(--color-subtle);',
        '      text-align: center;',
        '      line-height: 1.6;',
        '    }',
        '    @media print {',
        '      @page { size: A4 portrait; margin: 15mm 12mm 15mm 12mm; }',
        '      body { background: #FFFFFF !important; padding: 0 !important; color: #000000 !important; font-size: 10pt; }',
        '      .report-container { max-width: 100% !important; border: none !important; padding: 0 !important; }',
        '      .no-print, button { display: none !important; }',
        '      .issue-card, .evidence-grid, .timeline-section { break-inside: avoid; page-break-inside: avoid; }',
        '      details { display: block !important; border: 1px solid #E2E8F0 !important; }',
        '      summary { display: none !important; }',
        '      .details-content { display: block !important; }',
        '    }',
        '  </style>',
        '</head>',
        '<body>',
        '  <div class="report-container">',
        '    <!-- Header -->',
        '    <header class="header-bar">',
        '      <div class="header-meta">',
        '        <h1>散打学生练习分析图文报告</h1>',
        f'        <div class="student-info">学员：{student_display}</div>',
        f'        <div class="action-info">动作：{action_display}</div>',
        '      </div>',
        '      <div class="header-date">',
        f'        <div>记录时间：{time_display}</div>',
        f'        <div>记录版本：Rev {header.revision}</div>',
        f'        <div>规则版本：{safe_clean(header.rule_version)}</div>',
        '      </div>',
        '    </header>',
        '    <!-- Headline Banner -->',
        f'    <div class="headline-banner {headline_class}">',
        f'      <h2>{headline_title_escaped}</h2>',
        f'      <p>{headline_subtitle_escaped}</p>',
        '    </div>',
        '    <!-- Conserved Counts Bar -->',
        '    <div class="counts-bar">',
        f'      <div class="count-pill pill-candidate">{_get_svg_status_icon("candidate", 16)} 发现疑似问题：<strong>{counts.candidate_count}</strong>项</div>',
        f'      <div class="count-pill pill-not-observed">{_get_svg_status_icon("not_observed", 16)} 已检查未发现：<strong>{counts.not_observed_count}</strong>项</div>',
        f'      <div class="count-pill pill-unable">{_get_svg_status_icon("unable", 16)} 暂无法判断：<strong>{counts.unable_count}</strong>项</div>',
        f'      <div class="count-pill pill-pending">{_get_svg_status_icon("pending_rule", 16)} 需教师判断：<strong>{counts.pending_rule_count}</strong>项</div>',
        (f'      <div class="count-pill pill-revoked">{_get_svg_status_icon("revoked", 16)} 教师已撤销：<strong>{counts.revoked_count}</strong>项</div>' if counts.revoked_count > 0 else ''),
        f'      <div class="count-pill">检查项总计：<strong>{counts.total_checks}</strong>项 (守恒)</div>',
        '    </div>',
        '    <!-- Diagram-Design Action Phase Timeline -->',
        '    <section class="timeline-section">',
        '      <div class="section-title">动作阶段时序流向图</div>',
        f'      {timeline_svg}',
        '    </section>'
    ]

    # Section 1: Active Candidate Issues with Evidence Frames
    doc.append('    <!-- Candidate Issues -->')
    doc.append('    <section class="issues-section">')
    doc.append(f'      <div class="section-title">需要关注的问题项 ({len(model.candidate_checks)}项)</div>')

    if not model.candidate_checks:
        doc.append('      <div class="issue-card" style="text-align: center; color: var(--color-muted); padding: 24px;">')
        doc.append('        当前没有未撤销的候选问题；不代表全部动作标准合格。')
        doc.append('      </div>')
    else:
        for idx, check in enumerate(model.candidate_checks, 1):
            rev_badge = ('<span class="issue-tag tag-confirmed">教师已确认</span>'
                         if check.review == "confirmed" else
                         '<span class="issue-tag tag-pending">待教师复核</span>')
            clean_reason = safe_clean(check.reason)
            clean_standard = safe_clean(check.standard or "无特定标准文本")
            clean_name = safe_clean(check.name)
            clean_body = safe_clean(check.body_part)
            clean_seg = safe_clean(check.segment_label)
            clean_phase = safe_clean(check.phase_label)
            doc.append('      <div class="issue-card">')
            doc.append('        <div class="issue-header">')
            doc.append(f'          <div class="issue-name">{_get_svg_status_icon("candidate", 18)} {idx}. {clean_seg} / {clean_phase}：{clean_name}</div>')
            doc.append(f'          <div>{rev_badge}</div>')
            doc.append('        </div>')
            doc.append(f'        <div class="issue-standard"><strong>标准：</strong>{clean_standard}（部位：{clean_body}）</div>')
            doc.append(f'        <div class="issue-reason"><strong>依据：</strong>{clean_reason}</div>')

            if check.evidence_refs:
                doc.append('        <div class="evidence-grid">')
                for ev_ref in check.evidence_refs[:2]:
                    data_uri, caption = get_evidence_data_uri(ev_ref, check.code)
                    doc.append('          <div class="evidence-box">')
                    if data_uri:
                        doc.append(f'            <img src="{data_uri}" alt="{safe_clean(caption)}">')
                    else:
                        doc.append(f'            <div class="evidence-placeholder">{safe_clean(caption)}</div>')
                    doc.append(f'            <div class="evidence-caption">{safe_clean(caption)}</div>')
                    doc.append('          </div>')
                doc.append('        </div>')

            if check.measurements_summary or check.audit_logs:
                doc.append('        <details>')
                doc.append('          <summary>查看测量数据与复核日志</summary>')
                doc.append('          <div class="details-content">')
                if check.measurements_summary:
                    for line in check.measurements_summary.split("\n"):
                        doc.append(f'            <div>{safe_clean(line)}</div>')
                if check.audit_logs:
                    doc.append('            <div style="margin-top: 8px; font-weight: 600;">复核历史记录：</div>')
                    for audit in check.audit_logs:
                        teacher_esc = safe_clean(audit.get("teacher"))
                        reason_esc = safe_clean(audit.get("reason"))
                        decision_esc = "已确认" if audit.get("after") == "confirmed" else "已撤销"
                        time_esc = safe_clean(format_local_datetime(audit.get("at", "")))
                        doc.append(f'            <div>• {time_esc} 由 {teacher_esc} 复核为「{decision_esc}」；原因：{reason_esc}</div>')
                doc.append('          </div>')
                doc.append('        </details>')

            doc.append('      </div>')
    doc.append('    </section>')

    # Section 2: Incomplete checks (Unable & Pending Rule)
    incomplete_checks = model.unable_checks + model.pending_rule_checks
    if incomplete_checks:
        doc.append('    <!-- Incomplete / Unable Checks -->')
        doc.append('    <section class="incomplete-section">')
        doc.append(f'      <details {"open" if not model.candidate_checks else ""}>')
        doc.append(f'        <summary>未完成判断的项目 ({len(incomplete_checks)}项：无法判断{counts.unable_count}项，需教师判断{counts.pending_rule_count}项)</summary>')
        doc.append('        <div class="details-content">')
        doc.append('          <ul class="item-list">')
        for c in incomplete_checks:
            icon = _get_svg_status_icon(c.status, 14)
            status_text = safe_clean(c.status_display)
            clean_reason = safe_clean(c.reason)
            c_seg = safe_clean(c.segment_label)
            c_phase = safe_clean(c.phase_label)
            c_name = safe_clean(c.name)
            c_body = safe_clean(c.body_part)
            doc.append('            <li class="item-row">')
            doc.append(f'              <div>{icon} <strong>{c_seg}/{c_phase}</strong> · {c_name} ({c_body})</div>')
            doc.append(f'              <div style="color: var(--color-subtle);">{status_text}：{clean_reason}</div>')
            doc.append('            </li>')
        doc.append('          </ul>')
        doc.append('        </div>')
        doc.append('      </details>')
        doc.append('    </section>')

    # Section 3: Not Observed Checks
    if model.not_observed_checks:
        doc.append('    <!-- Not Observed Checks -->')
        doc.append('    <section class="not-observed-section">')
        doc.append('      <details>')
        doc.append(f'        <summary>已检查未发现问题的项目 ({len(model.not_observed_checks)}项)</summary>')
        doc.append('        <div class="details-content">')
        doc.append('          <p style="margin-bottom: 8px; color: var(--color-subtle);">说明：以下项目按二维几何规则检查完毕，未发现明显动作问题（不代表全部实战标准合格）。</p>')
        doc.append('          <ul class="item-list">')
        for c in model.not_observed_checks:
            icon = _get_svg_status_icon("not_observed", 14)
            c_seg = safe_clean(c.segment_label)
            c_phase = safe_clean(c.phase_label)
            c_name = safe_clean(c.name)
            doc.append('            <li class="item-row">')
            doc.append(f'              <div>{icon} <strong>{c_seg}/{c_phase}</strong> · {c_name}</div>')
            doc.append('              <div style="color: var(--color-ok);">已检查未发现</div>')
            doc.append('            </li>')
        doc.append('          </ul>')
        doc.append('        </div>')
        doc.append('      </details>')
        doc.append('    </section>')

    # Section 4: Revoked Checks (Excluded from candidates)
    if model.revoked_checks:
        doc.append('    <!-- Revoked Checks -->')
        doc.append('    <section class="revoked-section">')
        doc.append('      <details>')
        doc.append(f'        <summary>教师已撤销的项目 ({len(model.revoked_checks)}项)</summary>')
        doc.append('        <div class="details-content">')
        doc.append('          <p style="margin-bottom: 8px; color: var(--color-subtle);">说明：以下项目原为算法候选问题，经教师复核确认为误报并撤销，已从候选列表中移出。</p>')
        doc.append('          <ul class="item-list">')
        for c in model.revoked_checks:
            icon = _get_svg_status_icon("revoked", 14)
            c_seg = safe_clean(c.segment_label)
            c_phase = safe_clean(c.phase_label)
            c_name = safe_clean(c.name)
            doc.append('            <li class="item-row">')
            doc.append(f'              <div>{icon} <del>{c_seg}/{c_phase} · {c_name}</del></div>')
            doc.append('              <div style="color: #6B7280;">已由教师撤销</div>')
            doc.append('            </li>')
        doc.append('          </ul>')
        doc.append('        </div>')
        doc.append('      </details>')
        doc.append('    </section>')

    # Section 5: Diagnostics & Calibration Metadata (Desensitized)
    doc.append('    <!-- Diagnostics & Alignment -->')
    doc.append('    <section class="diagnostics-section">')
    doc.append('      <details>')
    doc.append('        <summary>采集与分段技术诊断 (离线分析元数据)</summary>')
    doc.append('        <div class="details-content">')
    doc.append(f'          <div>分析后端说明：{safe_clean(header.backend_description)}</div>')
    for view, label in [("front", "正面录像"), ("side", "侧面录像")]:
        diag = model.diagnostics.get(view)
        cap = model.capture.get(view, {})
        if not diag and not cap:
            doc.append(f'          <div>{label}：无元数据</div>')
            continue
        diag = diag or {}
        fps_val = _safe_float(cap.get('sourceFps'), 0.0) if cap else 0.0
        cap_str = f"录像{cap.get('width', 0)}×{cap.get('height', 0)} / {fps_val:.1f}fps" if cap else "未知规格"
        body_frames = diag.get("bodyValidFrames", 0)
        tot_frames = diag.get("totalFrames", 0)
        clean_diag_reason = safe_clean(diag.get("reason", "正常"))
        doc.append(f'          <div>{label}：{safe_clean(cap_str)}；可见躯干链 {body_frames}/{tot_frames} 帧；分段原因：{clean_diag_reason}</div>')

    align = model.alignment
    if align:
        offset_val = _safe_float(align.get("sideOffsetSeconds"), 0.0)
        doc.append(f'          <div>时间对齐：{safe_clean(str(align.get("method", "未知")))}；侧面偏移 {offset_val:.3f}s；可靠性：{"可靠" if align.get("reliable") else "不可靠"}</div>')
    doc.append('        </div>')
    doc.append('      </details>')
    doc.append('    </section>')

    # Footer
    doc.append('    <footer class="footer-note">')
    doc.append(f'      <div>武术散打动作识别系统 · 报告生成时间：{time_display} · 记录ID：{safe_clean(header.record_id)}</div>')
    doc.append('      <div>声明：本报告由规则与几何算法辅助生成，不计算成绩分值，评定结论需由专业教师核实。</div>')
    doc.append('    </footer>')
    doc.append('  </div>')
    doc.append('</body>')
    doc.append('</html>')

    return '\n'.join(doc)


def export_html_report(record_id: str, destination: Path, *, history: Any) -> Path:
    """Atomic export of single-file offline HTML report."""
    return history.export(record_id, destination)
