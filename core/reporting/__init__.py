"""报告层（L4）的公共 API：练习报告模型、证据帧、文本报告与历史导出渲染器、图文报告。"""
from core.reporting.evidence import RULE_JOINTS, extract_evidence_frame, frame_to_data_uri, resolve_record_video_path
from core.reporting.html import desensitize_paths, render_html_report
from core.reporting.practice_model import (
    GroupedReason,
    HeadlineSummary,
    NormalizedCheck,
    NormalizedReportModel,
    PhaseTimelineItem,
    ReportHeader,
    StatusCounts,
    build_report_model,
    format_local_datetime,
    get_headline_summary,
    group_checks_by_reason,
)
from core.reporting.renderers import ReportRenderer, practice_report_renderers
from core.reporting.text import format_fusion, format_report

__all__ = [
    "GroupedReason",
    "HeadlineSummary",
    "NormalizedCheck",
    "NormalizedReportModel",
    "PhaseTimelineItem",
    "RULE_JOINTS",
    "ReportHeader",
    "ReportRenderer",
    "StatusCounts",
    "build_report_model",
    "desensitize_paths",
    "extract_evidence_frame",
    "format_fusion",
    "format_local_datetime",
    "format_report",
    "frame_to_data_uri",
    "get_headline_summary",
    "group_checks_by_reason",
    "practice_report_renderers",
    "render_html_report",
    "resolve_record_video_path",
]
