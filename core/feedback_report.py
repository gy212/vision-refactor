"""练习报告门面：模型在 ``core.reporting.practice_model``，证据帧在 ``core.reporting.evidence``，
图文报告在 ``core.reporting.html``（P9 删除）。"""
from core.reporting.evidence import (  # noqa: F401
    RULE_JOINTS,
    SKELETON_BONES,
    extract_evidence_frame,
    frame_to_data_uri,
    resolve_record_video_path,
)
from core.reporting.html import (  # noqa: F401
    desensitize_paths,
    generate_phase_timeline_svg,
    render_html_report,
    safe_clean,
)
from core.reporting.practice_model import (  # noqa: F401
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
