"""离线组合流程（L4）的公共 API：比对、模板生成（含 body_core_v1）与自动裁剪复核、匹配预览导出、学生练习历史、录制性能测试。"""

from core.services import analyze
from core.services.body_core_templates import create_body_core_template
from core.services.analyze import (
    compare_dual,
    compare_templates,
    compare_video,
    score_segment,
    tech_eval_video,
)
from core.services.practice_history import HistoryStore, validate_identity
from core.services.preview import export_match_preview, write_pose_preview
from core.services.benchmark_probes import SystemProbes as BenchmarkSystemProbes
from core.services.recording_benchmark import (
    PHASES as BENCHMARK_PHASES,
    BenchmarkSetup,
    Phase as BenchmarkPhase,
    SessionLoad as BenchmarkSessionLoad,
    run_recording_benchmark,
    save_report as save_benchmark_report,
)
from core.services.templates import (
    MIN_CORE_VALID_RATIO,
    MIN_DETECTED_RATIO,
    MIN_TEMPLATE_FRAMES,
    VIEW_SUFFIXES,
    RecordingPair,
    TemplateAutoCropAnalysis,
    TemplateAutoCropReviewRequired,
    TemplateCreateResult,
    TemplateQualityError,
    TemplateQualityReport,
    analyze_template_auto_crop,
    create_template_from_video,
    create_template_pair,
    discover_recording_videos,
    planned_template_paths,
    sanitize_action_name,
)

__all__ = [
    "BENCHMARK_PHASES",
    "BenchmarkPhase",
    "BenchmarkSystemProbes",
    "BenchmarkSessionLoad",
    "BenchmarkSetup",
    "HistoryStore",
    "MIN_CORE_VALID_RATIO",
    "MIN_DETECTED_RATIO",
    "MIN_TEMPLATE_FRAMES",
    "RecordingPair",
    "TemplateAutoCropAnalysis",
    "TemplateAutoCropReviewRequired",
    "TemplateCreateResult",
    "TemplateQualityError",
    "TemplateQualityReport",
    "VIEW_SUFFIXES",
    "analyze",
    "analyze_template_auto_crop",
    "compare_dual",
    "create_body_core_template",
    "compare_templates",
    "compare_video",
    "create_template_from_video",
    "create_template_pair",
    "discover_recording_videos",
    "export_match_preview",
    "planned_template_paths",
    "run_recording_benchmark",
    "sanitize_action_name",
    "save_benchmark_report",
    "score_segment",
    "tech_eval_video",
    "validate_identity",
    "write_pose_preview",
]
