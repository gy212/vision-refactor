"""离线组合流程（L4）的公共 API：比对、模板生成与自动裁剪复核、匹配预览导出。"""
from core.services import analyze
from core.services.analyze import compare_dual, compare_templates, compare_video, score_segment
from core.services.preview import export_match_preview, write_pose_preview
from core.services.templates import (
    TemplateAutoCropAnalysis,
    TemplateAutoCropReviewRequired,
    analyze_template_auto_crop,
    create_template_from_video,
)

__all__ = [
    "TemplateAutoCropAnalysis",
    "TemplateAutoCropReviewRequired",
    "analyze",
    "analyze_template_auto_crop",
    "compare_dual",
    "compare_templates",
    "compare_video",
    "create_template_from_video",
    "export_match_preview",
    "score_segment",
    "write_pose_preview",
]
