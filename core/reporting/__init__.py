"""报告层（L4）的公共 API：学生练习结果的文本报告与历史导出渲染器。"""
from core.reporting.renderers import ReportRenderer, practice_report_renderers
from core.reporting.text import format_fusion, format_report

__all__ = ["ReportRenderer", "format_fusion", "format_report", "practice_report_renderers"]
