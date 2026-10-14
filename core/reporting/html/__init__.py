"""学生练习图文报告（HTML）的公共 API：渲染器、相位时间线 SVG 与路径脱敏。"""
from core.reporting.html.renderer import render_html_report
from core.reporting.html.sanitize import desensitize_paths, safe_clean
from core.reporting.html.timeline import generate_phase_timeline_svg

__all__ = ["desensitize_paths", "generate_phase_timeline_svg", "render_html_report", "safe_clean"]
