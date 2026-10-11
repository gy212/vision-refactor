"""学生练习历史导出的渲染器表：按扩展名选文字报告或图文报告，由调用方注入 ``FeedbackHistory``。"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.reporting.text import format_report

ReportRenderer = Callable[[dict, Path], str]  # (历史记录, 历史根目录) -> 报告全文


def _text(record: dict, history_root: Path) -> str:
    return format_report(record)


def _html(record: dict, history_root: Path) -> str:
    from core.feedback_report import render_html_report  # 图文报告依赖 cv2/PIL，导出时才加载
    return render_html_report(record, history_root=history_root)


def practice_report_renderers() -> dict[str, ReportRenderer]:
    """``.txt`` → 文字报告，``.html``/``.htm`` → 图文报告。"""
    return {".txt": _text, ".html": _html, ".htm": _html}
