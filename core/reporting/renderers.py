"""学生练习历史导出的 HTML 图文报告渲染器表，由调用方注入 ``HistoryStore``。"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.reporting.html import render_html_report

ReportRenderer = Callable[[dict, Path], str]  # (历史记录, 历史根目录) -> 报告全文


def _html(record: dict, history_root: Path) -> str:
    return render_html_report(record, history_root=history_root)


def practice_report_renderers() -> dict[str, ReportRenderer]:
    """``.html``/``.htm`` → 自包含的离线图文报告。"""
    return {".html": _html, ".htm": _html}
