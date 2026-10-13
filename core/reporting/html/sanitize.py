# -*- coding: utf-8 -*-
"""图文报告的文本清洗：脱敏本机绝对路径并做 HTML 转义，所有动态文字都经这里写入报告。"""
from __future__ import annotations

import html
import re
from typing import Any

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


def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely coerce any value to float, falling back to default on error."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default
