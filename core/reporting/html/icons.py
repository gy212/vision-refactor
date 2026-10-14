# -*- coding: utf-8 -*-
"""图文报告的状态配色与内联 SVG 状态图标（不引用任何外部资源）。"""
from __future__ import annotations

COLOR_CANDIDATE = "#92400E"       # Foreground alert
COLOR_CANDIDATE_BG = "#FFF7ED"    # Light bg
COLOR_CANDIDATE_BORDER = "#FED7AA"  # Border
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

_ICON_BODIES = {
    "candidate": (
        "疑似问题",
        f'<path d="M12 4L4 19.5H20L12 4Z" stroke="{COLOR_CANDIDATE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="{COLOR_CANDIDATE_BG}"/>'
        f'<line x1="12" y1="9" x2="12" y2="13.5" stroke="{COLOR_CANDIDATE}" stroke-width="2" stroke-linecap="round"/>'
        f'<circle cx="12" cy="16.5" r="1" fill="{COLOR_CANDIDATE}"/>',
    ),
    "not_observed": (
        "未发现问题",
        f'<circle cx="12" cy="12" r="9.5" stroke="{COLOR_NOT_OBSERVED}" stroke-width="2" fill="{COLOR_NOT_OBSERVED_BG}"/>'
        f'<path d="M7 12L10.5 15.5L17 9" stroke="{COLOR_NOT_OBSERVED}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    ),
    "unable": (
        "暂无法判断",
        f'<circle cx="12" cy="12" r="9.5" stroke="{COLOR_UNABLE}" stroke-width="2" fill="{COLOR_UNABLE_BG}"/>'
        f'<path d="M9.5 9.5C9.5 7.8 10.6 6.5 12 6.5C13.5 6.5 14.5 7.6 14.5 9C14.5 10.5 12 11.5 12 13" stroke="{COLOR_UNABLE}" stroke-width="2" stroke-linecap="round"/>'
        f'<circle cx="12" cy="16.5" r="1" fill="{COLOR_UNABLE}"/>',
    ),
    "pending_rule": (
        "需教师判断",
        f'<rect x="4.5" y="5.5" width="15" height="15" rx="2" stroke="{COLOR_PENDING}" stroke-width="2" fill="{COLOR_PENDING_BG}"/>'
        f'<path d="M8.5 3.5H15.5V6.5H8.5V3.5Z" stroke="{COLOR_PENDING}" stroke-width="1.8" fill="#FFFFFF"/>'
        f'<path d="M7.5 12.5L10.5 15.5L16.5 9.5" stroke="{COLOR_PENDING}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    ),
    "revoked": (
        "已撤销",
        f'<circle cx="12" cy="12" r="9.5" stroke="{COLOR_REVOKED}" stroke-width="2" fill="{COLOR_REVOKED_BG}"/>'
        f'<line x1="7" y1="12" x2="17" y2="12" stroke="{COLOR_REVOKED}" stroke-width="2" stroke-linecap="round"/>',
    ),
}


def status_icon(status: str, size: int = 20) -> str:
    """状态对应的内联 SVG 图标；未知状态返回空串。"""
    entry = _ICON_BODIES.get(status)
    if entry is None:
        return ""
    label, body = entry
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" role="img" aria-label="{label}">'
            f'{body}</svg>')
