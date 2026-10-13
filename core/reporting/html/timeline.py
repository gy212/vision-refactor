# -*- coding: utf-8 -*-
"""图文报告的相位时间线：把相位时间线数据画成自包含的内联 SVG（2px 圆角描边、4px 网格、无外部资源）。"""
from __future__ import annotations

import html
import math

from core.reporting.html.icons import (
    COLOR_CANDIDATE,
    COLOR_CANDIDATE_BG,
    COLOR_CANDIDATE_BORDER,
    COLOR_NOT_OBSERVED,
    COLOR_NOT_OBSERVED_BG,
    COLOR_PENDING,
    COLOR_PENDING_BORDER,
    COLOR_UNABLE,
    COLOR_UNABLE_BG,
    COLOR_UNABLE_BORDER,
)
from core.reporting.practice_model import PhaseTimelineItem, StatusCounts

_FONT = "font-family=\"-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei UI', sans-serif\""
_NODE_H = 68
_NODE_Y = 36


def _empty_svg(vb_w: int) -> str:
    return (
        f'<svg viewBox="0 0 {vb_w} 80" width="100%" height="auto" '
        'role="img" aria-labelledby="flow-title flow-desc">\n'
        '  <title id="flow-title">动作阶段时序流图</title>\n'
        '  <desc id="flow-desc">暂无记录的动作阶段时序信息。</desc>\n'
        '  <rect width="100%" height="100%" rx="6" fill="#F8FAFC" stroke="#D8E0EA" stroke-width="1"/>\n'
        f'  <text x="400" y="45" fill="#475569" font-size="12" text-anchor="middle" {_FONT}>未记录动作阶段时间</text>\n'
        '</svg>'
    )


def _svg_head(vb_w: int, vb_h: int) -> list[str]:
    return [
        f'<svg viewBox="0 0 {vb_w} {vb_h}" width="100%" height="auto" '
        'role="img" aria-labelledby="flow-title flow-desc">',
        '  <title id="flow-title">动作阶段时序流图</title>',
        '  <desc id="flow-desc">展示散打动作各阶段起止时间、检查项与疑似问题分布时序图。</desc>',
        '  <defs>',
        '    <marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">',
        '      <polygon points="0 0, 8 3, 0 6" fill="#475569"/>',
        '    </marker>',
        '    <marker id="arrow-warn" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">',
        '      <polygon points="0 0, 8 3, 0 6" fill="#92400E"/>',
        '    </marker>',
        '  </defs>',
        '  <rect width="100%" height="100%" rx="6" fill="#F8FAFC" stroke="#D8E0EA" stroke-width="1"/>',
    ]


def _node_style(i: int, phase: PhaseTimelineItem) -> tuple[str, str, str, str, str, str]:
    """节点的填充、描边、强调线、徽标文字、徽标底色与徽标描边。"""
    if phase.candidate_count > 0:
        return (COLOR_CANDIDATE_BG, COLOR_CANDIDATE_BORDER, COLOR_CANDIDATE,
                f"⚠ {phase.candidate_count}项", "#FEF3C7", COLOR_CANDIDATE)
    if phase.unable_count > 0:
        return (COLOR_UNABLE_BG, COLOR_UNABLE_BORDER, COLOR_UNABLE,
                f"? {phase.unable_count}项", "#E2E8F0", COLOR_UNABLE)
    return ("#FFFFFF", COLOR_PENDING_BORDER if i == 0 else "#D8E0EA", COLOR_PENDING if i == 0 else "#10B981",
            "✓", COLOR_NOT_OBSERVED_BG, COLOR_NOT_OBSERVED)


def _node(i: int, phase: PhaseTimelineItem, nx: int, node_w: int) -> list[str]:
    ny, node_h = _NODE_Y, _NODE_H
    step_num = f"{i + 1:02d}"
    fill_color, stroke_color, accent_line, badge_text, badge_fill, badge_stroke = _node_style(i, phase)
    badge_w = 40 if len(badge_text) > 2 else 22
    badge_x = nx + node_w - badge_w - 6
    return [
        f'  <!-- Stage {step_num}: {html.escape(phase.title)} -->',
        f'  <rect x="{nx}" y="{ny}" width="{node_w}" height="{node_h}" rx="6" fill="#FFFFFF"/>',
        f'  <rect x="{nx}" y="{ny}" width="{node_w}" height="{node_h}" rx="6" '
        f'fill="{fill_color}" stroke="{stroke_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
        f'  <line x1="{nx + 4}" y1="{ny + 8}" x2="{nx + 4}" y2="{ny + node_h - 8}" '
        f'stroke="{accent_line}" stroke-width="2" stroke-linecap="round"/>',
        f'  <text x="{nx + 12}" y="{ny + 20}" fill="#475569" font-size="10" font-weight="600" '
        f'{_FONT}>{step_num}</text>',
        f'  <rect x="{badge_x}" y="{ny + 8}" width="{badge_w}" height="16" rx="4" '
        f'fill="{badge_fill}" stroke="{badge_stroke}" stroke-width="1"/>',
        f'  <text x="{badge_x + badge_w // 2}" y="{ny + 20}" fill="{badge_stroke}" font-size="9" font-weight="600" '
        f'text-anchor="middle" {_FONT}>{html.escape(badge_text)}</text>',
        f'  <text x="{nx + 12}" y="{ny + 40}" fill="#1F2937" font-size="12" font-weight="600" '
        f'{_FONT}>{html.escape(phase.title)}</text>',
        f'  <text x="{nx + 12}" y="{ny + 58}" fill="#64748B" font-size="10" '
        f'{_FONT}>{html.escape(phase.time_range_text)}</text>',
    ]


def generate_phase_timeline_svg(phases: list[PhaseTimelineItem], counts: StatusCounts) -> str:
    """Generate self-contained inline SVG timeline compliant with diagram-design Process/Timeline standard.

    Adheres strictly to:
    - 2px stroke, round caps/joins (stroke-linecap="round" stroke-linejoin="round")
    - 4px grid alignment for structural bounds
    - Accessible contract (role="img", aria-labelledby, <title>, <desc>)
    - Zero external URLs or fonts (no http:// or https://)
    """
    n = len(phases)
    vb_w = 800
    if n == 0:
        return _empty_svg(vb_w)

    margin_x = 24
    gap = 20
    raw_node_w = (vb_w - 2 * margin_x - (n - 1) * gap) / n
    node_w = int(max(96, math.floor(raw_node_w / 4.0) * 4))
    center_y = _NODE_Y + _NODE_H // 2
    vb_w = max(vb_w, 2 * margin_x + n * node_w + (n - 1) * gap)

    parts = _svg_head(vb_w, 132)
    xs = [margin_x + i * (node_w + gap) for i in range(n)]
    for i in range(n - 1):  # 连线先画，节点压在上面
        warn = phases[i].candidate_count > 0 or phases[i + 1].candidate_count > 0
        marker_id = "arrow-warn" if warn else "arrow"
        stroke_color = "#92400E" if warn else "#475569"
        parts.append(
            f'  <line x1="{xs[i] + node_w}" y1="{center_y}" x2="{xs[i + 1]}" y2="{center_y}" '
            f'stroke="{stroke_color}" stroke-width="2" stroke-linecap="round" marker-end="url(#{marker_id})"/>'
        )
    for i, phase in enumerate(phases):
        parts.extend(_node(i, phase, xs[i], node_w))
    parts.append('</svg>')
    return '\n'.join(parts)
