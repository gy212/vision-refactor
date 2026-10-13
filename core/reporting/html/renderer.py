# -*- coding: utf-8 -*-
"""学生练习图文报告的渲染器：报告模型 + ``templates/`` 下的 HTML 模板 → 自包含的单文件离线报告。

不发起任何外部请求；所有动态文字经 ``safe_clean`` 脱敏与转义；证据帧以有界数量、有界尺寸的
Base64 JPEG 内嵌；样式表含 A4 打印规则。模板用 ``string.Template`` 的 ``${名字}`` 占位。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template

from core.reporting.evidence import RULE_JOINTS, extract_evidence_frame, frame_to_data_uri, resolve_record_video_path
from core.reporting.html.icons import status_icon
from core.reporting.html.sanitize import safe_clean, safe_float
from core.reporting.html.timeline import generate_phase_timeline_svg
from core.reporting.practice_model import NormalizedCheck, NormalizedReportModel, build_report_model, format_local_datetime

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


@lru_cache(maxsize=None)
def _template(name: str) -> Template:
    """读取模板；去掉结尾换行，并把 CRLF 统一成 LF（Windows 检出可能转换行尾）。"""
    text = (_TEMPLATE_DIR / f"{name}.html").read_text(encoding="utf-8").replace("\r\n", "\n")
    return Template(text[:-1] if text.endswith("\n") else text)


def _fill(template: str, /, **values: object) -> str:
    return _template(template).substitute(**values)


class _EvidenceEmbedder:
    """按报告的数量与尺寸上限截取证据帧；同一视角、帧号、高亮关节只截一次。"""

    def __init__(self, record: dict, model: NormalizedReportModel, history_root: Path | None,
                 max_image_edge: int, max_images: int) -> None:
        self._videos = record.get("videos", {})
        self._model = model
        self._root = history_root
        self._max_edge = max_image_edge
        self._max_images = max_images
        self._embedded = 0
        self._cache: dict[tuple[str, int, tuple[int, ...]], str] = {}

    def data_uri(self, ev_id: str, rule_code: str) -> tuple[str | None, str]:
        """证据帧的 data URI 与图注；截不到时 URI 为 None，图注说明原因。"""
        ev_data = self._model.evidence_catalog.get(ev_id)
        if not ev_data:
            return None, "未找到关联证据数据"
        if self._root is None:
            return None, "未指定历史目录，跳过截帧"
        view = ev_data.get("view", "front")
        frame_idx = ev_data.get("frame", 0)
        view_label = "正面" if view == "front" else "侧面"
        caption = f"{view_label}证据 · {ev_data.get('timeSeconds', 0.0):.3f}s · 第{frame_idx}帧"
        if self._embedded >= self._max_images:
            return None, f"{caption}（已省略次要截图以精简报告体积）"
        highlight_joints = tuple(sorted(RULE_JOINTS.get(rule_code, set())))
        cache_key = (view, frame_idx, highlight_joints)
        if cache_key in self._cache:
            return self._cache[cache_key], caption
        video_path = resolve_record_video_path(self._root, self._model.header.record_id, view, self._videos)
        if not video_path:
            return None, f"{caption}（原录像文件不可用）"
        try:
            frame = extract_evidence_frame(
                video_path, frame_idx,
                landmarks=ev_data.get("landmarks"), valid_mask=ev_data.get("validMask"),
                highlight_joints=set(highlight_joints) if highlight_joints else None,
            )
            if frame is None:
                return None, f"{caption}（截帧解码失败）"
            data_uri = frame_to_data_uri(frame, max_edge=self._max_edge)
        except Exception:
            return None, f"{caption}（截帧解码失败）"
        self._cache[cache_key] = data_uri
        self._embedded += 1
        return data_uri, caption


def _count_pills(model: NormalizedReportModel) -> str:
    counts = model.counts
    pills = [
        ("count-pill pill-candidate", f'{status_icon("candidate", 16)} 发现疑似问题：<strong>{counts.candidate_count}</strong>项'),
        ("count-pill pill-not-observed", f'{status_icon("not_observed", 16)} 已检查未发现：<strong>{counts.not_observed_count}</strong>项'),
        ("count-pill pill-unable", f'{status_icon("unable", 16)} 暂无法判断：<strong>{counts.unable_count}</strong>项'),
        ("count-pill pill-pending", f'{status_icon("pending_rule", 16)} 需教师判断：<strong>{counts.pending_rule_count}</strong>项'),
    ]
    lines = [_fill("count_pill", css_class=cls, content=content) for cls, content in pills]
    # 没有撤销项时保留一个空行，与旧版逐行拼接的输出逐字节一致
    lines.append(_fill("count_pill", css_class="count-pill pill-revoked",
                       content=f'{status_icon("revoked", 16)} 教师已撤销：<strong>{counts.revoked_count}</strong>项')
                 if counts.revoked_count > 0 else "")
    lines.append(_fill("count_pill", css_class="count-pill",
                       content=f"检查项总计：<strong>{counts.total_checks}</strong>项 (守恒)"))
    return "\n".join(lines)


def _evidence_grid(check: NormalizedCheck, embedder: _EvidenceEmbedder) -> str:
    boxes = []
    for ev_ref in check.evidence_refs[:2]:
        data_uri, caption = embedder.data_uri(ev_ref, check.code)
        if data_uri:
            boxes.append(_fill("evidence_image", src=data_uri, caption=safe_clean(caption)))
        else:
            boxes.append(_fill("evidence_placeholder", caption=safe_clean(caption)))
    return _fill("evidence_grid", boxes="\n".join(boxes))


def _issue_details(check: NormalizedCheck) -> str:
    lines = []
    if check.measurements_summary:
        lines.extend(f"<div>{safe_clean(line)}</div>" for line in check.measurements_summary.split("\n"))
    if check.audit_logs:
        lines.append('<div style="margin-top: 8px; font-weight: 600;">复核历史记录：</div>')
        for audit in check.audit_logs:
            decision = "已确认" if audit.get("after") == "confirmed" else "已撤销"
            when = safe_clean(format_local_datetime(audit.get("at", "")))
            lines.append(f'<div>• {when} 由 {safe_clean(audit.get("teacher"))} 复核为「{decision}」；'
                         f'原因：{safe_clean(audit.get("reason"))}</div>')
    return _fill("issue_details", lines="".join(f"\n            {line}" for line in lines))


def _issue_card(idx: int, check: NormalizedCheck, embedder: _EvidenceEmbedder) -> str:
    extras = []
    if check.evidence_refs:
        extras.append(_evidence_grid(check, embedder))
    if check.measurements_summary or check.audit_logs:
        extras.append(_issue_details(check))
    review_badge = ('<span class="issue-tag tag-confirmed">教师已确认</span>' if check.review == "confirmed"
                    else '<span class="issue-tag tag-pending">待教师复核</span>')
    return _fill(
        "issue_card",
        icon=status_icon("candidate", 18), index=idx,
        segment=safe_clean(check.segment_label), phase=safe_clean(check.phase_label), name=safe_clean(check.name),
        review_badge=review_badge,
        standard=safe_clean(check.standard or "无特定标准文本"), body_part=safe_clean(check.body_part),
        reason=safe_clean(check.reason),
        extras="".join(f"\n{block}" for block in extras),
    )


def _candidate_section(model: NormalizedReportModel, embedder: _EvidenceEmbedder) -> str:
    if model.candidate_checks:
        cards = "\n".join(_issue_card(i, c, embedder) for i, c in enumerate(model.candidate_checks, 1))
    else:
        cards = _fill("no_candidates")
    return _fill("candidate_section", count=len(model.candidate_checks), cards=cards)


def _row_names(c: NormalizedCheck) -> dict[str, str]:
    return {"segment": safe_clean(c.segment_label), "phase": safe_clean(c.phase_label), "name": safe_clean(c.name)}


def _status_sections(model: NormalizedReportModel) -> list[str]:
    """未完成判断、已检查未发现、教师已撤销三个折叠区，各自只在有项目时输出。"""
    sections = []
    incomplete = model.unable_checks + model.pending_rule_checks
    if incomplete:
        rows = [_fill("incomplete_row", icon=status_icon(c.status, 14), body_part=safe_clean(c.body_part),
                      status=safe_clean(c.status_display), reason=safe_clean(c.reason), **_row_names(c))
                for c in incomplete]
        sections.append(_fill("incomplete_section", open="open" if not model.candidate_checks else "",
                              total=len(incomplete), unable=model.counts.unable_count,
                              pending=model.counts.pending_rule_count, rows="\n".join(rows)))
    if model.not_observed_checks:
        rows = [_fill("not_observed_row", icon=status_icon("not_observed", 14), **_row_names(c))
                for c in model.not_observed_checks]
        sections.append(_fill("not_observed_section", count=len(model.not_observed_checks), rows="\n".join(rows)))
    if model.revoked_checks:
        rows = [_fill("revoked_row", icon=status_icon("revoked", 14), **_row_names(c)) for c in model.revoked_checks]
        sections.append(_fill("revoked_section", count=len(model.revoked_checks), rows="\n".join(rows)))
    return sections


def _diagnostics_section(model: NormalizedReportModel) -> str:
    lines = []
    for view, label in [("front", "正面录像"), ("side", "侧面录像")]:
        diag = model.diagnostics.get(view)
        cap = model.capture.get(view, {})
        if not diag and not cap:
            lines.append(f"<div>{label}：无元数据</div>")
            continue
        diag = diag or {}
        fps_val = safe_float(cap.get('sourceFps'), 0.0) if cap else 0.0
        cap_str = f"录像{cap.get('width', 0)}×{cap.get('height', 0)} / {fps_val:.1f}fps" if cap else "未知规格"
        lines.append(f"<div>{label}：{safe_clean(cap_str)}；可见躯干链 {diag.get('bodyValidFrames', 0)}/"
                     f"{diag.get('totalFrames', 0)} 帧；分段原因：{safe_clean(diag.get('reason', '正常'))}</div>")
    align = model.alignment
    if align:
        offset_val = safe_float(align.get("sideOffsetSeconds"), 0.0)
        lines.append(f'<div>时间对齐：{safe_clean(str(align.get("method", "未知")))}；侧面偏移 {offset_val:.3f}s；'
                     f'可靠性：{"可靠" if align.get("reliable") else "不可靠"}</div>')
    return _fill("diagnostics_section", backend=safe_clean(model.header.backend_description),
                 lines="".join(f"\n          {line}" for line in lines))


def render_html_report(
    record: dict,
    *,
    history_root: Path | None = None,
    max_image_edge: int = 480,
    max_images: int = 12
) -> str:
    """Render 100% offline, self-contained single-file HTML graphic report.

    Security & Offline Guarantees:
    - Zero external HTTP/HTTPS requests (no CDN, no web fonts, no external scripts).
    - Strict html.escape(quote=True) on all user and dynamic text.
    - Strips local absolute machine paths.
    - Embeds Base64 JPEG evidence thumbnails with memory and count limits.
    - Built-in @media print A4 printable stylesheet.
    """
    model = build_report_model(record)
    header = model.header
    embedder = _EvidenceEmbedder(record, model, history_root, max_image_edge, max_images)
    student_id = safe_clean(header.student_id, "未登记学号")
    time_display = safe_clean(header.created_at_local)
    sections = [_candidate_section(model, embedder), *_status_sections(model), _diagnostics_section(model)]
    return _fill(
        "report",
        student_id=student_id,
        student_display=f"{student_id} · {safe_clean(header.student_name)}" if header.student_name else student_id,
        action_display=f"{safe_clean(header.action_label)} · {safe_clean(header.stance_label)}",
        time_display=time_display,
        revision=header.revision,
        rule_version=safe_clean(header.rule_version),
        headline_class=f"banner-{model.headline.status_tone}",
        headline_title=safe_clean(model.headline.title),
        headline_subtitle=safe_clean(model.headline.subtitle),
        count_pills=_count_pills(model),
        timeline_svg=generate_phase_timeline_svg(model.phases, model.counts),
        sections="\n".join(sections),
        record_id=safe_clean(header.record_id),
    )
