# -*- coding: utf-8 -*-
"""离线组合流程：视频对模板比对、正侧双模板比对、片段目录的考试池复算、技术指标（CompareWindow、CLI、batch 共用，设计文档 §2.2）。

只做"提取 → 比对 → 可选导出"的串联，算法都在 ``core.evaluation``；计分方案按名字选（``scoring.schemes``）。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Event

from core.evaluation.matching.dual import DualCompareResult, compare_dual_streams, compare_dual_templates
from core.evaluation.matching.engine import (
    CompareResult,
    assert_feature_layout_match,
    match_templates,
    match_video,
    template_features,
)
from core.evaluation.scoring.cost import POSE33_BASELINE
from core.evaluation.scoring.exam_pool import ExamPoolResult, score_exam_pools, usable_pools
from core.evaluation.scoring.schemes import DUAL_STREAM, DUAL_TEMPLATE, Scheme
from core.evaluation.templates import COMPARE, load_template
from core.pose.features import ProgressCb
from core.pose.raw_series import extract_pose_raw_series
from core.services.preview import export_match_preview

__all__ = ["compare_dual", "compare_templates", "compare_video", "export_match_preview", "score_segment"]


def compare_video(
    template_path: str | Path,
    video_path: str | Path,
    *,
    pose_variant: str | None = None,
    workers: int = 1,
    preview_out: str | Path | None = None,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> CompareResult:
    """单模板比对（CompareWindow、``match_template`` CLI）；给了 ``preview_out`` 时导出匹配区间的骨架预览。

    取消只作用于提取与预览导出，单次 DTW 不响应取消（现状）。
    """
    stop_evt = stop_evt or Event()
    template = load_template(Path(template_path), COMPARE)
    video_path = Path(video_path)
    seq, fps, _views, pv, workers_eff = template_features(
        template, video_path, pose_variant=pose_variant, workers=workers, progress_cb=progress_cb, stop_evt=stop_evt,
    )
    video_label = f"video {video_path}"
    assert_feature_layout_match(template.features, seq, left_label=f"template {template.path}",
                                right_label=video_label, left_layout=template.layout, right_layout=template.layout)
    if progress_cb is not None:
        progress_cb("计算相似度", 0, 1)
    (m,) = match_templates(seq, [template], layout=template.layout, baseline=POSE33_BASELINE, video_label=video_label)
    result = CompareResult(
        template_path=template.path, video_path=video_path, pose_variant=pv, fps=float(fps),
        start_frame=m.start_frame, end_frame=m.end_frame, cost=m.cost, avg_cost=m.avg_cost, score=m.score,
        preview_path=None, workers_used=workers_eff,
    )
    if preview_out:
        preview = export_match_preview(video_path, out_path=Path(preview_out), start_frame=m.start_frame,
                                       end_frame=m.end_frame, pose_variant=pv, progress_cb=progress_cb, stop_evt=stop_evt)
        result = replace(result, preview_path=preview)
    return result


compare_templates = match_video  # 一条视频对多个模板（池），视频姿态只提取一次


def compare_dual(scheme: Scheme, front_template: str | Path, side_template: str | Path,
                 *videos: str | Path, **options) -> DualCompareResult:
    """按计分方案做正侧双模板比对：``DUAL_TEMPLATE`` 传一段视频，``DUAL_STREAM`` 传正、侧两段视频。"""
    if scheme is DUAL_TEMPLATE:
        (video,) = videos
        return compare_dual_templates(front_template, side_template, video, **options)
    if scheme is DUAL_STREAM:
        front_video, side_video = videos
        return compare_dual_streams(front_template, side_template, front_video, side_video, **options)
    raise ValueError(f"compare_dual 不支持计分方案 {scheme.name}")


def score_segment(front_video: str | Path, side_video: str | Path, front_templates, side_templates, *,
                  stop_evt: Event | None = None) -> ExamPoolResult:
    """离线复算一个双摄片段的考试分（``EXAM_POOL``，与 Tk 考试同一算法，D-4）；模板池按考试契约过滤。"""
    front_ok, side_ok = usable_pools(front_templates, side_templates)
    return score_exam_pools(front_ok, side_ok, Path(front_video), Path(side_video), compare=match_video,
                            raw_series=extract_pose_raw_series, stop_evt=stop_evt or Event())
