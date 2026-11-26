# -*- coding: utf-8 -*-
"""模板比对的唯一入口：``match_templates``（单模板、多模板、模板池是同一调用）与按模板取视频特征（设计文档 §7.5）。

D-9 未决：模板池混用 v2/v3 归一化时维持现状——视频特征按池内第一个模板的归一化版本提取，布局校验也
按第一个模板（``match_video``）。逐模板归一化作为 P3 之后的独立 PR。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import numpy as np

from core.evaluation.matching.dtw import subsequence_dtw
from core.evaluation.scoring.cost import (
    POSE33_BASELINE,
    avg_cost_of,
    score_from_avg_cost,
)
from core.evaluation.templates import (
    COMPARE,
    Template,
    canonical_feature_layout,
    load_template,
    normalizer_for,
)
from core.pose.features import ProgressCb, extract_features


def raise_if_cancelled(stop_evt: Event | None) -> None:
    if stop_evt is not None and stop_evt.is_set():
        raise InterruptedError("动作比对已取消")


@dataclass(frozen=True)
class CompareResult:
    template_path: Path
    video_path: Path
    pose_variant: str
    fps: float
    start_frame: int
    end_frame: int
    cost: float
    avg_cost: float
    score: float
    preview_path: Path | None = None
    workers_used: int | None = None


def _frame_shape(features: np.ndarray) -> tuple[int, ...]:
    arr = np.asarray(features)
    if arr.ndim <= 2:
        return tuple(int(x) for x in arr.shape)
    return tuple(int(x) for x in arr.shape[1:])


def assert_feature_layout_match(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_label: str,
    right_label: str,
    left_layout: str | None = None,
    right_layout: str | None = None,
) -> None:
    """两段特征的单帧形状与布局名（含历史别名）必须一致，否则报清晰错误，不静默比对。"""
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    left_shape = _frame_shape(left_arr)
    right_shape = _frame_shape(right_arr)
    left_layout_c = canonical_feature_layout(left_layout)
    right_layout_c = canonical_feature_layout(right_layout)
    layout_matches = (
        left_layout_c is None
        or right_layout_c is None
        or left_layout_c == right_layout_c
    )
    if (
        left_arr.ndim == 3
        and right_arr.ndim == 3
        and left_shape == right_shape
        and layout_matches
    ):
        return

    def _desc(
        label: str, arr: np.ndarray, frame_shape: tuple[int, ...], layout: str | None
    ) -> str:
        layout_part = "" if layout is None else f", feature_layout={layout!r}"
        return f"{label}: sequence_shape={tuple(arr.shape)}, frame_shape={frame_shape}{layout_part}"

    raise ValueError(
        "Feature layout mismatch: 不允许静默比对不同 layout 的特征。"
        f" {_desc(left_label, left_arr, left_shape, left_layout)};"
        f" {_desc(right_label, right_arr, right_shape, right_layout)}。"
        " 请使用相同 feature_layout 重新生成模板或特征。"
    )


def effective_workers(workers: int, *modes: str) -> int:
    """任一模板是 VIDEO（跟踪）模式时只用单线程，保证与模板生成时的跟踪方式一致。"""
    video_mode = any(mode == "video" for mode in modes)
    return max(1, 1 if (video_mode and int(workers) > 1) else int(workers))


def template_features(
    template: Template,
    video_path: Path,
    *,
    pose_variant: str | None,
    workers: int,
    compute_view: bool = False,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> tuple[np.ndarray, float, np.ndarray | None, str, int]:
    """按模板的归一化版本与运行模式提取视频特征，返回 ``(特征, fps, 视角分数, 模型规格, 实际线程数)``。"""
    pv = pose_variant or template.meta.get("pose_variant", "full")
    workers_eff = effective_workers(workers, template.running_mode)
    seq, fps, views = extract_features(
        video_path,
        pose_variant=pv,
        workers=workers_eff,
        normalizer=normalizer_for(template.normalizer_version),
        compute_view=compute_view,
        progress_cb=progress_cb,
        stop_evt=stop_evt,
    )
    return seq, fps, views, pv, workers_eff


@dataclass(frozen=True)
class TemplateMatch:
    template: Template
    cost: float
    avg_cost: float
    score: float
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class QueryMatch:
    """一段查询特征的子序列 DTW 结果：总代价、按查询帧数平均的代价与匹配区间（含端点）。"""

    cost: float
    avg_cost: float
    start_frame: int
    end_frame: int


def match_query(
    query: np.ndarray, seq: np.ndarray, *, stop_evt: Event | None = None
) -> QueryMatch:
    """查询特征 ``query`` 对视频特征 ``seq`` 做一次子序列 DTW（不校验布局；模板比对用 ``match_templates``）。"""
    cost, start, end = subsequence_dtw(query, seq, stop_evt=stop_evt)
    return QueryMatch(
        float(cost), float(avg_cost_of(cost, query.shape[0])), int(start), int(end)
    )


def _assert_pool_template_compatible(
    first: Template,
    template: Template,
    *,
    video_layout: str,
) -> None:
    """校验池内每个模板自身的 layout、normalizer_version 与 pose_variant 是否与基准模板一致。"""
    tpl_layout = template.layout if hasattr(template, "layout") else video_layout
    if canonical_feature_layout(tpl_layout) != canonical_feature_layout(video_layout):
        raise ValueError(
            "Feature layout mismatch: 模板池不允许混用不同 layout 的模板。"
            f" expected={video_layout!r}, actual={tpl_layout!r} ({template.path})。"
        )
    if not isinstance(first, Template) or not isinstance(template, Template):
        return
    if template.normalizer_version != first.normalizer_version:
        raise ValueError(
            "Template normalizer mismatch: 模板池不允许混用不同归一化版本"
            f" ({first.path}: {first.normalizer_version!r} vs {template.path}: {template.normalizer_version!r})。"
        )
    first_meta_norm = str(first.meta.get("normalizer_version") or "").strip().lower()
    tpl_meta_norm = str(template.meta.get("normalizer_version") or "").strip().lower()
    if first_meta_norm and tpl_meta_norm and tpl_meta_norm != first_meta_norm:
        raise ValueError(
            "Template normalizer mismatch: 模板池不允许混用不同归一化版本"
            f" ({first.path}: {first_meta_norm!r} vs {template.path}: {tpl_meta_norm!r})。"
        )
    first_pv = str(first.meta.get("pose_variant") or "full").strip().lower()
    tpl_pv = str(template.meta.get("pose_variant") or "full").strip().lower()
    if tpl_pv != first_pv:
        raise ValueError(
            "Template pose_variant mismatch: 模板池不允许混用不同姿态模型档位"
            f" ({first.path}: {first_pv!r} vs {template.path}: {tpl_pv!r})。"
        )


def _assert_pool_compatible(templates: Sequence[Template]) -> None:
    """在耗时姿态提取前前置校验模板池内全部模板的布局、归一化版本与模型档位是否兼容。"""
    if len(templates) < 2:
        return
    first = templates[0]
    if not isinstance(first, Template):
        return
    for template in templates[1:]:
        if not isinstance(template, Template):
            continue
        assert_feature_layout_match(
            template.features,
            first.features,
            left_label=f"template {template.path}",
            right_label=f"template {first.path}",
            left_layout=template.layout,
            right_layout=first.layout,
        )
        _assert_pool_template_compatible(first, template, video_layout=first.layout)


def _pool_workers(workers: int, templates: Sequence[Template]) -> int:
    modes = [t.running_mode for t in templates if isinstance(t, Template)]
    return effective_workers(workers, *modes) if modes else int(workers)


def match_templates(
    seq: np.ndarray,
    templates: Sequence[Template],
    *,
    layout: str,
    baseline: float = POSE33_BASELINE,
    stop_evt: Event | None = None,
    video_label: str = "video",
) -> list[TemplateMatch]:
    """视频特征 ``seq`` 对每个模板做一次子序列 DTW，按 ``templates`` 顺序返回。

    ``stop_evt`` 为 None 时 DTW 不响应取消（单模板比对的现状）；逐个校验每个模板自身的布局、
    归一化版本与模型档位是否与基准一致。
    """
    out: list[TemplateMatch] = []
    if not templates:
        return out
    first = templates[0]
    for template in templates:
        raise_if_cancelled(stop_evt)
        tpl_layout = template.layout if hasattr(template, "layout") else layout
        assert_feature_layout_match(
            template.features,
            seq,
            left_label=f"template {template.path}",
            right_label=video_label,
            left_layout=tpl_layout,
            right_layout=layout,
        )
        _assert_pool_template_compatible(first, template, video_layout=layout)
        m = match_query(template.features, seq, stop_evt=stop_evt)
        out.append(
            TemplateMatch(
                template,
                m.cost,
                m.avg_cost,
                score_from_avg_cost(m.avg_cost, baseline),
                m.start_frame,
                m.end_frame,
            )
        )
    return out


def match_video(
    template_paths: Sequence[str | Path],
    video_path: str | Path,
    *,
    pose_variant: str | None = None,
    workers: int = 1,
    baseline: float = POSE33_BASELINE,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
    load: Callable[[Path], Template] = lambda p: load_template(p, COMPARE),
) -> list[CompareResult]:
    """一条视频对多个模板：视频姿态只提取一次，复用给每个模板（原 ``compare_video_to_templates``）。

    以第一个模板决定归一化、模型规格与线程数（D-9 维持现状）；空列表返回空列表。
    """
    stop_evt = stop_evt or Event()
    templates = [load(Path(p)) for p in template_paths]
    if not templates:
        return []
    _assert_pool_compatible(templates)
    first = templates[0]
    pool_workers = _pool_workers(workers, templates)
    seq, fps, _views, pv, workers_eff = template_features(
        first,
        Path(video_path),
        pose_variant=pose_variant,
        workers=pool_workers,
        progress_cb=progress_cb,
        stop_evt=stop_evt,
    )
    matches = match_templates(
        seq,
        templates,
        layout=first.layout,
        baseline=baseline,
        stop_evt=stop_evt,
        video_label=f"video {video_path}",
    )
    return [_compare_result(m, Path(video_path), pv, fps, workers_eff) for m in matches]


def match_video_windows(
    template_paths: Sequence[str | Path],
    video_path: str | Path,
    windows: Sequence[tuple[int, int]],
    *,
    pose_variant: str | None = None,
    workers: int = 1,
    baseline: float = POSE33_BASELINE,
    stop_evt: Event | None = None,
    load: Callable[[Path], Template] = lambda p: load_template(p, COMPARE),
) -> list[list[CompareResult]]:
    """一条视频对多个模板，在每个帧范围 ``[起, 止]`` 内各做一次子序列 DTW（上步组合逐次评分，D-25）。

    视频特征与 ``match_video`` 一样只提取一次；返回外层按模板、内层按范围，帧号是整段视频的帧号。
    空范围（止 < 起或超出视频）跳过；模板列表为空时返回空列表。
    """
    stop_evt = stop_evt or Event()
    templates = [load(Path(p)) for p in template_paths]
    if not templates:
        return []
    _assert_pool_compatible(templates)
    first = templates[0]
    pool_workers = _pool_workers(workers, templates)
    seq, fps, _views, pv, workers_eff = template_features(
        first,
        Path(video_path),
        pose_variant=pose_variant,
        workers=pool_workers,
        stop_evt=stop_evt,
    )
    per_window = []
    for start, end in windows:
        start, end = max(0, int(start)), min(len(seq) - 1, int(end))
        if end <= start:
            continue
        matches = match_templates(
            seq[start : end + 1],
            templates,
            layout=first.layout,
            baseline=baseline,
            stop_evt=stop_evt,
            video_label=f"video {video_path}[{start}:{end}]",
        )
        per_window.append(
            [
                _compare_result(m, Path(video_path), pv, fps, workers_eff, offset=start)
                for m in matches
            ]
        )
    return [[window[i] for window in per_window] for i in range(len(templates))]


def _compare_result(
    m: TemplateMatch,
    video_path: Path,
    pv: str,
    fps: float,
    workers: int,
    offset: int = 0,
) -> CompareResult:
    return CompareResult(
        template_path=m.template.path,
        video_path=video_path,
        pose_variant=pv,
        fps=float(fps),
        start_frame=m.start_frame + offset,
        end_frame=m.end_frame + offset,
        cost=m.cost,
        avg_cost=m.avg_cost,
        score=m.score,
        preview_path=None,
        workers_used=workers,
    )
