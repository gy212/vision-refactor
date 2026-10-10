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
from core.evaluation.scoring.cost import POSE33_BASELINE, avg_cost_of, score_from_avg_cost
from core.evaluation.templates import COMPARE, Template, canonical_feature_layout, load_template, normalizer_for
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
    layout_matches = left_layout_c is None or right_layout_c is None or left_layout_c == right_layout_c
    if left_arr.ndim == 3 and right_arr.ndim == 3 and left_shape == right_shape and layout_matches:
        return

    def _desc(label: str, arr: np.ndarray, frame_shape: tuple[int, ...], layout: str | None) -> str:
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
        video_path, pose_variant=pv, workers=workers_eff,
        normalizer=normalizer_for(template.normalizer_version), compute_view=compute_view,
        progress_cb=progress_cb, stop_evt=stop_evt,
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

    ``stop_evt`` 为 None 时 DTW 不响应取消（单模板比对的现状）；布局按 ``layout`` 校验（见模块说明的 D-9）。
    """
    out: list[TemplateMatch] = []
    for template in templates:
        raise_if_cancelled(stop_evt)
        assert_feature_layout_match(
            template.features, seq, left_label=f"template {template.path}", right_label=video_label,
            left_layout=layout, right_layout=layout,
        )
        cost, start, end = subsequence_dtw(template.features, seq, stop_evt=stop_evt)
        avg_cost = avg_cost_of(cost, template.features.shape[0])
        out.append(TemplateMatch(template, float(cost), float(avg_cost),
                                 score_from_avg_cost(avg_cost, baseline), int(start), int(end)))
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
    first = templates[0]
    seq, fps, _views, pv, workers_eff = template_features(
        first, Path(video_path), pose_variant=pose_variant, workers=workers,
        progress_cb=progress_cb, stop_evt=stop_evt,
    )
    matches = match_templates(seq, templates, layout=first.layout, baseline=baseline, stop_evt=stop_evt,
                              video_label=f"video {video_path}")
    return [
        CompareResult(
            template_path=m.template.path, video_path=Path(video_path), pose_variant=pv, fps=float(fps),
            start_frame=m.start_frame, end_frame=m.end_frame, cost=m.cost, avg_cost=m.avg_cost, score=m.score,
            preview_path=None, workers_used=workers_eff,
        )
        for m in matches
    ]
