# -*- coding: utf-8 -*-
"""正侧双模板比对：单视频按视角自动切分（``compare_dual_templates``）与正侧两路视频（``compare_dual_streams``）（设计文档 §7.5）。

两者都由模板读取、视角切分、``repetition.score_view`` 与正式规则组合而成；计分方案分别是 ``DUAL_TEMPLATE``
与 ``DUAL_STREAM``（``scoring.schemes``）。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import numpy as np

from core.evaluation.matching.engine import assert_feature_layout_match, effective_workers, raise_if_cancelled
from core.evaluation.matching.repetition import JointErrorStat, RawSlice, RepetitionMatch, score_view, valid_mask_from_raw
from core.evaluation.rules.formal_v1 import RuleScore, RuleViolation, score_rules
from core.evaluation.scoring.cost import POSE33_BASELINE
from core.evaluation.segmentation import split_front_side
from core.evaluation.templates import COMPARE, Template, load_template, normalizer_for
from core.pose.features import ProgressCb, extract_features
from core.pose.raw_series import extract_pose_raw_series, slice_pose_raw_series


@dataclass(frozen=True)
class DualCompareResult:
    front_template_path: Path
    side_template_path: Path
    video_path: Path
    pose_variant: str
    fps: float
    front_score: float
    side_score: float
    combined_score: float
    combined_percent: int
    front_matches: tuple[RepetitionMatch, ...]
    side_matches: tuple[RepetitionMatch, ...]
    front_segment: tuple[int, int] | None = None  # 原视频中的 (start, end)
    side_segment: tuple[int, int] | None = None
    front_rule_score: int | None = None
    side_rule_score: int | None = None
    front_rule_deduction: int | None = None
    side_rule_deduction: int | None = None
    front_rule_violations: tuple[RuleViolation, ...] | None = None
    side_rule_violations: tuple[RuleViolation, ...] | None = None
    front_joint_errors: tuple[JointErrorStat, ...] | None = None
    side_joint_errors: tuple[JointErrorStat, ...] | None = None


@dataclass(frozen=True)
class _Pair:
    front: Template
    side: Template
    normalizer: Callable
    pose_variant: str
    workers: int


def _load_pair(front_path: Path, side_path: Path, pose_variant: str | None, workers: int) -> _Pair:
    """正侧模板布局必须一致、归一化版本必须相同；模型规格取正面、再取侧面、默认 full。"""
    front = load_template(front_path, COMPARE)
    side = load_template(side_path, COMPARE)
    assert_feature_layout_match(
        front.features, side.features, left_label=f"front template {front_path}",
        right_label=f"side template {side_path}", left_layout=front.layout, right_layout=side.layout,
    )
    if front.normalizer_version != side.normalizer_version:
        raise ValueError("Front/side templates use different feature layouts; regenerate templates with the same version.")
    pv = pose_variant or front.meta.get("pose_variant") or side.meta.get("pose_variant") or "full"
    return _Pair(front, side, normalizer_for(front.normalizer_version), pv,
                 effective_workers(workers, front.running_mode, side.running_mode))


def _lazy_raw(video: Path, pose_variant: str) -> RawSlice:
    """同一视频的原始序列只提取一次，规则与关节误差共用（``test_t008_single_pass_reuse``）。"""
    cache: list = []

    def raw_slice(start_frame: int | None, end_frame: int | None) -> tuple[np.ndarray, dict]:
        if not cache:
            cache.append(extract_pose_raw_series(video, pose_variant=pose_variant))
        return slice_pose_raw_series(cache[0], start_frame=start_frame, end_frame=end_frame)

    return raw_slice


def _rules(raw_slice: RawSlice, seg: tuple[int, int] | None, *, view: str, action_scope: str) -> RuleScore:
    s, e = (None, None) if seg is None else (int(seg[0]), int(seg[1]))
    raw, raw_meta = raw_slice(s, e)
    return score_rules(raw, view=view, action_scope=action_scope, valid_mask=valid_mask_from_raw(raw, raw_meta))


def _combine(w_front: float, w_side: float, front_score: float, side_score: float) -> tuple[float, int]:
    combined = float((w_front * float(front_score)) + (w_side * float(side_score)))
    combined = float(np.clip(combined, 0.0, 1.0))
    return combined, int(np.clip(int(round(combined * 100.0)), 0, 100))


def _result(pair: _Pair, video: Path, fps: float, w: tuple[float, float], front: tuple, side: tuple,
            segments: tuple, rules: tuple[RuleScore | None, RuleScore | None]) -> DualCompareResult:
    (f_score, f_matches, f_joints), (s_score, s_matches, s_joints) = front, side
    combined, pct = _combine(w[0], w[1], f_score, s_score)
    rf, rs = rules
    return DualCompareResult(
        front_template_path=pair.front.path, side_template_path=pair.side.path, video_path=video,
        pose_variant=pair.pose_variant, fps=float(fps), front_score=float(f_score), side_score=float(s_score),
        combined_score=float(combined), combined_percent=int(pct),
        front_matches=tuple(f_matches), side_matches=tuple(s_matches),
        front_segment=segments[0], side_segment=segments[1],
        front_rule_score=None if rf is None else int(rf.score),
        side_rule_score=None if rs is None else int(rs.score),
        front_rule_deduction=None if rf is None else int(rf.total_deduction),
        side_rule_deduction=None if rs is None else int(rs.total_deduction),
        front_rule_violations=None if rf is None else tuple(rf.violations),
        side_rule_violations=None if rs is None else tuple(rs.violations),
        front_joint_errors=None if f_joints is None else tuple(f_joints),
        side_joint_errors=None if s_joints is None else tuple(s_joints),
    )


def _check_video(template: Template, seq: np.ndarray, *, tpl_label: str, video_label: str) -> None:
    assert_feature_layout_match(template.features, seq, left_label=f"{tpl_label} {template.path}",
                                right_label=video_label, left_layout=template.layout, right_layout=template.layout)


def compare_dual_templates(
    front_template_path: str | Path,
    side_template_path: str | Path,
    video_path: str | Path,
    *,
    pose_variant: str | None = None,
    workers: int = 1,
    w_front: float = 0.4,
    w_side: float = 0.6,
    baseline: float = POSE33_BASELINE,
    enable_rules: bool = False,
    action_scope: str = "both",
    enable_error_analysis: bool = False,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> DualCompareResult:
    """一段同时含正面与侧面的学员视频对正/侧两个模板（``DUAL_TEMPLATE``）。

    按正面程度把视频切成正/侧两段（``median_always``），每段对各自模板取代表周期多次匹配、截尾均值，
    按 ``w_front``/``w_side`` 合成 0..1 与 0..100；可选在原始 Pose33 上做正式规则扣分与关节误差。
    """
    stop_evt = stop_evt or Event()
    video_path = Path(video_path)
    action_scope = str(action_scope or "both").lower()
    pair = _load_pair(Path(front_template_path), Path(side_template_path), pose_variant, workers)
    seq, fps, view_scores = extract_features(
        video_path, pose_variant=pair.pose_variant, workers=pair.workers, normalizer=pair.normalizer,
        compute_view=True, progress_cb=progress_cb, stop_evt=stop_evt,
    )
    _check_video(pair.front, seq, tpl_label="front template", video_label=f"video {video_path}")
    _check_video(pair.side, seq, tpl_label="side template", video_label=f"video {video_path}")
    raw_slice = _lazy_raw(video_path, pair.pose_variant)

    front_seg = side_seg = None
    if view_scores is not None and view_scores.size == seq.shape[0] and seq.shape[0] >= 10:
        front_seg, side_seg, _mode = split_front_side(view_scores, fps, mode="median_always")
    rules = (None, None)
    if enable_rules:  # 规则按"违规帧占比 ≥ 阈值则扣分"，降低单帧抖动影响
        rules = (_rules(raw_slice, front_seg, view="front", action_scope=action_scope),
                 _rules(raw_slice, side_seg, view="side", action_scope=action_scope))

    def view(template: Template, seg):  # 现状：单视频双模板的打分不响应取消
        return score_view(seq, raw_slice, template.features, template.meta, seg, fps=fps, baseline=baseline,
                          enable_error_analysis=enable_error_analysis)

    front = view(pair.front, front_seg)
    side = view(pair.side, side_seg)
    return _result(pair, video_path, fps, (w_front, w_side), front, side, (front_seg, side_seg), rules)


def compare_dual_streams(
    front_template_path: str | Path,
    side_template_path: str | Path,
    front_video_path: str | Path,
    side_video_path: str | Path,
    *,
    pose_variant: str | None = None,
    workers: int = 1,
    w_front: float = 0.4,
    w_side: float = 0.6,
    baseline: float = POSE33_BASELINE,
    enable_rules: bool = False,
    action_scope: str = "both",
    enable_error_analysis: bool = False,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> DualCompareResult:
    """正、侧两路已知视角的同步视频分别对正/侧模板（``DUAL_STREAM``，``batch --paired``）；每路整段参与，不切分。"""
    stop_evt = stop_evt or Event()
    raise_if_cancelled(stop_evt)
    front_video, side_video = Path(front_video_path), Path(side_video_path)
    action_scope = str(action_scope or "both").lower()
    pair = _load_pair(Path(front_template_path), Path(side_template_path), pose_variant, workers)

    def features(template: Template, video: Path, label: str):
        seq, fps, _views = extract_features(
            video, pose_variant=pair.pose_variant, workers=pair.workers, normalizer=pair.normalizer,
            compute_view=False, progress_cb=progress_cb, stop_evt=stop_evt,
        )
        raise_if_cancelled(stop_evt)
        _check_video(template, seq, tpl_label=f"{label} template", video_label=f"{label} video {video}")
        return seq, fps

    seq_f, fps_f = features(pair.front, front_video, "front")
    seq_s, fps_s = features(pair.side, side_video, "side")
    raw_f, raw_s = _lazy_raw(front_video, pair.pose_variant), _lazy_raw(side_video, pair.pose_variant)

    raise_if_cancelled(stop_evt)
    rules = (None, None)
    if enable_rules:
        rule_f = _rules(raw_f, None, view="front", action_scope=action_scope)
        raise_if_cancelled(stop_evt)
        rule_s = _rules(raw_s, None, view="side", action_scope=action_scope)
        raise_if_cancelled(stop_evt)
        rules = (rule_f, rule_s)

    raise_if_cancelled(stop_evt)
    front = score_view(seq_f, raw_f, pair.front.features, pair.front.meta, None, fps=fps_f, baseline=baseline,
                       enable_error_analysis=enable_error_analysis, stop_evt=stop_evt)
    raise_if_cancelled(stop_evt)
    side = score_view(seq_s, raw_s, pair.side.features, pair.side.meta, None, fps=fps_s, baseline=baseline,
                      enable_error_analysis=enable_error_analysis, stop_evt=stop_evt)
    raise_if_cancelled(stop_evt)
    segments = ((0, int(seq_f.shape[0] - 1)), (0, int(seq_s.shape[0] - 1)))
    result = _result(pair, front_video, fps_f, (w_front, w_side), front, side, segments, rules)
    raise_if_cancelled(stop_evt)
    return result
