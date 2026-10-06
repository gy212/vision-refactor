"""动作比对：模板加载与契约校验、姿态特征提取、单/双模板与正侧双流的 DTW 比对和模板生成（P3 按附录 B 拆分）。"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from typing import Callable

import numpy as np

from .pose_features import (
    DEFAULT_VALID_CONF_THR,
    MEDIAPIPE_VALIDITY_POLICY,
    derive_valid_mask,
    mirror_pose_features,
    normalize_pose_xy,
    normalize_pose_xy_v1,
    normalize_pose_xy_v3,
    pose_view_score,
    subsequence_dtw,
    subsequence_dtw_with_path,
)
from .evaluation.segmentation import active_window, estimate_period_frames as _estimate_period_frames  # noqa: F401
from .evaluation.segmentation import select_representative_cycle as _select_representative_cycle
from .evaluation.segmentation import split_front_side
from .foundation.signal import find_active_range, motion_energy, trimmed_mean
from .feature_layout import POSE33_V3, FeatureLayoutSpec, is_valid_feature_shape
from .paths import templates_dir
from .rule_scoring import (
    RuleViolation,
    extract_pose_raw,
    extract_pose_raw_series,
    score_rules,
    slice_pose_raw_series,
)
from .foundation.tasks import Cancelled, from_stopped
from .media.video_source import DecodeOptions, open_video
from .media.writer import open_video_writer
from .pose.engine import EngineSpec, create_engine
from .pose.extract import FORMAL, extract_pose_sequence, formal_engine
from .pose.sequence import GapPolicy


ProgressCb = Callable[[str, int, int], None]  # (stage, done, total)


def _raise_if_compare_cancelled(stop_evt: Event | None) -> None:
    if stop_evt is not None and stop_evt.is_set():
        raise InterruptedError("动作比对已取消")


# 历史模板曾把 feature_layout 写成 normalizer 描述字符串；Issue #6 后统一写注册表布局名。
LEGACY_DEFAULT_FEATURE_LAYOUT = "pose_indices_11_32_xy_rot_scale_norm"
LEGACY_POSE33_V3_FEATURE_LAYOUT = "pose_indices_11_32_xy_rot_scale_norm_v3"


# 模板 metadata 扩展（YOLO 迁移 S1 / Issue #6）：
# 新增字段全部为“增量、向后兼容”——旧模板缺这些键时由 normalize_template_meta 补默认值，
# 旧模板已有的历史 `feature_layout` 字符串不被覆盖，新模板则写入注册表布局名 `pose33_v3`。
def template_meta_defaults() -> dict:
    """返回 Issue #6 模板 metadata 字段及其默认值。

    用于：1) 写模板时补齐新字段；2) 加载旧模板（缺字段）时填默认值。
    ``validity_policy`` / ``valid_conf_thr`` 复用 ``core.pose_features`` 的集中式常量，
    与 raw 提取链路（``extract_pose_raw``）保持同一来源，避免字面量散落。
    """
    return {
        "backend": "mediapipe",
        "model_name": "pose_landmarker_full",
        "feature_layout": POSE33_V3.name,
        "normalizer_version": "v3",
        "confidence_kind": "visibility",
        "validity_policy": MEDIAPIPE_VALIDITY_POLICY,
        "valid_conf_thr": float(DEFAULT_VALID_CONF_THR),
    }


def normalize_template_meta(meta: dict) -> dict:
    """用默认值补齐缺失的新字段（不覆盖已存在的键），保证下游看到完整字段集。

    向后兼容关键点：旧模板没有这些键时补默认；旧模板已有的历史
    ``feature_layout`` 字符串不覆盖，保证 loader 仍可按后缀识别 v1/v2/v3。
    返回同一个 dict（原地补齐）。
    """
    if meta is None:
        meta = {}
    legacy_layout_name = meta.pop("feature_layout_name", None)
    if "feature_layout" not in meta and legacy_layout_name:
        meta["feature_layout"] = legacy_layout_name
    for key, value in template_meta_defaults().items():
        meta.setdefault(key, value)
    return meta


def _runtime_feature_layout(raw_meta: dict, normalized_meta: dict) -> str:
    """Return the layout that should drive matching, preserving old-template fallbacks."""
    if "feature_layout" in raw_meta:
        return str(raw_meta["feature_layout"])
    if "feature_layout_name" in raw_meta:
        return str(normalized_meta.get("feature_layout", raw_meta["feature_layout_name"]))
    return LEGACY_DEFAULT_FEATURE_LAYOUT


def _valid_mask_from_raw(landmarks: np.ndarray, meta: dict) -> np.ndarray:
    """Return the pose33 valid_mask emitted by raw extraction, deriving only for legacy meta."""
    mask = meta.get("valid_mask")
    if mask is not None:
        return np.asarray(mask, dtype=bool)
    thr = float(meta.get("valid_conf_thr") or DEFAULT_VALID_CONF_THR)
    return derive_valid_mask(landmarks, thr)


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


@dataclass(frozen=True)
class TemplateAutoCropAnalysis:
    """Diagnostics for the legacy single-action template auto crop."""

    start_frame: int
    end_frame: int
    frame_count: int
    fps: float
    activity_groups: tuple[tuple[int, int], ...]

    @property
    def selected_frames(self) -> int:
        return max(0, int(self.end_frame) - int(self.start_frame) + 1)

    @property
    def retained_ratio(self) -> float:
        return float(self.selected_frames) / float(max(1, int(self.frame_count)))

    @property
    def duration_s(self) -> float:
        return float(self.frame_count) / float(self.fps)

    @property
    def selected_duration_s(self) -> float:
        return float(self.selected_frames) / float(self.fps)

    @property
    def requires_review(self) -> bool:
        if self.duration_s < 30.0 or len(self.activity_groups) < 2:
            return False
        suspicious_limit_s = max(5.0, self.duration_s * 0.10)
        return self.selected_duration_s <= suspicious_limit_s

    def to_meta(self) -> dict:
        return {
            "auto_crop_strategy": "primary_activity_segment_v1",
            "auto_activity_group_count": len(self.activity_groups),
            "auto_activity_groups": [
                {"start_frame": int(start), "end_frame": int(end)}
                for start, end in self.activity_groups
            ],
            "auto_retained_ratio": float(self.retained_ratio),
            "auto_review_required": bool(self.requires_review),
        }


class TemplateAutoCropReviewRequired(ValueError):
    def __init__(self, analysis: TemplateAutoCropAnalysis):
        self.analysis = analysis
        ranges = ", ".join(
            f"{start}..{end} ({start / analysis.fps:.1f}-{end / analysis.fps:.1f}s)"
            for start, end in analysis.activity_groups[:12]
        )
        if len(analysis.activity_groups) > 12:
            ranges += f"，另有 {len(analysis.activity_groups) - 12} 段"
        super().__init__(
            "长视频自动裁剪需要人工确认："
            f"输入 {analysis.duration_s:.2f}s，当前算法只会保留 "
            f"{analysis.selected_duration_s:.2f}s "
            f"({analysis.retained_ratio * 100.0:.1f}%)，"
            f"并检测到 {len(analysis.activity_groups)} 个活动组。"
            "请填写 startFrame/endFrame（CLI 使用 --start/--end），"
            "或先按动作拆分视频后分别生成模板。"
            f"候选区间：{ranges}"
        )


@dataclass(frozen=True)
class RepetitionMatch:
    start_frame: int
    end_frame: int
    avg_cost: float
    score: float


@dataclass(frozen=True)
class JointErrorStat:
    joint: str
    mean_dist: float | None
    p90_dist: float | None
    max_dist: float | None
    valid_frames: int


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
    front_segment: tuple[int, int] | None = None  # (start,end) in original video frames
    side_segment: tuple[int, int] | None = None  # (start,end) in original video frames
    front_rule_score: int | None = None
    side_rule_score: int | None = None
    front_rule_deduction: int | None = None
    side_rule_deduction: int | None = None
    front_rule_violations: tuple[RuleViolation, ...] | None = None
    side_rule_violations: tuple[RuleViolation, ...] | None = None
    front_joint_errors: tuple[JointErrorStat, ...] | None = None
    side_joint_errors: tuple[JointErrorStat, ...] | None = None


def _extract_pose_features(
    video_path: Path,
    *,
    pose_variant: str,
    workers: int = 1,
    normalizer: Callable = normalize_pose_xy,
    compute_view: bool = False,
    layout: FeatureLayoutSpec = POSE33_V3,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """
    Extract normalized pose features for the entire video.
    Returns (features[T,J,2], fps, view_scores[T] or None), where J/D come from `layout`.

    ``FORMAL`` 提取 + ``features``（归一化返回 None 沿用上一帧，补零按 layout shape）+
    ``view_scores(HOLD_LAST)``。``workers > 1`` 走分段并行（每段前 15 帧跟踪预热）。
    """
    def progress(done: int, total: int) -> None:
        if progress_cb is not None:
            progress_cb("提取骨架特征", done, total)

    cancel = from_stopped(stop_evt.is_set) if stop_evt is not None else None
    try:
        seq = extract_pose_sequence(video_path, formal_engine(pose_variant), FORMAL, workers=workers,
                                    cancel=cancel, progress=progress)
        if cancel is not None:
            cancel.raise_if_cancelled()  # 最后一帧推理期间收到的取消也生效
    except Cancelled:
        raise InterruptedError("姿态特征提取已取消") from None
    if seq.decoded_count == 0:
        raise RuntimeError("视频为空或无法读取帧")
    zero_shape = tuple(int(x) for x in layout.shape)
    feats = seq.features(normalizer, zero_shape).copy()
    views = seq.view_scores(GapPolicy.HOLD_LAST).copy() if compute_view else None
    return feats, seq.meta.fps, views


def _swap_lr(name: str) -> str:
    if name.startswith("L_"):
        return "R_" + name[2:]
    if name.startswith("R_"):
        return "L_" + name[2:]
    return name


def _score_view_seq(
    seq: np.ndarray,
    raw_slice_fn,
    tpl_features: np.ndarray,
    tpl_meta: dict,
    seg: tuple[int, int] | None,
    *,
    fps: float,
    baseline: float,
    enable_error_analysis: bool,
    joint_names: list[str],
    src_idx: list[int],
    num_joints: int,
    stop_evt: Event | None = None,
) -> tuple[float, list[RepetitionMatch], list[JointErrorStat] | None]:
    _raise_if_compare_cancelled(stop_evt)
    # 2) 从标准模板中挑一个代表周期，避免整段重复导致匹配过长/过慢。
    tpl_fps = float(tpl_meta.get("fps") or fps)
    query = _select_representative_cycle(tpl_features, fps=tpl_fps)
    _raise_if_compare_cancelled(stop_evt)
    if query.size == 0:
        return 0.0, [], None

    # 3) 学员分段（缺失则整段）再收窄到运动活跃区间，减少静止帧对匹配的干扰。
    seg_seq, seg_offset = active_window(seq, seg)

    exclusion = max(3, int(round(0.2 * float(query.shape[0]))))
    max_matches = int(min(30, max(1, round(float(seg_seq.shape[0]) / max(1.0, float(query.shape[0]))))))

    matches = _multi_subsequence_matches(
        query,
        seg_seq,
        baseline=float(baseline),
        max_matches=max_matches,
        exclusion=exclusion,
        offset=seg_offset,
        stop_evt=stop_evt,
    )
    _raise_if_compare_cancelled(stop_evt)

    # 兜底：即便没有找到重复匹配，也保证有一个分数输出。
    if not matches and seg_seq.size > 0:
        cost, s, e = subsequence_dtw(query, seg_seq, stop_evt=stop_evt)
        avg = float(cost / max(1, int(query.shape[0])))
        matches = [
            RepetitionMatch(
                start_frame=int(seg_offset + s),
                end_frame=int(seg_offset + e),
                avg_cost=float(avg),
                score=float(float(baseline) / (float(baseline) + float(avg))),
            )
        ]

    scores = [m.score for m in matches]
    joint_stats: list[JointErrorStat] | None = None
    if enable_error_analysis and seg_seq.size > 0 and query.size > 0:
        _raise_if_compare_cancelled(stop_evt)
        q_m = mirror_pose_features(query)
        cost1, s1, e1, path1 = subsequence_dtw_with_path(
            query,
            seg_seq,
            stop_evt=stop_evt,
        )
        cost2, s2, e2, path2 = subsequence_dtw_with_path(
            q_m,
            seg_seq,
            stop_evt=stop_evt,
        )
        if float(cost2) < float(cost1):
            q_eff = q_m
            start_i = int(s2)
            end_i = int(e2)
            path = path2
            mirrored = True
        else:
            q_eff = query
            start_i = int(s1)
            end_i = int(e1)
            path = path1
            mirrored = False

        if start_i <= end_i and path:
            raw, raw_meta = raw_slice_fn(
                int(seg_offset),
                int(seg_offset + int(seg_seq.shape[0]) - 1),
            )
            raw_mask = _valid_mask_from_raw(raw, raw_meta)
            path_arr = np.asarray(path, dtype=np.int64)
            if path_arr.size:
                valid_path = (
                    (path_arr[:, 0] >= 0)
                    & (path_arr[:, 0] < int(q_eff.shape[0]))
                    & (path_arr[:, 1] >= 0)
                    & (path_arr[:, 1] < int(seg_seq.shape[0]))
                )
                path_arr = path_arr[valid_path]
            names = joint_names if not mirrored else [_swap_lr(n) for n in joint_names]
            joint_stats = []
            if path_arr.size:
                q_path = q_eff[path_arr[:, 0]].astype(np.float32, copy=False)
                s_path = seg_seq[path_arr[:, 1]].astype(np.float32, copy=False)
                distances = np.linalg.norm(q_path - s_path, axis=2).astype(np.float32, copy=False)
                visible = np.asarray(raw_mask[path_arr[:, 1]][:, src_idx], dtype=bool)
                finite = np.isfinite(distances)
                for k in range(num_joints):
                    _raise_if_compare_cancelled(stop_evt)
                    arr = distances[visible[:, k] & finite[:, k], k]
                    if arr.size == 0:
                        joint_stats.append(
                            JointErrorStat(
                                joint=str(names[k]),
                                mean_dist=None,
                                p90_dist=None,
                                max_dist=None,
                                valid_frames=0,
                            )
                        )
                    else:
                        joint_stats.append(
                            JointErrorStat(
                                joint=str(names[k]),
                                mean_dist=float(np.mean(arr)),
                                p90_dist=float(np.percentile(arr, 90)),
                                max_dist=float(np.max(arr)),
                                valid_frames=int(arr.size),
                            )
                        )
            else:
                joint_stats = [
                    JointErrorStat(
                        joint=str(name),
                        mean_dist=None,
                        p90_dist=None,
                        max_dist=None,
                        valid_frames=0,
                    )
                    for name in names
                ]

    _raise_if_compare_cancelled(stop_evt)
    return trimmed_mean(scores), matches, joint_stats


def _feature_frame_shape(features: np.ndarray) -> tuple[int, ...]:
    arr = np.asarray(features)
    if arr.ndim < 2:
        return tuple(int(x) for x in arr.shape)
    if arr.ndim == 2:
        return tuple(int(x) for x in arr.shape)
    return tuple(int(x) for x in arr.shape[1:])


def _assert_feature_layout_match(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_label: str,
    right_label: str,
    left_layout: str | None = None,
    right_layout: str | None = None,
) -> None:
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    left_shape = _feature_frame_shape(left_arr)
    right_shape = _feature_frame_shape(right_arr)
    left_layout_c = _canonical_feature_layout(left_layout)
    right_layout_c = _canonical_feature_layout(right_layout)
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


def _canonical_feature_layout(layout: str | None) -> str | None:
    """Map legacy layout aliases to the registered layout name used in new metadata."""
    if layout is None:
        return None
    layout_s = str(layout)
    if layout_s == POSE33_V3.name or layout_s == LEGACY_POSE33_V3_FEATURE_LAYOUT:
        return POSE33_V3.name
    return layout_s


def _normalizer_version_from_layout(layout: str) -> str:
    """Resolve normalizer version from current layout names and legacy aliases."""
    layout = str(layout)
    if _canonical_feature_layout(layout) == POSE33_V3.name or layout.endswith("_v3"):
        return "v3"
    if layout.endswith("_v2"):
        return "v2"
    return "v1"



def _multi_subsequence_matches(
    query: np.ndarray,
    seq: np.ndarray,
    *,
    baseline: float = 3.0,
    max_matches: int = 30,
    exclusion: int = 5,
    offset: int = 0,
    stop_evt: Event | None = None,
) -> list[RepetitionMatch]:
    """
    Greedy multi-match: repeatedly find the best subsequence DTW match, exclude it, and repeat.
    Returns matches in arbitrary order (not necessarily chronological).
    """
    _raise_if_compare_cancelled(stop_evt)
    if query.size == 0 or seq.size == 0:
        return []

    q = query
    q_m = mirror_pose_features(query)
    q_len = int(q.shape[0])
    if q_len <= 0:
        return []

    remaining: list[tuple[int, int]] = [(0, int(seq.shape[0] - 1))]
    out: list[RepetitionMatch] = []

    ref_avg: float | None = None

    while remaining and len(out) < int(max_matches):
        _raise_if_compare_cancelled(stop_evt)
        best: tuple[float, int, int] | None = None  # (avg_cost, start, end) in seq-local indices

        for seg_s, seg_e in remaining:
            _raise_if_compare_cancelled(stop_evt)
            if seg_e - seg_s + 1 < max(8, q_len // 2):
                continue
            sub = seq[seg_s : seg_e + 1]
            cost1, s1, e1 = subsequence_dtw(q, sub, stop_evt=stop_evt)
            cost2, s2, e2 = subsequence_dtw(q_m, sub, stop_evt=stop_evt)
            avg1 = float(cost1 / max(1, q_len))
            avg2 = float(cost2 / max(1, q_len))
            if avg2 < avg1:
                avg, s, e = avg2, int(s2), int(e2)
            else:
                avg, s, e = avg1, int(s1), int(e1)
            s += int(seg_s)
            e += int(seg_s)

            if best is None or avg < best[0]:
                best = (avg, s, e)

        if best is None:
            break

        avg, s, e = best

        # Stop once matching becomes much worse than the best (avoid collecting random matches).
        if ref_avg is None:
            ref_avg = float(avg)
        else:
            if float(avg) > float(ref_avg) * 2.5 + 0.5:
                break

        score = float(baseline / (baseline + float(avg)))
        out.append(
            RepetitionMatch(
                start_frame=int(offset + s),
                end_frame=int(offset + e),
                avg_cost=float(avg),
                score=float(score),
            )
        )

        # Exclude matched region (with padding) and keep remaining segments.
        exc_s = max(0, int(s - exclusion))
        exc_e = min(int(seq.shape[0] - 1), int(e + exclusion))
        new_remaining: list[tuple[int, int]] = []
        for seg_s, seg_e in remaining:
            if exc_e < seg_s or exc_s > seg_e:
                new_remaining.append((seg_s, seg_e))
                continue
            if seg_s < exc_s:
                new_remaining.append((seg_s, exc_s - 1))
            if exc_e < seg_e:
                new_remaining.append((exc_e + 1, seg_e))
        remaining = new_remaining

    _raise_if_compare_cancelled(stop_evt)
    return out


def analyze_template_auto_crop(
    energy: np.ndarray,
    *,
    fps: float,
) -> TemplateAutoCropAnalysis:
    """Analyze whether the single-action auto crop is unsafe for a long video.

    The selected range intentionally remains identical to ``find_active_range``.
    Activity groups are diagnostics only: they prevent a long multi-activity video
    from silently becoming a tiny template without changing scoring/exam callers.
    """
    values = np.asarray(energy, dtype=np.float32).reshape(-1)
    fps_value = float(fps or 0.0) or 30.0
    frame_count = int(values.size + 1)
    auto_start, auto_end = find_active_range(values, pad=10)

    if values.size == 0:
        groups: tuple[tuple[int, int], ...] = ()
    else:
        smooth_frames = max(3, int(round(0.30 * fps_value)))
        kernel = np.ones(smooth_frames, dtype=np.float32) / float(smooth_frames)
        smoothed = (
            np.convolve(values, kernel, mode="same")
            if values.size >= smooth_frames
            else values
        )
        active = smoothed > float(np.percentile(smoothed, 70))
        runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for index, is_active in enumerate(active.tolist()):
            if is_active and run_start is None:
                run_start = index
            elif (not is_active) and run_start is not None:
                runs.append((run_start, index - 1))
                run_start = None
        if run_start is not None:
            runs.append((run_start, int(active.size - 1)))

        min_active_frames = max(2, int(round(0.15 * fps_value)))
        runs = [
            (start, end)
            for start, end in runs
            if end - start + 1 >= min_active_frames
        ]
        merge_gap_frames = max(1, int(round(2.0 * fps_value)))
        merged: list[tuple[int, int]] = []
        for start, end in runs:
            if merged and start - merged[-1][1] - 1 <= merge_gap_frames:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))

        group_pad_frames = max(1, int(round(0.50 * fps_value)))
        groups = tuple(
            (
                max(0, int(start - group_pad_frames)),
                min(frame_count - 1, int(end + 1 + group_pad_frames)),
            )
            for start, end in merged
        )

    return TemplateAutoCropAnalysis(
        start_frame=int(auto_start),
        end_frame=int(auto_end),
        frame_count=frame_count,
        fps=fps_value,
        activity_groups=groups,
    )


def create_template_from_video(
    video_path: str | Path,
    *,
    pose_variant: str = "heavy",
    start: int | None = None,
    end: int | None = None,
    out_path: str | Path | None = None,
    workers: int = 1,
    preview: bool = False,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> Path:
    video_path = Path(video_path)
    features, fps, _view = _extract_pose_features(
        video_path,
        pose_variant=pose_variant,
        workers=workers,
        normalizer=normalize_pose_xy_v3,
        progress_cb=progress_cb,
        stop_evt=stop_evt,
    )

    seq = features.reshape(features.shape[0], -1)
    energy = motion_energy(seq)
    auto_crop = analyze_template_auto_crop(energy, fps=fps)
    auto_start, auto_end = auto_crop.start_frame, auto_crop.end_frame
    if start is None and end is None and auto_crop.requires_review:
        raise TemplateAutoCropReviewRequired(auto_crop)

    start_i = int(start) if start is not None else int(auto_start)
    end_i = int(end) if end is not None else int(auto_end)
    start_i = max(0, min(start_i, features.shape[0] - 1))
    end_i = max(start_i, min(end_i, features.shape[0] - 1))

    out_dir = templates_dir()
    out_path = Path(out_path) if out_path else (out_dir / f"{video_path.stem}_{pose_variant}.npz")

    meta = {
        "video": str(video_path),
        "fps": float(fps),
        "frame_count": int(features.shape[0]),
        "start_frame": int(start_i),
        "end_frame": int(end_i),
        "auto_start_frame": int(auto_start),
        "auto_end_frame": int(auto_end),
        "pose_variant": pose_variant,
        "feature_layout": POSE33_V3.name,
        "running_mode": "video",
        "cfg": asdict(EngineSpec(pose_variant=pose_variant, running_mode="video", enable_hands=False)),
        # 模板 metadata 扩展（YOLO 迁移 S1 / Issue #6）：增量字段，向后兼容。
        "backend": "mediapipe",
        "model_name": f"pose_landmarker_{pose_variant}",
        "normalizer_version": "v3",
        "confidence_kind": "visibility",
        "validity_policy": MEDIAPIPE_VALIDITY_POLICY,
        "valid_conf_thr": float(DEFAULT_VALID_CONF_THR),
        "crop_selection": "manual" if start is not None or end is not None else "auto",
        "selected_retained_ratio": float((end_i - start_i + 1) / features.shape[0]),
        "template_scope": "single_action",
        **auto_crop.to_meta(),
    }

    np.savez_compressed(
        out_path,
        features=features[start_i : end_i + 1],
        meta=np.array(meta, dtype=object),
    )

    if preview and progress_cb is not None:
        progress_cb("导出模板预览", 0, 1)
    if preview:
        preview_path = out_path.with_suffix(".preview.mp4")
        export_match_preview(
            video_path,
            out_path=preview_path,
            start_frame=start_i,
            end_frame=end_i,
            pose_variant=pose_variant,
            progress_cb=progress_cb,
            stop_evt=stop_evt,
        )

    return out_path


def compare_video_to_template(
    template_path: str | Path,
    video_path: str | Path,
    *,
    pose_variant: str | None = None,
    workers: int = 1,
    preview_out: str | Path | None = None,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> CompareResult:
    stop_evt = stop_evt or Event()
    template_path = Path(template_path)
    video_path = Path(video_path)

    tpl = np.load(template_path, allow_pickle=True)
    query = tpl["features"]
    raw_meta = dict(tpl["meta"].item() or {})
    meta = normalize_template_meta(dict(raw_meta))
    pv = pose_variant or meta.get("pose_variant", "full")
    layout = _runtime_feature_layout(raw_meta, meta)
    normalizer_version = _normalizer_version_from_layout(layout)
    if normalizer_version == "v3":
        normalizer = normalize_pose_xy_v3
    elif normalizer_version == "v2":
        normalizer = normalize_pose_xy
    else:
        normalizer = normalize_pose_xy_v1
    tpl_mode = str(meta.get("running_mode") or (meta.get("cfg") or {}).get("running_mode") or "video").lower()
    workers_eff = 1 if (tpl_mode == "video" and int(workers) > 1) else int(workers)
    workers_eff = max(1, workers_eff)

    seq, fps, _view = _extract_pose_features(
        video_path,
        pose_variant=pv,
        workers=workers_eff,
        normalizer=normalizer,
        progress_cb=progress_cb,
        stop_evt=stop_evt,
    )
    _assert_feature_layout_match(
        query,
        seq,
        left_label=f"template {template_path}",
        right_label=f"video {video_path}",
        left_layout=layout,
        right_layout=layout,
    )

    if progress_cb is not None:
        progress_cb("计算相似度", 0, 1)
    t0 = time.monotonic()
    cost, start, end = subsequence_dtw(query, seq)
    avg_cost = cost / max(1, int(query.shape[0]))
    # Baseline normalization: score=1.0 when avg_cost=0, score=0.5 when avg_cost=baseline
    baseline = 3.0
    score = float(baseline / (baseline + avg_cost))
    _ = time.monotonic() - t0

    actual_preview_path: Path | None = None
    if preview_out:
        actual_preview_path = export_match_preview(
            video_path,
            out_path=Path(preview_out),
            start_frame=int(start),
            end_frame=int(end),
            pose_variant=pv,
            progress_cb=progress_cb,
            stop_evt=stop_evt,
        )

    return CompareResult(
        template_path=template_path,
        video_path=video_path,
        pose_variant=pv,
        fps=float(fps),
        start_frame=int(start),
        end_frame=int(end),
        cost=float(cost),
        avg_cost=float(avg_cost),
        score=float(score),
        preview_path=actual_preview_path,
        workers_used=workers_eff,
    )


def compare_video_to_templates(
    template_paths: list[str | Path],
    video_path: str | Path,
    *,
    pose_variant: str | None = None,
    workers: int = 1,
    baseline: float = 3.0,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> list[CompareResult]:
    """一条视频 vs 多个模板：**视频姿态只提取一次**，复用给池内每个模板做 DTW。

    池内模板须同 feature_layout（正常均为 pose33_v3 heavy）；以第一个模板决定
    normalizer/pose_variant。返回与 ``template_paths`` 同序的 ``CompareResult`` 列表。
    空列表输入返回空列表。
    """
    stop_evt = stop_evt or Event()
    paths = [Path(p) for p in template_paths]
    if not paths:
        return []

    # 加载全部模板 query + 校验同布局（以首个为准）
    queries: list[np.ndarray] = []
    metas: list[dict] = []
    for tp in paths:
        tpl = np.load(tp, allow_pickle=True)
        queries.append(tpl["features"])
        metas.append(dict(tpl["meta"].item() or {}))

    raw_meta0 = metas[0]
    meta0 = normalize_template_meta(dict(raw_meta0))
    pv = pose_variant or meta0.get("pose_variant", "full")
    layout = _runtime_feature_layout(raw_meta0, meta0)
    normalizer_version = _normalizer_version_from_layout(layout)
    if normalizer_version == "v3":
        normalizer = normalize_pose_xy_v3
    elif normalizer_version == "v2":
        normalizer = normalize_pose_xy
    else:
        normalizer = normalize_pose_xy_v1
    tpl_mode = str(meta0.get("running_mode") or (meta0.get("cfg") or {}).get("running_mode") or "video").lower()
    workers_eff = 1 if (tpl_mode == "video" and int(workers) > 1) else int(workers)
    workers_eff = max(1, workers_eff)

    # 视频姿态只提取一次
    seq, fps, _view = _extract_pose_features(
        video_path,
        pose_variant=pv,
        workers=workers_eff,
        normalizer=normalizer,
        progress_cb=progress_cb,
        stop_evt=stop_evt,
    )

    results: list[CompareResult] = []
    for tp, query in zip(paths, queries):
        _raise_if_compare_cancelled(stop_evt)
        _assert_feature_layout_match(
            query,
            seq,
            left_label=f"template {tp}",
            right_label=f"video {video_path}",
            left_layout=layout,
            right_layout=layout,
        )
        cost, start, end = subsequence_dtw(query, seq, stop_evt=stop_evt)
        avg_cost = cost / max(1, int(query.shape[0]))
        score = float(baseline / (baseline + avg_cost))
        results.append(
            CompareResult(
                template_path=tp,
                video_path=Path(video_path),
                pose_variant=pv,
                fps=float(fps),
                start_frame=int(start),
                end_frame=int(end),
                cost=float(cost),
                avg_cost=float(avg_cost),
                score=float(score),
                preview_path=None,
                workers_used=workers_eff,
            )
        )
    return results


def compare_video_to_dual_templates(
    front_template_path: str | Path,
    side_template_path: str | Path,
    video_path: str | Path,
    *,
    pose_variant: str | None = None,
    workers: int = 1,
    w_front: float = 0.4,
    w_side: float = 0.6,
    baseline: float = 3.0,
    enable_rules: bool = False,
    action_scope: str = "both",
    enable_error_analysis: bool = False,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> DualCompareResult:
    """
    Compare a student's single long video (contains both front+side view) against two standard templates.

    - Auto-split student video into front/side segments using pose "frontness" score.
    - For each view, auto-pick a single repetition from the standard template as the DTW query.
    - Run greedy multi-match subsequence DTW to score multiple repetitions.
    - Aggregate by trimmed mean: drop max & min, then average (when >=3 reps).
    - Optional: rule-based scoring on raw Pose33 (enable_rules/action_scope).

    Returns both per-view scores (0..1) and combined integer percent (0..100).
    """
    stop_evt = stop_evt or Event()
    front_template_path = Path(front_template_path)
    side_template_path = Path(side_template_path)
    video_path = Path(video_path)
    action_scope = str(action_scope or "both").lower()

    tpl_f = np.load(front_template_path, allow_pickle=True)
    tpl_s = np.load(side_template_path, allow_pickle=True)

    feat_f = tpl_f["features"]
    feat_s = tpl_s["features"]
    raw_meta_f = dict(tpl_f["meta"].item() or {})
    raw_meta_s = dict(tpl_s["meta"].item() or {})
    meta_f = normalize_template_meta(dict(raw_meta_f))
    meta_s = normalize_template_meta(dict(raw_meta_s))

    layout_f = _runtime_feature_layout(raw_meta_f, meta_f)
    layout_s = _runtime_feature_layout(raw_meta_s, meta_s)
    _assert_feature_layout_match(
        feat_f,
        feat_s,
        left_label=f"front template {front_template_path}",
        right_label=f"side template {side_template_path}",
        left_layout=layout_f,
        right_layout=layout_s,
    )

    layout_ver_f = _normalizer_version_from_layout(layout_f)
    layout_ver_s = _normalizer_version_from_layout(layout_s)
    if layout_ver_f != layout_ver_s:
        raise ValueError("Front/side templates use different feature layouts; regenerate templates with the same version.")

    if layout_ver_f == "v3":
        normalizer = normalize_pose_xy_v3
    elif layout_ver_f == "v2":
        normalizer = normalize_pose_xy
    else:
        normalizer = normalize_pose_xy_v1
    pv = pose_variant or meta_f.get("pose_variant") or meta_s.get("pose_variant") or "full"

    tpl_mode_f = str(meta_f.get("running_mode") or (meta_f.get("cfg") or {}).get("running_mode") or "video").lower()
    tpl_mode_s = str(meta_s.get("running_mode") or (meta_s.get("cfg") or {}).get("running_mode") or "video").lower()
    tpl_video_mode = (tpl_mode_f == "video") or (tpl_mode_s == "video")
    workers_eff = 1 if (tpl_video_mode and int(workers) > 1) else int(workers)
    workers_eff = max(1, workers_eff)

    seq, fps, view_scores = _extract_pose_features(
        video_path,
        pose_variant=pv,
        workers=workers_eff,
        normalizer=normalizer,
        compute_view=True,
        progress_cb=progress_cb,
        stop_evt=stop_evt,
    )
    _assert_feature_layout_match(
        feat_f,
        seq,
        left_label=f"front template {front_template_path}",
        right_label=f"video {video_path}",
        left_layout=layout_f,
        right_layout=layout_f,
    )
    _assert_feature_layout_match(
        feat_s,
        seq,
        left_label=f"side template {side_template_path}",
        right_label=f"video {video_path}",
        left_layout=layout_s,
        right_layout=layout_s,
    )

    # 双模板比对核心流程：
    # A) 正/侧视角切分；B) DTW 多次匹配与聚合；C) 可选规则扣分与明细输出。
    raw_series = None

    def _raw_slice(start_frame: int | None, end_frame: int | None) -> tuple[np.ndarray, dict]:
        nonlocal raw_series
        if raw_series is None:
            raw_series = extract_pose_raw_series(video_path, pose_variant=pv)
        return slice_pose_raw_series(raw_series, start_frame=start_frame, end_frame=end_frame)

    # 1) 按“正面程度”把长视频拆成正/侧两段（无需手动标注转身点），转身附近的临界帧两端各收一段。
    front_seg: tuple[int, int] | None = None
    side_seg: tuple[int, int] | None = None
    if view_scores is not None and view_scores.size == seq.shape[0] and seq.shape[0] >= 10:
        front_seg, side_seg, _mode = split_front_side(view_scores, fps, mode="median_always")

    # 1.5) 可选规则评分：基于原始 Pose33，分视角段独立评估。
    #      规则采用“违规帧占比 >= 阈值则扣分”，降低单帧抖动影响。
    rule_front = None
    rule_side = None
    if enable_rules:
        def _score_rules_for_seg(seg: tuple[int, int] | None, *, view: str):
            if seg is None:
                s = e = None
            else:
                s, e = int(seg[0]), int(seg[1])
            raw, raw_meta = _raw_slice(s, e)
            return score_rules(
                raw,
                view=view,
                action_scope=action_scope,
                valid_mask=_valid_mask_from_raw(raw, raw_meta),
            )

        rule_front = _score_rules_for_seg(front_seg, view="front")
        rule_side = _score_rules_for_seg(side_seg, view="side")

    # 关节误差统计按 layout 取名/取数：本期双模板固定走 pose33_v3 布局（(22,2)）。
    # joint_names / source_indices / num_joints 全部来自 layout，热路径不再写死 22 / 列表字面量。
    error_layout = POSE33_V3
    joint_names = list(error_layout.joint_names)
    src_idx = list(error_layout.source_indices)
    num_joints = error_layout.num_joints

    def score_view(
        tpl_features: np.ndarray,
        tpl_meta: dict,
        seg: tuple[int, int] | None,
    ) -> tuple[float, list[RepetitionMatch], list[JointErrorStat] | None]:
        return _score_view_seq(
            seq,
            _raw_slice,
            tpl_features,
            tpl_meta,
            seg,
            fps=fps,
            baseline=baseline,
            enable_error_analysis=enable_error_analysis,
            joint_names=joint_names,
            src_idx=src_idx,
            num_joints=num_joints,
        )

    front_score, front_matches, front_joint_errors = score_view(feat_f, meta_f, front_seg)
    side_score, side_matches, side_joint_errors = score_view(feat_s, meta_s, side_seg)

    combined = float((w_front * float(front_score)) + (w_side * float(side_score)))
    combined = float(np.clip(combined, 0.0, 1.0))
    pct = int(np.clip(int(round(combined * 100.0)), 0, 100))

    return DualCompareResult(
        front_template_path=front_template_path,
        side_template_path=side_template_path,
        video_path=video_path,
        pose_variant=pv,
        fps=float(fps),
        front_score=float(front_score),
        side_score=float(side_score),
        combined_score=float(combined),
        combined_percent=int(pct),
        front_matches=tuple(front_matches),
        side_matches=tuple(side_matches),
        front_segment=front_seg,
        side_segment=side_seg,
        front_rule_score=None if rule_front is None else int(rule_front.score),
        side_rule_score=None if rule_side is None else int(rule_side.score),
        front_rule_deduction=None if rule_front is None else int(rule_front.total_deduction),
        side_rule_deduction=None if rule_side is None else int(rule_side.total_deduction),
        front_rule_violations=None if rule_front is None else tuple(rule_front.violations),
        side_rule_violations=None if rule_side is None else tuple(rule_side.violations),
        front_joint_errors=None if front_joint_errors is None else tuple(front_joint_errors),
        side_joint_errors=None if side_joint_errors is None else tuple(side_joint_errors),
    )


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
    baseline: float = 3.0,
    enable_rules: bool = False,
    action_scope: str = "both",
    enable_error_analysis: bool = False,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> DualCompareResult:
    """
    Compare two independently-synced streams (front camera + side camera) against
    front/side standard templates. Unlike compare_video_to_dual_templates (single
    video, frontness-based auto split), each stream is already known-view and is
    scored on its full length — no split needed.
    """
    stop_evt = stop_evt or Event()
    _raise_if_compare_cancelled(stop_evt)
    front_template_path = Path(front_template_path)
    side_template_path = Path(side_template_path)
    front_video_path = Path(front_video_path)
    side_video_path = Path(side_video_path)
    action_scope = str(action_scope or "both").lower()

    tpl_f = np.load(front_template_path, allow_pickle=True)
    tpl_s = np.load(side_template_path, allow_pickle=True)

    feat_f = tpl_f["features"]
    feat_s = tpl_s["features"]
    raw_meta_f = dict(tpl_f["meta"].item() or {})
    raw_meta_s = dict(tpl_s["meta"].item() or {})
    meta_f = normalize_template_meta(dict(raw_meta_f))
    meta_s = normalize_template_meta(dict(raw_meta_s))

    layout_f = _runtime_feature_layout(raw_meta_f, meta_f)
    layout_s = _runtime_feature_layout(raw_meta_s, meta_s)
    _assert_feature_layout_match(
        feat_f,
        feat_s,
        left_label=f"front template {front_template_path}",
        right_label=f"side template {side_template_path}",
        left_layout=layout_f,
        right_layout=layout_s,
    )

    layout_ver_f = _normalizer_version_from_layout(layout_f)
    layout_ver_s = _normalizer_version_from_layout(layout_s)
    if layout_ver_f != layout_ver_s:
        raise ValueError("Front/side templates use different feature layouts; regenerate templates with the same version.")

    if layout_ver_f == "v3":
        normalizer = normalize_pose_xy_v3
    elif layout_ver_f == "v2":
        normalizer = normalize_pose_xy
    else:
        normalizer = normalize_pose_xy_v1
    pv = pose_variant or meta_f.get("pose_variant") or meta_s.get("pose_variant") or "full"

    tpl_mode_f = str(meta_f.get("running_mode") or (meta_f.get("cfg") or {}).get("running_mode") or "video").lower()
    tpl_mode_s = str(meta_s.get("running_mode") or (meta_s.get("cfg") or {}).get("running_mode") or "video").lower()
    tpl_video_mode = (tpl_mode_f == "video") or (tpl_mode_s == "video")
    workers_eff = 1 if (tpl_video_mode and int(workers) > 1) else int(workers)
    workers_eff = max(1, workers_eff)

    seq_f, fps_f, _view_f = _extract_pose_features(
        front_video_path,
        pose_variant=pv,
        workers=workers_eff,
        normalizer=normalizer,
        compute_view=False,
        progress_cb=progress_cb,
        stop_evt=stop_evt,
    )
    _raise_if_compare_cancelled(stop_evt)
    _assert_feature_layout_match(
        feat_f,
        seq_f,
        left_label=f"front template {front_template_path}",
        right_label=f"front video {front_video_path}",
        left_layout=layout_f,
        right_layout=layout_f,
    )

    seq_s, fps_s, _view_s = _extract_pose_features(
        side_video_path,
        pose_variant=pv,
        workers=workers_eff,
        normalizer=normalizer,
        compute_view=False,
        progress_cb=progress_cb,
        stop_evt=stop_evt,
    )
    _raise_if_compare_cancelled(stop_evt)
    _assert_feature_layout_match(
        feat_s,
        seq_s,
        left_label=f"side template {side_template_path}",
        right_label=f"side video {side_video_path}",
        left_layout=layout_s,
        right_layout=layout_s,
    )

    raw_series_f = None
    raw_series_s = None

    def _raw_slice_f(start_frame: int | None, end_frame: int | None) -> tuple[np.ndarray, dict]:
        nonlocal raw_series_f
        if raw_series_f is None:
            raw_series_f = extract_pose_raw_series(front_video_path, pose_variant=pv)
        return slice_pose_raw_series(raw_series_f, start_frame=start_frame, end_frame=end_frame)

    def _raw_slice_s(start_frame: int | None, end_frame: int | None) -> tuple[np.ndarray, dict]:
        nonlocal raw_series_s
        if raw_series_s is None:
            raw_series_s = extract_pose_raw_series(side_video_path, pose_variant=pv)
        return slice_pose_raw_series(raw_series_s, start_frame=start_frame, end_frame=end_frame)

    rule_front = None
    rule_side = None
    _raise_if_compare_cancelled(stop_evt)
    if enable_rules:
        raw_f, raw_meta_f_slice = _raw_slice_f(None, None)
        _raise_if_compare_cancelled(stop_evt)
        rule_front = score_rules(
            raw_f,
            view="front",
            action_scope=action_scope,
            valid_mask=_valid_mask_from_raw(raw_f, raw_meta_f_slice),
        )
        _raise_if_compare_cancelled(stop_evt)
        raw_s, raw_meta_s_slice = _raw_slice_s(None, None)
        _raise_if_compare_cancelled(stop_evt)
        rule_side = score_rules(
            raw_s,
            view="side",
            action_scope=action_scope,
            valid_mask=_valid_mask_from_raw(raw_s, raw_meta_s_slice),
        )
        _raise_if_compare_cancelled(stop_evt)

    error_layout = POSE33_V3
    joint_names = list(error_layout.joint_names)
    src_idx = list(error_layout.source_indices)
    num_joints = error_layout.num_joints

    _raise_if_compare_cancelled(stop_evt)
    front_score, front_matches, front_joint_errors = _score_view_seq(
        seq_f,
        _raw_slice_f,
        feat_f,
        meta_f,
        None,
        fps=fps_f,
        baseline=baseline,
        enable_error_analysis=enable_error_analysis,
        joint_names=joint_names,
        src_idx=src_idx,
        num_joints=num_joints,
        stop_evt=stop_evt,
    )
    _raise_if_compare_cancelled(stop_evt)
    side_score, side_matches, side_joint_errors = _score_view_seq(
        seq_s,
        _raw_slice_s,
        feat_s,
        meta_s,
        None,
        fps=fps_s,
        baseline=baseline,
        enable_error_analysis=enable_error_analysis,
        joint_names=joint_names,
        src_idx=src_idx,
        num_joints=num_joints,
        stop_evt=stop_evt,
    )
    _raise_if_compare_cancelled(stop_evt)

    combined = float((w_front * float(front_score)) + (w_side * float(side_score)))
    combined = float(np.clip(combined, 0.0, 1.0))
    pct = int(np.clip(int(round(combined * 100.0)), 0, 100))
    _raise_if_compare_cancelled(stop_evt)

    return DualCompareResult(
        front_template_path=front_template_path,
        side_template_path=side_template_path,
        video_path=front_video_path,
        pose_variant=pv,
        fps=float(fps_f),
        front_score=float(front_score),
        side_score=float(side_score),
        combined_score=float(combined),
        combined_percent=int(pct),
        front_matches=tuple(front_matches),
        side_matches=tuple(side_matches),
        front_segment=(0, int(seq_f.shape[0] - 1)),
        side_segment=(0, int(seq_s.shape[0] - 1)),
        front_rule_score=None if rule_front is None else int(rule_front.score),
        side_rule_score=None if rule_side is None else int(rule_side.score),
        front_rule_deduction=None if rule_front is None else int(rule_front.total_deduction),
        side_rule_deduction=None if rule_side is None else int(rule_side.total_deduction),
        front_rule_violations=None if rule_front is None else tuple(rule_front.violations),
        side_rule_violations=None if rule_side is None else tuple(rule_side.violations),
        front_joint_errors=None if front_joint_errors is None else tuple(front_joint_errors),
        side_joint_errors=None if side_joint_errors is None else tuple(side_joint_errors),
    )


def write_pose_preview(
    video_path: Path,
    *,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    pose_variant: str,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> tuple[Path, str]:
    """把 ``[start_frame, end_frame]`` 逐帧推理并画骨架写成预览视频；返回 ``(实际路径, 编码)``。"""
    cancel = from_stopped(stop_evt.is_set) if stop_evt is not None else None
    total = max(0, int(end_frame - start_frame + 1))
    with open_video(video_path, DecodeOptions(start=int(start_frame), end=int(end_frame)), cancel) as source:
        meta = source.meta
        vw, actual_path, codec = open_video_writer(out_path, fps=meta.fps, size=(meta.width, meta.height))
        pipe = create_engine(formal_engine(pose_variant))
        done = 0
        try:
            for frame in source:
                annotated, _actions = pipe.annotate(frame.bgr, timestamp_ms=frame.t_ms)
                vw.write(annotated)
                done += 1
                if progress_cb is not None and (done % 5 == 0 or done == total):
                    progress_cb("导出匹配预览", done, total)
        except Cancelled:
            pass  # 取消时保留已写出的部分（与旧实现一致）
        finally:
            vw.release()
            pipe.close()
    return actual_path, codec


def export_match_preview(
    video_path: Path,
    *,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    pose_variant: str,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> Path:
    path, _codec = write_pose_preview(
        video_path, out_path=out_path, start_frame=start_frame, end_frame=end_frame,
        pose_variant=pose_variant, progress_cb=progress_cb, stop_evt=stop_evt,
    )
    return path
