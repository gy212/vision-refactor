# -*- coding: utf-8 -*-
"""重复动作的匹配与关节误差：贪心多次子序列 DTW、单视角打分（代表周期 → 活跃窗口 → 多次匹配 → 截尾均值）、逐关节误差统计。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event

import numpy as np

from core.evaluation.matching.dtw import subsequence_dtw, subsequence_dtw_with_path
from core.evaluation.matching.engine import raise_if_cancelled
from core.evaluation.scoring.cost import POSE33_BASELINE, avg_cost_of, score_from_avg_cost
from core.evaluation.segmentation import active_window, select_representative_cycle
from core.foundation.signal import trimmed_mean
from core.pose.layouts import POSE33_V3, FeatureLayoutSpec
from core.pose.normalize import mirror_pose_features
from core.pose.validity import DEFAULT_VALID_CONF_THR, derive_valid_mask

RawSlice = Callable[[int | None, int | None], tuple[np.ndarray, dict]]


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


def valid_mask_from_raw(landmarks: np.ndarray, meta: dict) -> np.ndarray:
    """原始提取带出的 ``(T,33)`` valid_mask；只有旧 meta 缺它时才按阈值现场推导。"""
    mask = meta.get("valid_mask")
    if mask is not None:
        return np.asarray(mask, dtype=bool)
    thr = float(meta.get("valid_conf_thr") or DEFAULT_VALID_CONF_THR)
    return derive_valid_mask(landmarks, thr)


def _best_of_mirror(q: np.ndarray, q_m: np.ndarray, sub: np.ndarray, q_len: int,
                    stop_evt: Event | None) -> tuple[float, int, int]:
    cost1, s1, e1 = subsequence_dtw(q, sub, stop_evt=stop_evt)
    cost2, s2, e2 = subsequence_dtw(q_m, sub, stop_evt=stop_evt)
    avg1 = float(avg_cost_of(cost1, q_len))
    avg2 = float(avg_cost_of(cost2, q_len))
    if avg2 < avg1:
        return avg2, int(s2), int(e2)
    return avg1, int(s1), int(e1)


def _exclude(remaining: list[tuple[int, int]], exc_s: int, exc_e: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for seg_s, seg_e in remaining:
        if exc_e < seg_s or exc_s > seg_e:
            out.append((seg_s, seg_e))
            continue
        if seg_s < exc_s:
            out.append((seg_s, exc_s - 1))
        if exc_e < seg_e:
            out.append((exc_e + 1, seg_e))
    return out


def multi_subsequence_matches(
    query: np.ndarray,
    seq: np.ndarray,
    *,
    baseline: float = POSE33_BASELINE,
    max_matches: int = 30,
    exclusion: int = 5,
    offset: int = 0,
    stop_evt: Event | None = None,
) -> list[RepetitionMatch]:
    """贪心多次匹配：反复取剩余区间里最好的子序列（原查询与镜像取优），排除其邻域后再找；不按时间排序。"""
    raise_if_cancelled(stop_evt)
    if query.size == 0 or seq.size == 0:
        return []
    q_m = mirror_pose_features(query)
    q_len = int(query.shape[0])
    if q_len <= 0:
        return []

    remaining: list[tuple[int, int]] = [(0, int(seq.shape[0] - 1))]
    out: list[RepetitionMatch] = []
    ref_avg: float | None = None
    while remaining and len(out) < int(max_matches):
        raise_if_cancelled(stop_evt)
        best: tuple[float, int, int] | None = None  # (avg_cost, start, end)，seq 内下标
        for seg_s, seg_e in remaining:
            raise_if_cancelled(stop_evt)
            if seg_e - seg_s + 1 < max(8, q_len // 2):
                continue
            avg, s, e = _best_of_mirror(query, q_m, seq[seg_s : seg_e + 1], q_len, stop_evt)
            if best is None or avg < best[0]:
                best = (avg, s + int(seg_s), e + int(seg_s))
        if best is None:
            break
        avg, s, e = best
        if ref_avg is None:
            ref_avg = float(avg)
        elif float(avg) > float(ref_avg) * 2.5 + 0.5:  # 比最好的一次差太多就停，避免收进随机匹配
            break
        out.append(RepetitionMatch(int(offset + s), int(offset + e), float(avg),
                                   float(score_from_avg_cost(float(avg), baseline))))
        remaining = _exclude(remaining, max(0, int(s - exclusion)), min(int(seq.shape[0] - 1), int(e + exclusion)))

    raise_if_cancelled(stop_evt)
    return out


def repeated_matches(query: np.ndarray, window: np.ndarray, offset: int, *, baseline: float,
                     stop_evt: Event | None) -> list[RepetitionMatch]:
    """窗口内的多次匹配；一次也没匹配上时用整窗单次 DTW 兜底，保证有分数。"""
    exclusion = max(3, int(round(0.2 * float(query.shape[0]))))
    max_matches = int(min(30, max(1, round(float(window.shape[0]) / max(1.0, float(query.shape[0]))))))
    matches = multi_subsequence_matches(query, window, baseline=float(baseline), max_matches=max_matches,
                                        exclusion=exclusion, offset=offset, stop_evt=stop_evt)
    raise_if_cancelled(stop_evt)
    if not matches and window.size > 0:
        cost, s, e = subsequence_dtw(query, window, stop_evt=stop_evt)
        avg = float(avg_cost_of(cost, query.shape[0]))
        matches = [RepetitionMatch(int(offset + s), int(offset + e), float(avg),
                                   float(score_from_avg_cost(float(avg), float(baseline))))]
    return matches


def _swap_lr(name: str) -> str:
    if name.startswith("L_"):
        return "R_" + name[2:]
    if name.startswith("R_"):
        return "L_" + name[2:]
    return name


def _joint_stat(name: str, arr: np.ndarray) -> JointErrorStat:
    if arr.size == 0:
        return JointErrorStat(joint=str(name), mean_dist=None, p90_dist=None, max_dist=None, valid_frames=0)
    return JointErrorStat(joint=str(name), mean_dist=float(np.mean(arr)), p90_dist=float(np.percentile(arr, 90)),
                          max_dist=float(np.max(arr)), valid_frames=int(arr.size))


def joint_error_stats(query: np.ndarray, window: np.ndarray, offset: int, raw_slice: RawSlice, *,
                      layout: FeatureLayoutSpec = POSE33_V3,
                      stop_evt: Event | None = None) -> list[JointErrorStat] | None:
    """最佳对齐路径（原查询与镜像取优）上逐关节的距离统计，只计原始点有效的帧；路径为空时为 None。"""
    q_m = mirror_pose_features(query)
    cost1, s1, e1, path1 = subsequence_dtw_with_path(query, window, stop_evt=stop_evt)
    cost2, s2, e2, path2 = subsequence_dtw_with_path(q_m, window, stop_evt=stop_evt)
    mirrored = float(cost2) < float(cost1)
    q_eff, start_i, end_i, path = (q_m, int(s2), int(e2), path2) if mirrored else (query, int(s1), int(e1), path1)
    if not (start_i <= end_i and path):
        return None

    raw, raw_meta = raw_slice(int(offset), int(offset + int(window.shape[0]) - 1))
    raw_mask = valid_mask_from_raw(raw, raw_meta)
    path_arr = np.asarray(path, dtype=np.int64)
    if path_arr.size:
        keep = ((path_arr[:, 0] >= 0) & (path_arr[:, 0] < int(q_eff.shape[0]))
                & (path_arr[:, 1] >= 0) & (path_arr[:, 1] < int(window.shape[0])))
        path_arr = path_arr[keep]
    names = list(layout.joint_names)
    names = names if not mirrored else [_swap_lr(n) for n in names]
    if not path_arr.size:
        return [_joint_stat(name, np.zeros(0)) for name in names]
    q_path = q_eff[path_arr[:, 0]].astype(np.float32, copy=False)
    s_path = window[path_arr[:, 1]].astype(np.float32, copy=False)
    distances = np.linalg.norm(q_path - s_path, axis=2).astype(np.float32, copy=False)
    visible = np.asarray(raw_mask[path_arr[:, 1]][:, list(layout.source_indices)], dtype=bool)
    finite = np.isfinite(distances)
    stats = []
    for k in range(layout.num_joints):
        raise_if_cancelled(stop_evt)
        stats.append(_joint_stat(names[k], distances[visible[:, k] & finite[:, k], k]))
    return stats


def score_view(
    seq: np.ndarray,
    raw_slice: RawSlice,
    tpl_features: np.ndarray,
    tpl_meta: dict,
    seg: tuple[int, int] | None,
    *,
    fps: float,
    baseline: float,
    enable_error_analysis: bool,
    layout: FeatureLayoutSpec = POSE33_V3,
    stop_evt: Event | None = None,
) -> tuple[float, list[RepetitionMatch], list[JointErrorStat] | None]:
    """单视角得分：模板取代表周期作查询，学员取分段内的活跃窗口，多次匹配后取截尾均值；可选关节误差。"""
    raise_if_cancelled(stop_evt)
    query = select_representative_cycle(tpl_features, fps=float(tpl_meta.get("fps") or fps))
    raise_if_cancelled(stop_evt)
    if query.size == 0:
        return 0.0, [], None

    window, offset = active_window(seq, seg)
    matches = repeated_matches(query, window, offset, baseline=baseline, stop_evt=stop_evt)
    joint_stats: list[JointErrorStat] | None = None
    if enable_error_analysis and window.size > 0 and query.size > 0:
        raise_if_cancelled(stop_evt)
        joint_stats = joint_error_stats(query, window, offset, raw_slice, layout=layout, stop_evt=stop_evt)
    raise_if_cancelled(stop_evt)
    return trimmed_mean([m.score for m in matches]), matches, joint_stats
