# -*- coding: utf-8 -*-
"""``PoseSequence``：所有"视频→姿态"结果的唯一数据契约，推理一次、多种派生视图（设计文档 §6.2）。

原始 landmark 以 float64 ``(T, 33, 4)`` 保存（MediaPipe 输出本身是 float32 精度，float64 无损）；派生视图
（原始序列、特征、视角分数）按各旧实现的 dtype 和补帧规则输出，以保证与旧实现位级一致。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np

from core.pose.normalize import normalize_pose_xy_v3, normalize_pose_xy_v3_batch
from core.pose.view_score import pose_view_score


def _normalize_rows(
    rows: np.ndarray,
    detected: np.ndarray,
    normalizer: Callable,
    zero_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """批量归一化 ``rows`` 中 ``detected`` 为真的帧；v3 走向量化核，其余正常器走同一返回契约。"""
    shape = tuple(int(x) for x in zero_shape)
    if normalizer is normalize_pose_xy_v3 and shape == (22, 2):
        return normalize_pose_xy_v3_batch(rows, detected)
    n = int(rows.shape[0])
    out = np.zeros((n, *shape), dtype=np.float32)
    valid = np.zeros((n,), dtype=bool)
    for t in np.flatnonzero(detected):
        idx = int(t)
        f = normalizer(rows[idx])
        if f is not None:
            out[idx] = f
            valid[idx] = True
    return out, valid


def _forward_fill_features(
    feats: np.ndarray,
    valid: np.ndarray,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """按 ``HOLD_LAST_FEATURE`` 补齐无效帧：沿用上一有效帧，首个有效帧前填 ``initial``（缺省补 0）。"""
    m = int(feats.shape[0])
    if m == 0 or valid.all():
        return feats
    idx = np.where(valid, np.arange(m, dtype=np.intp), -1)
    np.maximum.accumulate(idx, out=idx)
    out = np.zeros_like(feats)
    has_prev = idx >= 0
    if has_prev.any():
        out[has_prev] = feats[idx[has_prev]]
    if initial is not None and not has_prev.all():
        out[~has_prev] = initial
    return out


class GapPolicy(str, Enum):
    """未检出姿态的帧怎么补（设计文档 §6.2 的表）。"""

    ZERO = "zero"                            # 全 0，自然被判为无效（tech_eval、练习分析）
    HOLD_LAST_FEATURE = "hold_last_feature"  # 特征：沿用上一帧特征，首个有效帧之前补 0（模板特征）
    HOLD_LAST_INVALID = "hold_last_invalid"  # 原始序列：复制坐标但 visibility 置 0，补帧判为无效（正式规则、踢腿质量，D-2）
    HOLD_LAST = "hold_last"                  # 视角分数：沿用上一值（模板特征提取）
    NAN = "nan"                              # 视角分数：记为 NaN（tech_eval）


class Coords(str, Enum):
    NORMALIZED = "normalized"              # MediaPipe 归一化坐标：x/宽、y/高
    ASPECT_CORRECTED = "aspect_corrected"  # x、z 乘宽高比，统一为图像高度单位（练习分析 v4）


@dataclass(frozen=True)
class SourceMeta:
    video: str | None
    fps: float                 # 正式路径帧率（容器值为 0 时取 30）
    fps_reported: float        # 容器原值
    frame_count: int           # 容器报告的帧数
    width: int                 # 容器报告的尺寸
    height: int
    engine: object             # EngineSpec
    profile: str
    tracking: str              # "sequential" | "segmented"
    validity_policy: str
    valid_conf_thr: float
    frames_scanned: int = 0    # 实际读到的帧数（含抽帧跳过的）
    stride: int = 1
    inference_size: tuple[int, int] = (0, 0)  # 最后一帧送入推理的 (width, height)


@dataclass(frozen=True)
class SegmentWarmup:
    """分段并行时一段的跟踪预热：``rows`` 是 ``[warm_start, seg_start)`` 的检出结果（不进入输出）。"""

    seg_start: int
    seg_end: int
    landmarks: np.ndarray   # (W, 33, 4) float64
    detected: np.ndarray    # (W,) bool


@dataclass(frozen=True, eq=False)
class PoseSequence:
    landmarks: np.ndarray       # (T, 33, 4) float64；未检出帧为 0
    detected: np.ndarray        # (T,) bool：本帧检出姿态
    decoded: np.ndarray         # (T,) bool：本帧确实读到（分段时读不到的尾帧为 False）
    frame_index: np.ndarray     # (T,) int：原视频帧号
    t_ms: np.ndarray            # (T,) int：送入推理的时间戳
    source_wh: np.ndarray       # (T, 2) int：缩放前的帧尺寸，顺序与分段提取都会记录
    meta: SourceMeta
    segments: tuple[SegmentWarmup, ...] = ()
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __len__(self) -> int:
        return int(self.landmarks.shape[0])

    @property
    def decoded_count(self) -> int:
        return int(np.count_nonzero(self.decoded))

    def prefix(self, end: int) -> "PoseSequence":
        """前 ``end + 1`` 帧（供缓存按前缀复用；调用方须先确认 ``start==0`` 且逐帧，见 ``extract._prefix_safe``）。

        这类读法下行号与帧号一一对应，截前缀即少扫了 ``len(self) - stop`` 帧，meta 的
        ``frames_scanned`` 要跟着减，不能照抄原序列扫到结尾的值。
        """
        stop = min(len(self), max(0, int(end) + 1))
        meta = replace(self.meta, frames_scanned=min(self.meta.frames_scanned, stop))
        return replace(
            self,
            landmarks=self.landmarks[:stop], detected=self.detected[:stop], decoded=self.decoded[:stop],
            frame_index=self.frame_index[:stop], t_ms=self.t_ms[:stop], source_wh=self.source_wh[:stop],
            meta=meta,
        )

    # ------------------------------------------------------------------ 原始序列

    def raw(self, gap: GapPolicy = GapPolicy.ZERO, coords: Coords = Coords.NORMALIZED) -> np.ndarray:
        """``(T, 33, 4)`` float32 原始序列，按 ``gap`` 补未检出帧。"""
        key = ("raw", gap, coords)
        if key not in self._cache:
            self._cache[key] = self._raw(gap, coords)
        return self._cache[key]

    def _raw(self, gap: GapPolicy, coords: Coords) -> np.ndarray:
        lm = self.landmarks
        if coords is Coords.ASPECT_CORRECTED:
            lm = lm.copy()
            w = self.source_wh[:, 0].astype(np.float64)[:, None]
            h = self.source_wh[:, 1].astype(np.float64)[:, None]
            lm[:, :, 0] = lm[:, :, 0] * w / h
            lm[:, :, 2] = lm[:, :, 2] * w / h
        out = lm.astype(np.float32)
        if gap is GapPolicy.ZERO:
            return out
        if gap is not GapPolicy.HOLD_LAST_INVALID:
            raise ValueError(f"原始序列不支持补帧策略 {gap}")
        last = None
        for t in range(out.shape[0]):
            if self.detected[t]:
                last = out[t]
            elif last is not None:
                out[t] = last
                out[t, :, 3] = 0.0  # 补帧不得当作有效检测（D-2）
        return out

    # ------------------------------------------------------------------ 特征

    def features(self, normalizer: Callable, zero_shape: tuple[int, ...]) -> np.ndarray:
        """逐帧归一化特征；归一化返回 None 时沿用上一帧特征，首个有效帧之前补 0（``HOLD_LAST_FEATURE``）。"""
        key = ("features", normalizer, tuple(zero_shape))
        if key not in self._cache:
            self._cache[key] = (self._segmented_features(normalizer, zero_shape) if self.segments
                                else self._sequential_features(normalizer, zero_shape))
        return self._cache[key]

    def _frame(self, rows: np.ndarray, detected: np.ndarray, t: int):
        return rows[t] if detected[t] else None

    def _sequential_features(self, normalizer: Callable, zero_shape) -> np.ndarray:
        shape = tuple(int(x) for x in zero_shape)
        if len(self) == 0:
            return np.zeros((0, *shape), dtype=np.float32)
        feats, valid = _normalize_rows(self.landmarks, self.detected, normalizer, shape)
        return _forward_fill_features(feats, valid)

    def _segmented_features(self, normalizer: Callable, zero_shape) -> np.ndarray:
        n = len(self)
        shape = tuple(int(x) for x in zero_shape)
        out = np.zeros((n, *shape), dtype=np.float32)
        if n == 0:
            return out
        active = np.zeros((n,), dtype=bool)
        spans: list[tuple[SegmentWarmup, int, int]] = []
        for seg in self.segments:
            s, e = int(seg.seg_start), min(int(seg.seg_end) + 1, n)
            if s >= e:
                continue
            undec = np.flatnonzero(~self.decoded[s:e])
            stop = s + int(undec[0]) if undec.size else e
            if stop > s:
                active[s:stop] = True
                spans.append((seg, s, stop))
        if not spans:
            return out
        feats, valid = _normalize_rows(self.landmarks, self.detected & active, normalizer, shape)
        for seg, s, stop in spans:
            last = None
            if seg.landmarks.shape[0] > 0:
                w_feats, w_valid = _normalize_rows(seg.landmarks, seg.detected, normalizer, shape)
                w_ok = np.flatnonzero(w_valid)
                if w_ok.size:
                    last = w_feats[int(w_ok[-1])]
            out[s:stop] = _forward_fill_features(feats[s:stop], valid[s:stop], initial=last)
        return out

    # ------------------------------------------------------------------ 视角分数

    def view_scores(self, gap: GapPolicy) -> np.ndarray:
        """逐帧视角分数 float32；``HOLD_LAST`` 沿用上一值（首个之前为 0），``NAN`` 记为 NaN。"""
        key = ("view", gap)
        if key not in self._cache:
            self._cache[key] = self._view_scores(gap)
        return self._cache[key]

    def _score(self, t: int):
        return pose_view_score(self._frame(self.landmarks, self.detected, t))

    def _view_scores(self, gap: GapPolicy) -> np.ndarray:
        if gap is GapPolicy.NAN:
            vs = []
            for t in range(len(self)):
                s = self._score(t) if self.detected[t] else None
                vs.append(float("nan") if (s is None or not np.isfinite(s)) else float(s))
            return np.asarray(vs, dtype=np.float32)
        if gap is not GapPolicy.HOLD_LAST:
            raise ValueError(f"视角分数不支持补帧策略 {gap}")
        if not self.segments:
            views: list[float] = []
            last_view = 0.0
            for t in range(len(self)):
                v = self._score(t)
                if v is None:
                    v = last_view
                else:
                    last_view = float(v)
                views.append(float(v))
            return np.array(views, dtype=np.float32)
        out = np.zeros((len(self),), dtype=np.float32)
        for seg in self.segments:
            last_seg: float | None = None
            for t in range(seg.seg_start, seg.seg_end + 1):
                if not self.decoded[t]:
                    break
                v = self._score(t)
                if v is None:
                    v = float(last_seg or 0.0)
                else:
                    last_seg = float(v)
                out[t] = float(v)
        return out
