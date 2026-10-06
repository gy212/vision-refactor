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

from core.pose.view_score import pose_view_score


class GapPolicy(str, Enum):
    """未检出姿态的帧怎么补（设计文档 §6.2 的表）。"""

    ZERO = "zero"                            # 全 0，自然被判为无效（tech_eval、练习分析）
    HOLD_LAST_FEATURE = "hold_last_feature"  # 特征：沿用上一帧特征，首个有效帧之前补 0（模板特征）
    HOLD_LAST_LEGACY = "hold_last_legacy"    # 原始序列：复制上一帧 33×4 含 visibility，补帧会被判为有效（D-2）
    HOLD_LAST_INVALID = "hold_last_invalid"  # 原始序列：复制坐标但 visibility 置 0，补帧判为无效（D-2 批准后用）
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
    source_wh: np.ndarray       # (T, 2) int：缩放前的帧尺寸（仅缩放时记录，否则为 0）
    meta: SourceMeta
    segments: tuple[SegmentWarmup, ...] = ()
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __len__(self) -> int:
        return int(self.landmarks.shape[0])

    @property
    def decoded_count(self) -> int:
        return int(np.count_nonzero(self.decoded))

    def prefix(self, end: int) -> "PoseSequence":
        """前 ``end + 1`` 帧（顺序跟踪的序列，供缓存按前缀复用）。"""
        stop = min(len(self), max(0, int(end) + 1))
        return replace(
            self,
            landmarks=self.landmarks[:stop], detected=self.detected[:stop], decoded=self.decoded[:stop],
            frame_index=self.frame_index[:stop], t_ms=self.t_ms[:stop], source_wh=self.source_wh[:stop],
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
        if gap not in (GapPolicy.HOLD_LAST_LEGACY, GapPolicy.HOLD_LAST_INVALID):
            raise ValueError(f"原始序列不支持补帧策略 {gap}")
        last = None
        for t in range(out.shape[0]):
            if self.detected[t]:
                last = out[t]
            elif last is not None:
                out[t] = last
                if gap is GapPolicy.HOLD_LAST_INVALID:
                    out[t, :, 3] = 0.0
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
        feats: list[np.ndarray] = []
        for t in range(len(self)):
            f = normalizer(self._frame(self.landmarks, self.detected, t))
            if f is None:
                f = feats[-1].copy() if feats else np.zeros(zero_shape, dtype=np.float32)
            feats.append(f)
        return np.stack(feats, axis=0)

    def _segmented_features(self, normalizer: Callable, zero_shape) -> np.ndarray:
        out = np.zeros((len(self), *zero_shape), dtype=np.float32)
        for seg in self.segments:
            last = None
            for w in range(seg.landmarks.shape[0]):
                f = normalizer(self._frame(seg.landmarks, seg.detected, w))
                if f is not None:
                    last = f
            for t in range(seg.seg_start, seg.seg_end + 1):
                if not self.decoded[t]:
                    break
                f = normalizer(self._frame(self.landmarks, self.detected, t))
                if f is None:
                    f = last.copy() if last is not None else np.zeros(zero_shape, dtype=np.float32)
                else:
                    last = f
                out[t] = f
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
