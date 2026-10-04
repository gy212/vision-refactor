# -*- coding: utf-8 -*-
"""在线动作识别：滑动窗口 + 运动能量闸门 + 后台 DTW 模板库匹配。

预览实时流里，单帧几何规则（``_classify_pose_actions``）无法区分直拳/摆拳这类
「一段时间内的挥拳过程」。本模块做时序识别：维护一个特征滑动窗口，用**动作无关**
的运动能量闸门检测「发生了一次动作」，窗口闭合后对模板库逐个子序列 DTW 比对，取
最相似的模板作为识别结果。

设计约束（与项目硬规则对齐）：

- **触发器动作无关**：用运动能量越基线（而非直拳专用的肘角 FSM）触发，否则摆拳
  （肘不完全伸直、腕横扫）会被系统性漏掉。「是哪个动作」完全交给模板库 DTW 决定。
- **DTW 不上主路径**：预览关键路径已被推理吃满，匹配在后台线程异步算，只在窗口
  闭合那一刻派一次活；``push`` 永不阻塞。
- **hysteresis 用时间窗（ms）不是帧计数**：实时丢帧/帧率波动下阈值才不偏移。
- 在线单窗口匹配是**识别**不是离线双模板严谨评分，结果恒 ``score_authorized=False``。
- 复用现成纯函数：``subsequence_dtw`` / ``motion_energy`` 同款公式 / ``normalize_pose_xy_v3``
  布局（pose33_v3）。模板加载断言 layout/normalizer 与在线归一化逐字一致。
"""
from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .feature_layout import POSE33_V3
from .pose_features import subsequence_dtw

# pose33_v3 已标定 baseline：score=1.0@avg_cost=0，score=0.5@avg_cost=baseline。
# 2026-07-11 校准：2.0 → 3.0（同人自复现 avg_cost≈0.333 锚到 ~0.90 分）。
_POSE33_V3_BASELINE = 3.0


@dataclass(frozen=True)
class MatcherConfig:
    """在线匹配阈值。默认值是起点，**现场需对着摄像头标定**（能量/拒识/迟滞跨机位会失准）。"""

    # ponytail: 这些阈值是物理动作的标定旋钮，不是可推导常量；现场拧。
    arm_thr: float = 0.06          # 运动能量越过 idle 基线 + 此增量 → 武装窗口
    disarm_thr: float = 0.03       # 能量回落到 idle 基线 + 此增量以下 → 计入闭合（< arm 形成迟滞）
    hold_ms: float = 150.0         # 低能量需持续这么久才闭合窗口（去抖）
    min_window_ms: float = 200.0   # 窗口太短（误触发）则丢弃，不匹配
    max_window_ms: float = 1800.0  # 窗口上限，超时强制闭合（覆盖最长单动作）
    pre_roll: int = 5              # 武装前并入的帧数（动作起手常在能量越线前）
    reject_score_thr: float = 0.40 # 最佳模板分数低于此 → 判「无匹配」
    baseline_ema: float = 0.05     # idle 基线 EMA 系数（越小越平滑）
    buffer_frames: int = 90        # 环形缓冲容量（~1.5s @ 60fps，须 >= max_window 对应帧数）


@dataclass(frozen=True)
class ActionTemplate:
    name: str
    features: np.ndarray  # (M, 22, 2) pose33_v3


@dataclass(frozen=True)
class MatchResult:
    """一次窗口的识别结果。``action`` 为 None 表示窗口闭合但无模板过阈（无匹配）。"""

    action: str | None
    score: float
    start_frame: int
    end_frame: int
    score_authorized: bool = False  # 在线识别恒不授权对外评分


def load_template_library(template_dir: str | Path) -> list[ActionTemplate]:
    """加载 ``template_dir`` 下所有 ``*.npz`` 模板，文件 stem 作动作名。

    断言每个模板是 pose33_v3 / normalizer v3——与在线 ``normalize_pose_xy_v3`` 逐字一致，
    否则 DTW 分数无意义。布局不符直接抛错，不静默接受。
    """
    template_dir = Path(template_dir)
    templates: list[ActionTemplate] = []
    if not template_dir.is_dir():
        return templates
    for npz_path in sorted(template_dir.glob("*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        feats = np.asarray(data["features"], dtype=np.float32)
        raw_meta = dict(data["meta"].item() or {}) if "meta" in data else {}
        layout = str(raw_meta.get("feature_layout", "")) or "?"
        normalizer = str(raw_meta.get("normalizer_version", "")) or "?"
        if layout != POSE33_V3.name or normalizer != "v3":
            raise ValueError(
                f"在线模板 {npz_path.name} 布局/归一化不符："
                f"feature_layout={layout} normalizer_version={normalizer}，"
                f"在线识别要求 {POSE33_V3.name}/v3（与现场归一化逐字一致）"
            )
        if feats.ndim != 3 or feats.shape[1:] != POSE33_V3.shape:
            raise ValueError(f"在线模板 {npz_path.name} 特征形状 {feats.shape} 非 (M,{POSE33_V3.shape})")
        templates.append(ActionTemplate(name=npz_path.stem, features=feats))
    return templates


def match_window(window: np.ndarray, templates: list[ActionTemplate], cfg: MatcherConfig) -> MatchResult:
    """对一个窗口序列 ``(W,22,2)`` 逐模板子序列 DTW，取最小 avg_cost。低于拒识阈则无匹配。

    纯函数（不碰缓冲/线程状态），便于直接单测。``start_frame/end_frame`` 由调用方填。
    """
    best_name: str | None = None
    best_score = 0.0
    for tpl in templates:
        cost, _s, _e = subsequence_dtw(tpl.features, window)
        avg_cost = cost / max(1, int(tpl.features.shape[0]))
        score = float(_POSE33_V3_BASELINE / (_POSE33_V3_BASELINE + avg_cost))
        if score > best_score:
            best_score, best_name = score, tpl.name
    if best_name is None or best_score < cfg.reject_score_thr:
        return MatchResult(action=None, score=best_score, start_frame=0, end_frame=0)
    return MatchResult(action=best_name, score=best_score, start_frame=0, end_frame=0)


class OnlineActionMatcher:
    """喂入归一化单帧特征，后台 DTW 比对模板库，通过 ``on_result`` 回调产出识别结果。

    线程模型：``push`` 在预览主线程调用（仅入缓冲 + 闸门状态机，O(1)，永不阻塞）；
    窗口闭合时把快照投递队列，后台 worker 跑 DTW 后调 ``on_result``。``close`` 收尾。
    """

    def __init__(
        self,
        templates: list[ActionTemplate],
        on_result,
        *,
        cfg: MatcherConfig | None = None,
    ) -> None:
        self._templates = templates
        self._on_result = on_result
        self._cfg = cfg or MatcherConfig()
        self._buf: deque[tuple[np.ndarray, float, int]] = deque(maxlen=self._cfg.buffer_frames)
        self._baseline: float | None = None
        self._armed = False
        self._arm_start_idx: int | None = None
        self._arm_start_ts: float | None = None
        self._low_since_ms: float | None = None
        self._jobs: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run_worker, name="online-matcher-dtw", daemon=True)
        self._worker.start()

    def push(self, feature: np.ndarray, timestamp_ms: float, frame_index: int) -> None:
        """喂入一帧归一化特征 ``(22,2)``。``feature`` 为 None 由调用方过滤（缺失帧不入缓冲）。"""
        if self._stop.is_set() or not self._templates:
            return
        cfg = self._cfg
        prev = self._buf[-1][0] if self._buf else None
        self._buf.append((feature, float(timestamp_ms), int(frame_index)))

        # 运动能量：与上一帧的 RMS 位移（与 pose_features.motion_energy 同款公式）。
        if prev is None:
            return
        diff = feature.reshape(-1) - prev.reshape(-1)
        energy = float(np.sqrt(np.mean(diff * diff)))

        if self._baseline is None:
            self._baseline = energy
        elif not self._armed:
            # 仅在未武装（idle）时更新基线，避免动作能量污染 idle 基线。
            self._baseline += cfg.baseline_ema * (energy - self._baseline)

        base = self._baseline
        if not self._armed:
            if energy >= base + cfg.arm_thr:
                self._armed = True
                # pre_roll：把武装前若干帧并入窗口起点（动作起手常在越线前）。
                start_pos = max(0, len(self._buf) - 1 - cfg.pre_roll)
                self._arm_start_idx = self._buf[start_pos][2]
                self._arm_start_ts = self._buf[start_pos][1]
                self._low_since_ms = None
            return

        # 已武装：检测闭合条件（低能量持续 hold_ms 或超 max_window_ms）。
        if energy <= base + cfg.disarm_thr:
            if self._low_since_ms is None:
                self._low_since_ms = timestamp_ms
            elif timestamp_ms - self._low_since_ms >= cfg.hold_ms:
                self._close_window(timestamp_ms)
                return
        else:
            self._low_since_ms = None

        if self._arm_start_ts is not None and timestamp_ms - self._arm_start_ts >= cfg.max_window_ms:
            self._close_window(timestamp_ms)

    def _close_window(self, end_ts: float) -> None:
        cfg = self._cfg
        start_idx = self._arm_start_idx
        start_ts = self._arm_start_ts
        # 重置闸门状态（无论是否派活）。
        self._armed = False
        self._arm_start_idx = None
        self._arm_start_ts = None
        self._low_since_ms = None
        if start_idx is None or start_ts is None:
            return
        if end_ts - start_ts < cfg.min_window_ms:
            return  # 窗口过短（误触发），丢弃。
        window = np.stack([f for (f, _ts, idx) in self._buf if idx >= start_idx], axis=0)
        end_idx = self._buf[-1][2]
        if window.shape[0] < 2:
            return
        self._jobs.put((window, int(start_idx), int(end_idx)))

    def _run_worker(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                break
            window, start_idx, end_idx = job
            try:
                res = match_window(window, self._templates, self._cfg)
                res = MatchResult(
                    action=res.action,
                    score=res.score,
                    start_frame=start_idx,
                    end_frame=end_idx,
                )
                self._on_result(res)
            except Exception:  # noqa: BLE001 - 后台线程不得把异常吞进 join；识别失败跳过即可。
                continue

    def close(self) -> None:
        self._stop.set()
        self._jobs.put(None)
        self._worker.join(timeout=1.0)
