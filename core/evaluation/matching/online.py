# -*- coding: utf-8 -*-
"""在线动作识别：滑动窗口 + 运动能量闸门 + 后台 DTW 模板库匹配（原 ``core.online_matcher``，设计文档 §7.5）。

预览实时流里，单帧几何规则（``classify_pose_actions``）无法区分直拳/摆拳这类「一段时间内的挥拳过程」。
本模块维护特征滑动窗口，用动作无关的运动能量闸门检测「发生了一次动作」，窗口闭合后对模板库逐个
子序列 DTW 比对，取最相似的模板作为识别结果。

- 触发器动作无关：用运动能量越基线（而非直拳专用的肘角 FSM）触发，否则摆拳会被系统性漏掉。
- DTW 不上主路径：匹配在后台任务里异步算，只在窗口闭合时派一次活；``push`` 永不阻塞。
- 迟滞用时间窗（ms）不用帧计数：实时丢帧、帧率波动时阈值不偏移。
- 在线单窗口匹配是识别不是评分，结果恒 ``score_authorized=False``。
- 模板经 ``evaluation.templates`` 的 ``ONLINE`` 契约读取（pose33_v3 / v3，与现场归一化逐字一致），
  分数经 ``scoring.cost`` 换算（baseline 来自布局注册表）。
"""
from __future__ import annotations

import queue
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.evaluation.matching.dtw import subsequence_dtw
from core.evaluation.scoring.cost import POSE33_BASELINE, avg_cost_of, score_from_avg_cost
from core.evaluation.templates import ONLINE, TemplateError, TemplateStore
from core.foundation.tasks import CancelToken, TaskGroup


@dataclass(frozen=True)
class MatcherConfig:
    """在线匹配阈值。默认值是起点，现场需对着摄像头标定（能量、拒识、迟滞跨机位会失准）。"""

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
    """``template_dir`` 下所有 ``*.npz`` 模板，文件 stem 作动作名；布局或形状不符直接抛 ``ValueError``。"""
    try:
        return [ActionTemplate(name=t.name, features=t.features) for t in TemplateStore(ONLINE).library(template_dir)]
    except TemplateError as exc:
        raise ValueError(str(exc)) from None


def match_window(window: np.ndarray, templates: list[ActionTemplate], cfg: MatcherConfig) -> MatchResult:
    """窗口序列 ``(W,22,2)`` 对每个模板做子序列 DTW，取分数最高者；低于拒识阈为无匹配。帧号由调用方填。"""
    best_name: str | None = None
    best_score = 0.0
    for tpl in templates:
        cost, _s, _e = subsequence_dtw(tpl.features, window)
        score = score_from_avg_cost(avg_cost_of(cost, tpl.features.shape[0]), POSE33_BASELINE)
        if score > best_score:
            best_score, best_name = score, tpl.name
    if best_name is None or best_score < cfg.reject_score_thr:
        return MatchResult(action=None, score=best_score, start_frame=0, end_frame=0)
    return MatchResult(action=best_name, score=best_score, start_frame=0, end_frame=0)


class OnlineActionMatcher:
    """喂入归一化单帧特征，后台 DTW 比对模板库，通过 ``on_result`` 回调产出识别结果。

    ``push`` 在预览线程调用（只入缓冲 + 闸门状态机，O(1)，不阻塞）；窗口闭合时把快照投递队列，
    后台任务跑 DTW 后调 ``on_result``。``close`` 收尾。
    """

    def __init__(self, templates: list[ActionTemplate], on_result, *, cfg: MatcherConfig | None = None) -> None:
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
        self._tasks = TaskGroup("online-matcher")
        self._worker = self._tasks.spawn("dtw", self._run_worker, owner="online_matcher", close_budget_s=1.0)

    def push(self, feature: np.ndarray, timestamp_ms: float, frame_index: int) -> None:
        """喂入一帧归一化特征 ``(22,2)``；缺失帧由调用方过滤，不入缓冲。"""
        if self._worker.token.cancelled or not self._templates:
            return
        cfg = self._cfg
        prev = self._buf[-1][0] if self._buf else None
        self._buf.append((feature, float(timestamp_ms), int(frame_index)))

        # 运动能量：与上一帧的 RMS 位移（与 foundation.signal.motion_energy 同款公式）。
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
                start_pos = max(0, len(self._buf) - 1 - cfg.pre_roll)
                self._arm_start_idx = self._buf[start_pos][2]
                self._arm_start_ts = self._buf[start_pos][1]
                self._low_since_ms = None
            return

        # 已武装：低能量持续 hold_ms 或超过 max_window_ms 时闭合。
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
        self._armed = False
        self._arm_start_idx = None
        self._arm_start_ts = None
        self._low_since_ms = None
        if start_idx is None or start_ts is None:
            return
        if end_ts - start_ts < cfg.min_window_ms:
            return  # 窗口过短（误触发），丢弃
        window = np.stack([f for (f, _ts, idx) in self._buf if idx >= start_idx], axis=0)
        end_idx = self._buf[-1][2]
        if window.shape[0] < 2:
            return
        self._jobs.put((window, int(start_idx), int(end_idx)))

    def _run_worker(self, token: CancelToken) -> None:
        while not token.cancelled:
            try:
                job = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                break
            window, start_idx, end_idx = job
            try:
                res = match_window(window, self._templates, self._cfg)
                self._on_result(MatchResult(action=res.action, score=res.score, start_frame=start_idx,
                                            end_frame=end_idx))
            except Exception:  # noqa: BLE001 - 识别失败跳过即可，不让后台任务退出
                continue

    def close(self) -> None:
        self._tasks.cancel()
        self._jobs.put(None)
        self._worker.join(timeout=1.0)
