# -*- coding: utf-8 -*-
"""ROI 占用去抖闸门（无 UI / 无 MediaPipe 依赖）。

仅消费布尔 present 序列，输出进场稳定 / 空场稳定事件。
force_finish 不经过本闸门。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PresenceEventKind = Literal["enter_stable", "empty_stable"]


@dataclass(frozen=True)
class PresenceEvent:
    kind: PresenceEventKind
    at: float


@dataclass
class PresenceGateConfig:
    enter_stable_s: float = 0.8
    empty_hold_s: float = 2.0
    min_record_s: float = 3.0
    presence_vis_thr: float = 0.3  # 上层用于判 present，本闸门不直接用


class PresenceGate:
    """wait_enter / recording 两阶段使用方式由调用方选择 mode。"""

    def __init__(self, config: PresenceGateConfig | None = None) -> None:
        self.config = config or PresenceGateConfig()
        self.reset()

    def reset(self) -> None:
        self._present_since: float | None = None
        self._empty_since: float | None = None
        self._record_started_at: float | None = None
        self._enter_fired = False
        self._empty_fired = False

    def begin_wait_enter(self) -> None:
        self._present_since = None
        self._empty_since = None
        self._enter_fired = False
        self._empty_fired = False
        self._record_started_at = None

    def begin_recording(self, now: float) -> None:
        self._present_since = None
        self._empty_since = None
        self._enter_fired = True  # 已进场
        self._empty_fired = False
        self._record_started_at = float(now)

    def update(
        self,
        present: bool,
        now: float,
        *,
        mode: Literal["wait_enter", "recording"],
    ) -> list[PresenceEvent]:
        now = float(now)
        events: list[PresenceEvent] = []
        cfg = self.config

        if mode == "wait_enter":
            if present:
                if self._present_since is None:
                    self._present_since = now
                elif (
                    not self._enter_fired
                    and (now - self._present_since) >= cfg.enter_stable_s
                ):
                    self._enter_fired = True
                    events.append(PresenceEvent("enter_stable", now))
            else:
                self._present_since = None
            return events

        # recording
        if self._record_started_at is None:
            self._record_started_at = now

        if present:
            self._empty_since = None
            return events

        # absent
        if self._empty_since is None:
            self._empty_since = now
        elapsed_empty = now - self._empty_since
        elapsed_record = now - self._record_started_at
        if (
            not self._empty_fired
            and elapsed_empty >= cfg.empty_hold_s
            and elapsed_record >= cfg.min_record_s
        ):
            self._empty_fired = True
            events.append(PresenceEvent("empty_stable", now))
        return events


def hip_midpoint_in_roi(
    landmarks,
    roi: tuple[float, float, float, float],
    *,
    left_hip_idx: int = 23,
    right_hip_idx: int = 24,
    vis_thr: float = 0.3,
) -> bool:
    """从 MediaPipe 风格 landmark 列表判定髋中点是否在归一化 ROI 内。

    landmarks 支持：
    - 带 .x/.y/.visibility 的对象序列
    - (N, 4) 数组 [x,y,z,vis]
    """
    if landmarks is None:
        return False
    x0, y0, x1, y1 = roi
    if x1 <= x0 or y1 <= y0:
        return False

    def _get(idx: int) -> tuple[float, float, float] | None:
        try:
            if hasattr(landmarks, "__getitem__") and not hasattr(landmarks[0], "x"):
                # array-like
                pt = landmarks[idx]
                if len(pt) >= 4:
                    return float(pt[0]), float(pt[1]), float(pt[3])
                return float(pt[0]), float(pt[1]), 1.0
            lm = landmarks[idx]
            vis = float(getattr(lm, "visibility", getattr(lm, "presence", 1.0)))
            return float(lm.x), float(lm.y), vis
        except Exception:
            return None

    left = _get(left_hip_idx)
    right = _get(right_hip_idx)
    if left is None or right is None:
        return False
    if left[2] < vis_thr or right[2] < vis_thr:
        return False
    mx = 0.5 * (left[0] + right[0])
    my = 0.5 * (left[1] + right[1])
    return x0 <= mx <= x1 and y0 <= my <= y1
