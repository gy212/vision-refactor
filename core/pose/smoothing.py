# -*- coding: utf-8 -*-
"""预览专用骨架平滑（One Euro 滤波）。

只作用于**实时预览绘制**，绝不回流 raw 序列 / 模板匹配 / 规则评分 / tech_eval / golden。
设计约束（与项目硬规则对齐）：

- 滤波器持有跨帧状态，因此只能在「按帧序重排后的单线程边界」调用，不能塞进 ``annotate()``，
  更不能塞进并行 worker（每个 worker 只看到乱序帧子集，状态会被污染）。
- 只平滑 ``visibility >= vis_thr`` 的**有效点**；低于阈值的点原样透传（draw 本就跳过低可见点），
  **不 hold、不补点、不改写 visibility**——缺失点不得伪造成有效点。
- 点从无效→重新有效时重置该点滤波器，避免用 gap 前的陈旧状态外推。

One Euro 比固定 EMA 更适合动作识别：静止时强平滑，快速动作（出拳/踢腿）时自动放松，
减少峰值被抹平与拖影。参数 ``beta`` 是控制快动作拖影的主旋钮。
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SmoothedLandmark:
    """平滑后关键点；属性与 MediaPipe ``NormalizedLandmark`` 对齐（``.x/.y/.z/.visibility``），
    供 ``_visibility`` / ``_classify_pose_actions`` / ``_draw_pose`` 透明消费。"""

    x: float
    y: float
    z: float
    visibility: float


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class _OneEuroScalar:
    """单标量通道的 One Euro 滤波器。"""

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "_x_hat", "_dx_hat", "_init")

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_hat = 0.0
        self._dx_hat = 0.0
        self._init = False

    def reset(self) -> None:
        self._x_hat = 0.0
        self._dx_hat = 0.0
        self._init = False

    def filter(self, x: float, dt: float) -> float:
        if not self._init:
            self._x_hat = x
            self._dx_hat = 0.0
            self._init = True
            return x
        dx = (x - self._x_hat) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_hat
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_hat
        self._x_hat = x_hat
        self._dx_hat = dx_hat
        return x_hat


class PreviewLandmarkSmoother:
    """对 pose 关键点做 One Euro 平滑，**仅供预览绘制**。

    用法（单线程重排出口）::

        smoother = PreviewLandmarkSmoother()
        smoothed = smoother.feed(pose_landmarks, timestamp_ms=ts)  # 喂 raw，拿平滑后的点
        # 用 smoothed 绘制；raw pose_landmarks 仍用于评分/序列，互不影响

    v1 只平滑 33 个 pose 点（索引稳定）。hands 不在此处理。
    """

    # ponytail: 初值，现场调 beta 控出拳拖影；min_cutoff 调静止抖动。单位：cutoff=Hz、dt=秒。
    # beta 需 ~1–2 量级才在出拳时真正抬高 cutoff 跟手；<0.1 退化成固定 min_cutoff 强平滑（~5 帧可见滞后）。
    def __init__(
        self,
        *,
        min_cutoff: float = 1.0,
        beta: float = 2.0,
        d_cutoff: float = 1.0,
        vis_thr: float = 0.3,  # ponytail: 对齐 _draw_pose 的 0.3 绘制门限——画什么就平滑什么
        num_points: int = 33,
    ) -> None:
        self.vis_thr = float(vis_thr)
        self._pts = [
            (
                _OneEuroScalar(min_cutoff, beta, d_cutoff),
                _OneEuroScalar(min_cutoff, beta, d_cutoff),
                _OneEuroScalar(min_cutoff, beta, d_cutoff),
            )
            for _ in range(int(num_points))
        ]
        self._prev_ms: float | None = None

    def reset(self) -> None:
        for fx, fy, fz in self._pts:
            fx.reset()
            fy.reset()
            fz.reset()
        self._prev_ms = None

    def feed(self, pose_landmarks, *, timestamp_ms: float | int | None):
        """喂入一帧 raw landmarks，返回平滑后的列表（与输入等长）。

        - ``pose_landmarks is None``（整体丢失）→ 重置全部状态并返回 None。
        - 有效点（``visibility >= vis_thr``）→ 平滑 x/y/z，返回 :class:`SmoothedLandmark`。
        - 低可见点 → 原样透传 raw 对象（不改 visibility、不补点），并重置该点滤波器状态。
        """
        if pose_landmarks is None:
            self.reset()
            return None

        prev = self._prev_ms
        self._prev_ms = float(timestamp_ms) if timestamp_ms is not None else None
        if prev is None or timestamp_ms is None:
            dt = 1.0 / 30.0
        else:
            dt = (float(timestamp_ms) - prev) / 1000.0
            if dt <= 0.0:
                dt = 1e-3  # ponytail: clamp 非单调/重复时间戳；One Euro 需 dt>0

        n = len(self._pts)
        out = []
        for i, lm in enumerate(pose_landmarks):
            vis = float(getattr(lm, "visibility", 1.0))
            if i < n and vis >= self.vis_thr:
                fx, fy, fz = self._pts[i]
                x = fx.filter(float(lm.x), dt)
                y = fy.filter(float(lm.y), dt)
                z = fz.filter(float(getattr(lm, "z", 0.0)), dt)
                out.append(SmoothedLandmark(x, y, z, vis))
            else:
                if i < n:
                    fx, fy, fz = self._pts[i]
                    fx.reset()
                    fy.reset()
                    fz.reset()
                out.append(lm)  # 原样透传：不伪造有效点
        return out
