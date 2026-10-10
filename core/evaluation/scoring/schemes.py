# -*- coding: utf-8 -*-
"""计分方案登记：任何入口按名字选用现存的三种正式计分方案（设计文档 §7.6，D-4）。

| 方案 | 算法 | 使用者 |
|---|---|---|
| ``EXAM_POOL`` | 正对正池、侧对侧池逐模板子序列 DTW；踢腿几何加权；指定动作分数压缩；正侧权重合并 | Tk 考试与普通双摄录后自动比对；``batch --scheme exam_pool`` 离线复算 |
| ``DUAL_STREAM`` | 正侧两路各自多次匹配，加规则扣分 | ``batch_dual_compare --paired``（研究用） |
| ``DUAL_TEMPLATE`` | 单视频按视角自动切分后做双模板比对，加规则扣分 | ``batch_dual_compare``（非 paired）、golden、离线剖析 |

D-4（2026-10-06）：考试以 ``EXAM_POOL`` 为准；三种并存，batch 的默认方案不变。
CompareWindow 与 ``match_template`` CLI 直接用 ``match_templates``，不属于这三种方案。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scheme:
    name: str
    summary: str


EXAM_POOL = Scheme("exam_pool", "正对正池、侧对侧池逐模板 DTW + 踢腿几何 + 分数压缩 + 正侧权重")
DUAL_STREAM = Scheme("dual_stream", "正侧两路各自多次匹配 + 规则扣分")
DUAL_TEMPLATE = Scheme("dual_template", "单视频按视角切分后双模板比对 + 规则扣分")
SCHEMES = {s.name: s for s in (EXAM_POOL, DUAL_STREAM, DUAL_TEMPLATE)}


def scheme(name: str) -> Scheme:
    try:
        return SCHEMES[name]
    except KeyError:
        raise ValueError(f"未知的计分方案 {name!r}，可选：{', '.join(SCHEMES)}") from None
