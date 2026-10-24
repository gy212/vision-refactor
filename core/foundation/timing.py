# -*- coding: utf-8 -*-
"""耗时样本的汇总 ``ms_summary``（毫秒的 p50/p95/max），录像线程的写出统计与录制性能测试共用。"""
from __future__ import annotations

import statistics
from collections.abc import Iterable


def ms_summary(values: Iterable[float]) -> dict[str, float | None]:
    """``{"p50", "p95", "max"}``，保留两位小数；没有样本时都是 None。p95 取排序后第 ``int(0.95·n)`` 个。"""
    ordered = sorted(values)
    if not ordered:
        return {"p50": None, "p95": None, "max": None}
    return {
        "p50": round(statistics.median(ordered), 2),
        "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
        "max": round(ordered[-1], 2),
    }


__all__ = ["ms_summary"]
