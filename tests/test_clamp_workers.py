# -*- coding: utf-8 -*-
"""离线线程数钳制函数的属性测试。

设计来源：``.kiro/specs/ui-layout-redesign/design.md``（Correctness Property 7）。

被测函数：``apps.app_ui.clamp_workers(n) -> int``，将任意整数输入钳制到闭区间
``[1, os.cpu_count()]``（当 ``os.cpu_count()`` 返回 None 时上界回退为 1）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.app_ui import clamp_workers  # noqa: E402


def _expected(n: int) -> int:
    """参考实现：独立复刻期望的钳制语义。"""
    cpu = os.cpu_count()
    upper = cpu if cpu and cpu >= 1 else 1
    if n < 1:
        return 1
    if n > upper:
        return upper
    return n


# Feature: ui-layout-redesign, Property 7: 离线线程数钳制
# Validates: Requirements 7.2
@settings(max_examples=200)
@given(st.integers(min_value=-10_000, max_value=10_000))
def test_clamp_workers_within_bounds_and_matches_expected(n: int) -> None:
    cpu = os.cpu_count()
    upper = cpu if cpu and cpu >= 1 else 1

    result = clamp_workers(n)

    # 输出始终落在闭区间 [1, upper]
    assert 1 <= result <= upper
    # 输出等于期望的钳制值
    assert result == _expected(n)
