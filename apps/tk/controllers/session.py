# -*- coding: utf-8 -*-
"""实时会话的界面设置：离线/并行线程数的默认值与钳制（P8 B6 起会话控制器也在此）。"""
from __future__ import annotations

import os


def clamp_workers(n: int) -> int:
    """将离线线程数钳制到闭区间 [1, os.cpu_count()]。

    - 小于 1 的输入钳制为 1
    - 大于主机 CPU 逻辑核心数的输入钳制为核心数
    - 当 ``os.cpu_count()`` 返回 None（无法确定核心数）时，上界回退为 1

    Validates: Requirements 7.2
    """
    cpu = os.cpu_count()
    upper = cpu if cpu and cpu >= 1 else 1
    try:
        value = int(n)
    except (TypeError, ValueError):
        value = 1
    if value < 1:
        return 1
    if value > upper:
        return upper
    return value


def default_workers() -> int:
    """主预览默认并行 worker 数：按 CPU 逻辑核数自适应，尽量榨满多核吞吐。

    单条 VIDEO 管线在多核机上只用到少数核（32 核机整机 ~10% 利用率）；多 worker 并行
    IMAGE 推理可近线性扩展。留 2 个核给 UI/采集/系统；上限 8——实测 6 worker 已达
    ~57fps，再多是拿内存（每 worker 一份模型）换不到额外 fps。想更激进可在「线程数」
    里手调（上限 16）。
    """
    cpu = os.cpu_count() or 4
    return max(2, min(8, cpu - 2))
