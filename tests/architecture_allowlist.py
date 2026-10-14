# -*- coding: utf-8 -*-
"""架构测试的现有违规登记（棘轮，只减不增）；规则见 AGENTS.md"模块职责与体量"与设计文档 §13.2。

每条 ``键: (登记值, 计划消除阶段)``。实际值必须与登记值完全相等：变大判新增违规，
变小时在同一提交里把登记值改成实际值；消除后删掉该条，并把 ``ENTRY_CEILING`` 改成新的条数。
搬迁文件时条目随文件改到新路径，登记值沿用原值，不新增条目、不调高登记值。P9 清零。
"""

LAYERING: dict[str, tuple[int, str]] = {}

CYCLES: dict[str, tuple[int, str]] = {}

PRIVATE: dict[str, tuple[int, str]] = {}

PRIMITIVES: dict[str, tuple[int, str]] = {
}

CLONES: dict[str, tuple[int, str]] = {}

MODULE_SIZE: dict[str, tuple[int, str]] = {
    'core/media/camera/pool.py': (1115, 'P9'),  # 原 apps/camera_warmup.py（1125），D-15：P9 按职责拆分
}

FUNCTION_SIZE: dict[str, tuple[int, str]] = {}

DEFENSIVE_GETATTR: dict[str, tuple[int, str]] = {
}

ENTRY_CEILING = 1
