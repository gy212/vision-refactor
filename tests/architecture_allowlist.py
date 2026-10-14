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

FUNCTION_SIZE: dict[str, tuple[int, str]] = {
    'analysis/calibrate_body_core.py::run': (166, 'P9'),
    'analysis/offline_matching_profile.py::run_fixture_profile': (112, 'P9'),
    'batch/batch_dual_compare.py::main': (237, 'P9'),
    'batch/batch_export_skeleton.py::main': (120, 'P9'),
    'batch/batch_tech_eval.py::main': (190, 'P9'),
    'core/evaluation/techeval/cog.py::eval_cog_front': (101, 'P9'),  # 原 analysis/tech_eval.py，P3 迁入
    'core/evaluation/techeval/cog.py::eval_cog_side': (139, 'P9'),
    'core/evaluation/techeval/com.py::_compute_body_com_single': (176, 'P9'),
    'core/evaluation/techeval/com.py::eval_cog_com': (128, 'P9'),
    'core/evaluation/techeval/events.py::detect_retract_events_side': (103, 'P9'),
    'scripts/sim_exam_flow.py::scenario_5': (107, 'P9'),
}

DEFENSIVE_GETATTR: dict[str, tuple[int, str]] = {
}

ENTRY_CEILING = 12
