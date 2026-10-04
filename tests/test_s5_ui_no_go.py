# -*- coding: utf-8 -*-
"""S5 UI backend selector no-go regression tests (YOLO migration Issue #26)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

REPORT = _REPO_ROOT / "docs" / "yolo_gpu_recheck_report.md"
PLAN = _REPO_ROOT / "docs" / "yolo_migration_plan.md"
OPTIMIZED_PLAN = _REPO_ROOT / "docs" / "yolo_migration_plan_optimized.md"
APP_UI = _REPO_ROOT / "apps" / "app_ui.py"


def test_issue26_report_records_ui_no_go_decision():
    text = REPORT.read_text(encoding="utf-8")
    for keyword in (
        "UI 后端选择决议（Issue #26）",
        "当前仍不实现 `apps/app_ui.py` 的 YOLO 后端选择控件",
        "CUDA 可用，但 #23 实时阈值未达标",
        "6/6",
        "#25 仍关闭 / 不实现",
        "score_authorized=False",
        "另开 UI 实现子任务",
    ):
        assert keyword in text, f"#26 UI no-go 决议缺少关键依据或边界：{keyword}"


def test_plan_docs_do_not_recommend_ui_yolo_controls():
    for path in (PLAN, OPTIMIZED_PLAN):
        text = path.read_text(encoding="utf-8")
        for keyword in (
            "Issue #26",
            "docs/yolo_gpu_recheck_report.md",
            "当前不实现",
            "保持既有 MediaPipe UI",
        ):
            assert keyword in text, f"{path.name} 缺少 #26 UI no-go 同步说明：{keyword}"

    combined = "\n".join(path.read_text(encoding="utf-8") for path in (PLAN, OPTIMIZED_PLAN))
    legacy_promises = (
        "`apps/app_ui.py` 增加后端选择和规则完整度提示。",
        "`apps/app_ui.py` 增加后端选择、特征布局选择、规则完整度提示。",
        "UI 主预览、动作比对、直拳检测都能选择后端。",
        "CLI、UI、batch 都能选择后端并记录结果来源。",
    )
    for legacy in legacy_promises:
        assert legacy not in combined, f"计划文档仍保留会误导实现 UI YOLO 的旧表述：{legacy}"


def test_app_ui_keeps_mediapipe_only_ui_surface():
    source = APP_UI.read_text(encoding="utf-8")
    tree = ast.parse(source)

    import_names: list[str] = []
    string_literals: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            import_names.append(module)
            import_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            import_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            string_literals.append(node.value)

    joined_imports = "\n".join(import_names)
    joined_strings = "\n".join(string_literals)

    for forbidden in (
        "YoloPoseAdapter",
        "core.yolo_adapter",
        "batch.backend_options",
    ):
        assert forbidden not in joined_imports

    for forbidden in (
        "YOLO",
        "yolo",
        "body_core_v1",
        "feature_layout",
        "calibration_status",
        "score_authorized",
        "后端选择",
        "不完整评分",
    ):
        assert forbidden not in joined_strings
