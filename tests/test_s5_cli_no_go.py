# -*- coding: utf-8 -*-
"""S5 CLI realtime no-go regression tests (YOLO migration Issue #25)."""

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
MAIN = _REPO_ROOT / "apps" / "main.py"


def test_issue25_report_records_cli_no_go_decision():
    text = REPORT.read_text(encoding="utf-8")
    for keyword in (
        "CLI 实时预览决议（Issue #25）",
        "当前仍不实现 `apps/main.py --backend yolo`",
        "torch.cuda.is_available() = True",
        "6/6",
        "仅 1/6 达标",
        "0/6 达标",
        "Hands 开",
        "Hands 关",
        "score_authorized=False",
        "另开新的实现子任务",
    ):
        assert keyword in text, f"#25 CLI no-go 决议缺少关键依据或边界：{keyword}"


def test_plan_docs_do_not_recommend_cli_yolo_runtime():
    for path in (PLAN, OPTIMIZED_PLAN):
        text = path.read_text(encoding="utf-8")
        for keyword in (
            "Issue #25",
            "docs/yolo_gpu_recheck_report.md",
            "当前不实现",
            "另开实现子任务",
        ):
            assert keyword in text, f"{path.name} 缺少 #25 CLI no-go 同步说明：{keyword}"

    combined = "\n".join(path.read_text(encoding="utf-8") for path in (PLAN, OPTIMIZED_PLAN))
    legacy_promises = (
        "| `apps/main.py`、`apps/app_ui.py` | CLI/UI 实时和离线入口 | 增加 backend/layout/rules 完整度配置和提示 |",
        "`apps/main.py` 增加 `--backend`、`--feature-layout`，只作为显式参数。",
        "`apps/main.py` 增加 `--backend`、`--feature-layout`、`--rules-backend` 或等价参数。",
        "`apps/main.py --backend yolo --source input.mp4 --no-show --out out.mp4`",
    )
    for legacy in legacy_promises:
        assert legacy not in combined, f"计划文档仍保留会误导实现 CLI YOLO 的旧表述：{legacy}"


def test_apps_main_keeps_mediapipe_only_cli_surface():
    source = MAIN.read_text(encoding="utf-8")
    tree = ast.parse(source)

    parser_args = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]

    assert "--backend" not in parser_args
    assert "--feature-layout" not in parser_args
    assert "--rules-backend" not in parser_args

    assert "YoloPoseAdapter" not in source
    assert "yolo_adapter" not in source
    assert "MediaPipePipeline" in source
