# -*- coding: utf-8 -*-
"""artifact root 统一回归（YOLO 迁移 S1 / Issue #6 Part A）。

为什么需要它
------------
Issue #6 把散落各处的 ``Path(__file__).resolve().parent / "models|templates|outputs"``
统一收口到 ``core.paths``，三个顶层产物根一律锚定仓库根（``core/`` 的父目录）。
本测试验证：
  1. ``models_dir/templates_dir/outputs_dir`` 全部位于同一 ``repo_root()`` 下，
     且等于 ``<repo_root>/models|templates|outputs``；
  2. ``repo_root()`` 等于 ``core/__file__`` 的父父目录（即真正的仓库根）；
  3. 被改造的生产文件源码里不再残留 ``parent / "models|templates|outputs"`` 字面量
     （读源码做子串断言，避免回退到分散解析）。

验收命令
--------
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_artifact_roots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import core  # noqa: E402
from core import paths  # noqa: E402


# --------------------------------------------------------------------------- #
# 1) 三个根都在同一 repo_root 下，且路径精确相等
# --------------------------------------------------------------------------- #
def test_roots_under_same_repo_root():
    root = paths.repo_root()
    assert paths.models_dir() == root / "models"
    assert paths.templates_dir() == root / "templates"
    assert paths.outputs_dir() == root / "outputs"
    # 三者共享同一父目录（repo_root）
    assert paths.models_dir().parent == root
    assert paths.templates_dir().parent == root
    assert paths.outputs_dir().parent == root


def test_repo_root_is_parent_of_core_package():
    # repo_root 必须等于 core 包目录的父目录（即真正的仓库根）。
    core_file = Path(core.__file__).resolve()
    assert paths.repo_root() == core_file.parents[1]


def test_dirs_exist_after_resolve():
    # 解析后目录应已存在（mkdir(parents=True, exist_ok=True)）。
    assert paths.models_dir().is_dir()
    assert paths.templates_dir().is_dir()
    assert paths.outputs_dir().is_dir()


# --------------------------------------------------------------------------- #
# 2) 生产文件不再残留分散的 parent / "models|templates|outputs" 字面量
# --------------------------------------------------------------------------- #
# 仅检查本次改造涉及的生产文件，避免误伤 core/paths.py 自身的说明性 docstring。
_EDITED_FILES = (
    "core/action_compare.py",
    "core/rule_scoring.py",
    "analysis/tech_eval.py",
    "apps/main.py",
    "apps/app_ui.py",
    "apps/make_template.py",
    "apps/match_template.py",
    "batch/batch_export_skeleton.py",
    "batch/batch_tech_eval.py",
    "batch/batch_dual_compare.py",
)

_FORBIDDEN = (
    'parent / "models"',
    'parent / "templates"',
    'parent / "outputs"',
)


def test_no_scattered_parent_literals_in_edited_files():
    root = paths.repo_root()
    offenders: list[str] = []
    for rel in _EDITED_FILES:
        text = (root / rel).read_text(encoding="utf-8")
        for needle in _FORBIDDEN:
            if needle in text:
                offenders.append(f"{rel}: 残留 {needle!r}")
    assert not offenders, "发现未统一的路径解析：\n" + "\n".join(offenders)


# --------------------------------------------------------------------------- #
# 3) 代表性入口模块可正常 import 且引用统一的 core.paths
# --------------------------------------------------------------------------- #
def test_representative_modules_use_core_paths():
    import core.action_compare as ac
    import core.rule_scoring as rs
    import analysis.tech_eval as te
    import apps.make_template as mt
    import apps.match_template as mat
    import batch.batch_export_skeleton as bes

    # 这些模块都绑定了 core.paths 暴露的解析函数（models_dir/templates_dir/outputs_dir）。
    assert ac.models_dir is paths.models_dir
    assert ac.templates_dir is paths.templates_dir
    assert rs.models_dir is paths.models_dir
    assert te.models_dir is paths.models_dir
    assert mt.models_dir is paths.models_dir
    assert mt.templates_dir is paths.templates_dir
    assert mat.models_dir is paths.models_dir
    assert mat.templates_dir is paths.templates_dir
    assert bes.models_dir is paths.models_dir
    assert bes.outputs_dir is paths.outputs_dir
