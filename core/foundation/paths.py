# -*- coding: utf-8 -*-
"""产物根目录解析：仓库根或冻结包的数据根，以及其下的 models、templates、outputs（原 core/paths.py，YOLO 迁移 S1 / Issue #6）。

为什么需要它
------------
此前 ``models`` / ``templates`` / ``outputs`` 三个顶层产物目录散落在各模块里，
各自用 ``Path(__file__).resolve().parent / "models"`` 之类的写法推导。由于
``__file__`` 锚点随所在包不同（``core/`` 解析成 ``core/models``、``apps/`` 解析成
``apps/models`` ……），同一类产物会指向不同位置，既容易踩坑又难以维护。

源码运行时三个根目录统一锚定到**仓库根**
（即 ``core/`` 包目录的父目录，``Path(__file__).resolve().parents[2]``）：
  - ``models_dir()``     → ``<repo_root>/models``
  - ``templates_dir()``  → ``<repo_root>/templates``
  - ``outputs_dir()``    → ``<repo_root>/outputs``

约束
----
- 只依赖标准库与 ``core.foundation`` 内的模块，不导入业务模块，
  避免循环依赖；任何入口（core/apps/batch/analysis）都能安全 import。
- 用户偏好的读写在 ``core.foundation.prefs``，偏好文件路径只由 ``prefs._prefs_path()`` 提供。
- ``templates_dir`` / ``outputs_dir`` 在返回前 ``mkdir(parents=True, exist_ok=True)``。
- 内置模型的冻结桌面包使用 LocalAppData/VisionSanda，首次使用复制内置模型和模板；
  不覆盖用户已有的非空文件。其他冻结入口保持原有 EXE 同级路径。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from core.foundation.atomic_io import atomic_target


def repo_root() -> Path:
    """返回 artifact 根目录。

    - 源码运行：``core/`` 包目录的父目录（仓库根）。
    - 内置模型的离线桌面包：用户 LocalAppData/VisionSanda，普通用户也可写入。
    - 其他 PyInstaller 入口（包括 sidecar）：保持可执行文件所在目录。
    """
    if getattr(sys, "frozen", False):
        if _bundled_root() is not None:
            root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "VisionSanda"
            root.mkdir(parents=True, exist_ok=True)
            return root
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _bundled_root() -> Path | None:
    root = getattr(sys, "_MEIPASS", None)
    if getattr(sys, "frozen", False) and root and (Path(root) / "models").is_dir():
        return Path(root)
    return None


def _seed_bundled(directory: str) -> Path:
    """首次启动复制内置资源；原子替换避免中断留下半个模型，不覆盖用户文件。"""
    d = repo_root() / directory
    d.mkdir(parents=True, exist_ok=True)
    bundled = _bundled_root()
    if bundled is not None:
        source_root = bundled / directory
        for source in source_root.rglob("*"):
            if not source.is_file():
                continue
            target = d / source.relative_to(source_root)
            if target.is_file() and target.stat().st_size > 0:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with atomic_target(target) as temporary:
                with source.open("rb") as reader, temporary.open("wb") as writer:
                    shutil.copyfileobj(reader, writer)
    return d


def models_dir() -> Path:
    """返回持久化模型目录，离线包首次使用时释放内置模型。"""
    return _seed_bundled("models")


def templates_dir() -> Path:
    """返回可写模板目录，离线包附带仓库公开的在线识别模板。"""
    return _seed_bundled("templates")


def outputs_dir() -> Path:
    """返回 ``<repo_root>/outputs`` 并确保其存在。"""
    d = repo_root() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d
