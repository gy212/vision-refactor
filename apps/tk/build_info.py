# -*- coding: utf-8 -*-
"""读取构建时打进包内的 ``build_info.json``（commit、构建时间、工作区是否干净），供离线自检与录制性能测试写进报告。"""
from __future__ import annotations

import json
from pathlib import Path


def package_root() -> Path:
    """源码模式为仓库根，冻结模式为 PyInstaller 解包目录；与模块的 ``__file__`` 定位一致。"""
    return Path(__file__).resolve().parents[2]


def load_build_info(root: Path | None = None) -> dict | None:
    """读取 ``<root>/build_info.json``（默认包根）；源码运行没有该文件时返回 None。"""
    path = (package_root() if root is None else Path(root)) / "build_info.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
