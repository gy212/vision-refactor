# -*- coding: utf-8 -*-
"""旧导入路径的门面：转发 ``core.foundation.paths`` 与 ``core.foundation.prefs`` 的公开函数（P9 删除）。

只做转发，不含逻辑。测试打补丁请改新模块（``core.foundation.paths`` / ``core.foundation.prefs``），
改这里的属性不影响新模块内部的调用。
"""
from core.foundation.paths import models_dir, outputs_dir, repo_root, templates_dir
from core.foundation.prefs import (
    load_camera_selection,
    load_record_dir,
    save_camera_selection,
    save_record_dir,
)

__all__ = [
    "load_camera_selection",
    "load_record_dir",
    "models_dir",
    "outputs_dir",
    "repo_root",
    "save_camera_selection",
    "save_record_dir",
    "templates_dir",
]
