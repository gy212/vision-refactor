"""把构建出的 EXE 与附带文件组织成交付目录（P0-3）。

附带文件来自 ``packaging/delivery/``：使用说明、``feedback_config.example.json``、
``第三方许可/``。由 ``app_ui_onefile.spec`` 在 EXE 生成后按文件路径加载调用。
"""
from __future__ import annotations

import shutil
from pathlib import Path

ATTACHMENTS_DIR = Path(__file__).resolve().parents[1] / "delivery"


def delivery_name(info: dict) -> str:
    """交付目录名：``vision_ui_<构建日期>_<commit 前 7 位>``。"""
    return f"vision_ui_{info['builtAt'][:10].replace('-', '')}_{info['commit'][:7]}"


def assemble(exe: Path, build_info: Path, info: dict, dist: Path, attachments: Path = ATTACHMENTS_DIR) -> Path:
    """在 ``dist`` 下新建交付目录，复制 EXE、``build_info.json`` 与全部附带文件。"""
    target = dist / delivery_name(info)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(attachments, target)
    shutil.copy2(exe, target / exe.name)
    shutil.copy2(build_info, target / build_info.name)
    return target
