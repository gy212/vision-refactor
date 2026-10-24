# -*- coding: utf-8 -*-
"""定位 ffmpeg 可执行文件的唯一实现（``ffmpeg_executable``），转码与录像封装共用；另给出子进程不弹窗口的启动标志。

查找顺序：冻结包里随包的 ffmpeg（``app_ui_onefile.spec`` 把 imageio-ffmpeg 的可执行文件放进 ``ffmpeg/``）→
PATH 上的 ``ffmpeg`` → 源码环境里装了 imageio-ffmpeg（开发依赖）时它自带的可执行文件 → 都没有时返回 ``"ffmpeg"``，
由启动子进程时报找不到。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _bundled() -> str | None:
    if not getattr(sys, "frozen", False):
        return None
    found = sorted((Path(getattr(sys, "_MEIPASS", "")) / "ffmpeg").glob("*.exe"))
    return str(found[0]) if found else None


def _imageio_ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg  # 开发依赖；冻结包里不导入
    except ImportError:
        return None
    try:
        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except (OSError, RuntimeError):
        return None


@lru_cache(maxsize=1)
def ffmpeg_executable() -> str:
    """ffmpeg 的路径（或 ``"ffmpeg"``，交给系统按 PATH 查找）；进程内只查一次（首次查找含导入 imageio-ffmpeg，约 0.1 s）。"""
    return _bundled() or shutil.which("ffmpeg") or _imageio_ffmpeg() or "ffmpeg"


__all__ = ["NO_WINDOW", "ffmpeg_executable"]
