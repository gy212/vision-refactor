"""Tk 桌面入口壳：保留 ``apps/app_ui.py`` 的启动命令，实现在 ``apps.tk.app``（组合根、控制器与视图）。"""
from __future__ import annotations

from apps.tk.app import DesktopApp, main

__all__ = ["DesktopApp", "main"]

if __name__ == "__main__":
    main()
