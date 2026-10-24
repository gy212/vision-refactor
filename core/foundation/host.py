# -*- coding: utf-8 -*-
"""本机环境查询：用户桌面目录、CPU 概况、系统 CPU 累计时间与写进诊断报告的本机概况（内存见 ``memory``、电源见 ``power``）；
只用标准库（Windows 上经 ctypes 调系统 API），取不到时返回 None。"""
from __future__ import annotations

import ctypes
import os
import platform
import uuid
from ctypes import wintypes
from pathlib import Path

from core.foundation.memory import memory_gb, memory_snapshot
from core.foundation.power import power_status
from core.foundation.winapi import guid_from_uuid, is_windows

_FOLDERID_DESKTOP = uuid.UUID("{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}")


def desktop_dir() -> Path | None:
    """当前用户的桌面目录。Windows 上按系统登记的位置取（桌面被 OneDrive 或组策略重定向时也对），
    取不到时回退 ``%USERPROFILE%\\Desktop``；目录不存在时返回 None。"""
    if is_windows():
        try:
            guid = guid_from_uuid(_FOLDERID_DESKTOP)
            out = ctypes.c_wchar_p()
            shell32, ole32 = ctypes.windll.shell32, ctypes.windll.ole32
            if shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(out)) == 0:
                try:
                    path = Path(out.value)
                finally:
                    ole32.CoTaskMemFree(out)
                if path.is_dir():
                    return path
        except Exception:
            pass
    fallback = Path(os.environ.get("USERPROFILE") or Path.home()) / "Desktop"
    return fallback if fallback.is_dir() else None


def cpu_name() -> str:
    """处理器型号；Windows 上取注册表登记的名称，取不到时用 ``platform.processor()``。"""
    if is_windows():
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")


def system_cpu_times() -> tuple[int, int] | None:
    """全系统 CPU 累计时间 ``(忙, 总)``（100 ns 单位，含所有进程）；两次取值之差算占用率。非 Windows 返回 None。"""
    if not is_windows():
        return None
    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return None

    def value(t: wintypes.FILETIME) -> int:
        return (t.dwHighDateTime << 32) | t.dwLowDateTime

    total = value(kernel) + value(user)          # kernel 时间已包含 idle
    return total - value(idle), total


def cpu_percent(start: tuple[int, int] | None, end: tuple[int, int] | None) -> float | None:
    """两次 ``system_cpu_times`` 之间的全系统 CPU 占用率（%）。"""
    if start is None or end is None or end[1] <= start[1]:
        return None
    return round(100.0 * (end[0] - start[0]) / (end[1] - start[1]), 1)


def host_summary() -> dict:
    """写进诊断报告的本机概况；``memory`` 是物理内存与提交量快照，``power`` 含电源计划与电源模式。"""
    memory = memory_gb()
    return {
        "os": platform.platform(),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "memory_total_gb": None if memory is None else memory[0],
        "memory_available_gb": None if memory is None else memory[1],
        "memory": memory_snapshot(),
        "power": power_status(),
        "python": platform.python_version(),
    }


__all__ = ["cpu_name", "cpu_percent", "desktop_dir", "host_summary", "memory_gb", "power_status", "system_cpu_times"]
