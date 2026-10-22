# -*- coding: utf-8 -*-
"""本机环境查询：用户桌面目录、CPU 与内存概况、系统 CPU 累计时间；只用标准库（Windows 上经 ctypes 调系统 API），取不到时返回 None。"""
from __future__ import annotations

import ctypes
import os
import platform
import sys
import uuid
from ctypes import wintypes
from pathlib import Path

_FOLDERID_DESKTOP = uuid.UUID("{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}")


class _GUID(ctypes.Structure):
    _fields_ = [("data1", wintypes.DWORD), ("data2", wintypes.WORD), ("data3", wintypes.WORD),
                ("data4", ctypes.c_ubyte * 8)]


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD), ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong), ("total_page", ctypes.c_ulonglong),
                ("avail_page", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong), ("avail_ext", ctypes.c_ulonglong)]


def _windows() -> bool:
    return sys.platform == "win32"


def desktop_dir() -> Path | None:
    """当前用户的桌面目录。Windows 上按系统登记的位置取（桌面被 OneDrive 或组策略重定向时也对），
    取不到时回退 ``%USERPROFILE%\\Desktop``；目录不存在时返回 None。"""
    if _windows():
        try:
            guid = _GUID(_FOLDERID_DESKTOP.fields[0], _FOLDERID_DESKTOP.fields[1], _FOLDERID_DESKTOP.fields[2],
                         (ctypes.c_ubyte * 8)(*_FOLDERID_DESKTOP.bytes[8:]))
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
    if _windows():
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")


def memory_gb() -> tuple[float, float] | None:
    """物理内存 ``(总量, 可用)``，单位 GB；非 Windows 返回 None。"""
    if not _windows():
        return None
    status = _MemoryStatus()
    status.length = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return round(status.total_phys / 2**30, 1), round(status.avail_phys / 2**30, 1)


def system_cpu_times() -> tuple[int, int] | None:
    """全系统 CPU 累计时间 ``(忙, 总)``（100 ns 单位，含所有进程）；两次取值之差算占用率。非 Windows 返回 None。"""
    if not _windows():
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
    """写进诊断报告的本机概况。"""
    memory = memory_gb()
    return {
        "os": platform.platform(),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "memory_total_gb": None if memory is None else memory[0],
        "memory_available_gb": None if memory is None else memory[1],
        "python": platform.python_version(),
    }


__all__ = ["cpu_name", "cpu_percent", "desktop_dir", "host_summary", "memory_gb", "system_cpu_times"]
