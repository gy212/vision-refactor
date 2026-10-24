# -*- coding: utf-8 -*-
"""内存查询（录制性能测试与本机概况用）：物理内存与提交量快照（``GlobalMemoryStatusEx``），
以及全系统换页率（PDH 计数器 ``\\Memory\\Pages/sec``：为解决硬缺页而读写磁盘的页数，持续偏高说明内存不够在换页）。

计数器用 ``PdhAddEnglishCounterW`` 按英文名添加，系统语言不影响；它是速率计数器，``PagingCounter.read`` 给出
自上一次读取（或 ``start``）以来的平均值。取不到时返回 None 或在结果里写明原因，不抛出。
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Any

from core.foundation.winapi import is_windows, library

PAGES_PER_SEC = "\\Memory\\Pages/sec"
_PDH_FMT_DOUBLE = 0x00000200
_GB = 2**30


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD), ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong), ("total_page", ctypes.c_ulonglong),
                ("avail_page", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong), ("avail_ext", ctypes.c_ulonglong)]


class _CounterValue(ctypes.Structure):
    _fields_ = [("status", wintypes.DWORD), ("value", ctypes.c_double)]   # PDH_FMT_COUNTERVALUE（double 按 8 字节对齐）


def memory_snapshot() -> dict[str, float] | None:
    """``total_gb``、``available_gb``（物理内存）、``load_percent``（系统报告的内存占用）、
    ``commit_available_gb``（还能提交的内存，含页面文件）；非 Windows 或调用失败返回 None。"""
    if not is_windows():
        return None
    status = _MemoryStatus()
    status.length = ctypes.sizeof(_MemoryStatus)
    if not library("kernel32").GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return {
        "total_gb": round(status.total_phys / _GB, 2),
        "available_gb": round(status.avail_phys / _GB, 2),
        "load_percent": int(status.load),
        "commit_available_gb": round(status.avail_page / _GB, 2),
    }


def memory_gb() -> tuple[float, float] | None:
    """物理内存 ``(总量, 可用)``，单位 GB（保留一位小数）；取不到返回 None。"""
    snapshot = memory_snapshot()
    if snapshot is None:
        return None
    return round(snapshot["total_gb"], 1), round(snapshot["available_gb"], 1)


class PagingCounter:
    """全系统换页率。``start`` 打开 PDH 查询并取第一个样本；``read`` 返回 ``{"pages_per_s", "error"}``
    （自上次取样以来的平均值）；``close`` 释放查询。打开或读取失败时 ``pages_per_s`` 为 None、``error`` 写明原因。"""

    def __init__(self) -> None:
        self._query = wintypes.HANDLE()
        self._counter = wintypes.HANDLE()
        self._error: str | None = None

    def start(self) -> "PagingCounter":
        if not is_windows():
            self._error = "不是 Windows，没有换页计数器"
            return self
        try:
            pdh = library("pdh")
            self._check(pdh.PdhOpenQueryW(None, None, ctypes.byref(self._query)), "PdhOpenQueryW")
            self._check(pdh.PdhAddEnglishCounterW(self._query, PAGES_PER_SEC, None, ctypes.byref(self._counter)),
                        "PdhAddEnglishCounterW")
            self._check(pdh.PdhCollectQueryData(self._query), "PdhCollectQueryData")
        except Exception as exc:
            self._error = str(exc)
            self.close()
        return self

    def read(self) -> dict[str, Any]:
        if self._error is not None or not self._query:
            return {"pages_per_s": None, "error": self._error or "换页计数器没有打开"}
        try:
            pdh = library("pdh")
            self._check(pdh.PdhCollectQueryData(self._query), "PdhCollectQueryData")
            value = _CounterValue()
            self._check(pdh.PdhGetFormattedCounterValue(self._counter, _PDH_FMT_DOUBLE, None, ctypes.byref(value)),
                        "PdhGetFormattedCounterValue")
            return {"pages_per_s": round(value.value, 1), "error": None}
        except Exception as exc:
            return {"pages_per_s": None, "error": str(exc)}

    def close(self) -> None:
        if self._query:
            try:
                library("pdh").PdhCloseQuery(self._query)
            finally:
                self._query = wintypes.HANDLE()

    @staticmethod
    def _check(status: int, name: str) -> None:
        if status != 0:
            raise OSError(f"{name} 返回 0x{status & 0xFFFFFFFF:08X}")


__all__ = ["PAGES_PER_SEC", "PagingCounter", "memory_gb", "memory_snapshot"]
