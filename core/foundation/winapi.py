# -*- coding: utf-8 -*-
"""Windows API 调用的共用基元（ctypes）：GUID 结构与 ``uuid.UUID`` 互转、带类型声明的系统库句柄、是否 Windows。

各查询模块（``host``、``power``、``memory``、``volume``）共用；系统库用独立的 ``WinDLL`` 实例，
在这里声明的 argtypes/restype 不影响 ``ctypes.windll`` 上的同名函数。
"""
from __future__ import annotations

import ctypes
import sys
import uuid
from ctypes import wintypes
from functools import lru_cache


class GUID(ctypes.Structure):
    _fields_ = [("data1", wintypes.DWORD), ("data2", wintypes.WORD), ("data3", wintypes.WORD),
                ("data4", ctypes.c_ubyte * 8)]


def is_windows() -> bool:
    return sys.platform == "win32"


def guid_from_uuid(value: uuid.UUID) -> GUID:
    fields = value.fields
    return GUID(fields[0], fields[1], fields[2], (ctypes.c_ubyte * 8)(*value.bytes[8:]))


def uuid_from_guid(guid: GUID) -> uuid.UUID:
    return uuid.UUID(fields=(guid.data1, guid.data2, guid.data3, guid.data4[0], guid.data4[1],
                             int.from_bytes(bytes(guid.data4[2:]), "big")))


@lru_cache(maxsize=None)
def library(name: str) -> ctypes.WinDLL:
    """系统库 ``name``（如 ``kernel32``、``powrprof``、``pdh``）的独立实例，``use_last_error`` 打开。"""
    return ctypes.WinDLL(name, use_last_error=True)


def last_error_text(prefix: str) -> str:
    """``prefix``（失败的调用名）加上 ``GetLastError`` 的系统说明。"""
    code = ctypes.get_last_error()
    return f"{prefix} 失败（{code}：{ctypes.FormatError(code).strip()}）"


__all__ = ["GUID", "guid_from_uuid", "is_windows", "last_error_text", "library", "uuid_from_guid"]
