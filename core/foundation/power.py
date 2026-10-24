# -*- coding: utf-8 -*-
"""电源状态查询（录制性能测试报告用）：交流/电池、电量、节电模式、Windows 11 设置里的"电源模式"、当前电源计划。

只用文档化的系统 API（ctypes）：``GetSystemPowerStatus``；电源模式用 ``PowerGetEffectiveOverlayScheme``
（powersetting.h，Windows 10 1709 起有；它是覆盖在电源计划之上的一档，与电源计划不是一回事）；
电源计划用 ``PowerGetActiveScheme`` + ``PowerReadFriendlyName``（名称随系统语言变化，常见计划另按 GUID 给固定英文键）。
不调用 ``powercfg`` 解析本地化文字。哪一项取不到就记 None，原因写进 ``errors``，不抛出。
"""
from __future__ import annotations

import ctypes
import uuid
from ctypes import wintypes
from typing import Any

from core.foundation.winapi import GUID, is_windows, library, uuid_from_guid


class _PowerStatus(ctypes.Structure):
    _fields_ = [("ac_line", ctypes.c_ubyte), ("battery_flag", ctypes.c_ubyte), ("battery_percent", ctypes.c_ubyte),
                ("system_status", ctypes.c_ubyte), ("battery_life_s", wintypes.DWORD),
                ("battery_full_life_s", wintypes.DWORD)]


# Windows 设置里"电源模式"各档对应的覆盖方案（better_battery 是 Windows 10 用电池时的一档）
POWER_MODES = {
    uuid.UUID("961cc777-2547-4f9d-8174-7d86181b8a7a"): "best_efficiency",
    uuid.UUID("3af9b8d9-7c97-431d-ad78-34a8bfea439f"): "better_battery",
    uuid.UUID("00000000-0000-0000-0000-000000000000"): "balanced",
    uuid.UUID("ded574b5-45a0-4f42-8737-46345c09c238"): "best_performance",
}
# 系统自带的电源计划（名称随语言变化，GUID 固定）；其它计划（厂商或用户自建）记 other
POWER_PLANS = {
    uuid.UUID("381b4222-f694-41f0-9685-ff5bb260df2e"): "balanced",
    uuid.UUID("8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"): "high_performance",
    uuid.UUID("a1841308-3541-4fab-bc81-f71556f20b4a"): "power_saver",
    uuid.UUID("e9a42b02-d5df-448d-aa00-03f14749eb61"): "ultimate",
}
_NO_BATTERY_FLAGS = (128, 255)


def power_status() -> dict[str, Any]:
    """``ac_power``（接着电源 True、用电池 False）、``battery_percent``、``battery_saver``、
    ``power_mode``（best_efficiency/better_battery/balanced/best_performance，或未知 GUID 原文）、
    ``plan``（``{"guid", "name", "key"}``，key 为 balanced/high_performance/power_saver/ultimate/other）；
    取不到的项为 None，原因在 ``errors``（项名 → 说明）。"""
    result: dict[str, Any] = {"ac_power": None, "battery_percent": None, "battery_saver": None,
                              "power_mode": None, "plan": None, "errors": {}}
    if not is_windows():
        result["errors"] = {"all": "不是 Windows，没有电源信息"}
        return result
    for key, read in (("status", _read_status), ("power_mode", _read_power_mode), ("plan", _read_plan)):
        try:
            values = read()
        except Exception as exc:     # 系统 API 不存在或调用失败：这一项记原因，不影响其它项和测试
            result["errors"][key] = f"{type(exc).__name__}: {exc}"
            continue
        result.update(values)
    return result


def _read_status() -> dict[str, Any]:
    status = _PowerStatus()
    kernel32 = library("kernel32")
    if not kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GetSystemPowerStatus 失败")
    no_battery = status.battery_flag in _NO_BATTERY_FLAGS
    return {
        "ac_power": {0: False, 1: True}.get(status.ac_line),
        "battery_percent": None if no_battery or status.battery_percent == 255 else int(status.battery_percent),
        "battery_saver": bool(status.system_status & 1),
    }


def _read_power_mode() -> dict[str, Any]:
    powrprof = library("powrprof")
    try:
        function = powrprof.PowerGetEffectiveOverlayScheme
    except AttributeError:
        raise OSError("本系统没有 PowerGetEffectiveOverlayScheme（Windows 10 1709 之前），读不到电源模式") from None
    guid = GUID()
    code = function(ctypes.byref(guid))
    if code != 0:
        raise OSError(code, f"PowerGetEffectiveOverlayScheme 返回 {code}")
    value = uuid_from_guid(guid)
    return {"power_mode": POWER_MODES.get(value, str(value))}


def _read_plan() -> dict[str, Any]:
    powrprof, kernel32 = library("powrprof"), library("kernel32")
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    active = ctypes.POINTER(GUID)()
    code = powrprof.PowerGetActiveScheme(None, ctypes.byref(active))
    if code != 0:
        raise OSError(code, f"PowerGetActiveScheme 返回 {code}")
    try:
        value = uuid_from_guid(active.contents)
        name = _friendly_name(powrprof, active)
    finally:
        kernel32.LocalFree(ctypes.cast(active, ctypes.c_void_p))
    return {"plan": {"guid": str(value), "name": name, "key": POWER_PLANS.get(value, "other")}}


def _friendly_name(powrprof: Any, scheme: Any) -> str | None:
    """电源计划的显示名（UTF-16）；先问缓冲区大小再读，读不到返回 None（GUID 与 key 仍可用）。"""
    size = wintypes.DWORD(0)
    if powrprof.PowerReadFriendlyName(None, scheme, None, None, None, ctypes.byref(size)) != 0 or not size.value:
        return None
    buffer = ctypes.create_string_buffer(size.value)
    if powrprof.PowerReadFriendlyName(None, scheme, None, None, buffer, ctypes.byref(size)) != 0:
        return None
    return buffer.raw[:size.value].decode("utf-16-le", errors="replace").rstrip("\x00") or None


__all__ = ["POWER_MODES", "POWER_PLANS", "power_status"]
