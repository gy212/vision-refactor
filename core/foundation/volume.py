# -*- coding: utf-8 -*-
"""录像盘查询（录制性能测试用）：某个目录所在卷的盘符根、类型（本地/可移动/网络…）、文件系统、剩余空间，
所在磁盘的总线（USB、SATA、NVMe…）与是否有寻道延迟（机械硬盘）；以及绕过系统缓存的顺序读（测真实读速）。

只用文档化的系统 API（ctypes）：``GetVolumePathNameW``、``GetDriveTypeW``、``GetVolumeInformationW``、
``GetDiskFreeSpaceExW``，磁盘属性用 ``DeviceIoControl(IOCTL_STORAGE_QUERY_PROPERTY)``（打开卷设备只要 0 访问权限，
不需要管理员）。USB 移动硬盘的盘符类型也是"本地"，所以 USB 要看总线。哪一项取不到就记 None，原因写进 ``errors``。
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path
from typing import Any

from core.foundation.winapi import is_windows, last_error_text, library

_DRIVE_TYPES = {0: "unknown", 1: "no_root", 2: "removable", 3: "fixed", 4: "remote", 5: "cdrom", 6: "ramdisk"}
_BUS_TYPES = {0: "unknown", 1: "scsi", 2: "atapi", 3: "ata", 4: "1394", 5: "ssa", 6: "fibre", 7: "usb", 8: "raid",
              9: "iscsi", 10: "sas", 11: "sata", 12: "sd", 13: "mmc", 14: "virtual", 15: "file_backed_virtual",
              16: "spaces", 17: "nvme", 18: "scm", 19: "ufs"}
_IOCTL_STORAGE_QUERY_PROPERTY = 0x002D1400
_STORAGE_DEVICE_PROPERTY = 0
_STORAGE_SEEK_PENALTY_PROPERTY = 7
_FILE_SHARE_READ_WRITE = 0x1 | 0x2
_OPEN_EXISTING = 3
_GENERIC_READ = 0x80000000
_FILE_FLAG_NO_BUFFERING = 0x20000000
_FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
_INVALID_HANDLE = ctypes.c_void_p(-1).value
_ALIGN = 4096                       # 无缓冲读要求缓冲区地址与读取长度按扇区对齐；4096 覆盖 512 与 4K 扇区
_GB = 2**30


class _PropertyQuery(ctypes.Structure):
    _fields_ = [("property_id", wintypes.DWORD), ("query_type", wintypes.DWORD), ("extra", ctypes.c_ubyte * 1)]


class _DeviceDescriptor(ctypes.Structure):
    _fields_ = [("version", wintypes.DWORD), ("size", wintypes.DWORD), ("device_type", ctypes.c_ubyte),
                ("device_type_modifier", ctypes.c_ubyte), ("removable_media", ctypes.c_ubyte),
                ("command_queueing", ctypes.c_ubyte), ("vendor_offset", wintypes.DWORD),
                ("product_offset", wintypes.DWORD), ("revision_offset", wintypes.DWORD),
                ("serial_offset", wintypes.DWORD), ("bus_type", ctypes.c_int),
                ("raw_length", wintypes.DWORD), ("raw", ctypes.c_ubyte * 1)]


class _SeekPenalty(ctypes.Structure):
    _fields_ = [("version", wintypes.DWORD), ("size", wintypes.DWORD), ("incurs_seek_penalty", ctypes.c_ubyte)]


def _kernel32() -> Any:
    kernel32 = library("kernel32")
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                     wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.DeviceIoControl.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                         ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                         ctypes.c_void_p]
    kernel32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
                                  ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    return kernel32


def volume_info(path: Path | str) -> dict[str, Any]:
    """``path`` 所在卷：``root``、``drive_type``、``filesystem``、``label``、``free_gb``、``total_gb``、
    ``bus``（usb/sata/nvme/…）、``removable_media``、``product``、``seek_penalty``（True 是机械硬盘）；
    取不到的项为 None，原因在 ``errors``。"""
    result: dict[str, Any] = {key: None for key in ("root", "drive_type", "filesystem", "label", "free_gb",
                                                   "total_gb", "bus", "removable_media", "product", "seek_penalty")}
    result["errors"] = {}
    if not is_windows():
        result["errors"]["all"] = "不是 Windows，没有卷信息"
        return result
    kernel32 = _kernel32()
    root = ctypes.create_unicode_buffer(260)
    if not kernel32.GetVolumePathNameW(str(Path(path).resolve()), root, len(root)):
        result["errors"]["root"] = last_error_text("GetVolumePathNameW")
        return result
    result["root"] = root.value
    result["drive_type"] = _DRIVE_TYPES.get(kernel32.GetDriveTypeW(root.value), "unknown")
    for key, read in (("volume", _read_volume), ("space", _read_space), ("device", _read_device)):
        try:
            result.update(read(kernel32, root.value))
        except Exception as exc:
            result["errors"][key] = str(exc)
    return result


def _read_volume(kernel32: Any, root: str) -> dict[str, Any]:
    label, filesystem = ctypes.create_unicode_buffer(261), ctypes.create_unicode_buffer(261)
    if not kernel32.GetVolumeInformationW(root, label, len(label), None, None, None, filesystem, len(filesystem)):
        raise OSError(last_error_text("GetVolumeInformationW"))
    return {"filesystem": filesystem.value or None, "label": label.value}


def _read_space(kernel32: Any, root: str) -> dict[str, Any]:
    free, total = ctypes.c_ulonglong(), ctypes.c_ulonglong()
    if not kernel32.GetDiskFreeSpaceExW(root, ctypes.byref(free), ctypes.byref(total), None):
        raise OSError(last_error_text("GetDiskFreeSpaceExW"))
    return {"free_gb": round(free.value / _GB, 1), "total_gb": round(total.value / _GB, 1)}


def _read_device(kernel32: Any, root: str) -> dict[str, Any]:
    """卷所在磁盘的总线、型号与寻道延迟；网络盘、虚拟盘等可能不支持，失败时由调用方记原因。"""
    if not (len(root) >= 2 and root[1] == ":"):
        raise OSError(f"{root} 不是盘符卷，读不到磁盘属性")
    handle = kernel32.CreateFileW(f"\\\\.\\{root[:2]}", 0, _FILE_SHARE_READ_WRITE, None, _OPEN_EXISTING, 0, None)
    if handle in (None, _INVALID_HANDLE):
        raise OSError(last_error_text("打开卷设备"))
    try:
        buffer = ctypes.create_string_buffer(1024)
        _query(kernel32, handle, _STORAGE_DEVICE_PROPERTY, buffer)
        descriptor = _DeviceDescriptor.from_buffer_copy(buffer.raw[:ctypes.sizeof(_DeviceDescriptor)])
        names = [_string_at(buffer.raw, offset) for offset in (descriptor.vendor_offset, descriptor.product_offset)]
        result = {"bus": _BUS_TYPES.get(descriptor.bus_type, str(descriptor.bus_type)),
                  "removable_media": bool(descriptor.removable_media),
                  "product": " ".join(name for name in names if name) or None}
        penalty = ctypes.create_string_buffer(ctypes.sizeof(_SeekPenalty))
        try:
            _query(kernel32, handle, _STORAGE_SEEK_PENALTY_PROPERTY, penalty)
            result["seek_penalty"] = bool(_SeekPenalty.from_buffer_copy(penalty.raw).incurs_seek_penalty)
        except OSError:
            result["seek_penalty"] = None       # 部分 USB 桥接芯片不回答这一项
        return result
    finally:
        kernel32.CloseHandle(handle)


def _query(kernel32: Any, handle: int, property_id: int, out: Any) -> None:
    query = _PropertyQuery(property_id, 0)
    returned = wintypes.DWORD()
    if not kernel32.DeviceIoControl(handle, _IOCTL_STORAGE_QUERY_PROPERTY, ctypes.byref(query),
                                    ctypes.sizeof(query), out, len(out), ctypes.byref(returned), None):
        raise OSError(last_error_text("IOCTL_STORAGE_QUERY_PROPERTY"))


def _string_at(raw: bytes, offset: int) -> str:
    if not offset or offset >= len(raw):
        return ""
    return raw[offset:raw.index(b"\x00", offset)].decode("ascii", errors="replace").strip()


class UnbufferedReader:
    """按 ``chunk_bytes``（向上取整到 4096）顺序读整个文件：``read()`` 读下一块、返回读到的字节数（0 为文件尾）。
    Windows 上用 ``FILE_FLAG_NO_BUFFERING`` 绕过系统缓存（``uncached`` 为 True，读速反映磁盘本身）；
    打不开或不是 Windows 时退到普通读取（``uncached`` 为 False，刚写过的文件可能从缓存读出、偏快）。用 ``with`` 关闭。"""

    def __init__(self, path: Path | str, chunk_bytes: int) -> None:
        self.size = -(-int(chunk_bytes) // _ALIGN) * _ALIGN
        self.uncached = False
        self._handle = None
        self._stream = None
        if is_windows():
            kernel32 = _kernel32()
            handle = kernel32.CreateFileW(str(path), _GENERIC_READ, 0x1, None, _OPEN_EXISTING,
                                          _FILE_FLAG_NO_BUFFERING | _FILE_FLAG_SEQUENTIAL_SCAN, None)
            if handle not in (None, _INVALID_HANDLE):
                self._kernel32, self._handle, self.uncached = kernel32, handle, True
                self._storage = ctypes.create_string_buffer(self.size + _ALIGN)
                self._aligned = (ctypes.addressof(self._storage) + _ALIGN - 1) // _ALIGN * _ALIGN
                self._done = wintypes.DWORD()
        if self._handle is None:
            self._stream = open(path, "rb", buffering=0)
            self._buffer = bytearray(self.size)

    def read(self) -> int:
        if self._stream is not None:
            return self._stream.readinto(self._buffer) or 0
        if not self._kernel32.ReadFile(self._handle, self._aligned, self.size, ctypes.byref(self._done), None):
            raise OSError(last_error_text("ReadFile"))
        return int(self._done.value)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
        elif self._handle is not None:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None

    def __enter__(self) -> "UnbufferedReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["UnbufferedReader", "volume_info"]
