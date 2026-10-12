# -*- coding: utf-8 -*-
"""摄像头设备目录 ``DeviceCatalog``：枚举设备、生成设备身份（编号、名称、名称是否唯一），以及打开时的后端解析。

- 正常枚举只读 Media Foundation 设备名，不打开设备；顺序与 OpenCV ``CAP_MSMF`` 的编号一致。
- 设备名清单不可用时才逐编号探测兜底；兜底跳过本程序正在占用的设备并标为"使用中"（V-6），
  不打断池里的预热。
- ``backend_candidates`` 是后端解析的唯一实现，在采集子进程里调用。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Iterator

from core.foundation.tasks import TaskGroup

# 扫描上限与探测超时（见 requirements.md 需求 1.5 / 1.7）
DEFAULT_SCAN_LIMIT = 5
SCAN_LIMIT_MIN = 1
SCAN_LIMIT_MAX = 32
PROBE_TIMEOUT_S = 2.0
BUSY_SUFFIX = "（使用中）"

_probe_tasks = TaskGroup("camera-probe")


@dataclass(frozen=True)
class CameraDevice:
    """设备身份，也是下拉菜单中的一个条目。"""

    label: str  # 面向用户的显示文本，如 "摄像头 0: 1080P USB Camera"
    index: int  # 采集后端使用的非负整数编号
    name: str | None = None  # 设备友好名；拿不到时为 None
    name_unique: bool = True  # 同名设备不止一台时为 False，此时名称不能区分正/侧
    busy: bool = False  # 兜底探测时被本程序占用而跳过（V-6）


@dataclass(frozen=True)
class CatalogSnapshot:
    """一次枚举的结果；``ok=False`` 表示枚举失败，调用方保留原列表。"""

    devices: tuple[CameraDevice, ...] = ()
    ok: bool = True

    def name_of(self, index: int) -> str | None:
        for device in self.devices:
            if device.index == index:
                return device.name
        return None


def make_label(index: int, name: str | None = None, *, busy: bool = False) -> str:
    """由编号（及可选设备名）生成显示文本。需求 2.2。

    - 提供设备名时返回如 ``"摄像头 0: 1080P USB Camera"``；
    - 缺省（拿不到设备名）时回退为 ``"摄像头 0"``；
    - 兜底探测跳过的占用设备追加"（使用中）"。
    编号与文本保持一一对应（同一组枚举内编号唯一即可保证文本唯一）。
    """
    name = (name or "").strip()
    label = f"摄像头 {index}: {name}" if name else f"摄像头 {index}"
    return label + BUSY_SUFFIX if busy else label


# --------------------------------------------------------------------------- #
# Media Foundation 设备名
# --------------------------------------------------------------------------- #

_MF_VERSION = 0x00020070
_KEY_SRCTYPE = "{C60AC5FE-252A-478F-A0EF-BC8FA5F7CAD3}"
_VAL_VIDCAP = "{8AC3587A-4AE7-42D8-99E0-0A6013EEF90F}"
_KEY_FRIENDLY = "{60D0E559-52F8-4FA2-BBCE-ACDB34A8EC01}"


class _MediaFoundation:
    """``MFEnumDeviceSources`` 所需的 ctypes 绑定与 COM vtable 调用。"""

    def __init__(self) -> None:
        import ctypes as C
        from ctypes import wintypes as W

        class GUID(C.Structure):
            _fields_ = [
                ("Data1", W.DWORD),
                ("Data2", W.WORD),
                ("Data3", W.WORD),
                ("Data4", C.c_ubyte * 8),
            ]

        self.C, self.W, self.GUID = C, W, GUID
        self.ole32 = C.windll.ole32
        self.mfplat = C.windll.mfplat
        self.mf = C.windll.mf
        self.ole32.CLSIDFromString.argtypes = [C.c_wchar_p, C.POINTER(GUID)]
        self.ole32.CoTaskMemFree.argtypes = [C.c_void_p]
        self.mfplat.MFStartup.argtypes = [C.c_ulong, C.c_ulong]
        self.mfplat.MFCreateAttributes.argtypes = [C.POINTER(C.c_void_p), C.c_uint32]
        self.mf.MFEnumDeviceSources.argtypes = [
            C.c_void_p,
            C.POINTER(C.POINTER(C.c_void_p)),
            C.POINTER(W.UINT),
        ]

    def guid(self, text: str):
        g = self.GUID()
        if self.ole32.CLSIDFromString(text, self.C.byref(g)) != 0:
            raise OSError("CLSIDFromString failed: " + text)
        return g

    def vcall(self, ptr, idx, argtypes, *args):
        # 通过 COM vtable 序号调用接口方法（ptr 为接口指针）。
        C = self.C
        vtbl = C.cast(ptr, C.POINTER(C.c_void_p))[0]
        fn = C.cast(vtbl, C.POINTER(C.c_void_p))[idx]
        proto = C.WINFUNCTYPE(C.c_long, C.c_void_p, *argtypes)
        return proto(fn)(ptr, *args)

    def friendly_names(self, devices, count: int) -> list[str]:
        """逐个读取 ``IMFActivate`` 的友好名并释放接口。"""
        C, W = self.C, self.W
        key = self.guid(_KEY_FRIENDLY)
        names: list[str] = []
        for i in range(count):
            activate = C.c_void_p(devices[i])
            buf = C.c_wchar_p()
            length = W.UINT(0)
            # IMFActivate::GetAllocatedString = vtable index 13
            hr = self.vcall(
                activate, 13,
                [C.c_void_p, C.POINTER(C.c_wchar_p), C.POINTER(W.UINT)],
                C.byref(key), C.byref(buf), C.byref(length),
            )
            names.append(buf.value if hr == 0 and buf.value else "")
            if buf:
                self.ole32.CoTaskMemFree(C.cast(buf, C.c_void_p))
            # IUnknown::Release = vtable index 2
            self.vcall(activate, 2, [])
        return names

    def enumerate_video_sources(self) -> list[str] | None:
        """在已 ``MFStartup`` 的线程上枚举视频采集设备；任一步失败返回 None。"""
        C, W = self.C, self.W
        attrs = C.c_void_p()
        if self.mfplat.MFCreateAttributes(C.byref(attrs), 1) != 0 or not attrs:
            return None
        try:
            # IMFAttributes::SetGUID = vtable index 24
            if self.vcall(attrs, 24, [C.c_void_p, C.c_void_p],
                          C.byref(self.guid(_KEY_SRCTYPE)),
                          C.byref(self.guid(_VAL_VIDCAP))) != 0:
                return None
            devices = C.POINTER(C.c_void_p)()
            count = W.UINT(0)
            if self.mf.MFEnumDeviceSources(attrs, C.byref(devices), C.byref(count)) != 0:
                return None
            try:
                return self.friendly_names(devices, count.value)
            finally:
                self.ole32.CoTaskMemFree(C.cast(devices, C.c_void_p))
        finally:
            self.vcall(attrs, 2, [])  # IUnknown::Release


def media_foundation_names() -> list[str] | None:
    """用 Windows Media Foundation 枚举视频采集设备的友好名列表。

    返回顺序与 OpenCV ``CAP_MSMF`` 的设备 index 一致——二者同走 Media
    Foundation 的 ``MFEnumDeviceSources`` 枚举，因此与采集进程实际打开的设备一一对应。
    仅 Windows 可用。成功时返回名字列表（可能为空列表，表示确无设备）；任何失败返回
    ``None``，由 :func:`list_device_names` 回退到 pygrabber/DSHOW。
    """
    if sys.platform != "win32":
        return None
    try:
        api = _MediaFoundation()
        # CoInitializeEx：S_OK(0)/S_FALSE(1) 均为本次成功初始化，需配对 Uninit；
        # RPC_E_CHANGED_MODE 等表示线程已被别处初始化，沿用且不由我们 Uninit。
        co_inited = api.ole32.CoInitializeEx(None, 0) in (0, 1)
        mf_started = False
        try:
            if api.mfplat.MFStartup(_MF_VERSION, 0) != 0:
                return None
            mf_started = True
            return api.enumerate_video_sources()
        finally:
            if mf_started:
                api.mfplat.MFShutdown()
            if co_inited:
                api.ole32.CoUninitialize()
    except Exception:
        return None


def _directshow_names() -> list[str] | None:
    try:
        from pygrabber.dshow_graph import FilterGraph

        return list(FilterGraph().get_input_devices())
    except Exception:
        return None


def list_device_names() -> list[str]:
    """读取摄像头友好名列表，顺序与采集进程的设备 index 一致。

    优先用 Media Foundation 枚举（与 ``CAP_MSMF`` 后端同源，索引顺序一致，保证下拉名字与
    实际打开设备对应）。仅当 MF 不可用时回退 pygrabber（DirectShow 顺序）——该降级路径下
    DSHOW 与 MSMF 顺序可能不同、名字与设备可能重新错位，但 Win10/11 标准版 MF 始终可用，
    极少触发。任何失败都返回空列表，由调用方回退到 "摄像头 N" 命名。
    """
    names = media_foundation_names()
    if names is not None:
        return names
    return _directshow_names() or []


def backend_candidates(index: int) -> Iterator[tuple[int, int]]:
    """按顺序给出 ``(cv2 后端, 后端内编号)``；后端解析只在这里做。

    MF 可用时先 ``CAP_MSMF``；设备名唯一且 DirectShow 里也唯一时再按名字找 DSHOW 编号。
    同名双摄不能确定对应关系时不跨后端猜编号，避免录错正/侧摄像头。
    MF 不可用时使用 DirectShow 顺序（与枚举的回退一致）。
    """
    import cv2

    names = media_foundation_names()
    if names is None:
        yield cv2.CAP_DSHOW, index
        return
    yield cv2.CAP_MSMF, index
    if index >= len(names) or names.count(names[index]) != 1:
        return
    directshow = _directshow_names()
    if directshow is not None and directshow.count(names[index]) == 1:
        yield cv2.CAP_DSHOW, directshow.index(names[index])


def device_name(index: int) -> str | None:
    """按采集后端的编号顺序取设备名；拿不到时返回 None（此时不做名称复核）。"""
    names = media_foundation_names()
    if names is None or index >= len(names):
        return None
    return names[index]


# --------------------------------------------------------------------------- #
# 枚举
# --------------------------------------------------------------------------- #


def clamp_scan_limit(scan_limit: int) -> int:
    """将扫描上限钳制到 [SCAN_LIMIT_MIN, SCAN_LIMIT_MAX]。需求 1.7。"""
    return max(SCAN_LIMIT_MIN, min(int(scan_limit), SCAN_LIMIT_MAX))


def probe_camera(index: int, timeout_s: float = PROBE_TIMEOUT_S) -> bool:
    """探测单个编号是否可用（回退路径专用）。

    判定可用 = ``isOpened()`` 为真且能 ``read()`` 到非空帧；带超时与资源释放。
    需求 1.2、1.3、1.5、1.6。单测时可被替换。正常枚举不调用本函数：仅当
    Media Foundation / pygrabber 设备清单均不可用时，作逐编号扫描兜底。
    """
    import cv2  # 延迟导入，避免无 cv2 环境下导入本模块失败

    def _work(_token) -> bool:
        cap = None
        try:
            # 用 DSHOW（首帧快、与回退命名 pygrabber 的 DSHOW 顺序一致）。
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                return False
            ok, frame = cap.read()
            return bool(ok and frame is not None and getattr(frame, "size", 0) > 0)
        except Exception:
            # 驱动错误等：视为不可用（需求 1.3）。
            return False
        finally:
            # 无论成功与否都释放资源（需求 1.6）。
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    handle = _probe_tasks.spawn(f"probe-{index}", _work, owner="catalog")
    if not handle.join(timeout_s):
        # 探测超时：判不可用并继续（需求 1.5）。守护线程会在打开/读取返回后自行释放资源。
        return False
    return bool(handle.result)


def enumerate_cameras(
    scan_limit: int = DEFAULT_SCAN_LIMIT,
    probe: Callable[[int], bool] = probe_camera,
    names: Callable[[], list[str]] = list_device_names,
    busy: Callable[[int], bool] | None = None,
) -> list[CameraDevice]:
    """枚举可用摄像头，返回按 index 升序的设备列表。需求 1.1、1.4、1.7、1.8。

    1. **优先**：``names()`` 返回非空设备清单时直接据清单生成条目，不逐编号探测
       （MSMF 逐个 open+read 既慢又波动，会误杀 1080P 等慢首帧设备）。条目数受
       ``scan_limit`` 上限钳制。
    2. **回退**：设备清单为空时，从 0 逐编号 ``probe`` 扫描至钳制后上限，用无名
       "摄像头 N" 命名兜底。``busy(index)`` 为真的编号是本程序正占用的设备，不探测，
       直接列为"使用中"（V-6）。
    """
    upper = clamp_scan_limit(scan_limit)
    try:
        device_names = names()
    except Exception:
        device_names = []

    if device_names:
        # 优先路径：信任设备清单（与打开后端同顺序），不做慢速逐个 probe。
        return [
            CameraDevice(
                make_label(index, name),
                index,
                name=(name or "").strip() or None,
                name_unique=device_names.count(name) == 1,
            )
            for index, name in enumerate(device_names)
            if index <= upper
        ]

    devices: list[CameraDevice] = []
    for index in range(0, upper + 1):
        if busy is not None and busy(index):
            devices.append(CameraDevice(make_label(index, busy=True), index, busy=True))
        elif probe(index):
            devices.append(CameraDevice(make_label(index, None), index))
    return devices


class DeviceCatalog:
    """最近一次枚举的结果与刷新入口；刷新是阻塞调用，由调用方放到后台执行。"""

    def __init__(
        self,
        *,
        scan_limit: int = DEFAULT_SCAN_LIMIT,
        probe: Callable[[int], bool] = probe_camera,
        names: Callable[[], list[str]] = list_device_names,
    ) -> None:
        self._scan_limit = scan_limit
        self._probe = probe
        self._names = names
        self._snapshot = CatalogSnapshot()

    def snapshot(self) -> CatalogSnapshot:
        return self._snapshot

    def refresh(self, busy: Callable[[int], bool] | None = None) -> CatalogSnapshot:
        """重新枚举；失败时返回 ``ok=False`` 且保留上一次的结果。"""
        try:
            devices = enumerate_cameras(self._scan_limit, self._probe, self._names, busy)
        except Exception:
            return CatalogSnapshot(self._snapshot.devices, ok=False)
        self._snapshot = CatalogSnapshot(tuple(devices), ok=True)
        return self._snapshot


__all__ = [
    "BUSY_SUFFIX",
    "CameraDevice",
    "CatalogSnapshot",
    "DEFAULT_SCAN_LIMIT",
    "DeviceCatalog",
    "PROBE_TIMEOUT_S",
    "SCAN_LIMIT_MAX",
    "SCAN_LIMIT_MIN",
    "backend_candidates",
    "clamp_scan_limit",
    "device_name",
    "enumerate_cameras",
    "list_device_names",
    "make_label",
    "media_foundation_names",
    "probe_camera",
]
