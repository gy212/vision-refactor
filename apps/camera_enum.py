# -*- coding: utf-8 -*-
"""摄像头枚举与输入源状态模型。

本模块不依赖 tkinter，便于独立单元测试与属性测试：
- `enumerate_cameras` 的探测函数 `probe` 可注入，给定 probe 后输出完全确定。
- `InputSourceState` 是纯状态结构，维护摄像头/视频/无 三态互斥不变式。

真实的 OpenCV MSMF I/O 集中在 `probe_camera`，单测时可被替换。
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Callable

# 扫描上限与探测超时（见 requirements.md 需求 1.5 / 1.7）
DEFAULT_SCAN_LIMIT = 5
SCAN_LIMIT_MIN = 1
SCAN_LIMIT_MAX = 32
PROBE_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class CameraEntry:
    """下拉菜单中的单个摄像头条目。"""

    label: str  # 面向用户的显示文本，如 "摄像头 0"
    index: int  # 传给 cv2.VideoCapture 的非负整数编号


def make_label(index: int, name: str | None = None) -> str:
    """由编号（及可选设备名）生成显示文本。需求 2.2。

    - 提供设备名时返回如 ``"摄像头 0: 1080P USB Camera"``；
    - 缺省（拿不到设备名）时回退为 ``"摄像头 0"``。
    编号与文本保持一一对应（同一组枚举内编号唯一即可保证文本唯一）。
    """
    name = (name or "").strip()
    if name:
        return f"摄像头 {index}: {name}"
    return f"摄像头 {index}"


def _enumerate_media_foundation_names() -> list[str] | None:
    """用 Windows Media Foundation 枚举视频采集设备的友好名列表。

    返回顺序与 OpenCV ``CAP_MSMF`` 的设备 index 一致——二者同走 Media
    Foundation 的 ``MFEnumDeviceSources`` 枚举，因此与 :func:`open_camera`
    实际打开的设备一一对应。仅 Windows 可用。

    成功时返回名字列表（可能为空列表，表示确无设备）；任何失败返回
    ``None``，由 :func:`list_device_names` 回退到 pygrabber/DSHOW。
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes as C
        from ctypes import wintypes as W

        class GUID(C.Structure):
            _fields_ = [
                ("Data1", W.DWORD),
                ("Data2", W.WORD),
                ("Data3", W.WORD),
                ("Data4", C.c_ubyte * 8),
            ]

        ole32 = C.windll.ole32
        mfplat = C.windll.mfplat
        mf = C.windll.mf
        ole32.CLSIDFromString.argtypes = [C.c_wchar_p, C.POINTER(GUID)]
        ole32.CoTaskMemFree.argtypes = [C.c_void_p]
        mfplat.MFStartup.argtypes = [C.c_ulong, C.c_ulong]
        mfplat.MFCreateAttributes.argtypes = [C.POINTER(C.c_void_p), C.c_uint32]
        mf.MFEnumDeviceSources.argtypes = [
            C.c_void_p,
            C.POINTER(C.POINTER(C.c_void_p)),
            C.POINTER(W.UINT),
        ]

        def _guid(text: str) -> GUID:
            g = GUID()
            if ole32.CLSIDFromString(text, C.byref(g)) != 0:
                raise OSError("CLSIDFromString failed: " + text)
            return g

        def _vcall(ptr, idx, argtypes, *args):
            # 通过 COM vtable 序号调用接口方法（ptr 为接口指针）。
            vtbl = C.cast(ptr, C.POINTER(C.c_void_p))[0]
            fn = C.cast(vtbl, C.POINTER(C.c_void_p))[idx]
            proto = C.WINFUNCTYPE(C.c_long, C.c_void_p, *argtypes)
            return proto(fn)(ptr, *args)

        MF_VERSION = 0x00020070
        KEY_SRCTYPE = _guid("{C60AC5FE-252A-478F-A0EF-BC8FA5F7CAD3}")
        VAL_VIDCAP = _guid("{8AC3587A-4AE7-42D8-99E0-0A6013EEF90F}")
        KEY_FRIENDLY = _guid("{60D0E559-52F8-4FA2-BBCE-ACDB34A8EC01}")

        # CoInitializeEx：S_OK(0)/S_FALSE(1) 均为本次成功初始化，需配对 Uninit；
        # RPC_E_CHANGED_MODE 等表示线程已被别处初始化，沿用且不由我们 Uninit。
        co_inited = ole32.CoInitializeEx(None, 0) in (0, 1)
        mf_started = False
        try:
            if mfplat.MFStartup(MF_VERSION, 0) != 0:
                return None
            mf_started = True

            attrs = C.c_void_p()
            if mfplat.MFCreateAttributes(C.byref(attrs), 1) != 0 or not attrs:
                return None
            try:
                # IMFAttributes::SetGUID = vtable index 24
                if _vcall(attrs, 24, [C.c_void_p, C.c_void_p],
                          C.byref(KEY_SRCTYPE), C.byref(VAL_VIDCAP)) != 0:
                    return None

                devices = C.POINTER(C.c_void_p)()
                count = W.UINT(0)
                if mf.MFEnumDeviceSources(attrs, C.byref(devices),
                                          C.byref(count)) != 0:
                    return None
                try:
                    names: list[str] = []
                    for i in range(count.value):
                        activate = C.c_void_p(devices[i])
                        buf = C.c_wchar_p()
                        length = W.UINT(0)
                        # IMFActivate::GetAllocatedString = vtable index 13
                        hr = _vcall(
                            activate, 13,
                            [C.c_void_p, C.POINTER(C.c_wchar_p), C.POINTER(W.UINT)],
                            C.byref(KEY_FRIENDLY), C.byref(buf), C.byref(length),
                        )
                        names.append(buf.value if hr == 0 and buf.value else "")
                        if buf:
                            ole32.CoTaskMemFree(C.cast(buf, C.c_void_p))
                        # IUnknown::Release = vtable index 2
                        _vcall(activate, 2, [])
                    return names
                finally:
                    ole32.CoTaskMemFree(C.cast(devices, C.c_void_p))
            finally:
                _vcall(attrs, 2, [])  # IUnknown::Release
        finally:
            if mf_started:
                mfplat.MFShutdown()
            if co_inited:
                ole32.CoUninitialize()
    except Exception:
        return None


def list_device_names() -> list[str]:
    """读取摄像头友好名列表，顺序与 :func:`open_camera` 的设备 index 一致。

    优先用 Media Foundation 枚举（与 open_camera 的 ``CAP_MSMF`` 后端同源，
    索引顺序一致，保证下拉名字与实际打开设备对应）。仅当 MF 不可用时回退
    pygrabber（DirectShow 顺序）——该降级路径下 DSHOW 与 MSMF 顺序可能不同、
    名字与设备可能重新错位，但 Win10/11 标准版 MF 始终可用，极少触发。
    任何失败都返回空列表，由调用方回退到 "摄像头 N" 命名。
    """
    names = _enumerate_media_foundation_names()
    if names is not None:
        return names
    try:
        from pygrabber.dshow_graph import FilterGraph

        return list(FilterGraph().get_input_devices())
    except Exception:
        return []


def clamp_scan_limit(scan_limit: int) -> int:
    """将扫描上限钳制到 [SCAN_LIMIT_MIN, SCAN_LIMIT_MAX]。需求 1.7。"""
    return max(SCAN_LIMIT_MIN, min(int(scan_limit), SCAN_LIMIT_MAX))


def probe_camera(index: int, timeout_s: float = PROBE_TIMEOUT_S) -> bool:
    """探测单个编号是否可用（回退路径专用）。

    判定可用 = `isOpened()` 为真且能 `read()` 到非空帧；带超时与资源释放。
    需求 1.2、1.3、1.5、1.6。该函数封装真实 cv2 I/O，单测时可被替换。
    正常枚举不再调用本函数（见 :func:`enumerate_cameras`）：仅当 Media
    Foundation / pygrabber 设备清单均不可用时，作逐编号扫描兜底。
    """
    import cv2  # 延迟导入，避免无 cv2 环境下导入本模块失败

    result: dict[str, bool] = {"ok": False}

    def _work() -> None:
        cap = None
        try:
            # 仅回退路径使用：正常枚举走 list_device_names 的 Media Foundation
            # 清单，不调用本函数。此处用 DSHOW（首帧快、与回退命名 pygrabber 的
            # DSHOW 顺序一致），仅当 MF/pygrabber 均不可用时由 enumerate_cameras
            # 逐编号扫描兜底。
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                result["ok"] = False
                return
            ok, frame = cap.read()
            result["ok"] = bool(
                ok and frame is not None and getattr(frame, "size", 0) > 0
            )
        except Exception:
            # 驱动错误等：视为不可用（需求 1.3）。
            result["ok"] = False
        finally:
            # 无论成功与否都释放资源（需求 1.6）。
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    worker = threading.Thread(target=_work, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        # 探测超时：判不可用并继续（需求 1.5）。底层线程为 daemon，
        # 会在打开/读取返回后自行结束并释放资源。
        return False
    return bool(result["ok"])


def open_camera(index: int, *, width: int = 1280, height: int = 720):
    """打开摄像头并协商一个高帧率的采集格式。

    背景：在 Windows 上用 ``cv2.CAP_DSHOW`` 打开摄像头，请求 720p 时很多设备会
    退回到未压缩的 YUY2 像素格式。YUY2 受 USB 带宽限制，720p 往往只能跑到 ~10fps，
    导致主循环长时间阻塞在 ``cap.read()`` 上、CPU 利用率极低、整体 FPS 被采集端拖死。

    本 helper 优先用 ``CAP_MSMF`` 后端（通常能协商到压缩格式，720p 满 30fps），
    失败再回退到 ``CAP_DSHOW`` + 强制 MJPG，最后回退到朴素 DSHOW。返回已打开的
    ``cv2.VideoCapture``；若全部失败则返回最后一次尝试的 cap（由调用方检查
    ``isOpened()``）。
    """
    import cv2  # 延迟导入，保持本模块在无 cv2 环境下可被导入/测试

    def _try(backend: int, *, mjpg: bool):
        cap = cv2.VideoCapture(int(index), backend)
        if not cap.isOpened():
            return None
        if mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        return cap

    # 1) MSMF 默认（最优：本机实测 720p 30fps）。与 probe_camera 探测、
    #    list_device_names 的 Media Foundation 命名同后端，保证下拉所选 index
    #    与实际打开设备一致（DSHOW 与 MSMF 的设备枚举顺序可能不同）。
    cap = _try(cv2.CAP_MSMF, mjpg=False)
    if cap is not None:
        return cap
    # 2) DSHOW + 强制 MJPG（压缩格式，避免 YUY2 带宽瓶颈）。
    cap = _try(cv2.CAP_DSHOW, mjpg=True)
    if cap is not None:
        return cap
    # 3) 最后回退：朴素 DSHOW（即便慢，也优先保证能出图）。
    return cv2.VideoCapture(int(index), cv2.CAP_DSHOW)


def enumerate_cameras(
    scan_limit: int = DEFAULT_SCAN_LIMIT,
    probe: Callable[[int], bool] = probe_camera,
    names: Callable[[], list[str]] = list_device_names,
) -> list[CameraEntry]:
    """枚举可用摄像头，返回按 index 升序的条目列表。

    需求 1.1、1.4、1.7、1.8。两条路径：

    1. **优先**：当 ``names()`` 返回非空设备清单（Windows 正常情形下走
       Media Foundation，顺序与 :func:`open_camera` 的 ``CAP_MSMF`` index
       一致），直接据清单生成条目——快、稳，且名字与实际打开设备一一对应。
       不再逐编号调用 ``probe``（MSMF 逐个 open+read 既慢又波动，会误杀
       1080P 等慢首帧设备）。条目数受 ``scan_limit`` 上限钳制。
    2. **回退**：仅当设备清单为空（MF/pygrabber 均不可用）时，从 0 逐编号
       ``probe`` 扫描至钳制后上限，用无名 "摄像头 N" 命名兜底。

    - scan_limit 被钳制到 [SCAN_LIMIT_MIN, SCAN_LIMIT_MAX]。
    - probe / names 可注入，便于在不接触真实硬件的情况下做属性测试。
    - 返回列表按 index 升序，每个 index 至多出现一次。
    """
    upper = clamp_scan_limit(scan_limit)
    try:
        device_names = names()
    except Exception:
        device_names = []

    entries: list[CameraEntry] = []
    if device_names:
        # 优先路径：信任设备清单（与打开后端同顺序），不做慢速逐个 probe。
        for index, name in enumerate(device_names):
            if index > upper:
                break
            entries.append(CameraEntry(make_label(index, name), index))
        return entries

    # 回退路径：无设备清单时逐编号探测兜底。
    for index in range(0, upper + 1):
        if probe(index):
            entries.append(CameraEntry(make_label(index, None), index))
    return entries


class InputSourceState:
    """输入源状态模型：camera / video / none 三态互斥。

    需求 3.3、3.4。不变式：
    - kind == "camera" ⇒ value.isdigit()
    - kind == "video"  ⇒ not value.isdigit()
    - kind == "none"   ⇒ value == ""
    任一时刻至多存在一个生效输入源。
    """

    __slots__ = ("kind", "value")

    def __init__(self) -> None:
        self.kind: str = "none"
        self.value: str = ""

    def select_camera(self, index: int) -> None:
        """选择摄像头编号，清除任何视频残留。需求 3.3。"""
        self.kind = "camera"
        self.value = str(int(index))

    def select_video(self, path: str) -> None:
        """选择视频文件路径，清除摄像头选中语义。需求 3.4。"""
        self.kind = "video"
        self.value = path

    def clear(self) -> None:
        """清除输入源（无生效源）。"""
        self.kind = "none"
        self.value = ""

    def hint_text(self) -> str:
        """当前输入源的指示文本。需求 3.5。"""
        if self.kind == "camera":
            return f"当前输入源：摄像头 {self.value}"
        if self.kind == "video":
            return f"当前输入源：视频文件 {self.value}"
        return "当前输入源：未选择"
