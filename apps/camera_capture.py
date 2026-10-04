"""Tk 桌面摄像头：驱动只在一个子进程内使用，卡死时可单独回收。"""
from __future__ import annotations

import multiprocessing as mp
import threading
import time

import numpy as np


def _read_frame(capture):
    """短暂空帧可重试；异常帧不进入预览、推理或录像。"""
    for _ in range(5):
        ok, frame = capture.read()
        if (ok and isinstance(frame, np.ndarray) and frame.dtype == np.uint8
                and frame.ndim == 3 and frame.shape[2] == 3
                and frame.shape[0] > 0 and frame.shape[1] > 0):
            return np.ascontiguousarray(frame)
        time.sleep(0.03)
    raise RuntimeError("连续读不到有效 BGR 画面，请检查设备连接后重试")


def _camera_sources(index):
    import cv2
    from apps.camera_enum import _enumerate_media_foundation_names

    names = _enumerate_media_foundation_names()
    if names is None:
        # 桌面枚举此时也使用 DirectShow 顺序。
        yield cv2.CAP_DSHOW, index
        return
    yield cv2.CAP_MSMF, index
    if index >= len(names) or names.count(names[index]) != 1:
        return
    try:
        from pygrabber.dshow_graph import FilterGraph
        directshow = list(FilterGraph().get_input_devices())
    except Exception:
        return
    # 同名双摄不能确定对应关系时不跨后端猜编号，避免录错正/侧摄像头。
    if directshow.count(names[index]) == 1:
        yield cv2.CAP_DSHOW, directshow.index(names[index])


def _open_native(index: int, width: int, height: int):
    import cv2

    # 整组格式在首次 read 前设置，禁止对正在预热的 native capture 热切尺寸。
    error = None
    for backend, device in _camera_sources(index):
        for mjpg in (True, False):
            capture = cv2.VideoCapture(device, backend)
            try:
                if not capture.isOpened():
                    raise RuntimeError("无法打开设备，可能被其他程序占用")
                if mjpg:
                    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                return capture, _read_frame(capture)
            except Exception as exc:
                error = exc
                capture.release()
    raise RuntimeError(f"摄像头 {index} 格式协商/首帧失败：{error}") from error


def _capture_worker(connection, buffer, index, width, height, factory):
    capture = None
    phase = "open/first-frame"
    try:
        if factory is None:
            capture, first = _open_native(index, width, height)
        else:
            capture = factory(index, width=width, height=height)
            first = _read_frame(capture)
        storage = np.frombuffer(buffer, dtype=np.uint8)
        if first.size > storage.size:
            raise RuntimeError("设备返回的画面超过采集缓冲区，请降低摄像头分辨率")
        connection.send(("ready", {3: first.shape[1], 4: first.shape[0],
                                   5: float(capture.get(5) or 0.0)}))
        phase = "read"
        while True:
            command = connection.recv()
            if command == "close":
                return
            if command != "read":
                raise ValueError(f"unsupported camera command: {command}")
            frame = first if first is not None else _read_frame(capture)
            first = None
            if frame.size > storage.size:
                raise RuntimeError("设备返回的画面超过采集缓冲区")
            storage[:frame.size] = frame.reshape(-1)
            connection.send(("frame", frame.shape))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send(("error", f"{phase}: {type(exc).__name__}: {exc}"))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        try:
            if capture is not None:
                capture.release()
        finally:
            connection.close()


class ProcessCamera:
    """只传递帧元数据；像素走固定共享内存，每次 read 返回独立连续数组。"""

    def __init__(self, index: int, *, width: int = 1280, height: int = 720,
                 stop_event=None, startup_timeout: float = 12.0,
                 read_timeout: float = 2.0, _factory=None):
        if index < 0 or width <= 0 or height <= 0:
            raise ValueError("摄像头编号和画面尺寸无效")
        self.index = index
        self._timeout = read_timeout
        self._interrupted = threading.Event()
        self._io_lock = threading.RLock()
        self._closed = False
        self._properties = {}
        context = mp.get_context("spawn")
        # ponytail: 单槽覆盖最高请求尺寸或 1080p；需要 4K 时按请求尺寸分配。
        self._buffer = context.RawArray("B", max(width * height, 1920 * 1080) * 3)
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_capture_worker,
            args=(child, self._buffer, index, width, height, _factory),
            name=f"camera-{index}", daemon=True)
        try:
            self._process.start()
            child.close()
            kind, self._properties = self._receive(startup_timeout, stop_event)
            if kind != "ready":
                raise RuntimeError("摄像头启动响应无效")
            print(f"[camera-capture] index={index} requested={width}x{height} "
                  f"actual={self._properties[3]}x{self._properties[4]} "
                  f"fps={self._properties[5]}", flush=True)
        except BaseException:
            self.release()
            raise
        finally:
            child.close()

    def _receive(self, timeout, stop_event=None):
        deadline = time.monotonic() + timeout
        while True:
            if self._interrupted.is_set() or (stop_event is not None and stop_event.is_set()):
                raise RuntimeError(f"摄像头 {self.index} 采集已取消")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"摄像头 {self.index} 响应超时，采集进程将重置，可重新开始")
            if self._connection.poll(min(0.05, remaining)):
                try:
                    kind, value = self._connection.recv()
                except (EOFError, OSError) as exc:
                    raise RuntimeError(f"摄像头 {self.index} 采集进程已退出，可重新开始") from exc
                if kind == "error":
                    raise RuntimeError(f"摄像头 {self.index}：{value}")
                return kind, value
            if not self._process.is_alive():
                raise RuntimeError(f"摄像头 {self.index} 采集进程已退出，可重新开始")

    def isOpened(self):
        return not self._closed and not self._interrupted.is_set() and self._process.is_alive()

    def get(self, prop):
        return self._properties.get(prop, 0.0)

    def read(self):
        with self._io_lock:
            if self._closed:
                return False, None
            try:
                self._connection.send("read")
                kind, shape = self._receive(self._timeout)
                if kind != "frame" or len(shape) != 3 or shape[2] != 3:
                    raise RuntimeError("摄像头返回了无效帧格式")
                height, width, _ = shape
                if min(height, width) <= 0 or height * width * 3 > len(self._buffer):
                    raise RuntimeError("摄像头返回了无效帧尺寸")
                frame = np.frombuffer(self._buffer, dtype=np.uint8,
                                      count=height * width * 3).reshape(shape).copy()
                self._properties.update({3: width, 4: height})
                return True, frame
            except BaseException:
                self.release()
                raise

    def interrupt(self):
        # 不跨线程 release native capture；读帧等待看到标志后回收整个子进程。
        self._interrupted.set()

    def release(self):
        self.interrupt()
        with self._io_lock:
            if self._closed:
                return
            process = self._process
            if process.pid is not None:
                if process.is_alive():
                    try:
                        self._connection.send("close")
                    except (BrokenPipeError, EOFError, OSError):
                        pass
                    process.join(0.2)
                if process.is_alive():
                    process.terminate()
                    process.join(0.5)
                if process.is_alive():
                    process.kill()
                    process.join(0.5)
                if process.is_alive():
                    raise RuntimeError(f"摄像头 {self.index} 采集进程未退出，禁止重复打开")
                process.close()
            self._connection.close()
            self._closed = True


open_camera = ProcessCamera
