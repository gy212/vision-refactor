# -*- coding: utf-8 -*-
"""录制/暂停运行时控制状态机（与 Tkinter 解耦，可独立单元/属性测试）。

设计来源：``.kiro/specs/ui-layout-redesign/design.md``。

``RecordingController`` 是一个跨线程共享的录制状态机：
  - UI 线程通过 ``request_toggle()`` 请求状态变更（idle⇄recording⇄paused）。
  - Worker 线程每帧调用 ``write_frame()``，由控制器依据当前状态决定写或跳过。
  - 会话结束时调用 ``close_session()`` 释放 writer 并复位 idle。

所有真实 I/O（``VideoWriter`` 的创建）通过依赖注入的 ``writer_factory`` 完成，
便于属性测试用假对象替换；内部以单把 ``threading.Lock`` 串行化所有状态/写盘操作。

本文件对应任务 1.1：仅搭建模块骨架、类型别名与数据模型，并完整实现 ``__init__``、
默认 ``path_provider`` 与默认 ``writer_factory``；各状态机方法体由后续任务填充。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Optional, Protocol, Tuple

from core.paths import outputs_dir
from core.video_writer import open_video_writer

# 录制状态：空闲 / 录制中 / 已暂停
RecordingState = Literal["idle", "recording", "paused"]


class VideoWriterLike(Protocol):
    """writer 句柄的最小协议：写帧与释放。

    真实实现为 ``cv2.VideoWriter``；属性测试可注入记录调用的假对象。
    """

    def write(self, frame) -> None: ...

    def release(self) -> None: ...


# writer 工厂签名：给定目标路径 / fps / size，返回 (writer, 实际落盘路径, codec 标签)。
# 默认绑定到 core.video_writer.open_video_writer。
WriterFactory = Callable[[Path, float, Tuple[int, int]], Tuple[VideoWriterLike, Path, str]]


@dataclass(frozen=True)
class RecordingSnapshot:
    """不可变快照，供 UI 线程刷新状态显示。"""

    state: RecordingState
    result_path: Optional[Path]
    frames_written: int
    last_error: Optional[str]


def _default_path_provider() -> Path:
    """默认保存路径：``outputs_dir()/record_<timestamp>.mp4``。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return outputs_dir() / f"record_{timestamp}.mp4"


def _default_writer_factory(
    path: Path, fps: float, size: Tuple[int, int]
) -> Tuple[VideoWriterLike, Path, str]:
    """默认 writer 工厂：绑定 ``core.video_writer.open_video_writer``。"""
    return open_video_writer(path, fps=fps, size=size)


class RecordingController:
    """录制状态机控制器（线程安全）。"""

    def __init__(
        self,
        writer_factory: WriterFactory = _default_writer_factory,
        path_provider: Callable[[], Path] = _default_path_provider,
    ) -> None:
        self._writer_factory: WriterFactory = writer_factory
        self._path_provider: Callable[[], Path] = path_provider

        self._state: RecordingState = "idle"
        self._lock = threading.Lock()
        self._writer: Optional[VideoWriterLike] = None
        self._result_path: Optional[Path] = None
        self._session_active: bool = False
        self._fps: float = 30.0
        self._size: Tuple[int, int] = (0, 0)
        self._frames_written: int = 0
        self._last_error: Optional[str] = None

    @property
    def state(self) -> RecordingState:
        """返回当前录制状态。"""
        with self._lock:
            return self._state

    def begin_session(self, *, fps: float, size: Tuple[int, int]) -> None:
        """会话启动：登记本次会话写入参数，状态保持 idle，不创建 writer（懒创建）。"""
        with self._lock:
            self._fps = fps
            self._size = size
            self._session_active = True
            self._state = "idle"
            self._writer = None
            self._result_path = None
            self._frames_written = 0
            self._last_error = None

    def update_session_size(self, *, size: Tuple[int, int]) -> bool:
        """在 writer 懒创建前用首帧真实尺寸修正会话参数。

        摄像头驱动上报的 ``CAP_PROP_FRAME_WIDTH/HEIGHT`` 可能与实际帧不一致。
        会话未启动或 writer 已创建时拒绝修改，避免中途改变输出尺寸。
        """
        with self._lock:
            if not self._session_active or self._writer is not None:
                return False
            self._size = size
            return True

    @property
    def session_size(self) -> Tuple[int, int]:
        """当前会话登记的输出尺寸（writer 目标宽高）。"""
        with self._lock:
            return self._size

    def request_toggle(self) -> RecordingState:
        """UI 线程调用：idle→recording / recording→paused / paused→recording 循环切换。

        会话未运行（``_session_active`` 为 False）时为 no-op 并保持 ``idle``。
        ``idle→recording`` 仅切换状态：writer 保持 ``None`` 表示"需懒创建"，真正的
        创建延迟到首个 ``write_frame``（任务 2.3）。返回切换后的状态。
        """
        with self._lock:
            if not self._session_active:
                # 会话未运行：no-op，保持 idle（需求 5.1）
                return self._state

            if self._state == "idle":
                # idle→recording：仅置状态，不在此创建 writer（懒创建语义）。
                # writer 仍为 None，首个 write_frame 时再创建。
                self._state = "recording"
            elif self._state == "recording":
                # recording→paused：停止写新帧，保留已写帧与同一 writer。
                self._state = "paused"
            else:  # self._state == "paused"
                # paused→recording：续写同一文件（复用现有 writer，不新建）。
                self._state = "recording"

            return self._state

    def write_frame(self, frame) -> None:
        """Worker 线程每帧调用：仅当状态为 recording 时写盘（首帧懒创建 writer）。

        - 在锁内判定状态：仅 ``recording`` 写盘；``paused``/``idle`` 直接跳过且不
          丢弃已写帧（需求 5.5/I4）。
        - 首次写盘时经 ``_writer_factory`` 懒创建 writer，记录工厂返回的
          ``actual_path`` 为 ``_result_path``（需求 5.3）。
        - 写盘成功后 ``_frames_written += 1``。
        - 创建或写入抛异常时进入错误处理：释放任何半开资源、``_writer=None``、
          置 ``_last_error``、状态复位 ``idle``，异常不外泄到 worker 主循环（需求 5.11）。
        """
        with self._lock:
            # 仅录制中写盘；paused/idle 跳过，保留已写帧不丢弃（I4）。
            if self._state != "recording":
                return

            try:
                # 首帧懒创建 writer：取工厂返回的 actual_path 作为落盘路径（编码回退可能改后缀）。
                if self._writer is None:
                    writer, actual_path, _codec = self._writer_factory(
                        self._path_provider(), self._fps, self._size
                    )
                    self._writer = writer
                    self._result_path = actual_path

                self._writer.write(frame)
                self._frames_written += 1
            except Exception as exc:  # noqa: BLE001 - 任何创建/写入失败都需复位并记录
                # 释放任何半开资源（创建成功但 write 失败，或工厂部分初始化）。
                if self._writer is not None:
                    try:
                        self._writer.release()
                    except Exception:  # noqa: BLE001 - 释放阶段的二次异常忽略
                        pass
                self._writer = None
                self._last_error = str(exc)
                self._state = "idle"

    def stop_recording(self) -> Optional[Path]:
        """结束当前录制片段：释放 writer 并将状态复位为 ``idle``，但**保持会话运行**。

        与 ``close_session`` 的区别：本方法不结束会话（``_session_active`` 保持 True），
        因此用户可在同一识别会话内再次「开始录制」生成新的文件片段。用于 UI 上独立的
        「结束录制」按钮（与会话级「停止」解耦）。

        - 在锁内执行：无论当前为 ``recording``、``paused`` 还是 ``idle`` 都安全。
        - 释放 writer（``release()`` 后置 ``_writer=None``），复位 ``_state='idle'``，
          并清空本片段的 ``_result_path``、``_frames_written`` 与 ``_last_error``，
          使同一会话内的下次录制干净开始。
        - 会话未运行时为 no-op，返回 ``None``。
        - 返回本片段实际写入路径（曾录制过则为该路径，否则 ``None``）。
        """
        with self._lock:
            if not self._session_active:
                return None
            result_path = self._result_path
            if self._writer is not None:
                try:
                    self._writer.release()
                except Exception:  # noqa: BLE001 - 释放失败不得阻断状态复位
                    pass
            self._writer = None
            self._state = "idle"
            self._result_path = None
            self._frames_written = 0
            self._last_error = None
            return result_path

    def close_session(self) -> Optional[Path]:
        """会话结束/停止：释放 writer、复位 idle，返回本会话实际写入路径或 None。

        - 在锁内执行：无论当前为 ``recording``、``paused`` 还是 ``idle``，都释放
          writer（调用 ``release()`` 后置 ``_writer=None``）、``_session_active=False``、
          状态复位 ``idle``（需求 4.4/5.8）。
        - ``release()`` 自身可能抛异常：防御性吞掉，确保 state/_writer 始终被复位。
        - 清空 ``last_error``，防止已结束会话的错误污染下一种输入模式。
        - 返回本会话实际写入路径：曾录制过则为 ``_result_path``，否则 ``None``。
        """
        with self._lock:
            if self._writer is not None:
                try:
                    self._writer.release()
                except Exception:  # noqa: BLE001 - 释放失败不得阻断状态复位
                    pass
            self._writer = None
            self._session_active = False
            self._state = "idle"
            self._last_error = None
            return self._result_path

    def snapshot(self) -> RecordingSnapshot:
        """返回不可变快照（state、result_path、frames_written、last_error）。"""
        with self._lock:
            return RecordingSnapshot(
                state=self._state,
                result_path=self._result_path,
                frames_written=self._frames_written,
                last_error=self._last_error,
            )
