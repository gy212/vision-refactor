# -*- coding: utf-8 -*-
"""Tk 输入源状态模型 ``InputSourceState``：摄像头 / 视频 / 无 三态互斥（原 ``apps/camera_enum.py``，P8 随 Tk 层迁入 ``apps/tk/``）。"""
from __future__ import annotations


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
