# -*- coding: utf-8 -*-
"""旧导入路径的转发门面：摄像头枚举已迁到 ``core.media.camera.devices``（P5），输入源状态在 ``apps.input_source``；P9 删除。"""
from apps.input_source import InputSourceState
from core.media import CameraDevice as CameraEntry
from core.media import enumerate_cameras, list_device_names, make_label, probe_camera

__all__ = [
    "CameraEntry",
    "InputSourceState",
    "enumerate_cameras",
    "list_device_names",
    "make_label",
    "probe_camera",
]
