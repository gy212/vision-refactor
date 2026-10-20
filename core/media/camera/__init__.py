# -*- coding: utf-8 -*-
"""摄像头子系统（L1）的公共 API：设备目录、采集规格、池、采集子进程、会话 reader 与 ``CameraService`` 门面。"""
from core.media.camera.capture_process import ProcessCamera
from core.media.camera.devices import (
    CameraDevice,
    CatalogSnapshot,
    DeviceCatalog,
    enumerate_cameras,
    list_device_names,
    make_label,
    probe_camera,
)
from core.media.camera.pool import CameraLease, CameraPool
from core.media.camera.reader import CameraReadError, CaptureReader, LatestSlot
from core.media.camera.service import (
    CameraService,
    CameraStream,
    RoleSpecs,
)
from core.media.camera.spec import (
    PRIMARY,
    ROLES,
    SECONDARY,
    CameraWarmupError,
    CameraWarmupStopped,
    CameraWarmupTimeout,
    CaptureSpec,
)

__all__ = [
    "CameraDevice",
    "CameraLease",
    "CameraPool",
    "CameraReadError",
    "CameraService",
    "CameraStream",
    "CameraWarmupError",
    "CameraWarmupStopped",
    "CameraWarmupTimeout",
    "CaptureReader",
    "CaptureSpec",
    "CatalogSnapshot",
    "DeviceCatalog",
    "PRIMARY",
    "LatestSlot",
    "ProcessCamera",
    "ROLES",
    "RoleSpecs",
    "SECONDARY",
    "enumerate_cameras",
    "list_device_names",
    "make_label",
    "probe_camera",
]
