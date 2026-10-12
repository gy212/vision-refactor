# -*- coding: utf-8 -*-
"""摄像头子系统（L1）的公共 API：设备目录、采集规格、池、采集子进程与 ``CameraService`` 门面。"""
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
from core.media.camera.service import (
    CameraReadError,
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
    WarmFrame,
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
    "CaptureSpec",
    "CatalogSnapshot",
    "DeviceCatalog",
    "PRIMARY",
    "ProcessCamera",
    "ROLES",
    "RoleSpecs",
    "SECONDARY",
    "WarmFrame",
    "enumerate_cameras",
    "list_device_names",
    "make_label",
    "probe_camera",
]
