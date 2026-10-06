"""模型管理门面：转发到 ``core.pose.models``（P9 删除）。"""
from core.pose.models import (  # noqa: F401
    DEFAULT_MODEL_PROXY,
    ProgressCb,
    ModelSpec,
    MEDIAPIPE_MODELS,
    YOLO_MODELS,
    MODEL_SPECS,
    model_path,
    is_installed,
    installed_size_mb,
    download_model,
    model_download_proxy,
)

__all__ = [
    "DEFAULT_MODEL_PROXY",
    "ProgressCb",
    "ModelSpec",
    "MEDIAPIPE_MODELS",
    "YOLO_MODELS",
    "MODEL_SPECS",
    "model_path",
    "is_installed",
    "installed_size_mb",
    "download_model",
    "model_download_proxy",
]
