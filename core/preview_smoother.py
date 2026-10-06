"""预览平滑门面：转发到 ``core.pose.smoothing``（P9 删除）。"""
from core.pose.smoothing import (  # noqa: F401
    SmoothedLandmark,
    PreviewLandmarkSmoother,
)

__all__ = [
    "SmoothedLandmark",
    "PreviewLandmarkSmoother",
]
