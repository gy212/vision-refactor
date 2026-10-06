"""推理封装门面：引擎在 ``core.pose.engine``，绘制在 ``core.pose.render``，实时标签在 ``core.pose.live_labels``（P9 删除）。"""
from core.pose.engine import (  # noqa: F401
    EngineSpec,
    MediaPipePipeline,
    PipelineConfig,
    PoseEngine,
    create_engine,
)

__all__ = ["EngineSpec", "MediaPipePipeline", "PipelineConfig", "PoseEngine", "create_engine"]
