"""并行推理门面：转发到 ``core.pose.parallel``（P9 删除）。"""
from core.pose.parallel import (  # noqa: F401
    InferResult,
    PipelineFactory,
    default_pipeline_factory,
    ParallelPoseEngine,
)

__all__ = [
    "InferResult",
    "PipelineFactory",
    "default_pipeline_factory",
    "ParallelPoseEngine",
]
