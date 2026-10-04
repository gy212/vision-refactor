"""基础层（L0）的公共 API：几何与信号、原子写、偏好、路径、任务与取消、epoch、诊断、JSON 序列化。"""
from core.foundation.atomic_io import (
    atomic_target,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    replace_file,
)
from core.foundation.geometry import angle_deg, distance, landmark_xy, torso_length, unit
from core.foundation.signal import (
    contiguous_runs,
    find_active_range,
    longest_segment,
    motion_energy,
    smooth_1d,
    trimmed_mean,
)

__all__ = [
    "angle_deg",
    "atomic_target",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "contiguous_runs",
    "distance",
    "find_active_range",
    "landmark_xy",
    "longest_segment",
    "motion_energy",
    "replace_file",
    "smooth_1d",
    "torso_length",
    "trimmed_mean",
    "unit",
]
