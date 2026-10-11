"""测试用的已结束片段替身：任意片段 id 与目录（真实片段的 id 与目录由 ``SegmentLayout`` 按时间戳生成）。

``SegmentFinalizer`` 只用到 ``FinishedSegment`` 的 ``segment_id``、``directory``、``views``、``source(view)``、``errors``。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.recording import ViewRecording


@dataclass(frozen=True)
class SegmentDouble:
    segment_id: str
    directory: Path | None
    views: dict[str, ViewRecording] = field(default_factory=dict)
    multi: bool = True

    def source(self, view: str) -> Path:
        return self.views[view].path

    @property
    def errors(self) -> dict[str, str]:
        return {view: rec.error for view, rec in self.views.items() if rec.error}

    @property
    def has_activity(self) -> bool:
        return True


def dual_segment(
    segment_id: str,
    segment_dir: Path,
    front: Path,
    side: Path,
    *,
    front_frames: int = 12,
    side_frames: int = 12,
    front_error: str | None = None,
    side_error: str | None = None,
) -> SegmentDouble:
    return SegmentDouble(
        segment_id=segment_id,
        directory=Path(segment_dir),
        views={
            "front": ViewRecording(Path(front), int(front_frames), front_error),
            "side": ViewRecording(Path(side), int(side_frames), side_error),
        },
    )


def single_segment(segment_id: str, path: Path, frames: int = 12) -> SegmentDouble:
    return SegmentDouble(
        segment_id=segment_id,
        directory=None,
        views={"front": ViewRecording(Path(path), int(frames), None)},
        multi=False,
    )
