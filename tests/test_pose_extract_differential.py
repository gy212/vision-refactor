# -*- coding: utf-8 -*-
"""P2 差分测试：新 ``extract_pose_sequence`` 与迁移前 4 个提取循环（``legacy_extract_oracle``）逐元素一致。

fixture 回放用 golden harness，顺序与分段两种 tracking 都覆盖；``np.array_equal``（NaN 视为相等），不用 allclose。
真实视频的差分素材与结果记在 change.md（素材不入库）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis import tech_eval  # noqa: E402
from core import action_compare as ac  # noqa: E402
from core import feedback_geometry as fg  # noqa: E402
from core.pose import normalize_pose_xy, normalize_pose_xy_v1, normalize_pose_xy_v3  # noqa: E402
from core.rule_scoring import extract_pose_raw, extract_pose_raw_series  # noqa: E402
from tests import golden_harness as H  # noqa: E402
from tests import legacy_extract_oracle as old  # noqa: E402

FIX_DIR = Path(__file__).resolve().parent / "fixtures" / "pose33_v3"
VIDEOS = {}


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")


def _with_gaps(raw: np.ndarray) -> np.ndarray:
    """加入未检出帧（回放为 None）与 NaN 帧，覆盖补帧分支。"""
    out = raw.copy()
    out[[0, 1, 7, 8, 9, 30]] = np.nan
    return out


class _GapPipeline(H._FakePipeline):
    """NaN 帧回放为未检出（``pose_landmarks=None``）。"""

    def infer(self, frame, *, timestamp_ms=None):
        if np.isnan(frame.landmarks).all():
            return None, []
        return super().infer(frame, timestamp_ms=timestamp_ms)


@pytest.fixture(autouse=True)
def videos(monkeypatch):
    """同问题 3：对 ``create_engine`` 的 monkeypatch 必须在 ``replay_context()`` 内部撤销，
    否则 fixture 级 ``monkeypatch`` 之后才 undo，会把假引擎重新写回 ``extract.create_engine``。
    """
    H.clear_registry()
    for name in ("student_raw", "front_src_raw", "side_src_raw"):
        raw = np.asarray(np.load(FIX_DIR / f"{name}.npz", allow_pickle=True)["landmarks"], dtype=np.float32)
        key = H.register_video(f"golden://diff_{name}.mp4", _with_gaps(raw), fps=29.97)
        VIDEOS[name] = key
    monkeypatch.setattr(old, "MediaPipePipeline", _GapPipeline)
    with H.replay_context(), pytest.MonkeyPatch.context() as mp:
        from core.pose import extract

        mp.setattr(extract, "create_engine", lambda spec, **_kw: _GapPipeline(cfg=spec))
        yield
    H.clear_registry()


@pytest.mark.parametrize("workers", [1, 2, 3])
@pytest.mark.parametrize("normalizer", [normalize_pose_xy, normalize_pose_xy_v1, normalize_pose_xy_v3])
@pytest.mark.parametrize("name", ["student_raw", "front_src_raw", "side_src_raw"])
def test_features_and_view_scores_match_legacy(name, normalizer, workers):
    video = Path(VIDEOS[name])
    new = ac._extract_pose_features(video, pose_variant="full", workers=workers, normalizer=normalizer, compute_view=True)
    ref = old._extract_pose_features(video, pose_variant="full", workers=workers, normalizer=normalizer, compute_view=True)
    assert _same(new[0], ref[0]) and new[1] == ref[1] and _same(new[2], ref[2])


@pytest.mark.parametrize("name", ["student_raw", "front_src_raw"])
@pytest.mark.parametrize("end", [None, 0, 12, 10**6])
def test_rule_raw_series_matches_legacy(name, end):
    video = VIDEOS[name]
    if end is None:
        new = extract_pose_raw_series(video, pose_variant="full")
        ref = old._extract_pose_raw_prefix(Path(video), pose_variant="full", end_frame=None)
        assert _same(new.landmarks, ref)
        assert _same(new.meta["valid_mask"], ref[..., 3] >= 0.5)
    else:
        new, meta = extract_pose_raw(video, pose_variant="full", start_frame=0, end_frame=end)
        ref = old._extract_pose_raw_prefix(Path(video), pose_variant="full", end_frame=end)
        assert _same(new, ref)


@pytest.mark.parametrize("bounds", [(None, None), (5, 40), (0, 0), (20, 10**6)])
def test_tech_eval_raw_and_view_scores_match_legacy(bounds):
    video = Path(VIDEOS["side_src_raw"])
    start, end = bounds
    new = tech_eval.extract_pose_and_view_scores(video, pose_variant="full", start_frame=start, end_frame=end)
    ref = old.extract_pose_and_view_scores(video, pose_variant="full", start_frame=start, end_frame=end)
    assert _same(new[0], ref[0]) and _same(new[1], ref[1]) and new[2] == ref[2]


def _avi(path: Path, n: int, size: tuple[int, int], fps: float) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    for i in range(n):
        writer.write(np.full((size[1], size[0], 3), (i * 7) % 255, np.uint8))
    writer.release()
    return path


@pytest.mark.parametrize("cfg", [
    fg.FeedbackConfig(),
    fg.FeedbackConfig(max_edge=160),
    fg.FeedbackConfig(sample_fps=7),
])
def test_practice_series_matches_legacy(tmp_path, cfg):
    video = _avi(tmp_path / "practice.avi", 23, (320, 180), 30.0)

    class Pipeline:
        def __init__(self):
            self.n = 0

        def infer(self, frame, *, timestamp_ms):
            self.n += 1
            if self.n % 4 == 0:
                return None, []
            v = (self.n % 9) / 10
            return [SimpleNamespace(x=v, y=1 - v, z=v / 3, visibility=0.4 + v / 2) for _ in range(33)], []

        def close(self):
            pass

    with _real_capture():
        new = fg.extract_series(video, cfg, lambda: False, Pipeline)
        ref = old.extract_series(video, cfg, lambda: False, Pipeline)
    for key in ("landmarks", "valid_mask", "frames", "times"):
        assert _same(new[key], ref[key]), key
    new["capture"].pop("analysisSeconds"), ref["capture"].pop("analysisSeconds")
    assert new["capture"] == ref["capture"]


class _real_capture:
    """练习用真实 AVI 解码：临时恢复被 golden harness 替换的 ``cv2.VideoCapture``。"""

    def __enter__(self):
        self._fake = cv2.VideoCapture
        cv2.VideoCapture = _ORIG_CAPTURE

    def __exit__(self, *exc):
        cv2.VideoCapture = self._fake


_ORIG_CAPTURE = cv2.VideoCapture
