# -*- coding: utf-8 -*-
"""P2 评审遗留项（PR #4 评论）的回归：延迟导入 mediapipe、骨架导出只解码一遍、模型目录只解析一次、绘制接受数组。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from batch import batch_export_skeleton as bes  # noqa: E402
from core.media import video_source  # noqa: E402
from core.pose import extract as extract_mod  # noqa: E402
from core.pose import render  # noqa: E402
from core.pose.landmarks import RowLandmark  # noqa: E402

FIX = _REPO_ROOT / "tests" / "fixtures" / "pose33_v3" / "student_raw.npz"


def test_importing_pose_facades_does_not_load_mediapipe_or_touch_log_env():
    code = ("import os, sys; os.environ.pop('GLOG_minloglevel', None); import core.pose_features, core.rule_scoring, "
            "core.action_compare, core.pose; print('mediapipe' in sys.modules, os.environ.get('GLOG_minloglevel'))")
    env = {k: v for k, v in os.environ.items() if k not in ("GLOG_minloglevel", "TF_CPP_MIN_LOG_LEVEL")}
    out = subprocess.run([sys.executable, "-B", "-c", code], cwd=_REPO_ROOT, env=env, capture_output=True,
                         text=True, check=True)
    assert out.stdout.split() == ["False", "None"]


def test_mediapipe_runtime_sets_log_levels_before_import():
    code = ("import os, sys; from core.pose.mediapipe_runtime import mediapipe; mp = mediapipe(); "
            "print(mp is sys.modules['mediapipe'], os.environ['GLOG_minloglevel'], os.environ['TF_CPP_MIN_LOG_LEVEL'])")
    out = subprocess.run([sys.executable, "-B", "-c", code], cwd=_REPO_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["True", "3", "3"]


def _avi(path: Path, n: int, size=(96, 64), fps=10.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n):
        writer.write(np.full((size[1], size[0], 3), (i * 23) % 255, np.uint8))
    writer.release()
    return path


class _ScriptedEngine:
    """第 3、4 帧未检出，其余回放 fixture 关键点。"""

    def __init__(self):
        self.raw = np.load(FIX, allow_pickle=True)["landmarks"]
        self.n = 0

    def infer(self, frame, *, timestamp_ms):
        i, self.n = self.n, self.n + 1
        if i in (3, 4):
            return None, []
        return [SimpleNamespace(x=float(r[0]), y=float(r[1]), z=float(r[2]), visibility=float(r[3]))
                for r in self.raw[i]], []

    def close(self):
        pass


def test_skeleton_export_decodes_once_and_matches_two_pass_output(tmp_path, monkeypatch):
    video = _avi(tmp_path / "in.avi", 9)
    opened = []
    real_open = video_source.open_video

    def counting_open(*args, **kwargs):
        opened.append(args[0])
        return real_open(*args, **kwargs)

    monkeypatch.setattr(extract_mod, "open_video", counting_open)
    monkeypatch.setattr(extract_mod, "create_engine", lambda spec, **_kw: _ScriptedEngine())
    landmarks, meta = bes._extract_pose_and_video(video, out_video=tmp_path / "out.avi", pose_variant="full",
                                                  draw_face=False)
    assert len(opened) == 1 and landmarks.shape == (9, 33, 4) and meta["skeleton_video"].endswith("out.avi")

    # 旧的两遍做法：先提取，再重新解码逐帧画骨架。两者逐帧像素一致。
    raw = np.load(FIX, allow_pickle=True)["landmarks"]
    cap = cv2.VideoCapture(str(video))
    expected = []
    for i in range(9):
        ok, frame = cap.read()
        assert ok
        if i not in (3, 4):
            render.draw_skeleton(frame, raw[i].astype(np.float64))
        expected.append(frame)
    cap.release()
    out = cv2.VideoCapture(str(tmp_path / "out.avi"))
    assert int(out.get(cv2.CAP_PROP_FRAME_COUNT)) == 9
    out.release()
    # mp4v 有损，比对编码前的帧：直接用钩子收集
    frames = []

    class _Collect(bes._SkeletonVideoWriter):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)

        def __call__(self, frame, pose_landmarks):
            annotated = frame.bgr.copy()
            if pose_landmarks is not None:
                render.draw_skeleton(annotated, pose_landmarks)
            frames.append(annotated)

    monkeypatch.setattr(bes, "_SkeletonVideoWriter", _Collect)
    bes._extract_pose_and_video(video, out_video=tmp_path / "out2.avi", pose_variant="full", draw_face=False)
    assert len(frames) == 9 and all(np.array_equal(a, b) for a, b in zip(frames, expected))


def test_extract_resolves_models_dir_once_for_all_segments(monkeypatch, tmp_path):
    video = _avi(tmp_path / "seg.avi", 12)
    calls, dirs = [], []
    monkeypatch.setattr(extract_mod, "default_models_dir", lambda: calls.append(1) or tmp_path / "models")

    def fake_create(spec, *, models_dir=None, download=True):
        dirs.append(models_dir)
        return _ScriptedEngine()

    monkeypatch.setattr(extract_mod, "create_engine", fake_create)
    extract_mod.extract_pose_sequence(video, extract_mod.formal_engine("full"), workers=3)
    assert calls == [1] and len(dirs) == 3 and set(dirs) == {tmp_path / "models"}


def test_render_pose_and_joint_angles_accept_arrays():
    raw = np.load(FIX, allow_pickle=True)["landmarks"][10].astype(np.float64)
    objs = [RowLandmark(r) for r in raw]
    frame = np.zeros((120, 160, 3), np.uint8)
    a_img, a_actions = render.render_pose(frame, raw, [], draw_joint_angles_overlay=True)
    b_img, b_actions = render.render_pose(frame, objs, [], draw_joint_angles_overlay=True)
    assert np.array_equal(a_img, b_img) and a_actions == b_actions
    c, d = frame.copy(), frame.copy()
    render.draw_joint_angles(c, raw)
    render.draw_joint_angles(d, objs)
    assert np.array_equal(c, d) and c.any()
