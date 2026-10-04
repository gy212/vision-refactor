# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.exam_clip import (
    align_frame_counts,
    estimate_action_range,
    estimate_action_range_from_feature_seq,
    prepare_exam_pair,
)


def test_align_frame_counts_min() -> None:
    n, warnings = align_frame_counts(100, 97)
    assert n == 97
    assert any("frame_align" in w for w in warnings)


def test_estimate_action_range_fallback_time() -> None:
    start, end, warnings = estimate_action_range(
        90, fps=30.0, energy=None, smart_crop=True, trim_head_s=0.5, trim_tail_s=0.8
    )
    assert start == 15  # 0.5*30
    assert end == 90 - 24  # 0.8*30
    assert end > start


def test_exam_default_no_energy_keeps_full_tail() -> None:
    # 考试默认（smart_crop=False）：无 energy → 不裁尾，只固定掐头
    start, end, _ = estimate_action_range(90, fps=30.0, energy=None)
    assert start == 15  # 0.5*30 掐头
    assert end == 90  # 尾部不按时间掐


def test_exam_default_tail_scan_drops_trailing_quiet() -> None:
    # 尾部 2s 空场（能量低）应被回扫切掉，中段动作保留
    fps = 30.0
    n = 300  # 10s
    rng = np.random.default_rng(0)
    energy = np.full(n - 1, 0.001, dtype=np.float32)
    # 1.5s~5s 动作，能量有起伏（真实动作非恒定），末尾 ~5s 空场
    energy[45:150] = 0.3 + 0.2 * rng.random(105).astype(np.float32)
    start, end, _ = estimate_action_range(n, fps=fps, energy=energy)
    assert start == 15
    # 末动作 idx ~149 → +2 +0.3s(9) ≈ 160；尾部 ~140 帧空场被切
    assert 155 <= end <= 170


def test_exam_default_force_finish_keeps_last_action() -> None:
    # force_finish：尾部仍是动作（无空场）→ 回扫停在末帧附近，不吞末动作
    fps = 30.0
    n = 200
    energy = np.full(n - 1, 0.5, dtype=np.float32)  # 全程动作
    _, end, _ = estimate_action_range(n, fps=fps, energy=energy)
    assert end >= n - int(round(0.3 * fps)) - 2  # 末动作不被吞


def test_estimate_action_range_from_motion_features() -> None:
    # 头尾静止、中段运动
    t, j = 120, 4
    feats = np.zeros((t, j, 2), dtype=np.float32)
    for i in range(40, 80):
        feats[i, :, 0] = (i - 40) * 0.05
    start, end, _ = estimate_action_range_from_feature_seq(feats, fps=30.0)
    assert start < 50
    assert end > 70


def test_prepare_exam_pair_aligns_and_writes(tmp_path: Path) -> None:
    import cv2

    def write_video(path: Path, n: int, color: tuple[int, int, int]) -> None:
        w, h = 64, 48
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        wr = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
        for i in range(n):
            # 中段加噪声形成能量
            fr = np.full((h, w, 3), color, dtype=np.uint8)
            if 20 <= i < 50:
                fr = (fr + np.random.randint(0, 40, fr.shape, dtype=np.uint8)).clip(0, 255).astype(
                    np.uint8
                )
            wr.write(fr)
        wr.release()

    front = tmp_path / "front.mp4"
    side = tmp_path / "side.mp4"
    write_video(front, 60, (10, 10, 10))
    write_video(side, 55, (20, 20, 20))
    out = tmp_path / "out"
    result = prepare_exam_pair(
        front,
        side,
        front_frames=60,
        side_frames=55,
        out_dir=out,
        trim_head_s=0.2,
        trim_tail_s=0.2,
    )
    assert result.front_frames == result.side_frames
    assert result.front_frames > 0
    assert result.front_path.is_file()
    assert result.side_path.is_file()
    assert any("frame_align" in w for w in result.warnings)
    assert result.meta.get("streamed") is True
