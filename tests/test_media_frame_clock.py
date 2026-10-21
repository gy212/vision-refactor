# -*- coding: utf-8 -*-
"""``FrameClock`` 去抖：摄像头按网格曝光、主机到达带随机延迟（甲方实测量级）时，时间轴不再"重复一帧、跳过一帧"；
真实的掉帧、低帧率与卡顿照实保留；估计值不晚于到达、严格递增、不掉帧时不早于曝光。"""
from __future__ import annotations

import numpy as np
import pytest

from core.media import BoundaryShift, CapturedFrame, FrameClock, Timeline

P_NS = 1e9 / 30


def _camera(seconds=30.0, fps=30.0, delay_ms=8.0, seed=0, drop_every=None, stall_at_s=None):
    """返回 ``(曝光时刻, 到达时刻)``：曝光在 ``fps`` 网格上（可每 ``drop_every`` 帧掉一帧、在某时刻卡 0.5 s），
    到达 = 曝光 + 指数分布延迟（均值 8 ms 时帧间隔 p95 约 50 ms、最大约 85 ms，与甲方报告一致）。"""
    rng = np.random.default_rng(seed)
    index = np.arange(int((seconds + 10) * fps))
    if drop_every:
        index = index[index % drop_every != 0]
    exposure = index * (1e9 / fps) + 1e9
    if stall_at_s is not None:
        exposure = np.where(exposure > stall_at_s * 1e9, exposure + 0.5e9, exposure)
    arrival = exposure + rng.exponential(delay_ms * 1e6, len(index)) if delay_ms else exposure.copy()
    arrival = np.maximum.accumulate(arrival.astype(np.int64)) + np.arange(len(index))   # 到达严格递增
    return exposure.astype(np.int64), arrival


def _frames(arrival, *, seqs=None, generation=0):
    seqs = range(len(arrival)) if seqs is None else seqs
    return [CapturedFrame("front", int(q), int(a), None, generation) for q, a in zip(seqs, arrival)]


def _dejitter(frames, fps=30.0):
    clock = FrameClock(fps)
    out = [f for frame in frames for f in clock.push(frame)] + clock.flush()
    return out, clock


def _repeats(frames, seed, seconds=30.0, clock=None, phase=None):
    """30 s 录像的补帧数。开录时刻相对帧的相位随 ``seed`` 变化（或由 ``phase`` 指定，单位为周期）；
    给出 ``clock`` 时按 ``segment_shift`` 推迟这一路本段的边界。"""
    rng = np.random.default_rng(1000 + seed)
    timeline = Timeline(30)
    start = int(frames[150].captured_at_ns + (rng.uniform(0, P_NS) if phase is None else phase * P_NS))
    if clock is not None:
        start = clock.segment_shift(start).apply(start)
    timeline.start(start, max((f for f in frames if f.captured_at_ns <= start), key=lambda f: f.captured_at_ns))
    end = start + int(seconds * 1e9)
    for frame in frames:
        if start < frame.captured_at_ns <= end:
            timeline.push(frame)
    timeline.stop(end)
    assert timeline.stats.slots == 900
    return timeline.stats.repeated_slots


@pytest.mark.parametrize("seed", range(5))
def test_client_like_jitter_no_longer_stutters(seed):
    _exposure, arrival = _camera(seed=seed)
    raw = _frames(arrival)
    gaps = np.diff(arrival) / 1e6
    assert np.percentile(gaps, 95) > 45 and gaps.max() > 60          # 抖动量级与甲方报告相当
    estimated, clock = _dejitter(raw)
    assert _repeats(estimated, seed, clock=clock) == 0 and clock.stats.inferred_drops == 0
    assert sum(_repeats(_frames(_camera(seed=s)[1]), s) for s in range(5)) > 100   # 不去抖时整体明显重复


def test_segment_shift_keeps_slots_away_from_frames_at_every_phase():
    """帧恰在输出槽边界附近时，亚毫秒的估计误差就会来回跨界；按 ``segment_shift`` 推迟边界后任何相位都不重复。"""
    estimated, clock = _dejitter(_frames(_camera(seed=4)[1]))
    phases = np.linspace(0, 1, 40, endpoint=False)
    assert max(_repeats(estimated, 0, phase=ph) for ph in phases) > 0              # 不对齐时总有相位会重复
    assert [_repeats(estimated, 0, clock=clock, phase=ph) for ph in phases] == [0] * 40
    for at in (5_000_000_000, 5_012_345_678):
        shift = clock.segment_shift(at)
        assert 0 <= shift.ns < P_NS and shift.apply(at) == at + shift.ns
    assert FrameClock(30).segment_shift(123) == BoundaryShift(0)


def _record(frames, events, shift):
    """按调用约定驱动一段录制：每个边界（已用 ``shift`` 换算）之前推完采集于它之前的帧，再执行边界；返回输出帧数。"""
    timeline, pushed = Timeline(30), 0
    for kind, at in events:
        at = shift.apply(at)
        while pushed < len(frames) and frames[pushed].captured_at_ns <= at:
            if kind != "start":
                timeline.push(frames[pushed])
            pushed += 1
        if kind == "start":
            timeline.start(at, frames[pushed - 1])
        else:
            getattr(timeline, kind)(at)
    return timeline.stats.slots


def _phased_camera(phase, seed=0):
    """无掉帧的 30 fps 摄像头，曝光网格错开 ``phase`` 个周期，到达带甲方量级的抖动；返回去抖后的帧与时钟。"""
    _exposure, arrival = _camera(seed=seed)
    return _dejitter(_frames(arrival + int(phase * P_NS)))


def test_views_with_different_phases_keep_equal_frame_counts():
    """评审复现（PR #42，P1）：正侧相位差半帧、区间 30.3 帧时，每个边界各算一次平移会让两路区间长度不同、
    帧数 31 比 30；每路在段首算一次平移、整段沿用，两路帧数相同。"""
    front, front_clock = _phased_camera(0.0)
    side, side_clock = _phased_camera(0.5, seed=1)
    start = 6_000_000_000
    stop = start + int(30.3 * P_NS)
    events = [("start", start), ("stop", stop)]

    class PerBoundary:                  # 被评审否定的用法：每个边界各算一次平移
        def __init__(self, clock):
            self.clock = clock

        def apply(self, at):
            return self.clock.segment_shift(at).apply(at)

    per_boundary = (_record(front, events, PerBoundary(front_clock)), _record(side, events, PerBoundary(side_clock)))
    assert per_boundary[0] != per_boundary[1]
    counts = (_record(front, events, front_clock.segment_shift(start)),
              _record(side, events, side_clock.segment_shift(start)))
    assert counts[0] == counts[1] == 31


@pytest.mark.parametrize("seed", range(4))
def test_equal_frame_counts_with_pauses_at_random_boundaries(seed):
    rng = np.random.default_rng(seed)
    cameras = [_phased_camera(phase, seed=k) for k, phase in enumerate(rng.uniform(0, 1, 2))]
    for _ in range(25):
        start = int(6e9 + rng.uniform(0, 2e9))
        cuts = np.sort(rng.uniform(0.2e9, 20e9, 3)).astype(np.int64)
        events = [("start", start), ("pause", start + cuts[0]), ("resume", start + cuts[1]), ("stop", start + cuts[2])]
        counts = {_record(frames, events, clock.segment_shift(start)) for frames, clock in cameras}
        assert len(counts) == 1


@pytest.mark.parametrize("fps, max_repeats", [(29.97, 2), (30.03, 0)])
def test_small_rate_drift_is_followed(fps, max_repeats):
    for seed in range(3):
        estimated, clock = _dejitter(_frames(_camera(fps=fps, seed=seed)[1]))
        assert _repeats(estimated, seed, clock=clock) <= max_repeats   # 29.97 fps 每 30 s 真少约 1 帧


@pytest.mark.parametrize("seed", range(3))
def test_estimates_stay_between_exposure_and_arrival_without_drops(seed):
    exposure, arrival = _camera(delay_ms=12.0, seed=seed)
    estimated, clock = _dejitter(_frames(arrival))
    stamps = np.array([f.captured_at_ns for f in estimated])
    assert len(estimated) == len(arrival) and [f.seq for f in estimated] == list(range(len(arrival)))
    assert (stamps <= arrival).all() and (np.diff(stamps) > 0).all()
    assert (stamps >= exposure - 1_000_000).all()
    assert clock.stats.max_correction_ns == int((arrival - stamps).max())


@pytest.mark.parametrize("seed", range(3))
def test_camera_drops_are_kept_as_real_repeats(seed):
    """摄像头每 10 帧掉 1 帧（读帧序号仍连续）：补帧数等于真实空档，判定的掉帧数也对上。"""
    exposure, arrival = _camera(seed=seed, drop_every=10)
    estimated, clock = _dejitter(_frames(arrival))
    assert 88 <= _repeats(estimated, seed, clock=clock) <= 92
    true_drops = int(np.sum(np.diff(np.round((exposure - 1e9) / P_NS)) - 1))
    assert abs(clock.stats.inferred_drops - true_drops) <= 2
    stamps = np.array([f.captured_at_ns for f in estimated])
    assert (stamps <= arrival).all() and (stamps > exposure - P_NS).all()   # 个别帧可能偏早，但不到一个周期


def test_low_light_15fps_repeats_every_frame():
    exposure, arrival = _camera(fps=15.0)
    estimated, clock = _dejitter(_frames(arrival))
    assert _repeats(estimated, 0, clock=clock) == 450
    assert clock.stats.inferred_drops == len(arrival) - 1
    stamps = np.array([f.captured_at_ns for f in estimated])
    assert (stamps <= arrival).all() and (stamps > exposure - P_NS).all()


def test_half_second_stall_becomes_fifteen_repeats():
    _exposure, arrival = _camera(stall_at_s=10.0)
    estimated, clock = _dejitter(_frames(arrival))
    assert _repeats(estimated, 0, clock=clock) == 15 and clock.stats.inferred_drops == 15


def test_faster_camera_degrades_to_arrival_times():
    _exposure, arrival = _camera(fps=60.0)
    estimated, clock = _dejitter(_frames(arrival))
    stamps = np.array([f.captured_at_ns for f in estimated])
    assert (stamps <= arrival).all() and (np.diff(stamps) > 0).all() and clock.stats.inferred_drops == 0


def test_queue_drops_advance_by_seq_without_counting_camera_drops():
    """写盘队列满时丢掉的帧表现为 seq 跳号：按跳号推进网格，不算作摄像头掉帧。"""
    _exposure, arrival = _camera(delay_ms=0.0)
    keep = [i for i in range(len(arrival)) if i % 7 != 3]
    estimated, clock = _dejitter(_frames(arrival[keep], seqs=keep))
    assert clock.stats.seq_gaps == len(arrival) - len(keep) and clock.stats.inferred_drops == 0
    assert np.abs(np.array([f.captured_at_ns for f in estimated]) - arrival[keep]).max() < 1_000   # 无抖动时即到达时刻


def test_frames_are_released_after_lookahead_and_flush_releases_the_rest():
    clock = FrameClock(30, lookahead=4)
    raw = _frames(_camera(delay_ms=0.0)[1][:6])
    released = [len(clock.push(frame)) for frame in raw]
    assert released == [0, 0, 0, 0, 1, 1]
    rest = clock.flush()
    assert [f.seq for f in rest] == [2, 3, 4, 5] and clock.flush() == []
    assert clock.stats.frames == 6


def test_out_of_order_frames_are_ignored_and_pixels_are_kept():
    clock = FrameClock(30, lookahead=0)
    first = CapturedFrame("side", 0, 1_000_000_000, "pixels-0")
    assert clock.push(first)[0].bgr == "pixels-0"
    assert clock.push(CapturedFrame("side", 1, 999_000_000, "late")) == []
    second = clock.push(CapturedFrame("side", 1, 1_033_000_000, "pixels-1"))[0]
    assert (second.role, second.seq, second.bgr) == ("side", 1, "pixels-1")


def test_device_reopen_restarts_the_clock():
    _exposure, arrival = _camera(delay_ms=0.0)
    clock = FrameClock(30)
    first = [f for frame in _frames(arrival[:40]) for f in clock.push(frame)]
    later = arrival[200:240] + 5_000_000                       # 重开后相位不同、seq 从 0 重新计
    second = [f for frame in _frames(later, generation=1) for f in clock.push(frame)] + clock.flush()
    assert len(first) + len(second) == 80 and clock.stats.resets == 1
    assert [f.generation for f in second] == [0] * 4 + [1] * 40       # 先放出旧一代前瞻中的帧
    reopened = np.array([f.captured_at_ns for f in second[4:]])
    assert np.abs(reopened - later).max() < 1_000                       # 按新一代重新起算
    assert clock.stats.inferred_drops == 0


def test_fps_must_be_positive():
    with pytest.raises(ValueError):
        FrameClock(0)
