"""学生练习时间分段；保留原始点及无效帧，不影响模板/正式评分链路。"""
from __future__ import annotations

import math

import numpy as np


def torso_length(lm, mask):
    """只以可见的同侧肩髋链作尺度；不要求被遮挡的远侧也可见。"""
    lengths = np.linalg.norm(lm[:, [11, 12], :2] - lm[:, [23, 24], :2], axis=2)
    valid = mask[:, [11, 12]] & mask[:, [23, 24]] & np.isfinite(lengths) & (lengths > 1e-4)
    return np.where(valid, lengths, 0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)


def limb_indices(role, stance):
    left = (stance == "left") == (role != "rear")
    return (11, 13, 15, 23, 25, 27, 29, 31) if left else (12, 14, 16, 24, 26, 28, 30, 32)


def contiguous_runs(indices):
    if len(indices) == 0:
        return []
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)


def phase_windows(series: dict, action: str, stance: str, cfg, diagnostics=None) -> dict:
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.update(reason="没有找到持续稳定的起始护位", events=[])
    lm, mask = series["landmarks"], series["valid_mask"]
    times = np.asarray(series["times"], dtype=float)
    n = len(lm)
    if n < cfg.min_valid * 2 + 1:
        return {}
    dt = float(np.median(np.diff(times)))
    if not math.isfinite(dt) or dt <= 0:
        return {}
    edge = max(cfg.min_valid, int(math.ceil(cfg.stance_window_seconds / dt)))
    torso = torso_length(lm, mask)
    wrists = (lm[:, [15, 16], :2] - lm[:, [11, 12], :2]) / np.maximum(torso[:, None, None], 1e-4)
    stable = []
    for start in range(n - edge + 1):
        ix = np.arange(start, start + edge)
        if not (torso[ix] > 1e-4).all():
            continue
        # 每窗至少一条完整可见的护位链。远侧遮挡不否定近侧，但可见的移动手仍否定稳定。
        visible = 0
        for arm, (sh, wr, hip) in enumerate(((11, 15, 23), (12, 16, 24))):
            if not mask[ix][:, [sh, wr, hip]].all():
                continue
            visible += 1
            relative = wrists[ix, arm]
            if (np.max(np.linalg.norm(relative - np.median(relative, axis=0), axis=1)) > cfg.return_distance / 2
                    or np.max(np.linalg.norm(lm[ix, hip, :2] - np.median(lm[ix, hip, :2], axis=0), axis=1)) > .15 * np.median(torso[ix])
                    or not (lm[ix, wr, 1] < lm[ix, hip, 1] + .1 * torso[ix]).all()):
                visible = 0
                break
        if visible:
            stable.append(ix)
    if not stable:
        return {}
    baseline = stable[0]
    diagnostics["reason"] = "未检测到完整离开并回收的动作"
    windows = {("stance", "start"): baseline, ("stance", "end"): stable[-1]}
    segments = {"straight_combo": ["front_straight", "rear_straight"],
                "hook_combo": ["front_hook", "rear_hook"]}.get(action, [] if action == "stance" else [action])
    events = []
    for segment in segments:
        role = "front" if segment.startswith("front_") else "rear"
        sh, _, wr, hip_index, _, ankle, _, _ = limb_indices(role, stance)
        anchor, moving = (hip_index, ankle) if "kick" in segment else (sh, wr)
        valid = mask[:, [anchor, moving]].all(axis=1) & (torso > 1e-4)
        if not valid[baseline].all():
            continue
        relative = (lm[:, moving, :2] - lm[:, anchor, :2]) / np.maximum(torso[:, None], 1e-4)
        distance = np.linalg.norm(relative - np.median(relative[baseline], axis=0), axis=1)
        active_start = None
        last_valid = int(baseline[-1])
        rest_index = last_valid
        armed = True
        for i in range(last_valid + 1, n):
            if not valid[i]:
                if times[i] - times[last_valid] > cfg.max_gap_seconds:
                    active_start = None
                    armed = False
                continue
            if active_start is None:
                # 间断后必须先观察到护位，再观察到离开；不从伸到一半的位置虚构起手。
                if distance[i] <= cfg.return_distance:
                    armed, rest_index = True, i
                elif distance[i] > cfg.motion_distance and armed:
                    active_start = rest_index
                    armed = False
            elif distance[i] <= cfg.return_distance:
                ix = np.arange(active_start, i + 1)
                observed = ix[valid[ix]]
                if len(observed) >= cfg.min_valid and times[i] - times[active_start] >= cfg.min_motion_seconds:
                    peak = int(observed[np.argmax(distance[observed])])
                    events.append((segment, active_start, peak, i))
                active_start = None
                armed, rest_index = True, i
            last_valid = i
    events.sort(key=lambda event: event[2])
    diagnostics["events"] = [e[0] for e in events]
    if not events:
        return windows
    # 多次完整前后组合逐轮保留；错序/缺失不强行拼成一个动作。
    if len(segments) > 1:
        expected = segments * (len(events) // len(segments))
        if [e[0] for e in events] != expected:
            diagnostics["reason"] = "检测到的前后手次数或顺序不完整"
            return windows
    diagnostics["reason"] = "已检测到完整动作"
    for segment in segments:
        selected = [e for e in events if e[0] == segment]
        if not selected:
            continue
        windows[(segment, "motion")] = np.unique(np.concatenate([np.arange(s, e + 1) for _, s, _, e in selected]))
        peaks = []
        for _, start, peak, end in selected:
            radius = max(1, int(round(cfg.finish_window_seconds / dt)))
            indices = np.arange(max(start, peak - radius), min(end, peak + radius) + 1)
            if len(indices) >= cfg.min_valid:
                peaks.append(indices)
        if peaks:
            windows[(segment, "finish")] = np.unique(np.concatenate(peaks))
    before = [ix for ix in stable if ix[-1] <= events[0][1]]
    after = [ix for ix in stable if ix[0] >= events[-1][3]]
    if before:
        windows[("stance", "start")] = before[-1]
    if after:
        windows[("stance", "end")] = after[0]
    else:
        windows.pop(("stance", "end"), None)
    return windows


def align_views(views: dict, windows: dict, cfg) -> dict:
    """由至少两个对应峰值估计一个小的固定时间偏移，拒绝漂移/错配。"""
    alignment = {"method": "video_timestamps", "sideOffsetSeconds": 0.0, "anchors": 0, "reliable": True}
    if set(views) != {"front", "side"}:
        alignment["reliable"] = False
        return alignment
    offsets = []
    for key in windows["front"].keys() & windows["side"].keys():
        if key[1] != "finish":
            continue
        front = contiguous_runs(windows["front"][key])
        side = contiguous_runs(windows["side"][key])
        if len(front) != len(side):
            alignment.update(reliable=False, reason="正侧面检测到的动作次数不一致")
            return alignment
        for f, s in zip(front, side):
            offsets.append(float(np.median(np.asarray(views["front"]["times"])[f]) - np.median(np.asarray(views["side"]["times"])[s])))
    alignment["anchors"] = len(offsets)
    if len(offsets) >= 2:
        offset = float(np.median(offsets))
        residual = float(np.max(np.abs(np.asarray(offsets) - offset)))
        if abs(offset) > cfg.phase_tolerance_seconds or residual > .1:
            alignment.update(reliable=False, reason="正侧面峰值时间无法稳定对应")
        else:
            alignment.update(method="action_peaks", sideOffsetSeconds=offset, residualSeconds=residual)
    return alignment


def share_phase_timing(views, windows, alignment, diagnostics, cfg):
    """同步录像可共用可见机位的时间段；不借用它的关键点或测量结果。"""
    sources = {view: "own_view" for view in views}
    if set(views) != {"front", "side"} or not alignment["reliable"]:
        return sources
    for target, source in (("front", "side"), ("side", "front")):
        if any(key[1] == "motion" for key in windows[target]) or diagnostics[target].get("events"):
            continue  # 已有相反的次数/顺序证据时不能覆盖它。
        if not any(key[1] == "motion" for key in windows[source]):
            continue
        start = ("stance", "start")
        if start not in windows[target] or start not in windows[source]:
            continue
        times = {v: np.asarray(views[v]["times"]) + (alignment["sideOffsetSeconds"] if v == "side" else 0)
                 for v in (target, source)}
        if abs(np.median(times[target][windows[target][start]]) - np.median(times[source][windows[source][start]])) > cfg.phase_tolerance_seconds:
            continue
        projected = {}
        for key, indices in windows[source].items():
            runs = []
            for run in contiguous_runs(indices):
                lo, hi = times[source][run[[0, -1]]]
                if lo < times[target][0] - 1e-6 or hi > times[target][-1] + 1e-6:
                    continue
                runs.append(np.flatnonzero((times[target] >= lo - 1e-6) & (times[target] <= hi + 1e-6)))
            if runs:
                projected[key] = np.unique(np.concatenate(runs))
        windows[target] = projected
        sources[target] = source + "_synchronized_timing"
    return sources
