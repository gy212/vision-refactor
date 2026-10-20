# -*- coding: utf-8 -*-
"""录制性能测试：在实际机器上测各路 1080p 采集、写盘与识别同时运行时能否稳定 30 fps，只生成报告、不改任何设置（D-18、D-27）。

分四段依次运行，每段结束删除临时录像：

1. ``capture``：只采集，得到各路真实采集率与帧间隔；
2. ``record``：采集并各路一个写盘线程，写法与第 5 层的录像 writer 相同（每路一个有界队列、转正后写出）；
3. ``record_lite``：再加 Lite 站位检测（约每秒 10 次取最新主路画面，D-30）；
4. ``record_skeleton``：再加骨架识别（各路最新画面逐路推理并绘制，和普通模式骨架开时的处理循环相当）。

写盘队列按 16 帧开（内存有界），同时统计"到达时已有 8 帧排队"的次数：按生产的 8 帧容量运行时，
这些帧会让该段录制失败（D-27）。阈值判定见 ``phase_verdict``。
"""
from __future__ import annotations

import shutil
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from core.foundation import (
    CancelToken,
    Cancelled,
    TaskGroup,
    atomic_write_json,
    cpu_percent,
    host_summary,
    system_cpu_times,
)
from core.media import (
    PRIMARY,
    CameraStream,
    FrameQueue,
    apply_rotation,
    open_video_writer,
    rotated_size,
    view_for_role,
)
from core.pose import EngineSpec

TARGET_FPS = 30.0
PRODUCTION_QUEUE = 8           # 第 5 层录像队列的容量（D-27）
BENCH_QUEUE = 16               # 测试用的队列容量：两路 1080p 约 200 MB 上限
MIN_CAPTURE_FPS = 28.5         # 判定"采集达到 30 fps"的下限
LITE_INTERVAL_S = 0.1
REPORT_SCHEMA = 1
REPORT_PREFIX = "录制性能测试"

EngineFactory = Callable[[EngineSpec], Any]
Progress = Callable[[str, float], None]   # (说明文字, 总进度 0..1)


@dataclass(frozen=True)
class Phase:
    key: str
    label: str
    seconds: float
    write: bool = False
    lite: bool = False
    skeleton: bool = False


PHASES: tuple[Phase, ...] = (
    Phase("capture", "只采集", 10.0),
    Phase("record", "采集 + 写盘", 30.0, write=True),
    Phase("record_lite", "采集 + 写盘 + 站位检测（Lite）", 30.0, write=True, lite=True),
    Phase("record_skeleton", "采集 + 写盘 + 骨架识别", 30.0, write=True, skeleton=True),
)


@dataclass
class BenchmarkSetup:
    """一次测试的输入。``stream`` 未接管；``rotations``、``labels`` 按角色；``work_dir`` 放临时录像（测完删除）。"""

    stream: CameraStream
    rotations: Mapping[str, int]
    labels: Mapping[str, str]
    work_dir: Path
    engine_factory: EngineFactory
    pose_variant: str = "full"
    enable_hands: bool = True
    phases: tuple[Phase, ...] = PHASES
    extra: Mapping[str, Any] = field(default_factory=dict)


def _ms_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "p50": round(statistics.median(ordered), 2),
        "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
        "max": round(ordered[-1], 2),
    }


def _rate(count: int, seconds: float) -> float | None:
    return round(count / seconds, 2) if seconds > 0 else None


# --------------------------------------------------------------------------- #
# 各段的工作线程
# --------------------------------------------------------------------------- #


def _count_frames(queue: FrameQueue, stamps: list[int], token: CancelToken) -> None:
    """只记采集时刻（不持有像素），得到真实采集率与帧间隔。"""
    while not token.cancelled:
        frame = queue.get(0.2)
        if frame is None:
            if queue.closed:
                return
            continue
        stamps.append(frame.captured_at_ns)


def _open_writer(path: Path, size: tuple[int, int], result: dict) -> Any | None:
    """先打开 writer 再订阅帧（打开要几百毫秒，期间到达的帧会把队列排满）；记打开耗时与编码。"""
    result.update(write_ms=[], stamps=[])
    started = time.perf_counter()
    try:
        writer, actual, codec = open_video_writer(path, fps=TARGET_FPS, size=size)
    except Exception as exc:
        result["error"] = f"打开写出失败：{exc}"
        return None
    result.update(codec=codec, open_ms=round((time.perf_counter() - started) * 1000.0, 1), actual=actual)
    return writer


def _write_frames(queue: FrameQueue, writer: Any, rotation: int, result: dict, token: CancelToken) -> None:
    """按序取帧、转正、写出（第 5 层录像 writer 的写法），记每帧耗时；出错时记下原因并停止取帧。"""
    actual: Path = result["actual"]
    try:
        while not token.cancelled:
            frame = queue.get(0.2)
            if frame is None:
                if queue.closed:
                    return
                continue
            started = time.perf_counter()
            writer.write(apply_rotation(frame.bgr, rotation))
            result["write_ms"].append((time.perf_counter() - started) * 1000.0)
            result["stamps"].append(frame.captured_at_ns)
    except Exception as exc:
        result["error"] = f"写出失败：{exc}"
    finally:
        writer.release()
        result["bytes"] = actual.stat().st_size if actual.is_file() else 0


def _run_lite(stream: CameraStream, role: str, rotation: int, engine: Any, result: dict,
              token: CancelToken) -> None:
    """约每秒 10 次取最新主路画面做 Lite 推理（站位检测的负载）。"""
    result["infer_ms"] = []
    next_at = time.perf_counter()
    try:
        while not token.cancelled:
            frame = stream.latest(role)
            if frame is not None:
                started = time.perf_counter()
                engine.infer(apply_rotation(frame.bgr, rotation),
                             timestamp_ms=engine.next_timestamp_ms(is_file=False, fps_for_ts=TARGET_FPS))
                result["infer_ms"].append((time.perf_counter() - started) * 1000.0)
            next_at += LITE_INTERVAL_S
            token.wait(max(0.0, next_at - time.perf_counter()))
    except Exception as exc:
        result["error"] = f"Lite 推理失败：{exc}"


def _run_skeleton(stream: CameraStream, rotations: Mapping[str, int], engines: Mapping[str, Any],
                  result: dict, token: CancelToken) -> None:
    """各路最新画面逐路推理并绘制（普通模式骨架开时处理循环的负载）。"""
    result["loop_ms"] = []
    try:
        while not token.cancelled:
            started = time.perf_counter()
            for role, engine in engines.items():
                frame = stream.latest(role)
                if frame is None:
                    continue
                image = apply_rotation(frame.bgr, rotations.get(role, 0))
                pose, hands = engine.infer(
                    image, timestamp_ms=engine.next_timestamp_ms(is_file=False, fps_for_ts=TARGET_FPS))
                engine.draw(image.copy(), pose, hands)
            result["loop_ms"].append((time.perf_counter() - started) * 1000.0)
    except Exception as exc:
        result["error"] = f"骨架推理失败：{exc}"


# --------------------------------------------------------------------------- #
# 一段
# --------------------------------------------------------------------------- #


def _interval_stats(stamps: list[int]) -> dict[str, Any]:
    gaps = [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:])]
    seconds = (stamps[-1] - stamps[0]) / 1e9 if len(stamps) > 1 else 0.0
    return {"frames": len(stamps), "fps": _rate(len(stamps) - 1, seconds), "interval_ms": _ms_stats(gaps)}


def _start_phase_workers(setup: BenchmarkSetup, phase: Phase, sizes: Mapping[str, tuple[int, int]],
                         engines: Mapping[str, Any], tasks: TaskGroup, phase_dir: Path) -> dict:
    stream, state = setup.stream, {"roles": {}, "queues": {}}
    for role, size in sizes.items():
        rotation = setup.rotations.get(role, 0)
        role_state: dict[str, Any] = {"stamps": []}
        state["roles"][role] = role_state
        counter = stream.subscribe(role, 64)
        state["queues"][(role, "count")] = counter
        tasks.spawn(f"count-{role}", lambda t, q=counter, s=role_state["stamps"]: _count_frames(q, s, t),
                    owner="benchmark")
        if phase.write:
            result = role_state["writer"] = {}
            writer = _open_writer(phase_dir / f"{view_for_role(role)}.mp4", rotated_size(*size, rotation), result)
            queue = stream.subscribe(role, BENCH_QUEUE, watermark=PRODUCTION_QUEUE)
            state["queues"][(role, "write")] = queue
            if writer is not None:
                tasks.spawn(f"write-{role}", lambda t, q=queue, v=writer, r=rotation, w=result:
                            _write_frames(q, v, r, w, t), owner="benchmark")
    if phase.lite:
        state["lite"] = {}
        tasks.spawn("lite", lambda t: _run_lite(stream, PRIMARY, setup.rotations.get(PRIMARY, 0),
                                                engines["lite"], state["lite"], t), owner="benchmark")
    if phase.skeleton:
        state["skeleton"] = {}
        pose_engines = {role: engines[role] for role in sizes}
        tasks.spawn("skeleton", lambda t: _run_skeleton(stream, setup.rotations, pose_engines, state["skeleton"], t),
                    owner="benchmark")
    return state


def _wait_phase(phase: Phase, index: int, total: int, progress: Progress, cancel: CancelToken) -> None:
    started = time.monotonic()
    while (elapsed := time.monotonic() - started) < phase.seconds:
        cancel.raise_if_cancelled()
        done = (index + elapsed / phase.seconds) / total
        progress(f"第 {index + 1}/{total} 段：{phase.label}（{int(phase.seconds - elapsed)} 秒）", done)
        cancel.wait(min(0.5, phase.seconds - elapsed))


def _collect_phase(phase: Phase, state: dict) -> dict[str, Any]:
    result: dict[str, Any] = {"key": phase.key, "label": phase.label, "seconds": phase.seconds, "roles": {}}
    for role, role_state in state["roles"].items():
        entry = {"capture": _interval_stats(role_state["stamps"])}
        writer = role_state.get("writer")
        if writer is not None:
            queue = state["queues"][(role, "write")]
            written = len(writer.get("write_ms", []))
            stamps = writer.get("stamps", [])
            seconds = (stamps[-1] - stamps[0]) / 1e9 if len(stamps) > 1 else 0.0
            entry["write"] = {
                "error": writer.get("error"),
                "codec": writer.get("codec"),
                "open_ms": writer.get("open_ms"),
                "frames": written,
                "write_ms": _ms_stats(writer.get("write_ms", [])),
                "queue_peak": queue.peak,
                "over_production_queue": queue.over_watermark,
                "dropped": queue.overflows,
                "mb_per_s": round(writer.get("bytes", 0) / 2**20 / seconds, 1) if seconds > 0 else None,
            }
        result["roles"][view_for_role(role)] = entry
    if "lite" in state:
        times = state["lite"].get("infer_ms", [])
        result["lite"] = {"runs": len(times), "per_s": _rate(len(times), phase.seconds), "infer_ms": _ms_stats(times),
                          "error": state["lite"].get("error")}
    if "skeleton" in state:
        times = state["skeleton"].get("loop_ms", [])
        result["skeleton"] = {"loops": len(times), "fps": _rate(len(times), phase.seconds), "loop_ms": _ms_stats(times),
                              "error": state["skeleton"].get("error")}
    return result


def run_phase(setup: BenchmarkSetup, phase: Phase, index: int, sizes: Mapping[str, tuple[int, int]],
              engines: Mapping[str, Any], progress: Progress, cancel: CancelToken) -> dict[str, Any]:
    """运行一段并返回它的结果；临时录像在段末删除。"""
    phase_dir = setup.work_dir / phase.key
    phase_dir.mkdir(parents=True, exist_ok=True)
    tasks = TaskGroup(f"benchmark-{phase.key}")
    cpu_start = system_cpu_times()
    state = _start_phase_workers(setup, phase, sizes, engines, tasks, phase_dir)
    try:
        _wait_phase(phase, index, len(setup.phases), progress, cancel)
    finally:
        cpu_end = system_cpu_times()
        for (role, _kind), queue in state["queues"].items():
            setup.stream.unsubscribe(role, queue)
        tasks.cancel(reason="phase_end")
        tasks.join_until(time.monotonic() + 10.0)
        shutil.rmtree(phase_dir, ignore_errors=True)
    result = _collect_phase(phase, state)
    result["system_cpu_percent"] = cpu_percent(cpu_start, cpu_end)
    result["verdict"] = phase_verdict(result)
    return result


def phase_verdict(result: Mapping[str, Any]) -> dict[str, Any]:
    """这一段是否达标：各路采集不低于 ``MIN_CAPTURE_FPS``，写盘时没有一帧遇到 8 帧排队（否则生产配置会判录制失败）。"""
    problems: list[str] = []
    for view, entry in result["roles"].items():
        fps = entry["capture"]["fps"]
        if fps is None or fps < MIN_CAPTURE_FPS:
            problems.append(f"{view} 采集 {fps} fps，低于 {MIN_CAPTURE_FPS}")
        write = entry.get("write")
        if write is not None and write.get("error"):
            problems.append(f"{view} {write['error']}")
        elif write is not None and (write["over_production_queue"] or write["dropped"]):
            problems.append(f"{view} 写盘跟不上：{write['over_production_queue']} 帧遇到 {PRODUCTION_QUEUE} 帧排队")
    for key in ("lite", "skeleton"):
        if result.get(key, {}).get("error"):
            problems.append(result[key]["error"])
    return {"ok": not problems, "problems": problems}


# --------------------------------------------------------------------------- #
# 整体
# --------------------------------------------------------------------------- #


def _open_engines(setup: BenchmarkSetup, roles: tuple[str, ...], progress: Progress) -> dict[str, Any]:
    engines: dict[str, Any] = {}
    if any(phase.lite for phase in setup.phases):
        progress("正在加载 Lite 模型…", 0.0)
        engines["lite"] = setup.engine_factory(EngineSpec(pose_variant="lite", running_mode="video",
                                                          enable_hands=False))
    if any(phase.skeleton for phase in setup.phases):
        progress(f"正在加载骨架模型（{setup.pose_variant}）…", 0.0)
        for role in roles:
            engines[role] = setup.engine_factory(EngineSpec(pose_variant=setup.pose_variant,
                                                            enable_hands=setup.enable_hands))
    return engines


def _camera_entries(setup: BenchmarkSetup, sizes: Mapping[str, tuple[int, int]]) -> dict[str, Any]:
    specs = setup.stream.specs
    return {
        view_for_role(role): {
            "label": setup.labels.get(role, ""),
            "index": specs[role].index,
            "requested": list(specs[role].size),
            "actual": list(size),
            "driver_fps": setup.stream.fps(role),
            "rotation": setup.rotations.get(role, 0),
        }
        for role, size in sizes.items()
    }


def summary_lines(report: Mapping[str, Any]) -> list[str]:
    """给测试人员看的几行结论。"""
    names = {"front": "正面", "side": "侧面"}
    lines = []
    for phase in report.get("phases", []):
        rates = "，".join(f"{names.get(view, view)} {entry['capture']['fps']} fps"
                         for view, entry in phase["roles"].items())
        verdict = "达标" if phase["verdict"]["ok"] else "不达标：" + "；".join(phase["verdict"]["problems"])
        lines.append(f"{phase['label']}：{rates}（{verdict}）")
    if report.get("error"):
        lines.append(f"测试中断：{report['error']}")
    return lines


def run_recording_benchmark(setup: BenchmarkSetup, cancel: CancelToken, progress: Progress) -> dict[str, Any]:
    """接管摄像头、加载模型、依次运行各段，返回报告字典（出错或取消时也返回已完成的部分，并写明原因）。"""
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "kind": "recording_benchmark",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_fps": TARGET_FPS,
        "production_queue": PRODUCTION_QUEUE,
        "host": host_summary(),
        **dict(setup.extra),
        "phases": [],
    }
    engines: dict[str, Any] = {}
    stream = setup.stream
    try:
        progress("正在打开摄像头…", 0.0)
        stream.wait_ready(cancel)
        stream.claim(cancel)
        sizes = stream.actual_sizes()
        report["cameras"] = _camera_entries(setup, sizes)
        engines = _open_engines(setup, tuple(sizes), progress)
        for index, phase in enumerate(setup.phases):
            report["phases"].append(run_phase(setup, phase, index, sizes, engines, progress, cancel))
        report["ok"] = all(phase["verdict"]["ok"] for phase in report["phases"])
    except Cancelled:
        report["ok"], report["error"] = False, "已取消"
    except Exception as exc:
        report["ok"], report["error"] = False, f"{type(exc).__name__}: {exc}"
    finally:
        for engine in engines.values():
            try:
                engine.close()
            except Exception:
                pass
        try:
            stream.close(keep_warm=True)
        except Exception:
            pass
        shutil.rmtree(setup.work_dir, ignore_errors=True)
    report["summary"] = summary_lines(report)
    progress("测试完成" if report.get("ok") else "测试结束", 1.0)
    return report


def save_report(report: Mapping[str, Any], directory: Path, now: datetime | None = None) -> Path:
    """把报告写成 ``<directory>/录制性能测试_<YYYYMMDD_HHMMSS>.json``（UTF-8，原子写出），返回完整路径。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{REPORT_PREFIX}_{(now or datetime.now()).strftime('%Y%m%d_%H%M%S')}.json"
    atomic_write_json(path, dict(report), indent=2, trailing_newline=True)
    return path


__all__ = [
    "BenchmarkSetup",
    "PHASES",
    "PRODUCTION_QUEUE",
    "Phase",
    "phase_verdict",
    "run_phase",
    "REPORT_PREFIX",
    "run_recording_benchmark",
    "save_report",
    "summary_lines",
]
