# -*- coding: utf-8 -*-
"""录制性能测试：在实际机器上测各路 1080p 采集、写盘与识别同时运行时能否稳定 30 fps，只生成报告、不改任何设置（D-18、D-27）。

分四段依次运行，每段结束删除临时录像：

1. ``capture``：只采集，得到各路真实采集率与帧间隔；
2. ``record``：采集并各路一个写盘线程，写法与第 6 层的录像 writer 相同（每路一个有界队列、转正后写出）；
3. ``record_lite``：再加 Lite 站位检测（约每秒 10 次取最新主路画面，D-30）；
4. ``record_skeleton``：再加骨架识别（各路最新画面逐路推理并绘制，和普通模式骨架开时的处理循环相当）。

写盘队列按 16 帧开（内存有界），同时统计"到达时已有 8 帧排队"的次数：按生产的 8 帧容量运行时，
这些帧会让该段录制失败（D-27）。writer 的打开与生产的录像线程相同：先订阅、帧已经在流动，收到首帧后按它转正后的
尺寸在任务线程里打开（``WriterOpener``），期间写盘线程照常取帧、按序积压，打开后每轮先取一帧再写一帧积压；
报告记下编码、打开耗时、打开完成时的积压与追平耗时。打开超过生产上限（``PRODUCTION_OPEN_BUDGET_S``）时生产会判
开录失败，这一段也判不达标。阈值判定见 ``phase_verdict``。
"""
from __future__ import annotations

import math
import shutil
import statistics
import time
from collections import deque
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
    CapturedFrame,
    FrameQueue,
    WriterOpener,
    apply_rotation,
    capture_clock_ns,
    open_video_writer,
    rotated_size,
    view_for_role,
)
from core.pose import EngineSpec

TARGET_FPS = 30.0
PRODUCTION_QUEUE = 8           # 第 6 层录像队列的容量（D-27）
PRODUCTION_OPEN_BUDGET_S = 1.5  # 第 6 层后台打开 writer 的上限（``WRITER_OPEN_TIMEOUT_S``），超过即开录失败
BACKLOG_LIMIT = math.ceil(PRODUCTION_OPEN_BUDGET_S * TARGET_FPS)  # 生产允许的开录积压帧数
BENCH_QUEUE = 16               # 测试用的队列容量：两路 1080p 约 200 MB 上限
MIN_CAPTURE_FPS = 28.5         # 判定"采集达到 30 fps"的下限
LITE_INTERVAL_S = 0.1
STOP_TOLERANCE_S = 0.5         # 最后一帧离段末超过它即判该路中途停止（甲方实测最大帧间隔 85 ms）
REPORT_SCHEMA = 1
REPORT_PREFIX = "录制性能测试"

EngineFactory = Callable[[EngineSpec], Any]
# writer 工厂（与 ``RecordingController`` 的 ``writer_factory`` 同签名）：给定目标路径 / fps / size，
# 返回 (writer, 实际落盘路径, codec 标签)；默认绑定 ``open_video_writer``。
WriterFactory = Callable[[Path, float, tuple[int, int]], tuple[Any, Path, str]]
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


def _default_writer_factory(path: Path, fps: float, size: tuple[int, int]) -> tuple[Any, Path, str]:
    """默认 writer 工厂：绑定 ``open_video_writer``（生产录像用的同一个）。"""
    return open_video_writer(path, fps=fps, size=size)


@dataclass
class BenchmarkSetup:
    """一次测试的输入。``stream`` 未接管；``rotations``、``labels`` 按角色；``work_dir`` 放临时录像（测完删除）；
    ``writer_factory`` 打开各路 writer，默认 ``open_video_writer``。"""

    stream: CameraStream
    rotations: Mapping[str, int]
    labels: Mapping[str, str]
    work_dir: Path
    engine_factory: EngineFactory
    pose_variant: str = "full"
    enable_hands: bool = True
    phases: tuple[Phase, ...] = PHASES
    extra: Mapping[str, Any] = field(default_factory=dict)
    writer_factory: WriterFactory = _default_writer_factory


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


def _start_open(factory: WriterFactory, path: Path, frame: CapturedFrame, rotation: int,
                tasks: TaskGroup) -> WriterOpener:
    """按首帧转正后的尺寸在任务线程里打开 writer（与生产的录像线程相同）。"""
    height, width = frame.bgr.shape[:2]
    size = rotated_size(int(width), int(height), rotation)
    return WriterOpener(lambda: factory(path, TARGET_FPS, size), tasks,
                        name=f"open-{path.stem}", owner="benchmark", close_budget_s=10.0)


def _take_writer(opener: WriterOpener, backlog: deque, result: dict) -> Any | None:
    """打开完成时记下编码、打开耗时与此刻的积压，返回 writer；仍在打开返回 None；失败抛出。"""
    opened = opener.take(0.0)
    if opened is None:
        return None
    result.update(codec=opened.codec, open_ms=opened.open_ms, actual=opened.path, open_backlog=len(backlog),
                  opened_at=time.perf_counter())
    return opened.writer


def _write_one(writer: Any, frame: CapturedFrame, rotation: int, result: dict) -> None:
    started = time.perf_counter()
    writer.write(apply_rotation(frame.bgr, rotation))
    result["write_ms"].append((time.perf_counter() - started) * 1000.0)
    result["stamps"].append(frame.captured_at_ns)


def _write_frames(queue: FrameQueue, factory: WriterFactory, path: Path, rotation: int, tasks: TaskGroup,
                  result: dict, token: CancelToken) -> None:
    """按序取帧、转正、写出（第 6 层录像线程的写法）：首帧到达后在后台打开 writer，打开期间按序积压
    （超过生产允许的帧数就丢弃并计数），打开后每轮先取一帧再写一帧积压；记每帧耗时，出错时记下原因并停止取帧。"""
    result.update(write_ms=[], stamps=[], backlog_peak=0, backlog_dropped=0)
    backlog: deque[CapturedFrame] = deque()
    opener: WriterOpener | None = None
    writer: Any | None = None
    try:
        while not token.cancelled:
            frame = queue.get(0.0 if writer is not None and backlog else 0.2)
            if frame is not None:
                opener = opener or _start_open(factory, path, frame, rotation, tasks)
                if len(backlog) < BACKLOG_LIMIT:
                    backlog.append(frame)
                    result["backlog_peak"] = max(result["backlog_peak"], len(backlog))
                else:
                    result["backlog_dropped"] += 1
            elif queue.closed and (writer is None or not backlog):
                return
            if writer is None and opener is not None:
                writer = _take_writer(opener, backlog, result)
            if writer is not None and backlog:
                _write_one(writer, backlog.popleft(), rotation, result)
                if not backlog and "catch_up_ms" not in result:
                    result["catch_up_ms"] = round((time.perf_counter() - result["opened_at"]) * 1000.0, 1)
    except Exception as exc:
        result["error"] = f"打开录像文件失败：{exc}" if writer is None else f"写出失败：{exc}"
    finally:
        _close_writer(writer, opener, result)


def _close_writer(writer: Any | None, opener: WriterOpener | None, result: dict) -> None:
    """释放 writer（没打开完就放弃，由打开任务释放）；记下文件大小，没打开完时记下原因。"""
    if writer is not None:
        writer.release()
    elif opener is not None:
        opener.abandon()
        result.setdefault("error", f"打开录像文件 {opener.elapsed_s():.1f} 秒仍未完成")
    actual: Path | None = result.get("actual")
    result["bytes"] = actual.stat().st_size if actual is not None and actual.is_file() else 0


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


def _capture_stats(stamps: list[int], window: tuple[int, int], failure: BaseException | None) -> dict[str, Any]:
    """一路在本段窗口 ``(开始, 结束)`` 内的采集：采集率按整段窗口算（中途停止的那路不会因剩下的帧间隔正常而达标），
    ``covered_s`` 是最后一帧离段首的秒数，``error`` 是 reader 停止的原因。"""
    start, end = window
    inside = [t for t in stamps if start < t <= end]
    gaps = [(b - a) / 1e6 for a, b in zip(inside, inside[1:])]
    seconds = (end - start) / 1e9
    return {
        "frames": len(inside),
        "fps": _rate(len(inside), seconds),
        "interval_ms": _ms_stats(gaps),
        "covered_s": round(((inside[-1] if inside else start) - start) / 1e9, 2),
        "error": None if failure is None else str(failure),
    }


def _start_phase_workers(setup: BenchmarkSetup, phase: Phase, sizes: Mapping[str, tuple[int, int]],
                         engines: Mapping[str, Any], tasks: TaskGroup, phase_dir: Path) -> dict:
    stream, state = setup.stream, {"roles": {}, "queues": {}, "window": [capture_clock_ns(), None]}
    for role in sizes:                       # 先让各路都开始计数，段的窗口对各路相同
        role_state: dict[str, Any] = {"stamps": []}
        state["roles"][role] = role_state
        counter = stream.subscribe(role, 64)
        state["queues"][(role, "count")] = counter
        tasks.spawn(f"count-{role}", lambda t, q=counter, s=role_state["stamps"]: _count_frames(q, s, t),
                    owner="benchmark")
    for role in sizes:
        rotation, role_state = setup.rotations.get(role, 0), state["roles"][role]
        if phase.write:
            result = role_state["writer"] = {}
            path = phase_dir / f"{view_for_role(role)}.mp4"
            queue = stream.subscribe(role, BENCH_QUEUE, watermark=PRODUCTION_QUEUE)
            state["queues"][(role, "write")] = queue
            tasks.spawn(f"write-{role}", lambda t, q=queue, p=path, r=rotation, w=result:
                        _write_frames(q, setup.writer_factory, p, r, tasks, w, t), owner="benchmark")
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
    window = tuple(state["window"])
    for role, role_state in state["roles"].items():
        entry = {"capture": _capture_stats(role_state["stamps"], window, state["failures"].get(role))}
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
                "open_backlog": writer.get("open_backlog"),
                "backlog_peak": writer.get("backlog_peak", 0),
                "backlog_dropped": writer.get("backlog_dropped", 0),
                "catch_up_ms": writer.get("catch_up_ms"),
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
        state["window"][1] = capture_clock_ns()
        state["failures"] = {role: setup.stream.failure(role) for role in sizes}
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
    """这一段是否达标：各路采集一直持续到段末、整段不低于 ``MIN_CAPTURE_FPS``，写盘时没有一帧遇到 8 帧排队，
    打开 writer 不超过 ``PRODUCTION_OPEN_BUDGET_S``、开录积压不超过 ``BACKLOG_LIMIT``（否则生产配置会判录制失败）。"""
    problems: list[str] = []
    for view, entry in result["roles"].items():
        capture = entry["capture"]
        fps = capture["fps"]
        if capture.get("error"):
            problems.append(f"{view} 摄像头在第 {capture['covered_s']} 秒后停止出帧：{capture['error']}")
        elif capture["covered_s"] < result["seconds"] - STOP_TOLERANCE_S:
            problems.append(f"{view} 摄像头在第 {capture['covered_s']} 秒后不再出帧")
        if fps is None or fps < MIN_CAPTURE_FPS:
            problems.append(f"{view} 采集 {fps} fps，低于 {MIN_CAPTURE_FPS}")
        write = entry.get("write")
        if write is not None and write.get("error"):
            problems.append(f"{view} {write['error']}")
        elif write is not None and (write["over_production_queue"] or write["dropped"]):
            problems.append(f"{view} 写盘跟不上：{write['over_production_queue']} 帧遇到 {PRODUCTION_QUEUE} 帧排队")
        if write is not None:
            problems.extend(_open_problems(view, write))
    for key in ("lite", "skeleton"):
        if result.get(key, {}).get("error"):
            problems.append(result[key]["error"])
    return {"ok": not problems, "problems": problems}


def _open_problems(view: str, write: Mapping[str, Any]) -> list[str]:
    """打开 writer 超过生产上限、开录积压超过生产允许的帧数时，生产会判开录失败。"""
    problems = []
    open_ms = write.get("open_ms")
    if open_ms is not None and open_ms > PRODUCTION_OPEN_BUDGET_S * 1000.0:
        problems.append(f"{view} 打开录像文件用了 {open_ms:g} ms（{write.get('codec')}），"
                        f"超过生产上限 {PRODUCTION_OPEN_BUDGET_S * 1000.0:g} ms，录制开始时会失败")
    if write.get("backlog_dropped"):
        problems.append(f"{view} 开录积压超过 {BACKLOG_LIMIT} 帧（{write['backlog_dropped']} 帧放不下）")
    return problems


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
        writers = "、".join(f"{names.get(view, view)} {entry['write']['codec']} 打开 {entry['write']['open_ms']} ms"
                            for view, entry in phase["roles"].items() if "write" in entry)
        if writers:
            rates += f"；编码 {writers}"
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
        "production_open_budget_ms": PRODUCTION_OPEN_BUDGET_S * 1000.0,
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
    "BACKLOG_LIMIT",
    "BenchmarkSetup",
    "PHASES",
    "PRODUCTION_OPEN_BUDGET_S",
    "PRODUCTION_QUEUE",
    "Phase",
    "phase_verdict",
    "run_phase",
    "REPORT_PREFIX",
    "run_recording_benchmark",
    "save_report",
    "summary_lines",
]
