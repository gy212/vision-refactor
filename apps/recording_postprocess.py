"""双摄录后处理：片段转码、录像与模板校验、调用考试池计分并写结果 JSON（P4 迁入 recording/finalizer）。"""
from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal


from core import action_compare, model_manager, video_writer
from core.media import first_frame_ok
from core.pose import missing_model_message
from core.foundation import atomic_write_json, to_jsonable
from core.paths import templates_dir
from core.evaluation import TemplateError, exam_pool
from core.pose import extract_pose_raw_series

PostprocessStatus = Literal[
    "queued",
    "transcoding",
    "validating",
    "comparing",
    "completed",
    "failed",
    "skipped",
    "cancelled",
]

_TERMINAL_STATUSES = frozenset({"completed", "failed", "skipped", "cancelled"})
_QUEUE_SENTINEL = object()
def utc_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def default_template_paths() -> tuple[Path, Path]:
    root = templates_dir()
    return root / "standard_front_heavy.npz", root / "standard_side_heavy.npz"


@dataclass(frozen=True)
class DualRecordingJob:
    segment_id: str
    segment_dir: Path
    front_source: Path
    side_source: Path
    front_frames: int
    side_frames: int
    front_template: Path
    side_template: Path
    # 多模板池（正对正、侧对侧）。为空时 __post_init__ 用单字段回填，保持旧构造兼容。
    front_templates: tuple[Path, ...] = ()
    side_templates: tuple[Path, ...] = ()
    record_skeleton: bool = False
    # True：录制后自动比对（检测一条龙）；False：仅录制/转码，跳过 DTW 比对。
    auto_compare: bool = True
    front_error: str | None = None
    side_error: str | None = None
    created_at: str = ""
    # 考试模式可选元数据（非考试保持 None，旧路径不变）
    exam_run_id: str | None = None
    student_id: str | None = None
    student_name: str | None = None
    student_order: int | None = None
    attempt_index: int | None = None
    exam_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.created_at:
            object.__setattr__(self, "created_at", utc_timestamp())
        # 复数池为空时用单字段回填（旧构造兼容）；单字段用池首元素回填（透传/日志兼容）。
        if not self.front_templates and self.front_template:
            object.__setattr__(self, "front_templates", (Path(self.front_template),))
        if not self.side_templates and self.side_template:
            object.__setattr__(self, "side_templates", (Path(self.side_template),))

@dataclass(frozen=True)
class PostprocessUpdate:
    segment_id: str
    status: PostprocessStatus
    message: str
    front_score: float | None = None
    side_score: float | None = None
    combined_percent: int | None = None
    error_code: str | None = None
    # 逐动作子分明细：({"name","view","score","start","end"}, ...)
    action_scores: tuple[dict[str, Any], ...] = ()
    # 转码后的最终视频路径（学生练习等消费方用；中间态也可带当前源）
    front_video: Path | None = None
    side_video: Path | None = None


@dataclass
class _ProcessContext:
    front_video: Path | None = None
    side_video: Path | None = None
    warnings: list[str] = field(default_factory=list)


class PostprocessError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


TranscodeFn = Callable[[Path, threading.Event], Path | None]
CompareFn = Callable[..., Any]
VideoValidator = Callable[[Path], bool]
ModelAvailable = Callable[[], bool]
UpdateCallback = Callable[[PostprocessUpdate], None]
SegmentDirectoryKey = tuple[str, int, int] | tuple[str, str]


def _default_transcode(path: Path, stop_evt: threading.Event) -> Path | None:
    return video_writer.transcode_to_h264(path, stop_evt=stop_evt)


def _default_video_validator(path: Path) -> bool:
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    return first_frame_ok(path)


def _default_model_available() -> bool:
    spec = next((item for item in model_manager.MEDIAPIPE_MODELS if item.key == "pose_heavy"), None)
    return bool(spec is not None and model_manager.is_installed(spec))


def _segment_directory_text_key(path: Path) -> SegmentDirectoryKey:
    text = str(path)
    if os.name == "nt":
        folded = text.casefold()
        if folded.startswith("\\\\?\\unc\\"):
            text = "\\\\" + text[8:]
        elif folded.startswith("\\\\?\\"):
            text = text[4:]
    return ("path", os.path.normcase(os.path.normpath(text)))


def _segment_directory_stat(path: Path) -> os.stat_result:
    return path.stat()


def _segment_directory_keys(path: Path) -> tuple[SegmentDirectoryKey, ...]:
    resolved = Path(path).resolve(strict=False)
    text_key = _segment_directory_text_key(resolved)
    try:
        stat = _segment_directory_stat(resolved)
    except OSError:
        return (text_key,)

    inode = int(getattr(stat, "st_ino", 0) or 0)
    if inode:
        return (text_key, ("stat", int(stat.st_dev), inode))
    return (text_key,)


def _segment_directory_key(path: Path) -> SegmentDirectoryKey:
    return _segment_directory_keys(path)[-1]


def _postprocess_error(exc: TemplateError) -> PostprocessError:
    return PostprocessError(exc.code, str(exc))


def try_load_heavy_template(path: Path | str | None) -> Path | None:
    """模板路径可用则返回规范化 Path，否则 None（缺文件/无效均视为未填入）。"""
    return exam_pool.usable_template(path)


def validate_template_pair(front_path: Path, side_path: Path) -> None:
    """考试等正式路径：正/侧模板必须同时可用且布局一致。"""
    try:
        exam_pool.check_pair(Path(front_path), Path(side_path))
    except TemplateError as exc:
        raise _postprocess_error(exc) from exc


def validate_auto_compare_templates(
    front_path: Path | str | None,
    side_path: Path | str | None,
) -> tuple[Path | None, Path | None]:
    """一条龙自动比对：至少一侧可用即可；两侧都缺才 ``template_missing``，两侧都可用时校验布局一致。"""
    front = try_load_heavy_template(front_path)
    side = try_load_heavy_template(side_path)
    if front is None and side is None:
        raise PostprocessError(
            "template_missing",
            "正/侧标准模板均未填入或不可用（heavy + pose33_v3）。"
            "请在主界面「录制」区填入至少一侧模板。",
        )
    if front is not None and side is not None:
        validate_template_pair(front, side)
    return front, side


def template_action_name(path: Path | str) -> str:
    """动作名 = 模板文件 stem（如 直拳.npz → "直拳"）。"""
    return Path(path).stem


def validate_auto_compare_template_lists(
    front_list: list[Path | str] | None,
    side_list: list[Path | str] | None,
) -> tuple[list[Path], list[Path]]:
    """多模板池校验：各自过滤出可用的 heavy+pose33_v3 模板（单个坏模板跳过）；两池都空才报错。"""
    try:
        return exam_pool.usable_pools(front_list, side_list)
    except TemplateError as exc:
        raise _postprocess_error(exc) from exc


class DualRecordingPostProcessor:
    def __init__(
        self,
        *,
        on_update: UpdateCallback | None = None,
        transcode: TranscodeFn = _default_transcode,
        compare: CompareFn = action_compare.compare_video_to_templates,
        video_validator: VideoValidator = _default_video_validator,
        model_available: ModelAvailable = _default_model_available,
    ) -> None:
        self._on_update = on_update
        self._transcode = transcode
        # 池级比对函数（可注入测试）：(template_paths, video, ...) -> [CompareResult]。
        # 视频姿态在其内部只提取一次,复用给池内每个模板做 DTW。
        self._compare_pool = compare
        self._video_validator = video_validator
        self._model_available = model_available
        self._queue: queue.Queue[DualRecordingJob | object] = queue.Queue()
        self._lock = threading.RLock()
        self._persistence_lock = threading.Lock()
        self._submitted: set[str] = set()
        self._segment_dir_owners: dict[SegmentDirectoryKey, str] = {}
        self._cancelled_segments: set[str] = set()
        self._terminal_status: dict[str, PostprocessStatus] = {}
        self._closed = False
        self._sentinel_queued = False
        self._active_stop_evt: threading.Event | None = None
        self._active_job: DualRecordingJob | None = None
        self._active_context: _ProcessContext | None = None
        self._worker = threading.Thread(
            target=self._run,
            name="dual-recording-postprocess",
            daemon=True,
        )
        self._worker.start()

    def submit(self, job: DualRecordingJob) -> bool:
        try:
            segment_dir_keys = _segment_directory_keys(job.segment_dir)
        except (OSError, RuntimeError):
            return False
        with self._lock:
            if self._closed or job.segment_id in self._submitted:
                return False
            if any(key in self._segment_dir_owners for key in segment_dir_keys):
                return False
            self._submitted.add(job.segment_id)
            for key in segment_dir_keys:
                self._segment_dir_owners[key] = job.segment_id
            self._queue.put(job)
        return True

    def cancel_all(self) -> None:
        with self._lock:
            self._closed = True
            active = self._active_stop_evt
            self._cancelled_segments.update(
                segment_id
                for segment_id in self._submitted
                if segment_id not in self._terminal_status
            )
        if active is not None:
            active.set()

    def close(self, timeout: float) -> None:
        self.cancel_all()
        with self._lock:
            if self._worker.is_alive() and not self._sentinel_queued:
                self._sentinel_queued = True
                self._queue.put(_QUEUE_SENTINEL)
        if threading.current_thread() is self._worker:
            return
        self._worker.join(max(0.0, float(timeout)))

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _QUEUE_SENTINEL:
                self._queue.task_done()
                return
            job = item
            stop_evt = threading.Event()
            context = _ProcessContext()
            with self._lock:
                closed = self._closed
                self._active_stop_evt = stop_evt
                self._active_job = job
                self._active_context = context
            try:
                if closed:
                    self._publish_terminal_safely(
                        job,
                        "cancelled",
                        message="应用关闭，后台比对已取消",
                        error_code="app_closing",
                    )
                else:
                    self._process(job, stop_evt, context)
            except InterruptedError:
                self._publish_cancelled(job, context)
            except PostprocessError as exc:
                if self._is_cancelled(job):
                    self._publish_cancelled(job, context)
                else:
                    self._publish_terminal_safely(
                        job,
                        "failed",
                        message=str(exc),
                        error_code=exc.code,
                        front_video=context.front_video,
                        side_video=context.side_video,
                        warnings=context.warnings,
                    )
            except Exception as exc:  # noqa: BLE001 - 后台任务必须失败隔离
                if self._is_cancelled(job):
                    self._publish_cancelled(job, context)
                else:
                    self._publish_terminal_safely(
                        job,
                        "failed",
                        message=f"后台比对失败：{exc}",
                        error_code="compare_failed",
                        front_video=context.front_video,
                        side_video=context.side_video,
                        warnings=context.warnings,
                    )
            finally:
                with self._lock:
                    if self._active_stop_evt is stop_evt:
                        self._active_stop_evt = None
                    if self._active_job is job:
                        self._active_job = None
                        self._active_context = None
                self._queue.task_done()

    def _is_cancelled(self, job: DualRecordingJob) -> bool:
        with self._lock:
            return job.segment_id in self._cancelled_segments

    def _publish_cancelled(
        self,
        job: DualRecordingJob,
        context: _ProcessContext,
    ) -> None:
        app_closing = self._is_cancelled(job)
        self._publish_terminal_safely(
            job,
            "cancelled",
            message=(
                "应用关闭，后台比对已取消" if app_closing else "后台比对已取消"
            ),
            error_code="app_closing" if app_closing else "cancelled",
            front_video=context.front_video,
            side_video=context.side_video,
            warnings=context.warnings,
        )

    def _process(
        self,
        job: DualRecordingJob,
        stop_evt: threading.Event,
        context: _ProcessContext,
    ) -> None:
        self._validate_recording_snapshot(job)
        self._publish(job, "queued", "已进入后台比对队列")
        self._publish(job, "transcoding", "正在顺序完成正侧录像转码")

        context.front_video = self._transcode_one(
            job.front_source, stop_evt, "正面", context.warnings
        )
        context.side_video = self._transcode_one(
            job.side_source, stop_evt, "侧面", context.warnings
        )

        self._raise_if_cancelled(stop_evt)
        self._publish(
            job,
            "validating",
            "正在校验录像、模板和模型",
            front_video=context.front_video,
            side_video=context.side_video,
            warnings=context.warnings,
        )
        self._validate_video(context.front_video, "正面")
        self._validate_video(context.side_video, "侧面")

        self._raise_if_cancelled(stop_evt)
        if job.record_skeleton:
            self._publish_terminal(
                job,
                "skipped",
                message="带骨架录像未自动比对，请关闭骨架后重新录制",
                error_code="annotated_recording",
                front_video=context.front_video,
                side_video=context.side_video,
                warnings=context.warnings,
            )
            return
        if not job.auto_compare:
            self._publish_terminal(
                job,
                "skipped",
                message="仅录制模式，已保存视频未自动比对",
                error_code="auto_compare_disabled",
                front_video=context.front_video,
                side_video=context.side_video,
                warnings=context.warnings,
            )
            return

        front_tpls, side_tpls = validate_auto_compare_template_lists(
            list(job.front_templates), list(job.side_templates)
        )
        if not self._model_available():
            raise PostprocessError(
                "model_missing",
                missing_model_message("postprocess_heavy"),
            )

        self._raise_if_cancelled(stop_evt)
        if front_tpls and side_tpls:
            compare_msg = f"正在执行正侧多模板 DTW 比对（正{len(front_tpls)}/侧{len(side_tpls)}）"
        elif front_tpls:
            compare_msg = f"正在执行正面多模板 DTW 比对（{len(front_tpls)} 个，侧面池为空）"
            context.warnings.append("侧面模板池为空，仅比对正面")
        else:
            compare_msg = f"正在执行侧面多模板 DTW 比对（{len(side_tpls)} 个，正面池为空）"
            context.warnings.append("正面模板池为空，仅比对侧面")
        self._publish(
            job,
            "comparing",
            compare_msg,
            front_video=context.front_video,
            side_video=context.side_video,
            warnings=context.warnings,
        )
        result = self._run_auto_compare(
            front_tpls=front_tpls,
            side_tpls=side_tpls,
            front_video=context.front_video,
            side_video=context.side_video,
            stop_evt=stop_evt,
        )
        self._raise_if_cancelled(stop_evt)
        result_payload = self._result_payload(result)
        done_msg = (
            "后台多模板比对完成"
            if front_tpls and side_tpls
            else "后台单侧多模板比对完成"
        )
        self._publish_terminal(
            job,
            "completed",
            message=done_msg,
            front_video=context.front_video,
            side_video=context.side_video,
            warnings=context.warnings,
            result=result_payload,
        )

    @staticmethod
    def _validate_recording_snapshot(job: DualRecordingJob) -> None:
        if job.front_error:
            raise PostprocessError("front_recording_failed", f"正面录像失败：{job.front_error}")
        if job.side_error:
            raise PostprocessError("side_recording_failed", f"侧面录像失败：{job.side_error}")
        if not job.front_source or not job.side_source:
            raise PostprocessError("recording_missing", "正面或侧面录像路径缺失")
        segment_dir = Path(job.segment_dir).resolve(strict=False)
        for label, source in (
            ("正面", job.front_source),
            ("侧面", job.side_source),
        ):
            try:
                Path(source).resolve(strict=False).relative_to(segment_dir)
            except (OSError, ValueError) as exc:
                raise PostprocessError(
                    "recording_path_mismatch",
                    f"{label}录像路径不属于当前片段目录：{source}",
                ) from exc
        if job.front_frames <= 0 or job.side_frames <= 0:
            raise PostprocessError("recording_empty", "正面或侧面录像没有有效帧")
        if job.front_frames != job.side_frames:
            raise PostprocessError(
                "frame_count_mismatch",
                f"正侧录像帧数不一致：front={job.front_frames} side={job.side_frames}",
            )

    def _transcode_one(
        self,
        source: Path,
        stop_evt: threading.Event,
        label: str,
        warnings: list[str],
    ) -> Path:
        self._raise_if_cancelled(stop_evt)
        source = Path(source)
        try:
            final_path = self._transcode(source, stop_evt)
        except InterruptedError:
            raise
        except Exception as exc:
            raise PostprocessError(
                "transcode_failed", f"{label}录像转码失败：{exc}"
            ) from exc
        if final_path is None:
            raise PostprocessError("transcode_failed", f"{label}录像转码未返回文件路径")
        final_path = Path(final_path)
        if source.suffix.lower() == ".avi" and final_path.suffix.lower() == ".avi":
            warnings.append(f"{label}录像 H.264 转码失败，已回退使用 AVI")
        return final_path

    def _validate_video(self, path: Path, label: str) -> None:
        try:
            valid = self._video_validator(Path(path))
        except Exception as exc:
            raise PostprocessError("video_unreadable", f"{label}录像无法读取：{exc}") from exc
        if not valid:
            raise PostprocessError("video_unreadable", f"{label}录像不存在、为空或无法读取：{path}")

    @staticmethod
    def _raise_if_cancelled(stop_evt: threading.Event) -> None:
        if stop_evt.is_set():
            raise InterruptedError("后台比对已取消")

    def _run_auto_compare(
        self,
        *,
        front_tpls: list[Path],
        side_tpls: list[Path],
        front_video: Path,
        side_video: Path,
        stop_evt: threading.Event,
    ) -> Any:
        """正对正池、侧对侧池逐模板比对（计分方案 ``EXAM_POOL``）；池级比对函数可注入测试。"""
        return exam_pool.score_exam_pools(
            front_tpls, side_tpls, front_video, side_video,
            compare=self._compare_pool, raw_series=extract_pose_raw_series, stop_evt=stop_evt,
        )

    @staticmethod
    def _result_payload(result: Any) -> dict[str, Any]:
        def _segment(value: Any) -> dict[str, int] | None:
            if value is None:
                return None
            return {"start": int(value[0]), "end": int(value[1])}

        def _opt_float(value: Any) -> float | None:
            if value is None:
                return None
            return float(value)

        return {
            "front_score": _opt_float(getattr(result, "front_score", None)),
            "side_score": _opt_float(getattr(result, "side_score", None)),
            "combined_score": float(result.combined_score),
            "combined_percent": int(result.combined_percent),
            "front_matches": to_jsonable(getattr(result, "front_matches", ())),
            "side_matches": to_jsonable(getattr(result, "side_matches", ())),
            "front_segment": _segment(getattr(result, "front_segment", None)),
            "side_segment": _segment(getattr(result, "side_segment", None)),
            "action_scores": to_jsonable(getattr(result, "action_scores", ())),
        }

    def _publish(
        self,
        job: DualRecordingJob,
        status: PostprocessStatus,
        message: str,
        *,
        front_video: Path | None = None,
        side_video: Path | None = None,
        warnings: list[str] | None = None,
        result: dict[str, Any] | None = None,
        error_code: str | None = None,
        notify: bool = True,
    ) -> None:
        exam_meta = None
        if any(
            getattr(job, key, None) is not None
            for key in (
                "exam_run_id",
                "student_id",
                "student_name",
                "student_order",
                "attempt_index",
            )
        ) or getattr(job, "exam_warnings", ()):
            exam_meta = {
                "exam_run_id": job.exam_run_id,
                "student_id": job.student_id,
                "student_name": job.student_name,
                "student_order": job.student_order,
                "attempt_index": job.attempt_index,
            }
        merged_warnings = list(warnings or [])
        for w in getattr(job, "exam_warnings", ()) or ():
            if w not in merged_warnings:
                merged_warnings.append(str(w))
        payload = {
            "schema_version": 1,
            "status": status,
            "segment_id": job.segment_id,
            "created_at": job.created_at,
            "completed_at": utc_timestamp() if status in _TERMINAL_STATUSES else None,
            "record_skeleton": bool(job.record_skeleton),
            "auto_compare": bool(job.auto_compare),
            "front_video_path": str(front_video or job.front_source),
            "side_video_path": str(side_video or job.side_source),
            "front_template_path": str(job.front_template),
            "side_template_path": str(job.side_template),
            "warnings": merged_warnings,
            "result": result,
            "error": None if error_code is None else {"code": error_code, "message": message},
            "exam": exam_meta,
        }
        update: PostprocessUpdate | None = None
        with self._persistence_lock:
            with self._lock:
                existing_terminal = self._terminal_status.get(job.segment_id)
                cancelled_enrichment = (
                    existing_terminal == "cancelled" and status == "cancelled"
                )
                if existing_terminal is not None and not cancelled_enrichment:
                    return
                if (
                    job.segment_id in self._cancelled_segments
                    and status != "cancelled"
                ):
                    raise InterruptedError("后台比对已取消")
            try:
                self._write_result_json(job.segment_dir / "result.json", payload)
            except Exception as exc:
                if cancelled_enrichment:
                    return
                raise PostprocessError(
                    "result_write_failed", f"结果文件写入失败：{exc}"
                ) from exc
            with self._lock:
                existing_terminal = self._terminal_status.get(job.segment_id)
                cancelled_enrichment = (
                    existing_terminal == "cancelled" and status == "cancelled"
                )
                if existing_terminal is not None and not cancelled_enrichment:
                    return
                if (
                    job.segment_id in self._cancelled_segments
                    and status != "cancelled"
                ):
                    raise InterruptedError("后台比对已取消")
                if status in _TERMINAL_STATUSES:
                    self._terminal_status[job.segment_id] = status
                if notify and not cancelled_enrichment:
                    update = self._update_from_payload(payload, message)
        if update is not None:
            self._notify(update)

    def _publish_terminal(
        self,
        job: DualRecordingJob,
        status: Literal["completed", "failed", "skipped", "cancelled"],
        *,
        message: str,
        error_code: str | None = None,
        front_video: Path | None = None,
        side_video: Path | None = None,
        warnings: list[str] | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        self._publish(
            job,
            status,
            message,
            front_video=front_video,
            side_video=side_video,
            warnings=warnings,
            result=result,
            error_code=error_code,
        )

    def _publish_terminal_safely(
        self,
        job: DualRecordingJob,
        status: Literal["completed", "failed", "skipped", "cancelled"],
        *,
        message: str,
        error_code: str | None = None,
        front_video: Path | None = None,
        side_video: Path | None = None,
        warnings: list[str] | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        try:
            self._publish_terminal(
                job,
                status,
                message=message,
                error_code=error_code,
                front_video=front_video,
                side_video=side_video,
                warnings=warnings,
                result=result,
            )
        except InterruptedError:
            try:
                self._publish_terminal(
                    job,
                    "cancelled",
                    message="应用关闭，后台比对已取消",
                    error_code="app_closing",
                    front_video=front_video,
                    side_video=side_video,
                    warnings=warnings,
                )
            except Exception as exc:  # noqa: BLE001 - 持久化失败不得杀死 FIFO worker
                self._notify(
                    PostprocessUpdate(
                        segment_id=job.segment_id,
                        status="failed",
                        message=f"结果文件写入失败：{exc}",
                        error_code="result_write_failed",
                    )
                )
        except Exception as exc:  # noqa: BLE001 - 持久化失败不得杀死 FIFO worker
            self._notify(
                PostprocessUpdate(
                    segment_id=job.segment_id,
                    status="failed",
                    message=f"结果文件写入失败：{exc}",
                    error_code="result_write_failed",
                )
            )

    @staticmethod
    def _write_result_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, payload, trailing_newline=True)

    @staticmethod
    def _update_from_payload(payload: dict[str, Any], message: str) -> PostprocessUpdate:
        result = payload.get("result") or {}
        error = payload.get("error") or {}
        front_raw = payload.get("front_video_path")
        side_raw = payload.get("side_video_path")
        front_video = Path(str(front_raw)) if front_raw else None
        side_video = Path(str(side_raw)) if side_raw else None
        return PostprocessUpdate(
            segment_id=str(payload["segment_id"]),
            status=payload["status"],
            message=message,
            front_score=result.get("front_score"),
            side_score=result.get("side_score"),
            combined_percent=result.get("combined_percent"),
            error_code=error.get("code"),
            action_scores=tuple(result.get("action_scores") or ()),
            front_video=front_video,
            side_video=side_video,
        )

    def _notify(self, update: PostprocessUpdate) -> None:
        if self._on_update is None:
            return
        try:
            self._on_update(update)
        except Exception:
            pass
