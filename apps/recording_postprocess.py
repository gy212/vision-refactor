from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import cv2
import numpy as np

from analysis.tech_eval import to_jsonable
from core import action_compare, model_manager, video_writer
from core.kick_quality import score_kick_quality
from core.paths import repo_root, templates_dir
from core.rule_scoring import extract_pose_raw_series, slice_pose_raw_series

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
# 黑盒检测（录制后自动比对）统一走精度最高的 heavy + pose33_v3 链路。
_EXPECTED_TEMPLATE_LAYOUT = "pose33_v3"
_EXPECTED_TEMPLATE_POSE_VARIANT = "heavy"
_PREFS_FRONT_KEY = "auto_compare_front_template"
_PREFS_SIDE_KEY = "auto_compare_side_template"
# 多模板池（正对正、侧对侧,每边可多个动作模板）。缺省回落旧单值键。
_PREFS_FRONT_LIST_KEY = "auto_compare_front_templates"
_PREFS_SIDE_LIST_KEY = "auto_compare_side_templates"

# 踢腿几何质量评分：动作名（模板 stem）→ kick_quality 类型。列出的动作并入几何分。
# 低鞭腿不列入——实测其低/高鞭高度差不稳定,几何常误判,直接用纯 DTW 分。
_KICK_KINDS = {
    "高鞭腿": "high_whip",
    "侧踹腿": "kick_up",
    "正蹬腿": "kick_up",
}
# 踢腿动作子分 = 几何高度分 × w + DTW 相似度分 × (1-w)。校准旋钮。
KICK_GEOM_WEIGHT = 0.5

# DTW 分压缩映射（opt-in）：动作名（模板 stem）→ (下限, 上限)。
# 实测某些动作 DTW 定位不可靠（如背后七颠：低位移提踵,归一化特征抹掉高度,
# subsequence_dtw 开放边界退化匹配极短片段,虚高分与动作质量无关）。这类动作
# 考察不严「是人就过」,把 DTW 分线性压到高分窄区间：人人及格偏上,但保留同一条
# 视频恒定、可复现的微弱区分度(比固定分好、比随机数可复现且公平)。
# 未登记动作不受影响(散打零回归)。
_SCORE_SQUEEZE: dict[str, tuple[float, float]] = {
    "背后七颠": (0.75, 0.90),  # DTW 不可靠,压到 [0.75,0.90] 人人过、弱区分、可复现
}

# 同一动作多视角（正面/侧面）按权重合并成一个动作子分（opt-in，八段锦用）。
# key = 合并后动作名；value = {"front": 正面权重, "side": 侧面权重}（自动归一）。
# 未登记的动作保持池级各自出分（散打行为不变：模板 stem 无 _正面/_侧面 后缀、不在表内）。
# 触发条件：正池存在 "<key>_正面"、侧池存在 "<key>_侧面"，两者都在才合并。
_VIEW_WEIGHTS: dict[str, dict[str, float]] = {
    "两手攀足": {"side": 0.75, "front": 0.25},  # 实测正面看不清前俯深度,侧面为主
    "攒拳怒目": {"side": 0.50, "front": 0.50},  # 正侧分接近,合并去重双计数
}
_VIEW_FRONT_SUFFIX = "_正面"
_VIEW_SIDE_SUFFIX = "_侧面"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def default_template_paths() -> tuple[Path, Path]:
    root = templates_dir()
    return root / "standard_front_heavy.npz", root / "standard_side_heavy.npz"


def _prefs_path() -> Path:
    return repo_root() / "user_prefs.json"


def load_configured_template_paths() -> tuple[Path, Path]:
    """读取 UI/偏好里填入的正/侧模板路径；缺省回落默认 fixed 路径。"""
    front_default, side_default = default_template_paths()
    try:
        data = json.loads(_prefs_path().read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return front_default, side_default
        front_raw = str(data.get(_PREFS_FRONT_KEY) or "").strip()
        side_raw = str(data.get(_PREFS_SIDE_KEY) or "").strip()
        front = Path(front_raw) if front_raw else front_default
        side = Path(side_raw) if side_raw else side_default
        return front, side
    except (OSError, ValueError):
        return front_default, side_default


def save_configured_template_paths(
    front: Path | str | None = None,
    side: Path | str | None = None,
) -> None:
    """持久化一条龙模板路径（尽力而为，失败静默）。"""
    path = _prefs_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    if front is not None:
        data[_PREFS_FRONT_KEY] = str(front)
    if side is not None:
        data[_PREFS_SIDE_KEY] = str(side)
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def load_configured_template_lists() -> tuple[list[Path], list[Path]]:
    """读取多模板池；缺复数键则回落旧单值键（有则单元素）。均缺返回空列表。

    返回空列表表示该边未配置任何模板（不再自动回落 default_template_paths()——
    多模板语义下"未填"应显式为空,由调用方决定是否用默认；单模板旧行为仍由
    ``load_configured_template_paths()`` 保持）。
    """
    try:
        data = json.loads(_prefs_path().read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}

    def _read(list_key: str, single_key: str) -> list[Path]:
        raw = data.get(list_key)
        if isinstance(raw, list):
            out = [Path(str(p).strip()) for p in raw if str(p).strip()]
            if out:
                return out
        single = str(data.get(single_key) or "").strip()
        return [Path(single)] if single else []

    return (
        _read(_PREFS_FRONT_LIST_KEY, _PREFS_FRONT_KEY),
        _read(_PREFS_SIDE_LIST_KEY, _PREFS_SIDE_KEY),
    )


def save_configured_template_lists(
    front_list: list[Path | str] | None = None,
    side_list: list[Path | str] | None = None,
) -> None:
    """持久化多模板池（尽力而为，失败静默）。"""
    path = _prefs_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    if front_list is not None:
        data[_PREFS_FRONT_LIST_KEY] = [str(p) for p in front_list]
    if side_list is not None:
        data[_PREFS_SIDE_LIST_KEY] = [str(p) for p in side_list]
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


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
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        ok, frame = cap.read()
        return bool(ok and frame is not None and getattr(frame, "size", 0) > 0)
    finally:
        cap.release()


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


def _load_heavy_template(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """加载并校验 heavy + pose33_v3 单模板；失败抛 PostprocessError。"""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise PostprocessError("template_missing", f"固定模板不存在：{path}")
    try:
        with np.load(path, allow_pickle=True) as data:
            if "features" not in data or "meta" not in data:
                raise ValueError("缺少 features/meta")
            features = np.asarray(data["features"], dtype=np.float32)
            meta = dict(data["meta"].item() or {})
    except PostprocessError:
        raise
    except Exception as exc:
        raise PostprocessError("template_invalid", f"无法读取固定模板 {path}：{exc}") from exc
    if features.ndim != 3 or features.shape[0] <= 0 or features.shape[1:] != (22, 2):
        raise PostprocessError(
            "template_invalid",
            f"固定模板特征形状无效：{path} -> {tuple(features.shape)}",
        )
    if not np.isfinite(features).all():
        raise PostprocessError("template_invalid", f"固定模板包含非有限特征：{path}")
    if str(meta.get("pose_variant") or "") != _EXPECTED_TEMPLATE_POSE_VARIANT:
        raise PostprocessError(
            "template_invalid",
            f"固定模板必须使用 {_EXPECTED_TEMPLATE_POSE_VARIANT} 模型：{path}",
        )
    layout = str(meta.get("feature_layout") or "")
    if layout and layout != _EXPECTED_TEMPLATE_LAYOUT:
        raise PostprocessError(
            "template_incompatible",
            f"固定模板布局必须为 {_EXPECTED_TEMPLATE_LAYOUT}：{path} -> {layout}",
        )
    return features, meta


def try_load_heavy_template(path: Path | str | None) -> Path | None:
    """模板路径可用则返回规范化 Path，否则 None（缺文件/无效均视为未填入）。"""
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None
    try:
        _load_heavy_template(Path(raw))
    except PostprocessError:
        return None
    return Path(raw)


def validate_template_pair(front_path: Path, side_path: Path) -> None:
    """考试等正式路径：正/侧模板必须同时可用且布局一致。"""
    _front_features, front_meta = _load_heavy_template(Path(front_path))
    _side_features, side_meta = _load_heavy_template(Path(side_path))
    front_layout = str(front_meta.get("feature_layout") or "") or _EXPECTED_TEMPLATE_LAYOUT
    side_layout = str(side_meta.get("feature_layout") or "") or _EXPECTED_TEMPLATE_LAYOUT
    if front_layout != side_layout or front_layout != _EXPECTED_TEMPLATE_LAYOUT:
        raise PostprocessError(
            "template_incompatible",
            f"正侧模板布局不兼容：front={front_layout or '?'} side={side_layout or '?'}",
        )


def validate_auto_compare_templates(
    front_path: Path | str | None,
    side_path: Path | str | None,
) -> tuple[Path | None, Path | None]:
    """一条龙自动比对：至少一侧可用即可；两侧都缺才 ``template_missing``。

    返回 ``(usable_front_or_None, usable_side_or_None)``。
    两侧都可用时额外校验布局一致。
    """
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


def merge_view_weighted_actions(action_details: list[dict]) -> list[dict]:
    """把 ``_VIEW_WEIGHTS`` 登记的动作的正/侧子分按权重合并成单个动作子分。

    - 仅当正面 ``<key>_正面`` 与侧面 ``<key>_侧面`` **都存在**时合并；缺一则原样保留
      （不误合、不丢分）。
    - 合并项 ``name`` = 无后缀的动作名（成绩表出一列）；``score`` = 归一权重加权和。
    - 未登记动作与不成对的视角原样透传，保序（散打零影响：stem 无后缀/不在表内）。
    """
    by_name = {d.get("name"): d for d in action_details}
    merged: list[dict] = []
    consumed: set[int] = set()
    for idx, d in enumerate(action_details):
        if idx in consumed:
            continue
        name = str(d.get("name") or "")
        base = None
        if name.endswith(_VIEW_FRONT_SUFFIX):
            base = name[: -len(_VIEW_FRONT_SUFFIX)]
        elif name.endswith(_VIEW_SIDE_SUFFIX):
            base = name[: -len(_VIEW_SIDE_SUFFIX)]
        weights = _VIEW_WEIGHTS.get(base) if base else None
        front_d = by_name.get(f"{base}{_VIEW_FRONT_SUFFIX}") if base else None
        side_d = by_name.get(f"{base}{_VIEW_SIDE_SUFFIX}") if base else None
        if weights is None or front_d is None or side_d is None:
            merged.append(d)
            continue
        w_front = float(weights.get("front", 0.0))
        w_side = float(weights.get("side", 0.0))
        total = w_front + w_side
        if total <= 0:
            merged.append(d)
            continue
        w_front, w_side = w_front / total, w_side / total
        score = w_front * float(front_d["score"]) + w_side * float(side_d["score"])
        merged.append(
            {
                "name": base,
                "view": "combined",
                "score": float(score),
                "front_score": float(front_d["score"]),
                "side_score": float(side_d["score"]),
                "view_weights": {"front": w_front, "side": w_side},
                "front_start": int(front_d["start"]),
                "front_end": int(front_d["end"]),
                "side_start": int(side_d["start"]),
                "side_end": int(side_d["end"]),
            }
        )
        consumed.add(idx)
        for j, o in enumerate(action_details):
            if o is front_d or o is side_d:
                consumed.add(j)
    return merged


def validate_auto_compare_template_lists(
    front_list: list[Path | str] | None,
    side_list: list[Path | str] | None,
) -> tuple[list[Path], list[Path]]:
    """多模板池校验：过滤出可用的 heavy+pose33_v3 模板；两池都空才报错。

    返回 ``(usable_front_list, usable_side_list)``，每个元素为规范化 Path，同序去重。
    单个模板不可用则跳过（不整体失败），保证部分模板损坏不阻断其余动作评分。
    """
    def _filter(lst: list[Path | str] | None) -> list[Path]:
        out: list[Path] = []
        seen: set[str] = set()
        for raw in lst or []:
            usable = try_load_heavy_template(raw)
            if usable is None:
                continue
            key = str(usable.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            out.append(usable)
        return out

    front = _filter(front_list)
    side = _filter(side_list)
    if not front and not side:
        raise PostprocessError(
            "template_missing",
            "正/侧模板池均为空或不可用（需 heavy + pose33_v3）。"
            "请在主界面「录制」区至少为一侧添加模板。",
        )
    return front, side


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
                "缺少 pose_landmarker_heavy.task，请先在模型管理中安装 heavy 模型",
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
        """正对正池、侧对侧池：每个模板各跑一次 subsequence DTW 出一个动作子分。

        # ponytail: 弃用 compare_dual_streams 自动正侧拆分——考试双机位视角固定
        # （正机位只拍正、侧机位只拍侧），逐模板 subsequence DTW 更直接,正对正/侧对侧天然成立。
        每边综合 = 该池子分平均（供成绩表正/侧列）；combined = 所有动作子分直接平均。
        """
        from types import SimpleNamespace

        def _run_pool(tpls: list[Path], video: Path, view: str) -> tuple[list[dict], float | None]:
            if not tpls:
                return [], None
            self._raise_if_cancelled(stop_evt)
            # 视频姿态只提取一次,复用给池内每个模板做 DTW（避免逐模板重复提取）。
            results = self._compare_pool(
                tpls,
                video,
                pose_variant=_EXPECTED_TEMPLATE_POSE_VARIANT,
                workers=1,
                stop_evt=stop_evt,
            )
            names = [template_action_name(tpl) for tpl in tpls]

            # 踢腿几何质量分：仅当池内含踢腿动作时,对该视频提一次 raw Pose33 复用。
            raw_series = None
            if any(n in _KICK_KINDS for n in names):
                self._raise_if_cancelled(stop_evt)
                raw_series = extract_pose_raw_series(
                    video, pose_variant=_EXPECTED_TEMPLATE_POSE_VARIANT
                )

            details: list[dict] = []
            for name, res in zip(names, results):
                dtw_score = float(res.score)
                start, end = int(res.start_frame), int(res.end_frame)
                geom_score: float | None = None
                geom_detail = ""
                if raw_series is not None and name in _KICK_KINDS:
                    lm_win, meta_win = slice_pose_raw_series(
                        raw_series, start_frame=start, end_frame=end
                    )
                    geom_score, geom_detail = score_kick_quality(
                        lm_win,
                        meta_win["valid_mask"],
                        _KICK_KINDS[name],
                        width=meta_win.get("width"),
                        height=meta_win.get("height"),
                    )
                if geom_score is not None:
                    final = KICK_GEOM_WEIGHT * geom_score + (1.0 - KICK_GEOM_WEIGHT) * dtw_score
                else:
                    final = dtw_score
                # DTW 不可靠的动作（如背后七颠）：把分线性压到高分窄区间（人人过、
                # 弱区分、可复现）。只对已登记动作生效,散打/其余八段锦不受影响。
                squeeze = _SCORE_SQUEEZE.get(name)
                if squeeze is not None:
                    lo, hi = squeeze
                    final = lo + (hi - lo) * float(np.clip(dtw_score, 0.0, 1.0))
                details.append(
                    {
                        "name": name,
                        "view": view,
                        "score": float(final),
                        "dtw_score": dtw_score,
                        "geom_score": geom_score,
                        "geom_detail": geom_detail,
                        "start": start,
                        "end": end,
                    }
                )
            pool_mean = (
                float(np.mean([d["score"] for d in details])) if details else None
            )
            return details, pool_mean

        front_details, front_score = _run_pool(front_tpls, front_video, "front")
        side_details, side_score = _run_pool(side_tpls, side_video, "side")

        # 同动作多视角合并（_VIEW_WEIGHTS 登记的按权重合成一个子分；未登记原样透传）。
        # front_score/side_score 仍为各池原始均值（供成绩表正/侧列），只影响 combined 与
        # 逐动作 action_scores 列（避免正+侧双模板动作被双重计数）。
        action_scores = merge_view_weighted_actions(front_details + side_details)
        # combined：所有动作子分直接平均（正侧模板各自出分,不再套池权重）。
        all_scores = [d["score"] for d in action_scores]
        combined = float(np.mean(all_scores)) if all_scores else 0.0
        pct = int(np.clip(int(round(combined * 100.0)), 0, 100))
        # 兼容旧字段：front/side_segment 取各池首个匹配窗口（若有）
        front_seg = (
            (front_details[0]["start"], front_details[0]["end"]) if front_details else None
        )
        side_seg = (
            (side_details[0]["start"], side_details[0]["end"]) if side_details else None
        )
        return SimpleNamespace(
            front_score=front_score,
            side_score=side_score,
            combined_score=combined,
            combined_percent=pct,
            front_matches=(),
            side_matches=(),
            front_segment=front_seg,
            side_segment=side_seg,
            action_scores=tuple(action_scores),
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
                self._write_json_atomic(job.segment_dir / "result.json", payload)
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
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{path.stem}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as temporary:
                json.dump(payload, temporary, ensure_ascii=False, indent=2)
                temporary.write("\n")
                temporary_path = Path(temporary.name)
            temporary_path.replace(path)
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

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
