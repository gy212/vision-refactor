# -*- coding: utf-8 -*-
"""独立「制作模板」弹窗控制器：双摄/单摄视频加载、同步时间→帧换算、句柄生命周期、覆盖检查与后台生成。

视图只需实现 ``TemplateBuilderView`` 协议；视频解码句柄（``VideoSource``）只在主线程打开/抽帧/关闭，
后台线程按文件路径走 ``core.services.create_template_pair``，经 ``UiDispatcher`` 回传进度与体检结果。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from core.foundation import TaskGroup
from core.media import DecodeOptions, VideoMeta, VideoSource
from core.services import (
    TemplateCreateResult,
    TemplateQualityError,
    TemplateQualityReport,
    create_template_pair,
    discover_recording_videos,
    planned_template_paths,
    sanitize_action_name,
)

POSE_VARIANT = "heavy"
VIEW_LABELS = {"front": "正面", "side": "侧面", "single": "单视角"}
COMPARE_BACKFILL_OPTIONS = ("仅加入正面", "仅加入侧面", "仅保存不入池")
_COMPARE_BACKFILL_VIEW = {
    "仅加入正面": "front",
    "仅加入侧面": "side",
    "仅保存不入池": None,
}

logger = logging.getLogger(__name__)
Post = Callable[[Callable[[], Any]], Any]


class TemplateBuilderView(Protocol):
    def set_running(self, running: bool) -> None: ...
    def set_status(self, text: str) -> None: ...
    def show_progress(self, stage: str, done: int, total: int) -> None: ...
    def on_generated(
        self, results: list[TemplateCreateResult], summary: str
    ) -> None: ...
    def on_cancelled(self) -> None: ...
    def on_failed(self, message: str) -> None: ...


def sec_to_frame(t_s: float, fps: float, frame_count: int) -> int:
    """秒数换算到合法闭区间帧号 ``[0, frame_count - 1]``。"""
    if frame_count <= 0:
        return 0
    rate = float(fps) if fps and fps > 0 else 30.0
    idx = int(round(max(0.0, float(t_s)) * rate))
    return max(0, min(idx, int(frame_count) - 1))


def frame_ranges_for_times(
    metas: Mapping[str, VideoMeta],
    start_s: float,
    end_s: float,
) -> dict[str, tuple[int, int]]:
    """按各路实际 fps 把时间区间转为含端点的 ``(start_frame, end_frame)``，保底 ``start < end``。"""
    ranges: dict[str, tuple[int, int]] = {}
    for view, meta in metas.items():
        n = int(meta.frame_count)
        s = sec_to_frame(start_s, meta.fps, n)
        e = sec_to_frame(end_s, meta.fps, n)
        if e <= s and n > 1:
            if s < n - 1:
                e = s + 1
            else:
                s = max(0, e - 1)
        ranges[view] = (s, e)
    return ranges


def min_duration_seconds(metas: Mapping[str, VideoMeta]) -> float:
    """已加载各路视频的公共时长（秒）。"""
    if not metas:
        return 0.0
    return min(float(m.frame_count) / (float(m.fps) or 30.0) for m in metas.values())


def min_step_seconds(metas: Mapping[str, VideoMeta]) -> float:
    """一帧的最小时间间隔（用于滑块保底拉开 start < end）。"""
    if not metas:
        return 1.0 / 30.0
    min_fps = min(float(m.fps) or 30.0 for m in metas.values())
    return 1.0 / max(1.0, min_fps)


def nudge_time(
    current_s: float,
    delta_frames: int,
    fps: float,
    min_s: float,
    max_s: float,
) -> float:
    """按帧步进调整秒数并夹紧到 ``[min_s, max_s]``。"""
    rate = float(fps) if fps and fps > 0 else 30.0
    target = float(current_s) + (float(delta_frames) / rate)
    return max(float(min_s), min(float(max_s), target))


def format_sources_info(metas: Mapping[str, VideoMeta]) -> str:
    """已加载视频的摘要文本。"""
    if not metas:
        return "尚未选择视频（可选择双摄录制目录，或单个视频文件）"
    parts: list[str] = []
    for view, meta in metas.items():
        dur = float(meta.frame_count) / (float(meta.fps) or 30.0)
        parts.append(
            f"{VIEW_LABELS.get(view, view)}：{meta.path.name}"
            f"（{meta.width}x{meta.height}, {meta.fps:.1f}fps, {meta.frame_count}帧, {dur:.2f}s）"
        )
    return "\n".join(parts)


def format_range_summary(
    metas: Mapping[str, VideoMeta], start_s: float, end_s: float
) -> str:
    """已选区间摘要：秒数、时长与各视角帧范围。"""
    if not metas:
        return "已选区间：—"
    ranges = frame_ranges_for_times(metas, start_s, end_s)
    dur = max(0.0, float(end_s) - float(start_s))
    span_parts = [
        f"{VIEW_LABELS.get(view, view)} {s}..{e} 帧" for view, (s, e) in ranges.items()
    ]
    return f"已选区间：{start_s:.2f}s ~ {end_s:.2f}s（时长 {dur:.2f}s，{' / '.join(span_parts)}）"


def format_quality_report(report: TemplateQualityReport) -> str:
    """格式化单路质量体检报告。"""
    text = (
        f"帧数 {report.frame_count}，检出率 {report.detected_ratio * 100:.1f}%，"
        f"核心躯干有效率 {report.core_valid_ratio * 100:.1f}%，"
        f"四肢有效率 {report.limb_valid_ratio * 100:.1f}%"
    )
    if report.warnings:
        text += "；提示：" + "；".join(report.warnings)
    return text


def format_generation_summary(results: Sequence[TemplateCreateResult]) -> str:
    """生成成功后的多视角摘要（含输出路径、帧区间、质量体检与警告）。"""
    lines: list[str] = []
    for res in results:
        label = VIEW_LABELS.get(res.view, res.view)
        lines.append(
            f"[{label}] {res.path}（{res.start_frame}..{res.end_frame}，"
            f"共 {res.frame_count} 帧 @ {res.fps:.1f}fps）"
        )
        lines.append(f"  体检：{format_quality_report(res.quality)}")
        if res.preview_path is not None:
            lines.append(f"  预览：{res.preview_path}")
    return "\n".join(lines)


def format_quality_error(exc: TemplateQualityError) -> str:
    """质量校验失败文案：优先包含数值体检细节。"""
    if exc.report is None:
        return str(exc)
    return f"{exc}\n体检详情：{format_quality_report(exc.report)}"


def filter_backfill_results(
    results: Sequence[TemplateCreateResult],
    *,
    pool_mode: str,
    backfill_choice: str = COMPARE_BACKFILL_OPTIONS[0],
) -> list[TemplateCreateResult]:
    """按模板池模式过滤生成后需要自动回填的模板列表（不影响磁盘落盘）。"""
    if pool_mode == "none":
        return []
    if pool_mode == "dual":
        return [res for res in results if res.view in ("front", "side")]
    if pool_mode == "single":
        if len(results) <= 1:
            return list(results)
        target_view = _COMPARE_BACKFILL_VIEW.get(backfill_choice, "front")
        if target_view is None:
            return []
        return [res for res in results if res.view == target_view]
    return list(results)


class TemplateBuilderController:
    """管理制作模板弹窗的视频句柄、帧预览读取与后台创建任务。"""

    def __init__(
        self,
        view: TemplateBuilderView,
        post: Post,
        *,
        open_source: Callable[[Path], VideoSource] | None = None,
        create_pair: Callable[..., list[TemplateCreateResult]] = create_template_pair,
        out_dir: Path | None = None,
    ) -> None:
        self._view = view
        self._post = post
        self._open_source = open_source or (
            lambda p: VideoSource(p, DecodeOptions(), None)
        )
        self._create_pair = create_pair
        self._out_dir = out_dir
        self._sources: dict[str, VideoSource] = {}
        self._metas: dict[str, VideoMeta] = {}
        self._tasks = TaskGroup("template-builder")
        self._stop_evt = threading.Event()
        self._single_mode = False

    @property
    def busy(self) -> bool:
        return bool(self._tasks.pending())

    @property
    def metas(self) -> dict[str, VideoMeta]:
        return dict(self._metas)

    @property
    def single_mode(self) -> bool:
        return self._single_mode

    def close_sources(self) -> None:
        """安全关闭所有已打开的 ``VideoSource`` 句柄。"""
        for src in self._sources.values():
            try:
                src.close()
            except Exception as exc:
                logger.debug("关闭视频句柄失败：%s", exc)
        self._sources.clear()
        self._metas.clear()

    def _open_views(
        self, view_paths: Mapping[str, Path], *, single_mode: bool
    ) -> dict[str, VideoMeta]:
        opened: dict[str, VideoSource] = {}
        metas: dict[str, VideoMeta] = {}
        try:
            for view, path in view_paths.items():
                src = self._open_source(Path(path))
                if int(src.meta.frame_count) <= 0:
                    src.close()
                    raise ValueError(f"无法读取视频总帧数或视频为空：{path}")
                opened[view] = src
                metas[view] = src.meta
        except Exception:
            for src in opened.values():
                try:
                    src.close()
                except Exception as close_exc:
                    logger.debug("回滚关闭视频句柄失败：%s", close_exc)
            raise
        self.close_sources()
        self._sources = opened
        self._metas = metas
        self._single_mode = single_mode
        return dict(metas)

    def load_directory(self, dir_path: str | Path) -> dict[str, VideoMeta]:
        """加载双摄录制目录（自动识别 front/side）。"""
        pair = discover_recording_videos(dir_path)
        view_paths: dict[str, Path] = {}
        if pair.front is not None:
            view_paths["front"] = pair.front
        if pair.side is not None:
            view_paths["side"] = pair.side
        if not view_paths:
            raise ValueError(f"目录中未找到正面或侧面视频：{dir_path}")
        return self._open_views(view_paths, single_mode=False)

    def load_single_file(
        self, file_path: str | Path, view: str
    ) -> dict[str, VideoMeta]:
        """加载单个视频文件并绑定视角（front / side / single）。"""
        if view not in VIEW_LABELS:
            raise ValueError(f"不支持的视角：{view!r}")
        pair = discover_recording_videos(file_path)
        target = pair.front or pair.side
        if target is None:
            raise FileNotFoundError(f"视频文件不存在：{file_path}")
        return self._open_views({view: target}, single_mode=True)

    def set_single_view(self, view: str) -> dict[str, VideoMeta]:
        """单视频模式下切换视角标签（不重开文件句柄）。"""
        if not self._single_mode or len(self._sources) != 1:
            return dict(self._metas)
        if view not in VIEW_LABELS:
            raise ValueError(f"不支持的视角：{view!r}")
        old_view, src = next(iter(self._sources.items()))
        if old_view == view:
            return dict(self._metas)
        meta = self._metas.pop(old_view)
        self._sources.pop(old_view)
        self._sources[view] = src
        self._metas[view] = meta
        return dict(self._metas)

    def read_preview_frames(self, t_s: float) -> dict[str, tuple[int, Any]]:
        """按时刻读取各路当前帧，返回 ``{view: (frame_idx, bgr_or_none)}``。"""
        out: dict[str, tuple[int, Any]] = {}
        for view, src in self._sources.items():
            meta = self._metas[view]
            idx = sec_to_frame(t_s, meta.fps, meta.frame_count)
            out[view] = (idx, src.read_at(idx))
        return out

    def plan_paths(self, action_name: str) -> dict[str, Path]:
        """计算待生成模板路径；未加载视频时按当前单/双视角占位预览或返回空字典。"""
        if not self._metas:
            return {}
        clean = sanitize_action_name(action_name)
        return planned_template_paths(
            clean, views=list(self._metas.keys()), out_dir=self._out_dir
        )

    def existing_targets(self, action_name: str) -> list[Path]:
        """返回已存在、将被覆盖的目标模板路径。"""
        paths = self.plan_paths(action_name)
        return [p for p in paths.values() if p.exists()]

    def start_generate(
        self,
        *,
        action_name: str,
        start_s: float,
        end_s: float,
        preview: bool = False,
        overwrite: bool = True,
    ) -> bool:
        """校验入参并在后台线程执行模板创建。已有任务在跑时返回 False。"""
        if self.busy:
            return False
        if not self._metas:
            raise ValueError("请先选择视频来源。")
        clean_name = sanitize_action_name(action_name)
        frame_ranges_for_times(self._metas, start_s, end_s)
        videos = {view: meta.path for view, meta in self._metas.items()}

        self._stop_evt.clear()
        self._view.set_running(True)
        self._view.set_status("正在生成模板…")
        self._view.show_progress("准备中", 0, 0)

        def run(_token) -> None:
            try:
                results = self._create_pair(
                    videos,
                    action_name=clean_name,
                    start_s=float(start_s),
                    end_s=float(end_s),
                    out_dir=self._out_dir,
                    pose_variant=POSE_VARIANT,
                    preview=preview,
                    overwrite=overwrite,
                    progress_cb=self._progress,
                    stop_evt=self._stop_evt,
                )
                summary = format_generation_summary(results)
                self._post(lambda r=results, s=summary: self._view.on_generated(r, s))
            except InterruptedError:
                self._post(self._view.on_cancelled)
            except TemplateQualityError as exc:
                msg = format_quality_error(exc)
                self._post(lambda m=msg: self._view.on_failed(m))
            except Exception as exc:
                msg = str(exc)
                self._post(lambda m=msg: self._view.on_failed(m))
            finally:
                self._post(lambda: self._view.set_running(False))

        self._tasks.spawn("generate-pair", run, owner="template-builder")
        return True

    def _progress(self, stage: str, done: int, total: int) -> None:
        self._post(lambda: self._view.show_progress(stage, done, total))

    def stop(self) -> None:
        self._stop_evt.set()
        self._view.set_status("正在停止…")

    def close(self) -> None:
        self._stop_evt.set()
        self.close_sources()
