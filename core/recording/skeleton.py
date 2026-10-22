# -*- coding: utf-8 -*-
"""录后骨架成片步骤 ``RenderSkeleton``：采集期间只录原始画面，需要带骨架的成片时在录后从原始录像生成（D-26）。

每路：原始录像 → ``annotate_video`` 画骨架写到 ``<名>.skeleton`` 临时文件 → 转码 → 校验首帧可读 →
替换为该路的录像文件（文件名不变，H.264 时后缀同为 ``.mp4``）。原始录像保留到成片校验成功；
任一步失败或取消时删掉临时文件、保留原始录像，失败抛 ``PostprocessError("skeleton_render_failed")``。
跑在 analysis 线（逐帧推理，与比对评分同一条串行线）。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.foundation.atomic_io import replace_file
from core.foundation.tasks import Cancelled, CancelToken, from_stopped
from core.pose import EngineSpec, annotate_video
from core.recording.steps import (
    VIEW_LABELS,
    Lane,
    PostprocessError,
    StepContext,
    Terminal,
    TranscodeFn,
    VideoValidator,
    default_transcode,
    default_video_validator,
)

SkeletonRenderer = Callable[[Path, Path, EngineSpec, CancelToken], Path]   # (源, 目标, 引擎, 取消) -> 实际路径


def default_skeleton_renderer(source: Path, target: Path, engine: EngineSpec, cancel: CancelToken) -> Path:
    path, _codec, _frames = annotate_video(source, target, engine, cancel=cancel)
    return path


def skeleton_engine(pose_variant: str, enable_hands: bool) -> EngineSpec:
    """成片用会话的模型与手部设置，VIDEO 模式跟踪，和实时预览的骨架一致。"""
    return EngineSpec(pose_variant=pose_variant, running_mode="video", enable_hands=bool(enable_hands))


@dataclass(frozen=True)
class RenderSkeleton:
    """逐路把当前录像换成带骨架的成片。"""

    engine: EngineSpec
    render: SkeletonRenderer = default_skeleton_renderer
    transcode: TranscodeFn = default_transcode
    validator: VideoValidator = default_video_validator
    lane: Lane = "analysis"

    def run(self, ctx: StepContext, cancel: CancelToken) -> Terminal | None:
        stopped = from_stopped(lambda: cancel.cancelled or ctx.stop_evt.is_set())
        ctx.publish("rendering", "正在生成骨架成片")
        for view in ctx.views:
            if stopped.cancelled:
                raise Cancelled(cancel.reason or "骨架成片已取消")
            ctx.videos[view] = self._one(ctx, view, stopped)
        return None

    def _one(self, ctx: StepContext, view: str, stopped: CancelToken) -> Path:
        source = Path(ctx.video(view))
        label = VIEW_LABELS.get(view, view)
        # 渲染中途失败时实际路径未知：按 writer 可能用的两种后缀清理
        temporaries = [source.with_name(f"{source.stem}.skeleton{suffix}") for suffix in (".mp4", ".avi")]
        try:
            rendered = Path(self.render(source, source.with_name(f"{source.stem}.skeleton.mp4"), self.engine, stopped))
            temporaries.append(rendered)
            final = Path(self.transcode(rendered, ctx.stop_evt) or rendered)
            temporaries.append(final)
            if not self.validator(final):
                raise RuntimeError(f"成片不存在、为空或无法读取：{final.name}")
            target = source.with_suffix(final.suffix)
            if final.suffix.lower() == ".avi":
                ctx.warnings.append(f"{label}骨架成片 H.264 转码失败，已回退使用 AVI")
            replace_file(final, target)
            if target != source:
                source.unlink(missing_ok=True)
            return target
        except (Cancelled, InterruptedError):
            _remove(temporaries, keep=source)
            raise
        except Exception as exc:
            _remove(temporaries, keep=source)
            raise PostprocessError("skeleton_render_failed", f"{label}骨架成片失败（已保留原始录像）：{exc}") from exc


def _remove(paths: list[Path], *, keep: Path) -> None:
    for path in paths:
        if path != keep:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = ["RenderSkeleton", "SkeletonRenderer", "default_skeleton_renderer", "skeleton_engine"]
