# -*- coding: utf-8 -*-
"""逐帧处理步骤：转正、姿态推理、平滑、在线识别、标签与骨架、站位检测、HUD（设计文档 §8.2）。

每个 stage 实现 ``open(ctx)``（可能很慢，如加载模型）、``process(packet, ctx)``、``close()``；
runner 在自己的线程里依次调用，stage 之间只通过 ``FramePacket`` 传数据。并行推理时 ``PoseStage``
不在 ``process`` 里推理，而是由 runner 经 ``submit``/``take`` 投递与按序取回。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from core.evaluation import MatchResult, OnlineActionMatcher, load_template_library
from core.foundation.paths import models_dir, templates_dir
from core.media import FrameSet, apply_rotation
from core.pose import (
    EngineSpec,
    ParallelPoseEngine,
    PreviewLandmarkSmoother,
    create_engine,
    default_pipeline_factory,
    draw_cli_hud,
    draw_roi,
    draw_tk_hud,
    normalize_pose_xy_v3,
    render_pose,
)
from core.live.config import LiveControls, PresenceProbe, SessionConfig

_MATCH_SUFFIXES = ("_正面", "_侧面", "_左侧", "_右侧")




def match_text(result: MatchResult) -> str:
    """在线识别命中的界面文本：去掉模板名的视角后缀。"""
    name = result.action or ""
    for suffix in _MATCH_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return f"识别到：{name} ({result.score:.2f})"


@dataclass
class PresenceSample:
    """站位检测的一次采样：人是否在 ROI 内、采样时刻（``time.monotonic``）。"""

    present: bool
    sampled_at: float


@dataclass
class FramePacket:
    """一轮各路帧在 stage 之间传递的数据。``frames`` 是转正后的原始帧，``out`` 是叠加后的帧。"""

    index: int
    frames: FrameSet
    timestamp_ms: int = 0
    captured_at_ns: int = 0          # 主路采集时刻（``capture_clock_ns``）
    pose: dict[str, object] = field(default_factory=dict)
    hands: dict[str, list] = field(default_factory=dict)
    drawn_pose: dict[str, object] = field(default_factory=dict)
    out: FrameSet = field(default_factory=dict)
    actions: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class StageContext:
    """stage 共用的会话信息与回调；``emit`` 发送会话事件（在线识别命中、站位采样）。"""

    config: SessionConfig
    controls: LiveControls
    emit: Callable[[str, object], None]
    fps: float = 30.0
    fps_text: str = ""


class Stage(Protocol):
    def open(self, ctx: StageContext) -> None: ...
    def process(self, packet: FramePacket, ctx: StageContext) -> FramePacket: ...
    def close(self) -> None: ...


class TransformStage:
    """推理前按每路的运行时旋转转正（视频文件不旋转），保证关键点、预览与录制同一坐标系。"""

    def open(self, ctx: StageContext) -> None:
        return None

    def process(self, packet: FramePacket, ctx: StageContext) -> FramePacket:
        if not ctx.config.is_file:
            packet.frames = {
                role: apply_rotation(frame, ctx.controls.rotation_of(role))
                for role, frame in packet.frames.items()
            }
        return packet

    def close(self) -> None:
        return None


class PoseStage:
    """每路一个顺序 VIDEO 引擎；或一个 ``ParallelPoseEngine``（IMAGE 模式，摄像头满则丢帧、视频保序）。"""

    def __init__(self, roles: tuple[str, ...]) -> None:
        self._roles = roles
        self._engines: dict[str, object] = {}
        self._parallel: ParallelPoseEngine | None = None

    @property
    def parallel(self) -> ParallelPoseEngine | None:
        return self._parallel

    def open(self, ctx: StageContext) -> None:
        cfg = ctx.config
        directory = models_dir()
        if cfg.parallel:
            self._parallel = ParallelPoseEngine(
                pipeline_factory=default_pipeline_factory(
                    models_dir=directory, pose_variant=cfg.pose_variant,
                    enable_hands=cfg.enable_hands, delegate=cfg.delegate,
                ),
                workers=cfg.workers,
                drop_when_full=not cfg.is_file,
                defer_draw=True,
            )
            self._parallel.start()
            return
        spec = EngineSpec(
            pose_variant=cfg.pose_variant, running_mode="video",
            enable_hands=cfg.enable_hands, delegate=cfg.delegate,
        )
        for role in self._roles:
            self._engines[role] = create_engine(spec, models_dir=directory)

    def process(self, packet: FramePacket, ctx: StageContext) -> FramePacket:
        for role, frame in packet.frames.items():
            engine = self._engines[role]
            if ctx.config.is_file:
                ts = int(packet.index * 1000.0 / max(1e-6, ctx.fps))
            else:
                ts = engine.next_timestamp_ms(is_file=False, fps_for_ts=ctx.fps)
            if role == self._roles[0]:
                packet.timestamp_ms = ts
            packet.pose[role], packet.hands[role] = engine.infer(frame, timestamp_ms=ts)
        return packet

    def close(self) -> None:
        if self._parallel is not None:
            self._parallel.close()
        for engine in self._engines.values():
            try:
                engine.close()
            except Exception:
                pass
        self._engines = {}


class SmoothStage:
    """预览平滑：只影响骨架绘制，原始关键点照常用于在线识别与动作标签。"""

    def __init__(self) -> None:
        self._smoothers: dict[str, PreviewLandmarkSmoother] = {}

    def open(self, ctx: StageContext) -> None:
        self._smoothers = {role: PreviewLandmarkSmoother() for role in ctx.config.roles}

    def process(self, packet: FramePacket, ctx: StageContext) -> FramePacket:
        for role, pose in packet.pose.items():
            packet.drawn_pose[role] = self._smoothers[role].feed(pose, timestamp_ms=packet.timestamp_ms)
        return packet

    def close(self) -> None:
        self._smoothers = {}


class OnlineMatchStage:
    """主路原始关键点送入在线 matcher；模板库缺失或不兼容时不识别，预览照常。命中发 ``match`` 事件。"""

    def __init__(self) -> None:
        self._matcher: OnlineActionMatcher | None = None

    def open(self, ctx: StageContext) -> None:
        def on_result(result: MatchResult) -> None:
            if result.action is not None:
                ctx.emit("match", match_text(result))

        try:
            templates = load_template_library(templates_dir() / "online")
            self._matcher = OnlineActionMatcher(templates, on_result) if templates else None
        except Exception:
            self._matcher = None

    def process(self, packet: FramePacket, ctx: StageContext) -> FramePacket:
        pose = packet.pose.get(ctx.config.roles[0])
        if self._matcher is not None and pose is not None:
            feature = normalize_pose_xy_v3(pose)
            if feature is not None:
                self._matcher.push(feature, packet.timestamp_ms, packet.index)
        return packet

    def close(self) -> None:
        if self._matcher is not None:
            self._matcher.close()
            self._matcher = None


class LabelsStage:
    """画骨架与手部，并按原始关键点给出实时标签（V 手势、举手、下蹲）。"""

    def open(self, ctx: StageContext) -> None:
        self._spec = EngineSpec(pose_variant=ctx.config.pose_variant)

    def process(self, packet: FramePacket, ctx: StageContext) -> FramePacket:
        for role, frame in packet.frames.items():
            raw = packet.pose.get(role)
            packet.out[role], packet.actions[role] = render_pose(
                frame,
                packet.drawn_pose.get(role, raw),
                packet.hands.get(role) or [],
                draw_face=self._spec.draw_pose_face,
                draw_joint_angles_overlay=self._spec.draw_joint_angles,
                action_pose_landmarks=raw,
            )
        return packet

    def close(self) -> None:
        return None


class OccupancyStage:
    """推理关闭且站位检测已启用时，每 ``stride`` 帧用独立 Lite 引擎推理主路，发 ``presence`` 事件，并在预览上画 ROI。

    引擎在启用后第一次需要时构造、停用后关闭；推理或绘制失败不影响会话。
    """

    def __init__(self, probe: PresenceProbe) -> None:
        self._probe = probe
        self._engine = None

    def open(self, ctx: StageContext) -> None:
        return None

    def process(self, packet: FramePacket, ctx: StageContext) -> FramePacket:
        settings = ctx.controls.occupancy.get()
        if not settings.armed:
            self._close_engine()
            return packet
        primary = ctx.config.roles[0]
        try:
            if self._engine is None:
                self._engine = create_engine(
                    EngineSpec(pose_variant="lite", running_mode="video", enable_hands=False),
                    models_dir=models_dir(),
                )
            if packet.index % max(1, int(settings.stride or 3)) == 0:
                ts = self._engine.next_timestamp_ms(is_file=False, fps_for_ts=ctx.fps)
                sampled_at = time.monotonic()
                pose, _ = self._engine.infer(packet.frames[primary], timestamp_ms=ts)
                ctx.emit("presence", PresenceSample(bool(self._probe(pose, settings.roi)), sampled_at))
            draw_roi(packet.out[primary], settings.roi)
        except Exception:
            pass
        return packet

    def _close_engine(self) -> None:
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception:
                pass
            self._engine = None

    def close(self) -> None:
        self._close_engine()


class OverlayStage:
    """在主路叠加 HUD：Tk 为左上角 FPS，CLI 为标签加 FPS；底部提示按 ``hud_hint``。"""

    def open(self, ctx: StageContext) -> None:
        return None

    def process(self, packet: FramePacket, ctx: StageContext) -> FramePacket:
        primary = ctx.config.roles[0]
        frame = packet.out[primary]
        if ctx.config.hud == "cli":
            draw_cli_hud(frame, packet.actions.get(primary, []), ctx.fps_text, hint=ctx.config.hud_hint)
        else:
            draw_tk_hud(frame, ctx.fps_text, hint=ctx.config.hud_hint)
        return packet

    def close(self) -> None:
        return None


def build_stages(config: SessionConfig, *, probe: PresenceProbe | None = None) -> tuple[PoseStage | None, list]:
    """按配置组装 stage：推理关闭时没有 ``PoseStage``；返回 ``(pose, 推理之后的 stages)``。"""
    pose = PoseStage(config.roles) if config.inference else None
    after: list = []
    if config.inference:
        if config.smoothing:
            after.append(SmoothStage())
        if config.online_match:
            after.append(OnlineMatchStage())
        after.append(LabelsStage())
    if config.occupancy and probe is not None:
        after.append(OccupancyStage(probe))
    return pose, after


__all__ = [
    "FramePacket",
    "LabelsStage",
    "OccupancyStage",
    "OnlineMatchStage",
    "OverlayStage",
    "PoseStage",
    "PresenceSample",
    "SmoothStage",
    "Stage",
    "StageContext",
    "TransformStage",
    "build_stages",
    "match_text",
]
