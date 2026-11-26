# -*- coding: utf-8 -*-
"""从标准动作视频生成 pose33_v3 比对模板、双摄配对建库与长视频自动裁剪复核（CompareWindow、制作模板弹窗、CLI、batch 共用）。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
import uuid

import numpy as np

from core.foundation.atomic_io import atomic_commit_staged, temporary_sibling
from core.foundation.paths import templates_dir
from core.foundation.signal import motion_energy
from core.foundation.tasks import Cancelled, from_stopped
from core.pose.engine import EngineSpec
from core.pose.extract import FORMAL, extract_pose_sequence, formal_engine
from core.pose.features import ProgressCb, extract_features
from core.pose.layouts import POSE33_V3
from core.pose.normalize import normalize_pose_xy_v3
from core.pose.validity import DEFAULT_VALID_CONF_THR, MEDIAPIPE_VALIDITY_POLICY
from core.services.preview import export_match_preview
from core.services.template_quality import (
    MIN_CORE_VALID_RATIO,
    MIN_DETECTED_RATIO,
    MIN_TEMPLATE_FRAMES,
    VIEW_SUFFIXES,
    RecordingPair,
    TemplateAutoCropAnalysis,
    TemplateAutoCropReviewRequired,
    TemplateQualityError,
    TemplateQualityReport,
    analyze_template_auto_crop,
    assess_template_quality,
    discover_recording_videos,
    planned_template_paths,
    sanitize_action_name,
    validate_frame_bounds,
)

_DEFAULT_EXTRACT_FEATURES = extract_features


@dataclass(frozen=True)
class TemplateCreateResult:
    """单路模板建库产物及质量摘要。"""

    view: str
    path: Path
    start_frame: int
    end_frame: int
    frame_count: int
    fps: float
    quality: TemplateQualityReport
    preview_path: Path | None = None


def _extract_pose_and_features(
    video_path: Path,
    *,
    pose_variant: str,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> tuple[object | None, np.ndarray, float]:
    """单次顺序提取姿态序列与 pose33_v3 特征；支持测试替身对 ``extract_features`` 的 monkeypatch。"""
    if extract_features is not _DEFAULT_EXTRACT_FEATURES:
        feats, fps, _view = extract_features(
            video_path,
            pose_variant=pose_variant,
            workers=1,
            normalizer=normalize_pose_xy_v3,
            progress_cb=progress_cb,
            stop_evt=stop_evt,
        )
        arr = np.asarray(feats, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] <= 0:
            raise RuntimeError("视频为空或无法读取帧")
        return None, arr, float(fps)

    def progress(done: int, total: int) -> None:
        if progress_cb is not None:
            progress_cb("提取骨架特征", done, total)

    cancel = from_stopped(stop_evt.is_set) if stop_evt is not None else None
    try:
        pose_seq = extract_pose_sequence(
            video_path,
            formal_engine(pose_variant),
            FORMAL,
            workers=1,
            cancel=cancel,
            progress=progress,
        )
        if cancel is not None:
            cancel.raise_if_cancelled()
    except Cancelled:
        raise InterruptedError("姿态特征提取已取消") from None
    if pose_seq.decoded_count == 0 or len(pose_seq) == 0:
        raise RuntimeError("视频为空或无法读取帧")
    feats = pose_seq.features(normalize_pose_xy_v3, tuple(POSE33_V3.shape)).copy()
    return pose_seq, feats, float(pose_seq.meta.fps)


def _resolve_crop_range(
    features: np.ndarray,
    *,
    fps: float,
    start: int | None,
    end: int | None,
    auto_crop: bool,
) -> tuple[int, int, TemplateAutoCropAnalysis, str]:
    """校验显式起止帧或执行自动裁剪/整段截取，返回 ``(start_i, end_i, auto_crop, crop_selection)``。"""
    n_frames = int(features.shape[0])
    energy = motion_energy(features.reshape(n_frames, -1))
    analysis = analyze_template_auto_crop(energy, fps=fps)

    if start is None and end is None:
        if auto_crop:
            if analysis.requires_review:
                raise TemplateAutoCropReviewRequired(analysis)
            return analysis.start_frame, analysis.end_frame, analysis, "auto"
        return 0, n_frames - 1, analysis, "full"

    start_i, end_i = validate_frame_bounds(n_frames, start, end)
    return start_i, end_i, analysis, "manual"


def _build_template_meta(
    video_path: Path,
    *,
    fps: float,
    total_frames: int,
    start_i: int,
    end_i: int,
    pose_variant: str,
    crop_selection: str,
    analysis: TemplateAutoCropAnalysis,
    quality: TemplateQualityReport,
) -> dict:
    return {
        "video": str(video_path),
        "fps": float(fps),
        "frame_count": int(total_frames),
        "start_frame": int(start_i),
        "end_frame": int(end_i),
        "auto_start_frame": int(analysis.start_frame),
        "auto_end_frame": int(analysis.end_frame),
        "pose_variant": pose_variant,
        "feature_layout": POSE33_V3.name,
        "running_mode": "video",
        "cfg": asdict(
            EngineSpec(
                pose_variant=pose_variant, running_mode="video", enable_hands=False
            )
        ),
        # 模板 metadata 扩展（YOLO 迁移 S1 / Issue #6）：增量字段，向后兼容。
        "backend": "mediapipe",
        "model_name": f"pose_landmarker_{pose_variant}",
        "normalizer_version": "v3",
        "confidence_kind": "visibility",
        "validity_policy": MEDIAPIPE_VALIDITY_POLICY,
        "valid_conf_thr": float(DEFAULT_VALID_CONF_THR),
        "crop_selection": crop_selection,
        "selected_retained_ratio": float((end_i - start_i + 1) / max(1, total_frames)),
        "template_scope": "single_action",
        "template_quality": quality.to_meta(),
        **analysis.to_meta(),
    }


def _prepare_template_payload(
    video_path: Path,
    *,
    pose_variant: str,
    start: int | None = None,
    end: int | None = None,
    auto_crop: bool = True,
    validate_quality: bool = True,
    bounds_fn=None,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> tuple[np.ndarray, dict, TemplateQualityReport, int, int, float]:
    pose_seq, features, fps = _extract_pose_and_features(
        video_path,
        pose_variant=pose_variant,
        progress_cb=progress_cb,
        stop_evt=stop_evt,
    )
    if bounds_fn is not None:
        start, end = bounds_fn(int(features.shape[0]), float(fps))
    start_i, end_i, analysis, crop_selection = _resolve_crop_range(
        features, fps=fps, start=start, end=end, auto_crop=auto_crop
    )
    seg_feats = features[start_i : end_i + 1]
    quality = assess_template_quality(
        pose_seq, seg_feats, start_i, end_i, validate_quality=validate_quality
    )
    meta = _build_template_meta(
        video_path,
        fps=fps,
        total_frames=int(features.shape[0]),
        start_i=start_i,
        end_i=end_i,
        pose_variant=pose_variant,
        crop_selection=crop_selection,
        analysis=analysis,
        quality=quality,
    )
    return seg_feats, meta, quality, start_i, end_i, float(fps)


def create_template_from_video(
    video_path: str | Path,
    *,
    pose_variant: str = "heavy",
    start: int | None = None,
    end: int | None = None,
    out_path: str | Path | None = None,
    workers: int = 1,
    preview: bool = False,
    auto_crop: bool = True,
    overwrite: bool = True,
    validate_quality: bool = True,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> Path:
    """从单段视频生成 ``pose33_v3`` 模板（强制顺序跟踪 ``workers=1``、严格边界与质量体检、原子落盘）。"""
    del workers  # 模板元数据固定为 VIDEO 顺序跟踪，统一走单线程保证与比对侧跟踪一致
    video_path = Path(video_path)
    out_dir = templates_dir()
    resolved_out = (
        Path(out_path)
        if out_path
        else (out_dir / f"{video_path.stem}_{pose_variant}.npz")
    )
    if not overwrite and resolved_out.exists():
        raise FileExistsError(f"模板文件已存在：{resolved_out}")

    seg_feats, meta, quality, start_i, end_i, fps = _prepare_template_payload(
        video_path,
        pose_variant=pose_variant,
        start=start,
        end=end,
        auto_crop=auto_crop,
        validate_quality=validate_quality,
        progress_cb=progress_cb,
        stop_evt=stop_evt,
    )
    results = _commit_staged_pair(
        [
            (
                "single",
                video_path,
                resolved_out,
                seg_feats,
                meta,
                quality,
                start_i,
                end_i,
                fps,
            )
        ],
        pose_variant=pose_variant,
        preview=preview,
        overwrite=overwrite,
        progress_cb=progress_cb,
        stop_evt=stop_evt,
    )
    return results[0].path


def _preview_target_and_stale(target: Path, actual_preview: Path) -> tuple[Path, Path]:
    if actual_preview.suffix.lower() == ".avi":
        return target.with_suffix(".preview.avi"), target.with_suffix(".preview.mp4")
    return target.with_suffix(".preview.mp4"), target.with_suffix(".preview.avi")


def _stage_npz_and_previews(
    staged: Sequence[
        tuple[str, Path, Path, np.ndarray, dict, TemplateQualityReport, int, int, float]
    ],
    *,
    pose_variant: str,
    preview: bool,
    progress_cb: ProgressCb | None,
    stop_evt: Event | None,
    tmp_cleanup: list[Path],
) -> tuple[list[tuple[Path, Path]], list[Path], list[Path | None]]:
    commit_pairs: list[tuple[Path, Path]] = []
    stale_previews: list[Path] = []
    staged_previews: list[Path | None] = []
    for _view, _vp, target, seg_feats, meta, _q, _s, _e, _fps in staged:
        if stop_evt is not None and stop_evt.is_set():
            raise InterruptedError("模板生成已取消")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_npz = temporary_sibling(target)
        tmp_cleanup.append(tmp_npz)
        with open(tmp_npz, "wb") as fh:
            np.savez_compressed(
                fh, features=seg_feats, meta=np.array(meta, dtype=object)
            )
        commit_pairs.append((tmp_npz, target))

    for _view, video_path, target, _sf, _m, _q, start_i, end_i, _fps in staged:
        if not preview:
            staged_previews.append(None)
            continue
        if stop_evt is not None and stop_evt.is_set():
            raise InterruptedError("模板生成已取消")
        if progress_cb is not None:
            progress_cb("导出模板预览", 0, 1)
        tmp_req = target.with_name(f"_tmp_{uuid.uuid4().hex}.preview.mp4")
        tmp_cleanup.extend((tmp_req, tmp_req.with_suffix(".avi")))
        actual_tmp = export_match_preview(
            video_path,
            out_path=tmp_req,
            start_frame=start_i,
            end_frame=end_i,
            pose_variant=pose_variant,
            progress_cb=progress_cb,
            stop_evt=stop_evt,
        )
        tmp_cleanup.append(actual_tmp)
        final_prev, stale_prev = _preview_target_and_stale(target, actual_tmp)
        commit_pairs.append((actual_tmp, final_prev))
        stale_previews.append(stale_prev)
        staged_previews.append(final_prev)
    return commit_pairs, stale_previews, staged_previews


def _commit_staged_pair(
    staged: Sequence[
        tuple[str, Path, Path, np.ndarray, dict, TemplateQualityReport, int, int, float]
    ],
    *,
    pose_variant: str,
    preview: bool,
    overwrite: bool,
    progress_cb: ProgressCb | None,
    stop_evt: Event | None,
) -> list[TemplateCreateResult]:
    if stop_evt is not None and stop_evt.is_set():
        raise InterruptedError("模板生成已取消")
    for _view, _vpath, target, *_rest in staged:
        if not overwrite and target.exists():
            raise FileExistsError(f"模板文件已存在：{target}")

    tmp_cleanup: list[Path] = []
    try:
        commit_pairs, stale_previews, staged_previews = _stage_npz_and_previews(
            staged,
            pose_variant=pose_variant,
            preview=preview,
            progress_cb=progress_cb,
            stop_evt=stop_evt,
            tmp_cleanup=tmp_cleanup,
        )
        if stop_evt is not None and stop_evt.is_set():
            raise InterruptedError("模板生成已取消")
        for _view, _vpath, target, *_rest in staged:
            if not overwrite and target.exists():
                raise FileExistsError(f"模板文件已存在：{target}")
        atomic_commit_staged(commit_pairs)
        for stale in stale_previews:
            stale.unlink(missing_ok=True)
    finally:
        for path in dict.fromkeys(tmp_cleanup):
            path.unlink(missing_ok=True)

    return [
        TemplateCreateResult(
            view=view,
            path=target,
            start_frame=start_i,
            end_frame=end_i,
            frame_count=int(seg_feats.shape[0]),
            fps=float(fps),
            quality=quality,
            preview_path=prev_path,
        )
        for (
            view,
            _vp,
            target,
            seg_feats,
            _meta,
            quality,
            start_i,
            end_i,
            fps,
        ), prev_path in zip(staged, staged_previews)
    ]


def create_template_pair(
    videos: Mapping[str, str | Path],
    *,
    action_name: str,
    start_s: float | None = None,
    end_s: float | None = None,
    frame_ranges: Mapping[str, tuple[int, int]] | None = None,
    out_dir: str | Path | None = None,
    pose_variant: str = "heavy",
    preview: bool = False,
    overwrite: bool = True,
    progress_cb: ProgressCb | None = None,
    stop_evt: Event | None = None,
) -> list[TemplateCreateResult]:
    """按动作名与选段为单路或正/侧双路建库（全视角提取体检通过后再统一原子写盘）。"""
    if not videos:
        raise ValueError("至少需要提供一路视频")
    if start_s is not None and end_s is not None and float(start_s) > float(end_s):
        raise ValueError(f"起始时间 {start_s:.2f}s 不能大于结束时间 {end_s:.2f}s")

    paths = planned_template_paths(
        action_name, views=list(videos.keys()), out_dir=out_dir
    )
    if not overwrite:
        for target in paths.values():
            if target.exists():
                raise FileExistsError(f"模板文件已存在：{target}")

    staged = []
    for view, raw_video in videos.items():
        if stop_evt is not None and stop_evt.is_set():
            raise InterruptedError("姿态特征提取已取消")
        video_path = Path(raw_video)
        bounds_fn = lambda n_frames, fps, v=view: _pair_frame_bounds(
            view=v,
            n_frames=n_frames,
            fps=fps,
            start_s=start_s,
            end_s=end_s,
            frame_ranges=frame_ranges,
        )
        seg_feats, meta, quality, start_i, end_i, fps = _prepare_template_payload(
            video_path,
            pose_variant=pose_variant,
            auto_crop=False,
            validate_quality=True,
            bounds_fn=bounds_fn,
            progress_cb=progress_cb,
            stop_evt=stop_evt,
        )
        staged.append(
            (
                view,
                video_path,
                paths[view],
                seg_feats,
                meta,
                quality,
                start_i,
                end_i,
                fps,
            )
        )

    return _commit_staged_pair(
        staged,
        pose_variant=pose_variant,
        preview=preview,
        overwrite=overwrite,
        progress_cb=progress_cb,
        stop_evt=stop_evt,
    )


def _pair_frame_bounds(
    *,
    view: str,
    n_frames: int,
    fps: float,
    start_s: float | None,
    end_s: float | None,
    frame_ranges: Mapping[str, tuple[int, int]] | None,
) -> tuple[int, int]:
    if frame_ranges is not None and view in frame_ranges:
        s_raw, e_raw = frame_ranges[view]
        return int(s_raw), int(e_raw)
    if start_s is not None or end_s is not None:
        fps_val = float(fps or 30.0)
        s_i = (
            0
            if start_s is None
            else max(0, min(n_frames - 1, int(round(float(start_s) * fps_val))))
        )
        e_i = (
            (n_frames - 1)
            if end_s is None
            else max(0, min(n_frames - 1, int(round(float(end_s) * fps_val))))
        )
        return s_i, e_i
    return 0, n_frames - 1
