"""真实视频差分（README"待办"第 2 项）：录下 4 个提取入口的输出（landmarks、掩码、特征、视角分数、时间戳）。

用法（仓库根，或旧 commit 的 worktree 根）：
    python -B tests/real_video_diff/record.py <视频目录> <输出目录>
视频目录里放 single.*（单摄）、front.*、side.*（双摄正侧）、practice.*（学生练习 1080p）。
只用 P2 前后都存在的公开入口，可以在 P2 之前的 commit（5721f1d）上原样运行；两次输出用 compare.py 比对。
视频不入库。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))

from core.evaluation import extract_pose_and_view_scores  # noqa: E402
from core.pose.features import extract_features as _extract_pose_features  # noqa: E402
from core.evaluation import FeedbackConfig, extract_series  # noqa: E402
from core.pose import normalize_pose_xy, normalize_pose_xy_v3  # noqa: E402
from core.pose.backends import extract_body_core_features  # noqa: E402
from core.pose import extract_pose_raw, extract_pose_raw_series  # noqa: E402

VIDEOS: dict[str, Path] = {}


def _find_videos(directory: Path) -> dict[str, Path]:
    found = {}
    for key in ("single", "front", "side", "practice"):
        hits = sorted(p for p in directory.glob(f"{key}.*") if p.is_file())
        if not hits:
            raise SystemExit(f"{directory} 里缺少 {key}.*")
        found[key] = hits[0]
    return found


def save(out: Path, name: str, **arrays) -> None:
    np.savez(out / f"{name}.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    print("saved", name, {k: np.asarray(v).shape for k, v in arrays.items()}, flush=True)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("用法: record.py <视频目录> <输出目录>")
    VIDEOS.update(_find_videos(Path(sys.argv[1])))
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for key in ("single", "front", "side"):
        video = VIDEOS[key]
        for workers in (1, 3):
            for norm_name, norm in (("v2", normalize_pose_xy), ("v3", normalize_pose_xy_v3)):
                feats, fps, views = _extract_pose_features(
                    video, pose_variant="lite", workers=workers, normalizer=norm, compute_view=True)
                save(out, f"feat_{key}_w{workers}_{norm_name}", feats=feats, fps=fps, views=views)
        series = extract_pose_raw_series(video, pose_variant="lite")
        save(out, f"rawseries_{key}", landmarks=series.landmarks, valid_mask=series.meta["valid_mask"],
             fps=series.meta["fps"], frame_count=series.meta["frame_count"])
        lm, meta = extract_pose_raw(video, pose_variant="lite", start_frame=20, end_frame=90)
        save(out, f"rawslice_{key}", landmarks=lm, valid_mask=meta["valid_mask"],
             start=meta["start_frame"], end=meta["end_frame"])
        lm, vs, meta = extract_pose_and_view_scores(video, pose_variant="lite")
        save(out, f"techeval_{key}", landmarks=lm, views=vs, end=meta["end_frame"])
        lm, vs, meta = extract_pose_and_view_scores(video, pose_variant="lite", start_frame=30, end_frame=100)
        save(out, f"techevalslice_{key}", landmarks=lm, views=vs, start=meta["start_frame"], end=meta["end_frame"])
        feats, fps, _meta = extract_body_core_features(video, backend="mediapipe", pose_variant="lite")
        save(out, f"bodycore_{key}", feats=feats, fps=fps)

    for name, cfg in (
        ("default", FeedbackConfig(pose_variant="lite")),
        ("resize", FeedbackConfig(pose_variant="lite", max_edge=1280)),
        ("sampled", FeedbackConfig(pose_variant="lite", sample_fps=10)),
    ):
        for key in ("practice", "single"):
            data = extract_series(VIDEOS[key], cfg, lambda: False)
            cap = data["capture"]
            save(out, f"practice_{name}_{key}", landmarks=data["landmarks"], valid_mask=data["valid_mask"],
                 frames=data["frames"], times=data["times"],
                 capture=np.array([cap["sourceFps"], cap["sourceFrames"], cap["analyzedFrames"], cap["stride"],
                                   cap["width"], cap["height"], cap["inferenceWidth"], cap["inferenceHeight"]]))


if __name__ == "__main__":
    main()
