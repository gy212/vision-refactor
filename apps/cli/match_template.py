"""``match_template`` CLI：把待测视频与模板做子序列 DTW 比对并输出分数（含 body_core 调试路径）。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from core.evaluation import (
    COMPARE,
    LEGACY_DEFAULT_FEATURE_LAYOUT,
    POSE33_BASELINE,
    MultiPersonReviewRequiredError,
    Template,
    load_template,
    match_body_core_template,
    match_templates,
    normalizer_for,
    normalizer_version_from_layout,
)
from core.foundation import templates_dir
from core.media import VideoOpenError
from core.pose import FORMAL, POSE33_V3, extract_pose_sequence, formal_engine
from core.services import write_pose_preview

_GATE_SOURCE_LABELS = {
    "template": "模板来源视频",
    "video": "待匹配视频",
    "template+video": "模板来源视频与待匹配视频",
}


def _extract_features(video_path: Path, *, pose_variant: str, normalizer) -> tuple[np.ndarray, float]:
    """``FORMAL`` 顺序提取；未检出帧沿用上一帧特征，首帧前补零。"""
    try:
        seq = extract_pose_sequence(video_path, formal_engine(pose_variant), FORMAL)
    except VideoOpenError:
        raise RuntimeError(f"Cannot open video: {video_path}") from None
    if len(seq) == 0:
        raise RuntimeError("No frames read")
    return seq.features(normalizer, tuple(POSE33_V3.shape)), seq.meta.fps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="Template .npz produced by make_template.py")
    ap.add_argument("--video", required=True, help="Target video to search")
    ap.add_argument("--pose", default=None, choices=["lite", "full", "heavy"], help="Override pose variant")
    ap.add_argument("--preview", action="store_true", help="Export best-match preview video")
    # YOLO 迁移 Issue #8 / S2：body_core_v1 离线闭环匹配（显式 opt-in）。
    ap.add_argument(
        "--backend",
        default=None,
        choices=["mediapipe", "yolo"],
        help="Pose backend override（默认按模板 meta.backend）。body_core_v1 模板才支持 yolo。",
    )
    ap.add_argument(
        "--feature-layout",
        dest="feature_layout",
        default=None,
        choices=["pose33_v3", "body_core_v1"],
        help="Feature layout override（默认按模板 meta.feature_layout 推断）。",
    )
    ap.add_argument(
        "--allow-multi-person",
        dest="allow_multi_person",
        action="store_true",
        help=(
            "多人场景降级而非拒绝（Issue #9）：检出多人时不报错，但仍不产出分数"
            "（标记『需人工复核』）。默认拒绝多人视频出分。"
        ),
    )
    args = ap.parse_args()

    # 按模板 meta 或显式参数判定是否走 body_core_v1 闭环，绝不改动 pose33_v3 默认路径。
    template = load_template(args.template, COMPARE)
    tpl_layout = str(template.raw_meta.get("feature_layout", ""))
    want_body_core = (
        args.feature_layout == "body_core_v1"
        or args.backend == "yolo"
        or tpl_layout == "body_core_v1"
    )
    if want_body_core:
        _match_body_core(args)
        return
    _match_pose33_v3(args, template)


def _match_body_core(args) -> None:
    """body_core_v1 模板匹配（未标定调试分数，不得对外评分）。"""
    try:
        res = match_body_core_template(
            args.template,
            Path(args.video),
            backend=args.backend,
            pose_variant=args.pose,
            reject_multi_person=not args.allow_multi_person,
        )
    except MultiPersonReviewRequiredError as e:
        source_label = _GATE_SOURCE_LABELS.get(getattr(e, "gate_source", "video"), "多人来源")
        print(f"Template: {args.template}")
        print(f"Video:    {args.video}")
        print(
            f"多人场景闸门：{source_label}检出多人（max_persons="
            f"{e.max_persons}，multi_person_frames={e.multi_person_frames}），"
            "拒绝出分 → 需人工复核（不混入正常评分结果）。"
        )
        print("  （如需查看降级结果，可加 --allow-multi-person，但分数仍不产出。）")
        return

    print(f"Template: {args.template}")
    print(f"Video:    {res.video_path}")
    print(f"Backend:  {res.backend}  feature_layout={res.feature_layout}")
    print(
        f"Match:    frames {res.start_frame}..{res.end_frame}  "
        f"(t={res.start_frame / res.fps:.2f}s..{res.end_frame / res.fps:.2f}s)"
    )
    if res.review_required or res.score is None:
        source_label = _GATE_SOURCE_LABELS.get(res.multi_person_gate_source or "video", "多人来源")
        print(
            f"Cost:     total={res.cost:.2f}  avg/frame={res.avg_cost:.3f}  "
            f"score=N/A  baseline={res.baseline:.3f}"
        )
        print(
            f"多人场景闸门：{source_label}检出多人（max_persons="
            f"{res.max_persons}，multi_person_frames={res.multi_person_frames}），"
            "降级为『需人工复核』，不产出对外分数。"
        )
    else:
        print(
            f"Cost:     total={res.cost:.2f}  avg/frame={res.avg_cost:.3f}  "
            f"score={res.score:.3f}  baseline={res.baseline:.3f}"
        )
    print(f"Calibration: {res.calibration_status}（仅预览 / 内部标定参考，不得对外评分）")
    if res.valid_frame_ratio is not None:
        print(f"Valid body_core frames: {res.valid_frame_ratio * 100:.1f}%")


def _match_pose33_v3(args, template: Template) -> None:
    # 文件里的 meta 原值：meta=None 的模板与迁移前一样在这里报错（AttributeError），不改错误路径的行为。
    meta = template.declared_meta
    pose_variant = args.pose or meta.get("pose_variant", "full")
    # 归一化只看文件里写的 feature_layout（缺省按最旧的 v1），与模板生成时一致。
    layout = str(meta.get("feature_layout", LEGACY_DEFAULT_FEATURE_LAYOUT))
    normalizer = normalizer_for(normalizer_version_from_layout(layout))

    video_path = Path(args.video)
    seq, fps = _extract_features(video_path, pose_variant=pose_variant, normalizer=normalizer)
    # baseline 来自布局注册表（pose33_v3：avg_cost=0 → 1.0 分，avg_cost=baseline → 0.5 分）。
    (m,) = match_templates(seq, [template], layout=template.layout, baseline=POSE33_BASELINE,
                           video_label=f"video {video_path}")
    start, end = m.start_frame, m.end_frame

    print(f"Template: {args.template}")
    print(f"Video:    {video_path}")
    print(f"Match:    frames {start}..{end}  (t={start/fps:.2f}s..{end/fps:.2f}s)")
    print(f"Cost:     total={m.cost:.2f}  avg/frame={m.avg_cost:.3f}  score={m.score:.3f}")

    if args.preview:
        out_dir = templates_dir()
        out_path = out_dir / f"{video_path.stem}.match_{Path(args.template).stem}.mp4"

        actual_path, codec = write_pose_preview(
            video_path, out_path=out_path, start_frame=start, end_frame=end, pose_variant=pose_variant,
        )
        print(f"Saved match preview: {actual_path} (codec={codec})")
