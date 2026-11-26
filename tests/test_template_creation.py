# -*- coding: utf-8 -*-
"""模板建库服务、质量体检闸门、模板池契约校验与 make_template CLI 的回归测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apps.cli import make_template as make_template_cli  # noqa: E402
from core.evaluation import LEGACY_DEFAULT_FEATURE_LAYOUT, match_video  # noqa: E402
from core.evaluation.templates import LEGACY_POSE33_V3_FEATURE_LAYOUT  # noqa: E402
from core.evaluation.scoring import exam_pool  # noqa: E402
from core.pose import POSE33_V3, extract as pose_extract  # noqa: E402
from core.services import (  # noqa: E402
    MIN_TEMPLATE_FRAMES,
    TemplateQualityError,
    create_template_from_video,
    create_template_pair,
    discover_recording_videos,
    planned_template_paths,
    sanitize_action_name,
)
from core.services import templates as services_templates  # noqa: E402
from tests import golden_harness as H  # noqa: E402

FIX_DIR = Path(__file__).resolve().parent / "fixtures" / "pose33_v3"


def _load_raw(name: str) -> np.ndarray:
    return np.asarray(
        np.load(FIX_DIR / name, allow_pickle=True)["landmarks"], dtype=np.float32
    )


def _write_tpl(
    path: Path,
    *,
    shape: tuple[int, ...] = (10, 22, 2),
    layout: str = POSE33_V3.name,
    normalizer_version: str = "v3",
    pose_variant: str = "heavy",
) -> Path:
    meta = {
        "feature_layout": layout,
        "normalizer_version": normalizer_version,
        "pose_variant": pose_variant,
        "running_mode": "video",
    }
    np.savez_compressed(
        path,
        features=np.zeros(shape, dtype=np.float32),
        meta=np.array(meta, dtype=object),
    )
    return path


# --------------------------------------------------------------------------- #
# 1) 录制目录发现与动作名/路径规划
# --------------------------------------------------------------------------- #
def test_discover_recording_videos_handles_mp4_avi_single_and_errors(
    tmp_path: Path,
) -> None:
    dual_dir = tmp_path / "dual"
    dual_dir.mkdir()
    (dual_dir / "front.avi").write_bytes(b"a")
    (dual_dir / "front.mp4").write_bytes(b"b")
    (dual_dir / "side.avi").write_bytes(b"c")
    pair = discover_recording_videos(dual_dir)
    assert pair.front == dual_dir / "front.mp4"
    assert pair.side == dual_dir / "side.avi"

    single_file = tmp_path / "record_01.mp4"
    single_file.write_bytes(b"x")
    single_pair = discover_recording_videos(single_file)
    assert single_pair.front == single_file and single_pair.side is None

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="front.mp4"):
        discover_recording_videos(empty_dir)

    with pytest.raises(FileNotFoundError):
        discover_recording_videos(tmp_path / "missing_path")


@pytest.mark.parametrize(
    "bad_name",
    [
        "",
        "   ",
        ".",
        "..",
        "a/b",
        "a\\b",
        "a:b",
        "a*b",
        'a"b',
        "a<b",
        "a>b",
        "a|b",
        "a..b",
    ],
)
def test_sanitize_action_name_rejects_empty_and_illegal(bad_name: str) -> None:
    with pytest.raises(ValueError):
        sanitize_action_name(bad_name)


def test_planned_template_paths_uses_view_suffixes_without_heavy(
    tmp_path: Path,
) -> None:
    paths = planned_template_paths(
        "  高鞭腿 ", views=["front", "side", "single"], out_dir=tmp_path
    )
    assert paths == {
        "front": tmp_path / "高鞭腿_正面.npz",
        "side": tmp_path / "高鞭腿_侧面.npz",
        "single": tmp_path / "高鞭腿.npz",
    }
    with pytest.raises(ValueError, match="视角"):
        planned_template_paths("直拳", views=["unknown"], out_dir=tmp_path)


# --------------------------------------------------------------------------- #
# 2) create_template_from_video：起止帧校验、单线程强制、质量体检与原子写盘
# --------------------------------------------------------------------------- #
def test_create_template_validates_bounds_and_supports_single_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feats = np.ones((40, 22, 2), dtype=np.float32)
    monkeypatch.setattr(
        services_templates,
        "extract_features",
        lambda *_a, **_k: (feats, 30.0, None),
    )

    for bad_kw in ({"start": -1}, {"end": 40}, {"start": 25, "end": 10}):
        with pytest.raises(ValueError):
            create_template_from_video("v.mp4", out_path=tmp_path / "bad.npz", **bad_kw)

    p_head = create_template_from_video(
        "v.mp4", start=15, out_path=tmp_path / "head.npz"
    )
    with np.load(p_head, allow_pickle=True) as d:
        meta = dict(d["meta"].item())
        assert d["features"].shape[0] == 25
        assert (meta["start_frame"], meta["end_frame"]) == (15, 39)

    p_tail = create_template_from_video("v.mp4", end=19, out_path=tmp_path / "tail.npz")
    with np.load(p_tail, allow_pickle=True) as d:
        meta = dict(d["meta"].item())
        assert d["features"].shape[0] == 20
        assert (meta["start_frame"], meta["end_frame"]) == (0, 19)


def test_create_template_no_auto_crop_keeps_full_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 构造 40s 稀疏多活动序列：开启 auto_crop 会触发复核，关闭 auto_crop 应直接保留整段。
    energy = np.full(40 * 30 - 1, 0.002, dtype=np.float32)
    energy[100:140] = 0.2
    energy[700:740] = 0.2
    values = np.concatenate(
        [np.zeros(1, dtype=np.float32), np.cumsum(energy, dtype=np.float32)]
    )
    feats = np.broadcast_to(values[:, None, None], (values.size, 22, 2)).copy()
    monkeypatch.setattr(
        services_templates,
        "extract_features",
        lambda *_a, **_k: (feats, 30.0, None),
    )

    out_path = create_template_from_video(
        "long.mp4", auto_crop=False, out_path=tmp_path / "full.npz"
    )
    with np.load(out_path, allow_pickle=True) as d:
        meta = dict(d["meta"].item())
        assert d["features"].shape[0] == feats.shape[0]
        assert meta["crop_selection"] == "full"


def test_create_template_forces_sequential_extraction_even_if_workers_gt_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    front_raw = _load_raw("front_src_raw.npz")
    seen_workers: list[int] = []
    orig_extract_seq = pose_extract.extract_pose_sequence

    def spy_extract_seq(
        video, engine, profile=pose_extract.FORMAL, *, workers=1, **kwargs
    ):
        seen_workers.append(int(workers))
        return orig_extract_seq(video, engine, profile, workers=workers, **kwargs)

    monkeypatch.setattr(services_templates, "extract_pose_sequence", spy_extract_seq)
    H.clear_registry()
    H.register_video("golden://seq_only.mp4", front_raw, fps=30.0)
    try:
        with H.replay_context():
            create_template_from_video(
                "golden://seq_only.mp4",
                pose_variant="heavy",
                workers=8,
                out_path=tmp_path / "seq.npz",
            )
    finally:
        H.clear_registry()

    assert seen_workers == [1]


def test_create_template_quality_gate_rejects_short_empty_and_low_core_valid(
    tmp_path: Path,
) -> None:
    front_raw = _load_raw("front_src_raw.npz")
    empty_raw = np.zeros((30, 33, 4), dtype=np.float32)

    low_core_raw = front_raw[:30].copy()
    low_core_raw[:15, 11, 3] = 0.0  # 一半帧左肩失效 -> core_valid_ratio = 0.5 < 0.6

    H.clear_registry()
    H.register_video("golden://front_ok.mp4", front_raw, fps=30.0)
    H.register_video("golden://empty.mp4", empty_raw, fps=30.0)
    H.register_video("golden://low_core.mp4", low_core_raw, fps=30.0)
    try:
        with H.replay_context():
            with pytest.raises(TemplateQualityError, match="过短") as short_exc:
                create_template_from_video(
                    "golden://front_ok.mp4",
                    start=0,
                    end=MIN_TEMPLATE_FRAMES - 2,
                    out_path=tmp_path / "short.npz",
                )
            assert short_exc.value.report is not None

            with pytest.raises(TemplateQualityError, match="全为空帧"):
                create_template_from_video(
                    "golden://empty.mp4",
                    auto_crop=False,
                    out_path=tmp_path / "empty.npz",
                )

            with pytest.raises(TemplateQualityError, match="质量不达标") as low_exc:
                create_template_from_video(
                    "golden://low_core.mp4",
                    auto_crop=False,
                    out_path=tmp_path / "low_core.npz",
                )
            assert low_exc.value.report is not None
            assert low_exc.value.report.core_valid_ratio == pytest.approx(0.5)
    finally:
        H.clear_registry()

    assert not list(tmp_path.glob("*.npz"))


def test_create_template_records_limb_warning_and_respects_overwrite_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    front_raw = _load_raw("front_src_raw.npz")[:30].copy()
    front_raw[:, 13:17, 3] = 0.1  # 上肢肘腕低置信度，核心躯干 100% 有效 -> 放行但写告警
    out_path = tmp_path / "warn.npz"

    H.clear_registry()
    H.register_video("golden://limb_warn.mp4", front_raw, fps=30.0)
    try:
        with H.replay_context():
            saved = create_template_from_video(
                "golden://limb_warn.mp4",
                auto_crop=False,
                out_path=out_path,
            )
    finally:
        H.clear_registry()

    with np.load(saved, allow_pickle=True) as d:
        meta = dict(d["meta"].item())
    assert meta["template_quality"]["core_valid_ratio"] == pytest.approx(1.0)
    assert any("四肢" in w for w in meta["template_quality"]["warnings"])
    assert not list(tmp_path.glob("*.tmp"))

    # overwrite=False 在推理提取之前快速失败
    monkeypatch.setattr(
        services_templates,
        "extract_pose_sequence",
        lambda *_a, **_k: pytest.fail("overwrite=False 命中已存在文件时不该启动推理"),
    )
    with pytest.raises(FileExistsError):
        create_template_from_video(
            "golden://limb_warn.mp4", overwrite=False, out_path=out_path
        )


# --------------------------------------------------------------------------- #
# 3) create_template_pair：双摄时间转帧、全有或全无原子落盘与预览
# --------------------------------------------------------------------------- #
def test_create_template_pair_dual_fps_conversion_and_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    front_raw = _load_raw("front_src_raw.npz")[:60]  # 2.0s @ 30fps
    side_raw = _load_raw("side_src_raw.npz")[:30]  # 2.0s @ 15fps
    preview_calls: list[tuple[Path, int, int]] = []

    def fake_preview(video_path, *, out_path, start_frame, end_frame, **_k):
        Path(out_path).write_bytes(b"preview")
        preview_calls.append((Path(out_path), start_frame, end_frame))
        return Path(out_path)

    monkeypatch.setattr(services_templates, "export_match_preview", fake_preview)
    H.clear_registry()
    H.register_video("golden://pair_front.mp4", front_raw, fps=30.0)
    H.register_video("golden://pair_side.mp4", side_raw, fps=15.0)
    try:
        with H.replay_context():
            results = create_template_pair(
                {"front": "golden://pair_front.mp4", "side": "golden://pair_side.mp4"},
                action_name="高鞭腿",
                start_s=0.5,
                end_s=1.5,
                out_dir=tmp_path,
                preview=True,
            )
    finally:
        H.clear_registry()

    by_view = {r.view: r for r in results}
    assert by_view["front"].path == tmp_path / "高鞭腿_正面.npz"
    assert (by_view["front"].start_frame, by_view["front"].end_frame) == (15, 45)
    assert by_view["side"].path == tmp_path / "高鞭腿_侧面.npz"
    assert (by_view["side"].start_frame, by_view["side"].end_frame) == (8, 22)
    assert len(preview_calls) == 2
    assert by_view["front"].preview_path == tmp_path / "高鞭腿_正面.preview.mp4"


def test_create_template_pair_is_all_or_nothing_on_quality_failure(
    tmp_path: Path,
) -> None:
    front_raw = _load_raw("front_src_raw.npz")[:30]
    empty_side = np.zeros((30, 33, 4), dtype=np.float32)

    H.clear_registry()
    H.register_video("golden://pair_good_f.mp4", front_raw, fps=30.0)
    H.register_video("golden://pair_bad_s.mp4", empty_side, fps=30.0)
    try:
        with H.replay_context():
            with pytest.raises(TemplateQualityError):
                create_template_pair(
                    {
                        "front": "golden://pair_good_f.mp4",
                        "side": "golden://pair_bad_s.mp4",
                    },
                    action_name="直拳",
                    out_dir=tmp_path,
                )
    finally:
        H.clear_registry()

    assert not (tmp_path / "直拳_正面.npz").exists()
    assert not (tmp_path / "直拳_侧面.npz").exists()


# --------------------------------------------------------------------------- #
# 4) exam_pool 视角后缀剥离 & matching.engine 混用拦截 & CLI 收口
# --------------------------------------------------------------------------- #
def test_exam_pool_strips_view_suffix_for_squeeze_kick_and_rules(
    tmp_path: Path,
) -> None:
    lo, hi = exam_pool.SCORE_SQUEEZE["背后七颠"]
    assert exam_pool.action_score("背后七颠_正面", 1.0, None) == pytest.approx(hi)
    assert exam_pool.action_score("背后七颠_侧面", 0.0, None) == pytest.approx(lo)
    assert exam_pool.rule_action_for("上步前后手直拳组合_正面") == "step_straight_combo"

    tpl = _write_tpl(tmp_path / "高鞭腿_侧面.npz")
    raw_calls: list[Path] = []

    def fake_raw(video, **_kw):
        raw_calls.append(Path(video))
        lm = np.zeros((12, 33, 4), dtype=np.float32)
        lm[:, :, 3] = 0.95
        lm[:, 0, :2] = (0.5, 0.2)  # NOSE
        lm[:, 23, :2] = (0.48, 0.5)  # L_HIP
        lm[:, 24, :2] = (0.52, 0.5)  # R_HIP
        lm[:, 25, :2] = (0.48, 0.7)  # L_KNEE
        lm[:, 27, :2] = (0.48, 0.9)  # L_ANKLE (站立腿)
        for t in range(12):
            phase = min(t, 11 - t) / 5.0
            lm[t, 26, :2] = (0.54 + 0.1 * phase, 0.7 - 0.3 * phase)  # R_KNEE
            lm[t, 28, :2] = (
                0.54 + 0.2 * phase,
                0.9 - 0.7 * phase,
            )  # R_ANKLE (高鞭踢起)
        return SimpleNamespace(
            landmarks=lm,
            meta={
                "video": str(video),
                "fps": 30.0,
                "frame_count": 12,
                "width": 1280,
                "height": 720,
                "start_frame": 0,
                "valid_conf_thr": 0.5,
            },
        )

    def fake_compare(templates, video, **_kw):
        return [
            SimpleNamespace(score=0.8, start_frame=0, end_frame=10) for _ in templates
        ]

    details, _mean, _targets = exam_pool.score_pool(
        [tpl],
        Path("side.mp4"),
        "side",
        compare=fake_compare,
        raw_series=fake_raw,
        stop_evt=SimpleNamespace(is_set=lambda: False),
    )
    assert len(raw_calls) == 1
    assert details[0]["geom_score"] is not None


def test_match_video_rejects_mixed_pool_variants_normalizers_and_layouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core.evaluation.matching import engine as match_engine

    monkeypatch.setattr(
        match_engine,
        "template_features",
        lambda *_a, **_k: (
            np.zeros((20, 22, 2), dtype=np.float32),
            30.0,
            None,
            "heavy",
            1,
        ),
    )

    t_v3_heavy = _write_tpl(
        tmp_path / "a.npz", layout=POSE33_V3.name, pose_variant="heavy"
    )
    t_alias_heavy = _write_tpl(
        tmp_path / "alias.npz",
        layout=LEGACY_POSE33_V3_FEATURE_LAYOUT,
        pose_variant="heavy",
    )
    t_v1_heavy = _write_tpl(
        tmp_path / "v1.npz",
        layout=LEGACY_DEFAULT_FEATURE_LAYOUT,
        normalizer_version="v1",
        pose_variant="heavy",
    )
    t_v3_lite = _write_tpl(
        tmp_path / "lite.npz", layout=POSE33_V3.name, pose_variant="lite"
    )
    t_body_core = _write_tpl(
        tmp_path / "bc.npz",
        shape=(10, 12, 2),
        layout="body_core_v1",
        pose_variant="heavy",
    )

    # 同档位同归一化且互为布局别名时允许通过
    res = match_video([t_v3_heavy, t_alias_heavy], "v.mp4")
    assert len(res) == 2

    with pytest.raises(ValueError, match="Feature layout mismatch"):
        match_video([t_v3_heavy, t_v1_heavy], "v.mp4")

    with pytest.raises(ValueError, match="Feature layout mismatch"):
        match_video([t_v3_heavy, t_body_core], "v.mp4")

    with pytest.raises(ValueError, match="pose_variant mismatch"):
        match_video([t_v3_heavy, t_v3_lite], "v.mp4")


def test_make_template_cli_defaults_to_heavy_and_forwards_no_auto_crop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}
    out_file = tmp_path / "out.npz"

    def fake_create(video_path, **kwargs):
        captured["video_path"] = video_path
        captured.update(kwargs)
        _write_tpl(out_file, pose_variant=kwargs["pose_variant"])
        return out_file

    monkeypatch.setattr(make_template_cli, "create_template_from_video", fake_create)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_template.py",
            "--video",
            "demo.mp4",
            "--out",
            str(out_file),
            "--no-auto-crop",
        ],
    )
    make_template_cli.main()

    assert captured["pose_variant"] == "heavy"
    assert captured["auto_crop"] is False
