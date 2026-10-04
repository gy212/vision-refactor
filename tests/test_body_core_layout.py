# -*- coding: utf-8 -*-
"""
body_core_v1 布局 + 离线模板闭环回归（YOLO 迁移 Issue #8 / S2）。

验收目标（对应 Issue #8）
-------------------------
1. ``body_core_v1`` 的 shape / mirror pairs / joint names / layout mismatch 行为正确。
2. 能生成 YOLO ``body_core_v1`` 模板并匹配同一视频，产出预览分数，且该分数 metadata 标
   ``calibration_status=unvalidated``、未进入对外报告（S3 仅落参数，不授权评分）。
3. MediaPipe 旧 ``pose33_v3`` 模板仍能正常比对（不在本文件，由 golden 套件保证；
   此处补一条 smoke：MediaPipe 也能生成 body_core_v1 模板，证明布局共享）。
4. 不同 layout 比对报清晰错误。

确定性 / 无网络
--------------
不下载模型、不读真实视频：YOLO 路径用 ``FakeYoloAdapter`` + ``patch_cv2_capture`` 驱动；
MediaPipe 路径用 golden harness 回放 ``(T,33,4)`` landmark 序列。

验收命令
--------
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_body_core_layout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.action_compare import _assert_feature_layout_match  # noqa: E402
from core.body_core_compare import (  # noqa: E402
    BODY_CORE_V1_CALIBRATED_BASELINE,
    CALIBRATION_STATUS_UNVALIDATED,
    MultiPersonReviewRequiredError,
    create_body_core_template,
    extract_body_core_features,
    match_body_core_template,
)
from core.feature_layout import BODY_CORE_V1, POSE33_V3, get_layout, has_layout  # noqa: E402
from core.pose_features import mirror_pose_features, normalize_pose_body_core_v1  # noqa: E402
from core.yolo_adapter import BODY_CORE_V1_VALID_INDICES  # noqa: E402
from tests import golden_harness as H  # noqa: E402
from tests.yolo_fakes import FakeYoloAdapter, FakeYoloResult, patch_cv2_capture  # noqa: E402

ABS_TOL = 1e-6
FIX_DIR = Path(__file__).resolve().parent / "fixtures" / "pose33_v3"


# --------------------------------------------------------------------------- #
# 1) layout 注册与基本属性
# --------------------------------------------------------------------------- #
def test_body_core_v1_registered_with_shape_and_names():
    assert has_layout("body_core_v1")
    spec = get_layout("body_core_v1")
    assert spec is BODY_CORE_V1
    assert spec.shape == (12, 2)
    assert spec.num_joints == 12
    assert len(spec.source_indices) == 12
    assert spec.source_indices == (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    assert spec.joint_names == (
        "L_SHOULDER",
        "R_SHOULDER",
        "L_ELBOW",
        "R_ELBOW",
        "L_WRIST",
        "R_WRIST",
        "L_HIP",
        "R_HIP",
        "L_KNEE",
        "R_KNEE",
        "L_ANKLE",
        "R_ANKLE",
    )


def test_body_core_v1_baseline_is_calibrated_s3():
    # S3（#10）已标定：layout 默认 baseline = 1.2826（尺度对齐 pose33_v3 反推），
    # 闭环 baseline 取自 layout，二者一致；占位 2.0 已移除。
    assert BODY_CORE_V1.default_baseline == 1.2826
    assert BODY_CORE_V1_CALIBRATED_BASELINE == 1.2826
    assert CALIBRATION_STATUS_UNVALIDATED == "unvalidated"


def test_body_core_v1_mirror_pairs_are_adjacent():
    assert BODY_CORE_V1.mirror_pairs == tuple((i, i + 1) for i in range(0, 12, 2))


def test_yolo_body_core_indices_match_layout_source_indices():
    # yolo_adapter 内的冗余索引必须与 layout.source_indices 完全一致（防漂移）。
    assert tuple(BODY_CORE_V1_VALID_INDICES) == tuple(BODY_CORE_V1.source_indices)


# --------------------------------------------------------------------------- #
# 2) mirror pairs 行为：左右交换 + x 取反
# --------------------------------------------------------------------------- #
def test_mirror_body_core_v1_swaps_left_right():
    rng = np.random.default_rng(20260208)
    feats = rng.standard_normal((12, 2)).astype(np.float32)
    out = mirror_pose_features(feats, layout="body_core_v1")

    assert out.shape == (12, 2)
    # 逐对验证：L 与 R 交换，且 x 取反。
    for a, b in BODY_CORE_V1.mirror_pairs:
        assert out[a, 0] == pytest.approx(-feats[b, 0], abs=ABS_TOL)
        assert out[a, 1] == pytest.approx(feats[b, 1], abs=ABS_TOL)
        assert out[b, 0] == pytest.approx(-feats[a, 0], abs=ABS_TOL)
        assert out[b, 1] == pytest.approx(feats[a, 1], abs=ABS_TOL)


def test_mirror_body_core_v1_resolved_by_shape():
    # (12,2) 唯一对应 body_core_v1，未显式传 layout 也能按 shape 反查。
    rng = np.random.default_rng(7)
    feats = rng.standard_normal((12, 2)).astype(np.float32)
    by_name = mirror_pose_features(feats, layout="body_core_v1")
    by_shape = mirror_pose_features(feats)
    assert np.allclose(by_name, by_shape, atol=ABS_TOL)


# --------------------------------------------------------------------------- #
# 3) 共享 normalizer：MediaPipe landmark 对象 与 YOLO (33,4) 行 都能产出 (12,2)
# --------------------------------------------------------------------------- #
class _LM:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)


def _fake_blaze33_landmarks(seed: int = 1):
    rng = np.random.default_rng(seed)
    lms = []
    for i in range(33):
        # 给一个有躯干结构的姿态：肩在上、髋在下。
        x = 0.5 + 0.1 * rng.standard_normal()
        y = 0.3 + 0.01 * i
        lms.append(_LM(x, y, 0.0, 1.0))
    return lms


def test_normalizer_accepts_mediapipe_landmarks():
    lms = _fake_blaze33_landmarks()
    out = normalize_pose_body_core_v1(lms)
    assert out is not None
    assert out.shape == (12, 2)
    assert np.isfinite(out).all()


def test_normalizer_accepts_yolo_row():
    # YOLO (33,4) 行：给 body_core 的 12 个 BlazePose 索引填真实坐标。
    row = np.zeros((33, 4), dtype=np.float32)
    rng = np.random.default_rng(3)
    for idx in BODY_CORE_V1.source_indices:
        row[idx, 0] = 0.5 + 0.1 * rng.standard_normal()
        row[idx, 1] = 0.2 + 0.02 * idx
        row[idx, 3] = 0.9
    out = normalize_pose_body_core_v1(row)
    assert out is not None
    assert out.shape == (12, 2)
    assert np.isfinite(out).all()


def test_normalizer_returns_none_on_garbage():
    assert normalize_pose_body_core_v1(None) is None
    assert normalize_pose_body_core_v1(np.zeros((10, 4), dtype=np.float32)) is None


# --------------------------------------------------------------------------- #
# 4) layout mismatch 报清晰错误（不静默比对）
# --------------------------------------------------------------------------- #
def test_layout_mismatch_raises_clear_error():
    body_core = np.zeros((8, 12, 2), dtype=np.float32)
    pose33 = np.zeros((10, 22, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="Feature layout mismatch"):
        _assert_feature_layout_match(
            body_core,
            pose33,
            left_label="body_core template",
            right_label="pose33 video",
            left_layout="body_core_v1",
            right_layout="pose33_v3",
        )


def test_match_rejects_non_body_core_template(tmp_path):
    # 用一个 pose33_v3 的 fake 模板喂给 body_core 匹配入口，应报清晰错误。
    bad_tpl = tmp_path / "pose33_tpl.npz"
    meta = {"feature_layout": POSE33_V3.name, "backend": "mediapipe"}
    np.savez_compressed(
        bad_tpl,
        features=np.zeros((5, 22, 2), dtype=np.float32),
        meta=np.array(meta, dtype=object),
    )
    with pytest.raises(ValueError, match="body_core_v1"):
        match_body_core_template(bad_tpl, "fake://video.mp4", backend="mediapipe")


# --------------------------------------------------------------------------- #
# 5) YOLO body_core_v1 离线闭环：生成模板 → 匹配同一视频 → 未标定分数
# --------------------------------------------------------------------------- #
def _make_body_core_coco17(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """合成 COCO17：body_core 需要的 COCO 5..16 给高置信度真实点。"""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.3, 0.7, size=(17, 2)).astype(np.float32)
    # 给一个有结构的躯干（肩 5/6 在上、髋 11/12 在下、踝 15/16 最下）。
    layout_rows = {
        5: (0.45, 0.30), 6: (0.55, 0.30),
        7: (0.40, 0.45), 8: (0.60, 0.45),
        9: (0.38, 0.60), 10: (0.62, 0.60),
        11: (0.46, 0.55), 12: (0.54, 0.55),
        13: (0.45, 0.75), 14: (0.55, 0.75),
        15: (0.45, 0.95), 16: (0.55, 0.95),
    }
    for c, (x, y) in layout_rows.items():
        xy[c] = (x, y)
    conf = np.full((17,), 0.9, dtype=np.float32)
    return xy, conf


def _periodic_yolo_frames(n: int):
    """构造 n 帧带轻微周期摆动的单人 YOLO 结果。"""
    frames = []
    for t in range(n):
        xy, conf = _make_body_core_coco17(seed=0)
        # 让腕部随时间摆动，制造可被 DTW 匹配的运动。
        swing = 0.1 * np.sin(2.0 * np.pi * t / 15.0)
        xy[9] = (0.38 + swing, 0.60)
        xy[10] = (0.62 - swing, 0.60)
        frames.append(FakeYoloResult.single(xy, conf))
    return frames


def test_yolo_body_core_template_and_match_closed_loop(tmp_path):
    n = 60
    out_tpl = tmp_path / "yolo_body_core_v1.npz"

    # 1) 生成 YOLO body_core_v1 模板（fake adapter，无模型/无视频）。
    frames_make = _periodic_yolo_frames(n)
    adapter_make = FakeYoloAdapter(frames=frames_make, valid_conf_thr=0.5)
    with patch_cv2_capture(n_frames=n, fps=30.0):
        tpl_path = create_body_core_template(
            "fake://std.mp4",
            backend="yolo",
            out_path=out_tpl,
            yolo_model=adapter_make,
        )
    assert tpl_path.exists()

    d = np.load(tpl_path, allow_pickle=True)
    meta = dict(d["meta"].item())
    feats = d["features"]
    assert feats.ndim == 3 and feats.shape[1:] == (12, 2)
    assert meta["raw_layout"] == "pose33_like_coco17"
    assert meta["feature_layout"] == "body_core_v1"
    assert meta["capability"] == "body_only"
    assert meta["backend"] == "yolo"
    # S3（#10）只落参数，不授权评分；分数仍不得进入对外报告。
    assert meta["calibration_status"] == CALIBRATION_STATUS_UNVALIDATED
    assert meta["baseline_calibrated"] is True
    assert meta["score_authorized"] is False
    assert meta["display_scope"] == "internal"
    assert "calibration_note" in meta

    # 2) 用该模板匹配同一视频，产出分数。
    frames_match = _periodic_yolo_frames(n)
    adapter_match = FakeYoloAdapter(frames=frames_match, valid_conf_thr=0.5)
    with patch_cv2_capture(n_frames=n, fps=30.0):
        res = match_body_core_template(
            tpl_path,
            "fake://std.mp4",
            backend="yolo",
            yolo_model=adapter_match,
        )

    assert res.feature_layout == "body_core_v1"
    assert res.backend == "yolo"
    assert np.isfinite(res.score)
    assert 0.0 <= res.score <= 1.0
    # 同一视频自匹配：分数应较高（avg_cost 较小）。
    assert res.score > 0.5
    # 关键：结果标未标定，分数不得作为对外评分。
    assert res.calibration_status == CALIBRATION_STATUS_UNVALIDATED


def test_yolo_body_core_ignores_invalid_core_frame():
    xy0, conf0 = _make_body_core_coco17(seed=0)
    xy_bad, conf_bad = _make_body_core_coco17(seed=0)
    xy_bad[9] = (0.95, 0.95)
    conf_bad[9] = 0.1
    xy2, conf2 = _make_body_core_coco17(seed=0)
    xy2[9] = (0.20, 0.60)
    frames = [
        FakeYoloResult.single(xy0, conf0),
        FakeYoloResult.single(xy_bad, conf_bad),
        FakeYoloResult.single(xy2, conf2),
    ]

    adapter = FakeYoloAdapter(frames=frames, valid_conf_thr=0.5)
    with patch_cv2_capture(n_frames=len(frames), fps=30.0):
        features, _fps, meta = extract_body_core_features(
            "fake://low_conf.mp4",
            backend="yolo",
            yolo_model=adapter,
        )

    assert features.shape == (3, 12, 2)
    # 第 2 帧有 body_core 低置信点，必须按 valid_mask 当作缺帧，沿用上一帧特征。
    assert np.allclose(features[1], features[0], atol=ABS_TOL)
    assert not np.allclose(features[2], features[1], atol=ABS_TOL)
    assert meta["body_core_valid_frame_ratio"] == pytest.approx(2.0 / 3.0, abs=ABS_TOL)


def test_create_body_core_template_start_end_slices_full_sequence(tmp_path):
    n = 60
    out_tpl = tmp_path / "sliced_body_core_v1.npz"
    adapter = FakeYoloAdapter(frames=_periodic_yolo_frames(n), valid_conf_thr=0.5)

    with patch_cv2_capture(n_frames=n, fps=30.0):
        tpl_path = create_body_core_template(
            "fake://std.mp4",
            backend="yolo",
            start=10,
            end=20,
            out_path=out_tpl,
            yolo_model=adapter,
        )

    d = np.load(tpl_path, allow_pickle=True)
    meta = dict(d["meta"].item())
    feats = d["features"]
    assert feats.shape[0] == 11
    assert meta["frame_count"] == n
    assert meta["start_frame"] == 10
    assert meta["end_frame"] == 20


def test_yolo_backend_rejects_pose33_layout_template(tmp_path):
    # YOLO 只允许 body_core_v1：拿 pose33_v3 模板用 yolo 后端匹配应报错。
    bad_tpl = tmp_path / "pose33.npz"
    np.savez_compressed(
        bad_tpl,
        features=np.zeros((5, 22, 2), dtype=np.float32),
        meta=np.array({"feature_layout": "pose33_v3", "backend": "mediapipe"}, dtype=object),
    )
    with pytest.raises(ValueError, match="body_core_v1"):
        match_body_core_template(bad_tpl, "fake://v.mp4", backend="yolo")


# --------------------------------------------------------------------------- #
# 6) MediaPipe 也能生成 body_core_v1 模板（证明布局共享，供 #10 三方对比）
# --------------------------------------------------------------------------- #
def _load_raw(name: str) -> np.ndarray:
    d = np.load(FIX_DIR / name, allow_pickle=True)
    return np.asarray(d["landmarks"], dtype=np.float32)


def test_mediapipe_can_generate_body_core_v1_template(tmp_path):
    front_raw = _load_raw("front_src_raw.npz")
    out_tpl = tmp_path / "mp_body_core_v1.npz"

    H.clear_registry()
    H.register_video("golden://front_src.mp4", front_raw, fps=30.0)
    try:
        with H.replay_context():
            tpl_path = create_body_core_template(
                "golden://front_src.mp4",
                backend="mediapipe",
                pose_variant="full",
                out_path=out_tpl,
            )
    finally:
        H.clear_registry()

    d = np.load(tpl_path, allow_pickle=True)
    meta = dict(d["meta"].item())
    feats = d["features"]
    assert feats.ndim == 3 and feats.shape[1:] == (12, 2)
    assert meta["feature_layout"] == "body_core_v1"
    assert meta["backend"] == "mediapipe"
    assert meta["calibration_status"] == CALIBRATION_STATUS_UNVALIDATED


# --------------------------------------------------------------------------- #
# 7) 多人场景闸门（Issue #9 / S2）：拒绝 / 降级出分，不静默选最大框
# --------------------------------------------------------------------------- #
def _multi_person_body_core_frame(seed: int = 0):
    """构造一帧两人（两个不同大小框）的 body_core YOLO 结果。"""
    xy, conf = _make_body_core_coco17(seed=seed)
    return FakeYoloResult.multi(
        [
            (xy, conf, (0.3, 0.5, 0.2, 0.4)),   # 较小框
            (xy, conf, (0.65, 0.5, 0.5, 0.9)),  # 较大框 → 会被最大框选中
        ]
    )


def _yolo_body_core_template(tmp_path, n=60):
    out_tpl = tmp_path / "yolo_body_core_v1.npz"
    adapter_make = FakeYoloAdapter(frames=_periodic_yolo_frames(n), valid_conf_thr=0.5)
    with patch_cv2_capture(n_frames=n, fps=30.0):
        return create_body_core_template(
            "fake://std.mp4", backend="yolo", out_path=out_tpl, yolo_model=adapter_make
        )


def test_match_rejects_multi_person_video_by_default(tmp_path):
    # 默认 reject_multi_person=True：多人视频抛 MultiPersonReviewRequiredError，
    # 不混入正常评分结果。
    tpl_path = _yolo_body_core_template(tmp_path)

    n = 30
    frames = [_periodic_yolo_frames(1)[0] for _ in range(n)]
    frames[10] = _multi_person_body_core_frame()   # 插入一帧多人
    frames[20] = _multi_person_body_core_frame()
    adapter = FakeYoloAdapter(frames=frames, valid_conf_thr=0.5)

    with patch_cv2_capture(n_frames=n, fps=30.0):
        with pytest.raises(MultiPersonReviewRequiredError) as ei:
            match_body_core_template(tpl_path, "fake://multi.mp4", backend="yolo", yolo_model=adapter)
    assert ei.value.max_persons == 2
    assert ei.value.multi_person_frames == 2


def test_match_degrades_multi_person_video_when_not_rejecting(tmp_path):
    # reject_multi_person=False：降级——返回 score=None + review_required=True，
    # 仍不产出对外分数。
    tpl_path = _yolo_body_core_template(tmp_path)

    n = 30
    frames = [_periodic_yolo_frames(1)[0] for _ in range(n)]
    frames[5] = _multi_person_body_core_frame()
    adapter = FakeYoloAdapter(frames=frames, valid_conf_thr=0.5)

    with patch_cv2_capture(n_frames=n, fps=30.0):
        res = match_body_core_template(
            tpl_path,
            "fake://multi.mp4",
            backend="yolo",
            yolo_model=adapter,
            reject_multi_person=False,
        )

    assert res.review_required is True
    assert res.multi_person_detected is True
    assert res.max_persons == 2
    assert res.multi_person_frames == 1
    # 降级模式不产出分数，避免多人视频混入正常评分。
    assert res.score is None
    assert res.calibration_status == CALIBRATION_STATUS_UNVALIDATED


def test_match_single_person_video_unaffected_by_gate(tmp_path):
    # 单人样本不受影响：照常出分，review_required=False。
    tpl_path = _yolo_body_core_template(tmp_path)

    n = 60
    adapter = FakeYoloAdapter(frames=_periodic_yolo_frames(n), valid_conf_thr=0.5)
    with patch_cv2_capture(n_frames=n, fps=30.0):
        res = match_body_core_template(
            tpl_path, "fake://single.mp4", backend="yolo", yolo_model=adapter
        )

    assert res.review_required is False
    assert res.multi_person_detected is False
    assert res.score is not None
    assert 0.0 <= res.score <= 1.0


def test_multi_person_template_meta_flags_review_required(tmp_path):
    # 生成模板的视频若多人，模板 meta 也应透传 multi_person_detected / review_required，
    # 便于下游识别该模板来源不可信。
    out_tpl = tmp_path / "multi_tpl.npz"
    n = 30
    frames = _periodic_yolo_frames(n)
    frames[12] = _multi_person_body_core_frame()
    adapter = FakeYoloAdapter(frames=frames, valid_conf_thr=0.5)
    with patch_cv2_capture(n_frames=n, fps=30.0):
        tpl_path = create_body_core_template(
            "fake://multi_std.mp4", backend="yolo", out_path=out_tpl, yolo_model=adapter
        )

    meta = dict(np.load(tpl_path, allow_pickle=True)["meta"].item())
    assert meta["multi_person_detected"] is True
    assert meta["review_required"] is True
    assert meta["max_persons"] == 2


def test_match_rejects_multi_person_template_by_default(tmp_path):
    # 多人来源模板本身也不得进入正常评分；默认拒绝时无需等目标视频推理完成。
    out_tpl = tmp_path / "multi_tpl.npz"
    n = 30
    frames = _periodic_yolo_frames(n)
    frames[7] = _multi_person_body_core_frame()
    adapter_make = FakeYoloAdapter(frames=frames, valid_conf_thr=0.5)
    with patch_cv2_capture(n_frames=n, fps=30.0):
        tpl_path = create_body_core_template(
            "fake://multi_std.mp4", backend="yolo", out_path=out_tpl, yolo_model=adapter_make
        )

    with pytest.raises(MultiPersonReviewRequiredError) as ei:
        match_body_core_template(tpl_path, "fake://single.mp4", backend="yolo")

    assert ei.value.gate_source == "template"
    assert ei.value.max_persons == 2
    assert ei.value.multi_person_frames == 1
