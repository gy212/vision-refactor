# -*- coding: utf-8 -*-
"""
YOLO 后端契约回归（YOLO 迁移 Issue #7 / S2）。

验收目标（对应 Issue #7）
-------------------------
- 无人帧 / 空结果 → 零行 + 全 False mask，不崩。
- track_id 在段边界重置（``reset_tracker`` 语义）；序列层每次调用都重置。
- 序列层输出仅为 numpy 数组（landmarks/valid_mask），meta 携带规定字段
  （backend/model_name/running_mode/confidence_kind/validity_policy/valid_conf_thr
   /frame_count/fps + calibration/track_reset note）。
- 暴露 ``num_persons``，并据此判定多人闸门（Issue #9）：``num_persons>1`` 触发
  ``multi_person_detected`` / ``review_required`` / ``gate_status``；单人不受影响。
- 懒加载：导入本模块/adapter 不触发 ultralytics import。

确定性 / 无网络
--------------
不下载模型、不读真实视频：用 ``FakeYoloAdapter`` + ``patch_cv2_capture`` 驱动序列层，
用 ``FakeYoloResult`` 驱动单帧解析。

验收命令
--------
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_yolo_backend_contract.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.yolo_adapter import (  # noqa: E402
    GATE_STATUS_MULTI_PERSON,
    GATE_STATUS_OK,
    SingleTargetTracker,
    YoloPoseAdapter,
    empty_frame_result,
    empty_sequence_row,
    evaluate_multi_person_gate,
    extract_yolo_landmark_series,
    select_main_person,
    extract_persons,
    yolo_result_to_arrays,
    yolo_result_to_frame,
)
from tests.yolo_fakes import (  # noqa: E402
    FakeYoloAdapter,
    FakeYoloResult,
    make_coco17,
    patch_cv2_capture,
)


# --------------------------------------------------------------------------- #
# 懒加载：导入 adapter 不应触发 ultralytics import
# --------------------------------------------------------------------------- #
def test_adapter_import_is_lazy():
    # 本测试进程中导入 core.yolo_adapter 后，ultralytics 不应进入 sys.modules
    # （除非别的依赖显式引入；本仓库 fake 路径不引入）。
    assert "core.yolo_adapter" in sys.modules
    assert "ultralytics" not in sys.modules


def test_realtime_annotate_returns_body_only_preview_meta_without_authorizing_score():
    xy, conf = make_coco17(conf_value=0.9)
    frame_result = FakeYoloResult.multi(
        [
            (xy, conf, (0.3, 0.5, 0.2, 0.4)),
            (xy, conf, (0.6, 0.5, 0.5, 0.9)),
        ]
    )

    class StaticYoloAdapter(YoloPoseAdapter):
        def __init__(self):
            super().__init__(model_path="fake-yolo26n-pose.pt")

        def _predict_raw(self, frame):
            return frame_result

    adapter = StaticYoloAdapter()
    annotated, actions, meta = adapter.annotate(np.zeros((64, 64, 3), dtype=np.uint8))

    assert annotated.shape == (64, 64, 3)
    assert isinstance(actions, list)
    assert meta["backend"] == "yolo"
    assert meta["model_name"] == "fake-yolo26n-pose.pt"
    assert meta["running_mode"] == "realtime_preview"
    assert meta["raw_layout"] == "pose33_like_coco17"
    assert meta["feature_layout"] == "body_core_v1"
    assert meta["score_authorized"] is False
    assert meta["calibration_status"] == "unvalidated"
    assert meta["display_scope"] == "limited"
    assert meta["multi_person_detected"] is True
    assert meta["person_count"] == 2
    assert meta["review_required"] is True
    assert meta["target_policy"] == "select_main_person_largest_box_highest_score"


# --------------------------------------------------------------------------- #
# 无人帧 / 空结果
# --------------------------------------------------------------------------- #
def test_empty_sequence_row_zero_and_false():
    row, valid = empty_sequence_row()
    assert row.shape == (33, 4)
    assert valid.shape == (33,)
    assert np.all(row == 0.0)
    assert not np.any(valid)


def test_no_person_frame_result_pose_none():
    fr = empty_frame_result(num_persons=0)
    assert fr.pose33 is None
    assert fr.track_id is None
    assert fr.meta is not None and fr.meta["num_persons"] == 0
    assert fr.meta["backend"] == "yolo"
    assert fr.meta["raw_layout"] == "pose33_like_coco17"
    assert fr.meta["feature_layout"] == "body_core_v1"
    assert fr.meta["capability"] == "body_only"
    assert fr.meta["calibration_status"] == "unvalidated"
    assert fr.meta["score_authorized"] is False
    assert fr.meta["display_scope"] == "limited"
    assert "eval_scope" not in fr.meta
    assert "internal_use_only" not in fr.meta


def test_yolo_result_empty_does_not_crash():
    res = FakeYoloResult.empty()
    row, valid, num_persons, sel = yolo_result_to_arrays(res, valid_conf_thr=0.5)
    assert np.all(row == 0.0)
    assert not np.any(valid)
    assert num_persons == 0
    assert sel is None
    # None 结果也不崩
    row2, valid2, n2, sel2 = yolo_result_to_arrays(None, valid_conf_thr=0.5)
    assert np.all(row2 == 0.0) and not np.any(valid2) and n2 == 0 and sel2 is None
    # 边界层同理
    fr = yolo_result_to_frame(res, valid_conf_thr=0.5)
    assert fr.pose33 is None


# --------------------------------------------------------------------------- #
# num_persons 暴露（Issue #9 闸门的基础；本期只暴露不拦截）
# --------------------------------------------------------------------------- #
def test_num_persons_exposed_for_multi_person():
    xy, conf = make_coco17(conf_value=0.9)
    res = FakeYoloResult.multi(
        [
            (xy, conf, (0.3, 0.5, 0.2, 0.4)),   # 较小框
            (xy, conf, (0.6, 0.5, 0.5, 0.9)),   # 较大框 → 应被选中
        ]
    )
    persons = extract_persons(res)
    assert len(persons) == 2
    # 最大框选择：index 1 面积更大
    assert select_main_person(persons) == 1

    _row, _valid, num_persons, sel = yolo_result_to_arrays(res, valid_conf_thr=0.5)
    assert num_persons == 2   # 暴露人数
    assert sel == 1
    # num_persons 被如实暴露；多人闸门的拒绝/降级在序列层 meta + 下游评分入口实现
    # （见下方 test_multi_person_gate_* 与 test_body_core_layout 的拒绝/降级用例）。


# --------------------------------------------------------------------------- #
# 多人场景闸门（Issue #9 / S2）
# --------------------------------------------------------------------------- #
def test_evaluate_multi_person_gate_single_person_ok():
    gate = evaluate_multi_person_gate([1, 1, 0, 1])
    assert gate["multi_person_detected"] is False
    assert gate["review_required"] is False
    assert gate["gate_status"] == GATE_STATUS_OK
    assert gate["max_persons"] == 1
    assert gate["multi_person_frames"] == 0
    assert gate["gate_note"] == ""


def test_evaluate_multi_person_gate_flags_and_counts():
    # 帧人数：1,3,2,1,0 → 两帧多人，max=3。
    gate = evaluate_multi_person_gate([1, 3, 2, 1, 0])
    assert gate["multi_person_detected"] is True
    assert gate["review_required"] is True
    assert gate["gate_status"] == GATE_STATUS_MULTI_PERSON
    assert gate["max_persons"] == 3
    assert gate["multi_person_frames"] == 2
    assert "review_required" in gate["gate_note"].lower()


def test_evaluate_multi_person_gate_empty_sequence():
    gate = evaluate_multi_person_gate([])
    assert gate["multi_person_detected"] is False
    assert gate["max_persons"] == 0
    assert gate["gate_status"] == GATE_STATUS_OK


def _multi_person_frame():
    """构造一帧两人（两个不同大小框）的 YOLO 结果。"""
    xy, conf = make_coco17(conf_value=0.9)
    return FakeYoloResult.multi(
        [
            (xy, conf, (0.3, 0.5, 0.2, 0.4)),
            (xy, conf, (0.6, 0.5, 0.5, 0.9)),
        ]
    )


def test_sequence_layer_multi_person_meta_flags_review_required():
    # 第 2 帧多人 → 整段必须标 multi_person_detected / review_required，不静默选最大框。
    xy, conf = make_coco17(conf_value=0.9)
    frames = [
        FakeYoloResult.single(xy, conf),
        _multi_person_frame(),
        FakeYoloResult.single(xy, conf),
    ]
    adapter = FakeYoloAdapter(frames=frames, valid_conf_thr=0.5)
    with patch_cv2_capture(n_frames=len(frames), fps=30.0):
        _lm, _vm, meta = extract_yolo_landmark_series("fake://multi.mp4", yolo_model=adapter)

    assert meta["max_persons"] == 2
    assert meta["multi_person_frames"] == 1
    assert meta["multi_person_detected"] is True
    assert meta["review_required"] is True
    assert meta["gate_status"] == GATE_STATUS_MULTI_PERSON
    assert meta["num_persons_per_frame"] == [1, 2, 1]
    assert "review_required" in meta["gate_note"].lower()
    assert meta["score_authorized"] is False
    assert meta["display_scope"] == "internal"


def test_sequence_layer_single_person_not_flagged():
    # 单人样本不受影响：不触发闸门。
    xy, conf = make_coco17(conf_value=0.9)
    frames = [FakeYoloResult.single(xy, conf) for _ in range(4)]
    adapter = FakeYoloAdapter(frames=frames, valid_conf_thr=0.5)
    with patch_cv2_capture(n_frames=len(frames), fps=30.0):
        _lm, _vm, meta = extract_yolo_landmark_series("fake://single.mp4", yolo_model=adapter)

    assert meta["max_persons"] == 1
    assert meta["multi_person_detected"] is False
    assert meta["review_required"] is False
    assert meta["gate_status"] == GATE_STATUS_OK


# --------------------------------------------------------------------------- #
# SingleTargetTracker 段边界重置语义
# --------------------------------------------------------------------------- #
def test_tracker_assigns_and_persists_within_segment():
    tr = SingleTargetTracker()
    assert tr.update(True) == 1   # 首次检出 → id 1
    assert tr.update(True) == 1   # 持续检出 → 沿用
    assert tr.update(False) is None  # 未检出帧 track_id 为 None
    assert tr.update(True) == 1   # 短暂丢失后仍沿用（未超过 max_missed）


def test_tracker_drops_after_max_missed_then_new_id():
    tr = SingleTargetTracker(max_missed=2)
    assert tr.update(True) == 1
    assert tr.update(False) is None  # missed=1
    assert tr.update(False) is None  # missed=2
    assert tr.update(False) is None  # missed=3 > 2 → 丢弃 active
    assert tr.update(True) == 2      # 重新检出 → 新 id 2


def test_tracker_reset_restarts_ids_at_segment_boundary():
    tr = SingleTargetTracker()
    assert tr.update(True) == 1
    assert tr.update(True) == 1
    # 段边界：reset 后 id 从 1 重新开始（不跨段复用）
    tr.reset()
    assert tr.active_id is None
    assert tr.next_id == 1
    assert tr.update(True) == 1


# --------------------------------------------------------------------------- #
# 序列层：只返回 numpy + meta 字段 + 段边界 track 重置（用 FakeYoloAdapter）
# --------------------------------------------------------------------------- #
def _build_frames(n_person_frames: int, n_empty_tail: int = 0):
    xy, conf = make_coco17(conf_value=0.9)
    frames = [FakeYoloResult.single(xy, conf) for _ in range(n_person_frames)]
    frames += [FakeYoloResult.empty() for _ in range(n_empty_tail)]
    return frames


def test_sequence_layer_returns_numpy_only_with_meta():
    frames = _build_frames(5)
    adapter = FakeYoloAdapter(frames=frames, valid_conf_thr=0.5)
    with patch_cv2_capture(n_frames=len(frames), fps=24.0):
        landmarks, valid_mask, meta = extract_yolo_landmark_series(
            "fake://video.mp4", yolo_model=adapter
        )

    assert isinstance(landmarks, np.ndarray)
    assert isinstance(valid_mask, np.ndarray)
    assert landmarks.shape == (5, 33, 4)
    assert valid_mask.shape == (5, 33)
    assert landmarks.dtype == np.float32
    assert valid_mask.dtype == bool

    # meta 必备字段
    for key in (
        "backend",
        "model_name",
        "running_mode",
        "confidence_kind",
        "validity_policy",
        "valid_conf_thr",
        "frame_count",
        "fps",
    ):
        assert key in meta, f"meta 缺少字段 {key}"
    assert meta["backend"] == "yolo"
    assert meta["raw_layout"] == "pose33_like_coco17"
    assert meta["feature_layout"] == "body_core_v1"
    assert meta["capability"] == "body_only"
    assert meta["confidence_kind"] == "yolo_conf"
    assert meta["validity_policy"] == "confidence_thr"
    assert meta["calibration_status"] == "unvalidated"
    assert meta["score_authorized"] is False
    assert meta["display_scope"] == "internal"
    assert "eval_scope" not in meta
    assert "internal_use_only" not in meta
    note = meta["calibration_note"].lower()
    assert "preview" in note or "must not enter outward-facing scoring" in note
    assert meta["fps"] == pytest.approx(24.0)
    # 段边界 track 重置说明写入 meta
    assert "reset" in meta["track_reset_note"].lower()
    # num_persons 暴露
    assert meta["max_persons"] == 1
    assert meta["multi_person_frames"] == 0
    assert len(meta["num_persons_per_frame"]) == 5


def test_sequence_layer_handles_no_person_frames():
    # 全部无人帧 → 零行 + 全 False mask，不崩。
    frames = _build_frames(0, n_empty_tail=4)
    adapter = FakeYoloAdapter(frames=frames, valid_conf_thr=0.5)
    with patch_cv2_capture(n_frames=len(frames), fps=30.0):
        landmarks, valid_mask, meta = extract_yolo_landmark_series(
            "fake://empty.mp4", yolo_model=adapter
        )
    assert landmarks.shape == (4, 33, 4)
    assert valid_mask.shape == (4, 33)
    assert np.all(landmarks == 0.0)
    assert not np.any(valid_mask)
    assert meta["max_persons"] == 0
    # 无人帧 track_id 全 None
    assert all(t is None for t in meta["track_ids"])


def test_sequence_layer_track_ids_reset_across_calls():
    # 两次独立调用（模拟两个段/两个视频）：track_id 都应从 1 起，不跨调用累加。
    frames_a = _build_frames(3)
    frames_b = _build_frames(3)
    adapter_a = FakeYoloAdapter(frames=frames_a, valid_conf_thr=0.5)
    adapter_b = FakeYoloAdapter(frames=frames_b, valid_conf_thr=0.5)

    with patch_cv2_capture(n_frames=3, fps=30.0):
        _lm_a, _vm_a, meta_a = extract_yolo_landmark_series("fake://seg_a.mp4", yolo_model=adapter_a)
    with patch_cv2_capture(n_frames=3, fps=30.0):
        _lm_b, _vm_b, meta_b = extract_yolo_landmark_series("fake://seg_b.mp4", yolo_model=adapter_b)

    # 段内首个有人帧 track_id 都应为 1（段边界重置语义）
    first_a = next(t for t in meta_a["track_ids"] if t is not None)
    first_b = next(t for t in meta_b["track_ids"] if t is not None)
    assert first_a == 1
    assert first_b == 1


def test_sequence_layer_reuse_adapter_resets_tracker():
    # 复用同一个 adapter 跑两段：第二段开始 extract 会 reset_tracker，track_id 重新从 1。
    frames = _build_frames(3)
    adapter = FakeYoloAdapter(frames=frames, valid_conf_thr=0.5)

    with patch_cv2_capture(n_frames=3, fps=30.0):
        _lm1, _vm1, meta1 = extract_yolo_landmark_series("fake://seg1.mp4", yolo_model=adapter)
    # 第二段：重置 cursor（FakeYoloAdapter.reset_tracker 由 extract 内部调用）
    with patch_cv2_capture(n_frames=3, fps=30.0):
        _lm2, _vm2, meta2 = extract_yolo_landmark_series("fake://seg2.mp4", yolo_model=adapter)

    first1 = next(t for t in meta1["track_ids"] if t is not None)
    first2 = next(t for t in meta2["track_ids"] if t is not None)
    assert first1 == 1
    assert first2 == 1  # 段边界重置，未跨段累加


# --------------------------------------------------------------------------- #
# adapter 构造：默认模型路径解析 + tracker 重置（不加载 ultralytics）
# --------------------------------------------------------------------------- #
def test_adapter_default_model_path_and_no_eager_load():
    adapter = YoloPoseAdapter()  # 不调用 _load，不触发 ultralytics import
    assert adapter.model_name == "yolo11n-pose.pt"
    assert adapter.valid_conf_thr == 0.6  # S3（#10）落库阈值；仍不授权对外评分
    assert adapter.confidence_kind == "yolo_conf"
    assert adapter.imgsz == 640
    assert adapter.device == "cpu"
    assert adapter.requested_half is False
    assert adapter.half is False
    assert adapter.warmup is False
    assert adapter._warmup_done is False
    # 构造后仍未导入 ultralytics
    assert "ultralytics" not in sys.modules
    # reset_tracker 可用
    adapter.reset_tracker()
    assert adapter.last_num_persons == 0


def test_adapter_half_is_opt_in_and_cpu_falls_back():
    adapter = YoloPoseAdapter(device="cpu", half=True, warmup=True, warmup_shape=(480, 640))

    assert adapter.requested_half is True
    assert adapter.half is False
    assert adapter.warmup is True
    assert adapter.warmup_shape == (480, 640, 3)
    assert "ultralytics" not in sys.modules


def test_adapter_cuda_half_predict_and_warmup_only_once(monkeypatch):
    xy, conf = make_coco17(conf_value=0.9)
    result = FakeYoloResult.single(xy, conf)

    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def predict(self, frame, **kwargs):
            self.calls.append(
                {
                    "shape": tuple(frame.shape),
                    "imgsz": kwargs["imgsz"],
                    "device": kwargs["device"],
                    "half": kwargs["half"],
                    "verbose": kwargs["verbose"],
                }
            )
            return [result]

    fake_model = FakeModel()
    adapter = YoloPoseAdapter(
        model_path="fake.pt",
        imgsz=512,
        device="cuda",
        half=True,
        warmup=True,
        warmup_shape=(32, 48),
    )
    monkeypatch.setattr(adapter, "_load", lambda: fake_model)

    frame = np.zeros((80, 96, 3), dtype=np.uint8)
    adapter.infer_frame(frame)
    adapter.infer_frame(frame)

    assert len(fake_model.calls) == 3
    assert fake_model.calls[0] == {
        "shape": (32, 48, 3),
        "imgsz": 512,
        "device": "cuda",
        "half": True,
        "verbose": False,
    }
    assert fake_model.calls[1]["shape"] == (80, 96, 3)
    assert fake_model.calls[2]["shape"] == (80, 96, 3)
    assert all(call["half"] is True for call in fake_model.calls)
    assert adapter._warmup_done is True


def test_extract_yolo_landmark_series_passes_adapter_options(monkeypatch):
    created: list[YoloPoseAdapter] = []
    original_init = YoloPoseAdapter.__init__

    def tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(YoloPoseAdapter, "__init__", tracking_init)

    xy, conf = make_coco17(conf_value=0.9)
    result = FakeYoloResult.single(xy, conf)

    class FakeModel:
        def predict(self, frame, **kwargs):
            return [result]

    monkeypatch.setattr(YoloPoseAdapter, "_load", lambda self: FakeModel())

    with patch_cv2_capture(n_frames=1, fps=30.0):
        _landmarks, _valid_mask, meta = extract_yolo_landmark_series(
            "fake://options.mp4",
            yolo_model="fake.pt",
            imgsz=512,
            device="cuda",
            half=True,
            warmup=True,
            warmup_shape=(32, 48),
        )

    assert len(created) == 1
    adapter = created[0]
    assert adapter.imgsz == 512
    assert adapter.device == "cuda"
    assert adapter.requested_half is True
    assert adapter.half is True
    assert adapter.warmup is True
    assert adapter._warmup_done is True
    assert meta["imgsz"] == 512
    assert meta["device"] == "cuda"
    assert meta["requested_half"] is True
    assert meta["half"] is True
    assert meta["warmup"] is True
    assert meta["warmup_shape"] == (32, 48, 3)
