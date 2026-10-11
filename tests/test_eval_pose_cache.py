# -*- coding: utf-8 -*-
"""评分流程接入 ``pose_cache``（issue #11，设计文档 §6.3）：统计底层 ``infer()`` 次数。

双模板比对的"特征 + 规则/误差原始序列"、考试池的"模板比对 + 踢腿几何"在顺序跟踪下各只推理一次；
分段提取（``workers > 1``）的特征不被规则复用，规则仍独立顺序提取。分数与不走缓存的旧行为一致。
"""
from __future__ import annotations

import sys
from pathlib import Path
from threading import Event

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.evaluation import compare_dual_streams, compare_dual_templates  # noqa: E402
from core.evaluation.scoring import exam_pool  # noqa: E402
from core.pose import extract as extract_mod  # noqa: E402
from core.pose import extract_pose_raw_series, pose_cache  # noqa: E402
from core.services.analyze import match_video  # noqa: E402
from tests import golden_harness as H  # noqa: E402

FIX_DIR = Path(__file__).resolve().parent / "fixtures" / "pose33_v3"
FRONT_TPL = FIX_DIR / "front_template.npz"
SIDE_TPL = FIX_DIR / "side_template.npz"
FRONT_SRC = "golden://cache_front_src.mp4"
SIDE_SRC = "golden://cache_side_src.mp4"
STUDENT = "golden://cache_student.mp4"
FPS = 30.0


def _raw(name: str) -> np.ndarray:
    return np.asarray(np.load(FIX_DIR / name, allow_pickle=True)["landmarks"], dtype=np.float32)


@pytest.fixture()
def infer_calls():
    """golden 回放 + 统计假引擎的 ``infer()`` 调用（底层推理次数，不是提取函数的调用次数）。"""
    H.clear_registry()
    H.register_video(FRONT_SRC, _raw("front_src_raw.npz"), fps=FPS)
    H.register_video(SIDE_SRC, _raw("side_src_raw.npz"), fps=FPS)
    H.register_video(STUDENT, _raw("student_raw.npz"), fps=FPS)
    calls: list[int] = [0]
    with H.replay_context(), pytest.MonkeyPatch.context() as mp:
        fake_create = extract_mod.create_engine

        def counted(spec, **kw):
            pipe = fake_create(spec, **kw)
            infer = pipe.infer

            def counted_infer(frame, **kwargs):
                calls[0] += 1
                return infer(frame, **kwargs)

            pipe.infer = counted_infer
            return pipe

        mp.setattr(extract_mod, "create_engine", counted)
        yield calls
    H.clear_registry()


def _dual_fields(r) -> tuple:
    return (r.front_score, r.side_score, r.combined_score, r.combined_percent, r.front_segment, r.side_segment,
            r.front_rule_score, r.side_rule_score, r.front_rule_violations, r.side_rule_violations,
            r.front_joint_errors, r.side_joint_errors)


def test_dual_templates_infers_once_with_rules_and_error_analysis(infer_calls):
    result = compare_dual_templates(FRONT_TPL, SIDE_TPL, STUDENT, enable_rules=True, enable_error_analysis=True)
    assert infer_calls[0] == 180  # 180 帧 fixture：特征与规则/误差原始序列共用一次推理
    assert result.front_rule_score is not None and result.front_joint_errors is not None


def test_dual_templates_scores_match_per_consumer_extraction(infer_calls):
    cached = compare_dual_templates(FRONT_TPL, SIDE_TPL, STUDENT, enable_rules=True, enable_error_analysis=True)
    infer_calls[0] = 0
    fresh = compare_dual_templates.__wrapped__(FRONT_TPL, SIDE_TPL, STUDENT,  # 去掉作用域：旧行为参照
                                               enable_rules=True, enable_error_analysis=True)
    assert infer_calls[0] == 360  # 不走缓存时特征、原始序列各推理一次
    assert _dual_fields(cached) == _dual_fields(fresh)


def test_dual_streams_infers_each_video_once(infer_calls):
    result = compare_dual_streams(FRONT_TPL, SIDE_TPL, FRONT_SRC, SIDE_SRC, enable_rules=True,
                                  enable_error_analysis=True)
    assert infer_calls[0] == 90 + 90
    assert result.front_rule_score is not None and result.side_joint_errors is not None


def test_segmented_features_do_not_feed_sequential_rules(infer_calls, tmp_path):
    """``workers > 1`` 的特征是分段跟踪，规则要顺序跟踪：缓存不命中，规则独立提取。"""
    front = _template(tmp_path / "front.npz", FRONT_TPL, running_mode="image")  # VIDEO 模板会强制单线程
    side = _template(tmp_path / "side.npz", SIDE_TPL, running_mode="image")
    compare_dual_templates(front, side, STUDENT, workers=3)
    segmented_only = infer_calls[0]
    assert segmented_only > 180  # 分段含每段 15 帧预热
    infer_calls[0] = 0
    compare_dual_templates(front, side, STUDENT, workers=3, enable_rules=True)
    assert infer_calls[0] == segmented_only + 180  # 规则另做一次完整顺序推理


def test_scope_is_reentrant_and_ends_with_outer_scope(infer_calls):
    with pose_cache():
        compare_dual_templates(FRONT_TPL, SIDE_TPL, STUDENT, enable_rules=True)
        extract_pose_raw_series(STUDENT, pose_variant="full")  # 外层作用域仍持有内层的结果
    assert infer_calls[0] == 180
    extract_pose_raw_series(STUDENT, pose_variant="full")  # 作用域外不缓存
    assert infer_calls[0] == 360


def _template(out: Path, src: Path, **meta_overrides) -> Path:
    data = np.load(src, allow_pickle=True)
    meta = {**data["meta"].item(), **meta_overrides}
    np.savez_compressed(out, features=data["features"], meta=np.array(meta, dtype=object))
    return out


def _kick_pool(tmp_path: Path) -> list[Path]:
    """把 fixture 模板改写成考试池契约（heavy + pose33_v3），名字取踢腿动作与非踢腿动作各一。"""
    return [_template(tmp_path / f"{name}.npz", src, pose_variant="heavy")
            for name, src in (("高鞭腿", FRONT_TPL), ("直拳", SIDE_TPL))]


def test_exam_pool_with_kick_infers_each_video_once(infer_calls, tmp_path):
    pool = _kick_pool(tmp_path)
    raw_videos: list[str] = []

    def raw_series(video, **kw):
        raw_videos.append(str(video))
        return extract_pose_raw_series(video, **kw)

    exam_pool.score_exam_pools(pool, pool, Path(FRONT_SRC), Path(SIDE_SRC), compare=match_video,
                               raw_series=raw_series, stop_evt=Event())
    assert len(raw_videos) == 2  # 两路都走了踢腿几何的原始序列
    assert infer_calls[0] == 90 + 90  # 模板比对与踢腿几何共用每路视频的一次推理


def test_exam_pool_scores_match_per_consumer_extraction(infer_calls, tmp_path, monkeypatch):
    pool = _kick_pool(tmp_path)

    def run():
        return exam_pool.score_exam_pools(pool, [], Path(FRONT_SRC), Path(SIDE_SRC), compare=match_video,
                                          raw_series=extract_pose_raw_series, stop_evt=Event())

    cached = run()
    assert infer_calls[0] == 90
    infer_calls[0] = 0
    monkeypatch.setattr(exam_pool, "score_pool", exam_pool.score_pool.__wrapped__)  # 去掉作用域：旧行为参照
    fresh = run()
    assert infer_calls[0] == 180  # 不走缓存时比对与踢腿几何各推理一次
    assert cached == fresh

