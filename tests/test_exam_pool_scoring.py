# -*- coding: utf-8 -*-
"""考试池计分方案（``evaluation.scoring.exam_pool``）：正侧权重合并、分数压缩、踢腿几何加权与模板池校验。

前 4 个用例原在 ``test_recording_postprocess.py``，P3 把评分策略移出 apps 层后随实现迁到这里（断言不变）。
"""
from __future__ import annotations

from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from core.evaluation import TemplateError
from core.evaluation.scoring import exam_pool


def _d(name: str, score: float, start: int = 0, end: int = 10) -> dict:
    return {"name": name, "view": "x", "score": score, "start": start, "end": end}


def test_merge_view_weighted_actions_paired_uses_registered_weights() -> None:
    # 两手攀足：侧 0.75 + 正 0.25（实测正面看不清前俯深度）。
    out = exam_pool.merge_view_weighted_actions([_d("两手攀足_正面", 0.708), _d("两手攀足_侧面", 0.855)])
    assert len(out) == 1
    assert out[0]["name"] == "两手攀足"
    assert out[0]["score"] == pytest.approx(0.75 * 0.855 + 0.25 * 0.708)
    # 攒拳怒目：50/50 去重双计数。
    out = exam_pool.merge_view_weighted_actions([_d("攒拳怒目_正面", 0.812), _d("攒拳怒目_侧面", 0.807)])
    assert len(out) == 1 and out[0]["score"] == pytest.approx(0.8095)


def test_merge_view_weighted_actions_is_noop_for_sanda() -> None:
    # 散打模板 stem 无 _正面/_侧面 后缀、不在权重表内 → 原样透传、保序（零回归）。
    sanda = [_d("直拳", 0.8), _d("高鞭腿", 0.7), _d("front", 0.86), _d("side", 0.86)]
    assert exam_pool.merge_view_weighted_actions(sanda) == sanda


def test_merge_view_weighted_actions_keeps_unpaired_view() -> None:
    # 缺侧面则不合并，正面原样保留（不误合、不丢分）。
    out = exam_pool.merge_view_weighted_actions([_d("两手攀足_正面", 0.708)])
    assert len(out) == 1 and out[0]["name"] == "两手攀足_正面"


def test_score_squeeze_backhand_keeps_range_and_reproducible() -> None:
    # 背后七颠：DTW 不可靠 → 压到 [0.75,0.90]（人人过、弱区分、可复现）。原测试只验公式本身，
    # 现直接验 action_score（分数压缩的唯一实现）。
    lo, hi = exam_pool.SCORE_SQUEEZE["背后七颠"]

    def sq(d: float) -> float:
        return exam_pool.action_score("背后七颠", d, None)

    # 任意 DTW 输入都落在窄区间内（人人及格偏上）。
    for dtw in (0.0, 0.3, 0.7, 0.936, 0.942, 1.0):
        assert lo - 1e-9 <= sq(dtw) <= hi + 1e-9
    # 单调保留微弱区分度：认真 > 敷衍 > 没做。
    assert sq(0.942) > sq(0.70) > sq(0.30)
    # 可复现：同输入同输出（无随机）。
    assert sq(0.936) == sq(0.936)


def test_kick_geometry_weighted_with_dtw_and_squeeze_ignores_geometry() -> None:
    assert exam_pool.action_score("高鞭腿", 0.8, 0.4) == pytest.approx(0.5 * 0.4 + 0.5 * 0.8)
    assert exam_pool.action_score("高鞭腿", 0.8, None) == 0.8
    assert exam_pool.action_score("直拳", 0.8, 0.1) == pytest.approx(0.5 * 0.1 + 0.5 * 0.8)  # 是否算几何由池决定
    assert exam_pool.action_score("背后七颠", 2.0, 0.0) == 0.90


def _write(path: Path, *, variant="heavy", layout="pose33_v3", shape=(5, 22, 2)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"pose_variant": variant, "feature_layout": layout}
    np.savez_compressed(path, features=np.zeros(shape, np.float32), meta=np.array(meta, dtype=object))
    return path


def test_usable_pools_filters_bad_templates_and_dedups(tmp_path) -> None:
    good = _write(tmp_path / "直拳.npz")
    bad = _write(tmp_path / "bad.npz", variant="full")
    front, side = exam_pool.usable_pools([good, bad, str(good), " ", None], [bad])
    assert front == [good] and side == []
    with pytest.raises(TemplateError) as exc:
        exam_pool.usable_pools([bad], None)
    assert exc.value.code == "template_missing"


def test_score_exam_pools_reuses_one_extraction_per_view_and_merges(tmp_path) -> None:
    tpls = [_write(tmp_path / f"{n}.npz") for n in ("直拳", "两手攀足_正面")]
    side_tpls = [_write(tmp_path / "side" / "两手攀足_侧面.npz")]
    calls = []

    def compare(templates, video, **kwargs):
        calls.append((tuple(templates), video, kwargs["pose_variant"]))
        return [SimpleNamespace(score=0.5 + 0.1 * i, start_frame=i, end_frame=i + 3) for i in range(len(templates))]

    res = exam_pool.score_exam_pools(tpls, side_tpls, Path("f.mp4"), Path("s.mp4"), compare=compare,
                                     raw_series=lambda *_a, **_k: pytest.fail("无踢腿动作不该提原始序列"),
                                     stop_evt=Event())
    assert [c[1] for c in calls] == [Path("f.mp4"), Path("s.mp4")] and {c[2] for c in calls} == {"heavy"}
    assert res.front_score == pytest.approx(0.55) and res.side_score == pytest.approx(0.5)
    names = [a["name"] for a in res.action_scores]
    assert names == ["直拳", "两手攀足"]
    assert res.combined_score == pytest.approx((0.5 + 0.25 * 0.6 + 0.75 * 0.5) / 2)
    assert res.front_segment == (0, 3) and res.side_segment == (0, 3)

