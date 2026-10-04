# -*- coding: utf-8 -*-
"""
pose33_v3 golden 回归基线（YOLO 迁移 Issue #3 / S1 安全网，先行）。

为什么需要它
------------
S1 后续 Issue（#4 layout shape 参数化 + 周期裁切修复、#5 valid_mask 契约迁移）会改动
这些热路径：
  - core.action_compare._extract_pose_features() 缺帧补零形状
  - core.pose_features.mirror_pose_features()
  - core.action_compare._select_representative_cycle()
  - 双模板关节误差统计
  - analysis.tech_eval._valid / core.rule_scoring._valid_frame 的有效性判据

`py_compile` 对“行为不变”零保证。本测试把当前 pose33_v3 默认路径的行为冻结成 golden，
作为这些重构“行为不变”的唯一硬门槛。任意改动后本测试必须保持全绿。

覆盖的三类输出（对应 Issue #3 验收标准）
----------------------------------------
1. 单模板 compare_video_to_template 分数；
2. 双模板 compare_video_to_dual_templates 的 combined_percent + 各视角分
   （含规则扣分明细、关节误差统计）；
3. tech_eval evaluate_video_full 各指标 status + 关键 detail。

断言精度
--------
- 数值用极小容差 np.allclose / pytest.approx（abs=1e-4）；
- 状态 / 分类 / 整数（combined_percent、rule_score、status、segment、primary_cause 等）
  用精确相等。

确定性
------
不依赖 MediaPipe 实际推理：通过 tests/golden_harness.py 回放已提交的
(T,33,4) landmark fixture，所有上层入口都走真实代码路径但输入确定，
因此 golden 只反映本仓库代码行为，不受模型版本影响。

如何在“有意变更行为”时重新生成 golden
---------------------------------------
1. 确认行为变更是预期且正确的（例如重新标定阈值、修复算法 bug）；
2. 运行：
       .\\.venv\\Scripts\\python.exe tests\\fixtures\\regen_pose33_v3_golden.py
   该脚本会用已提交的 *_raw.npz fixture 重新生成 front_template.npz / side_template.npz
   / golden.json；
3. review golden.json 的 diff，确认每处变化都符合预期后再提交。
   （若是 #4 / #5 这类“行为不变”重构，则不应产生任何 diff——出现 diff 即说明漂移。）

如需重建 raw fixture 本身（仅当源样本或裁剪范围有意调整时）：
       .\\.venv\\Scripts\\python.exe tests\\fixtures\\_build_raw_fixtures.py

验收命令
--------
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_pose33_v3_golden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.action_compare import (  # noqa: E402
    compare_video_to_dual_templates,
    compare_video_to_template,
)
from analysis.tech_eval import evaluate_video_full  # noqa: E402
from tests import golden_harness as H  # noqa: E402

FIX_DIR = Path(__file__).resolve().parent / "fixtures" / "pose33_v3"

FRONT_TPL = FIX_DIR / "front_template.npz"
SIDE_TPL = FIX_DIR / "side_template.npz"
GOLDEN_JSON = FIX_DIR / "golden.json"

# 与 regen 脚本一致的虚拟视频 key 与 fps。
FRONT_SRC = "golden://front_src.mp4"
SIDE_SRC = "golden://side_src.mp4"
STUDENT = "golden://student.mp4"
FPS = 30.0

# 数值断言绝对容差（极小）。状态/整数用精确相等。
ABS_TOL = 1e-4


def _load_raw(name: str) -> np.ndarray:
    d = np.load(FIX_DIR / name, allow_pickle=True)
    return np.asarray(d["landmarks"], dtype=np.float32)


@pytest.fixture(scope="module")
def golden() -> dict:
    assert GOLDEN_JSON.exists(), (
        "缺少 golden.json，请先运行 tests/fixtures/regen_pose33_v3_golden.py 生成。"
    )
    with open(GOLDEN_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def registered_videos():
    """注册三段确定性视频；模板沿用已提交的 *.npz。"""
    assert FRONT_TPL.exists() and SIDE_TPL.exists(), (
        "缺少模板 fixture，请先运行 tests/fixtures/regen_pose33_v3_golden.py 生成。"
    )
    H.clear_registry()
    H.register_video(FRONT_SRC, _load_raw("front_src_raw.npz"), fps=FPS)
    H.register_video(SIDE_SRC, _load_raw("side_src_raw.npz"), fps=FPS)
    H.register_video(STUDENT, _load_raw("student_raw.npz"), fps=FPS)
    yield
    H.clear_registry()


# --------------------------------------------------------------------------- #
# 1) 单模板分数
# --------------------------------------------------------------------------- #
def test_single_template_score(golden, registered_videos):
    g = golden["single_template"]
    with H.replay_context():
        res = compare_video_to_template(FRONT_TPL, STUDENT, pose_variant="full")

    # 数值：极小容差
    assert res.score == pytest.approx(g["score"], abs=ABS_TOL)
    assert res.avg_cost == pytest.approx(g["avg_cost"], abs=ABS_TOL)
    assert res.cost == pytest.approx(g["cost"], abs=ABS_TOL)
    assert res.fps == pytest.approx(g["fps"], abs=ABS_TOL)
    # 匹配区间：精确相等（DTW 路径不应漂移）
    assert int(res.start_frame) == int(g["start_frame"])
    assert int(res.end_frame) == int(g["end_frame"])


# --------------------------------------------------------------------------- #
# 2) 双模板：combined_percent + 各视角分 + 规则扣分 + 关节误差
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def dual_result(registered_videos):
    with H.replay_context():
        return compare_video_to_dual_templates(
            FRONT_TPL,
            SIDE_TPL,
            STUDENT,
            pose_variant="full",
            enable_rules=True,
            enable_error_analysis=True,
        )


def test_dual_scores(golden, dual_result):
    g = golden["dual_template"]
    # combined_percent：整数，精确相等（这是对外评分的核心数字）
    assert int(dual_result.combined_percent) == int(g["combined_percent"])
    # 各视角分与组合分：极小容差
    assert dual_result.combined_score == pytest.approx(g["combined_score"], abs=ABS_TOL)
    assert dual_result.front_score == pytest.approx(g["front_score"], abs=ABS_TOL)
    assert dual_result.side_score == pytest.approx(g["side_score"], abs=ABS_TOL)


def test_dual_segments(golden, dual_result):
    g = golden["dual_template"]
    # 视角切分区间：精确相等
    fseg = None if dual_result.front_segment is None else list(dual_result.front_segment)
    sseg = None if dual_result.side_segment is None else list(dual_result.side_segment)
    assert fseg == g["front_segment"]
    assert sseg == g["side_segment"]


def test_dual_rule_scores(golden, dual_result):
    g = golden["dual_template"]
    # 规则总分/扣分：整数精确相等
    assert dual_result.front_rule_score == g["front_rule_score"]
    assert dual_result.side_rule_score == g["side_rule_score"]
    assert dual_result.front_rule_deduction == g["front_rule_deduction"]
    assert dual_result.side_rule_deduction == g["side_rule_deduction"]


@pytest.mark.parametrize("view", ["front", "side"])
def test_dual_rule_violations(golden, dual_result, view):
    g = golden["dual_template"][f"{view}_rule_violations"]
    got = dual_result.front_rule_violations if view == "front" else dual_result.side_rule_violations
    assert got is not None and g is not None
    assert len(got) == len(g)
    # 违规明细按 rule_id 对齐，逐条核对
    got_by_id = {v.rule_id: v for v in got}
    for exp in g:
        rid = exp["rule_id"]
        assert rid in got_by_id, f"缺少规则 {rid}"
        v = got_by_id[rid]
        # 结构化字段精确相等（state 由字符串拼出，#11 才结构化；此处冻结当前行为）
        assert v.name == exp["name"]
        assert int(v.penalty) == int(exp["penalty"])
        assert int(v.valid_frames) == int(exp["valid_frames"])
        assert int(v.total_frames) == int(exp["total_frames"])
        assert v.detail == exp["detail"]
        # 违规占比：极小容差
        assert float(v.violation_ratio) == pytest.approx(exp["violation_ratio"], abs=ABS_TOL)


@pytest.mark.parametrize("view", ["front", "side"])
def test_dual_joint_errors(golden, dual_result, view):
    g = golden["dual_template"][f"{view}_joint_errors"]
    got = dual_result.front_joint_errors if view == "front" else dual_result.side_joint_errors
    assert got is not None and g is not None
    assert len(got) == len(g)
    # 关节顺序敏感（mirror 会交换 L/R 名称，必须冻结），逐位核对
    for j, exp in zip(got, g):
        assert j.joint == exp["joint"]
        assert int(j.valid_frames) == int(exp["valid_frames"])
        for attr in ("mean_dist", "p90_dist", "max_dist"):
            jv = getattr(j, attr)
            ev = exp[attr]
            if ev is None:
                assert jv is None, f"{j.joint}.{attr} 期望 None，实际 {jv}"
            else:
                assert jv is not None and float(jv) == pytest.approx(ev, abs=ABS_TOL)


# --------------------------------------------------------------------------- #
# 3) tech_eval 各指标 status + 关键 detail
# --------------------------------------------------------------------------- #
_TECH_DETAIL_KEYS = (
    "valid_frames",
    "total_frames",
    "forward_frames",
    "backward_frames",
    "center_frames",
    "unknown_frames",
    "primary_cause",
    "front_leg",
    "events_total",
    "events_used",
    "events_ok",
    "events_bad",
)

_INDICATORS = ("cog_side", "cog_front", "cog_final", "cog_com", "retract_speed", "force_sequence", "wrist_angle")


@pytest.fixture(scope="module")
def tech_eval_full(registered_videos):
    with H.replay_context():
        return evaluate_video_full(STUDENT, pose_variant="full", stance="left", view_hint="auto")


def test_tech_eval_view_and_segments(golden, tech_eval_full):
    g = golden["tech_eval"]
    assert tech_eval_full.get("view_mode") == g["view_mode"]
    assert tech_eval_full.get("front_segment") == g["front_segment"]
    assert tech_eval_full.get("side_segment") == g["side_segment"]


@pytest.mark.parametrize("indicator", _INDICATORS)
def test_tech_eval_indicator(golden, tech_eval_full, indicator):
    g = golden["tech_eval"][indicator]
    got = tech_eval_full.get(indicator)
    if g is None:
        assert got is None
        return
    assert got is not None, f"指标 {indicator} 缺失"
    # status / reason：精确相等（分类断言）
    assert got.get("status") == g["status"], f"{indicator} status 漂移"
    assert got.get("reason") == g["reason"], f"{indicator} reason 漂移"

    # 关键 detail 字段：白名单提取后精确相等
    detail = got.get("detail") or {}
    exp_detail = g["detail_keys"]
    for k in _TECH_DETAIL_KEYS:
        if k in exp_detail:
            assert k in detail, f"{indicator}.detail 缺少 {k}"
            ev = exp_detail[k]
            dv = detail[k]
            if isinstance(ev, float):
                assert float(dv) == pytest.approx(ev, abs=ABS_TOL)
            else:
                assert dv == ev, f"{indicator}.detail[{k}] 漂移：{dv} != {ev}"
