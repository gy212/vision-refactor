# -*- coding: utf-8 -*-
"""
重生成 pose33_v3 golden（YOLO 迁移 Issue #3 / S1 安全网）。

何时运行
--------
仅当你“有意”改变了 MediaPipe 默认路径（pose33_v3）的行为，并确认新行为正确时，
才重跑本脚本刷新 golden。日常重构（#4 / #5）的目标是 **不触发 golden 变化**——
若本脚本输出与已提交 golden 不一致，说明行为漂移，需先确认是否符合预期。

输入（已提交、确定性）
----------------------
  tests/fixtures/pose33_v3/front_src_raw.npz   (T,33,4) 正面模板源
  tests/fixtures/pose33_v3/side_src_raw.npz    (T,33,4) 侧面模板源
  tests/fixtures/pose33_v3/student_raw.npz     (T,33,4) 学员长视频（正面段+侧面段）

产物
----
  tests/fixtures/pose33_v3/front_template.npz  pose33_v3 正面模板
  tests/fixtures/pose33_v3/side_template.npz   pose33_v3 侧面模板
  tests/fixtures/pose33_v3/golden.json         三类回归基线：
      1) 单模板 compare_video_to_template 分数
      2) 双模板 compare_video_to_dual_templates 的 combined_percent + 各视角分
         （含规则扣分、关节误差统计）
      3) tech_eval evaluate_video_full 各指标 status + 关键 detail

运行：
    .\\.venv\\Scripts\\python.exe tests\\fixtures\\regen_pose33_v3_golden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# 允许从仓库根直接运行本脚本。
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.action_compare import (  # noqa: E402
    compare_video_to_dual_templates,
    compare_video_to_template,
    create_template_from_video,
)
from analysis.tech_eval import evaluate_video_full  # noqa: E402
from tests import golden_harness as H  # noqa: E402

FIX_DIR = Path(__file__).resolve().parent / "pose33_v3"

FRONT_SRC = "golden://front_src.mp4"
SIDE_SRC = "golden://side_src.mp4"
STUDENT = "golden://student.mp4"

FPS = 30.0


def _load_raw(name: str) -> np.ndarray:
    d = np.load(FIX_DIR / name, allow_pickle=True)
    return np.asarray(d["landmarks"], dtype=np.float32)


def build_templates() -> tuple[Path, Path]:
    """从 raw fixture 生成 pose33_v3 正/侧模板（确定性回放）。"""
    front_raw = _load_raw("front_src_raw.npz")
    side_raw = _load_raw("side_src_raw.npz")

    front_tpl = FIX_DIR / "front_template.npz"
    side_tpl = FIX_DIR / "side_template.npz"

    H.clear_registry()
    H.register_video(FRONT_SRC, front_raw, fps=FPS)
    H.register_video(SIDE_SRC, side_raw, fps=FPS)

    with H.replay_context():
        create_template_from_video(FRONT_SRC, pose_variant="full", out_path=front_tpl)
        create_template_from_video(SIDE_SRC, pose_variant="full", out_path=side_tpl)

    return front_tpl, side_tpl


def compute_golden() -> dict:
    front_tpl, side_tpl = build_templates()

    front_raw = _load_raw("front_src_raw.npz")
    side_raw = _load_raw("side_src_raw.npz")
    student_raw = _load_raw("student_raw.npz")

    H.clear_registry()
    H.register_video(FRONT_SRC, front_raw, fps=FPS)
    H.register_video(SIDE_SRC, side_raw, fps=FPS)
    H.register_video(STUDENT, student_raw, fps=FPS)

    golden: dict = {}

    with H.replay_context():
        # 1) 单模板分数：用正面模板比对学员视频。
        single = compare_video_to_template(front_tpl, STUDENT, pose_variant="full")
        golden["single_template"] = {
            "score": float(single.score),
            "avg_cost": float(single.avg_cost),
            "cost": float(single.cost),
            "start_frame": int(single.start_frame),
            "end_frame": int(single.end_frame),
            "fps": float(single.fps),
        }

        # 2) 双模板：combined_percent + 各视角分 + 规则扣分 + 关节误差。
        dual = compare_video_to_dual_templates(
            front_tpl,
            side_tpl,
            STUDENT,
            pose_variant="full",
            enable_rules=True,
            enable_error_analysis=True,
        )
        golden["dual_template"] = {
            "combined_percent": int(dual.combined_percent),
            "combined_score": float(dual.combined_score),
            "front_score": float(dual.front_score),
            "side_score": float(dual.side_score),
            "front_segment": None if dual.front_segment is None else list(dual.front_segment),
            "side_segment": None if dual.side_segment is None else list(dual.side_segment),
            "front_rule_score": dual.front_rule_score,
            "side_rule_score": dual.side_rule_score,
            "front_rule_deduction": dual.front_rule_deduction,
            "side_rule_deduction": dual.side_rule_deduction,
            "front_rule_violations": _violations_to_list(dual.front_rule_violations),
            "side_rule_violations": _violations_to_list(dual.side_rule_violations),
            "front_joint_errors": _joint_errors_to_list(dual.front_joint_errors),
            "side_joint_errors": _joint_errors_to_list(dual.side_joint_errors),
        }

        # 3) tech_eval 各指标 status + 关键 detail。
        full = evaluate_video_full(STUDENT, pose_variant="full", stance="left", view_hint="auto")
        golden["tech_eval"] = _tech_eval_summary(full)

    return golden


def _violations_to_list(violations) -> list | None:
    if violations is None:
        return None
    out = []
    for v in violations:
        out.append(
            {
                "rule_id": v.rule_id,
                "name": v.name,
                "penalty": int(v.penalty),
                "violation_ratio": float(v.violation_ratio),
                "valid_frames": int(v.valid_frames),
                "total_frames": int(v.total_frames),
                "detail": v.detail,
            }
        )
    return out


def _joint_errors_to_list(joint_errors) -> list | None:
    if joint_errors is None:
        return None
    out = []
    for j in joint_errors:
        out.append(
            {
                "joint": j.joint,
                "mean_dist": None if j.mean_dist is None else float(j.mean_dist),
                "p90_dist": None if j.p90_dist is None else float(j.p90_dist),
                "max_dist": None if j.max_dist is None else float(j.max_dist),
                "valid_frames": int(j.valid_frames),
            }
        )
    return out


# tech_eval detail 中需要锁定的“关键”数值字段（按指标白名单提取，避免 golden 过脆）。
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


def _indicator_summary(ind: dict | None) -> dict | None:
    if ind is None:
        return None
    detail = ind.get("detail") or {}
    kept = {k: detail[k] for k in _TECH_DETAIL_KEYS if k in detail}
    return {
        "status": ind.get("status"),
        "reason": ind.get("reason"),
        "detail_keys": kept,
    }


def _tech_eval_summary(full: dict) -> dict:
    summary = {
        "view_mode": full.get("view_mode"),
        "front_segment": full.get("front_segment"),
        "side_segment": full.get("side_segment"),
    }
    for key in ("cog_side", "cog_front", "cog_final", "cog_com", "retract_speed", "force_sequence", "wrist_angle"):
        summary[key] = _indicator_summary(full.get(key))
    return summary


def main() -> None:
    golden = compute_golden()
    out = FIX_DIR / "golden.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2, sort_keys=True)
    print("golden 已写入：", out)
    print(json.dumps(golden, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
