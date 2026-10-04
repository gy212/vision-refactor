# -*- coding: utf-8 -*-
"""
S3 标定落配置回归（YOLO 迁移 Issue #10）。

验收目标（对应 Issue #10）
-------------------------
- ``body_core_v1`` baseline 已落配置（``FeatureLayoutSpec.default_baseline``），
  不再写死占位 2.0。
- YOLO ``valid_conf_thr`` 已标定入库（``DEFAULT_YOLO_VALID_CONF_THR``），占位 0.5 移除。
- body_core 闭环 baseline 取自 layout（与占位脱钩），但 calibration_status 仍不授权评分。
- 标定报告 ``docs/yolo_body_core_calibration.md`` 存在且头部含预注册数字与结论四选一。

这些断言把「标定值入库」固化为回归——若有人误改回占位值或改坏报告头部，测试会失败。

验收命令
--------
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_s3_calibration.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.body_core_compare import (  # noqa: E402
    BODY_CORE_V1_CALIBRATED_BASELINE,
    CALIBRATION_STATUS_UNVALIDATED,
)
from core.feature_layout import BODY_CORE_V1, POSE33_V3  # noqa: E402
from core.yolo_adapter import (  # noqa: E402
    DEFAULT_YOLO_VALID_CONF_THR,
    YOLO_CALIBRATION_STATUS,
)
from analysis import bench_annotate_fps  # noqa: E402

CALIB_BASELINE = 1.2826
CALIB_YOLO_CONF_THR = 0.6
REPORT = _REPO_ROOT / "docs" / "yolo_body_core_calibration.md"


# --------------------------------------------------------------------------- #
# 1) baseline 已落配置（不再写死 2.0）
# --------------------------------------------------------------------------- #
def test_body_core_baseline_calibrated_in_layout():
    # body_core_v1 的 baseline 已标定入 layout（替换占位 2.0 / None）。
    assert BODY_CORE_V1.default_baseline == CALIB_BASELINE
    assert BODY_CORE_V1.default_baseline is not None
    # 闭环 baseline 取自 layout，与之一致。
    assert BODY_CORE_V1_CALIBRATED_BASELINE == CALIB_BASELINE
    # pose33_v3 的 baseline 已于 2026-07-11 从 2.0 校准到 3.0（同人自复现锚点）。
    assert POSE33_V3.default_baseline == 3.0


# --------------------------------------------------------------------------- #
# 2) YOLO valid_conf_thr 已标定入库（占位 0.5 移除）
# --------------------------------------------------------------------------- #
def test_yolo_valid_conf_thr_calibrated():
    assert DEFAULT_YOLO_VALID_CONF_THR == CALIB_YOLO_CONF_THR
    # 占位 0.5 已移除（YOLO 侧不再等于 MediaPipe 的 0.5）。
    assert DEFAULT_YOLO_VALID_CONF_THR != 0.5


def test_offline_benchmark_default_model_is_yolo26l(tmp_path):
    assert bench_annotate_fps._default_yolo_model_path(tmp_path) == tmp_path / "yolo26l-pose.pt"


# --------------------------------------------------------------------------- #
# 3) calibration_status 仍不授权评分（S3 只落参数）
# --------------------------------------------------------------------------- #
def test_calibration_status_remains_unvalidated():
    assert YOLO_CALIBRATION_STATUS == "unvalidated"
    assert CALIBRATION_STATUS_UNVALIDATED == "unvalidated"


# --------------------------------------------------------------------------- #
# 4) 标定报告头部含预注册数字与结论四选一
# --------------------------------------------------------------------------- #
def test_calibration_report_exists_and_has_preregistered_header():
    assert REPORT.exists(), f"标定报告缺失：{REPORT}"
    text = REPORT.read_text(encoding="utf-8")
    # 预注册判据关键词必须出现在报告中。
    for kw in (
        "预注册",
        "相关性",
        "pass/fail",
        "MAE",
        "skip",
        "失败帧率",
    ):
        assert kw in text, f"报告缺少预注册判据关键词：{kw}"
    # 标定后的数字必须写入报告。
    assert "1.2826" in text, "报告未写入 body_core_v1 标定 baseline"
    assert "0.6" in text, "报告未写入 YOLO valid_conf_thr 标定值"
    assert "self-match" in text, "报告未说明 self-match 不计入主判据"
    # 结论四选一之一必须明确出现。
    conclusions = ["仅预览", "可模板匹配", "skip-aware partial eval", "不建议"]
    assert any(c in text for c in conclusions), "报告未给出四选一结论"


def test_calibration_report_states_passfail_criteria_source():
    text = REPORT.read_text(encoding="utf-8")
    # 报告须明确 pass/fail 判定口径、阈值/标签来源与样本范围（不允许事后改口径）。
    for kw in ("判定口径", "样本范围", "阈值"):
        assert kw in text, f"报告缺少 pass/fail 口径说明关键词：{kw}"
