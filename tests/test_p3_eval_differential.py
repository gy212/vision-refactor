# -*- coding: utf-8 -*-
"""P3 差分测试：评估层迁移前录下的基准输出（``tests/fixtures/p3_eval_baseline.json.gz``）与当前实现逐项一致。

用例见 ``tests/eval_diff/``：DTW 与代表周期、单/多模板比对、双模板与正侧双流、正式规则、踢腿质量、
考试池录后处理全流程、模板校验、在线识别、模板生成与自动裁剪、动作区间、正侧切分、技术指标、body_core、
CLI、batch、离线剖析，以及学生练习（检查清单、逐帧序列、相位与融合、文本报告、``feedback_config.json`` 覆盖）。
浮点逐位相等（NaN 视为相等），dtype、tuple/list、dataclass 名、报错类型与文本都参与比较；大数组比 SHA-256。
基准在 P3 开工前的 ``main``（``136c8d0``）上用 ``python -B -m tests.eval_diff.record fixture`` 录制；
只有 §11.2 已批准的决策可以改变基准，且单独 PR、单独重录。
"""
from __future__ import annotations

from tests.eval_diff import record


def test_eval_outputs_match_pre_p3_baseline(tmp_path):
    expected = record.load(record.BASELINE)
    actual = record.roundtrip(record.record_fixture(tmp_path))  # 与基准一样经过 JSON 往返再比
    problems = record.compare(expected, actual)
    assert not problems, f"{len(problems)} 处差异（最多列 60 条）：\n" + "\n".join(problems[:60])
