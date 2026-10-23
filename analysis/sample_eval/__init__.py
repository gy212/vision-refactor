# -*- coding: utf-8 -*-
"""样本评估工具（规则检出修复计划步骤 1）：按样本清单批量评估 v5 规则，输出误报率、检出率、阶段检出与对齐失败的对比报告。

离线工具，只用 ``core`` 各包的公共 API，不接入生产路径。命令行见 ``python -m analysis.sample_eval --help``。
"""
from analysis.sample_eval.manifest import ManifestError, PairRole, SamplePair, pair_role, read_manifest
from analysis.sample_eval.metrics import summarize
from analysis.sample_eval.pipeline import run_evaluation, write_report
from analysis.sample_eval.report import render_report

__all__ = [
    "ManifestError",
    "PairRole",
    "SamplePair",
    "pair_role",
    "read_manifest",
    "render_report",
    "run_evaluation",
    "summarize",
    "write_report",
]
