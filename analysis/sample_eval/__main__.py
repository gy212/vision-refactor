# -*- coding: utf-8 -*-
r"""样本评估工具的命令行入口：``run`` 跑一次评估，``report`` 重新生成报告，``compare`` 生成修改前后的对比报告。

示例（PowerShell，仓库根）::

    $env:VISION_SAMPLES = "D:\VisionSamples\2026-10-16_client_batch1"
    .\.venv\Scripts\python.exe -m analysis.sample_eval run --manifest docs/samples/2026-10-16_client_batch1.csv `
        --out outputs/sample_eval/before --workers 6
    # 改规则后
    .\.venv\Scripts\python.exe -m analysis.sample_eval run --manifest docs/samples/2026-10-16_client_batch1.csv `
        --out outputs/sample_eval/after --workers 6
    .\.venv\Scripts\python.exe -m analysis.sample_eval compare --before outputs/sample_eval/before `
        --after outputs/sample_eval/after --out outputs/sample_eval/compare.md
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import freeze_support
from pathlib import Path

from analysis.sample_eval.pipeline import resolve_samples_root, run_evaluation, write_report


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m analysis.sample_eval", description="样本评估：规则误报率、检出率、阶段检出与对齐")
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="补齐姿态缓存并评估全部样本，写运行目录与 report.md")
    run.add_argument("--manifest", required=True, type=Path, help="样本清单 CSV（见 docs/样本清单.md）")
    run.add_argument("--samples-root", default=None, help="样本根目录；缺省取环境变量 VISION_SAMPLES")
    run.add_argument("--out", required=True, type=Path, help="运行目录，如 outputs/sample_eval/before")
    run.add_argument("--cache", default=None, type=Path, help="姿态缓存目录；缺省 outputs/sample_eval/pose_cache/<样本根目录名>")
    run.add_argument("--models-dir", default=None, type=Path, help="MediaPipe 模型目录；缺省为仓库的 models/（子进程不下载模型）")
    run.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)), help="提取姿态的进程数")
    run.add_argument("--label", default="", help="报告里的运行标签；缺省为运行目录名")
    rep = sub.add_parser("report", help="由运行目录重新生成 report.md")
    rep.add_argument("--run", required=True, type=Path)
    rep.add_argument("--out", default=None, type=Path)
    cmp_ = sub.add_parser("compare", help="修改前（--before）与修改后（--after）的对比报告")
    cmp_.add_argument("--before", required=True, type=Path)
    cmp_.add_argument("--after", required=True, type=Path)
    cmp_.add_argument("--out", required=True, type=Path)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        run_dir = run_evaluation(args.manifest, resolve_samples_root(args.samples_root), args.out,
                                 cache_dir=args.cache, models=args.models_dir, workers=args.workers, label=args.label)
        print(f"完成：{run_dir / 'report.md'}")
    elif args.command == "report":
        print(f"完成：{write_report(args.run, out=args.out)}")
    else:
        print(f"完成：{write_report(args.after, args.before, args.out)}")
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
