# -*- coding: utf-8 -*-
"""样本评估的一次运行：读清单 → 补齐姿态缓存 → 逐对逐动作评估 → 写运行目录与单次报告。

样本根目录由参数或环境变量 ``VISION_SAMPLES`` 给出，视频与缓存都不在仓库里；
运行目录默认在 ``outputs/sample_eval/`` 下（已被 git 忽略）。
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from analysis.sample_eval.cache import POSE_VARIANT, build_cache, load_views, plan_jobs
from analysis.sample_eval.evaluate import default_config, evaluate_pair
from analysis.sample_eval.manifest import pair_role, read_manifest
from analysis.sample_eval.metrics import summarize
from analysis.sample_eval.report import render_report
from analysis.sample_eval.runs import REPORT_MD, file_sha256, git_commit, read_run, write_run
from core.evaluation import RULE_VERSION
from core.foundation import atomic_write_text, outputs_dir

SAMPLES_ENV = "VISION_SAMPLES"


def resolve_samples_root(arg: str | None) -> Path:
    """参数优先，其次环境变量 ``VISION_SAMPLES``；都没有或目录不存在时报错。"""
    value = arg or os.environ.get(SAMPLES_ENV, "")
    if not value:
        raise SystemExit(f"请用 --samples-root 或环境变量 {SAMPLES_ENV} 指定样本根目录（见 docs/样本清单.md）")
    root = Path(value)
    if not root.is_dir():
        raise SystemExit(f"样本根目录不存在：{root}")
    return root


def default_cache_dir(samples_root: Path) -> Path:
    """姿态缓存默认按样本批次分目录，修改前后的两次运行共用。"""
    return outputs_dir() / "sample_eval" / "pose_cache" / Path(samples_root).name


def _evaluate_all(pairs, cache_dir: Path, results_dir: Path, echo: Callable[[str], None]) -> tuple[list, list, int]:
    cfg = default_config()
    check_rows, phase_rows, evaluated = [], [], 0
    results_dir.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        if not pair.rule_actions:
            continue
        views = load_views(pair, cache_dir)
        if not views:
            echo(f"{pair.pair_id}: 没有可用的姿态缓存，跳过")
            continue
        evaluated += 1
        role = pair_role(pair)
        for action in pair.rule_actions:
            result, rows, phase = evaluate_pair(pair, role, action, views, cfg)
            atomic_write_text(results_dir / f"{pair.pair_id}__{action}.json",
                              json.dumps(result, ensure_ascii=False, default=str))
            check_rows.extend(rows)
            phase_rows.append(phase)
            echo(f"{pair.pair_id} {action} {result['status']}")
    return check_rows, phase_rows, evaluated


def run_evaluation(manifest: Path, samples_root: Path, out_dir: Path, *, cache_dir: Path | None = None,
                   models: Path | None = None, workers: int = 4, label: str = "",
                   echo: Callable[[str], None] = print) -> Path:
    """跑一次完整评估，返回运行目录。``models`` 缺省为仓库的 ``models/``。"""
    pairs = read_manifest(manifest)
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir(samples_root)
    jobs, missing = plan_jobs(pairs, samples_root, cache_dir, models)
    for item in missing:
        echo(f"缺失视频：{item}")
    cache = build_cache(jobs, cache_dir, workers, echo)
    started = time.time()
    check_rows, phase_rows, evaluated = _evaluate_all(pairs, cache_dir, Path(out_dir) / "results", echo)
    meta = {"label": label or Path(out_dir).name, "rule_version": RULE_VERSION, "commit": git_commit(),
            "manifest": str(manifest), "manifest_sha256": file_sha256(manifest)[:12], "samples_root": str(samples_root),
            "cache_dir": str(cache_dir), "pose_variant": POSE_VARIANT, "pairs": len(pairs),
            "evaluated_pairs": evaluated, "evaluations": len(phase_rows), "missing_videos": len(missing),
            "cache": f"{cache['reused']}/{cache['extracted']}/{cache['failed']}",
            "config": asdict(default_config()), "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluate_sec": round(time.time() - started, 1)}
    write_run(out_dir, meta, check_rows, phase_rows)
    write_report(out_dir)
    return Path(out_dir)


def write_report(run_dir: Path, baseline_dir: Path | None = None, out: Path | None = None) -> Path:
    """由运行目录（及可选的基线运行目录）生成报告；默认写到运行目录的 ``report.md``。"""
    meta, rows, phases = read_run(run_dir)
    before_meta = before = None
    if baseline_dir is not None:
        before_meta, before_rows, before_phases = read_run(baseline_dir)
        before = summarize(before_rows, before_phases)
    target = Path(out) if out else Path(run_dir) / REPORT_MD
    atomic_write_text(target, render_report(meta, summarize(rows, phases), before_meta, before))
    return target
