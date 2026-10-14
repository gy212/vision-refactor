"""P3 评估差分的录制与比对入口。

用法（仓库根，或旧 commit 的 worktree 根）：
    python -B -m tests.eval_diff.record fixture <输出.json.gz>
    python -B -m tests.eval_diff.record real <视频目录> <输出.json.gz>
    python -B -m tests.eval_diff.record compare <a.json.gz> <b.json.gz>
real 模式的视频目录放 single.*、front.*、side.*、practice.*（同 tests/real_video_diff）。用例只经旧入口与门面调用，
P3 前后都能原样运行；fixture 模式的基准入库为 tests/fixtures/p3_eval_baseline.json.gz，由 test_p3_eval_differential 比对。
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests import golden_harness as H  # noqa: E402
from tests.eval_diff import cases_match, cases_misc, cases_practice, cases_rules, synth  # noqa: E402
from tests.eval_diff.canon import Canon, diff, dump, load  # noqa: E402
from tests.eval_diff.ctx import Ctx  # noqa: E402

BASELINE = _ROOT / "tests" / "fixtures" / "p3_eval_baseline.json.gz"
_FIXTURE_RAW = {"student": "student_raw", "front": "front_src_raw", "side": "side_src_raw"}


def _recorder(work: Path, store: dict):
    canon = Canon(work)

    def rec(key: str, fn) -> None:
        assert key not in store, f"重复的用例键：{key}"
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001 - 报错本身也是要比对的行为
            value = exc
        store[key] = canon(value)

    return rec


def record_fixture(work: Path) -> dict:
    from core.pose import extract

    store: dict = {}
    rec = _recorder(work, store)
    H.clear_registry()
    with H.replay_context():
        extract.create_engine = lambda spec, **_kw: synth.GapPipeline(cfg=spec)
        v = synth.fixture_videos()
        c = Ctx(rec, v, synth.write_templates(work), work, "fixture",
                place=lambda dst, name: H.register_video(dst, synth.load_raw(_FIXTURE_RAW[name]), fps=synth.FPS),
                raw=lambda name: synth.load_raw(_FIXTURE_RAW[name]))
        cases_misc.body_core(c)
        cases_match.run(c)
        cases_rules.run(c)
        cases_misc.run(c)
    H.clear_registry()
    cases_practice.run(c)  # 练习解码真实 AVI，不在回放上下文里
    return store


def record_real(video_dir: Path, work: Path) -> dict:
    from core.pose import pose_cache
    from core.pose import extract_pose_raw_series

    found = {}
    for key in ("single", "front", "side", "practice"):
        hits = sorted(p for p in video_dir.glob(f"{key}.*") if p.is_file())
        if not hits:
            raise SystemExit(f"{video_dir} 里缺少 {key}.*")
        found[key] = hits[0]
    v = {"student": found["single"], "front": found["front"], "side": found["side"], "gappy": found["side"],
         "mirror": found["front"], "long": found["single"], "short": found["single"],
         "practice_front": found["practice"], "practice_side": found["side"]}
    name_of = {"student": "single", "front": "front", "side": "side"}
    raws: dict[str, np.ndarray] = {}

    def raw(name: str) -> np.ndarray:
        if name not in raws:
            raws[name] = np.asarray(extract_pose_raw_series(found[name_of[name]], pose_variant="full").landmarks)
        return raws[name]

    store: dict = {}
    with pose_cache():
        c = Ctx(_recorder(work, store), v, synth.write_templates(work), work, "real",
                place=lambda dst, name: shutil.copyfile(found[name_of[name]], dst), raw=raw)
        cases_misc.body_core(c)
        cases_match.run(c)
        cases_rules.run(c)
        cases_misc.run(c)
        cases_practice.real_practice(c)
    return store


def roundtrip(data: dict) -> dict:
    import json

    return json.loads(json.dumps(data, ensure_ascii=False, allow_nan=True))


def compare(a: dict, b: dict) -> list[str]:
    problems = [f"缺少用例：{k}" for k in a if k not in b] + [f"多出用例：{k}" for k in b if k not in a]
    for key in a:
        if key in b:
            problems += diff(a[key], b[key], key)
    return problems


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] in ("fixture", "real"):
        out = Path(argv[-1])
        with tempfile.TemporaryDirectory(prefix="p3diff_", ignore_cleanup_errors=True) as tmp:
            data = record_fixture(Path(tmp)) if argv[0] == "fixture" else record_real(Path(argv[1]), Path(tmp))
        dump(out, data)
        print(f"recorded {len(data)} cases -> {out}")
        return 0
    if len(argv) == 3 and argv[0] == "compare":
        problems = compare(load(Path(argv[1])), load(Path(argv[2])))
        print("\n".join(problems[:200]) or "0 差异")
        return 1 if problems else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
