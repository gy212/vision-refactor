# -*- coding: utf-8 -*-
"""P0b 打包：build_info、干净构建、交付目录与自检图标检查。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apps.tk.self_check import check_feedback_icons, load_build_info  # noqa: E402
from core.evaluation import RULE_VERSION  # noqa: E402
from core.evaluation import FeedbackConfig  # noqa: E402

SPEC = _REPO_ROOT / "app_ui_onefile.spec"


def _load(name: str):
    path = _REPO_ROOT / "packaging" / "pyinstaller" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_vision_build_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_info = _load("build_info")
delivery = _load("delivery")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    _git(tmp_path, "config", "core.autocrlf", "false")
    rule_file = tmp_path / build_info.RULE_VERSION_SOURCE
    rule_file.parent.mkdir(parents=True)
    rule_file.write_text('RULE_VERSION = "rule-v-test"\n', encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "keep.py").write_text("", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_clean_repo_passes_and_build_info_has_required_fields(repo: Path):
    model = repo / "m.task"
    model.write_bytes(b"model")
    (repo / ".gitignore").write_text("m.task\n", encoding="utf-8")  # 根目录文件不在打包路径内

    build_info.require_clean(repo)
    info = build_info.collect(repo, [model], pyinstaller_version="6.22.3")

    assert info["clean"] is True
    assert len(info["commit"]) == 40
    assert info["ruleVersion"] == "rule-v-test"
    assert info["models"] == {"m.task": hashlib.sha256(b"model").hexdigest()}
    assert info["pyinstaller"] == "6.22.3"
    assert info["python"] == f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    assert info["builtAt"].endswith("+00:00")


@pytest.mark.parametrize("make_dirty", ["modified", "untracked"])
def test_dirty_source_refuses_build(repo: Path, make_dirty: str):
    if make_dirty == "modified":
        (repo / build_info.RULE_VERSION_SOURCE).write_text('RULE_VERSION = "changed"\n', encoding="utf-8")
    else:
        (repo / "core" / "new_module.py").write_text("", encoding="utf-8")  # v4 曾从未跟踪源码出包

    with pytest.raises(SystemExit, match="拒绝构建"):
        build_info.require_clean(repo)
    assert build_info.collect(repo, [], pyinstaller_version="x")["clean"] is False


def test_changes_outside_packaged_paths_do_not_block_build(repo: Path):
    (repo / "tests" / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "notes.md").write_text("", encoding="utf-8")

    build_info.require_clean(repo)


def test_real_rule_version_is_read_without_importing():
    assert build_info.read_rule_version(_REPO_ROOT) == RULE_VERSION


def test_delivery_assembles_exe_build_info_and_attachments(tmp_path: Path):
    exe = tmp_path / "vision_ui.exe"
    exe.write_bytes(b"exe")
    info = {"commit": "abcdef1234567890", "builtAt": "2026-10-03T12:00:00+00:00"}
    info_path = build_info.write(info, tmp_path / "build" / build_info.BUILD_INFO_NAME)

    target = delivery.assemble(exe, info_path, info, tmp_path / "dist")

    assert target.name == "vision_ui_20261003_abcdef1"
    names = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()}
    assert {"vision_ui.exe", "build_info.json", "使用说明.txt", "feedback_config.example.json"} <= names
    assert any(name.startswith("第三方许可/") for name in names)
    assert json.loads((target / "build_info.json").read_text(encoding="utf-8")) == info


def test_feedback_config_example_matches_current_defaults():
    example = json.loads((delivery.ATTACHMENTS_DIR / "feedback_config.example.json").read_text(encoding="utf-8"))
    defaults = asdict(FeedbackConfig())
    defaults.pop("pose_variant")  # 由界面的分析模型下拉框决定

    assert example == defaults
    FeedbackConfig(**example)


def test_source_tree_has_all_feedback_icons_and_no_build_info():
    assert check_feedback_icons(_REPO_ROOT) == 24
    assert load_build_info(_REPO_ROOT) is None


def test_missing_feedback_icon_is_reported(tmp_path: Path):
    (tmp_path / "assets" / "feedback").mkdir(parents=True)
    with pytest.raises(AssertionError, match="缺少结果页图标"):
        check_feedback_icons(tmp_path)


def test_single_tk_spec_bundles_feedback_assets_and_build_info():
    assert sorted(p.name for p in _REPO_ROOT.glob("*.spec")) == ["app_ui_onefile.spec"]
    text = SPEC.read_text(encoding="utf-8")
    assert '("assets/feedback", "assets/feedback")' in text
    assert "build_info.require_clean" in text
    assert "delivery.assemble" in text


def test_every_runtime_requirement_has_a_license_in_delivery():
    licenses = delivery.ATTACHMENTS_DIR / "第三方许可"
    manifest = (licenses / "清单.txt").read_text(encoding="utf-8-sig")
    lines = (_REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    runtime = []
    for line in lines:
        if "测试依赖" in line:
            break
        if line.startswith("#") or not line.strip():
            continue
        runtime.append(line.split("==")[0].split(">=")[0].strip())
    for name in runtime + ["imageio-ffmpeg"]:
        files = list((licenses / name).glob("*"))
        assert files, f"第三方许可缺少 {name}"
        assert name in manifest
