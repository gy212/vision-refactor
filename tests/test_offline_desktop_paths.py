import sys

from core import paths


def test_offline_bundle_seeds_writable_data_without_overwriting_user_files(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    (bundle / "models").mkdir(parents=True)
    (bundle / "models" / "pose_landmarker_lite.task").write_bytes(b"bundled-model")
    (bundle / "templates" / "online").mkdir(parents=True)
    (bundle / "templates" / "online" / "example.npz").write_bytes(b"template")
    local = tmp_path / "用户 数据"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    assert paths.repo_root() == local / "VisionSanda"
    model = paths.models_dir() / "pose_landmarker_lite.task"
    assert model.read_bytes() == b"bundled-model"
    model.write_bytes(b"user-model")
    assert (paths.models_dir() / model.name).read_bytes() == b"user-model"
    model.write_bytes(b"")
    assert (paths.models_dir() / model.name).read_bytes() == b"bundled-model"
    assert (paths.templates_dir() / "online" / "example.npz").read_bytes() == b"template"
    assert paths.outputs_dir().is_dir()
    (bundle / "models" / "pose_landmarker_lite.task").unlink()
    (bundle / "models").rmdir()
    assert paths.repo_root() == paths.Path(sys.executable).resolve().parent


def test_windows_unicode_model_uses_bytes(tmp_path, monkeypatch):
    from core import vision_pipeline
    model = tmp_path / "中文模型.task"
    model.write_bytes(b"model-data")
    monkeypatch.setattr(vision_pipeline.os, "name", "nt")
    options = vision_pipeline._base_options(model, delegate="cpu")
    assert options.model_asset_buffer == b"model-data"
    assert options.model_asset_path is None
