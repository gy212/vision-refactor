# -*- coding: utf-8 -*-
"""S5 realtime Hands toggle tests (Issue #41) & opt-in default contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apps.cli import main as app_main  # noqa: E402
from apps.tk.controllers.flags import LiveFlags  # noqa: E402
from core.foundation.prefs import CAMERA_SELECTION, PreferenceStore  # noqa: E402
from core.live import LiveControls, SessionConfig, camera_session, video_session  # noqa: E402
from core.live.stages import PoseStage, StageContext  # noqa: E402
from core.media import PRIMARY, SECONDARY, CaptureSpec  # noqa: E402
from core.pose import EngineSpec, create_engine  # noqa: E402
from core.pose import engine as pose_engine_mod  # noqa: E402
from core.pose.backends import BackendRouteRequest, TaskType  # noqa: E402
from core.services import BenchmarkSetup  # noqa: E402
from core.workflows import PlanSettings, build_segment_plan  # noqa: E402


def test_cli_defaults_to_hands_disabled(monkeypatch):
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(sys, "argv", ["main.py", "--source", "0", "--no-show"])
    monkeypatch.setattr(
        app_main,
        "run",
        lambda source, **kwargs: calls.append({"source": source, **kwargs}),
    )

    app_main.main()

    assert calls == [
        {
            "source": "0",
            "show": False,
            "out_path": None,
            "pose_variant": "full",
            "enable_hands": False,
            "delegate": "cpu",
        }
    ]


def test_cli_no_hands_disables_single_thread_preview(monkeypatch):
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(sys, "argv", ["main.py", "--source", "0", "--no-show", "--no-hands"])
    monkeypatch.setattr(
        app_main,
        "run",
        lambda source, **kwargs: calls.append({"source": source, **kwargs}),
    )

    app_main.main()

    assert calls[0]["enable_hands"] is False


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        (["--hands"], True),
        (["--no-hands"], False),
        (["--no-hands", "--hands"], True),
        (["--hands", "--no-hands"], False),
    ],
)
def test_cli_hands_and_no_hands_flags_toggle_single_thread_preview(
    monkeypatch, extra_args: list[str], expected: bool
):
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(sys, "argv", ["main.py", "--source", "0", "--no-show", *extra_args])
    monkeypatch.setattr(
        app_main,
        "run",
        lambda source, **kwargs: calls.append({"source": source, **kwargs}),
    )

    app_main.main()

    assert len(calls) == 1
    assert calls[0]["enable_hands"] is expected


def test_cli_no_hands_disables_parallel_offline_pipeline(monkeypatch):
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--source", "sample.mp4", "--workers", "2", "--no-show", "--no-hands"],
    )
    monkeypatch.setattr(
        app_main,
        "run_parallel",
        lambda source, **kwargs: calls.append({"source": source, **kwargs}),
    )

    app_main.main()

    assert len(calls) == 1
    assert calls[0]["source"] == "sample.mp4"
    assert calls[0]["enable_hands"] is False
    assert calls[0]["pose_variant"] == "full"
    assert calls[0]["workers"] == 2


def test_cli_hands_flag_enables_parallel_and_latest_frame_pipelines(monkeypatch):
    parallel_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--source", "sample.mp4", "--workers", "2", "--no-show", "--hands"],
    )
    monkeypatch.setattr(
        app_main,
        "run_parallel",
        lambda source, **kwargs: parallel_calls.append({"source": source, **kwargs}),
    )
    app_main.main()
    assert len(parallel_calls) == 1
    assert parallel_calls[0]["enable_hands"] is True

    smoke_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--source",
            "sample.mp4",
            "--realtime-latest-frame",
            "--limit-frames",
            "3",
            "--no-show",
            "--hands",
        ],
    )
    monkeypatch.setattr(
        app_main,
        "run_realtime_latest_frame_smoke",
        lambda source, **kwargs: smoke_calls.append({"source": source, **kwargs})
        or app_main.RealtimeLatestFrameMetrics(3, 3, 3, 0, 30.0, 30.0, 30.0, 1.0),
    )
    app_main.main()
    assert len(smoke_calls) == 1
    assert smoke_calls[0]["enable_hands"] is True


def test_all_config_defaults_disable_hands_until_opted_in():
    assert EngineSpec().enable_hands is False
    assert EngineSpec(enable_hands=True).enable_hands is True

    assert SessionConfig(video_path=Path("clip.mp4")).enable_hands is False
    assert camera_session({PRIMARY: CaptureSpec(0)}, skeleton=True).enable_hands is False
    assert (
        camera_session({PRIMARY: CaptureSpec(0)}, skeleton=True, enable_hands=True).enable_hands
        is True
    )
    assert video_session("clip.mp4").enable_hands is False
    assert video_session("clip.mp4", enable_hands=True).enable_hands is True

    default_settings = PlanSettings(record_skeleton=True, auto_compare=False)
    opted_settings = PlanSettings(record_skeleton=True, auto_compare=False, enable_hands=True)
    assert default_settings.enable_hands is False
    assert opted_settings.enable_hands is True
    single_seg = SimpleNamespace(multi=False)
    assert build_segment_plan(single_seg, default_settings).steps[1].engine.enable_hands is False
    assert build_segment_plan(single_seg, opted_settings).steps[1].engine.enable_hands is True

    assert LiveFlags().enable_hands is False
    assert BackendRouteRequest(task_type=TaskType.REALTIME_PREVIEW.value).enable_hands is False
    setup = BenchmarkSetup(
        stream=None,  # type: ignore[arg-type]
        rotations={},
        labels={},
        work_dir=Path("."),
        engine_factory=lambda spec: spec,
    )
    assert setup.enable_hands is False


def test_engine_loads_and_runs_hand_landmarker_only_when_opted_in(tmp_path, monkeypatch):
    pose_file = tmp_path / "pose_landmarker_full.task"
    pose_file.write_bytes(b"pose-model")

    events: list[str] = []

    class FakePoseLandmarker:
        def detect_for_video(self, _img, _ts):
            events.append("pose_infer")
            return SimpleNamespace(pose_landmarks=["pose_lm"])

        def close(self) -> None:
            events.append("pose_close")

    class FakeHandLandmarker:
        def detect_for_video(self, _img, _ts):
            events.append("hand_infer")
            return SimpleNamespace(hand_landmarks=["hand_lm"])

        def close(self) -> None:
            events.append("hand_close")

    class FakePoseFactory:
        @staticmethod
        def create_from_options(options):
            events.append(f"pose_create:{Path(options.base_options.model_asset_path).name}")
            return FakePoseLandmarker()

    class FakeHandFactory:
        @staticmethod
        def create_from_options(options):
            events.append(f"hand_create:{Path(options.base_options.model_asset_path).name}")
            return FakeHandLandmarker()

    fake_mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB="SRGB"),
        Image=lambda **kwargs: SimpleNamespace(**kwargs),
        tasks=SimpleNamespace(
            BaseOptions=lambda **kwargs: SimpleNamespace(**kwargs),
            vision=SimpleNamespace(
                RunningMode=SimpleNamespace(VIDEO="VIDEO", IMAGE="IMAGE"),
                PoseLandmarker=FakePoseFactory,
                PoseLandmarkerOptions=lambda **kwargs: SimpleNamespace(**kwargs),
                HandLandmarker=FakeHandFactory,
                HandLandmarkerOptions=lambda **kwargs: SimpleNamespace(**kwargs),
            ),
        ),
    )

    monkeypatch.setattr(pose_engine_mod, "mediapipe", lambda: fake_mp)
    monkeypatch.setattr(
        pose_engine_mod.urllib.request,
        "urlretrieve",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    # 默认 enable_hands=False：不要求 hand_landmarker.task、不构造也不调用 HandLandmarker
    default_engine = create_engine(EngineSpec(), models_dir=tmp_path, download=False)
    try:
        assert default_engine.hand_landmarker is None
        pose_lm, hands = default_engine.infer(frame, timestamp_ms=10)
        assert pose_lm == "pose_lm"
        assert hands == []
    finally:
        default_engine.close()

    assert events == [
        "pose_create:pose_landmarker_full.task",
        "pose_infer",
        "pose_close",
    ]

    # 显式开启 enable_hands=True：完整加载并运行 HandLandmarker
    events.clear()
    (tmp_path / "hand_landmarker.task").write_bytes(b"hand-model")
    optin_engine = create_engine(EngineSpec(enable_hands=True), models_dir=tmp_path, download=False)
    try:
        assert optin_engine.hand_landmarker is not None
        pose_lm, hands = optin_engine.infer(frame, timestamp_ms=20)
        assert pose_lm == "pose_lm"
        assert hands == ["hand_lm"]
    finally:
        optin_engine.close()

    assert events == [
        "pose_create:pose_landmarker_full.task",
        "hand_create:hand_landmarker.task",
        "pose_infer",
        "hand_infer",
        "pose_close",
        "hand_close",
    ]


def test_user_prefs_preserves_existing_and_unknown_keys(tmp_path):
    prefs_file = tmp_path / "user_prefs.json"
    prefs_file.write_text(
        json.dumps(
            {
                "camera_selection": {"primary_index": 1, "secondary_index": 2},
                "legacy_custom_flag": True,
            }
        ),
        encoding="utf-8",
    )
    store = PreferenceStore(lambda: prefs_file)
    assert store.get(CAMERA_SELECTION) == (1, 2)
    store.set(CAMERA_SELECTION, (0, 2), strict=True)
    saved = json.loads(prefs_file.read_text(encoding="utf-8"))
    assert saved["camera_selection"] == {"primary_index": 0, "secondary_index": 2}
    assert saved["legacy_custom_flag"] is True


def test_live_pose_stage_respects_default_false_and_optin_true(monkeypatch):
    """不依赖 Tk：实时会话默认关闭手部，显式 enable_hands=True 时为每路引擎开启手部。"""
    import core.live.stages as stages

    specs: list[EngineSpec] = []

    class _Engine:
        def __init__(self, spec: EngineSpec) -> None:
            specs.append(spec)

        def next_timestamp_ms(self, **_kwargs):
            return 0

        def infer(self, frame, *, timestamp_ms=None):
            return None, []

        def close(self) -> None:
            pass

    monkeypatch.setattr(stages, "create_engine", lambda spec, models_dir=None: _Engine(spec))
    monkeypatch.setattr(stages, "models_dir", lambda: Path("models"))

    default_configs = (
        camera_session({PRIMARY: CaptureSpec(0)}, pose_variant="heavy", skeleton=True),
        camera_session(
            {PRIMARY: CaptureSpec(0), SECONDARY: CaptureSpec(1)},
            pose_variant="heavy",
            skeleton=True,
        ),
        video_session("video.mp4", pose_variant="heavy"),
    )
    for config in default_configs:
        assert config.enable_hands is False
        specs.clear()
        PoseStage(config.roles).open(StageContext(config, LiveControls(), lambda *_: None))
        assert specs and all(spec.enable_hands is False for spec in specs)
        assert all(spec.pose_variant == "heavy" for spec in specs)

    optin_configs = (
        camera_session(
            {PRIMARY: CaptureSpec(0)}, pose_variant="heavy", skeleton=True, enable_hands=True
        ),
        camera_session(
            {PRIMARY: CaptureSpec(0), SECONDARY: CaptureSpec(1)},
            pose_variant="heavy",
            skeleton=True,
            enable_hands=True,
        ),
        video_session("video.mp4", pose_variant="heavy", enable_hands=True),
    )
    for config in optin_configs:
        assert config.enable_hands is True
        specs.clear()
        PoseStage(config.roles).open(StageContext(config, LiveControls(), lambda *_: None))
        assert specs and all(spec.enable_hands is True for spec in specs)
        assert all(spec.pose_variant == "heavy" for spec in specs)


def test_ui_session_pipelines_use_state_enable_hands(monkeypatch):
    """界面的手部开关进入会话配置，推理 stage 按配置构造引擎；站位检测固定 lite、不跑手部（bf8c908）。

    P6 起取代对 app_ui 源码的 AST 检查：引擎在 core.live 的 stage 里构造。
    """
    pytest.importorskip("tkinter")

    from apps.tk.controllers.session import UiState, session_config
    import core.live.presence as presence
    import core.live.stages as stages
    from core.live import PresenceSampler
    from core.media import CapturedFrame

    specs = []

    class _Engine:
        def __init__(self, spec):
            specs.append(spec)

        def next_timestamp_ms(self, **_kwargs):
            return 0

        def infer(self, frame, *, timestamp_ms=None):
            return None, []

        def close(self):
            pass

    monkeypatch.setattr(stages, "create_engine", lambda spec, models_dir=None: _Engine(spec))
    monkeypatch.setattr(stages, "models_dir", lambda: Path("models"))
    monkeypatch.setattr(presence, "create_engine", lambda spec, models_dir=None: _Engine(spec))
    monkeypatch.setattr(presence, "models_dir", lambda: Path("models"))

    for source, source2 in (("0", None), ("0", "1"), ("video.mp4", None)):
        for hands in (True, False):
            state = UiState(
                source=source,
                source2=source2,
                pose_variant="heavy",
                workers=1,
                enable_hands=hands,
                record_skeleton=True,
            )
            config = session_config(state)
            assert config.enable_hands is hands
            specs.clear()
            PoseStage(config.roles).open(StageContext(config, LiveControls(), lambda *_: None))
            assert specs and all(spec.enable_hands is hands for spec in specs)
            assert all(spec.pose_variant == "heavy" for spec in specs)

    # 站位检测管线：骨架关的双摄会话里启用后构造，恰一个，lite + 无手部，与界面开关无关。
    state = UiState(source="0", source2="1", pose_variant="heavy", workers=1, enable_hands=True)
    config = session_config(state)
    controls = LiveControls()
    controls.arm_occupancy("exam", True)
    specs.clear()
    assert config.occupancy
    sampler = PresenceSampler(lambda _pose, _roi: True)

    class _Feed:
        def latest(self, role):
            return CapturedFrame(role, 1, 0, np.zeros((8, 8, 3), np.uint8))

    sampler._sample(_Feed(), controls, lambda *_: None)
    assert len(specs) == 1
    assert specs[0].pose_variant == "lite" and specs[0].enable_hands is False
