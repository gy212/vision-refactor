# -*- coding: utf-8 -*-
"""RecordingController 属性测试套件。

设计来源：``.kiro/specs/ui-layout-redesign/design.md``（Correctness Properties P1–P6）。

本文件承载与 Tkinter 解耦的纯状态机 ``RecordingController`` 的属性测试。任务 3.1
仅实现**共享测试夹具**：

  - ``FakeWriter``：记录 ``write``/``release`` 调用次数的假 writer，可配置第 N 次
    ``write`` 抛错（1-based）。
  - ``make_fake_writer_factory``：构造可注入的假 writer 工厂，可配置创建时抛错，
    返回 ``FakeWriter`` 实例与确定性路径。
  - ``ReferenceModel``：用于 model-based 断言的简单引用模型，独立复刻控制器应有的
    状态转换与写入计数语义，供属性测试对照。

各条正确性属性的属性测试（Property 1–6）由后续任务 3.2–3.7 追加到本文件。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.recording_controller import (  # noqa: E402
    RecordingController,
    RecordingState,
)


# ---------------------------------------------------------------------------
# 测试夹具：假 writer 与假 writer 工厂
# ---------------------------------------------------------------------------


class FakeWriter:
    """记录调用的假 ``VideoWriterLike``，可配置在第 N 次 ``write`` 抛错。

    满足 ``core.recording_controller.VideoWriterLike`` 协议（``write`` / ``release``）。

    参数：
        raise_on_write: 1-based 的写入序号，表示第几次 ``write`` 调用应抛出
            ``RuntimeError``；为 ``None`` 时永不抛错。例如 ``raise_on_write=1``
            表示首次 ``write`` 即抛错。
        raise_on_release: 为 True 时 ``release`` 调用抛错（用于验证释放阶段二次异常
            被吞掉、状态仍能复位）。

    属性：
        write_calls: 累计收到的 ``write`` 调用次数（含抛错的那一次）。
        release_calls: 累计收到的 ``release`` 调用次数。
        frames: 成功写入（未抛错）的帧对象列表，便于断言写入内容。
    """

    def __init__(
        self,
        *,
        raise_on_write: Optional[int] = None,
        raise_on_release: bool = False,
    ) -> None:
        self.raise_on_write = raise_on_write
        self.raise_on_release = raise_on_release
        self.write_calls = 0
        self.release_calls = 0
        self.frames: List[object] = []

    def write(self, frame) -> None:
        self.write_calls += 1
        if self.raise_on_write is not None and self.write_calls == self.raise_on_write:
            raise RuntimeError(
                f"FakeWriter: 模拟第 {self.write_calls} 次 write 失败"
            )
        self.frames.append(frame)

    def release(self) -> None:
        self.release_calls += 1
        if self.raise_on_release:
            raise RuntimeError("FakeWriter: 模拟 release 失败")


class FakeWriterFactory:
    """可注入的假 writer 工厂，匹配 ``WriterFactory`` 签名。

    可调用为 ``factory(path, fps, size) -> (writer, actual_path, codec)``。

    参数：
        raise_on_create: 为 True 时在被调用（创建 writer）时抛出 ``RuntimeError``，
            模拟 ``open_video_writer`` 编码器全部回退失败的场景（需求 5.11）。
        raise_on_write: 透传给所创建的 ``FakeWriter``，配置其第 N 次 write 抛错。
        raise_on_release: 透传给所创建的 ``FakeWriter``。
        actual_suffix: 注入到返回 ``actual_path`` 的后缀，用于模拟编码回退改后缀
            （例如传入路径为 ``.mp4`` 而实际落盘为 ``.avi``）。
        codec: 返回的 codec 标签。

    属性：
        create_calls: 累计被调用（尝试创建 writer）的次数。
        writers: 已创建的 ``FakeWriter`` 实例列表（按创建顺序）。
        last_writer: 最近一次创建的 ``FakeWriter``（无则为 None）。
    """

    def __init__(
        self,
        *,
        raise_on_create: bool = False,
        raise_on_write: Optional[int] = None,
        raise_on_release: bool = False,
        actual_suffix: Optional[str] = None,
        codec: str = "fake",
    ) -> None:
        self.raise_on_create = raise_on_create
        self.raise_on_write = raise_on_write
        self.raise_on_release = raise_on_release
        self.actual_suffix = actual_suffix
        self.codec = codec
        self.create_calls = 0
        self.writers: List[FakeWriter] = []

    @property
    def last_writer(self) -> Optional[FakeWriter]:
        return self.writers[-1] if self.writers else None

    def __call__(
        self, path: Path, fps: float, size: Tuple[int, int]
    ) -> Tuple[FakeWriter, Path, str]:
        self.create_calls += 1
        if self.raise_on_create:
            raise RuntimeError("FakeWriterFactory: 模拟 writer 创建失败")

        # 确定性 actual_path：可选地替换后缀以模拟编码回退改后缀。
        path = Path(path)
        actual_path = path.with_suffix(self.actual_suffix) if self.actual_suffix else path

        writer = FakeWriter(
            raise_on_write=self.raise_on_write,
            raise_on_release=self.raise_on_release,
        )
        self.writers.append(writer)
        return writer, actual_path, self.codec


def make_fake_writer_factory(
    *,
    raise_on_create: bool = False,
    raise_on_write: Optional[int] = None,
    raise_on_release: bool = False,
    actual_suffix: Optional[str] = None,
    codec: str = "fake",
) -> FakeWriterFactory:
    """便捷构造器：返回一个 ``FakeWriterFactory`` 实例（见其文档）。"""
    return FakeWriterFactory(
        raise_on_create=raise_on_create,
        raise_on_write=raise_on_write,
        raise_on_release=raise_on_release,
        actual_suffix=actual_suffix,
        codec=codec,
    )


# 确定性路径生成器：供 RecordingController.path_provider 注入，避免依赖时间戳/真实目录。
DEFAULT_FAKE_PATH = Path("fake_outputs") / "record_fake.mp4"


def make_fake_path_provider(path: Path = DEFAULT_FAKE_PATH):
    """返回一个总是产出确定性 ``path`` 的 path_provider。"""

    def _provider() -> Path:
        return path

    return _provider


# ---------------------------------------------------------------------------
# 引用模型（reference model）：复刻控制器应有语义，供 model-based 断言对照
# ---------------------------------------------------------------------------


class ReferenceModel:
    """RecordingController 的简单引用模型。

    独立于被测实现，按设计文档（状态机图、不变式 I1–I4）复刻应有语义：

      - 状态仅取 ``idle``/``recording``/``paused``。
      - ``begin_session`` 登记会话参数、复位计数，状态保持 ``idle``，不创建 writer。
      - ``request_toggle`` 在会话运行时循环切换 idle→recording→paused→recording；
        会话未运行时为 no-op 且保持 ``idle``。
      - ``write_frame`` 仅在 ``recording`` 状态计入写入；若该次写入触发注入错误
        （writer 工厂创建失败或 writer.write 抛错），则复位为 ``idle`` 并记录错误。
      - ``close_session`` 无论当前状态都复位 ``idle`` 并结束会话。

    本模型据注入的失败配置预测「期望写入帧数」与「期望状态」，属性测试据此对照
    被测控制器的 ``snapshot()`` 与注入 ``FakeWriter`` 的实际计数。

    参数：
        raise_on_create: 与 ``FakeWriterFactory.raise_on_create`` 对应。
        raise_on_write: 与 ``FakeWriter.raise_on_write`` 对应（1-based 写入序号）。
    """

    VALID_STATES = ("idle", "recording", "paused")

    def __init__(
        self,
        *,
        raise_on_create: bool = False,
        raise_on_write: Optional[int] = None,
    ) -> None:
        self.raise_on_create = raise_on_create
        self.raise_on_write = raise_on_write

        self.state: RecordingState = "idle"
        self.session_active = False
        self.writer_created = False
        # expected_writes：成功写入控制器内部 writer 的帧数（不含抛错那次）。
        self.expected_writes = 0
        # writer_write_attempts：对 writer.write 的尝试次数（含抛错那次），
        # 用于与 FakeWriter.write_calls 对照。
        self.writer_write_attempts = 0
        self.last_error: Optional[str] = None
        self.result_path: Optional[Path] = None

    def begin_session(self, *, fps: float = 30.0, size: Tuple[int, int] = (0, 0)) -> None:
        self.session_active = True
        self.state = "idle"
        self.writer_created = False
        self.expected_writes = 0
        self.writer_write_attempts = 0
        self.last_error = None
        self.result_path = None

    def request_toggle(self) -> RecordingState:
        if not self.session_active:
            return self.state
        if self.state == "idle":
            self.state = "recording"
        elif self.state == "recording":
            self.state = "paused"
        else:  # paused
            self.state = "recording"
        return self.state

    def write_frame(self, frame=None, *, result_path: Optional[Path] = None) -> None:
        """对照控制器 write_frame 的预期效果。

        ``result_path`` 为该会话 path_provider/工厂应产出的确定性落盘路径，仅在首帧
        懒创建成功时被采纳为 ``result_path``。
        """
        if self.state != "recording":
            return

        # 首帧懒创建 writer。
        if not self.writer_created:
            if self.raise_on_create:
                # 创建失败：复位、记录错误，不计写入。
                self.last_error = "create failed"
                self.state = "idle"
                return
            self.writer_created = True
            self.result_path = result_path

        # writer 已就绪：尝试写入。
        self.writer_write_attempts += 1
        if (
            self.raise_on_write is not None
            and self.writer_write_attempts == self.raise_on_write
        ):
            # 写入失败：释放并复位 idle，记录错误。
            self.last_error = "write failed"
            self.writer_created = False
            self.state = "idle"
            return

        self.expected_writes += 1

    def close_session(self) -> Optional[Path]:
        self.session_active = False
        self.state = "idle"
        self.last_error = None
        return self.result_path


# ---------------------------------------------------------------------------
# 冒烟测试：验证夹具与引用模型基本可用（实际属性测试见任务 3.2–3.7）
# ---------------------------------------------------------------------------


def _make_controller(factory: FakeWriterFactory) -> RecordingController:
    return RecordingController(
        writer_factory=factory,
        path_provider=make_fake_path_provider(),
    )


def test_fixture_smoke_records_only_in_recording_state() -> None:
    """冒烟：recording 状态写盘、paused/idle 跳过，FakeWriter 计数符合预期。"""
    factory = make_fake_writer_factory()
    rc = _make_controller(factory)

    rc.begin_session(fps=30.0, size=(64, 48))
    # idle 下 write 不写盘
    rc.write_frame(object())
    assert factory.last_writer is None

    # idle→recording
    assert rc.request_toggle() == "recording"
    rc.write_frame(object())
    rc.write_frame(object())
    assert factory.last_writer is not None
    assert factory.last_writer.write_calls == 2

    # recording→paused：暂停后不再新增写入
    assert rc.request_toggle() == "paused"
    rc.write_frame(object())
    assert factory.last_writer.write_calls == 2

    # 结束复位 + 释放
    rc.close_session()
    assert rc.state == "idle"
    assert factory.last_writer.release_calls == 1


def test_fixture_smoke_factory_create_failure_resets() -> None:
    """冒烟：工厂创建失败时控制器复位 idle 且记录 last_error。"""
    factory = make_fake_writer_factory(raise_on_create=True)
    rc = _make_controller(factory)

    rc.begin_session(fps=30.0, size=(64, 48))
    rc.request_toggle()  # idle→recording
    rc.write_frame(object())  # 触发懒创建 → 创建失败

    snap = rc.snapshot()
    assert snap.state == "idle"
    assert snap.last_error is not None


def test_fixture_smoke_write_failure_resets() -> None:
    """冒烟：第 N 次 write 抛错时控制器释放并复位 idle。"""
    factory = make_fake_writer_factory(raise_on_write=2)
    rc = _make_controller(factory)

    rc.begin_session(fps=30.0, size=(64, 48))
    rc.request_toggle()  # idle→recording
    rc.write_frame(object())  # 第 1 次 write 成功
    rc.write_frame(object())  # 第 2 次 write 抛错 → 复位

    snap = rc.snapshot()
    assert snap.state == "idle"
    assert snap.last_error is not None
    assert factory.last_writer is not None
    assert factory.last_writer.release_calls == 1


def test_first_frame_size_overrides_reported_capture_size_before_writer_creation() -> None:
    """首帧真实尺寸应覆盖不可靠的 CAP_PROP 尺寸，但 writer 创建后不得再变更。"""
    create_args: list[tuple[Path, float, Tuple[int, int]]] = []
    writer = FakeWriter()

    def _factory(path: Path, fps: float, size: Tuple[int, int]):
        create_args.append((Path(path), fps, size))
        return writer, Path(path), "fake"

    rc = RecordingController(
        writer_factory=_factory,
        path_provider=make_fake_path_provider(),
    )

    assert rc.update_session_size(size=(720, 1280)) is False
    rc.begin_session(fps=30.0, size=(1280, 720))
    assert rc.request_toggle() == "recording"
    assert rc.update_session_size(size=(720, 1280)) is True

    rc.write_frame(object())

    assert create_args == [(DEFAULT_FAKE_PATH, 30.0, (720, 1280))]
    assert rc.update_session_size(size=(640, 480)) is False


def test_stop_recording_clears_last_error_for_same_session_retry() -> None:
    factory = make_fake_writer_factory(raise_on_create=True)
    rc = _make_controller(factory)
    rc.begin_session(fps=30.0, size=(64, 48))
    assert rc.request_toggle() == "recording"
    rc.write_frame(object())
    assert rc.snapshot().last_error is not None

    rc.stop_recording()

    assert rc.snapshot().last_error is None
    assert rc.request_toggle() == "recording"


def test_close_session_clears_last_error_before_next_mode() -> None:
    factory = make_fake_writer_factory(raise_on_create=True)
    rc = _make_controller(factory)
    rc.begin_session(fps=30.0, size=(64, 48))
    assert rc.request_toggle() == "recording"
    rc.write_frame(object())
    assert rc.snapshot().last_error is not None

    rc.close_session()

    snap = rc.snapshot()
    assert snap.state == "idle"
    assert snap.last_error is None


def test_reference_model_matches_controller_on_simple_sequence() -> None:
    """冒烟：引用模型与控制器在一段简单动作序列上的状态/写入计数一致。"""
    factory = make_fake_writer_factory()
    rc = _make_controller(factory)
    model = ReferenceModel()

    rc.begin_session(fps=30.0, size=(10, 10))
    model.begin_session(fps=30.0, size=(10, 10))

    # 动作序列：toggle, write, write, toggle(pause), write, toggle(resume), write
    rc.request_toggle(); model.request_toggle()
    rc.write_frame(object()); model.write_frame(result_path=DEFAULT_FAKE_PATH)
    rc.write_frame(object()); model.write_frame(result_path=DEFAULT_FAKE_PATH)
    rc.request_toggle(); model.request_toggle()
    rc.write_frame(object()); model.write_frame(result_path=DEFAULT_FAKE_PATH)
    rc.request_toggle(); model.request_toggle()
    rc.write_frame(object()); model.write_frame(result_path=DEFAULT_FAKE_PATH)

    assert rc.state == model.state
    assert rc.snapshot().frames_written == model.expected_writes
    assert factory.last_writer.write_calls == model.writer_write_attempts


# ---------------------------------------------------------------------------
# Property 1: 状态转换合法性
# ---------------------------------------------------------------------------
# Feature: ui-layout-redesign, Property 1: 状态转换合法性
#
# For any 由 begin_session、request_toggle、close_session 组成的动作序列，
# RecordingController 的状态在任意时刻都只取 idle/recording/paused 之一，且每一步
# 转换都属于合法转换集合 {idle→recording, recording→paused, paused→recording,
# recording→idle, paused→idle}；当会话未运行（未 begin_session 或已 close_session）
# 时，request_toggle 为 no-op 且状态保持 idle。
#
# Validates: Requirements 4.1, 5.1

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

# 合法转换集合：设计文档定义的真实状态变更，加上恒等转换（不改变状态的动作，
# 例如 recording 下 write_frame 写盘、会话未运行时 toggle no-op）。
_LEGAL_TRANSITIONS = {
    ("idle", "recording"),
    ("recording", "paused"),
    ("paused", "recording"),
    ("recording", "idle"),
    ("paused", "idle"),
    # 恒等转换：未改变状态的动作（write 写盘 / no-op toggle / 重复 close）均合法。
    ("idle", "idle"),
    ("recording", "recording"),
    ("paused", "paused"),
}

# 动作字母表：begin / toggle / write / close。
_ACTIONS = st.sampled_from(["begin", "toggle", "write", "close"])


@settings(max_examples=200)
@given(actions=st.lists(_ACTIONS, max_size=40))
def test_property1_state_transitions_are_always_legal(actions) -> None:
    """Property 1: 任意动作序列下状态恒合法、每步转换合法、会话未运行时 toggle 为 no-op。"""
    factory = make_fake_writer_factory()
    rc = _make_controller(factory)
    model = ReferenceModel()

    session_active = False  # 引用侧追踪会话是否运行，用于断言 no-op 语义。

    for action in actions:
        prev_state = rc.state

        if action == "begin":
            rc.begin_session(fps=30.0, size=(16, 16))
            model.begin_session(fps=30.0, size=(16, 16))
            session_active = True
        elif action == "toggle":
            returned = rc.request_toggle()
            model.request_toggle()
            # 返回值必须等于切换后的实际状态。
            assert returned == rc.state
            # 会话未运行时 toggle 必须为 no-op：状态保持 idle 不变。
            if not session_active:
                assert prev_state == "idle"
                assert rc.state == "idle"
        elif action == "write":
            rc.write_frame(object())
            model.write_frame(result_path=DEFAULT_FAKE_PATH)
        else:  # close
            rc.close_session()
            model.close_session()
            session_active = False
            # 结束后状态必复位 idle。
            assert rc.state == "idle"

        cur_state = rc.state

        # 不变式：状态永远落在合法状态集合内。
        assert cur_state in ReferenceModel.VALID_STATES

        # 不变式：每一步转换都属于合法转换集合。
        assert (prev_state, cur_state) in _LEGAL_TRANSITIONS, (
            f"非法转换 {prev_state}->{cur_state} 由动作 {action!r} 触发"
        )

        # 与引用模型对照：状态一致。
        assert cur_state == model.state


# ---------------------------------------------------------------------------
# Property 2: 仅录制中写盘
# ---------------------------------------------------------------------------
# Feature: ui-layout-redesign, Property 2: 仅录制中写盘
#
# For any 动作序列与穿插其间的帧序列，注入 writer 收到的写入帧数恰好等于「调用
# write_frame 时控制器处于 recording 状态」的次数；处于 paused 或 idle 时调用
# write_frame 不产生任何写入，且不丢弃此前已写入的帧。
#
# Validates: Requirements 5.3, 5.5


def _total_writer_calls(factory: FakeWriterFactory) -> int:
    """工厂迄今创建的所有 writer 累计 write 调用次数（跨会话求和）。"""
    return sum(w.write_calls for w in factory.writers)


def _total_writer_frames(factory: FakeWriterFactory) -> int:
    """工厂迄今创建的所有 writer 成功写入的帧总数（跨会话求和）。"""
    return sum(len(w.frames) for w in factory.writers)


@settings(max_examples=200)
@given(actions=st.lists(_ACTIONS, max_size=60))
def test_property2_writes_only_while_recording(actions) -> None:
    """Property 2: 注入 writer 收到的写入帧数 == recording 态下的 write_frame 次数；
    paused/idle 下 write 不写盘且不丢弃既有帧。

    动作序列可含多次 ``begin``（多会话），每个会话至多懒创建一个 writer，故按
    工厂创建的所有 writer 累计求和对照「全程 recording 态写盘次数」；并按当前会话
    与 ReferenceModel.expected_writes 逐步交叉校验。
    """
    # 默认工厂：不注入创建/写入失败，使「写盘当且仅当 recording」语义可被干净验证。
    factory = make_fake_writer_factory()
    rc = _make_controller(factory)
    model = ReferenceModel()

    # 全程：调用 write_frame 时控制器恰处于 recording 状态的次数（期望总写盘数）。
    total_writes_while_recording = 0
    # 当前会话内的 recording 态写盘次数，begin 时复位，用于对照模型（模型 begin 也复位）。
    session_writes_while_recording = 0
    # 监视器：跨会话累计写盘数永不减少（不丢弃既有帧）。
    prev_total_calls = 0

    for action in actions:
        if action == "begin":
            rc.begin_session(fps=30.0, size=(16, 16))
            model.begin_session(fps=30.0, size=(16, 16))
            session_writes_while_recording = 0
        elif action == "toggle":
            rc.request_toggle()
            model.request_toggle()
        elif action == "write":
            state_before = rc.state
            before = _total_writer_calls(factory)
            rc.write_frame(object())
            model.write_frame(result_path=DEFAULT_FAKE_PATH)
            after = _total_writer_calls(factory)

            if state_before == "recording":
                total_writes_while_recording += 1
                session_writes_while_recording += 1
                # recording 态：本次 write 必恰好产生一次写盘。
                assert after == before + 1
            else:
                # paused / idle 态：本次 write 不得产生任何写盘。
                assert after == before
        else:  # close
            rc.close_session()
            model.close_session()

        # 不丢弃既有帧：跨会话累计写盘数单调不减。
        cur_total = _total_writer_calls(factory)
        assert cur_total >= prev_total_calls
        prev_total_calls = cur_total

        # 与引用模型逐步交叉校验：当前会话的期望写盘数一致。
        assert model.expected_writes == session_writes_while_recording

    # 终态对照：所有注入 writer 累计写入帧数恰好等于全程 recording 态写盘次数。
    assert _total_writer_calls(factory) == total_writes_while_recording
    # 无失败注入时，成功写入的 frames 总数也应等于写盘次数（无丢弃）。
    assert _total_writer_frames(factory) == total_writes_while_recording


# ---------------------------------------------------------------------------
# Property 3: 单会话单文件
# ---------------------------------------------------------------------------
# Feature: ui-layout-redesign, Property 3: 单会话单文件
#
# For any 单次会话（一次 begin_session 到对应 close_session）内的任意 toggle/写帧
# 交织序列，本会话产生的 result_path 至多为一个确定路径；paused→recording 恢复录制
# 时复用同一 writer 与同一路径，不新建文件、不覆盖既有内容。
#
# Validates: Requirements 5.7

# 会话内动作字母表：仅 toggle / write（不含 begin/close，以约束在「单会话」边界内）。
_SESSION_ACTIONS = st.sampled_from(["toggle", "write"])


@settings(max_examples=200)
@given(actions=st.lists(_SESSION_ACTIONS, max_size=60))
def test_property3_single_session_single_file(actions) -> None:
    """Property 3: 单会话内 result_path 至多一个；paused→recording 复用同一 writer
    与同一路径，不新建文件、不覆盖既有内容。

    断言要点：
      - 会话内工厂创建次数 ``create_calls <= 1``（至多一个文件/一个 writer）。
      - 一旦 ``result_path`` 被设定，其后保持稳定不变（不切换为不同路径）。
      - 工厂至多创建一个 ``FakeWriter`` 实例。
      - 每次 ``paused→recording`` 恢复后继续 ``write`` 复用的仍是同一 writer 实例
        （不触发新的工厂创建、不覆盖既有内容）。
    """
    factory = make_fake_writer_factory()
    rc = _make_controller(factory)
    model = ReferenceModel()

    # 单会话开始。
    rc.begin_session(fps=30.0, size=(16, 16))
    model.begin_session(fps=30.0, size=(16, 16))

    seen_result_path: Optional[Path] = None  # 一旦设定即应保持稳定。
    first_writer: Optional[FakeWriter] = None  # 会话内首个（也应是唯一）writer 实例。

    for action in actions:
        prev_state = rc.state
        if action == "toggle":
            rc.request_toggle()
            model.request_toggle()
        else:  # write
            rc.write_frame(object())
            model.write_frame(result_path=DEFAULT_FAKE_PATH)

        # 不变式：会话内工厂至多创建一个 writer（单文件）。
        assert factory.create_calls <= 1, (
            f"单会话内工厂创建了 {factory.create_calls} 次，违反单文件约束"
        )
        assert len(factory.writers) <= 1

        # 一旦工厂创建了 writer，记录其实例，后续应始终是同一个实例。
        if factory.last_writer is not None:
            if first_writer is None:
                first_writer = factory.last_writer
            else:
                assert factory.last_writer is first_writer, (
                    "paused→recording 恢复后未复用同一 writer 实例（疑似新建文件）"
                )

        # result_path 稳定性：一旦非空即固定，不得变更为不同路径。
        snap = rc.snapshot()
        if snap.result_path is not None:
            if seen_result_path is None:
                seen_result_path = snap.result_path
            else:
                assert snap.result_path == seen_result_path, (
                    f"result_path 发生变更：{seen_result_path} -> {snap.result_path}"
                )

        # paused→recording 恢复录制后立即写帧：必须复用同一 writer（create_calls 不增）。
        if prev_state == "paused" and rc.state == "recording" and first_writer is not None:
            calls_before = factory.create_calls
            rc.write_frame(object())
            model.write_frame(result_path=DEFAULT_FAKE_PATH)
            assert factory.create_calls == calls_before, (
                "paused→recording 恢复后写帧触发了新的 writer 创建"
            )
            assert factory.last_writer is first_writer

    # 会话结束：复用的同一 writer 被释放，返回路径与会话内观察到的稳定路径一致。
    returned_path = rc.close_session()
    model.close_session()
    if seen_result_path is not None:
        assert returned_path == seen_result_path
        assert first_writer is not None
        assert first_writer.release_calls == 1


# ---------------------------------------------------------------------------
# Property 4: 结束必复位
# ---------------------------------------------------------------------------
# Feature: ui-layout-redesign, Property 4: 结束必复位
#
# For any 会话结束前所处的状态（idle、recording 或 paused），调用 close_session 后
# 控制器状态恒为 idle，且本会话的 writer 已被释放（release 恰被调用且其后 writer
# 句柄为 None）。
#
# Validates: Requirements 4.4, 5.8


@settings(max_examples=200)
@given(actions=st.lists(_SESSION_ACTIONS, max_size=60))
def test_property4_close_session_always_resets(actions) -> None:
    """Property 4: 任意会话内动作序列把控制器驱入随机状态（idle/recording/paused）后，
    close_session 必复位 idle，且若本会话创建过 writer 则其 release 恰被调用一次。

    断言要点：
      - close_session 后 ``rc.state == "idle"``（需求 4.4/5.8）。
      - 若会话内曾懒创建 writer（``factory.last_writer is not None``），则该 writer
        的 ``release_calls == 1``（释放恰一次）。
      - writer 句柄释放后不再被持有：通过「再开一个全新会话并正常录制/释放」验证控制器
        未残留旧 writer 句柄（idle ⇒ writer is None 不变式，需求 5.8）。
    """
    factory = make_fake_writer_factory()
    rc = _make_controller(factory)
    model = ReferenceModel()

    # 单会话开始，按随机动作序列驱入某个状态（idle/recording/paused）。
    rc.begin_session(fps=30.0, size=(16, 16))
    model.begin_session(fps=30.0, size=(16, 16))

    for action in actions:
        if action == "toggle":
            rc.request_toggle()
            model.request_toggle()
        else:  # write
            rc.write_frame(object())
            model.write_frame(result_path=DEFAULT_FAKE_PATH)

    # 记录结束前是否已创建 writer（用于断言释放次数）。
    writer_before_close = factory.last_writer

    # 结束会话：无论结束前状态如何，都必复位 idle。
    rc.close_session()
    model.close_session()

    assert rc.state == "idle", "close_session 后状态未复位为 idle"
    assert rc.snapshot().state == "idle"
    assert model.state == "idle"

    # 若本会话创建过 writer，则其 release 恰被调用一次（释放且仅释放一次）。
    if writer_before_close is not None:
        assert writer_before_close.release_calls == 1, (
            f"会话 writer 的 release 调用次数为 {writer_before_close.release_calls}，应为 1"
        )

    # writer 句柄已释放、不再被持有：开启全新会话并正常录制一帧后释放，
    # 必定创建一个全新 writer（说明旧句柄未残留），且新 writer 同样被释放恰一次。
    prev_create_calls = factory.create_calls
    rc.begin_session(fps=30.0, size=(16, 16))
    rc.request_toggle()  # idle→recording
    rc.write_frame(object())  # 触发新会话的懒创建
    assert factory.create_calls == prev_create_calls + 1, (
        "新会话未触发全新 writer 创建，疑似残留旧 writer 句柄"
    )
    new_writer = factory.last_writer
    rc.close_session()
    assert rc.state == "idle"
    assert new_writer is not None
    assert new_writer.release_calls == 1


# ---------------------------------------------------------------------------
# Property 5: writer 与路径生命周期
# ---------------------------------------------------------------------------
# Feature: ui-layout-redesign, Property 5: writer 与路径生命周期
#
# For any 动作序列，当状态为 idle 时控制器持有的 writer 句柄必为 None（空闲态绝不
# 持有打开的 writer）；当状态为 recording 或 paused 且本会话已至少写入过一帧时，
# snapshot().result_path 非空。
#
# Validates: Requirements 5.9, 5.10


@settings(max_examples=200)
@given(actions=st.lists(_ACTIONS, max_size=60))
def test_property5_writer_and_path_lifecycle(actions) -> None:
    """Property 5: idle ⇒ writer 句柄为 None；recording/paused 且 frames_written>=1
    ⇒ result_path 非空。

    断言要点（每步动作后检查不变式）：
      - 空闲态绝不持有打开的 writer：``rc.state == "idle"`` 时 ``rc._writer is None``
        （需求 5.9/5.10；直接检查私有句柄，测试内允许）。
      - 录制/暂停且本会话已写入至少一帧时，落盘路径已就绪：``rc.state in
        ("recording", "paused")`` 且 ``snapshot().frames_written >= 1`` 时
        ``snapshot().result_path is not None``。
    """
    # 默认工厂：不注入创建/写入失败，使写盘必然成功、result_path 必被设定，
    # 从而「写过帧 ⇒ 路径非空」语义可被干净验证。
    factory = make_fake_writer_factory()
    rc = _make_controller(factory)

    for action in actions:
        if action == "begin":
            rc.begin_session(fps=30.0, size=(16, 16))
        elif action == "toggle":
            rc.request_toggle()
        elif action == "write":
            rc.write_frame(object())
        else:  # close
            rc.close_session()

        snap = rc.snapshot()

        # 不变式一：空闲态绝不持有打开的 writer 句柄。
        if rc.state == "idle":
            assert rc._writer is None, (
                "idle 态仍持有 writer 句柄，违反 writer 生命周期不变式"
            )

        # 不变式二：录制/暂停且已写入至少一帧 ⇒ result_path 非空。
        if rc.state in ("recording", "paused") and snap.frames_written >= 1:
            assert snap.result_path is not None, (
                f"状态 {rc.state} 且已写入 {snap.frames_written} 帧，"
                "但 result_path 仍为 None"
            )


# ---------------------------------------------------------------------------
# Property 6: 错误条件复位
# ---------------------------------------------------------------------------
# Feature: ui-layout-redesign, Property 6: 错误条件复位
#
# For any 使 writer 工厂在创建时抛错、或使 writer.write 在写入时抛错的注入场景，
# 相应的 write_frame 调用返回后控制器状态恒为 idle，writer 句柄被释放并置为 None，
# 且 snapshot().last_error 为非空错误原因。
#
# Validates: Requirements 5.11

# 失败场景策略：二选一注入 —— 工厂创建时抛错，或 writer 第 N 次 write 抛错。
_FAILURE_CONFIG = st.one_of(
    st.fixed_dictionaries({"kind": st.just("create")}),
    st.fixed_dictionaries(
        {"kind": st.just("write"), "n": st.integers(min_value=1, max_value=5)}
    ),
)


@settings(max_examples=200)
@given(
    failure=_FAILURE_CONFIG,
    actions=st.lists(_SESSION_ACTIONS, max_size=60),
)
def test_property6_error_condition_resets(failure, actions) -> None:
    """Property 6: 任意「创建抛错 / 第 N 次 write 抛错」注入场景下，触发失败的那次
    write_frame 返回后控制器必复位 idle、writer 句柄置为 None、last_error 非空，
    且异常不外泄（write_frame 正常返回）。

    断言要点（每次触发失败后检查）：
      - write_frame 调用未抛出异常（异常被控制器收敛，不外泄到 worker 主循环）。
      - 触发失败后 ``rc.state == "idle"``（错误条件复位，需求 5.11）。
      - ``rc._writer is None``（半开资源被释放、句柄不残留）。
      - ``snapshot().last_error`` 为非空字符串（记录失败原因）。
    """
    if failure["kind"] == "create":
        factory = make_fake_writer_factory(raise_on_create=True)
    else:
        factory = make_fake_writer_factory(raise_on_write=failure["n"])

    rc = _make_controller(factory)
    rc.begin_session(fps=30.0, size=(16, 16))

    triggered = False

    def _do_write() -> None:
        """执行一次 write_frame，检测是否触发注入失败并断言复位不变式。

        失败被触发的判据：调用前处于 recording 状态，调用后落到 idle —— 在 recording
        态下 write_frame 唯一会复位 idle 的路径就是错误处理路径（需求 5.11）。
        """
        nonlocal triggered
        state_before = rc.state
        # 异常不得外泄：write_frame 必须正常返回（控制器自行收敛错误）。
        try:
            rc.write_frame(object())
        except Exception as exc:  # noqa: BLE001 - 任何外泄都视为属性违反
            raise AssertionError(
                f"write_frame 让注入异常外泄到调用方：{exc!r}"
            )

        if state_before == "recording" and rc.state == "idle":
            # 触发了错误条件复位：断言完整复位语义。
            triggered = True
            snap = rc.snapshot()
            assert snap.state == "idle", "错误复位后状态应为 idle"
            assert rc._writer is None, "错误复位后 writer 句柄应被释放并置为 None"
            assert snap.last_error is not None, "错误复位后 last_error 不应为空"
            assert isinstance(snap.last_error, str) and snap.last_error != "", (
                "last_error 应为非空错误原因字符串"
            )

    # 先按随机动作序列驱动（可能在中途触发失败）。
    for action in actions:
        if action == "toggle":
            rc.request_toggle()
        else:  # write
            _do_write()

    # 确保该注入场景至少被实际触发一次：强制进入 recording 并写满足够的帧，
    # 使断言主体（错误复位不变式）必然被执行至少一次。
    forced = 0
    while not triggered and forced < 12:
        if rc.state != "recording":
            rc.request_toggle()
            # toggle 仅在会话运行时改变状态；此处会话恒为运行中，故必能进入 recording。
            if rc.state != "recording":
                rc.request_toggle()
        _do_write()
        forced += 1

    assert triggered, "注入的失败场景未被触发，无法验证错误条件复位"

    # 触发复位后调用 close_session 仍应安全且保持 idle。
    rc.close_session()
    assert rc.state == "idle"
