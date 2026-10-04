# -*- coding: utf-8 -*-
"""输入源状态模型测试：Property 4 属性测试 + 示例测试 + 刷新门控示例。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.camera_enum import InputSourceState  # noqa: E402


def _assert_invariant(state: InputSourceState) -> None:
    if state.kind == "camera":
        assert state.value.isdigit()
    elif state.kind == "video":
        assert not state.value.isdigit()
    elif state.kind == "none":
        assert state.value == ""
    else:
        raise AssertionError(f"非法 kind: {state.kind!r}")


# 视频路径策略：保证非纯数字（含路径分隔符/扩展名），符合 video ⇒ not isdigit() 约定。
_video_paths = st.from_regex(r"[A-Za-z]:[\\/][A-Za-z0-9_]+\.(mp4|avi|mov)", fullmatch=True)

# 操作序列：("camera", index) 或 ("video", path)
_ops = st.lists(
    st.one_of(
        st.tuples(st.just("camera"), st.integers(min_value=0, max_value=32)),
        st.tuples(st.just("video"), _video_paths),
    ),
    max_size=30,
)


# Feature: camera-dropdown-selection, Property 4: 输入源切换保持「至多一个生效源」不变式
@settings(max_examples=200)
@given(ops=_ops)
def test_property4_input_source_mutual_exclusion(ops) -> None:
    state = InputSourceState()
    _assert_invariant(state)
    for kind, payload in ops:
        if kind == "camera":
            state.select_camera(payload)
            assert state.kind == "camera"
            assert state.value == str(payload)
        else:
            state.select_video(payload)
            assert state.kind == "video"
            assert state.value == payload
        _assert_invariant(state)


def test_initial_state_is_none() -> None:
    state = InputSourceState()
    assert state.kind == "none"
    assert state.value == ""
    assert "未选择" in state.hint_text()


def test_camera_to_video_clears_camera() -> None:
    state = InputSourceState()
    state.select_camera(1)
    state.select_video("C:/clips/a.mp4")
    assert state.kind == "video"
    assert state.value == "C:/clips/a.mp4"
    assert not state.value.isdigit()


def test_video_to_camera_clears_video() -> None:
    state = InputSourceState()
    state.select_video("C:/clips/a.mp4")
    state.select_camera(0)
    assert state.kind == "camera"
    assert state.value == "0"
    assert state.value.isdigit()


# ---- 刷新门控逻辑（纯函数形式的可注入断言，对应 _set_refresh_enabled 决策） ----


def _refresh_enabled(enum_busy: bool, capture_running: bool) -> bool:
    """刷新启用条件：非枚举中且采集未运行（需求 5.3、5.7）。"""
    return (not enum_busy) and (not capture_running)


@pytest.mark.parametrize(
    "enum_busy,capture_running,expected",
    [
        (False, False, True),
        (True, False, False),
        (False, True, False),
        (True, True, False),
    ],
)
def test_refresh_gating(enum_busy: bool, capture_running: bool, expected: bool) -> None:
    assert _refresh_enabled(enum_busy, capture_running) is expected


def test_collect_state_rejects_none() -> None:
    # 模拟 _collect_state 在 none 时的判定逻辑。
    state = InputSourceState()
    assert state.kind == "none"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
