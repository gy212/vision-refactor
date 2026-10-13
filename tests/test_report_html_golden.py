"""图文报告模板化的差分：P8 前用旧 ``render_html_report`` 录下的基准，新渲染器逐字节一致（证据帧按摘要比较）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.reporting.html import render_html_report
from tests.report_html_cases import FIXTURE_DIR, build_cases, normalize

_CASES = sorted(p.stem for p in FIXTURE_DIR.glob("*.html"))


@pytest.fixture(scope="module")
def cases(tmp_path_factory) -> dict:
    return build_cases(tmp_path_factory.mktemp("history"))


def test_fixture_set_matches_cases(cases):
    assert sorted(cases) == _CASES


@pytest.mark.parametrize("name", _CASES)
def test_report_html_is_byte_identical_to_pre_template_baseline(cases, name: str):
    record, kwargs = cases[name]
    expected = (FIXTURE_DIR / f"{name}.html").read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert normalize(render_html_report(record, **kwargs)) == expected
