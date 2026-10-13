"""结果页拆分的差分：拆分前旧面板录下的控件树快照摘要，与拆分后的面板逐步一致（文字、配色、证据帧尺寸、相位条、列表）。"""
from __future__ import annotations

import json
import tkinter as tk

import pytest

from apps.tk.views.practice_result import FeedbackResultPanel
from tests.result_page_snapshot import FIXTURE, build_records, digest, drive


@pytest.fixture
def tk_root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


def test_result_page_matches_pre_split_snapshots(tk_root, tmp_path):
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    records = build_records(tmp_path)
    assert sorted(records) == sorted(expected)
    for name, record in records.items():
        panel = FeedbackResultPanel(tk_root, history_root=tmp_path, on_review_toggle=lambda *a: None)
        panel.pack()
        steps = drive(panel, record, select_phase=panel.select_phase, select_status=panel.select_status,
                      select_check=panel.select_check)
        panel.destroy()
        assert [step for step, _ in steps] == [step for step, _ in expected[name]], name
        for (step, snap), (_, want) in zip(steps, expected[name]):
            assert digest(snap) == want, f"{name} 在步骤 {step} 与拆分前不一致"
