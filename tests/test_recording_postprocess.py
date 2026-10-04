from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import apps.recording_postprocess as recording_postprocess
from apps.recording_postprocess import (
    DualRecordingJob,
    DualRecordingPostProcessor,
    PostprocessError,
    PostprocessUpdate,
    _SCORE_SQUEEZE,
    default_template_paths,
    merge_view_weighted_actions,
    validate_auto_compare_templates,
    validate_template_pair,
)


def _d(name: str, score: float, start: int = 0, end: int = 10) -> dict:
    return {"name": name, "view": "x", "score": score, "start": start, "end": end}


def test_merge_view_weighted_actions_paired_uses_registered_weights() -> None:
    # 两手攀足：侧 0.75 + 正 0.25（实测正面看不清前俯深度）。
    out = merge_view_weighted_actions([_d("两手攀足_正面", 0.708), _d("两手攀足_侧面", 0.855)])
    assert len(out) == 1
    assert out[0]["name"] == "两手攀足"
    assert out[0]["score"] == pytest.approx(0.75 * 0.855 + 0.25 * 0.708)
    # 攒拳怒目：50/50 去重双计数。
    out = merge_view_weighted_actions([_d("攒拳怒目_正面", 0.812), _d("攒拳怒目_侧面", 0.807)])
    assert len(out) == 1 and out[0]["score"] == pytest.approx(0.8095)


def test_merge_view_weighted_actions_is_noop_for_sanda() -> None:
    # 散打模板 stem 无 _正面/_侧面 后缀、不在权重表内 → 原样透传、保序（零回归）。
    sanda = [_d("直拳", 0.8), _d("高鞭腿", 0.7), _d("front", 0.86), _d("side", 0.86)]
    assert merge_view_weighted_actions(sanda) == sanda


def test_merge_view_weighted_actions_keeps_unpaired_view() -> None:
    # 缺侧面则不合并，正面原样保留（不误合、不丢分）。
    out = merge_view_weighted_actions([_d("两手攀足_正面", 0.708)])
    assert len(out) == 1 and out[0]["name"] == "两手攀足_正面"


def test_score_squeeze_backhand_keeps_range_and_reproducible() -> None:
    # 背后七颠：DTW 不可靠 → 压到 [0.75,0.90]（人人过、弱区分、可复现）。
    lo, hi = _SCORE_SQUEEZE["背后七颠"]

    def sq(d: float) -> float:
        return lo + (hi - lo) * min(max(d, 0.0), 1.0)

    # 任意 DTW 输入都落在窄区间内（人人及格偏上）。
    for dtw in (0.0, 0.3, 0.7, 0.936, 0.942, 1.0):
        assert lo - 1e-9 <= sq(dtw) <= hi + 1e-9
    # 单调保留微弱区分度：认真 > 敷衍 > 没做。
    assert sq(0.942) > sq(0.70) > sq(0.30)
    # 可复现：同输入同输出（无随机）。
    assert sq(0.936) == sq(0.936)



def _job(
    root: Path,
    segment_id: str,
    *,
    skeleton: bool = False,
    auto_compare: bool = True,
    front_frames: int = 12,
    side_frames: int = 12,
    suffix: str = ".mp4",
) -> DualRecordingJob:
    segment_dir = root / segment_id
    segment_dir.mkdir(parents=True)
    front = segment_dir / f"front{suffix}"
    side = segment_dir / f"side{suffix}"
    front.write_bytes(b"front-video")
    side.write_bytes(b"side-video")
    # 用临时 heavy 正/侧模板，避免依赖仓库 templates/ 是否齐全（仅一侧存在会走真实单流比对）。
    front_template = segment_dir / "front_template.npz"
    side_template = segment_dir / "side_template.npz"
    _write_template(front_template)
    _write_template(side_template)
    return DualRecordingJob(
        segment_id=segment_id,
        segment_dir=segment_dir,
        front_source=front,
        side_source=side,
        front_frames=front_frames,
        side_frames=side_frames,
        front_template=front_template,
        side_template=side_template,
        record_skeleton=skeleton,
        auto_compare=auto_compare,
    )


def _result(score: float = 0.86):
    """单模板比对结果（CompareResult 形状）：score + 匹配窗口。"""
    return SimpleNamespace(
        score=score,
        start_frame=0,
        end_frame=10,
    )


def _pool_result(template_paths, score: float = 0.86):
    """池级比对结果：与 template_paths 同序的 CompareResult 列表。"""
    return [_result(score) for _ in template_paths]


def _wait(event: threading.Event) -> None:
    assert event.wait(3.0), "postprocess worker did not reach a terminal state"


def test_fixed_templates_default_paths_and_auto_compare_contract(tmp_path: Path) -> None:
    """默认路径名固定；一条龙仅当两侧都缺才 template_missing。"""
    front, side = default_template_paths()
    assert front.name == "standard_front_heavy.npz"
    assert side.name == "standard_side_heavy.npz"
    # 两侧都缺 → 报错
    with pytest.raises(PostprocessError) as both_missing:
        validate_auto_compare_templates(
            tmp_path / "no_front.npz", tmp_path / "no_side.npz"
        )
    assert both_missing.value.code == "template_missing"
    # 仅一侧可用 → 通过
    only = tmp_path / "only_side.npz"
    _write_template(only)
    f, s = validate_auto_compare_templates(tmp_path / "no_front.npz", only)
    assert f is None
    assert s == only


def test_fifo_single_consumer_and_atomic_completed_results(tmp_path: Path) -> None:
    updates: list[PostprocessUpdate] = []
    terminal = threading.Event()
    order: list[str] = []
    concurrency = 0
    max_concurrency = 0
    lock = threading.Lock()

    def compare(template_paths, video, **kwargs):
        nonlocal concurrency, max_concurrency
        assert kwargs == {
            "pose_variant": "heavy",
            "workers": 1,
            "stop_evt": kwargs["stop_evt"],
        }
        with lock:
            concurrency += 1
            max_concurrency = max(max_concurrency, concurrency)
        order.append(Path(video).parent.name)
        time.sleep(0.03)
        with lock:
            concurrency -= 1
        return _pool_result(template_paths)

    def on_update(update: PostprocessUpdate) -> None:
        updates.append(update)
        if sum(item.status == "completed" for item in updates) == 2:
            terminal.set()

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=lambda path, _stop: path,
        compare=compare,
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    first = _job(tmp_path, "record_first")
    second = _job(tmp_path, "record_second")
    try:
        assert processor.submit(first)
        assert processor.submit(second)
        assert not processor.submit(first)
        _wait(terminal)
    finally:
        processor.close(1.0)

    # 每个 job 正/侧各一池 → 各调一次池比对；FIFO 保证 first 两次全部先于 second。
    assert order == ["record_first", "record_first", "record_second", "record_second"]
    assert max_concurrency == 1
    assert [item.segment_id for item in updates if item.status == "queued"] == [
        "record_first",
        "record_second",
    ]
    for segment_id in ("record_first", "record_second"):
        segment_updates = [item for item in updates if item.segment_id == segment_id]
        assert segment_updates[0].status == "queued"
    for job in (first, second):
        result_path = job.segment_dir / "result.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert set(payload) == {
            "schema_version",
            "status",
            "segment_id",
            "created_at",
            "completed_at",
            "record_skeleton",
            "auto_compare",
            "front_video_path",
            "side_video_path",
            "front_template_path",
            "side_template_path",
            "warnings",
            "result",
            "error",
            "exam",
        }
        assert payload["status"] == "completed"
        assert payload["segment_id"] == job.segment_id
        assert payload["created_at"] == job.created_at
        assert payload["completed_at"]
        assert payload["record_skeleton"] is False
        assert payload["auto_compare"] is True
        assert payload["front_video_path"] == str(job.front_source)
        assert payload["side_video_path"] == str(job.side_source)
        assert payload["front_template_path"] == str(job.front_template)
        assert payload["side_template_path"] == str(job.side_template)
        assert payload["warnings"] == []
        assert set(payload["result"]) == {
            "front_score",
            "side_score",
            "combined_score",
            "combined_percent",
            "front_matches",
            "side_matches",
            "front_segment",
            "side_segment",
            "action_scores",
        }
        # 正/侧各 1 模板，各出 0.86 → 加权综合仍 0.86 → 86
        assert payload["result"]["front_score"] == pytest.approx(0.86)
        assert payload["result"]["side_score"] == pytest.approx(0.86)
        assert payload["result"]["combined_percent"] == 86
        assert payload["result"]["front_segment"] == {"start": 0, "end": 10}
        assert len(payload["result"]["action_scores"]) == 2
        assert payload["error"] is None
        assert list(job.segment_dir.glob(".result.*.tmp")) == []


def test_submit_reserves_resolved_segment_directory_for_processor_lifetime(
    tmp_path: Path,
) -> None:
    completed = threading.Event()
    compare_ids: list[str] = []
    updates: list[PostprocessUpdate] = []

    first = _job(tmp_path, "record_directory_owner")
    alias_dir = first.segment_dir / ".." / first.segment_dir.name
    second = DualRecordingJob(
        segment_id="record_directory_intruder",
        segment_dir=alias_dir,
        front_source=alias_dir / first.front_source.name,
        side_source=alias_dir / first.side_source.name,
        front_frames=first.front_frames,
        side_frames=first.side_frames,
        front_template=first.front_template,
        side_template=first.side_template,
    )

    def compare(template_paths, video, **_kwargs):
        compare_ids.append(Path(video).parent.name)
        return _pool_result(template_paths)

    def on_update(update: PostprocessUpdate) -> None:
        updates.append(update)
        if update.segment_id == first.segment_id and update.status == "completed":
            completed.set()

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=lambda path, _stop: path,
        compare=compare,
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    try:
        assert processor.submit(first)
        _wait(completed)
        result_path = first.segment_dir / "result.json"
        original_result = result_path.read_text(encoding="utf-8")

        assert alias_dir.resolve() == first.segment_dir.resolve()
        assert not processor.submit(second)
        processor._queue.join()
    finally:
        processor.close(1.0)

    # 正/侧各一模板 → 比对两次，均属 first
    assert compare_ids == [first.segment_id, first.segment_id]
    assert {update.segment_id for update in updates} == {first.segment_id}
    assert result_path.read_text(encoding="utf-8") == original_result
    assert json.loads(original_result)["segment_id"] == first.segment_id


@pytest.mark.skipif(os.name != "nt", reason="Windows namespace alias contract")
def test_submit_reserves_windows_extended_path_directory_identity(
    tmp_path: Path,
) -> None:
    completed = threading.Event()
    compare_calls = 0
    first = _job(tmp_path, "record_windows_directory_owner")
    extended_dir = Path("\\\\?\\" + str(first.segment_dir.resolve()))
    assert os.path.samefile(first.segment_dir, extended_dir)

    def alias_job(segment_id: str) -> DualRecordingJob:
        return DualRecordingJob(
            segment_id=segment_id,
            segment_dir=extended_dir,
            front_source=extended_dir / first.front_source.name,
            side_source=extended_dir / first.side_source.name,
            front_frames=first.front_frames,
            side_frames=first.side_frames,
            front_template=first.front_template,
            side_template=first.side_template,
        )

    def compare(*_args, **_kwargs):
        nonlocal compare_calls
        compare_calls += 1
        return _pool_result(_args[0])

    processor = DualRecordingPostProcessor(
        on_update=lambda update: completed.set()
        if update.segment_id == first.segment_id and update.status == "completed"
        else None,
        transcode=lambda path, _stop: path,
        compare=compare,
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    try:
        assert processor.submit(first)
        assert not processor.submit(alias_job("record_windows_alias_concurrent"))
        _wait(completed)
        result_path = first.segment_dir / "result.json"
        original_result = result_path.read_text(encoding="utf-8")
        assert not processor.submit(alias_job("record_windows_alias_late"))
        processor._queue.join()
    finally:
        processor.close(1.0)

    assert compare_calls == 2  # 正/侧各一模板，各比对一次
    assert result_path.read_text(encoding="utf-8") == original_result
    assert json.loads(original_result)["segment_id"] == first.segment_id


@pytest.mark.skipif(os.name != "nt", reason="Windows namespace fallback contract")
def test_submit_reserves_windows_alias_when_filesystem_has_no_inode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    completed = threading.Event()
    compare_calls = 0
    first = _job(tmp_path, "record_zero_inode_owner")
    extended_dir = Path("\\\\?\\" + str(first.segment_dir.resolve()))
    monkeypatch.setattr(
        recording_postprocess,
        "_segment_directory_stat",
        lambda _path: SimpleNamespace(st_dev=17, st_ino=0),
    )

    def alias_job(segment_id: str) -> DualRecordingJob:
        return DualRecordingJob(
            segment_id=segment_id,
            segment_dir=extended_dir,
            front_source=extended_dir / first.front_source.name,
            side_source=extended_dir / first.side_source.name,
            front_frames=first.front_frames,
            side_frames=first.side_frames,
            front_template=first.front_template,
            side_template=first.side_template,
        )

    def compare(*_args, **_kwargs):
        nonlocal compare_calls
        compare_calls += 1
        return _pool_result(_args[0])

    processor = DualRecordingPostProcessor(
        on_update=lambda update: completed.set()
        if update.segment_id == first.segment_id and update.status == "completed"
        else None,
        transcode=lambda path, _stop: path,
        compare=compare,
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    try:
        assert processor.submit(first)
        assert not processor.submit(alias_job("record_zero_inode_concurrent"))
        _wait(completed)
        result_path = first.segment_dir / "result.json"
        original_result = result_path.read_text(encoding="utf-8")
        assert not processor.submit(alias_job("record_zero_inode_late"))
        processor._queue.join()
    finally:
        processor.close(1.0)

    assert compare_calls == 2  # 正/侧各一模板，各比对一次
    assert result_path.read_text(encoding="utf-8") == original_result
    assert json.loads(original_result)["segment_id"] == first.segment_id


@pytest.mark.skipif(os.name != "nt", reason="Windows namespace fallback contract")
@pytest.mark.parametrize(
    ("extended", "normal"),
    [
        (r"\\?\C:\recordings\segment", r"C:\recordings\segment"),
        (r"\\?\unc\server\share\segment", r"\\server\share\segment"),
    ],
)
def test_windows_namespace_text_fallback_is_case_insensitive(
    extended: str,
    normal: str,
) -> None:
    assert recording_postprocess._segment_directory_text_key(Path(extended)) == (
        recording_postprocess._segment_directory_text_key(Path(normal))
    )


@pytest.mark.parametrize(
    ("first_mode", "later_mode"),
    [
        pytest.param("stat_error", "inode", id="stat-error-to-inode"),
        pytest.param("zero_inode", "inode", id="zero-inode-to-inode"),
        pytest.param("inode", "stat_error", id="inode-to-stat-error"),
        pytest.param("inode", "zero_inode", id="inode-to-zero-inode"),
    ],
)
def test_submit_reserves_directory_when_identity_mode_changes(
    tmp_path: Path,
    monkeypatch,
    first_mode: str,
    later_mode: str,
) -> None:
    compare_started = threading.Event()
    release_compare = threading.Event()
    completed = threading.Event()
    compare_calls = 0
    stat_calls = 0
    first = _job(tmp_path, f"record_identity_transition_{first_mode}_{later_mode}")

    def changing_stat(_path: Path):
        nonlocal stat_calls
        stat_calls += 1
        mode = first_mode if stat_calls == 1 else later_mode
        if mode == "stat_error":
            raise OSError("transient stat failure")
        if mode == "zero_inode":
            return SimpleNamespace(st_dev=17, st_ino=0)
        return SimpleNamespace(st_dev=17, st_ino=23)

    monkeypatch.setattr(recording_postprocess, "_segment_directory_stat", changing_stat)

    def duplicate_job(segment_id: str) -> DualRecordingJob:
        return DualRecordingJob(
            segment_id=segment_id,
            segment_dir=first.segment_dir,
            front_source=first.front_source,
            side_source=first.side_source,
            front_frames=first.front_frames,
            side_frames=first.side_frames,
            front_template=first.front_template,
            side_template=first.side_template,
        )

    def compare(*_args, **_kwargs):
        nonlocal compare_calls
        compare_calls += 1
        compare_started.set()
        assert release_compare.wait(2.0)
        return _pool_result(_args[0])

    processor = DualRecordingPostProcessor(
        on_update=lambda update: completed.set()
        if update.segment_id == first.segment_id and update.status == "completed"
        else None,
        transcode=lambda path, _stop: path,
        compare=compare,
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    try:
        assert processor.submit(first)
        _wait(compare_started)
        assert not processor.submit(duplicate_job("record_identity_transition_concurrent"))
        release_compare.set()
        _wait(completed)
        result_path = first.segment_dir / "result.json"
        original_result = result_path.read_text(encoding="utf-8")
        assert not processor.submit(duplicate_job("record_identity_transition_late"))
        processor._queue.join()
    finally:
        release_compare.set()
        processor.close(1.0)

    assert compare_calls == 2  # 正/侧各一模板，各比对一次
    assert result_path.read_text(encoding="utf-8") == original_result
    assert json.loads(original_result)["segment_id"] == first.segment_id


def test_submit_returns_before_blocking_queued_callback_runs(tmp_path: Path) -> None:
    queued_callback_started = threading.Event()
    release_callback = threading.Event()
    callback_threads: list[str] = []

    def on_update(update: PostprocessUpdate) -> None:
        if update.status != "queued":
            return
        callback_threads.append(threading.current_thread().name)
        queued_callback_started.set()
        assert release_callback.wait(2.0)

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=lambda path, _stop: path,
        compare=lambda tpls, _video, **_kwargs: _pool_result(tpls),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    job = _job(tmp_path, "record_nonblocking_submit")
    started_at = time.monotonic()
    try:
        assert processor.submit(job)
        elapsed = time.monotonic() - started_at
        assert elapsed < 0.1
        _wait(queued_callback_started)
        payload = json.loads(
            (job.segment_dir / "result.json").read_text(encoding="utf-8")
        )
        assert payload["status"] == "queued"
        assert callback_threads == ["dual-recording-postprocess"]
        cancel_started_at = time.monotonic()
        processor.cancel_all()
        assert time.monotonic() - cancel_started_at < 0.1
    finally:
        release_callback.set()
        processor.close(1.0)


def test_submit_returns_while_another_result_write_is_blocked(tmp_path: Path) -> None:
    write_started = threading.Event()
    release_write = threading.Event()
    submit_returned = threading.Event()
    two_completed = threading.Event()
    submitted: list[bool] = []
    completed_ids: list[str] = []

    def on_update(update: PostprocessUpdate) -> None:
        if update.status != "completed":
            return
        completed_ids.append(update.segment_id)
        if len(completed_ids) == 2:
            two_completed.set()

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=lambda path, _stop: path,
        compare=lambda tpls, _video, **_kwargs: _pool_result(tpls),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    first = _job(tmp_path, "record_blocked_write")
    second = _job(tmp_path, "record_submit_during_write")
    original_write = processor._write_json_atomic

    def blocking_write(path: Path, payload: dict) -> None:
        if (
            payload["segment_id"] == first.segment_id
            and payload["status"] == "queued"
        ):
            write_started.set()
            assert release_write.wait(2.0)
        original_write(path, payload)

    def submit_second() -> None:
        submitted.append(processor.submit(second))
        submit_returned.set()

    processor._write_json_atomic = blocking_write
    submit_thread = threading.Thread(target=submit_second)
    try:
        assert processor.submit(first)
        _wait(write_started)
        submit_thread.start()
        assert submit_returned.wait(0.5), "submit was blocked by result persistence"
        assert submitted == [True]
        assert not release_write.is_set()
        release_write.set()
        _wait(two_completed)
    finally:
        release_write.set()
        submit_thread.join(1.0)
        processor.close(1.0)

    assert completed_ids == [first.segment_id, second.segment_id]
    for job in (first, second):
        payload = json.loads(
            (job.segment_dir / "result.json").read_text(encoding="utf-8")
        )
        assert payload["status"] == "completed"


@pytest.mark.parametrize("blocked_status", ["queued", "completed"])
def test_cancel_all_returns_during_result_write_and_wins_publication_race(
    tmp_path: Path,
    blocked_status: str,
) -> None:
    write_started = threading.Event()
    release_write = threading.Event()
    cancel_returned = threading.Event()
    cancelled = threading.Event()
    updates: list[PostprocessUpdate] = []

    def on_update(update: PostprocessUpdate) -> None:
        updates.append(update)
        if update.status == "cancelled":
            cancelled.set()

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=lambda path, _stop: path,
        compare=lambda tpls, _video, **_kwargs: _pool_result(tpls),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    job = _job(tmp_path, f"record_cancel_during_{blocked_status}_write")
    original_write = processor._write_json_atomic

    def blocking_write(path: Path, payload: dict) -> None:
        original_write(path, payload)
        if payload["status"] == blocked_status:
            write_started.set()
            assert release_write.wait(2.0)

    def cancel() -> None:
        processor.cancel_all()
        cancel_returned.set()

    processor._write_json_atomic = blocking_write
    cancel_thread = threading.Thread(target=cancel)
    try:
        assert processor.submit(job)
        _wait(write_started)
        cancel_thread.start()
        assert cancel_returned.wait(0.5), "cancel_all was blocked by result persistence"
        assert not release_write.is_set()
        release_write.set()
        _wait(cancelled)
        processor._queue.join()
    finally:
        release_write.set()
        cancel_thread.join(1.0)
        processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "cancelled"
    assert payload["error"]["code"] == "app_closing"
    assert payload["result"] is None
    assert [
        update.status
        for update in updates
        if update.status in {"completed", "failed", "skipped", "cancelled"}
    ] == ["cancelled"]
    assert all(update.status != blocked_status for update in updates)


def test_annotated_recording_is_transcoded_then_skipped(tmp_path: Path) -> None:
    terminal = threading.Event()
    calls: list[Path] = []
    updates: list[PostprocessUpdate] = []

    def on_update(update: PostprocessUpdate) -> None:
        updates.append(update)
        if update.status == "skipped":
            terminal.set()

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=lambda path, _stop: calls.append(path) or path,
        compare=lambda *_args, **_kwargs: pytest.fail("annotated video must not be compared"),
        video_validator=lambda _path: True,
        model_available=lambda: pytest.fail("skipped job must not require a model"),
    )
    job = _job(tmp_path, "record_annotated", skeleton=True)
    try:
        assert processor.submit(job)
        _wait(terminal)
    finally:
        processor.close(1.0)

    assert calls == [job.front_source, job.side_source]
    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "skipped"
    assert payload["error"]["code"] == "annotated_recording"
    assert payload["result"] is None


def test_auto_compare_disabled_is_transcoded_then_skipped(tmp_path: Path) -> None:
    terminal = threading.Event()
    calls: list[Path] = []

    def on_update(update: PostprocessUpdate) -> None:
        if update.status == "skipped":
            terminal.set()

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=lambda path, _stop: calls.append(path) or path,
        compare=lambda *_args, **_kwargs: pytest.fail(
            "record-only mode must not be compared"
        ),
        video_validator=lambda _path: True,
        model_available=lambda: pytest.fail("record-only job must not require a model"),
    )
    job = _job(tmp_path, "record_only", auto_compare=False)
    try:
        assert processor.submit(job)
        _wait(terminal)
    finally:
        processor.close(1.0)

    assert calls == [job.front_source, job.side_source]
    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "skipped"
    assert payload["auto_compare"] is False
    assert payload["error"]["code"] == "auto_compare_disabled"
    assert payload["result"] is None


def test_annotated_recording_cancelled_during_validation_is_not_skipped(
    tmp_path: Path,
) -> None:
    validating = threading.Event()
    release_validation = threading.Event()
    cancelled = threading.Event()
    validation_calls = 0

    def validate(_path: Path) -> bool:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 2:
            validating.set()
            assert release_validation.wait(2.0)
        return True

    def on_update(update: PostprocessUpdate) -> None:
        if update.status == "cancelled":
            cancelled.set()

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=lambda path, _stop: path,
        compare=lambda *_args, **_kwargs: pytest.fail("cancelled job must not compare"),
        video_validator=validate,
        model_available=lambda: pytest.fail("annotated job must not require a model"),
    )
    job = _job(tmp_path, "record_annotated_cancel", skeleton=True)
    try:
        assert processor.submit(job)
        _wait(validating)
        processor.cancel_all()
        release_validation.set()
        _wait(cancelled)
    finally:
        release_validation.set()
        processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "cancelled"
    assert payload["error"]["code"] == "app_closing"


def test_invalid_recording_snapshot_fails_before_transcode(tmp_path: Path) -> None:
    terminal = threading.Event()
    updates: list[PostprocessUpdate] = []

    def on_update(update: PostprocessUpdate) -> None:
        updates.append(update)
        if update.status == "failed":
            terminal.set()

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=lambda *_args: pytest.fail("invalid recording must not transcode"),
        compare=lambda *_args, **_kwargs: pytest.fail("invalid recording must not compare"),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    job = _job(tmp_path, "record_mismatch", front_frames=12, side_frames=11)
    try:
        assert processor.submit(job)
        _wait(terminal)
    finally:
        processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "frame_count_mismatch"


@pytest.mark.parametrize("mismatched_view", ["front", "side"])
def test_recording_source_outside_segment_fails_before_transcode(
    tmp_path: Path,
    mismatched_view: str,
) -> None:
    terminal = threading.Event()
    transcode_calls: list[Path] = []
    compare_calls = 0
    base = _job(tmp_path, f"record_path_mismatch_{mismatched_view}")
    foreign_dir = tmp_path / "other_segment"
    foreign_dir.mkdir(exist_ok=True)
    foreign_source = foreign_dir / f"{mismatched_view}.mp4"
    foreign_source.write_bytes(b"foreign-video")
    job = DualRecordingJob(
        **{
            **base.__dict__,
            f"{mismatched_view}_source": foreign_source,
        }
    )

    def compare(*_args, **_kwargs):
        nonlocal compare_calls
        compare_calls += 1
        return _pool_result(_args[0])

    processor = DualRecordingPostProcessor(
        on_update=lambda update: terminal.set() if update.status == "failed" else None,
        transcode=lambda path, _stop: transcode_calls.append(path) or path,
        compare=compare,
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    try:
        assert processor.submit(job)
        _wait(terminal)
    finally:
        processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "recording_path_mismatch"
    assert transcode_calls == []
    assert compare_calls == 0


def test_zero_frame_recording_has_stable_failure_code(tmp_path: Path) -> None:
    terminal = threading.Event()
    processor = DualRecordingPostProcessor(
        on_update=lambda update: terminal.set() if update.status == "failed" else None,
        transcode=lambda *_args: pytest.fail("empty recording must not transcode"),
        compare=lambda *_args, **_kwargs: pytest.fail("empty recording must not compare"),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    job = _job(tmp_path, "record_empty", front_frames=0, side_frames=0)
    try:
        assert processor.submit(job)
        _wait(terminal)
    finally:
        processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "recording_empty"


def test_writer_error_has_stable_side_failure_code(tmp_path: Path) -> None:
    terminal = threading.Event()
    base = _job(tmp_path, "record_side_writer_failed")
    job = DualRecordingJob(
        **{
            **base.__dict__,
            "side_error": "side writer unavailable",
        }
    )
    processor = DualRecordingPostProcessor(
        on_update=lambda update: terminal.set() if update.status == "failed" else None,
        transcode=lambda *_args: pytest.fail("writer failure must not transcode"),
        compare=lambda *_args, **_kwargs: pytest.fail("writer failure must not compare"),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    try:
        assert processor.submit(job)
        _wait(terminal)
    finally:
        processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"] == {
        "code": "side_recording_failed",
        "message": "侧面录像失败：side writer unavailable",
    }


def test_avi_fallback_is_compared_and_recorded_as_warning(tmp_path: Path) -> None:
    terminal = threading.Event()

    def on_update(update: PostprocessUpdate) -> None:
        if update.status == "completed":
            terminal.set()

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=lambda path, _stop: path,
        compare=lambda tpls, _video, **_kwargs: _pool_result(tpls),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    job = _job(tmp_path, "record_avi", suffix=".avi")
    try:
        assert processor.submit(job)
        _wait(terminal)
    finally:
        processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert len(payload["warnings"]) == 2
    assert all("回退使用 AVI" in item for item in payload["warnings"])


def test_transcode_dependency_error_has_stage_specific_failure_code(
    tmp_path: Path,
) -> None:
    terminal = threading.Event()
    compare_calls = 0

    def transcode(_path: Path, _stop: threading.Event) -> Path:
        raise OSError("ffmpeg unavailable")

    def compare(*_args, **_kwargs):
        nonlocal compare_calls
        compare_calls += 1
        return _pool_result(_args[0])

    processor = DualRecordingPostProcessor(
        on_update=lambda update: terminal.set() if update.status == "failed" else None,
        transcode=transcode,
        compare=compare,
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    job = _job(tmp_path, "record_transcode_error")
    try:
        assert processor.submit(job)
        _wait(terminal)
    finally:
        processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "transcode_failed"
    assert "正面录像转码失败" in payload["error"]["message"]
    assert compare_calls == 0


def test_missing_model_fails_without_calling_compare(tmp_path: Path) -> None:
    terminal = threading.Event()

    def on_update(update: PostprocessUpdate) -> None:
        if update.status == "failed":
            terminal.set()

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=lambda path, _stop: path,
        compare=lambda *_args, **_kwargs: pytest.fail("missing model must not compare"),
        video_validator=lambda _path: True,
        model_available=lambda: False,
    )
    job = _job(tmp_path, "record_no_model")
    try:
        assert processor.submit(job)
        _wait(terminal)
    finally:
        processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "model_missing"


@pytest.mark.parametrize("validator_mode", ["false", "raises"])
def test_unreadable_video_has_stable_failure_code(
    tmp_path: Path, validator_mode: str
) -> None:
    terminal = threading.Event()

    def validate(_path: Path) -> bool:
        if validator_mode == "raises":
            raise OSError("decoder unavailable")
        return False

    processor = DualRecordingPostProcessor(
        on_update=lambda update: terminal.set() if update.status == "failed" else None,
        transcode=lambda path, _stop: path,
        compare=lambda *_args, **_kwargs: pytest.fail("unreadable video must not compare"),
        video_validator=validate,
        model_available=lambda: True,
    )
    job = _job(tmp_path, f"record_unreadable_{validator_mode}")
    try:
        assert processor.submit(job)
        _wait(terminal)
    finally:
        processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "video_unreadable"
    assert payload["result"] is None


def test_compare_exception_fails_one_job_and_fifo_continues(tmp_path: Path) -> None:
    terminal = threading.Event()

    def compare(template_paths, video, **_kwargs):
        if Path(video).parent.name == "record_compare_fails":
            raise RuntimeError("DTW crashed")
        return _pool_result(template_paths)

    processor = DualRecordingPostProcessor(
        on_update=lambda update: terminal.set()
        if update.segment_id == "record_after_compare_failure"
        and update.status == "completed"
        else None,
        transcode=lambda path, _stop: path,
        compare=compare,
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    failed = _job(tmp_path, "record_compare_fails")
    following = _job(tmp_path, "record_after_compare_failure")
    try:
        assert processor.submit(failed)
        assert processor.submit(following)
        _wait(terminal)
    finally:
        processor.close(1.0)

    failed_payload = json.loads(
        (failed.segment_dir / "result.json").read_text(encoding="utf-8")
    )
    following_payload = json.loads(
        (following.segment_dir / "result.json").read_text(encoding="utf-8")
    )
    assert failed_payload["status"] == "failed"
    assert failed_payload["error"]["code"] == "compare_failed"
    assert failed_payload["result"] is None
    assert following_payload["status"] == "completed"


def test_failure_after_transcode_keeps_final_paths(tmp_path: Path) -> None:
    terminal = threading.Event()

    def transcode(path: Path, _stop: threading.Event) -> Path:
        final_path = path.with_suffix(".mp4")
        final_path.write_bytes(b"h264")
        return final_path

    def on_update(update: PostprocessUpdate) -> None:
        if update.status == "failed":
            terminal.set()

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=transcode,
        compare=lambda *_args, **_kwargs: pytest.fail("missing model must not compare"),
        video_validator=lambda _path: True,
        model_available=lambda: False,
    )
    job = _job(tmp_path, "record_final_paths", suffix=".avi")
    try:
        assert processor.submit(job)
        _wait(terminal)
    finally:
        processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["error"]["code"] == "model_missing"
    assert payload["front_video_path"].endswith("front.mp4")
    assert payload["side_video_path"].endswith("side.mp4")


def test_failure_after_avi_fallback_keeps_transcode_warnings(tmp_path: Path) -> None:
    terminal = threading.Event()
    processor = DualRecordingPostProcessor(
        on_update=lambda update: terminal.set() if update.status == "failed" else None,
        transcode=lambda path, _stop: path,
        compare=lambda *_args, **_kwargs: pytest.fail("missing model must not compare"),
        video_validator=lambda _path: True,
        model_available=lambda: False,
    )
    job = _job(tmp_path, "record_fallback_then_fail", suffix=".avi")
    try:
        assert processor.submit(job)
        _wait(terminal)
    finally:
        processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "model_missing"
    assert len(payload["warnings"]) == 2
    assert all("回退使用 AVI" in warning for warning in payload["warnings"])


def test_close_cancels_active_transcode_and_writes_cancelled_result(tmp_path: Path) -> None:
    started = threading.Event()
    cancelled = threading.Event()

    def transcode(_path: Path, stop_evt: threading.Event):
        started.set()
        assert stop_evt.wait(2.0)
        raise InterruptedError("cancelled")

    def on_update(update: PostprocessUpdate) -> None:
        if update.status == "cancelled":
            cancelled.set()

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=transcode,
        compare=lambda *_args, **_kwargs: pytest.fail("cancelled job must not compare"),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    job = _job(tmp_path, "record_cancelled")
    assert processor.submit(job)
    _wait(started)
    processor.close(2.0)
    _wait(cancelled)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "cancelled"
    assert payload["error"]["code"] == "app_closing"
    assert not processor.submit(_job(tmp_path, "record_after_close"))


def test_cancel_all_marks_active_and_queued_jobs_cancelled(tmp_path: Path) -> None:
    active_started = threading.Event()
    two_cancelled = threading.Event()
    cancelled_ids: list[str] = []
    callback_threads: list[str] = []
    transcode_calls: list[str] = []

    def transcode(path: Path, stop_evt: threading.Event):
        transcode_calls.append(path.parent.name)
        active_started.set()
        assert stop_evt.wait(2.0)
        raise InterruptedError("cancelled")

    def on_update(update: PostprocessUpdate) -> None:
        if update.status == "cancelled":
            cancelled_ids.append(update.segment_id)
            callback_threads.append(threading.current_thread().name)
            if len(cancelled_ids) == 2:
                two_cancelled.set()

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=transcode,
        compare=lambda *_args, **_kwargs: pytest.fail("cancelled jobs must not compare"),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    active = _job(tmp_path, "record_active")
    queued = _job(tmp_path, "record_queued")
    assert processor.submit(active)
    assert processor.submit(queued)
    _wait(active_started)

    processor.cancel_all()
    _wait(two_cancelled)
    processor.close(1.0)

    assert transcode_calls == ["record_active"]
    assert cancelled_ids == [active.segment_id, queued.segment_id]
    assert callback_threads == ["dual-recording-postprocess"] * 2
    for job in (active, queued):
        payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
        assert payload["status"] == "cancelled"
        assert payload["result"] is None
    assert not processor.submit(_job(tmp_path, "record_rejected"))


def test_cancel_after_successful_transcode_keeps_final_video_path(tmp_path: Path) -> None:
    cancelled = threading.Event()
    calls = 0

    def transcode(path: Path, stop_evt: threading.Event) -> Path:
        nonlocal calls
        calls += 1
        final_path = path.with_suffix(".mp4")
        final_path.write_bytes(b"h264")
        path.unlink()
        stop_evt.set()
        return final_path

    processor = DualRecordingPostProcessor(
        on_update=lambda update: cancelled.set() if update.status == "cancelled" else None,
        transcode=transcode,
        compare=lambda *_args, **_kwargs: pytest.fail("cancelled job must not compare"),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    job = _job(tmp_path, "record_cancel_after_transcode", suffix=".avi")
    try:
        assert processor.submit(job)
        _wait(cancelled)
    finally:
        processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert calls == 1
    assert payload["status"] == "cancelled"
    assert payload["front_video_path"].endswith("front.mp4")
    assert Path(payload["front_video_path"]).is_file()


def test_cancel_all_defers_terminal_until_worker_finalizes_transcode_path(
    tmp_path: Path,
) -> None:
    output_ready = threading.Event()
    allow_return = threading.Event()
    calls = 0

    def transcode(path: Path, _stop_evt: threading.Event) -> Path:
        nonlocal calls
        calls += 1
        final_path = path.with_suffix(".mp4")
        final_path.write_bytes(b"h264")
        path.unlink()
        output_ready.set()
        assert allow_return.wait(2.0)
        return final_path

    processor = DualRecordingPostProcessor(
        transcode=transcode,
        compare=lambda *_args, **_kwargs: pytest.fail("cancelled job must not compare"),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    job = _job(tmp_path, "record_cancel_inflight_transcode", suffix=".avi")
    assert processor.submit(job)
    _wait(output_ready)

    processor.cancel_all()
    immediate = json.loads(
        (job.segment_dir / "result.json").read_text(encoding="utf-8")
    )
    assert immediate["status"] == "transcoding"

    allow_return.set()
    processor.close(1.0)
    final = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert calls == 1
    assert final["status"] == "cancelled"
    assert final["error"]["code"] == "app_closing"
    assert final["front_video_path"].endswith("front.mp4")
    assert Path(final["front_video_path"]).is_file()


def test_cancelled_result_path_enrichment_does_not_notify_twice(
    tmp_path: Path,
) -> None:
    terminal = threading.Event()
    cancelled_updates: list[PostprocessUpdate] = []
    callback_threads: list[str] = []
    job = _job(tmp_path, "record_cancelled_enrichment", suffix=".avi")
    processor: DualRecordingPostProcessor

    def transcode(path: Path, stop_evt: threading.Event) -> Path:
        final_path = path.with_suffix(".mp4")
        final_path.write_bytes(b"h264")
        path.unlink()
        processor._publish_terminal_safely(
            job,
            "cancelled",
            message="后台比对已取消",
            error_code="cancelled",
        )
        stop_evt.set()
        return final_path

    def on_update(update: PostprocessUpdate) -> None:
        if update.status == "cancelled":
            cancelled_updates.append(update)
            callback_threads.append(threading.current_thread().name)
            terminal.set()

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=transcode,
        compare=lambda *_args, **_kwargs: pytest.fail("cancelled job must not compare"),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    try:
        assert processor.submit(job)
        _wait(terminal)
        processor._queue.join()
    finally:
        processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "cancelled"
    assert payload["front_video_path"].endswith("front.mp4")
    assert Path(payload["front_video_path"]).is_file()
    assert [update.segment_id for update in cancelled_updates] == [job.segment_id]
    assert callback_threads == ["dual-recording-postprocess"]


def test_close_from_worker_callback_returns_and_allows_callback_to_finish(
    tmp_path: Path,
) -> None:
    callback_finished = threading.Event()
    cancelled = threading.Event()
    callback_steps: list[str] = []
    processor: DualRecordingPostProcessor

    def on_update(update: PostprocessUpdate) -> None:
        if update.status == "queued":
            callback_steps.append("before-close")
            processor.close(1.0)
            callback_steps.append("after-close")
            callback_finished.set()
        elif update.status == "cancelled":
            cancelled.set()

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=lambda path, _stop: path,
        compare=lambda *_args, **_kwargs: pytest.fail("closed job must not compare"),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    job = _job(tmp_path, "record_callback_close")

    assert processor.submit(job)
    _wait(callback_finished)
    _wait(cancelled)
    processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert callback_steps == ["before-close", "after-close"]
    assert payload["status"] == "cancelled"
    assert payload["error"]["code"] == "app_closing"
    assert not processor._worker.is_alive()


def test_cancelled_enrichment_write_failure_preserves_original_terminal(
    tmp_path: Path,
) -> None:
    terminal = threading.Event()
    enrichment_attempted = threading.Event()
    terminal_updates: list[PostprocessUpdate] = []
    original_result: list[str] = []
    job = _job(tmp_path, "record_enrichment_write_failure", suffix=".avi")
    processor: DualRecordingPostProcessor

    def transcode(path: Path, stop_evt: threading.Event) -> Path:
        final_path = path.with_suffix(".mp4")
        final_path.write_bytes(b"h264")
        path.unlink()
        processor._publish_terminal_safely(
            job,
            "cancelled",
            message="后台比对已取消",
            error_code="cancelled",
        )
        stop_evt.set()
        return final_path

    def on_update(update: PostprocessUpdate) -> None:
        if update.status in {"completed", "failed", "skipped", "cancelled"}:
            terminal_updates.append(update)
            terminal.set()

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=transcode,
        compare=lambda *_args, **_kwargs: pytest.fail("cancelled job must not compare"),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    original_write = processor._write_json_atomic

    def fail_enrichment(path: Path, payload: dict) -> None:
        if (
            payload["status"] == "cancelled"
            and payload["front_video_path"].endswith("front.mp4")
        ):
            original_result.append(path.read_text(encoding="utf-8"))
            enrichment_attempted.set()
            raise OSError("disk unavailable during enrichment")
        original_write(path, payload)

    processor._write_json_atomic = fail_enrichment
    try:
        assert processor.submit(job)
        _wait(terminal)
        _wait(enrichment_attempted)
        processor._queue.join()
    finally:
        processor.close(1.0)

    result_path = job.segment_dir / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert result_path.read_text(encoding="utf-8") == original_result[0]
    assert payload["status"] == "cancelled"
    assert payload["front_video_path"].endswith("front.avi")
    assert [update.status for update in terminal_updates] == ["cancelled"]
    assert terminal_updates[0].error_code == "cancelled"


def test_cancel_all_returns_before_slow_terminal_callback_and_notifies_once(
    tmp_path: Path,
) -> None:
    transcode_started = threading.Event()
    callback_started = threading.Event()
    release_callback = threading.Event()
    callback_threads: list[str] = []
    cancelled_updates: list[PostprocessUpdate] = []

    def transcode(_path: Path, stop_evt: threading.Event) -> Path:
        transcode_started.set()
        assert stop_evt.wait(2.0)
        raise InterruptedError("cancelled")

    def on_update(update: PostprocessUpdate) -> None:
        if update.status != "cancelled":
            return
        cancelled_updates.append(update)
        callback_threads.append(threading.current_thread().name)
        callback_started.set()
        assert release_callback.wait(2.0)

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=transcode,
        compare=lambda *_args, **_kwargs: pytest.fail("cancelled job must not compare"),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    job = _job(tmp_path, "record_nonblocking_cancel")
    assert processor.submit(job)
    _wait(transcode_started)

    started_at = time.monotonic()
    processor.cancel_all()
    elapsed = time.monotonic() - started_at
    assert elapsed < 0.1
    _wait(callback_started)
    assert callback_threads == ["dual-recording-postprocess"]

    release_callback.set()
    processor.close(1.0)
    assert [update.segment_id for update in cancelled_updates] == [job.segment_id]


def test_cancel_between_scoring_and_terminal_publish_still_finishes_cancelled(
    tmp_path: Path,
) -> None:
    terminal = threading.Event()
    cancelled_updates: list[PostprocessUpdate] = []

    def on_update(update: PostprocessUpdate) -> None:
        if update.status == "cancelled":
            cancelled_updates.append(update)
            terminal.set()

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=lambda path, _stop: path,
        compare=lambda tpls, _video, **_kwargs: _pool_result(tpls),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    job = _job(tmp_path, "record_cancel_before_terminal")
    original_result_payload = processor._result_payload

    def cancel_before_terminal(result):
        processor.cancel_all()
        return original_result_payload(result)

    processor._result_payload = cancel_before_terminal
    try:
        assert processor.submit(job)
        _wait(terminal)
    finally:
        processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "cancelled"
    assert payload["error"]["code"] == "app_closing"
    assert payload["result"] is None
    assert [update.segment_id for update in cancelled_updates] == [job.segment_id]


@pytest.mark.parametrize("compare_outcome", ["returns", "raises"])
def test_cancel_all_marks_task_while_compare_is_still_blocked(
    tmp_path: Path, compare_outcome: str
) -> None:
    compare_started = threading.Event()
    release_compare = threading.Event()

    def compare(*_args, **_kwargs):
        compare_started.set()
        assert release_compare.wait(2.0)
        if compare_outcome == "raises":
            raise RuntimeError("late DTW failure")
        return _pool_result(_args[0])

    processor = DualRecordingPostProcessor(
        transcode=lambda path, _stop: path,
        compare=compare,
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    job = _job(tmp_path, f"record_blocked_compare_{compare_outcome}")
    assert processor.submit(job)
    _wait(compare_started)

    processor.close(0.05)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert processor._worker.is_alive()
    assert payload["status"] == "comparing"
    assert payload["error"] is None

    release_compare.set()
    processor.close(1.0)
    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert not processor._worker.is_alive()
    assert payload["status"] == "cancelled"
    assert payload["result"] is None


def test_missing_template_has_stable_failure_code(tmp_path: Path) -> None:
    """正/侧模板都缺时一条龙失败；仅缺一侧不走 template_missing。"""
    terminal = threading.Event()

    def on_update(update: PostprocessUpdate) -> None:
        if update.status == "failed":
            terminal.set()

    job = _job(tmp_path, "record_no_template")
    job = DualRecordingJob(
        **{
            **job.__dict__,
            "front_template": tmp_path / "missing-front.npz",
            "side_template": tmp_path / "missing-side.npz",
            "front_templates": (),
            "side_templates": (),
        }
    )
    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=lambda path, _stop: path,
        compare=lambda *_args, **_kwargs: pytest.fail("missing template must not compare"),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    try:
        assert processor.submit(job)
        _wait(terminal)
    finally:
        processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "template_missing"


def test_auto_compare_runs_with_only_side_template(tmp_path: Path, monkeypatch) -> None:
    """仅侧面模板填入时仍可完成一条龙（单流比对）。"""
    terminal = threading.Event()
    updates: list[PostprocessUpdate] = []
    single_calls: list[tuple[str, str]] = []

    def on_update(update: PostprocessUpdate) -> None:
        updates.append(update)
        if update.status in {"completed", "failed", "skipped", "cancelled"}:
            terminal.set()

    side_tpl = tmp_path / "side_only.npz"
    _write_template(side_tpl)
    job = _job(tmp_path, "record_side_only")
    job = DualRecordingJob(
        **{
            **job.__dict__,
            "front_template": tmp_path / "missing-front.npz",
            "side_template": side_tpl,
            "front_templates": (),
            "side_templates": (),
        }
    )

    def fake_pool(template_paths, video_path, **kwargs):
        for tp in template_paths:
            single_calls.append((str(tp), str(video_path)))
        return [
            SimpleNamespace(score=0.77, start_frame=1, end_frame=10)
            for _ in template_paths
        ]

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=lambda path, _stop: path,
        compare=fake_pool,
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    try:
        assert processor.submit(job)
        _wait(terminal)
    finally:
        processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["result"]["front_score"] is None
    assert payload["result"]["side_score"] == pytest.approx(0.77)
    assert payload["result"]["combined_percent"] == 77
    assert single_calls and "side" in single_calls[0][1]
    assert any(u.status == "completed" for u in updates)


def _write_template(
    path: Path,
    *,
    shape: tuple[int, ...] = (3, 22, 2),
    pose_variant: str = "heavy",
    layout: str = "pose33_v3",
    nonfinite: bool = False,
) -> None:
    features = np.zeros(shape, dtype=np.float32)
    if nonfinite:
        features.reshape(-1)[0] = np.nan
    np.savez(
        path,
        features=features,
        meta={"pose_variant": pose_variant, "feature_layout": layout},
    )


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("shape", "template_invalid"),
        ("nonfinite", "template_invalid"),
        ("pose_variant", "template_invalid"),
        ("layout_mismatch", "template_incompatible"),
    ],
)
def test_template_pair_rejects_invalid_metadata_and_features(
    tmp_path: Path, mutation: str, expected_code: str
) -> None:
    front = tmp_path / f"front_{mutation}.npz"
    side = tmp_path / f"side_{mutation}.npz"
    _write_template(front)
    _write_template(side)
    if mutation == "shape":
        _write_template(front, shape=(3, 21, 2))
    elif mutation == "nonfinite":
        _write_template(front, nonfinite=True)
    elif mutation == "pose_variant":
        _write_template(front, pose_variant="full")
    else:
        _write_template(side, layout="other_layout")

    with pytest.raises(PostprocessError) as exc_info:
        validate_template_pair(front, side)

    assert exc_info.value.code == expected_code


def test_result_write_failure_does_not_stop_fifo_worker(tmp_path: Path) -> None:
    terminal = threading.Event()
    updates: list[PostprocessUpdate] = []

    def on_update(update: PostprocessUpdate) -> None:
        updates.append(update)
        if update.segment_id == "record_second" and update.status == "completed":
            terminal.set()

    processor = DualRecordingPostProcessor(
        on_update=on_update,
        transcode=lambda path, _stop: path,
        compare=lambda tpls, _video, **_kwargs: _pool_result(tpls),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    first = _job(tmp_path, "record_first")
    second = _job(tmp_path, "record_second")
    original_write = processor._write_json_atomic

    def flaky_write(path: Path, payload: dict) -> None:
        if path.parent.name == "record_first":
            raise OSError("disk full")
        original_write(path, payload)

    processor._write_json_atomic = flaky_write
    try:
        assert processor.submit(first)
        assert processor.submit(second)
        _wait(terminal)
    finally:
        processor.close(1.0)

    assert any(
        item.segment_id == "record_first" and item.error_code == "result_write_failed"
        for item in updates
    )
    payload = json.loads((second.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"


def test_one_shot_result_write_failure_has_stable_error_code(tmp_path: Path) -> None:
    terminal = threading.Event()
    compare_calls = 0

    def compare(*_args, **_kwargs):
        nonlocal compare_calls
        compare_calls += 1
        return _pool_result(_args[0])

    processor = DualRecordingPostProcessor(
        on_update=lambda update: terminal.set() if update.status == "failed" else None,
        transcode=lambda path, _stop: path,
        compare=compare,
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    job = _job(tmp_path, "record_one_shot_write_failure")
    original_write = processor._write_json_atomic
    failed_once = False

    def flaky_write(path: Path, payload: dict) -> None:
        nonlocal failed_once
        if payload["status"] == "comparing" and not failed_once:
            failed_once = True
            raise OSError("disk hiccup")
        original_write(path, payload)

    processor._write_json_atomic = flaky_write
    try:
        assert processor.submit(job)
        _wait(terminal)
    finally:
        processor.close(1.0)

    payload = json.loads((job.segment_dir / "result.json").read_text(encoding="utf-8"))
    assert failed_once is True
    assert compare_calls == 0
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "result_write_failed"


def test_repeated_close_keeps_exit_sentinel(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    def transcode(path: Path, _stop: threading.Event) -> Path:
        started.set()
        release.wait(2.0)
        return path

    processor = DualRecordingPostProcessor(
        transcode=transcode,
        compare=lambda tpls, _video, **_kwargs: _pool_result(tpls),
        video_validator=lambda _path: True,
        model_available=lambda: True,
    )
    assert processor.submit(_job(tmp_path, "record_repeated_close"))
    _wait(started)
    processor.close(0.0)
    processor.close(0.0)
    release.set()
    processor.close(2.0)
    assert not processor._worker.is_alive()
