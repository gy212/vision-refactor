"""正式报告的有限图片名额优先覆盖各阶段和次数的判定依据。"""

from copy import deepcopy

from core.reporting import build_report_model, group_action_issues, select_evidence
from tests.test_evidence_contract import _record_with_evidence, _render_requests


def _decoded(requested):
    return [(view, frame) for view, frame, _ in requested]


def _add_prior_issues(record, count, *, with_observation=False):
    template = deepcopy(record["result"]["checks"][0])
    preceding = []
    for number in range(count):
        check = deepcopy(template)
        check.update(id=f"prior:{number}", phase="start", phaseLabel="开始实战式")
        details = []
        for view in ("side", "front") if with_observation else ("side",):
            detail = deepcopy(next(d for d in template["evidenceDetails"] if d["view"] == view))
            ref = f"{view}:{100 + number}"
            detail.update(ref=ref, checkId=check["id"], phaseLabel=check["phaseLabel"])
            details.append(detail)
            record["result"]["evidence"].append(
                {"id": ref, "view": view, "frame": 100 + number, "timeSeconds": number + 10.0}
            )
        check["evidenceDetails"] = details
        check["evidence"] = [detail["ref"] for detail in details]
        preceding.append(check)
    record["result"]["checks"] = preceding + record["result"]["checks"]


def test_two_slots_keep_process_and_finish_problem_images(monkeypatch):
    record = _record_with_evidence(grouped=True)
    requested, rendered = _render_requests(monkeypatch, record, max_images=2)

    assert _decoded(requested) == [("side", 10), ("side", 20)]
    assert "动作结束瞬间 · 侧面问题依据 · 0.667s · 第20帧（图片数量" not in rendered
    assert "动作过程 · 正面观测画面 · 0.333s · 第10帧（图片数量" in rendered


def test_default_remaining_two_slots_keep_both_stages(monkeypatch):
    record = _record_with_evidence(grouped=True)
    _add_prior_issues(record, 10)
    requested, rendered = _render_requests(monkeypatch, record)

    assert _decoded(requested) == [("side", 100), ("side", 10), ("side", 20)] + [
        ("side", 100 + i) for i in range(1, 10)
    ]
    assert rendered.count("data:image/jpeg;base64,") == 12


def test_second_problem_view_does_not_replace_another_stages_first_image(monkeypatch):
    record = _record_with_evidence(grouped=True)
    for check in record["result"]["checks"]:
        for detail in check["evidenceDetails"]:
            detail["purpose"] = "violation"
    requested, _ = _render_requests(monkeypatch, record, max_images=2)

    assert _decoded(requested) == [("side", 10), ("side", 20)]


def test_prior_cards_observations_cannot_consume_later_stages_budget(monkeypatch):
    record = _record_with_evidence(grouped=True)
    _add_prior_issues(record, 5, with_observation=True)
    requested, _ = _render_requests(monkeypatch, record, max_images=7)

    assert _decoded(requested) == [("side", 100), ("side", 10), ("side", 20)] + [
        ("side", 100 + i) for i in range(1, 5)
    ]


def test_every_repetition_gets_support_before_observation(monkeypatch):
    record = _record_with_evidence(grouped=True)
    for check in record["result"]["checks"]:
        for detail in check["evidenceDetails"]:
            detail["repetition"] = int(detail["ref"].partition(":")[2]) % 10 + 1
    model = build_report_model(record)
    grouped = group_action_issues(model.candidate_checks, record["action"])[0]
    selected = select_evidence(grouped, model.evidence_catalog)
    requested, _ = _render_requests(monkeypatch, record, max_images=6)

    assert {(item.check_id, item.repetition) for item in selected[:6]} == {
        (check["id"], repetition)
        for check in record["result"]["checks"]
        for repetition in (1, 2, 3)
    }
    assert _decoded(requested) == [("side", frame) for frame in (10, 11, 12, 20, 21, 22)]


def test_required_reference_uses_primary_sample_and_precedes_observation(monkeypatch):
    record = _record_with_evidence(grouped=True, paired=True)
    process = record["result"]["checks"][0]
    # 参照的保存顺序与主图不同，只能取主图实际使用的同采样参照。
    process["evidenceDetails"].reverse()
    for detail in process["evidenceDetails"]:
        if detail["view"] == "side":
            detail["weight"] = 1.0 if detail["ref"] == "side:10" else 0.8
        else:
            detail["referenceFor"] = detail["ref"].replace("front:", "side:")
    requested, rendered = _render_requests(monkeypatch, record, max_images=4)

    assert _decoded(requested) == [("side", 10), ("side", 20), ("front", 10), ("front", 20)]
    assert "正面测量参照" in rendered
    assert "正面问题依据" not in rendered


def test_shared_frame_keeps_each_repetition_caption_with_one_image_slot(monkeypatch):
    record = _record_with_evidence()
    check = record["result"]["checks"][0]
    detail = next(d for d in check["evidenceDetails"] if d["ref"] == "side:10")
    check["evidence"] = ["side:10"]
    check["evidenceDetails"] = [{**detail, "repetition": n} for n in (1, 2)]
    requested, rendered = _render_requests(monkeypatch, record, max_images=1)

    assert _decoded(requested) == [("side", 10)]
    assert "第1次 · 动作过程 · 侧面问题依据" in rendered
    assert "第2次 · 动作过程 · 侧面问题依据" in rendered
    assert "图片数量已达上限" not in rendered


def test_reference_for_observation_cannot_displace_another_stages_problem(monkeypatch):
    record = _record_with_evidence(grouped=True, paired=True)
    process, finish = record["result"]["checks"]
    for check in (process, finish):
        for detail in check["evidenceDetails"]:
            detail["repetition"] = 1
            if detail["purpose"] == "reference":
                detail["referenceFor"] = detail["ref"].replace("front:", "side:")
    for detail in process["evidenceDetails"]:
        if detail["ref"].endswith(":11"):
            detail["repetition"] = 2
            if detail["view"] == "side":
                detail["purpose"] = "observation"
    requested, _ = _render_requests(monkeypatch, record, max_images=4)

    assert _decoded(requested) == [("side", 10), ("side", 20), ("front", 10), ("front", 20)]


def test_customer_sized_report_keeps_finish_diagnostic_with_default_budget(monkeypatch):
    record = _record_with_evidence()
    template = deepcopy(record["result"]["checks"][0])
    checks, catalog = [], []
    for number in range(17):
        check = deepcopy(template)
        unable = number >= 7
        phase = "finish" if number >= 12 else "motion" if unable else "start"
        label = {"start": "开始实战式", "motion": "动作过程", "finish": "动作结束瞬间"}[phase]
        check.update(
            id=f"rear_straight:{phase}:front:item{number}",
            code=f"item{number}",
            status="unable" if unable else "candidate",
            phase=phase,
            phaseLabel=label,
            reason="左腕有效帧不足" if unable else "发现疑似问题",
        )
        check["evidenceDetails"] = []
        for view, purpose in (
            ("side", "diagnostic" if unable else "violation"),
            ("front", "reference"),
        ):
            frame = 100 + number
            ref = f"{view}:{frame}"
            check["evidenceDetails"].append(
                {
                    "ref": ref,
                    "checkId": check["id"],
                    "phaseLabel": label,
                    "view": view,
                    "purpose": purpose,
                    "weight": 1,
                    "highlightJoints": [11, 15],
                    "referenceFor": f"side:{frame}" if view == "front" else "",
                }
            )
            catalog.append({"id": ref, "view": view, "frame": frame, "timeSeconds": frame / 30})
        check["evidence"] = [detail["ref"] for detail in check["evidenceDetails"]]
        checks.append(check)
    record["result"]["checks"] = checks
    record["result"]["evidence"] = catalog

    requested, rendered = _render_requests(monkeypatch, record)

    assert len(requested) == 12
    assert ("side", 112) in _decoded(requested)
    assert "动作结束瞬间 · 侧面未能判定的取帧 · 3.733s · 第112帧（图片数量" not in rendered
    assert "动作结束瞬间 · 侧面未能判定的取帧 · 3.733s · 第112帧" in rendered
