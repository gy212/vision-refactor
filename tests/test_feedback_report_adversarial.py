# -*- coding: utf-8 -*-
"""Adversarial stress-test harness and security challenge for HTML report export in core/feedback_report.py.

Empirical verification of:
1. XSS immunity and attribute breakout resistance across all user and dynamic fields.
2. Windows and Unix absolute path desensitization across all fields and path types.
3. Offline 100% self-contained compliance (zero https?:// external requests).
4. Large evidence simulation (Base64 embedding cap, memory and size limits).
5. Corrupt file and malformed record resilience (graceful degradation without tracebacks).
"""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
import pytest

from core.feedback_report import (
    render_html_report,
    build_report_model,
    desensitize_paths,
    generate_phase_timeline_svg,
)
from tests.test_feedback_html_export import make_html_test_record


ADVERSARIAL_XSS_PAYLOADS = [
    '<script>alert(1)</script>',
    '<img src=x onerror=alert(1)>',
    '"><svg onload=alert(1)>',
    'javascript:void(0)',
    'javascript:alert(1)',
    '"><script>alert(document.domain)</script>',
    '</title><script>alert(1)</script>',
    '</style><script>alert(1)</script>',
    '"><input onfocus=alert(1) autofocus>',
    '<iframe src="javascript:alert(1)">',
    '<body onload=alert(1)>',
    '--> <script>alert(1)</script>',
    '\"><svg onload=alert(1)>',
    '"><div style="background-image: url(javascript:alert(1))">',
    '&lt;script&gt;alert(1)&lt;/script&gt;',
    '\' onmouseover=\'alert(1)\'',
    '" onfocus="alert(1)"',
    '<a href="javascript:alert(1)">click me</a>',
    '<svg><script xlink:href="data:text/javascript,alert(1)"/>',
    '"><details open ontoggle=alert(1)>',
]


RAW_WINDOWS_SECRET_PATH = r"D:\CodeProject\vision\outputs\practice_feedback\secret\video.mp4"
RAW_WINDOWS_SPACED_PATH = r"C:\Program Files\vision\outputs\practice_feedback\secret\video.mp4"
RAW_WINDOWS_USER_PATH = r"C:\Users\John Doe\AppData\Local\secret\video.mp4"
RAW_UNC_PATH = r"\\server\share\secret\video.mp4"
RAW_UNIX_PATH = "/home/ubuntu/secret/video.mp4"


class XSSAuditParser(HTMLParser):
    """Parses rendered HTML to detect any unescaped tags or active JavaScript event handlers."""

    def __init__(self):
        super().__init__()
        self.dangerous_tags: list[str] = []
        self.dangerous_attributes: list[tuple[str, str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        tag_lower = tag.lower()
        if tag_lower in {"script", "iframe", "object", "embed", "applet"}:
            self.dangerous_tags.append(tag)
        for attr, val in attrs:
            attr_lower = attr.lower()
            val_str = str(val or "").lower()
            if attr_lower.startswith("on"):
                self.dangerous_attributes.append((tag, attr, val or ""))
            elif val_str.startswith("javascript:") or "javascript:" in val_str:
                self.dangerous_attributes.append((tag, attr, val or ""))


def assert_no_xss_in_html(html_content: str):
    """Strictly assert that no dangerous tags or executable event handlers exist in HTML DOM."""
    parser = XSSAuditParser()
    parser.feed(html_content)
    assert len(parser.dangerous_tags) == 0, f"XSS Failure: Found dangerous unescaped tags: {parser.dangerous_tags}"
    assert len(parser.dangerous_attributes) == 0, f"XSS Failure: Found dangerous executable attributes: {parser.dangerous_attributes}"


# ===========================================================================
# 1. Adversarial XSS & Attribute Breakout Vectors
# ===========================================================================

class TestAdversarialXSS:
    """Stress-test all entry points for potential unescaped HTML/script execution."""

    @pytest.mark.parametrize("payload", ADVERSARIAL_XSS_PAYLOADS)
    def test_student_name_xss(self, payload: str):
        record = make_html_test_record(student_name=payload)
        out = render_html_report(record)
        assert_no_xss_in_html(out)

    @pytest.mark.parametrize("payload", ADVERSARIAL_XSS_PAYLOADS)
    def test_student_id_xss(self, payload: str):
        record = make_html_test_record(student_id=payload)
        out = render_html_report(record)
        assert_no_xss_in_html(out)

    @pytest.mark.parametrize("payload", ADVERSARIAL_XSS_PAYLOADS)
    def test_check_standard_xss(self, payload: str):
        record = make_html_test_record()
        record["result"]["checks"][0]["standard"] = payload
        out = render_html_report(record)
        assert_no_xss_in_html(out)

    @pytest.mark.parametrize("payload", ADVERSARIAL_XSS_PAYLOADS)
    def test_check_reason_xss(self, payload: str):
        record = make_html_test_record()
        record["result"]["checks"][0]["reason"] = payload
        out = render_html_report(record)
        assert_no_xss_in_html(out)

    @pytest.mark.parametrize("payload", ADVERSARIAL_XSS_PAYLOADS)
    def test_check_name_and_body_part_xss(self, payload: str):
        record = make_html_test_record()
        record["result"]["checks"][0]["name"] = payload
        record["result"]["checks"][0]["bodyPart"] = payload
        out = render_html_report(record)
        assert_no_xss_in_html(out)

    @pytest.mark.parametrize("payload", ADVERSARIAL_XSS_PAYLOADS)
    def test_action_and_stance_xss(self, payload: str):
        record = make_html_test_record(action=payload, stance=payload)
        out = render_html_report(record)
        assert_no_xss_in_html(out)

    @pytest.mark.parametrize("payload", ADVERSARIAL_XSS_PAYLOADS)
    def test_review_history_xss(self, payload: str):
        record = make_html_test_record()
        record["result"]["reviewHistory"] = [
            {
                "checkId": "front_straight:start:front:stance_elbow",
                "teacher": payload,
                "after": "confirmed",
                "at": "2026-09-26T14:00:00Z",
                "reason": payload,
            }
        ]
        out = render_html_report(record)
        assert_no_xss_in_html(out)


# ===========================================================================
# 2. Path Desensitization & Privacy Leak Vectors
# ===========================================================================

class TestPathDesensitization:
    """Challenge path desensitization across all fields and path types."""

    def test_desensitize_paths_unit_regex(self):
        """Test the pure desensitize_paths regex directly."""
        assert desensitize_paths(RAW_WINDOWS_SECRET_PATH) == "[已脱敏路径]"
        assert desensitize_paths(RAW_UNIX_PATH) == "[已脱敏路径]"

    def test_path_in_check_reason_desensitized(self):
        record = make_html_test_record()
        record["result"]["checks"][0]["reason"] = f"视频参考自 {RAW_WINDOWS_SECRET_PATH}"
        out = render_html_report(record)
        assert RAW_WINDOWS_SECRET_PATH not in out
        assert "[已脱敏路径]" in out

    def test_path_in_check_standard_desensitized(self):
        record = make_html_test_record()
        record["result"]["checks"][0]["standard"] = f"标准参见 {RAW_WINDOWS_SECRET_PATH}"
        out = render_html_report(record)
        assert RAW_WINDOWS_SECRET_PATH not in out
        assert "[已脱敏路径]" in out

    def test_path_with_spaces_leakage_discovery(self):
        """Verify desensitize_paths properly desensitizes paths with spaces."""
        res_spaced = desensitize_paths(RAW_WINDOWS_SPACED_PATH)
        res_user = desensitize_paths(RAW_WINDOWS_USER_PATH)
        assert res_spaced == "[已脱敏路径]"
        assert res_user == "[已脱敏路径]"

    def test_unc_path_leakage_discovery(self):
        """Verify desensitize_paths properly desensitizes UNC network paths."""
        res_unc = desensitize_paths(RAW_UNC_PATH)
        assert res_unc == "[已脱敏路径]"

    def test_path_in_student_name_leaks(self):
        """Verify student_name is passed through desensitize_paths."""
        record = make_html_test_record(student_name=RAW_WINDOWS_SECRET_PATH)
        out = render_html_report(record)
        assert RAW_WINDOWS_SECRET_PATH not in out
        assert "[已脱敏路径]" in out

    def test_path_in_student_id_leaks(self):
        """Verify student_id is passed through desensitize_paths."""
        record = make_html_test_record(student_id=RAW_WINDOWS_SECRET_PATH)
        out = render_html_report(record)
        assert RAW_WINDOWS_SECRET_PATH not in out
        assert "[已脱敏路径]" in out

    def test_path_in_check_name_leaks(self):
        """Verify check.name is passed through desensitize_paths."""
        record = make_html_test_record()
        record["result"]["checks"][0]["name"] = f"异常文件 {RAW_WINDOWS_SECRET_PATH}"
        out = render_html_report(record)
        assert RAW_WINDOWS_SECRET_PATH not in out
        assert "[已脱敏路径]" in out

    def test_path_in_review_teacher_leaks(self):
        """Verify reviewHistory teacher is passed through desensitize_paths."""
        record = make_html_test_record()
        record["result"]["reviewHistory"] = [
            {
                "checkId": "front_straight:start:front:stance_elbow",
                "teacher": RAW_WINDOWS_SECRET_PATH,
                "after": "confirmed",
                "at": "2026-09-26T14:00:00Z",
                "reason": "通过",
            }
        ]
        out = render_html_report(record)
        assert RAW_WINDOWS_SECRET_PATH not in out
        assert "[已脱敏路径]" in out


# ===========================================================================
# 3. Offline Zero-Network Compliance
# ===========================================================================

class TestOfflineZeroNetwork:
    """Verify zero external network calls or URL references across rendered HTML."""

    def test_rendered_html_has_strictly_zero_http_links(self):
        record = make_html_test_record()
        out = render_html_report(record)
        hits = re.findall(r"https?://", out, re.IGNORECASE)
        assert len(hits) == 0, f"Found unexpected external URLs: {hits}"

    def test_no_external_resources_in_svg(self):
        record = make_html_test_record()
        model = build_report_model(record)
        svg = generate_phase_timeline_svg(model.phases, model.counts)
        hits = re.findall(r"https?://", svg, re.IGNORECASE)
        assert len(hits) == 0, f"Found external URLs in SVG: {hits}"
        assert "xmlns" not in svg  # verify no external schema urls


# ===========================================================================
# 4. Large Evidence Simulation & Memory Robustness
# ===========================================================================

class TestLargeEvidenceSimulation:
    """Verify Base64 embedding cap and large evidence handling."""

    def test_max_images_capping_and_thumbnail_downsampling(self, tmp_path: Path):
        history_root = tmp_path / "history"
        rec_id = "e" * 32
        rec_dir = history_root / rec_id
        rec_dir.mkdir(parents=True)

        # Create a video with large frames (1920x1080)
        vid_file = rec_dir / "front.mp4"
        writer = cv2.VideoWriter(str(vid_file), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1920, 1080))
        for i in range(20):
            frame = np.full((1080, 1920, 3), (i * 12) % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        record = make_html_test_record()
        record["id"] = rec_id
        record["videos"] = {"front": "front.mp4"}

        # Add 30 evidence items and 30 checks
        evidence_list = []
        checks_list = []
        for i in range(30):
            ev_id = f"front:{i % 20}"
            evidence_list.append({
                "id": ev_id,
                "view": "front",
                "frame": i % 20,
                "timeSeconds": i * 0.1,
                "landmarks": [[0.5, 0.5, 0.0, 1.0]] * 33,
                "validMask": [True] * 33,
            })
            checks_list.append({
                "id": f"check_large_{i}",
                "code": "stance_elbow",
                "name": f"大负荷测试项_{i}",
                "bodyPart": "手臂",
                "segment": "front_straight",
                "segmentLabel": "前手直拳",
                "phase": "motion",
                "phaseLabel": "动作过程",
                "standard": "测试标准",
                "status": "candidate",
                "review": "pending",
                "reason": f"疑似异常_{i}",
                "evidence": [ev_id],
            })

        record["result"]["evidence"] = evidence_list
        record["result"]["checks"] = checks_list

        # Render with max_images=8
        max_imgs = 8
        out = render_html_report(record, history_root=history_root, max_images=max_imgs)

        # Count embedded data URIs
        b64_matches = re.findall(r"data:image/jpeg;base64,[A-Za-z0-9+/=]+", out)
        assert len(b64_matches) == max_imgs, f"Expected {max_imgs} embedded images, got {len(b64_matches)}"

        # Verify omitted notice exists
        assert "已省略次要截图以精简报告体积" in out

        # Verify total file size is well under 500 KB
        html_size_kb = len(out.encode("utf-8")) / 1024
        assert html_size_kb < 500


# ===========================================================================
# 5. Missing / Corrupt Video & Malformed Record Robustness
# ===========================================================================

class TestCorruptAndMissingFiles:
    """Verify graceful degradation when media or metadata are corrupt or missing."""

    def test_missing_video_file_graceful(self, tmp_path: Path):
        history_root = tmp_path / "empty_dir"
        history_root.mkdir(parents=True)
        record = make_html_test_record()
        record["id"] = "a" * 32
        record["videos"] = {"front": "nonexistent.mp4"}
        out = render_html_report(record, history_root=history_root)
        assert "原录像文件不可用" in out

    def test_zero_byte_video_file_graceful(self, tmp_path: Path):
        history_root = tmp_path / "history"
        rec_id = "b" * 32
        rec_dir = history_root / rec_id
        rec_dir.mkdir(parents=True)
        (rec_dir / "front.mp4").write_bytes(b"")

        record = make_html_test_record()
        record["id"] = rec_id
        record["videos"] = {"front": "front.mp4"}
        out = render_html_report(record, history_root=history_root)
        assert "截帧解码失败" in out or "原录像文件不可用" in out

    def test_corrupt_binary_video_file_graceful(self, tmp_path: Path):
        history_root = tmp_path / "history"
        rec_id = "c" * 32
        rec_dir = history_root / rec_id
        rec_dir.mkdir(parents=True)
        (rec_dir / "front.mp4").write_bytes(b"\x00\x01\x02\x03\xFF\xFE\xFD" * 1000)

        record = make_html_test_record()
        record["id"] = rec_id
        record["videos"] = {"front": "front.mp4"}
        out = render_html_report(record, history_root=history_root)
        assert "截帧解码失败" in out or "原录像文件不可用" in out

    def test_directory_instead_of_video_graceful(self, tmp_path: Path):
        history_root = tmp_path / "history"
        rec_id = "d" * 32
        rec_dir = history_root / rec_id
        rec_dir.mkdir(parents=True)
        (rec_dir / "front.mp4").mkdir()

        record = make_html_test_record()
        record["id"] = rec_id
        record["videos"] = {"front": "front.mp4"}
        out = render_html_report(record, history_root=history_root)
        assert "原录像文件不可用" in out


class TestMalformedRecordRobustness:
    """Verify graceful handling without exceptions when encountering None or malformed types."""

    def test_null_student_id_graceful(self):
        record = make_html_test_record()
        record["studentId"] = None
        out = render_html_report(record)
        assert "未登记学号" in out

    def test_null_rule_version_graceful(self):
        record = make_html_test_record()
        record["result"]["ruleVersion"] = None
        out = render_html_report(record)
        assert out is not None

    def test_null_check_name_graceful(self):
        record = make_html_test_record()
        record["result"]["checks"][0]["name"] = None
        out = render_html_report(record)
        assert "未命名检查项" in out

    def test_null_check_body_part_graceful(self):
        record = make_html_test_record()
        record["result"]["checks"][0]["bodyPart"] = None
        out = render_html_report(record)
        assert "未指定部位" in out

    def test_null_review_history_fields_graceful(self):
        record = make_html_test_record()
        record["result"]["reviewHistory"] = [
            {
                "checkId": "front_straight:start:front:stance_elbow",
                "teacher": None,
                "after": "confirmed",
                "at": None,
                "reason": None,
            }
        ]
        out = render_html_report(record)
        assert out is not None

    def test_string_source_fps_graceful(self):
        record = make_html_test_record()
        record["result"]["capture"]["front"]["sourceFps"] = "invalid_fps"
        out = render_html_report(record)
        assert "0.0fps" in out

    def test_string_side_offset_seconds_graceful(self):
        record = make_html_test_record()
        record["result"]["alignment"] = {
            "method": "dtw",
            "sideOffsetSeconds": "invalid_offset",
            "reliable": True,
        }
        out = render_html_report(record)
        assert "0.000s" in out

    def test_string_revision_graceful(self):
        record = make_html_test_record()
        record["revision"] = "not_an_int"
        out = render_html_report(record)
        assert out is not None
