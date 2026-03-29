"""
Tests for SocialOSINTAgent.process_stdin() — output and saving behaviour.

Covers:
- --no-auto-save + markdown: report printed as plain text to stdout, exits 0
- --no-auto-save + json: structured {"success", "metadata", "report"} JSON to stdout
- auto-save + markdown: file created, stdout JSON contains "output_file" pointing to a real file
- auto-save + json: .json output file created with the correct structure
- successful stdin flow always exits with code 0
"""

import argparse
import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from socialosintagent.analyzer import SocialOSINTAgent
from socialosintagent.cache import CacheManager
from socialosintagent.client_manager import ClientManager
from socialosintagent.llm import LLMAnalyzer


SAMPLE_RESULT = {
    "error": False,
    "report": "# OSINT Analysis Report\n\nThis is the report body.",
    "metadata": {
        "query": "test query",
        "targets": {"hackernews": ["pg"]},
        "generated_utc": "2026-01-01 00:00:00 UTC",
        "mode": "Online",
        "models": {"text": "gpt-4", "image": "gpt-4v"},
        "fetch_stats": {"successful": 1, "failed": 0, "rate_limited": 0},
        "vision_stats": {},
    },
}

STDIN_PAYLOAD = {"platforms": {"hackernews": ["pg"]}, "query": "test query"}


def _make_agent(monkeypatch, tmp_path, no_auto_save=True, fmt="markdown"):
    """Helper: build a fully mocked SocialOSINTAgent pointed at tmp_path."""
    monkeypatch.setenv("LLM_API_KEY", "test_key")
    monkeypatch.setenv("LLM_API_BASE_URL", "https://test.api/v1")
    monkeypatch.setenv("IMAGE_ANALYSIS_MODEL", "test_vision_model")
    monkeypatch.setenv("ANALYSIS_MODEL", "test_text_model")

    args = argparse.Namespace(
        offline=False,
        no_auto_save=no_auto_save,
        format=fmt,
        unsafe_allow_external_media=False,
    )
    mock_cache = create_autospec(CacheManager, instance=True)
    with patch("socialosintagent.llm._load_prompt", return_value="mock prompt"):
        mock_llm = create_autospec(LLMAnalyzer, instance=True)
    mock_cm = create_autospec(ClientManager, instance=True)
    mock_cm.get_available_platforms.return_value = ["hackernews"]

    agent = SocialOSINTAgent(args, mock_cache, mock_llm, mock_cm)
    agent.base_dir = tmp_path
    (tmp_path / "outputs").mkdir(parents=True, exist_ok=True)
    agent.analyze = MagicMock(return_value=SAMPLE_RESULT)
    return agent


def _run(agent, payload):
    """Feed payload JSON into process_stdin and return (exit_code, captured)."""
    with patch("sys.stdin", StringIO(json.dumps(payload))):
        with pytest.raises(SystemExit) as exc_info:
            agent.process_stdin()
    return exc_info.value.code


class TestProcessStdinOutput:

    def test_no_auto_save_markdown_prints_report_to_stdout(
        self, monkeypatch, tmp_path, capsys
    ):
        """--no-auto-save + markdown: raw report text printed to stdout."""
        agent = _make_agent(monkeypatch, tmp_path, no_auto_save=True, fmt="markdown")
        code = _run(agent, STDIN_PAYLOAD)

        assert code == 0
        assert "# OSINT Analysis Report" in capsys.readouterr().out

    def test_no_auto_save_json_format_prints_json_to_stdout(
        self, monkeypatch, tmp_path, capsys
    ):
        """--no-auto-save + json: structured JSON with success/metadata/report printed."""
        agent = _make_agent(monkeypatch, tmp_path, no_auto_save=True, fmt="json")
        code = _run(agent, STDIN_PAYLOAD)

        assert code == 0
        parsed = json.loads(capsys.readouterr().out)
        assert parsed["success"] is True
        assert "metadata" in parsed
        assert "report" in parsed

    def test_auto_save_writes_markdown_file_and_prints_path_json(
        self, monkeypatch, tmp_path, capsys
    ):
        """auto-save + markdown: file created, stdout JSON has output_file pointing to it."""
        agent = _make_agent(monkeypatch, tmp_path, no_auto_save=False, fmt="markdown")
        code = _run(agent, STDIN_PAYLOAD)

        assert code == 0
        parsed = json.loads(capsys.readouterr().out)
        assert parsed["success"] is True
        assert "output_file" in parsed

        output_path = Path(parsed["output_file"])
        assert output_path.exists(), f"Expected output file at {output_path}"
        assert "# OSINT Analysis Report" in output_path.read_text(encoding="utf-8")

    def test_auto_save_json_format_writes_json_file(
        self, monkeypatch, tmp_path, capsys
    ):
        """auto-save + json: .json file created with analysis_metadata and report keys."""
        agent = _make_agent(monkeypatch, tmp_path, no_auto_save=False, fmt="json")
        code = _run(agent, STDIN_PAYLOAD)

        assert code == 0
        parsed = json.loads(capsys.readouterr().out)
        output_path = Path(parsed["output_file"])
        assert output_path.suffix == ".json"

        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert "analysis_metadata" in saved
        assert "analysis_report_markdown" in saved

    def test_successful_stdin_exits_0(self, monkeypatch, tmp_path):
        """A valid stdin request always exits with code 0."""
        agent = _make_agent(monkeypatch, tmp_path, no_auto_save=True, fmt="markdown")
        assert _run(agent, STDIN_PAYLOAD) == 0
