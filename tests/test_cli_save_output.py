"""
Tests for CliHandler._save_output() — interactive mode saving.

Covers:
- Saves a .md file with correct report content
- Saves a .json file with the expected structure
- Prints a confirmation message to the console after saving
"""

import argparse
import json
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from socialosintagent.analyzer import SocialOSINTAgent
from socialosintagent.cache import CacheManager
from socialosintagent.cli_handler import CliHandler
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


@pytest.fixture
def cli_handler(monkeypatch, tmp_path):
    """CliHandler wired to a temp data directory with a mocked console."""
    monkeypatch.setenv("LLM_API_KEY", "test_key")
    monkeypatch.setenv("LLM_API_BASE_URL", "https://test.api/v1")
    monkeypatch.setenv("IMAGE_ANALYSIS_MODEL", "test_vision_model")
    monkeypatch.setenv("ANALYSIS_MODEL", "test_text_model")

    args = argparse.Namespace(
        offline=False,
        no_auto_save=False,
        format="markdown",
        unsafe_allow_external_media=False,
    )
    mock_cache = create_autospec(CacheManager, instance=True)
    with patch("socialosintagent.llm._load_prompt", return_value="mock prompt"):
        mock_llm = create_autospec(LLMAnalyzer, instance=True)
    mock_cm = create_autospec(ClientManager, instance=True)
    mock_cm.get_available_platforms.return_value = ["hackernews"]

    agent_instance = SocialOSINTAgent(args, mock_cache, mock_llm, mock_cm)
    agent_instance.base_dir = tmp_path
    (tmp_path / "outputs").mkdir(parents=True, exist_ok=True)

    cli = CliHandler(agent_instance, args)
    cli.console = MagicMock()
    cli.base_dir = tmp_path
    return cli, tmp_path


class TestCliSaveOutput:

    def test_saves_markdown_file(self, cli_handler):
        """Saves a .md file containing the report text."""
        cli, base = cli_handler
        cli._save_output(SAMPLE_RESULT, "markdown")

        outputs = list((base / "outputs").glob("*.md"))
        assert len(outputs) == 1
        assert "# OSINT Analysis Report" in outputs[0].read_text(encoding="utf-8")

    def test_saves_json_file(self, cli_handler):
        """Saves a .json file with the expected top-level keys."""
        cli, base = cli_handler
        cli._save_output(SAMPLE_RESULT, "json")

        outputs = list((base / "outputs").glob("*.json"))
        assert len(outputs) == 1
        data = json.loads(outputs[0].read_text(encoding="utf-8"))
        assert "analysis_metadata" in data
        assert "analysis_report_markdown" in data

    def test_prints_confirmation_to_console(self, cli_handler):
        """Console.print is called with a message referencing the saved path."""
        cli, _ = cli_handler
        cli._save_output(SAMPLE_RESULT, "markdown")

        cli.console.print.assert_called()
        printed = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "saved" in printed.lower() or "analysis" in printed.lower()
