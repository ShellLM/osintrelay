"""
Tests for SocialOSINTAgent._save_output_headless().

Covers:
- Creates a .md file containing the full report text
- Creates a .json file with the expected top-level keys and content
- Filename embeds the platform name and a query slug
- Saved JSON is valid and parseable
- Markdown file content matches the report string exactly
"""

import argparse
import json
from pathlib import Path
from unittest.mock import create_autospec, patch

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


@pytest.fixture
def agent(monkeypatch, tmp_path):
    """SocialOSINTAgent wired to a temp data directory."""
    monkeypatch.setenv("LLM_API_KEY", "test_key")
    monkeypatch.setenv("LLM_API_BASE_URL", "https://test.api/v1")
    monkeypatch.setenv("IMAGE_ANALYSIS_MODEL", "test_vision_model")
    monkeypatch.setenv("ANALYSIS_MODEL", "test_text_model")

    args = argparse.Namespace(
        offline=False,
        no_auto_save=True,
        format="markdown",
        unsafe_allow_external_media=False,
    )
    mock_cache = create_autospec(CacheManager, instance=True)
    with patch("socialosintagent.llm._load_prompt", return_value="mock prompt"):
        mock_llm = create_autospec(LLMAnalyzer, instance=True)
    mock_cm = create_autospec(ClientManager, instance=True)
    mock_cm.get_available_platforms.return_value = ["hackernews"]

    instance = SocialOSINTAgent(args, mock_cache, mock_llm, mock_cm)
    instance.base_dir = tmp_path
    (tmp_path / "outputs").mkdir(parents=True, exist_ok=True)
    return instance


class TestSaveOutputHeadless:

    def test_saves_markdown_file(self, agent):
        """Creates a .md file containing the report text."""
        path = agent._save_output_headless(SAMPLE_RESULT, "markdown")

        assert path.exists(), "Expected output file to exist"
        assert path.suffix == ".md"
        content = path.read_text(encoding="utf-8")
        assert "# OSINT Analysis Report" in content
        assert "This is the report body." in content

    def test_markdown_file_contains_full_report(self, agent):
        """The markdown file content matches the report string exactly."""
        path = agent._save_output_headless(SAMPLE_RESULT, "markdown")
        assert path.read_text(encoding="utf-8") == SAMPLE_RESULT["report"]

    def test_saves_json_file(self, agent):
        """Creates a .json file with the expected top-level keys and content."""
        path = agent._save_output_headless(SAMPLE_RESULT, "json")

        assert path.exists()
        assert path.suffix == ".json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "analysis_metadata" in data
        assert "analysis_report_markdown" in data
        assert data["analysis_metadata"]["query"] == "test query"
        assert "# OSINT Analysis Report" in data["analysis_report_markdown"]

    def test_json_file_is_valid_json(self, agent):
        """The saved JSON file is parseable without errors."""
        path = agent._save_output_headless(SAMPLE_RESULT, "json")
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            pytest.fail(f"Saved JSON is not valid: {exc}")

    def test_filename_contains_platform_and_query_slug(self, agent):
        """The filename embeds the platform name and a truncated query slug."""
        path = agent._save_output_headless(SAMPLE_RESULT, "markdown")
        assert "hackernews" in path.name
        assert "test" in path.name
