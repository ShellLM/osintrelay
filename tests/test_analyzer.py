"""
Tests for socialosintagent/analyzer.py

Covers:
- analyze() orchestration happy path.
- FetchResult unit tests (add_success, add_failure, add_rate_limit, has_any_data,
  get_summary), analyze() returns error dict when every fetch fails, offline mode skips
  vision analysis, process_stdin rejects missing query (exit 1), process_stdin rejects
  missing platforms (exit 1), process_stdin exits 2 on analysis error.
"""

import argparse
import json
import sys
from io import StringIO
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from socialosintagent.analyzer import FetchResult, SocialOSINTAgent
from socialosintagent.image_processor import ImageProcessingResult, ProcessingStatus
from socialosintagent.cache import CacheManager
from socialosintagent.client_manager import ClientManager
from socialosintagent.exceptions import UserNotFoundError
from socialosintagent.llm import LLMAnalyzer
from socialosintagent.utils import UserData

# Fixtures

@pytest.fixture
def mock_dependencies(mocker):
    """Provides mocked versions of the agent's dependencies."""
    mock_cache = create_autospec(CacheManager, instance=True)
    with patch("socialosintagent.llm._load_prompt", return_value="mock prompt"):
        mock_llm = create_autospec(LLMAnalyzer, instance=True)
    mock_client_manager = create_autospec(ClientManager, instance=True)
    return mock_cache, mock_llm, mock_client_manager


@pytest.fixture
def agent(mock_dependencies, monkeypatch):
    """Provides a SocialOSINTAgent instance with mocked dependencies for testing."""
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
    mock_cache, mock_llm, mock_client_manager = mock_dependencies

    agent_instance = SocialOSINTAgent(args, mock_cache, mock_llm, mock_client_manager)
    return agent_instance


@pytest.fixture
def offline_agent(mock_dependencies, monkeypatch):
    """Agent with offline=True so vision analysis is skipped."""
    monkeypatch.setenv("LLM_API_KEY", "test_key")
    monkeypatch.setenv("LLM_API_BASE_URL", "https://test.api/v1")
    monkeypatch.setenv("IMAGE_ANALYSIS_MODEL", "test_vision_model")
    monkeypatch.setenv("ANALYSIS_MODEL", "test_text_model")

    args = argparse.Namespace(
        offline=True,
        no_auto_save=True,
        format="markdown",
        unsafe_allow_external_media=False,
    )
    mock_cache, mock_llm, mock_client_manager = mock_dependencies
    return SocialOSINTAgent(args, mock_cache, mock_llm, mock_client_manager)

# TEST

def test_analyze_method_orchestration(agent, mocker):
    """Tests the main analyze() method's orchestration with the UserData model."""
    mock_user_data: UserData = {
        "profile": {"platform": "twitter", "username": "testuser", "id": "123"},
        "posts": [
            {
                "id": "t1",
                "media": [
                    {
                        "local_path": "/fake/path/image.jpg",
                        "url": "http://example.com/image.jpg",
                    }
                ],
            }
        ],
    }

    mock_fetcher = mocker.MagicMock(return_value=mock_user_data)
    mock_twitter_client = MagicMock()
    agent.client_manager.get_platform_client.return_value = mock_twitter_client

    mocker.patch("pathlib.Path.exists", return_value=True)
    path_mock = mocker.patch("socialosintagent.analyzer.Path")
    path_mock.return_value.suffix = ".jpg"
    path_mock.return_value.exists.return_value = True

    mocker.patch("socialosintagent.analyzer.FETCHERS", {"twitter": mock_fetcher})

    # Mock ImageProcessor to avoid real file system access and ensure callback execution
    mock_processor = mocker.MagicMock()
    agent.image_processor = mock_processor

    def side_effect(file_path, analyze_func, **kwargs):
        # Simulate successful analysis call
        # Extract args passed to process_single_image to satisfy the autospec signature of analyze_func
        source_url = kwargs.get("source_url", "http://example.com/image.jpg")
        context = kwargs.get("context", "")
        
        # Call the mocked analyze_func with required arguments
        res = analyze_func(file_path, source_url=source_url, context=context)
        
        return ImageProcessingResult(
            url=source_url,
            status=ProcessingStatus.SUCCESS,
            analysis=res,
            local_path=file_path
        )

    mock_processor.process_single_image.side_effect = side_effect

    agent.llm.analyze_image.return_value = "This is an image analysis."

    # run_analysis returns a (report_str, entities_dict) tuple.
    # Setting return_value to a plain string caused a ValueError at the
    # `report, entities = self.llm.run_analysis(...)` unpack in analyzer.py.
    agent.llm.run_analysis.return_value = ("This is the final report.", {})

    platforms_to_query = {"twitter": ["testuser"]}
    query = "analyze this user"

    result = agent.analyze(platforms_to_query, query, force_refresh=False)

    agent.client_manager.get_platform_client.assert_called_once_with("twitter")
    mock_fetcher.assert_called_once()

    agent.llm.analyze_image.assert_called_once()
    agent.llm.run_analysis.assert_called_once()

    assert isinstance(result, dict)
    assert result["report"].endswith("This is the final report.")
    assert not result["error"]

# FetchResult unit tests

class TestFetchResult:
    def test_initial_state(self):
        """A fresh FetchResult has empty lists and has_any_data is False."""
        fr = FetchResult()
        assert fr.successful == []
        assert fr.failed == []
        assert fr.rate_limited == []
        assert fr.has_any_data is False

    def test_add_success_populates_successful(self):
        fr = FetchResult()
        data = {"profile": {}, "posts": []}
        fr.add_success("twitter", "alice", data)

        assert fr.has_any_data is True
        assert len(fr.successful) == 1
        assert fr.successful[0] == ("twitter", "alice", data)

    def test_add_failure_populates_failed(self):
        fr = FetchResult()
        fr.add_failure("reddit", "bob", "NotFound", "User does not exist")

        assert fr.has_any_data is False
        assert len(fr.failed) == 1
        assert fr.failed[0] == ("reddit", "bob", "NotFound", "User does not exist")

    def test_add_rate_limit_populates_rate_limited(self):
        fr = FetchResult()
        fr.add_rate_limit("github", "charlie")

        assert len(fr.rate_limited) == 1
        assert fr.rate_limited[0] == ("github", "charlie")

    def test_get_summary_all_categories(self):
        fr = FetchResult()
        fr.add_success("twitter", "a", {})
        fr.add_success("reddit", "b", {})
        fr.add_failure("github", "c", "Err", "msg")
        fr.add_rate_limit("bluesky", "d")

        summary = fr.get_summary()
        assert "2 successful" in summary
        assert "1 failed" in summary
        assert "1 rate-limited" in summary

    def test_get_summary_empty(self):
        fr = FetchResult()
        assert fr.get_summary() == "no results"

# analyze() error paths

class TestAnalyzeErrorPaths:
    def test_all_fetches_fail_returns_error_dict(self, agent, mocker):
        """When every platform fetch raises UserNotFoundError, the result is an error."""

        def failing_fetcher(**kwargs):
            raise UserNotFoundError("not found")

        mocker.patch(
            "socialosintagent.analyzer.FETCHERS", {"hackernews": failing_fetcher}
        )
        agent.client_manager.get_platform_client.return_value = None

        result = agent.analyze({"hackernews": ["nobody"]}, "find info")

        assert result["error"] is True
        assert "failed" in result["report"].lower()

    def test_offline_mode_skips_vision_and_still_reports(self, offline_agent, mocker):
        """In offline mode, vision analysis is skipped but LLM synthesis still runs."""
        mock_data: UserData = {
            "profile": {"platform": "hackernews", "username": "coder", "id": "1"},
            "posts": [
                {
                    "id": "h1",
                    "text": "Great talk at the conference.",
                    "media": [
                        {
                            "local_path": "/fake/image.jpg",
                            "url": "http://example.com/img.jpg",
                        }
                    ],
                }
            ],
        }

        def fetcher(**kwargs):
            return mock_data

        mocker.patch(
            "socialosintagent.analyzer.FETCHERS", {"hackernews": fetcher}
        )
        offline_agent.client_manager.get_platform_client.return_value = None

        # run_analysis returns a (report_str, entities_dict) tuple.
        # The plain string "Offline report content." would cause a ValueError
        # at the `report, entities = self.llm.run_analysis(...)` unpack.
        offline_agent.llm.run_analysis.return_value = ("Offline report content.", {})

        result = offline_agent.analyze({"hackernews": ["coder"]}, "summarize")

        # Vision analysis must NOT have been called
        offline_agent.llm.analyze_image.assert_not_called()
        # But LLM text synthesis must have run
        offline_agent.llm.run_analysis.assert_called_once()
        assert not result["error"]
        assert "Offline report content." in result["report"]

# process_stdin validation

class TestProcessStdin:
    def _run_stdin(self, agent, json_input):
        """Helper: patch sys.stdin and call process_stdin, returning the exit code."""
        with patch("sys.stdin", StringIO(json.dumps(json_input))):
            with pytest.raises(SystemExit) as exc_info:
                agent.process_stdin()
        return exc_info.value.code

    def test_missing_query_exits_1(self, agent):
        """A JSON payload without a 'query' field exits with code 1."""
        code = self._run_stdin(agent, {"platforms": {"hackernews": ["pg"]}})
        assert code == 1

    def test_empty_query_exits_1(self, agent):
        """A JSON payload with a blank query string exits with code 1."""
        code = self._run_stdin(
            agent, {"platforms": {"hackernews": ["pg"]}, "query": "   "}
        )
        assert code == 1

    def test_missing_platforms_exits_1(self, agent):
        """A JSON payload without 'platforms' exits with code 1."""
        code = self._run_stdin(agent, {"query": "find info"})
        assert code == 1

    def test_no_valid_configured_platforms_exits_1(self, agent):
        """When none of the requested platforms are available, exit 1."""
        # client_manager reports nothing is available
        agent.client_manager.get_available_platforms.return_value = []
        code = self._run_stdin(
            agent,
            {
                "platforms": {"twitter": ["someone"]},
                "query": "analyse",
            },
        )
        assert code == 1

    def test_analysis_error_exits_2(self, agent, mocker):
        """When analyze() returns an error result, process_stdin exits with code 2."""
        agent.client_manager.get_available_platforms.return_value = ["hackernews"]

        # Make analyze return an error dict
        agent.analyze = MagicMock(
            return_value={
                "error": True,
                "report": "Something failed",
                "metadata": {},
            }
        )

        code = self._run_stdin(
            agent,
            {
                "platforms": {"hackernews": ["pg"]},
                "query": "test query",
            },
        )
        assert code == 2