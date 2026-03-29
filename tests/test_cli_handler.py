"""
Tests for socialosintagent/cli_handler.py

Covers:
_handle_loadmore_command()
  - Specific platform/user target with count updates fetch_options and re-runs last query
  - Non-numeric count is rejected with an error message
  - Raw command string "/loadmore platform/user count" preserves the slash in the
    target after lstrip("/") parsing (regression for the replace("/","") bug)

_handle_add_command()
  - Adds a valid new target to the platforms dict
  - Uses the session default_count when no count is specified in the path
  - Stores a per-target count override when count is specified in the path
  - Prints a confirmation message on success
  - Refuses to add if the platform is not configured/available
  - Refuses to add a target that is already in the session (idempotent)
  - Rejects malformed /add input (wrong number of path segments)
  - Rejects a non-numeric count in /add platform/user/count form
  - Creates a new platform key when the platform is not yet in the session

_handle_remove_command()
  - Removes an existing target from the platforms dict
  - Cleans up the per-target fetch_options entry when removing
  - Refuses to remove the last remaining target across all platforms
  - Refuses to remove a target not in the current session
  - Rejects malformed /remove input (no slash separator)
  - Rejects a /remove command with no argument
  - Removes the platform key entirely when its last user is removed
  - Prints a confirmation message on success

_handle_status_command()
  - Calls console.print for any non-empty platforms dict
  - Does not raise when no cache files exist for any target
  - Reads the cache file and passes a rich Panel to console.print when data exists
  - Handles multi-platform sessions with multiple users per platform
"""

import argparse
import json as _json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest
from rich.panel import Panel

from socialosintagent.cli_handler import CliHandler


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.client_manager.get_available_platforms.return_value = [
        "bluesky", "github", "hackernews", "mastodon", "reddit", "twitter"
    ]
    agent.cache.load.return_value = None
    return agent


@pytest.fixture
def cli(mock_agent):
    args = argparse.Namespace(offline=False, no_auto_save=True, format="markdown")
    handler = CliHandler(mock_agent, args)
    handler.console = MagicMock()
    return handler


def _fresh_platforms():
    return {"twitter": ["alice"], "github": ["torvalds"]}


def _fresh_fetch_options():
    return {"default_count": 50, "targets": {}}


# ── _handle_loadmore_command ──────────────────────────────────────────────────

class TestHandleLoadmoreCommand:
    def test_handle_loadmore_command_specific_target(self, cli):
        """loadmore <platform/user> <count> updates fetch_options and signals re-run."""
        parts = ["loadmore", "twitter/testuser", "100"]
        platforms = {"twitter": ["testuser"]}
        fetch_options = {"default_count": 50, "targets": {}}
        should_run, query, force_refresh = cli._handle_loadmore_command(
            parts, platforms, fetch_options, "find connections"
        )
        assert should_run is True
        assert query == "find connections"
        assert force_refresh is True
        # New count = default 50 + 100 added = 150
        assert fetch_options["targets"]["twitter:testuser"]["count"] == 150

    def test_handle_loadmore_command_invalid_count(self, cli):
        """A non-numeric count prints an error and returns should_run=False."""
        parts = ["loadmore", "twitter/testuser", "invalid"]
        should_run, query, force_refresh = cli._handle_loadmore_command(
            parts, {"twitter": ["testuser"]}, {}, ""
        )
        assert should_run is False
        cli.console.print.assert_called_with("[red]Invalid count: 'invalid'.[/red]")

    def test_loadmore_raw_command_string_preserves_slash_in_target(self, cli):
        """
        Regression test for the slash-stripping bug in _run_analysis_loop().

        The original caller used `user_input.replace("/", "").split()` which
        destroyed the separator in "platform/user" targets:

            "/loadmore twitter/alice 50"
            → replace("/","") → "loadmore twitteralice 50"
            → split()        → ["loadmore", "twitteralice", "50"]

        _handle_loadmore_command then called target_str.split('/', 1) on
        "twitteralice", found no slash, and reported the target as not found.

        The fix uses `user_input.lstrip("/").split()` which removes only the
        leading slash that prefixes the command name:

            "/loadmore twitter/alice 50"
            → lstrip("/") → "loadmore twitter/alice 50"
            → split()     → ["loadmore", "twitter/alice", "50"]

        This test simulates what the fixed caller now produces and verifies the
        method succeeds for a target that contains a slash.
        """
        raw_command = "/loadmore twitter/alice 50"

        # Reproduce what the OLD (broken) caller produced — method should FAIL
        broken_parts = raw_command.replace("/", "").split()
        # broken_parts == ["loadmore", "twitteralice", "50"]
        platforms = {"twitter": ["alice"]}
        fetch_options = {"default_count": 50, "targets": {}}
        should_run_broken, _, _ = cli._handle_loadmore_command(
            broken_parts, platforms, fetch_options, "last query"
        )
        # With the broken parts the target "twitteralice" is not in the session
        assert should_run_broken is False

        # Reproduce what the FIXED caller produces — method should SUCCEED
        fixed_parts = raw_command.lstrip("/").split()
        # fixed_parts == ["loadmore", "twitter/alice", "50"]
        fetch_options_2 = {"default_count": 50, "targets": {}}
        should_run_fixed, query_fixed, refresh_fixed = cli._handle_loadmore_command(
            fixed_parts, platforms, fetch_options_2, "last query"
        )
        assert should_run_fixed is True
        assert query_fixed == "last query"
        assert refresh_fixed is True
        assert fetch_options_2["targets"]["twitter:alice"]["count"] == 100


# ── _handle_add_command ───────────────────────────────────────────────────────

class TestHandleAddCommand:
    def test_adds_valid_new_target(self, cli):
        platforms = _fresh_platforms()
        cli._handle_add_command("/add reddit/spez", platforms, _fresh_fetch_options())
        assert "reddit" in platforms
        assert "spez" in platforms["reddit"]

    def test_uses_default_count_when_none_specified(self, cli):
        platforms = _fresh_platforms()
        fetch_options = {"default_count": 75, "targets": {}}
        cli._handle_add_command("/add hackernews/pg", platforms, fetch_options)
        # No per-target override when the fetch count equals the default
        assert "hackernews:pg" not in fetch_options.get("targets", {})

    def test_stores_per_target_count_override(self, cli):
        platforms = _fresh_platforms()
        fetch_options = _fresh_fetch_options()
        cli._handle_add_command("/add hackernews/pg/100", platforms, fetch_options)
        assert fetch_options["targets"]["hackernews:pg"]["count"] == 100

    def test_prints_confirmation_on_success(self, cli):
        platforms = _fresh_platforms()
        cli._handle_add_command("/add hackernews/pg", platforms, _fresh_fetch_options())
        cli.console.print.assert_called()
        printed = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "added" in printed.lower()

    def test_refuses_unconfigured_platform(self, cli):
        platforms = _fresh_platforms()
        cli._handle_add_command("/add instagram/someone", platforms, _fresh_fetch_options())
        assert "instagram" not in platforms
        printed = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "not configured" in printed.lower() or "unavailable" in printed.lower()

    def test_refuses_duplicate_target(self, cli):
        platforms = {"twitter": ["alice"]}
        cli._handle_add_command("/add twitter/alice", platforms, _fresh_fetch_options())
        assert platforms["twitter"].count("alice") == 1
        printed = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "already" in printed.lower()

    def test_rejects_wrong_number_of_parts(self, cli):
        platforms = _fresh_platforms()
        cli._handle_add_command("/add twitter", platforms, _fresh_fetch_options())
        printed = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "invalid" in printed.lower() or "format" in printed.lower()
        assert platforms == _fresh_platforms()

    def test_rejects_non_numeric_count(self, cli):
        platforms = _fresh_platforms()
        cli._handle_add_command("/add hackernews/pg/notanumber", platforms, _fresh_fetch_options())
        assert "hackernews" not in platforms
        printed = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "invalid" in printed.lower()

    def test_creates_new_platform_key_when_needed(self, cli):
        platforms = {"twitter": ["alice"]}  # no github yet
        cli._handle_add_command("/add github/newuser", platforms, _fresh_fetch_options())
        assert "github" in platforms
        assert "newuser" in platforms["github"]


# ── _handle_remove_command ────────────────────────────────────────────────────

class TestHandleRemoveCommand:
    def test_removes_existing_target(self, cli):
        platforms = {"twitter": ["alice", "bob"], "github": ["carol"]}
        cli._handle_remove_command("/remove twitter/alice", platforms, _fresh_fetch_options())
        assert "alice" not in platforms["twitter"]
        assert "bob" in platforms["twitter"]

    def test_prints_confirmation_on_success(self, cli):
        platforms = {"twitter": ["alice", "bob"]}
        cli._handle_remove_command("/remove twitter/alice", platforms, _fresh_fetch_options())
        printed = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "removed" in printed.lower()

    def test_cleans_up_fetch_options_for_removed_target(self, cli):
        platforms = {"twitter": ["alice", "bob"]}
        fetch_options = {"default_count": 50, "targets": {"twitter:alice": {"count": 200}}}
        cli._handle_remove_command("/remove twitter/alice", platforms, fetch_options)
        assert "twitter:alice" not in fetch_options["targets"]

    def test_removes_platform_key_when_last_user_gone(self, cli):
        platforms = {"twitter": ["alice"], "github": ["carol"]}
        cli._handle_remove_command("/remove twitter/alice", platforms, _fresh_fetch_options())
        assert "twitter" not in platforms

    def test_refuses_to_remove_last_target(self, cli):
        platforms = {"twitter": ["alice"]}
        cli._handle_remove_command("/remove twitter/alice", platforms, _fresh_fetch_options())
        assert "alice" in platforms["twitter"]
        printed = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "last" in printed.lower() or "cannot" in printed.lower()

    def test_refuses_target_not_in_session(self, cli):
        platforms = {"twitter": ["alice"]}
        cli._handle_remove_command("/remove twitter/bob", platforms, _fresh_fetch_options())
        assert "alice" in platforms["twitter"]
        printed = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "not in" in printed.lower()

    def test_rejects_malformed_input_no_slash(self, cli):
        platforms = _fresh_platforms()
        cli._handle_remove_command("/remove twitteralice", platforms, _fresh_fetch_options())
        assert platforms == _fresh_platforms()
        printed = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "invalid" in printed.lower() or "format" in printed.lower()

    def test_rejects_missing_argument(self, cli):
        platforms = _fresh_platforms()
        cli._handle_remove_command("/remove", platforms, _fresh_fetch_options())
        assert platforms == _fresh_platforms()


# ── _handle_status_command ────────────────────────────────────────────────────

class TestHandleStatusCommand:
    def test_prints_something_for_valid_platforms(self, cli):
        platforms = {"twitter": ["alice"], "github": ["torvalds"]}
        cli.agent.cache.get_cache_path.return_value = MagicMock(exists=lambda: False)
        cli._handle_status_command(platforms)
        cli.console.print.assert_called()

    def test_handles_no_cached_data_without_error(self, cli):
        """Must not raise even when no cache files exist for any target."""
        platforms = {"hackernews": ["pg"]}
        cli.agent.cache.get_cache_path.return_value = MagicMock(exists=lambda: False)
        cli._handle_status_command(platforms)  # no assertion needed — just must not raise

    def test_reads_cache_file_and_prints_panel(self, cli, tmp_path):
        """When a cache file exists the command reads it and passes a rich Panel
        to console.print containing the status table."""
        platforms = {"hackernews": ["pg"]}
        cache_file = tmp_path / "hackernews_pg.json"
        ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        cache_file.write_text(_json.dumps({
            "timestamp": ts,
            "profile": {"username": "pg"},
            "posts": [{"id": str(i)} for i in range(42)],
        }), encoding="utf-8")
        cli.agent.cache.get_cache_path.return_value = cache_file
        cli._handle_status_command(platforms)
        cli.console.print.assert_called()
        # The argument passed to print must be a rich Panel wrapping the table
        assert isinstance(cli.console.print.call_args[0][0], Panel)

    def test_handles_multi_platform_multi_user_session(self, cli):
        """Multi-platform sessions with multiple users per platform must not crash."""
        platforms = {
            "twitter": ["alice", "bob"],
            "github": ["carol"],
            "hackernews": ["pg"],
        }
        cli.agent.cache.get_cache_path.return_value = MagicMock(exists=lambda: False)
        cli._handle_status_command(platforms)
        cli.console.print.assert_called()