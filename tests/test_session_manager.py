"""
Tests for socialosintagent/session_manager.py

Covers:
Session model
  - Defaults: dismissed_contacts is [], query_history is [], fetch_options has defaults
  - add_query_result appends entry, returns a query_id, updates updated_at
  - Multiple add_query_result calls accumulate correctly
  - to_dict / from_dict round-trip preserves all fields
  - from_dict on legacy data (no dismissed_contacts key) defaults to []
  - summary() returns lightweight dict, correct target_count, no report content

SessionManager
  - create() persists session file inside sessions_dir and returns Session
  - load() returns None for unknown session_id
  - load() returns None for corrupt JSON
  - save() + load() round-trip preserves full session state
  - delete() removes the file and returns True
  - delete() returns False for unknown session_id
  - list_all() returns summaries sorted newest-updated first
  - list_all() silently skips corrupt files
  - rename() updates name and persists
  - rename() returns None for unknown session_id
  - update_targets() replaces platforms and persists
  - update_targets() replaces fetch_options when provided
  - update_targets() preserves fetch_options when not provided
  - update_targets() returns None for unknown session_id
  - _session_path() sanitises session_id to prevent path traversal
"""

import time
from pathlib import Path

import pytest

from socialosintagent.session_manager import Session, SessionManager


# ── Session model ────────────────────────────────────────────────────────────

class TestSession:
    def test_defaults(self):
        s = Session("s1", "Test", {"twitter": ["alice"]})
        assert s.dismissed_contacts == []
        assert s.query_history == []
        assert s.fetch_options == {"default_count": 50, "targets": {}}
        assert s.monitoring_rules == []

    def test_add_query_result_appends_and_returns_id(self):
        s = Session("s1", "Test", {"twitter": ["alice"]})
        qid = s.add_query_result("What are they posting?", "# Report\n\nContent.", {"mode": "Online"})
        assert isinstance(qid, str) and len(qid) > 0
        assert len(s.query_history) == 1
        assert s.query_history[0]["query_id"] == qid
        assert s.query_history[0]["query"] == "What are they posting?"
        assert s.query_history[0]["report"] == "# Report\n\nContent."

    def test_add_query_result_updates_updated_at(self):
        s = Session("s1", "Test", {"twitter": ["alice"]})
        before = s.updated_at
        time.sleep(0.02)
        s.add_query_result("q", "r", {})
        assert s.updated_at > before

    def test_multiple_queries_accumulate(self):
        s = Session("s1", "Test", {"twitter": ["alice"]})
        s.add_query_result("q1", "r1", {})
        s.add_query_result("q2", "r2", {})
        assert len(s.query_history) == 2
        assert s.query_history[0]["query"] == "q1"
        assert s.query_history[1]["query"] == "q2"

    def test_to_dict_from_dict_round_trip(self):
        s = Session(
            "abc-123", "Investigation Alpha",
            {"github": ["torvalds"], "twitter": ["tux"]},
            fetch_options={"default_count": 75, "targets": {}}
        )
        s.dismissed_contacts = ["twitter/spammer"]
        s.add_query_result("What repos?", "## Report body", {"mode": "Online"})

        restored = Session.from_dict(s.to_dict())
        assert restored.session_id == "abc-123"
        assert restored.name == "Investigation Alpha"
        assert restored.platforms == {"github": ["torvalds"], "twitter": ["tux"]}
        assert restored.fetch_options["default_count"] == 75
        assert restored.dismissed_contacts == ["twitter/spammer"]
        assert len(restored.query_history) == 1
        assert restored.query_history[0]["query"] == "What repos?"
        assert restored.query_history[0]["report"] == "## Report body"

    def test_from_dict_legacy_no_dismissed_contacts(self):
        """Old session files without dismissed_contacts key must deserialise safely."""
        old = {
            "session_id": "old-1", "name": "Legacy",
            "platforms": {"hackernews": ["pg"]},
            "fetch_options": {"default_count": 50, "targets": {}},
            "query_history": [],
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            # no "dismissed_contacts" key
        }
        s = Session.from_dict(old)
        assert s.dismissed_contacts == []
        assert s.monitoring_rules == []

    def test_summary_omits_query_history_report_content(self):
        s = Session("s1", "Test", {"twitter": ["alice"]})
        s.add_query_result("q", "big report text that must not leak", {})
        summary = s.summary()
        assert "session_id" in summary
        assert "name" in summary
        assert "query_count" in summary
        assert summary["query_count"] == 1
        assert "big report text" not in str(summary)

    def test_summary_target_count_is_correct(self):
        s = Session("s1", "Test", {"twitter": ["alice", "bob"], "github": ["carol"]})
        assert s.summary()["target_count"] == 3

    def test_summary_with_no_queries_has_zero_count(self):
        s = Session("s1", "Test", {"twitter": ["alice"]})
        assert s.summary()["query_count"] == 0


# ── SessionManager ───────────────────────────────────────────────────────────

class TestSessionManager:
    @pytest.fixture
    def sm(self, tmp_path):
        return SessionManager(tmp_path)

    def test_create_returns_session_and_persists_in_sessions_dir(self, sm):
        s = sm.create("Op Falcon", {"twitter": ["hawk"]})
        assert s.session_id
        assert s.name == "Op Falcon"
        # File must live inside sessions_dir, not the base dir
        assert sm._session_path(s.session_id).exists()

    def test_load_returns_none_for_unknown_id(self, sm):
        assert sm.load("nonexistent-id") is None

    def test_load_returns_none_for_corrupt_json(self, sm):
        bad_path = sm.sessions_dir / "corrupt-id.json"
        bad_path.write_text("{not valid json", encoding="utf-8")
        assert sm.load("corrupt-id") is None

    def test_save_then_load_round_trip(self, sm):
        s = sm.create("Round Trip", {"reddit": ["spez"]})
        s.dismissed_contacts = ["reddit/troll"]
        s.add_query_result("test query", "test report", {"mode": "Online"})
        sm.save(s)
        loaded = sm.load(s.session_id)
        assert loaded.name == "Round Trip"
        assert loaded.dismissed_contacts == ["reddit/troll"]
        assert len(loaded.query_history) == 1
        assert loaded.query_history[0]["report"] == "test report"

    def test_delete_removes_file_and_returns_true(self, sm):
        s = sm.create("To Delete", {"hackernews": ["pg"]})
        assert sm._session_path(s.session_id).exists()
        assert sm.delete(s.session_id) is True
        assert not sm._session_path(s.session_id).exists()
        assert sm.load(s.session_id) is None

    def test_delete_returns_false_for_unknown_id(self, sm):
        assert sm.delete("ghost-session") is False

    def test_list_all_returns_summaries(self, sm):
        sm.create("Session A", {"twitter": ["a"]})
        sm.create("Session B", {"twitter": ["b"]})
        results = sm.list_all()
        assert len(results) == 2
        names = {r["name"] for r in results}
        assert "Session A" in names
        assert "Session B" in names

    def test_list_all_sorted_newest_updated_first(self, sm):
        sm.create("Old Session", {"twitter": ["old"]})
        time.sleep(0.05)
        sm.create("New Session", {"twitter": ["new"]})
        results = sm.list_all()
        assert results[0]["name"] == "New Session"
        assert results[1]["name"] == "Old Session"

    def test_list_all_skips_corrupt_files(self, sm):
        sm.create("Good Session", {"twitter": ["good"]})
        (sm.sessions_dir / "bad-file.json").write_text("{{broken", encoding="utf-8")
        results = sm.list_all()
        assert len(results) == 1
        assert results[0]["name"] == "Good Session"

    def test_list_all_empty_when_no_sessions(self, sm):
        assert sm.list_all() == []

    def test_rename_updates_name_and_persists(self, sm):
        s = sm.create("Original Name", {"twitter": ["alice"]})
        updated = sm.rename(s.session_id, "New Name")
        assert updated.name == "New Name"
        assert sm.load(s.session_id).name == "New Name"

    def test_rename_returns_none_for_unknown_session(self, sm):
        assert sm.rename("ghost-id", "New Name") is None

    def test_update_targets_replaces_platforms_and_persists(self, sm):
        s = sm.create("Targets Test", {"twitter": ["alice"]})
        updated = sm.update_targets(s.session_id, {"github": ["torvalds"], "reddit": ["spez"]})
        assert updated.platforms == {"github": ["torvalds"], "reddit": ["spez"]}
        reloaded = sm.load(s.session_id)
        assert "github" in reloaded.platforms
        assert "twitter" not in reloaded.platforms

    def test_update_targets_replaces_fetch_options_when_provided(self, sm):
        s = sm.create("FO Replace", {"twitter": ["alice"]})
        sm.update_targets(s.session_id, {"twitter": ["alice"]},
                          fetch_options={"default_count": 100, "targets": {}})
        assert sm.load(s.session_id).fetch_options["default_count"] == 100

    def test_update_targets_preserves_fetch_options_when_not_provided(self, sm):
        s = sm.create("FO Preserve", {"twitter": ["alice"]},
                      fetch_options={"default_count": 75, "targets": {}})
        sm.update_targets(s.session_id, {"twitter": ["bob"]})
        assert sm.load(s.session_id).fetch_options["default_count"] == 75

    def test_update_targets_returns_none_for_unknown_session(self, sm):
        assert sm.update_targets("ghost-id", {"twitter": ["alice"]}) is None

    def test_session_path_sanitises_path_traversal_attempt(self, sm):
        """A session_id containing path traversal characters must not escape sessions_dir."""
        dangerous_id = "../../etc/passwd"
        path = sm._session_path(dangerous_id)
        assert path.parent == sm.sessions_dir
        assert ".." not in path.name
