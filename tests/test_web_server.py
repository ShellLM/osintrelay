"""
Tests for socialosintagent/web_server.py — session and cache REST endpoints.

The contacts endpoints are covered by test_network_contacts.py.
All tests use FastAPI's TestClient and mock _get_components() so no real
API clients, LLM calls, or platform fetches are made.

Covers:
GET  /api/v1/platforms
  - Returns all known platforms with available flag set correctly

GET  /api/v1/sessions
  - Returns empty list when no sessions exist
  - Returns session summaries when sessions exist

POST /api/v1/sessions
  - Creates session and returns 201 with full session data
  - Returns 422 when platforms field is missing
  - Returns 422 when platforms dict is empty
  - Returns 422 when a platform has an empty usernames list

GET  /api/v1/sessions/{id}
  - Returns full session dict
  - Returns 404 for unknown session

DELETE /api/v1/sessions/{id}
  - Deletes session and returns 200 with deleted id
  - Returns 404 for unknown session

PATCH /api/v1/sessions/{id}/rename
  - Updates name and returns summary
  - Returns 404 for unknown session

PUT /api/v1/sessions/{id}/targets
  - Updates platforms and returns summary
  - Returns 404 for unknown session

POST /api/v1/sessions/{id}/analyse
  - Returns 202 with job_id for valid session
  - Returns 404 for unknown session
  - Returns 409 when a job is already running for the session

GET /api/v1/jobs/{id}
  - Returns job status for known job
  - Returns 404 for unknown job

GET /api/v1/cache
  - Returns CacheStatusResponse with entries list

POST /api/v1/cache/purge
  - Returns purged list for valid targets
  - Returns 422 for invalid purge target
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def web_client(tmp_path, monkeypatch):
    """TestClient wired to a temp data directory with mocked components."""
    from socialosintagent.session_manager import SessionManager
    from socialosintagent.cache import CacheManager

    real_session_manager = SessionManager(tmp_path)
    real_cache_manager = CacheManager(tmp_path, is_offline=False)
    mock_llm = MagicMock()
    mock_clients = MagicMock()
    mock_clients.get_available_platforms.return_value = ["github", "hackernews", "twitter"]

    def fake_get_components():
        return real_cache_manager, mock_llm, mock_clients, real_session_manager

    monkeypatch.setattr("socialosintagent.web_server._get_components", fake_get_components)
    # Ensure _JOBS is empty for each test
    monkeypatch.setattr("socialosintagent.web_server._JOBS", {})

    from socialosintagent.web_server import app
    return TestClient(app, raise_server_exceptions=True), real_session_manager, tmp_path


# ── GET /api/v1/platforms ─────────────────────────────────────────────────────

class TestGetPlatforms:
    def test_returns_all_known_platforms(self, web_client):
        client, _, _ = web_client
        resp = client.get("/api/v1/platforms")
        assert resp.status_code == 200
        platforms = {p["name"]: p["available"] for p in resp.json()["platforms"]}
        # All six known platforms should be present
        assert set(platforms.keys()) == {"twitter", "reddit", "bluesky", "mastodon", "github", "hackernews"}

    def test_available_flag_reflects_client_manager(self, web_client):
        client, _, _ = web_client
        resp = client.get("/api/v1/platforms")
        platforms = {p["name"]: p["available"] for p in resp.json()["platforms"]}
        # mock returns twitter, github, hackernews as available
        assert platforms["twitter"] is True
        assert platforms["github"] is True
        assert platforms["hackernews"] is True
        assert platforms["reddit"] is False
        assert platforms["bluesky"] is False


# ── Sessions CRUD ─────────────────────────────────────────────────────────────

class TestListSessions:
    def test_empty_list_when_no_sessions(self, web_client):
        client, _, _ = web_client
        resp = client.get("/api/v1/sessions")
        assert resp.status_code == 200
        assert resp.json()["sessions"] == []

    def test_returns_summaries_when_sessions_exist(self, web_client):
        client, sm, _ = web_client
        sm.create("Alpha", {"twitter": ["alice"]})
        sm.create("Beta", {"github": ["bob"]})
        resp = client.get("/api/v1/sessions")
        assert resp.status_code == 200
        names = {s["name"] for s in resp.json()["sessions"]}
        assert "Alpha" in names
        assert "Beta" in names


class TestCreateSession:
    def test_creates_session_and_returns_201(self, web_client):
        client, _, _ = web_client
        resp = client.post("/api/v1/sessions", json={"name": "Op Falcon", "platforms": {"twitter": ["hawk"]}})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Op Falcon"
        assert "session_id" in data
        assert data["platforms"] == {"twitter": ["hawk"]}

    def test_missing_platforms_returns_422(self, web_client):
        client, _, _ = web_client
        resp = client.post("/api/v1/sessions", json={"name": "No Platforms"})
        assert resp.status_code == 422

    def test_empty_platforms_dict_returns_422(self, web_client):
        client, _, _ = web_client
        resp = client.post("/api/v1/sessions", json={"name": "Empty", "platforms": {}})
        assert resp.status_code == 422

    def test_platform_with_empty_usernames_returns_422(self, web_client):
        client, _, _ = web_client
        resp = client.post("/api/v1/sessions", json={"name": "Bad", "platforms": {"twitter": []}})
        assert resp.status_code == 422

    def test_fetch_options_stored_when_provided(self, web_client):
        client, sm, _ = web_client
        resp = client.post("/api/v1/sessions", json={
            "name": "With Options",
            "platforms": {"hackernews": ["pg"]},
            "fetch_options": {"default_count": 100, "targets": {}}
        })
        assert resp.status_code == 201
        session = sm.load(resp.json()["session_id"])
        assert session.fetch_options["default_count"] == 100


class TestGetSession:
    def test_returns_full_session(self, web_client):
        client, sm, _ = web_client
        s = sm.create("Detail Test", {"github": ["torvalds"]})
        resp = client.get(f"/api/v1/sessions/{s.session_id}")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == s.session_id
        assert resp.json()["name"] == "Detail Test"
        assert "query_history" in resp.json()

    def test_returns_404_for_unknown_session(self, web_client):
        client, _, _ = web_client
        resp = client.get("/api/v1/sessions/does-not-exist")
        assert resp.status_code == 404


class TestDeleteSession:
    def test_deletes_session_and_returns_200(self, web_client):
        client, sm, _ = web_client
        s = sm.create("To Delete", {"hackernews": ["pg"]})
        resp = client.delete(f"/api/v1/sessions/{s.session_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == s.session_id
        assert sm.load(s.session_id) is None

    def test_returns_404_for_unknown_session(self, web_client):
        client, _, _ = web_client
        resp = client.delete("/api/v1/sessions/ghost-id")
        assert resp.status_code == 404


class TestRenameSession:
    def test_renames_session_and_returns_summary(self, web_client):
        client, sm, _ = web_client
        s = sm.create("Old Name", {"twitter": ["a"]})
        resp = client.patch(f"/api/v1/sessions/{s.session_id}/rename", json={"name": "New Name"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "New Name"
        assert sm.load(s.session_id).name == "New Name"

    def test_returns_404_for_unknown_session(self, web_client):
        client, _, _ = web_client
        resp = client.patch("/api/v1/sessions/ghost-id/rename", json={"name": "X"})
        assert resp.status_code == 404

    def test_empty_name_returns_422(self, web_client):
        client, sm, _ = web_client
        s = sm.create("Valid Name", {"twitter": ["a"]})
        resp = client.patch(f"/api/v1/sessions/{s.session_id}/rename", json={"name": ""})
        assert resp.status_code == 422


class TestUpdateTargets:
    def test_updates_platforms_and_returns_summary(self, web_client):
        client, sm, _ = web_client
        s = sm.create("Targets", {"twitter": ["old"]})
        resp = client.put(f"/api/v1/sessions/{s.session_id}/targets",
                          json={"platforms": {"github": ["torvalds"]}})
        assert resp.status_code == 200
        reloaded = sm.load(s.session_id)
        assert "github" in reloaded.platforms
        assert "twitter" not in reloaded.platforms

    def test_returns_404_for_unknown_session(self, web_client):
        client, _, _ = web_client
        resp = client.put("/api/v1/sessions/ghost-id/targets", json={"platforms": {"hackernews": ["pg"]}})
        assert resp.status_code == 404

    def test_empty_platforms_returns_422(self, web_client):
        client, sm, _ = web_client
        s = sm.create("Valid", {"twitter": ["a"]})
        resp = client.put(f"/api/v1/sessions/{s.session_id}/targets", json={"platforms": {}})
        assert resp.status_code == 422


# ── Analysis job endpoints ────────────────────────────────────────────────────

class TestStartAnalysis:
    def test_returns_202_with_job_id_for_valid_session(self, web_client):
        client, sm, _ = web_client
        s = sm.create("Analysis Test", {"hackernews": ["pg"]})
        resp = client.post(f"/api/v1/sessions/{s.session_id}/analyse",
                           json={"query": "What are their interests?"})
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "running"
        assert data["session_id"] == s.session_id

    def test_returns_404_for_unknown_session(self, web_client):
        client, _, _ = web_client
        resp = client.post("/api/v1/sessions/ghost-id/analyse",
                           json={"query": "test query"})
        assert resp.status_code == 404

    def test_returns_409_when_job_already_running(self, web_client, monkeypatch):
        client, sm, _ = web_client
        s = sm.create("Conflict Test", {"hackernews": ["pg"]})
        # Inject a fake running job for this session
        monkeypatch.setitem(
            __import__("socialosintagent.web_server", fromlist=["_JOBS"])._JOBS,
            "existing-job-id",
            {"job_id": "existing-job-id", "session_id": s.session_id, "status": "running", "query": "old query", "events": []}
        )
        resp = client.post(f"/api/v1/sessions/{s.session_id}/analyse",
                           json={"query": "new query"})
        assert resp.status_code == 409

    def test_empty_query_returns_422(self, web_client):
        client, sm, _ = web_client
        s = sm.create("Query Validation", {"hackernews": ["pg"]})
        resp = client.post(f"/api/v1/sessions/{s.session_id}/analyse",
                           json={"query": ""})
        assert resp.status_code == 422


class TestGetJobStatus:
    def test_returns_job_status_for_known_job(self, web_client, monkeypatch):
        client, _, _ = web_client
        monkeypatch.setitem(
            __import__("socialosintagent.web_server", fromlist=["_JOBS"])._JOBS,
            "test-job-id",
            {"job_id": "test-job-id", "session_id": "s1", "status": "complete",
             "query": "find patterns", "query_id": "abc123", "error": None, "progress": None, "events": []}
        )
        resp = client.get("/api/v1/jobs/test-job-id")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "test-job-id"
        assert data["status"] == "complete"
        assert data["query"] == "find patterns"

    def test_returns_404_for_unknown_job(self, web_client):
        client, _, _ = web_client
        resp = client.get("/api/v1/jobs/ghost-job")
        assert resp.status_code == 404


# ── Cache endpoints ───────────────────────────────────────────────────────────

class TestCacheStatus:
    def test_returns_empty_entries_when_no_cache(self, web_client):
        client, _, _ = web_client
        resp = client.get("/api/v1/cache")
        assert resp.status_code == 200
        assert resp.json()["entries"] == []

    def test_returns_entry_for_cached_platform_data(self, web_client, tmp_path, monkeypatch):
        from socialosintagent.cache import CacheManager
        from socialosintagent.utils import UserData
        import socialosintagent.web_server as ws
        client, _, _ = web_client
        # Redirect the web server's BASE_DIR to tmp_path so it reads the
        # cache we write here rather than the real data/ directory.
        monkeypatch.setattr(ws, "BASE_DIR", tmp_path)
        cache = CacheManager(tmp_path, is_offline=False)
        data: UserData = {"profile": {"platform": "hackernews", "username": "pg", "id": "pg"}, "posts": []}
        cache.save("hackernews", "pg", data)
        resp = client.get("/api/v1/cache")
        assert resp.status_code == 200
        entries = resp.json()["entries"]
        assert len(entries) == 1
        assert entries[0]["platform"] == "hackernews"
        assert entries[0]["username"] == "pg"


class TestPurgeCache:
    def test_purge_cache_returns_purged_list(self, web_client, tmp_path):
        client, _, _ = web_client
        # Create some directories to purge
        (tmp_path / "cache").mkdir(exist_ok=True)
        (tmp_path / "media").mkdir(exist_ok=True)
        resp = client.post("/api/v1/cache/purge", json={"targets": ["cache", "media"]})
        assert resp.status_code == 200
        purged = resp.json()["purged"]
        assert "cache" in purged
        assert "media" in purged

    def test_purge_all_expands_to_all_targets(self, web_client, tmp_path):
        client, _, _ = web_client
        resp = client.post("/api/v1/cache/purge", json={"targets": ["all"]})
        assert resp.status_code == 200
        purged = resp.json()["purged"]
        # "all" should expand to cache, media, outputs
        assert set(purged) == {"cache", "media", "outputs"}

    def test_invalid_purge_target_returns_422(self, web_client):
        client, _, _ = web_client
        resp = client.post("/api/v1/cache/purge", json={"targets": ["passwords"]})
        assert resp.status_code == 422
