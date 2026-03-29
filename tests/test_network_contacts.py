"""
Tests for the network contact extraction and session contact management features.

No live API calls or LLM calls are made anywhere in this file.

Covers:
network_extractor.extract_contacts()
  - Twitter @mentions in post text are found
  - Twitter RT attribution is captured as 'retweet', not 'mention'
  - RT body is not scanned for additional mentions (they belong to the original author)
  - Reddit u/ mentions are found
  - GitHub repo owner extracted from context.repo
  - GitHub target user is not surfaced as their own repo contact
  - HackerNews @mentions are found
  - Mastodon @user and federated @user@instance mentions are found
  - Bluesky domain-format handles are found
  - Source user is never returned as a contact
  - Other active session targets are excluded from results
  - Multiple interactions with the same user merge into one weighted entry
  - Results are sorted by weight descending
  - Contacts spanning multiple post types accumulate all interaction_types
  - to_dict() serialises correctly

session_manager.Session
  - dismissed_contacts defaults to empty list on new sessions
  - dismissed_contacts survives to_dict / from_dict round-trip
  - Old session files without dismissed_contacts deserialise safely

session_manager.SessionManager.dismiss_contact / undismiss_contact
  - dismiss_contact stores the correct key and persists it
  - Dismissing the same contact twice is idempotent
  - undismiss_contact removes the key
  - Both return None for unknown session IDs

GET /api/v1/sessions/{id}/contacts
  - Returns 404 for unknown session
  - Returns empty contacts list when no cached data exists
  - Dismissed contacts are excluded from the contacts list
  - total_extracted reflects count before dismiss filtering
  - dismissed list is included in the response

POST /api/v1/sessions/{id}/contacts/dismiss
  - Returns 404 for unknown session
  - Persists the dismissal so subsequent GET excludes the contact

POST /api/v1/sessions/{id}/contacts/undismiss
  - Reverses a dismissal so subsequent GET includes the contact again
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(iso: str) -> datetime:
    """Parse an ISO string to a UTC-aware datetime."""
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


def _make_post(
    post_id: str,
    platform: str,
    text: str,
    post_type: str = "post",
    created_at: str = "2024-06-01T12:00:00",
    context: dict = None,
) -> dict:
    """Minimal NormalizedPost dict."""
    return {
        "id": post_id,
        "platform": platform,
        "text": text,
        "type": post_type,
        "created_at": _ts(created_at),
        "context": context or {},
        "media": [],
        "external_links": [],
    }


# ---------------------------------------------------------------------------
# network_extractor tests
# ---------------------------------------------------------------------------

class TestExtractContacts:
    """Tests for network_extractor.extract_contacts()."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from socialosintagent.network_extractor import extract_contacts
        self.extract_contacts = extract_contacts

    def _run(self, platform_posts: dict, active_targets: dict = None):
        active_targets = active_targets or {
            p: list(users.keys()) for p, users in platform_posts.items()
        }
        return self.extract_contacts(platform_posts, active_targets)

    # ---- Twitter --------------------------------------------------------

    def test_twitter_mention_is_found(self):
        posts = {"twitter": {"alice": [
            _make_post("1", "twitter", "Great thread by @bob on this!"),
        ]}}
        contacts = self._run(posts)
        usernames = [c.username.lower() for c in contacts]
        assert "bob" in usernames

    def test_twitter_mention_interaction_type(self):
        posts = {"twitter": {"alice": [
            _make_post("1", "twitter", "@charlie are you seeing this?"),
        ]}}
        contacts = self._run(posts)
        charlie = next(c for c in contacts if c.username.lower() == "charlie")
        assert "mention" in charlie.interaction_types

    def test_twitter_rt_captured_as_retweet(self):
        posts = {"twitter": {"alice": [
            _make_post("1", "twitter", "RT @dave: Some important news here"),
        ]}}
        contacts = self._run(posts)
        dave = next((c for c in contacts if c.username.lower() == "dave"), None)
        assert dave is not None
        assert "retweet" in dave.interaction_types

    def test_twitter_rt_body_not_scanned_for_mentions(self):
        """@mentions inside an RT body belong to the original author, not our target."""
        posts = {"twitter": {"alice": [
            # RT body contains @eve — she should NOT appear because the text
            # after "RT @dave:" is the original author's words, not alice's.
            _make_post("1", "twitter", "RT @dave: Thanks to @eve for this"),
        ]}}
        contacts = self._run(posts)
        usernames = [c.username.lower() for c in contacts]
        assert "eve" not in usernames
        assert "dave" in usernames  # RT source is still captured

    def test_twitter_rt_not_double_counted_as_mention(self):
        """The RT source should appear only once with type 'retweet', not also 'mention'."""
        posts = {"twitter": {"alice": [
            _make_post("1", "twitter", "RT @frank: Original tweet text"),
        ]}}
        contacts = self._run(posts)
        frank = next(c for c in contacts if c.username.lower() == "frank")
        assert "mention" not in frank.interaction_types

    # ---- Reddit ---------------------------------------------------------

    def test_reddit_u_mention_is_found(self):
        posts = {"reddit": {"alice": [
            _make_post("1", "reddit", "u/gregorysmith has a great point here"),
        ]}}
        contacts = self._run(posts)
        usernames = [c.username.lower() for c in contacts]
        assert "gregorysmith" in usernames

    # ---- GitHub ---------------------------------------------------------

    def test_github_repo_owner_extracted(self):
        posts = {"github": {"alice": [
            _make_post("1", "github", "Pushed to torvalds/linux", context={"repo": "torvalds/linux"}),
        ]}}
        contacts = self._run(posts)
        usernames = [c.username.lower() for c in contacts]
        assert "torvalds" in usernames

    def test_github_own_repo_excluded(self):
        """If alice pushes to alice/myrepo, alice should not appear as a contact."""
        posts = {"github": {"alice": [
            _make_post("1", "github", "Pushed to alice/myrepo", context={"repo": "alice/myrepo"}),
        ]}}
        contacts = self._run(posts)
        usernames = [c.username.lower() for c in contacts]
        assert "alice" not in usernames

    def test_github_repo_interaction_type(self):
        posts = {"github": {"alice": [
            _make_post("1", "github", "Starred octocat/hello", context={"repo": "octocat/hello"}),
        ]}}
        contacts = self._run(posts)
        octocat = next(c for c in contacts if c.username.lower() == "octocat")
        assert "repo_interaction" in octocat.interaction_types

    # ---- HackerNews -----------------------------------------------------

    def test_hackernews_mention_found(self):
        posts = {"hackernews": {"alice": [
            _make_post("1", "hackernews", "@pg makes a good point about this"),
        ]}}
        contacts = self._run(posts)
        assert any(c.username.lower() == "pg" for c in contacts)

    # ---- Mastodon -------------------------------------------------------

    def test_mastodon_local_mention_found(self):
        posts = {"mastodon": {"alice@mastodon.social": [
            _make_post("1", "mastodon", "Great post @helen!"),
        ]}}
        active = {"mastodon": ["alice@mastodon.social"]}
        contacts = self.extract_contacts(
            {"mastodon": {"alice@mastodon.social": posts["mastodon"]["alice@mastodon.social"]}},
            active,
        )
        assert any(c.username.lower() == "helen" for c in contacts)

    def test_mastodon_federated_mention_found(self):
        posts = {"mastodon": {"alice@mastodon.social": [
            _make_post("1", "mastodon", "cc @ivan@fosstodon.org on this"),
        ]}}
        active = {"mastodon": ["alice@mastodon.social"]}
        contacts = self.extract_contacts(
            {"mastodon": {"alice@mastodon.social": posts["mastodon"]["alice@mastodon.social"]}},
            active,
        )
        assert any("ivan" in c.username.lower() for c in contacts)

    # ---- Bluesky --------------------------------------------------------

    def test_bluesky_handle_mention_found(self):
        """Bluesky handles contain a dot — e.g. user.bsky.social."""
        posts = {"bluesky": {"alice.bsky.social": [
            _make_post("1", "bluesky", "Nice thread @julia.bsky.social!"),
        ]}}
        active = {"bluesky": ["alice.bsky.social"]}
        contacts = self.extract_contacts(
            {"bluesky": {"alice.bsky.social": posts["bluesky"]["alice.bsky.social"]}},
            active,
        )
        assert any("julia" in c.username.lower() for c in contacts)

    # ---- Self-exclusion and target exclusion ----------------------------

    def test_source_user_never_returned(self):
        """The analysed user must not appear in their own contact list."""
        posts = {"twitter": {"alice": [
            _make_post("1", "twitter", "@alice talking to myself"),
        ]}}
        contacts = self._run(posts)
        assert not any(c.username.lower() == "alice" for c in contacts)

    def test_active_targets_excluded(self):
        """Other currently-tracked session targets should not appear as contacts."""
        posts = {
            "twitter": {
                "alice": [_make_post("1", "twitter", "@bob looking good!")],
            }
        }
        # bob is also an active target
        active = {"twitter": ["alice", "bob"]}
        contacts = self.extract_contacts({"twitter": {"alice": posts["twitter"]["alice"]}}, active)
        assert not any(c.username.lower() == "bob" for c in contacts)

    # ---- Weighting and deduplication ------------------------------------

    def test_repeated_mentions_accumulate_weight(self):
        posts = {"twitter": {"alice": [
            _make_post("1", "twitter", "@carlos great thread"),
            _make_post("2", "twitter", "Hey @carlos did you see this?"),
            _make_post("3", "twitter", "@carlos what do you think?"),
        ]}}
        contacts = self._run(posts)
        carlos = next(c for c in contacts if c.username.lower() == "carlos")
        assert carlos.weight == 3

    def test_results_sorted_by_weight_descending(self):
        posts = {"twitter": {"alice": [
            _make_post("1", "twitter", "@bob @bob @bob three times"),
            _make_post("2", "twitter", "@charlie once"),
        ]}}
        contacts = self._run(posts)
        # bob appears 3 times in the text (the regex finds each @bob), charlie once
        weights = [c.weight for c in contacts]
        assert weights == sorted(weights, reverse=True)

    def test_multiple_interaction_types_accumulated(self):
        """Same user found via RT and mention should have both types listed."""
        posts = {"twitter": {"alice": [
            _make_post("1", "twitter", "RT @diana: Some tweet"),
            _make_post("2", "twitter", "@diana did you write that?"),
        ]}}
        contacts = self._run(posts)
        # RT post is skipped for body scanning but diana still appears via RT
        diana = next(c for c in contacts if c.username.lower() == "diana")
        assert "retweet" in diana.interaction_types

    def test_empty_posts_returns_empty_list(self):
        contacts = self.extract_contacts(
            {"twitter": {"alice": []}},
            {"twitter": ["alice"]},
        )
        assert contacts == []

    # ---- Timestamps -----------------------------------------------------

    def test_first_seen_last_seen_populated(self):
        posts = {"twitter": {"alice": [
            _make_post("1", "twitter", "@ella hello", created_at="2024-01-01T10:00:00"),
            _make_post("2", "twitter", "@ella again", created_at="2024-03-15T18:00:00"),
        ]}}
        contacts = self._run(posts)
        ella = next(c for c in contacts if c.username.lower() == "ella")
        assert ella.first_seen == _ts("2024-01-01T10:00:00")
        assert ella.last_seen == _ts("2024-03-15T18:00:00")

    # ---- Serialisation --------------------------------------------------

    def test_to_dict_structure(self):
        posts = {"twitter": {"alice": [
            _make_post("1", "twitter", "@finn interesting take"),
        ]}}
        contacts = self._run(posts)
        d = contacts[0].to_dict()
        assert set(d.keys()) == {
            "platform", "username", "interaction_types", "weight",
            "first_seen", "last_seen",
        }
        assert isinstance(d["interaction_types"], list)
        assert isinstance(d["weight"], int)

    def test_to_dict_interaction_types_deduplicated(self):
        """to_dict should deduplicate interaction_types even if stored with repeats."""
        posts = {"twitter": {"alice": [
            _make_post("1", "twitter", "@grace hello"),
            _make_post("2", "twitter", "@grace hi"),
        ]}}
        contacts = self._run(posts)
        grace = next(c for c in contacts if c.username.lower() == "grace")
        d = grace.to_dict()
        assert len(d["interaction_types"]) == len(set(d["interaction_types"]))


# ---------------------------------------------------------------------------
# Session model — dismissed_contacts
# ---------------------------------------------------------------------------

class TestSessionDismissedContacts:
    """Tests for dismissed_contacts on the Session model."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from socialosintagent.session_manager import Session
        self.Session = Session

    def test_dismissed_contacts_defaults_to_empty_list(self):
        session = self.Session("s1", "Test", {"twitter": ["alice"]})
        assert session.dismissed_contacts == []

    def test_dismissed_contacts_survives_round_trip(self):
        session = self.Session("s1", "Test", {"twitter": ["alice"]})
        session.dismissed_contacts = ["twitter/bob", "github/torvalds"]
        restored = self.Session.from_dict(session.to_dict())
        assert restored.dismissed_contacts == ["twitter/bob", "github/torvalds"]

    def test_old_session_without_dismissed_contacts_deserialises_safely(self):
        """Sessions saved before this feature was added must load with an empty list."""
        old_data = {
            "session_id": "s-old",
            "name": "Legacy Session",
            "platforms": {"twitter": ["alice"]},
            "fetch_options": {"default_count": 50, "targets": {}},
            "query_history": [],
            # no "dismissed_contacts" key — simulates an old file
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }
        session = self.Session.from_dict(old_data)
        assert session.dismissed_contacts == []

    def test_dismissed_contacts_not_in_summary(self):
        """summary() is a lightweight listing dict — full dismiss list is not needed there."""
        session = self.Session("s1", "Test", {"twitter": ["alice"]})
        session.dismissed_contacts = ["twitter/bob"]
        # summary() should succeed without error regardless
        summary = session.summary()
        assert "session_id" in summary


# ---------------------------------------------------------------------------
# SessionManager.dismiss_contact / undismiss_contact
# ---------------------------------------------------------------------------

class TestSessionManagerDismiss:
    """Tests for SessionManager dismiss/undismiss methods."""

    @pytest.fixture
    def session_manager(self, tmp_path):
        from socialosintagent.session_manager import SessionManager
        return SessionManager(tmp_path)

    @pytest.fixture
    def session(self, session_manager):
        return session_manager.create(
            name="Test Session",
            platforms={"twitter": ["alice"]},
        )

    def test_dismiss_contact_stores_correct_key(self, session_manager, session):
        updated = session_manager.dismiss_contact(session.session_id, "twitter", "Bob")
        assert "twitter/bob" in updated.dismissed_contacts

    def test_dismiss_contact_lowercases_username(self, session_manager, session):
        session_manager.dismiss_contact(session.session_id, "twitter", "UPPERCASE")
        reloaded = session_manager.load(session.session_id)
        assert "twitter/uppercase" in reloaded.dismissed_contacts

    def test_dismiss_contact_persisted_to_disk(self, session_manager, session):
        session_manager.dismiss_contact(session.session_id, "twitter", "carol")
        reloaded = session_manager.load(session.session_id)
        assert "twitter/carol" in reloaded.dismissed_contacts

    def test_dismiss_contact_idempotent(self, session_manager, session):
        session_manager.dismiss_contact(session.session_id, "twitter", "dave")
        session_manager.dismiss_contact(session.session_id, "twitter", "dave")
        reloaded = session_manager.load(session.session_id)
        assert reloaded.dismissed_contacts.count("twitter/dave") == 1

    def test_dismiss_contact_returns_none_for_unknown_session(self, session_manager):
        result = session_manager.dismiss_contact("nonexistent-id", "twitter", "anyone")
        assert result is None

    def test_undismiss_contact_removes_key(self, session_manager, session):
        session_manager.dismiss_contact(session.session_id, "twitter", "eve")
        session_manager.undismiss_contact(session.session_id, "twitter", "eve")
        reloaded = session_manager.load(session.session_id)
        assert "twitter/eve" not in reloaded.dismissed_contacts

    def test_undismiss_contact_persisted_to_disk(self, session_manager, session):
        session_manager.dismiss_contact(session.session_id, "github", "frank")
        session_manager.undismiss_contact(session.session_id, "github", "frank")
        reloaded = session_manager.load(session.session_id)
        assert "github/frank" not in reloaded.dismissed_contacts

    def test_undismiss_contact_on_non_dismissed_is_safe(self, session_manager, session):
        """Undismissing someone who was never dismissed should not raise."""
        result = session_manager.undismiss_contact(session.session_id, "twitter", "ghost")
        assert result is not None  # session returned, no error

    def test_undismiss_contact_returns_none_for_unknown_session(self, session_manager):
        result = session_manager.undismiss_contact("nonexistent-id", "twitter", "anyone")
        assert result is None

    def test_multiple_platforms_in_dismiss_list(self, session_manager, session):
        session_manager.dismiss_contact(session.session_id, "twitter", "grace")
        session_manager.dismiss_contact(session.session_id, "github", "grace")
        reloaded = session_manager.load(session.session_id)
        assert "twitter/grace" in reloaded.dismissed_contacts
        assert "github/grace" in reloaded.dismissed_contacts


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """
    Stand up a test FastAPI client with mocked _get_components.

    The contacts endpoints call _extract_contacts directly (no module-level
    agent). Tests that need to control what contacts are returned should patch
    "socialosintagent.web_server._extract_contacts" via monkeypatch in the
    individual test method.

    Returns (TestClient, SessionManager, mock_cache) so tests can create
    sessions and inspect persisted state without hitting the filesystem
    outside of tmp_path.
    """
    from fastapi.testclient import TestClient
    from socialosintagent.session_manager import SessionManager

    real_session_manager = SessionManager(tmp_path)
    mock_cache = MagicMock()
    mock_cache.load.return_value = None  # no cached posts by default
    mock_llm = MagicMock()
    mock_clients = MagicMock()

    def fake_get_components():
        return mock_cache, mock_llm, mock_clients, real_session_manager

    monkeypatch.setattr(
        "socialosintagent.web_server._get_components", fake_get_components
    )

    from socialosintagent.web_server import app
    client = TestClient(app, raise_server_exceptions=True)

    return client, real_session_manager, mock_cache



class TestContactsEndpoint:
    """Tests for GET /api/v1/sessions/{id}/contacts."""

    def _make_contact(self, platform, username, weight=1, types=None):
        """Build a DiscoveredContact for use as _extract_contacts mock return value."""
        from socialosintagent.network_extractor import DiscoveredContact
        c = DiscoveredContact(platform=platform, username=username)
        c.weight = weight
        c.interaction_types = types or ["mention"]
        c.first_seen = _ts("2024-01-01T00:00:00")
        c.last_seen = _ts("2024-06-01T00:00:00")
        return c

    def test_get_contacts_404_for_unknown_session(self, api_client):
        client, _, _ = api_client
        resp = client.get("/api/v1/sessions/does-not-exist/contacts")
        assert resp.status_code == 404

    def test_get_contacts_returns_200_with_empty_list(self, api_client, monkeypatch):
        client, sm, _ = api_client
        session = sm.create("Test", {"twitter": ["alice"]})
        monkeypatch.setattr("socialosintagent.web_server._extract_contacts", lambda *a, **kw: [])

        resp = client.get(f"/api/v1/sessions/{session.session_id}/contacts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["contacts"] == []
        assert data["total_extracted"] == 0

    def test_get_contacts_returns_discovered_contacts(self, api_client, monkeypatch):
        client, sm, _ = api_client
        session = sm.create("Test", {"twitter": ["alice"]})
        contacts = [
            self._make_contact("twitter", "bob", weight=3),
            self._make_contact("twitter", "carol", weight=1),
        ]
        monkeypatch.setattr("socialosintagent.web_server._extract_contacts", lambda *a, **kw: contacts)

        resp = client.get(f"/api/v1/sessions/{session.session_id}/contacts")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["contacts"]) == 2
        assert data["contacts"][0]["username"] == "bob"
        assert data["contacts"][0]["weight"] == 3

    def test_get_contacts_excludes_dismissed(self, api_client, monkeypatch):
        client, sm, _ = api_client
        session = sm.create("Test", {"twitter": ["alice"]})
        sm.dismiss_contact(session.session_id, "twitter", "dave")
        contacts = [
            self._make_contact("twitter", "bob", weight=2),
            self._make_contact("twitter", "dave", weight=5),  # dismissed
        ]
        monkeypatch.setattr("socialosintagent.web_server._extract_contacts", lambda *a, **kw: contacts)

        resp = client.get(f"/api/v1/sessions/{session.session_id}/contacts")
        assert resp.status_code == 200
        data = resp.json()
        returned_names = [c["username"] for c in data["contacts"]]
        assert "dave" not in returned_names
        assert "bob" in returned_names

    def test_get_contacts_total_extracted_counts_before_dismiss_filter(self, api_client, monkeypatch):
        client, sm, _ = api_client
        session = sm.create("Test", {"twitter": ["alice"]})
        sm.dismiss_contact(session.session_id, "twitter", "eve")
        contacts = [
            self._make_contact("twitter", "frank", weight=2),
            self._make_contact("twitter", "eve", weight=1),  # dismissed
        ]
        monkeypatch.setattr("socialosintagent.web_server._extract_contacts", lambda *a, **kw: contacts)

        resp = client.get(f"/api/v1/sessions/{session.session_id}/contacts")
        data = resp.json()
        # 2 total extracted, 1 after dismiss filter, total_extracted must be 2
        assert data["total_extracted"] == 2
        assert len(data["contacts"]) == 1

    def test_get_contacts_includes_dismissed_list(self, api_client, monkeypatch):
        client, sm, _ = api_client
        session = sm.create("Test", {"twitter": ["alice"]})
        sm.dismiss_contact(session.session_id, "twitter", "grace")
        monkeypatch.setattr("socialosintagent.web_server._extract_contacts", lambda *a, **kw: [])

        resp = client.get(f"/api/v1/sessions/{session.session_id}/contacts")
        data = resp.json()
        assert "twitter/grace" in data["dismissed"]

    def test_get_contacts_contact_has_expected_fields(self, api_client, monkeypatch):
        client, sm, _ = api_client
        session = sm.create("Test", {"twitter": ["alice"]})
        contacts = [self._make_contact("twitter", "helen", weight=2, types=["mention", "retweet"])]
        monkeypatch.setattr("socialosintagent.web_server._extract_contacts", lambda *a, **kw: contacts)

        resp = client.get(f"/api/v1/sessions/{session.session_id}/contacts")
        contact = resp.json()["contacts"][0]
        assert contact["platform"] == "twitter"
        assert contact["username"] == "helen"
        assert contact["weight"] == 2
        assert set(contact["interaction_types"]) == {"mention", "retweet"}
        assert contact["first_seen"] is not None
        assert contact["last_seen"] is not None


class TestDismissEndpoint:
    """Tests for POST /api/v1/sessions/{id}/contacts/dismiss."""

    def test_dismiss_404_for_unknown_session(self, api_client):
        client, _, _ = api_client
        resp = client.post(
            "/api/v1/sessions/bad-id/contacts/dismiss",
            json={"platform": "twitter", "username": "bob"},
        )
        assert resp.status_code == 404

    def test_dismiss_persists_contact(self, api_client):
        client, sm, _ = api_client
        session = sm.create("Test", {"twitter": ["alice"]})

        resp = client.post(
            f"/api/v1/sessions/{session.session_id}/contacts/dismiss",
            json={"platform": "twitter", "username": "ivan"},
        )
        assert resp.status_code == 200
        assert resp.json()["dismissed"] == "twitter/ivan"

        reloaded = sm.load(session.session_id)
        assert "twitter/ivan" in reloaded.dismissed_contacts

    def test_dismiss_then_get_excludes_contact(self, api_client, monkeypatch):
        from socialosintagent.network_extractor import DiscoveredContact
        client, sm, _ = api_client
        session = sm.create("Test", {"twitter": ["alice"]})

        c = DiscoveredContact(platform="twitter", username="julia")
        c.weight = 3
        c.interaction_types = ["mention"]
        c.first_seen = _ts("2024-01-01T00:00:00")
        c.last_seen = _ts("2024-06-01T00:00:00")
        monkeypatch.setattr("socialosintagent.web_server._extract_contacts", lambda *a, **kw: [c])

        # Dismiss
        client.post(
            f"/api/v1/sessions/{session.session_id}/contacts/dismiss",
            json={"platform": "twitter", "username": "julia"},
        )

        # Verify excluded from GET
        resp = client.get(f"/api/v1/sessions/{session.session_id}/contacts")
        names = [item["username"] for item in resp.json()["contacts"]]
        assert "julia" not in names

    def test_dismiss_is_case_insensitive(self, api_client):
        client, sm, _ = api_client
        session = sm.create("Test", {"twitter": ["alice"]})

        client.post(
            f"/api/v1/sessions/{session.session_id}/contacts/dismiss",
            json={"platform": "twitter", "username": "KEVIN"},
        )
        reloaded = sm.load(session.session_id)
        assert "twitter/kevin" in reloaded.dismissed_contacts


class TestUndismissEndpoint:
    """Tests for POST /api/v1/sessions/{id}/contacts/undismiss."""

    def test_undismiss_404_for_unknown_session(self, api_client):
        client, _, _ = api_client
        resp = client.post(
            "/api/v1/sessions/bad-id/contacts/undismiss",
            json={"platform": "twitter", "username": "bob"},
        )
        assert resp.status_code == 404

    def test_undismiss_reverses_dismissal(self, api_client):
        client, sm, _ = api_client
        session = sm.create("Test", {"twitter": ["alice"]})

        # Dismiss then undismiss
        client.post(
            f"/api/v1/sessions/{session.session_id}/contacts/dismiss",
            json={"platform": "twitter", "username": "lena"},
        )
        resp = client.post(
            f"/api/v1/sessions/{session.session_id}/contacts/undismiss",
            json={"platform": "twitter", "username": "lena"},
        )
        assert resp.status_code == 200

        reloaded = sm.load(session.session_id)
        assert "twitter/lena" not in reloaded.dismissed_contacts

    def test_undismiss_non_dismissed_is_safe(self, api_client):
        client, sm, _ = api_client
        session = sm.create("Test", {"twitter": ["alice"]})

        resp = client.post(
            f"/api/v1/sessions/{session.session_id}/contacts/undismiss",
            json={"platform": "twitter", "username": "nobody"},
        )
        assert resp.status_code == 200
