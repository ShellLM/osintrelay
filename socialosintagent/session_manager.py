"""
Session management for the web interface.

Sessions are first-class persistent objects stored as JSON files in
data/sessions/. Each session holds its target list, fetch options,
query history, and the full report for each completed analysis.

The CLI interface is session-less (ephemeral state in memory), so this
module is only used by the web server. Both interfaces share the same
underlying CacheManager and agent, so cached platform data is shared.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("SocialOSINTAgent.SessionManager")


class Session:
    """
    Represents a persistent OSINT investigation session.

    A session groups together:
    - A set of targets (platforms + usernames)
    - Fetch options (default count, per-target overrides)
    - A chronological query history with results
    - Metadata (name, created/updated timestamps)
    """

    def __init__(
        self,
        session_id: str,
        name: str,
        platforms: Dict[str, List[str]],
        fetch_options: Optional[Dict[str, Any]] = None,
        monitoring_rules: Optional[List[Dict[str, Any]]] = None,
    ):
        self.session_id = session_id
        self.name = name
        self.platforms = platforms  # e.g. {"twitter": ["naval"], "github": ["torvalds"]}
        self.fetch_options = fetch_options or {"default_count": 50, "targets": {}}
        # Continuous monitoring rules (used by watcher.py). Each rule is a plain
        # dict so we can evolve it without strict schema migrations.
        self.monitoring_rules: List[Dict[str, Any]] = monitoring_rules or []
        self.query_history: List[Dict[str, Any]] = []
        # Contacts the user has explicitly dismissed from the network panel.
        # Stored as "platform/username_lowercase" strings so they survive
        # username casing changes and are trivially serialisable to JSON.
        self.dismissed_contacts: List[str] = []
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at

    def add_query_result(self, query: str, report: str, metadata: Dict[str, Any], entities: Dict[str, Any] = None) -> str:
        """
        Appends a completed analysis result to the query history.

        Args:
            query:    The natural language query that was run.
            report:   The markdown report from the LLM.
            metadata: Analysis metadata (targets, models, stats, etc.).
            entities: Extracted OSINT entities from the analysis.

        Returns:
            The query_id of the new history entry.
        """
        query_id = str(uuid.uuid4())[:8]
        entry = {
            "query_id": query_id,
            "query": query,
            "report": report,
            "metadata": metadata,
            "entities": entities or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.query_history.append(entry)
        self.updated_at = entry["timestamp"]
        return query_id

    def to_dict(self) -> Dict[str, Any]:
        """Serialises the session to a plain dict for JSON storage."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "platforms": self.platforms,
            "fetch_options": self.fetch_options,
            "monitoring_rules": self.monitoring_rules,
            "query_history": self.query_history,
            "dismissed_contacts": self.dismissed_contacts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Deserialises a session from a plain dict (loaded from JSON)."""
        session = cls(
            session_id=data["session_id"],
            name=data["name"],
            platforms=data.get("platforms", {}),
            fetch_options=data.get("fetch_options", {"default_count": 50, "targets": {}}),
            monitoring_rules=data.get("monitoring_rules", []),
        )
        session.query_history = data.get("query_history", [])
        session.dismissed_contacts = data.get("dismissed_contacts", [])
        session.created_at = data.get("created_at", session.created_at)
        session.updated_at = data.get("updated_at", session.updated_at)
        return session

    def summary(self) -> Dict[str, Any]:
        """
        Returns a lightweight summary dict suitable for listing sessions
        without returning full report content.
        """
        target_count = sum(len(users) for users in self.platforms.values())
        return {
            "session_id": self.session_id,
            "name": self.name,
            "platforms": self.platforms,
            "target_count": target_count,
            "query_count": len(self.query_history),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class SessionManager:
    """
    Handles persistence of Session objects to/from the data/sessions/ directory.

    Sessions are stored as individual JSON files named by session ID.
    No database required — consistent with the project's file-based cache approach.
    """

    def __init__(self, base_dir: Path):
        """
        Args:
            base_dir: The root data directory (e.g. Path("data")).
                      Sessions are stored in base_dir/sessions/.
        """
        self.sessions_dir = base_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SessionManager initialised. Sessions directory: {self.sessions_dir}")

    def _session_path(self, session_id: str) -> Path:
        """Returns the file path for a given session ID."""
        # Sanitize session_id to prevent path traversal
        safe_id = "".join(c for c in session_id if c.isalnum() or c in ["-", "_"])
        return self.sessions_dir / f"{safe_id}.json"

    def create(
        self,
        name: str,
        platforms: Dict[str, List[str]],
        fetch_options: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """
        Creates and persists a new session.

        Args:
            name:          Human-readable session name.
            platforms:     Initial target platforms and usernames.
            fetch_options: Optional fetch configuration.

        Returns:
            The newly created Session object.
        """
        session_id = str(uuid.uuid4())
        session = Session(session_id, name, platforms, fetch_options)
        self.save(session)
        logger.info(f"Created session '{name}' ({session_id})")
        return session

    def load(self, session_id: str) -> Optional[Session]:
        """
        Loads a session by ID.

        Args:
            session_id: The UUID of the session.

        Returns:
            The Session object, or None if not found or unreadable.
        """
        path = self._session_path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Session.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def save(self, session: Session) -> None:
        """
        Persists a session to disk.

        Args:
            session: The Session object to save.
        """
        path = self._session_path(session.session_id)
        try:
            path.write_text(
                json.dumps(session.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug(f"Saved session {session.session_id}")
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")

    def delete(self, session_id: str) -> bool:
        """
        Deletes a session file.

        Args:
            session_id: The UUID of the session to delete.

        Returns:
            True if deleted, False if not found.
        """
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted session {session_id}")
            return True
        return False

    def list_all(self) -> List[Dict[str, Any]]:
        """
        Returns summary dicts for all sessions, sorted by most recently updated.

        Returns:
            List of session summary dicts (no full report content).
        """
        sessions = []
        for path in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                session = Session.from_dict(data)
                sessions.append(session.summary())
            except Exception as e:
                logger.warning(f"Could not read session file {path.name}: {e}")
        
        # Sort by most recently updated first
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions

    def update_targets(
        self,
        session_id: str,
        platforms: Dict[str, List[str]],
        fetch_options: Optional[Dict[str, Any]] = None,
    ) -> Optional[Session]:
        """
        Updates the target list and fetch options for an existing session.

        Args:
            session_id:    The session to update.
            platforms:     New platforms dict (replaces existing).
            fetch_options: New fetch options (replaces existing if provided).

        Returns:
            Updated Session, or None if session not found.
        """
        session = self.load(session_id)
        if not session:
            return None
        session.platforms = platforms
        if fetch_options is not None:
            session.fetch_options = fetch_options
        session.updated_at = datetime.now(timezone.utc).isoformat()
        self.save(session)
        return session

    def rename(self, session_id: str, new_name: str) -> Optional[Session]:
        """
        Renames a session.

        Args:
            session_id: The session to rename.
            new_name:   The new human-readable name.

        Returns:
            Updated Session, or None if session not found.
        """
        session = self.load(session_id)
        if not session:
            return None
        session.name = new_name
        session.updated_at = datetime.now(timezone.utc).isoformat()
        self.save(session)
        return session

    def dismiss_contact(self, session_id: str, platform: str, username: str) -> Optional[Session]:
        """
        Adds a contact to the session's dismissed list so it is hidden from
        the network panel on future loads.

        The key is stored as "platform/username_lowercase" for stable identity.

        Args:
            session_id: The session to update.
            platform:   The platform the contact was found on.
            username:   The contact's username.

        Returns:
            Updated Session, or None if session not found.
        """
        session = self.load(session_id)
        if not session:
            return None
        key = f"{platform}/{username.lower()}"
        if key not in session.dismissed_contacts:
            session.dismissed_contacts.append(key)
            session.updated_at = datetime.now(timezone.utc).isoformat()
            self.save(session)
        return session

    def undismiss_contact(self, session_id: str, platform: str, username: str) -> Optional[Session]:
        """
        Removes a contact from the session's dismissed list, making it visible
        in the network panel again.

        Args:
            session_id: The session to update.
            platform:   The platform the contact was found on.
            username:   The contact's username.

        Returns:
            Updated Session, or None if session not found.
        """
        session = self.load(session_id)
        if not session:
            return None
        key = f"{platform}/{username.lower()}"
        if key in session.dismissed_contacts:
            session.dismissed_contacts.remove(key)
            session.updated_at = datetime.now(timezone.utc).isoformat()
            self.save(session)
        return session