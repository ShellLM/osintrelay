"""
Background watcher (Phase 3): continuous monitoring + keyword-triggered alerts.

The watcher:
  - Loads persisted monitoring rules from SessionManager (data/sessions/*.json)
  - Every X seconds, fetches recently updated posts for each monitored target
  - Runs the cheap triage router (LLMAnalyzer.run_triage_evaluation)
  - Sends chat alerts only when a match occurs (or when injection is quarantined)
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aiogram import Bot
import httpx

from .llm import LLMAnalyzer
from .platforms import FETCHERS
from .session_manager import SessionManager
from .utils import get_sort_key

logger = logging.getLogger("SocialOSINTAgent.watcher")

# Telegram hard limit per message (leave margin for formatting).
TELEGRAM_MAX_LEN = 4000


def chunk_telegram_text(text: str, max_len: int = TELEGRAM_MAX_LEN) -> List[str]:
    """Split long text into Telegram-sized chunks, preferring paragraph breaks."""
    if len(text) <= max_len:
        return [text]
    chunks: List[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break
        window = remaining[:max_len]
        split_at = window.rfind("\n\n")
        if split_at < max_len // 2:
            split_at = window.rfind("\n")
        if split_at < max_len // 2:
            split_at = max_len
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    return chunks


def _parse_target(target: str) -> Tuple[str, str]:
    """
    Parse `platform/username` into (platform, username).

    We intentionally do not allow slashes inside usernames.
    """
    if not target or "/" not in target:
        raise ValueError(f"Invalid target format: {target!r}. Expected platform/username.")
    platform, username = target.split("/", 1)
    return platform.lower().strip(), username.strip()


def _parse_dt(value: Any) -> datetime:
    """Parse ISO 8601 strings into timezone-aware UTC datetimes."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


class MonitoringWatcher:
    def __init__(
        self,
        *,
        agent: Any,  # SocialOSINTAgent
        session_manager: SessionManager,
        bot: Bot,
        poll_interval_seconds: int,
        fetch_limit: int,
        triage_post_limit: int,
    ):
        self.agent = agent
        self.session_manager = session_manager
        self.bot = bot
        self.poll_interval_seconds = poll_interval_seconds
        self.fetch_limit = fetch_limit
        self.triage_post_limit = triage_post_limit

    async def send_telegram_alert(self, *, chat_id: int, text: str) -> None:
        for chunk in chunk_telegram_text(text):
            # parse_mode intentionally unset: safe plain-text relay.
            await self.bot.send_message(chat_id=chat_id, text=chunk)

    async def send_discord_webhook_alert(self, *, webhook_url: str, text: str) -> None:
        # Discord "content" max length is 2000 chars; use a margin.
        chunks = chunk_telegram_text(text, max_len=1900)
        async with httpx.AsyncClient(timeout=20.0) as client:
            for chunk in chunks:
                try:
                    await client.post(
                        webhook_url,
                        json={"content": chunk},
                        headers={"User-Agent": "SocialOSINTAgent"},
                    )
                except Exception:
                    logger.exception("Failed to send Discord webhook alert.")

    async def send_rule_alert(self, *, rule: Dict[str, Any], text: str) -> None:
        alert_type = rule.get("alert_type")
        alert_channel = rule.get("alert_channel")

        # Backward-compatible heuristics:
        # - Telegram: numeric chat_id
        # - Discord: looks like a webhook URL
        if alert_type is None:
            if isinstance(alert_channel, int) or (isinstance(alert_channel, str) and alert_channel.isdigit()):
                alert_type = "telegram"
            else:
                alert_type = "discord"

        if alert_type == "telegram":
            chat_id = int(alert_channel)
            await self.send_telegram_alert(chat_id=chat_id, text=text)
        elif alert_type == "discord":
            webhook_url = str(alert_channel)
            await self.send_discord_webhook_alert(webhook_url=webhook_url, text=text)
        else:
            logger.warning("Unknown alert_type=%r; dropping alert.", alert_type)

    def _fetch_target_posts(
        self,
        platform: str,
        username: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Synchronous fetch wrapper for platforms' FETCHERS.

        Returns:
            Normalized UserData-like dict or None.
        """
        if not (fetcher := FETCHERS.get(platform)):
            return None

        client = None
        try:
            client = self.agent.client_manager.get_platform_client(platform)
        except Exception:
            # Missing or invalid credentials. Fetcher will raise if attempted;
            # we catch here to keep watcher resilient.
            logger.exception("Failed to initialize platform client for %s/%s", platform, username)
            return None

        kwargs: Dict[str, Any] = {
            "username": username,
            "cache": self.agent.cache,
            "force_refresh": True,
            "fetch_limit": self.fetch_limit,
            "allow_external_media": getattr(self.agent.args, "unsafe_allow_external_media", False),
        }

        platforms_requiring_client = ["twitter", "reddit", "bluesky"]
        if platform == "mastodon":
            kwargs["clients"], kwargs["default_client"] = client
        elif platform in platforms_requiring_client:
            kwargs["client"] = client

        return fetcher(**kwargs)

    async def _evaluate_rule(self, session_id: str, session_name: str, rule: Dict[str, Any]) -> bool:
        """
        Evaluate a single monitoring rule and possibly send an alert.

        Returns:
            True if the rule was updated (last_seen changed), else False.
        """
        enabled = bool(rule.get("enabled", True))
        if not enabled:
            return False

        target = str(rule.get("target", "")).strip()
        condition = str(rule.get("condition", "")).strip()
        alert_channel = rule.get("alert_channel")
        if not target or not condition or alert_channel is None:
            return False

        try:
            platform, username = _parse_target(target)
        except ValueError:
            return False

        last_seen = _parse_dt(rule.get("last_seen_post_created_at") or rule.get("created_at"))

        # Fetch in a worker thread to avoid blocking the event loop.
        user_data = await asyncio.to_thread(self._fetch_target_posts, platform, username)
        if not user_data:
            return False

        posts: List[Dict[str, Any]] = user_data.get("posts", []) or []
        if not posts:
            return False

        new_posts = [
            p for p in posts if get_sort_key(p, "created_at") > last_seen
        ]
        if not new_posts:
            return False

        # Limit evidence size for cheaper triage.
        new_posts_for_triage = new_posts[: self.triage_post_limit]
        for p in new_posts_for_triage:
            # Ensure triage formatting knows the platform.
            p["platform"] = platform

        triage_posts_ts = max(get_sort_key(p, "created_at") for p in new_posts)
        triage_posts_ts_iso = triage_posts_ts.isoformat()

        # Run triage evaluation (cheap model) off the event loop.
        match, details = await asyncio.to_thread(
            self.agent.llm.run_triage_evaluation,
            new_posts_for_triage,
            condition,
        )

        # Always advance last_seen on newly processed posts to avoid repeat alerts.
        rule["last_seen_post_created_at"] = triage_posts_ts_iso
        rule["last_checked_at"] = datetime.now(timezone.utc).isoformat()

        if details.get("quarantined"):
            warnings_preview = details.get("security_warnings") or []
            warn_excerpt = warnings_preview[0] if warnings_preview else "injected content detected"
            msg = (
                "⚠️ Prompt Injection attempt detected in monitored evidence.\n"
                f"Target: {target}\n"
                f"Condition: {condition}\n"
                f"Reason: {details.get('reason','')}\n"
                f"Example: {warn_excerpt}\n"
            )
            logger.info("Quarantined monitoring rule alert sent for %s", target)
            await self.send_rule_alert(rule=rule, text=msg)
            return True

        if not match:
            return True

        matched_keywords = details.get("matched_keywords") or []
        matched_keywords_str = ", ".join(matched_keywords) if matched_keywords else "n/a"

        # Include short snippets for operator UX.
        snippet_lines: List[str] = []
        for p in new_posts_for_triage[:3]:
            created = get_sort_key(p, "created_at").strftime("%Y-%m-%d %H:%M UTC")
            post_id = str(p.get("id") or "")
            text = (p.get("text") or "").strip().replace("\n", " ")[:180]
            snippet_lines.append(f"- {created} {post_id}: {text}")

        msg = (
            "🔎 OSINT Monitoring match\n"
            f"Target: {target}\n"
            f"Condition: {condition}\n"
            f"Reason: {details.get('reason','')}\n"
            f"Matched keywords: {matched_keywords_str}\n"
            "\n"
            "Matched evidence snippets:\n"
            + "\n".join(snippet_lines)
            + "\n"
        )
        logger.info("Monitoring match alert sent for %s", target)
        await self.send_rule_alert(rule=rule, text=msg)
        return True

    async def run_forever(self) -> None:
        while True:
            try:
                await self._run_once()
            except Exception as e:
                logger.exception("Watcher iteration failed: %s", e)
            await asyncio.sleep(self.poll_interval_seconds)

    async def _run_once(self) -> None:
        # Load persisted sessions on each iteration; small number expected.
        any_updates = 0
        for path in self.session_manager.sessions_dir.glob("*.json"):
            session_id = path.stem
            session = self.session_manager.load(session_id)
            if not session:
                continue

            if not session.monitoring_rules:
                continue

            session_changed = False
            for rule in session.monitoring_rules:
                updated = await self._evaluate_rule(session_id, session.name, rule)
                if updated:
                    session_changed = True
                    any_updates += 1

            if session_changed:
                self.session_manager.save(session)

        if any_updates:
            logger.debug("Watcher advanced %s monitoring rule(s)", any_updates)


async def run_watcher_forever(
    *,
    agent: Any,  # SocialOSINTAgent
    session_manager: SessionManager,
    bot: Bot,
    poll_interval_seconds: Optional[int] = None,
    fetch_limit: Optional[int] = None,
    triage_post_limit: Optional[int] = None,
) -> None:
    """
    Convenience entrypoint used by `socialosintagent.bot` to run Phase 3.
    """
    interval = poll_interval_seconds or int(os.getenv("OSINT_WATCH_INTERVAL_SECONDS", "300"))
    f_limit = fetch_limit or int(os.getenv("OSINT_WATCH_FETCH_LIMIT", "20"))
    t_limit = triage_post_limit or int(os.getenv("OSINT_WATCH_TRIAGE_POST_LIMIT", "8"))

    watcher = MonitoringWatcher(
        agent=agent,
        session_manager=session_manager,
        bot=bot,
        poll_interval_seconds=interval,
        fetch_limit=f_limit,
        triage_post_limit=t_limit,
    )
    await watcher.run_forever()

