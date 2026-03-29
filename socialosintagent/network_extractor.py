"""
Deterministic network contact extraction from normalised posts.

Extracts accounts that a target user has directly interacted with —
mentions, retweets, boosts, replies, and repository interactions —
without any inference or LLM involvement.

Design principles
-----------------
- Zero inference: only structurally unambiguous signals are extracted.
- Platform-scoped: contacts are tied to the platform they were found on.
  Cross-platform identity matching is deliberately out of scope here.
- Self-exclusion: the source user is never listed as their own contact.
- Caller-supplied exclusions: other active session targets can be excluded
  so they don't clutter the discovered contacts panel.

Interaction types
-----------------
mention         @-mention in post text (all platforms)
retweet         "RT @user:" attribution in Twitter post text
repost_boost    Bluesky/Mastodon repost/boost (where author is parseable)
repo_owner      GitHub: the owner of a repo the target pushed to/starred
co_author       GitHub: @-mention in a commit message (deep analysis only)
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("SocialOSINTAgent.network_extractor")

# ---------------------------------------------------------------------------
# Per-platform mention patterns
# ---------------------------------------------------------------------------

# Twitter/X: 1–15 alphanumeric + underscore chars
_TWITTER_MENTION = re.compile(r'(?<!\w)@([A-Za-z0-9_]{1,15})(?!\w)')
# RT attribution at start of tweet: "RT @handle: ..."
_TWITTER_RT = re.compile(r'^RT @([A-Za-z0-9_]{1,15}):', re.IGNORECASE)

# Bluesky handles are domain-format: user.bsky.social or user.custom.tld
# Must contain at least one dot (distinguishes from plain @word)
_BLUESKY_MENTION = re.compile(
    r'(?<!\w)@([a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?'
    r'(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)+)'
)

# Mastodon: @user or federated @user@instance.domain
_MASTODON_MENTION = re.compile(r'(?<!\w)@([A-Za-z0-9_]+(?:@[A-Za-z0-9._-]+)?)(?!\w)')

# HackerNews: simple @username
_HN_MENTION = re.compile(r'(?<!\w)@([A-Za-z0-9_-]{2,25})(?!\w)')

# GitHub: @username in commit messages (1–39 chars, GitHub's limit)
_GITHUB_MENTION = re.compile(r'(?<!\w)@([A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)(?!\w)')

# Reddit: u/username (3–20 chars)
_REDDIT_MENTION = re.compile(r'(?:^|[\s,(])u/([A-Za-z0-9_-]{3,20})')

# Map platform name -> its mention pattern
_MENTION_PATTERNS: Dict[str, re.Pattern] = {
    "twitter":    _TWITTER_MENTION,
    "bluesky":    _BLUESKY_MENTION,
    "mastodon":   _MASTODON_MENTION,
    "hackernews": _HN_MENTION,
    "github":     _GITHUB_MENTION,
    "reddit":     _REDDIT_MENTION,
}

# Maximum source post IDs stored per contact (keeps the object small)
_MAX_SOURCE_POSTS = 10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DiscoveredContact:
    """
    A single network contact discovered from a target user's posts.

    Attributes:
        platform:          The platform this contact was found on.
        username:          The contact's username as it appeared in the post.
        interaction_types: Deduplicated list of how the contact was found.
        weight:            Total number of interactions observed.
        first_seen:        Timestamp of the earliest observed interaction.
        last_seen:         Timestamp of the most recent observed interaction.
        source_post_ids:   IDs of up to _MAX_SOURCE_POSTS posts that
                           generated this contact (for drill-down if needed).
    """
    platform: str
    username: str
    interaction_types: List[str] = field(default_factory=list)
    weight: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    source_post_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON / API responses."""
        return {
            "platform": self.platform,
            "username": self.username,
            "interaction_types": sorted(set(self.interaction_types)),
            "weight": self.weight,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }


# ---------------------------------------------------------------------------
# Internal accumulator
# ---------------------------------------------------------------------------

class _ContactAccumulator:
    """
    Collects (platform, username) → DiscoveredContact mappings, merging
    repeated interactions into a single weighted entry.

    Keyed on (platform, username_lowercase) to handle case variations.
    """

    def __init__(self) -> None:
        self._data: Dict[Tuple[str, str], DiscoveredContact] = {}

    def add(
        self,
        platform: str,
        username: str,
        interaction_type: str,
        post_id: str,
        timestamp: Optional[datetime],
    ) -> None:
        key = (platform, username.lower())
        if key not in self._data:
            self._data[key] = DiscoveredContact(
                platform=platform,
                # Preserve the first-seen casing for display
                username=username,
                first_seen=timestamp,
                last_seen=timestamp,
            )

        contact = self._data[key]
        contact.weight += 1

        if interaction_type not in contact.interaction_types:
            contact.interaction_types.append(interaction_type)

        if len(contact.source_post_ids) < _MAX_SOURCE_POSTS:
            if post_id not in contact.source_post_ids:
                contact.source_post_ids.append(post_id)

        if timestamp:
            if contact.first_seen is None or timestamp < contact.first_seen:
                contact.first_seen = timestamp
            if contact.last_seen is None or timestamp > contact.last_seen:
                contact.last_seen = timestamp

    def results(self) -> List[DiscoveredContact]:
        """Return contacts sorted by weight (highest first)."""
        return sorted(self._data.values(), key=lambda c: c.weight, reverse=True)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def _parse_timestamp(value) -> Optional[datetime]:
    """Coerce a post's created_at value to a timezone-aware datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _extract_from_posts(
    posts: List[dict],
    source_platform: str,
    source_username: str,
    exclude_usernames: Set[str],  # lowercase; other tracked targets to skip
    acc: _ContactAccumulator,
) -> None:
    """
    Walk a list of NormalizedPost dicts and feed all discovered interactions
    into the accumulator.

    This function is intentionally side-effect-only (writes to acc) so that
    a single accumulator can be shared across multiple targets, producing a
    naturally merged result without a separate merge pass.

    Args:
        posts:            Normalised post dicts for a single target.
        source_platform:  The platform these posts are from.
        source_username:  The owning target's username (excluded from results).
        exclude_usernames: Additional lowercase usernames to skip on this platform.
        acc:              Accumulator to write discovered contacts into.
    """
    source_lower = source_username.lower()
    # Combined exclusion set: always skip the source user themselves
    skip = exclude_usernames | {source_lower}

    for post in posts:
        platform = post.get("platform", source_platform)
        post_id  = str(post.get("id", ""))
        ts       = _parse_timestamp(post.get("created_at"))
        text     = post.get("text") or ""
        ptype    = post.get("type") or ""
        context  = post.get("context") or {}

        # ----------------------------------------------------------------
        # Twitter-specific: RT attribution
        # RT text format is "RT @handle: <original tweet text>".
        # The @mentions *inside* the RT body belong to the original author,
        # not to our target, so we record the RT source and skip further
        # mention scanning of this post's text.
        # ----------------------------------------------------------------
        if platform == "twitter":
            if m := _TWITTER_RT.match(text):
                handle = m.group(1)
                if handle.lower() not in skip:
                    acc.add(platform, handle, "retweet", post_id, ts)
                # Do not scan RT body for additional mentions — they belong
                # to the original tweet, not to our target's own words.
                continue

        # ----------------------------------------------------------------
        # Text @mentions (platform-specific pattern)
        # ----------------------------------------------------------------
        if text:
            pattern = _MENTION_PATTERNS.get(platform)
            if pattern:
                for m in pattern.finditer(text):
                    handle = m.group(1)
                    if handle.lower() not in skip:
                        acc.add(platform, handle, "mention", post_id, ts)

        # ----------------------------------------------------------------
        # GitHub: repo owner extraction
        # context["repo"] is "owner/reponame". The owner is a discoverable
        # GitHub user whenever they differ from the target.
        # ----------------------------------------------------------------
        if platform == "github":
            repo = context.get("repo", "")
            if "/" in repo:
                owner = repo.split("/", 1)[0]
                if owner.lower() not in skip:
                    acc.add(platform, owner, "repo_interaction", post_id, ts)

        # ----------------------------------------------------------------
        # Mastodon repost
        # The current normaliser sets type="repost" for reblogs but does not
        # store the original author's username in context (the Mastodon API
        # does provide it via status["reblog"]["account"]["acct"], but that
        # would require a normaliser change). Nothing to extract here yet —
        # this branch is a placeholder for when the normaliser is extended.
        # ----------------------------------------------------------------
        # if platform == "mastodon" and ptype == "repost":
        #     reblog_author = context.get("reblog_author")
        #     if reblog_author and reblog_author.lower() not in skip:
        #         acc.add(platform, reblog_author, "repost_boost", post_id, ts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_contacts(
    platform_posts: Dict[str, Dict[str, List[dict]]],
    active_targets: Dict[str, List[str]],
) -> List[DiscoveredContact]:
    """
    Extract all discovered contacts across an entire session's worth of posts.

    Args:
        platform_posts: Nested dict of platform → username → [NormalizedPost].
                        Only posts that are already loaded (from cache) should
                        be passed here — this function makes no API calls.
        active_targets: The session's current platforms dict
                        (platform → [usernames]). Used to build the exclusion
                        set so already-tracked users are not re-surfaced as
                        discovered contacts.

    Returns:
        List of DiscoveredContact sorted by weight descending, with all active
        targets already excluded.
    """
    # Build per-platform exclusion sets from all active targets
    exclude_per_platform: Dict[str, Set[str]] = {
        platform: {u.lower() for u in usernames}
        for platform, usernames in active_targets.items()
    }

    acc = _ContactAccumulator()

    for platform, user_posts in platform_posts.items():
        platform_exclude = exclude_per_platform.get(platform, set())
        for username, posts in user_posts.items():
            # Exclude all other tracked users on this platform, not just self
            _extract_from_posts(
                posts=posts,
                source_platform=platform,
                source_username=username,
                exclude_usernames=platform_exclude - {username.lower()},
                acc=acc,
            )

    return acc.results()
