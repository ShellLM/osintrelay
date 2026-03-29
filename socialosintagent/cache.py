"""Manages the file-based caching of API responses and media."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .utils import DateTimeEncoder, UserData, get_sort_key

logger = logging.getLogger("SocialOSINTAgent.cache")

MAX_CACHE_ITEMS = 200
CACHE_EXPIRY_HOURS = 24

class CacheManager:
    """Handles saving and loading of normalized UserData to/from JSON files."""
    def __init__(self, base_dir: Path, is_offline: bool):
        """
        Initializes the CacheManager.

        Args:
            base_dir: The root directory for all data (e.g., 'data/').
            is_offline: If True, expired cache will still be returned.
        """
        self.base_dir = base_dir
        self.is_offline = is_offline
        self.cache_dir = self.base_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Instance-level path cache replaces @lru_cache, which leaks memory on
        # instance methods by holding a strong reference to `self` indefinitely,
        # preventing garbage collection of CacheManager instances.
        self._path_cache: dict = {}

    def get_cache_path(self, platform: str, username: str) -> Path:
        """
        Generates a standardized, safe file path for a given platform and username.
        
        Results are memoized in an instance-level dict for performance, avoiding
        repeated path constructions without the memory-leak risk of @lru_cache
        on instance methods.

        Args:
            platform: The name of the social media platform.
            username: The username of the target.

        Returns:
            A Path object for the cache file.
            
        Raises:
            ValueError: If username becomes empty after sanitization.
        """
        key = (platform, username)
        if key in self._path_cache:
            return self._path_cache[key]

        # Sanitize username to create a safe filename
        # Only allow: alphanumeric, hyphen, underscore, @
        # Explicitly EXCLUDE "." to prevent path traversal (e.g., "../../../etc/passwd")
        safe_username = "".join(c for c in username if c.isalnum() or c in ["-", "_", "@"])[:100]
        
        if not safe_username:
            raise ValueError(f"Username '{username}' is invalid after sanitization (became empty)")
        
        path = self.cache_dir / f"{platform}_{safe_username}.json"
        self._path_cache[key] = path
        return path

    def load(self, platform: str, username: str) -> Optional[UserData]:
        """
        Loads and validates a user's data from the cache.

        - Checks for file existence.
        - Validates that the file contains the required keys for the standard UserData model.
        - Checks for cache expiry, returning None if expired (unless in offline mode).

        Args:
            platform: The name of the social media platform.
            username: The username of the target.

        Returns:
            A UserData dictionary if a valid, non-expired cache file is found, otherwise None.
        """
        cache_path = self.get_cache_path(platform, username)
        if not cache_path.exists():
            return None
        
        try:
            data: UserData = json.loads(cache_path.read_text(encoding="utf-8"))
            
            # Universal validation: Ensure the cache file conforms to our standard data model.
            # This prevents loading of old, incompatible cache formats.
            if not all(k in data for k in ["timestamp", "profile", "posts"]):
                logger.warning(f"Cache file for {platform}/{username} is incomplete or in an old format. Discarding.")
                cache_path.unlink(missing_ok=True)
                return None
            
            # Fix top-level timestamp
            if isinstance(data.get("timestamp"), str):
                data["timestamp"] = get_sort_key(data, "timestamp")

            # Fix profile created_at
            if "profile" in data and isinstance(data["profile"].get("created_at"), str):
                data["profile"]["created_at"] = get_sort_key(data["profile"], "created_at")

            # Fix all posts created_at
            if "posts" in data:
                for post in data["posts"]:
                    if isinstance(post.get("created_at"), str):
                        post["created_at"] = get_sort_key(post, "created_at")

            timestamp = data["timestamp"]
            
            if self.is_offline:
                logger.info(f"Offline mode: Using potentially stale cache for {platform}/{username}.")
                return data

            # In online mode, check if the cache is expired.
            if (datetime.now(timezone.utc) - timestamp) < timedelta(hours=CACHE_EXPIRY_HOURS):
                logger.info(f"Cache hit and valid for {platform}/{username}")
                return data
            else:
                logger.info(f"Cache expired for {platform}/{username}. Discarding.")
                # We don't delete the file here, as the fetcher will overwrite it.
                return None

        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            logger.warning(f"Failed to load or parse cache for {platform}/{username}: {e}. Discarding.")
            cache_path.unlink(missing_ok=True)
            return None

    def save(self, platform: str, username: str, data: UserData):
        """
        Saves a UserData object to a JSON file in the cache.

        Automatically adds a timestamp and sorts posts before saving.

        Args:
            platform: The name of the social media platform.
            username: The username of the target.
            data: The UserData dictionary to save.
        """
        cache_path = self.get_cache_path(platform, username)
        try:
            # Ensure posts are always sorted chronologically, newest first.
            if "posts" in data and isinstance(data["posts"], list):
                data["posts"].sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True)
            
            # Add metadata before saving
            data["timestamp"] = datetime.now(timezone.utc)
            data["stats"] = {"total_posts_cached": len(data.get("posts", []))}
            
            cache_path.write_text(json.dumps(data, indent=2, cls=DateTimeEncoder), encoding="utf-8")
            logger.info(f"Saved cache for {platform}/{username} to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache for {platform}/{username}: {e}", exc_info=True)

    def delete(self, platform: str, username: str):
        """Deletes a specific user's cache file."""
        path = self.get_cache_path(platform, username)
        if path.exists():
            path.unlink()
            key = (platform, username)
            if key in self._path_cache:
                del self._path_cache[key]