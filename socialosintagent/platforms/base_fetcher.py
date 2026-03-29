"""
Unified base fetcher with consistent rate limit handling and error management.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

from ..cache import MAX_CACHE_ITEMS, CacheManager
from ..exceptions import (
    AccessForbiddenError,
    RateLimitExceededError,
    UserNotFoundError,
)
from ..utils import UserData, NormalizedProfile, NormalizedPost, get_sort_key

logger = logging.getLogger("SocialOSINTAgent.base_fetcher")


class BaseFetcher(ABC):
    def __init__(self, platform_name: str):
        self.platform_name = platform_name
        self.logger = logging.getLogger(f"SocialOSINTAgent.platforms.{platform_name}")

    def fetch_data(
        self,
        username: str,
        cache: CacheManager,
        force_refresh: bool = False,
        fetch_limit: int = 50,
        **kwargs,
    ) -> Optional[UserData]:
        # Handle offline mode
        cached_data = cache.load(self.platform_name, username)
        if cache.is_offline:
            return cached_data

        # Cache Hit Optimization
        if not force_refresh and cached_data:
            if len(cached_data.get("posts", [])) >= fetch_limit:
                self.logger.info(f"Cache hit for {self.platform_name}/{username}")
                return cached_data

        # Profile Logic
        profile_obj = cached_data.get("profile") if (not force_refresh and cached_data) else None

        if not profile_obj:
            try:
                profile_obj = self._fetch_profile(username, **kwargs)
            except Exception as e:
                self._handle_api_error(e, username)

        if not profile_obj:
            raise UserNotFoundError(f"{self.platform_name} user '{username}' not found")

        # Post Logic: Centralized Loop
        all_posts = (
            cached_data.get("posts", [])
            if not force_refresh and cached_data
            else []
        )
        processed_post_ids = {p["id"] for p in all_posts}
        
        pagination_state = None
        while len(all_posts) < fetch_limit:
            needed = fetch_limit - len(all_posts)
            try:
                batch, pagination_state = self._fetch_batch(
                    username, profile_obj, needed, pagination_state, cache=cache, **kwargs
                )
                
                if not batch:
                    break
                
                for item in batch:
                    normalized = self._normalize(item, profile_obj, cache=cache, **kwargs)
                    if normalized["id"] not in processed_post_ids:
                        all_posts.append(normalized)
                        processed_post_ids.add(normalized["id"])
                
                if not pagination_state:
                    break
            except Exception as e:
                self._handle_api_error(e, username)
                break

        # Finalization
        final_posts = sorted(
            all_posts, key=lambda x: get_sort_key(x, "created_at"), reverse=True
        )[: max(fetch_limit, MAX_CACHE_ITEMS)]

        user_data = UserData(profile=profile_obj, posts=final_posts)
        cache.save(self.platform_name, username, user_data)
        return user_data

    @abstractmethod
    def _fetch_profile(self, username: str, **kwargs) -> Optional[NormalizedProfile]:
        pass

    @abstractmethod
    def _fetch_batch(self, username: str, profile: NormalizedProfile, needed: int, state: Any, **kwargs) -> Tuple[List[Any], Any]:
        pass

    @abstractmethod
    def _normalize(self, item: Any, profile: NormalizedProfile, **kwargs) -> NormalizedPost:
        pass

    def _handle_api_error(self, error: Exception, username: str) -> None:
        error_str = str(error).lower()
        if any(p in error_str for p in ["rate limit", "too many requests", "429"]):
            raise RateLimitExceededError(f"{self.platform_name} API rate limit exceeded", original_exception=error)
        if any(p in error_str for p in ["not found", "404", "does not exist"]):
            raise UserNotFoundError(f"{self.platform_name} user '{username}' not found") from error
        if any(p in error_str for p in ["forbidden", "403", "private", "suspended"]):
            raise AccessForbiddenError(f"Access to {self.platform_name} user '{username}' is forbidden") from error
        raise error

class RateLimitHandler:
    @staticmethod
    def check_response_headers(headers: Dict[str, str], platform: str) -> None:
        if "x-ratelimit-remaining" in headers:
            remaining = headers.get("x-ratelimit-remaining", "1")
            if remaining.isdigit() and int(remaining) == 0:
                raise RateLimitExceededError(f"{platform} API rate limit exceeded")
        if "retry-after" in headers:
            raise RateLimitExceededError(f"{platform} API rate limit exceeded")