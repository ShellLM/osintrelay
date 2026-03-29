import logging
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from .base_fetcher import BaseFetcher
from ..utils import (NormalizedMedia, NormalizedPost, NormalizedProfile,
                     SAFE_CDN_DOMAINS, download_media, extract_and_resolve_urls)

logger = logging.getLogger("SocialOSINTAgent.platforms.mastodon")

class MastodonFetcher(BaseFetcher):
    def __init__(self):
        super().__init__(platform_name="mastodon")

    def _get_client(self, username, **kwargs):
        instance = username.split('@')[1]
        instance_url = f"https://{instance}"
        return kwargs.get("clients", {}).get(instance_url) or kwargs.get("default_client")

    def _fetch_profile(self, username: str, **kwargs) -> Optional[NormalizedProfile]:
        client = self._get_client(username, **kwargs)
        account = client.account_lookup(acct=username)
        return NormalizedProfile(
            platform="mastodon", id=str(account["id"]), username=account["acct"],
            display_name=account["display_name"], profile_url=account["url"],
            bio=BeautifulSoup(account.get("note", ""), "html.parser").get_text(separator=" ", strip=True),
            created_at=account["created_at"],
            metrics={"followers": account["followers_count"], "post_count": account["statuses_count"]}
        )

    def _fetch_batch(self, username, profile, needed, state, **kwargs) -> Tuple[List[Any], Any]:
        client = self._get_client(username, **kwargs)
        statuses = client.account_statuses(id=profile["id"], limit=min(needed, 40), max_id=state)
        next_state = statuses[-1]['id'] if statuses else None
        return statuses, next_state

    def _normalize(self, status: Any, profile: NormalizedProfile, **kwargs) -> NormalizedPost:
        cache = kwargs.get("cache")
        allow_ext = kwargs.get("allow_external_media", False)

        # Extract the instance domain to trust it dynamically.
        # This was previously imported inside the for loop below (on every
        # iteration), which is both inefficient and semantically misleading.
        # SAFE_CDN_DOMAINS is now imported at module level alongside the
        # other utils imports.
        user_instance_domain = profile["username"].split('@')[1].lower() if '@' in profile["username"] else None

        cleaned_text = BeautifulSoup(status["content"], "html.parser").get_text(separator=" ", strip=True)
        media_items = []

        for att in status.get("media_attachments", []):
            media_url = att["url"]
            media_domain = urlparse(media_url).netloc.lower()

            # Check Dynamic Trust (matches the user's home server)
            is_internal_host = False
            if user_instance_domain:
                is_internal_host = (
                    media_domain == user_instance_domain
                    or media_domain.endswith(f".{user_instance_domain}")
                )

            # Check Static Trust (matches the list in utils.py)
            is_globally_safe = any(
                media_domain == d or media_domain.endswith(f".{d}")
                for d in SAFE_CDN_DOMAINS.get("mastodon", [])
            )

            # Download is allowed if ANY of these are true
            effective_allow_external = allow_ext or is_internal_host or is_globally_safe

            if path := download_media(
                cache.base_dir,
                media_url,
                cache.is_offline,
                "mastodon",
                allow_external=effective_allow_external
            ):
                media_items.append(NormalizedMedia(url=media_url, local_path=str(path), type=att["type"]))

        return NormalizedPost(
            platform="mastodon", id=str(status["id"]), created_at=status["created_at"],
            author_username=profile["username"], text=cleaned_text, media=media_items,
            external_links=extract_and_resolve_urls(cleaned_text), post_url=status["url"],
            metrics={"likes": status["favourites_count"], "reposts": status["reblogs_count"]},
            type="repost" if status.get("reblog") else "post"
        )


def fetch_data(**kwargs):
    return MastodonFetcher().fetch_data(kwargs.pop("username"), kwargs.pop("cache"), **kwargs)