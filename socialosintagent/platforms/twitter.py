import logging
from typing import Any, Dict, List, Optional, Tuple
import tweepy
from .base_fetcher import BaseFetcher
from ..utils import NormalizedMedia, NormalizedPost, NormalizedProfile, download_media, extract_and_resolve_urls

logger = logging.getLogger("SocialOSINTAgent.platforms.twitter")

class TwitterFetcher(BaseFetcher):
    def __init__(self):
        super().__init__(platform_name="twitter")

    def _fetch_profile(self, username: str, **kwargs) -> Optional[NormalizedProfile]:
        client: tweepy.Client = kwargs.get("client")
        res = client.get_user(
            username=username,
            user_fields=["created_at", "public_metrics", "description", "location", "verified"]
        )
        if not res or not res.data: return None
        u = res.data
        return NormalizedProfile(
            platform="twitter", id=str(u.id), username=u.username,
            display_name=u.name, bio=u.description, created_at=u.created_at,
            profile_url=f"https://twitter.com/{u.username}",
            metrics={
                "followers": u.public_metrics.get("followers_count", 0),
                "post_count": u.public_metrics.get("tweet_count", 0),
                "location": u.location or "N/A"
            }
        )

    def _fetch_batch(self, username: str, profile: NormalizedProfile, needed: int, state: Any, **kwargs) -> Tuple[List[Any], Any]:
        client = kwargs.get("client")
        limit = max(min(needed, 100), 5)
        res = client.get_users_tweets(
            id=profile["id"], 
            max_results=limit, 
            pagination_token=state,
            tweet_fields=["created_at", "public_metrics", "attachments", "in_reply_to_user_id", "author_id"],
            expansions=["attachments.media_keys", "author_id"],
            media_fields=["url", "preview_image_url", "type"],
            user_fields=["username"]
        )
        if not res or not res.data: return [], None
        
        media_map = {m.media_key: m for m in res.includes.get("media", [])} if res.includes else {}
        user_map = {u.id: u for u in res.includes.get("users", [])} if res.includes else {}
        
        wrapped_items = [{"tweet": t, "media_map": media_map, "user_map": user_map} for t in res.data]
        return wrapped_items, res.meta.get("next_token")

    def _normalize(self, item: Any, profile: NormalizedProfile, **kwargs) -> NormalizedPost:
        t, media_map, user_map = item["tweet"], item["media_map"], item["user_map"]
        cache, allow_ext, client = kwargs.get("cache"), kwargs.get("allow_external_media", False), kwargs.get("client")
        
        media_items = []
        if t.attachments and "media_keys" in t.attachments:
            for k in t.attachments["media_keys"]:
                if m := media_map.get(k):
                    url = m.url or m.preview_image_url
 
                    if path := download_media(cache.base_dir, url, cache.is_offline, "twitter", {"bearer_token": client.bearer_token}, allow_ext):
                        media_items.append(NormalizedMedia(url=url, local_path=str(path), type=m.type))
        
        author_user = user_map.get(t.author_id)
        author_handle = author_user.username if author_user else profile["username"]
        
        return NormalizedPost(
            platform="twitter", id=str(t.id), created_at=t.created_at, 
            author_username=author_handle, text=t.text, media=media_items, 
            external_links=extract_and_resolve_urls(t.text),
            post_url=f"https://twitter.com/i/status/{t.id}",
            metrics={"likes": t.public_metrics.get("like_count", 0), "reposts": t.public_metrics.get("retweet_count", 0)},
            type="reply" if t.in_reply_to_user_id else "post"
        )

def fetch_data(**kwargs):
    return TwitterFetcher().fetch_data(kwargs.pop("username"), kwargs.pop("cache"), **kwargs)