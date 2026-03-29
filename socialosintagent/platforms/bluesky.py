import logging
from typing import Any, List, Optional, Tuple
from atproto import Client
from .base_fetcher import BaseFetcher
from ..utils import (NormalizedMedia, NormalizedPost, NormalizedProfile, download_media, get_sort_key)

logger = logging.getLogger("SocialOSINTAgent.platforms.bluesky")

class BlueskyFetcher(BaseFetcher):
    def __init__(self):
        super().__init__(platform_name="bluesky")

    def _fetch_profile(self, username: str, **kwargs) -> Optional[NormalizedProfile]:
        client = kwargs.get("client")
        p = client.get_profile(actor=username)
        return NormalizedProfile(
            platform="bluesky", id=p.did, username=p.handle,
            display_name=p.display_name, bio=p.description,
            profile_url=f"https://bsky.app/profile/{p.handle}",
            metrics={"followers": p.followers_count, "posts": p.posts_count, "did": p.did}
        )

    def _fetch_batch(self, username, profile, needed, state, **kwargs) -> Tuple[List[Any], Any]:
        client = kwargs.get("client")
        batch_limit = min(max(needed, 20), 100)
        resp = client.get_author_feed(actor=username, cursor=state, limit=batch_limit)
        return (resp.feed or []), resp.cursor

    def _normalize(self, item: Any, profile: NormalizedProfile, **kwargs) -> NormalizedPost:
        post = item.post
        client, cache, allow_ext = kwargs.get("client"), kwargs.get("cache"), kwargs.get("allow_external_media", False)
        media = []
        
        if hasattr(post, 'embed') and post.embed:
            images = []
            if hasattr(post.embed, 'images'): images = post.embed.images
            elif hasattr(post.embed, 'media') and hasattr(post.embed.media, 'images'): images = post.embed.media.images
            
            auth = {"access_jwt": getattr(client._session, 'access_jwt', None)}
            for img in images:
                url = getattr(img, 'thumb', f"https://cdn.bsky.app/img/feed_fullsize/plain/{post.author.did}/{getattr(img, 'cid', '')}")
                if p := download_media(cache.base_dir, url, cache.is_offline, "bluesky", auth, allow_ext):
                    media.append(NormalizedMedia(url=url, local_path=str(p), type="image"))

        return NormalizedPost(
            platform="bluesky", id=post.uri, author_username=profile["username"],
            text=getattr(post.record, "text", ""), media=media,
            created_at=get_sort_key({"ts": getattr(post.record, "created_at", None)}, "ts"),
            post_url=f"https://bsky.app/profile/{post.author.handle}/post/{post.uri.split('/')[-1]}",
            metrics={"likes": post.like_count, "replies": post.reply_count},
            type="reply" if hasattr(post.record, 'reply') and post.record.reply else "post"
        )

def fetch_data(**kwargs):
    return BlueskyFetcher().fetch_data(kwargs.pop("username"), kwargs.pop("cache"), **kwargs)