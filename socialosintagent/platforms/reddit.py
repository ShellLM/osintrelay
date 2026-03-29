import logging
from typing import Any, List, Optional, Tuple
from datetime import datetime, timezone
from .base_fetcher import BaseFetcher
from ..utils import (SUPPORTED_IMAGE_EXTENSIONS, NormalizedMedia,
                     NormalizedPost, NormalizedProfile, download_media)

logger = logging.getLogger("SocialOSINTAgent.platforms.reddit")

class RedditFetcher(BaseFetcher):
    def __init__(self):
        super().__init__(platform_name="reddit")

    def _fetch_profile(self, username: str, **kwargs) -> Optional[NormalizedProfile]:
        r = kwargs.get("client").redditor(username)
        return NormalizedProfile(
            platform="reddit", id=r.id, username=r.name,
            created_at=datetime.fromtimestamp(r.created_utc, tz=timezone.utc),
            profile_url=f"https://reddit.com/u/{r.name}",
            metrics={"post_karma": r.link_karma, "comment_karma": r.comment_karma}
        )

    def _fetch_batch(self, username: str, profile: NormalizedProfile, needed: int, state: Any, **kwargs) -> Tuple[List[Any], Any]:
        if state == "done": return [], None
        r = kwargs.get("client").redditor(username)
        items = []
        for item in r.submissions.new(limit=needed):
            items.append({"data": item, "type": "submission"})
        for item in r.comments.new(limit=needed):
            items.append({"data": item, "type": "comment"})
        return items, "done"

    def _normalize(self, item: Any, profile: NormalizedProfile, **kwargs) -> NormalizedPost:
        obj, itype = item["data"], item["type"]
        cache, allow_ext = kwargs.get("cache"), kwargs.get("allow_external_media", False)
        
        text = f"Title: {getattr(obj, 'title', '')}\n\n{getattr(obj, 'selftext', '')}" if itype == "submission" else obj.body
        media = []
        
        if itype == "submission" and hasattr(obj, 'url'):
            if any(obj.url.lower().endswith(ex) for ex in SUPPORTED_IMAGE_EXTENSIONS):
                p = download_media(cache.base_dir, obj.url, cache.is_offline, "reddit", allow_external=allow_ext)
                if p: media.append(NormalizedMedia(url=obj.url, local_path=str(p), type="image"))

        return NormalizedPost(
            platform="reddit", id=obj.id, 
            created_at=datetime.fromtimestamp(obj.created_utc, tz=timezone.utc),
            author_username=profile["username"], text=text, media=media,
            post_url=f"https://reddit.com{obj.permalink}", 
            metrics={"score": obj.score}, type=itype,
            context={"subreddit": str(obj.subreddit.display_name)}
        )

def fetch_data(**kwargs):
    return RedditFetcher().fetch_data(kwargs.pop("username"), kwargs.pop("cache"), **kwargs)