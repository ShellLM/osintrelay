import logging, httpx
from typing import Any, List, Optional, Tuple
from .base_fetcher import BaseFetcher
from ..utils import NormalizedPost, NormalizedProfile, extract_and_resolve_urls, get_sort_key

logger = logging.getLogger("SocialOSINTAgent.platforms.hackernews")

class HackerNewsFetcher(BaseFetcher):
    def __init__(self):
        super().__init__(platform_name="hackernews")

    def _fetch_profile(self, username: str, **kwargs) -> Optional[NormalizedProfile]:
        url = f"https://hacker-news.firebaseio.com/v0/user/{username}.json"
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            d = resp.json()
            if not d: return None
            return NormalizedProfile(
                platform="hackernews", id=username, username=username,
                bio=d.get("about", ""), created_at=get_sort_key({"ts": d.get("created")}, "ts"),
                profile_url=f"https://news.ycombinator.com/user?id={username}",
                metrics={"karma": d.get("karma", 0)}
            )

    def _fetch_batch(self, username, profile, needed, state, **kwargs) -> Tuple[List[Any], Any]:
        if state == "done": return [], None
        url = f"https://hn.algolia.com/api/v1/search?tags=author_{username}&hitsPerPage={needed}"
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
            return hits, "done"

    def _normalize(self, hit: Any, profile: NormalizedProfile, **kwargs) -> NormalizedPost:
        from bs4 import BeautifulSoup
        raw_text = hit.get("comment_text") or hit.get("story_text") or ""
        clean_text = BeautifulSoup(raw_text, "html.parser").get_text(separator=" ")
        itype = "comment" if "comment" in hit.get("_tags", []) else "story"
        if itype == "story" and hit.get("title"): clean_text = f"Title: {hit['title']}\n\n{clean_text}".strip()
        
        links = extract_and_resolve_urls(clean_text)
        if hit.get("url"): links.append(hit["url"])
        
        return NormalizedPost(
            platform="hackernews", id=hit["objectID"], author_username=profile["username"],
            text=clean_text, created_at=get_sort_key({"ts": hit["created_at_i"]}, "ts"),
            post_url=f"https://news.ycombinator.com/item?id={hit['objectID']}",
            metrics={"score": hit.get("points", 0), "comment_count": hit.get("num_comments", 0)},
            type=itype, external_links=list(set(links))
        )

def fetch_data(**kwargs):
    return HackerNewsFetcher().fetch_data(kwargs.pop("username"), kwargs.pop("cache"), **kwargs)