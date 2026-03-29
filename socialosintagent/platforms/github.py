import logging, os, re, httpx
from typing import Any, Dict, List, Optional, Set, Tuple
from .base_fetcher import BaseFetcher, RateLimitHandler
from ..utils import NormalizedPost, NormalizedProfile, get_sort_key, extract_and_resolve_urls

logger = logging.getLogger("SocialOSINTAgent.platforms.github")

class GitHubFetcher(BaseFetcher):
    def __init__(self):
        super().__init__(platform_name="github")
        self.base_url = "https://api.github.com"

    def _fetch_profile(self, username: str, **kwargs) -> Optional[NormalizedProfile]:
        headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "SocialOSINTAgent"}
        if tk := os.getenv("GITHUB_TOKEN"): headers["Authorization"] = f"Bearer {tk}"
        with httpx.Client(headers=headers, timeout=20.0) as client:
            resp = client.get(f"{self.base_url}/users/{username}")
            RateLimitHandler.check_response_headers(resp.headers, "github")
            resp.raise_for_status()
            d = resp.json()
            return NormalizedProfile(
                platform="github", id=str(d["id"]), username=d["login"],
                display_name=d.get("name"), bio=d.get("bio"),
                created_at=get_sort_key(d, "created_at"), profile_url=d["html_url"],
                metrics={"followers": d.get("followers", 0), "repos": d.get("public_repos", 0)}
            )

    def _fetch_batch(self, username, profile, needed, state, **kwargs) -> Tuple[List[Any], Any]:
        page = state or 1
        headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "SocialOSINTAgent"}
        if tk := os.getenv("GITHUB_TOKEN"): headers["Authorization"] = f"Bearer {tk}"
        
        # Get deep analysis options from kwargs
        deep_analysis = kwargs.get("github_deep_analysis", False)
        max_patches = kwargs.get("github_max_patches", 20)
        patches_fetched = kwargs.get("_patches_fetched", 0)
        
        with httpx.Client(headers=headers, timeout=20.0) as client:
            resp = client.get(f"{self.base_url}/users/{username}/events/public", params={"per_page": 100, "page": page})
            RateLimitHandler.check_response_headers(resp.headers, "github")
            events = resp.json()
            
            # Optionally analyze patches for PushEvents
            if deep_analysis and patches_fetched < max_patches:
                for ev in events:
                    if patches_fetched >= max_patches:
                        break
                    if ev["type"] == "PushEvent" and self._is_interesting_commit(ev):
                        patch_data = self._analyze_patch(ev, client)
                        if patch_data:
                            if "context" not in ev:
                                ev["context"] = {}
                            ev["context"]["patch_analysis"] = patch_data
                            patches_fetched += 1
                
                # Store patch count for next iteration
                kwargs["_patches_fetched"] = patches_fetched
                if patches_fetched > 0:
                    logger.info(f"Analyzed {patches_fetched} patches for {username}")
            
            return events, (page + 1 if len(events) >= 100 else None)

    def _normalize(self, ev: Any, profile: NormalizedProfile, **kwargs) -> NormalizedPost:
        etype, repo = ev["type"], ev.get("repo", {}).get("name", "unknown")
        payload = ev.get("payload", {})
        text = f"Activity on {repo}"
        
        if etype == "PushEvent":
            commits = payload.get('commits', [])
            text = f"Pushed to {repo}:\n" + "\n".join([f"- {c['sha'][:7]}: {c['message']}" for c in commits])
        elif etype == "WatchEvent": 
            text = f"Starred {repo}"

        # Extract any patch analysis from context
        context = {"repo": repo}
        if "context" in ev and "patch_analysis" in ev["context"]:
            context["patch_analysis"] = ev["context"]["patch_analysis"]

        return NormalizedPost(
            platform="github", id=ev["id"], created_at=get_sort_key(ev, "created_at"),
            author_username=profile["username"], text=text, media=[],
            external_links=extract_and_resolve_urls(text), post_url=f"https://github.com/{repo}", 
            type=etype, context=context
        )

    def _is_interesting_commit(self, event: Dict) -> bool:
        """Quick heuristic to decide if commit is worth analyzing."""
        import random
        payload = event.get("payload", {})
        commits = payload.get("commits", [])
        
        if not commits:
            return False
        
        # Check for security keywords or large commits
        for commit in commits:
            msg = commit.get("message", "").lower()
            if len(msg) > 200 or any(kw in msg for kw in ["security", "cve", "auth", "crypto"]):
                return True
        
        # 10% random sampling
        return random.random() < 0.10

    def _analyze_patch(self, event: Dict, client: httpx.Client) -> Optional[Dict[str, Any]]:
        """Fetch patch and extract key OSINT signals."""
        payload = event.get("payload", {})
        commits = payload.get("commits", [])
        repo_name = event.get("repo", {}).get("name", "unknown")
        
        if not commits:
            return None
        
        commit = commits[0]
        commit_sha = commit.get("sha")
        if not commit_sha:
            return None
        
        patch_url = f"https://github.com/{repo_name}/commit/{commit_sha}.patch"
        
        try:
            resp = client.get(patch_url, timeout=15.0)
            resp.raise_for_status()
            patch = resp.text
            
            # Extract key signals
            return {
                "commit_sha": commit_sha[:7],
                "commit_url": f"https://github.com/{repo_name}/commit/{commit_sha}",
                "author_email": self._extract_email(patch, "From:"),
                "committer_email": self._extract_email(patch, "Committer:"),
                "languages": list(self._detect_languages(patch)),
                "lines_changed": self._count_changes(patch),
                "has_tests": "test" in patch.lower(),
                "security_related": any(kw in patch.lower() for kw in ["security", "auth", "crypto", "password"]),
            }
        
        except Exception as e:
            logger.debug(f"Could not analyze patch for {commit_sha[:7]}: {e}")
            return None

    def _extract_email(self, patch: str, prefix: str) -> Optional[str]:
        """Extract email from patch header."""
        match = re.search(f'^{prefix}.*?<(.+?)>', patch, re.MULTILINE)
        return match.group(1) if match else None

    def _detect_languages(self, patch: str) -> Set[str]:
        """Detect languages from file extensions."""
        paths = re.findall(r'diff --git a/(.+?) b/', patch)
        ext_map = {
            '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript', 
            '.java': 'Java', '.go': 'Go', '.rs': 'Rust', '.cpp': 'C++',
            '.rb': 'Ruby', '.php': 'PHP', '.swift': 'Swift'
        }
        langs = set()
        for path in paths[:20]:
            for ext, lang in ext_map.items():
                if path.lower().endswith(ext):
                    langs.add(lang)
        return langs

    def _count_changes(self, patch: str) -> Dict[str, int]:
        """Count added/removed lines."""
        lines = patch.split('\n')
        return {
            "added": sum(1 for l in lines if l.startswith('+') and not l.startswith('+++')),
            "removed": sum(1 for l in lines if l.startswith('-') and not l.startswith('---'))
        }


def fetch_data(**kwargs):
    return GitHubFetcher().fetch_data(kwargs.pop("username"), kwargs.pop("cache"), **kwargs)