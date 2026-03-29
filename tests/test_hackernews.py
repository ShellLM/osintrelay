"""
Covers:
- Cache miss: fetches via Algolia, normalises stories and comments correctly.
- Story normalisation: title is prefixed, text content is included.
- Comment normalisation: no title prefix, comment_text is included.
- HTML stripping: tags in story_text / comment_text are removed by BeautifulSoup.
- 429 rate limit: raises RateLimitExceededError.
- Offline mode: returns cached data directly without any HTTP call.
- Cache hit with sufficient posts: skips the API entirely.
- Empty hits list: returns a profile with zero posts.
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

import httpx

from socialosintagent.platforms import hackernews as hn_fetcher
from socialosintagent.exceptions import RateLimitExceededError

# Helpers

def _make_request():
    return MagicMock()


def _algolia_response(hits):
    """Build an httpx.Response wrapping a standard Algolia hits payload."""
    return httpx.Response(
        200,
        json={"hits": hits, "nbHits": len(hits)},
        request=_make_request(),
    )


def _story_hit(object_id="story_1", title="Big News", story_text="The body text.", created_at_i=1700000000, url="https://example.com/story"):
    return {
        "objectID": object_id,
        "_tags": ["story"],
        "title": title,
        "story_text": story_text,
        "comment_text": None,
        "created_at_i": created_at_i,
        "url": url,
        "points": 150,
        "num_comments": 42,
        "author": "hn_user",
    }


def _comment_hit(object_id="comment_1", comment_text="Great article!", created_at_i=1700000100):
    return {
        "objectID": object_id,
        "_tags": ["comment"],
        "title": None,
        "story_text": None,
        "comment_text": comment_text,
        "created_at_i": created_at_i,
        "url": None,
        "points": None,
        "num_comments": None,
        "author": "hn_user",
    }

# Fixtures

@pytest.fixture
def mock_cache(mocker):
    cache = mocker.MagicMock()
    cache.is_offline = False
    cache.load.return_value = None
    return cache


def _wire_hn_client(mocker, hits):
    """Patch httpx.Client so the Algolia GET returns the given hits."""
    mock_resp = _algolia_response(hits)
    mock_client_instance = mocker.MagicMock(spec=httpx.Client)
    mock_client_instance.get.return_value = mock_resp

    ctx = mocker.MagicMock()
    ctx.__enter__.return_value = mock_client_instance
    ctx.__exit__.return_value = False
    mocker.patch("httpx.Client", return_value=ctx)
    return mock_client_instance

# Tests

class TestHNFetchHappyPath:
    def test_cache_miss_fetches_and_normalises(self, mock_cache, mocker):
        """On a cache miss, data is fetched, normalised, saved, and returned."""
        hits = [_story_hit(), _comment_hit()]
        _wire_hn_client(mocker, hits)

        result = hn_fetcher.fetch_data(
            username="hn_user", cache=mock_cache, fetch_limit=50
        )

        assert result is not None
        assert result["profile"]["platform"] == "hackernews"
        assert result["profile"]["username"] == "hn_user"
        assert len(result["posts"]) == 2
        mock_cache.save.assert_called_once_with("hackernews", "hn_user", result)

    def test_empty_hits_returns_profile_with_no_posts(self, mock_cache, mocker):
        """An empty hits array produces a valid result with zero posts."""
        _wire_hn_client(mocker, [])

        result = hn_fetcher.fetch_data(
            username="quiet_user", cache=mock_cache, fetch_limit=50
        )

        assert result is not None
        assert result["profile"]["username"] == "quiet_user"
        assert len(result["posts"]) == 0


class TestHNNormalisation:
    def test_story_text_includes_title_prefix(self, mock_cache, mocker):
        """A story's normalised text begins with 'Title: <title>'."""
        hits = [_story_hit(title="Important Update", story_text="Details here.")]
        _wire_hn_client(mocker, hits)

        result = hn_fetcher.fetch_data(
            username="hn_user", cache=mock_cache, fetch_limit=50
        )

        story = result["posts"][0]
        assert story["type"] == "story"
        assert story["text"].startswith("Title: Important Update")
        assert "Details here." in story["text"]

    def test_comment_text_has_no_title_prefix(self, mock_cache, mocker):
        """A comment's normalised text is just the comment body, no title prefix."""
        hits = [_comment_hit(comment_text="Nice work!")]
        _wire_hn_client(mocker, hits)

        result = hn_fetcher.fetch_data(
            username="hn_user", cache=mock_cache, fetch_limit=50
        )

        comment = result["posts"][0]
        assert comment["type"] == "comment"
        assert comment["text"] == "Nice work!"
        assert "Title:" not in comment["text"]

    def test_html_tags_are_stripped_from_story_text(self, mock_cache, mocker):
        """HTML tags in story_text are removed by BeautifulSoup."""
        html_text = "<p>First paragraph.</p><br/><p>Second <b>bold</b> paragraph.</p>"
        hits = [_story_hit(story_text=html_text, title="HTML Story")]
        _wire_hn_client(mocker, hits)

        result = hn_fetcher.fetch_data(
            username="hn_user", cache=mock_cache, fetch_limit=50
        )

        story = result["posts"][0]
        # No HTML tags should remain
        assert "<p>" not in story["text"]
        assert "<b>" not in story["text"]
        # But the text content should be present
        assert "First paragraph." in story["text"]
        assert "bold" in story["text"]

    def test_html_tags_are_stripped_from_comment_text(self, mock_cache, mocker):
        """HTML tags in comment_text are removed."""
        html_comment = "<p>This is <a href='http://x.com'>linked</a> text.</p>"
        hits = [_comment_hit(comment_text=html_comment)]
        _wire_hn_client(mocker, hits)

        result = hn_fetcher.fetch_data(
            username="hn_user", cache=mock_cache, fetch_limit=50
        )

        comment = result["posts"][0]
        assert "<a" not in comment["text"]
        assert "linked" in comment["text"]

    def test_story_metrics_are_populated(self, mock_cache, mocker):
        """Story posts include score and comment_count in metrics."""
        hits = [_story_hit()]
        _wire_hn_client(mocker, hits)

        result = hn_fetcher.fetch_data(
            username="hn_user", cache=mock_cache, fetch_limit=50
        )

        story = result["posts"][0]
        assert story["metrics"]["score"] == 150
        assert story["metrics"]["comment_count"] == 42

    def test_post_url_points_to_hn_item(self, mock_cache, mocker):
        """The post_url is the canonical HN item URL."""
        hits = [_story_hit(object_id="42")]
        _wire_hn_client(mocker, hits)

        result = hn_fetcher.fetch_data(
            username="hn_user", cache=mock_cache, fetch_limit=50
        )

        assert result["posts"][0]["post_url"] == "https://news.ycombinator.com/item?id=42"

    def test_story_external_links_extracted_from_url(self, mock_cache, mocker):
        """The story's top-level URL is captured as an external link."""
        hits = [_story_hit(url="https://techcrunch.com/2024/big-news")]
        _wire_hn_client(mocker, hits)

        result = hn_fetcher.fetch_data(
            username="hn_user", cache=mock_cache, fetch_limit=50
        )

        story = result["posts"][0]
        assert "https://techcrunch.com/2024/big-news" in story["external_links"]


class TestHNCacheBehaviour:
    def test_offline_returns_cached_data(self, mocker):
        """Offline mode returns cached data without any HTTP call."""
        cached = {
            "profile": {"username": "offline_hn"},
            "posts": [{"id": "1", "text": "cached"}],
        }
        cache = mocker.MagicMock()
        cache.is_offline = True
        cache.load.return_value = cached

        # Wire a client that would crash if touched
        mock_client = mocker.MagicMock(spec=httpx.Client)
        mock_client.get.side_effect = AssertionError("No HTTP in offline mode")
        ctx = mocker.MagicMock()
        ctx.__enter__.return_value = mock_client
        mocker.patch("httpx.Client", return_value=ctx)

        result = hn_fetcher.fetch_data(
            username="offline_hn", cache=cache, fetch_limit=50
        )
        assert result is cached

    def test_cache_hit_sufficient_posts_skips_api(self, mocker):
        """When cached posts >= fetch_limit, no HTTP call is made."""
        cached = {
            "profile": {"username": "full_hn"},
            "posts": [{"id": str(i)} for i in range(100)],
        }
        cache = mocker.MagicMock()
        cache.is_offline = False
        cache.load.return_value = cached

        mock_client = mocker.MagicMock(spec=httpx.Client)
        mock_client.get.side_effect = AssertionError("No HTTP needed")
        ctx = mocker.MagicMock()
        ctx.__enter__.return_value = mock_client
        mocker.patch("httpx.Client", return_value=ctx)

        result = hn_fetcher.fetch_data(
            username="full_hn", cache=cache, fetch_limit=100
        )
        assert result is cached
        cache.save.assert_not_called()


class TestHNErrorPaths:
    def test_429_raises_rate_limit_error(self, mock_cache, mocker):
        """A 429 HTTP response raises RateLimitExceededError."""
        mock_resp = httpx.Response(
            429, request=_make_request(), text="rate limited"
        )
        mock_client = mocker.MagicMock(spec=httpx.Client)
        mock_client.get.return_value = mock_resp

        ctx = mocker.MagicMock()
        ctx.__enter__.return_value = mock_client
        mocker.patch("httpx.Client", return_value=ctx)

        with pytest.raises(RateLimitExceededError):
            hn_fetcher.fetch_data(
                username="hn_user", cache=mock_cache, fetch_limit=50
            )