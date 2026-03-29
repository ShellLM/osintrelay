from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
import tweepy

from socialosintagent.exceptions import UserNotFoundError
from socialosintagent.platforms import twitter as twitter_fetcher


@pytest.fixture
def mock_tweepy_client(mocker):
    client = mocker.MagicMock(spec=tweepy.Client)
    client.bearer_token = "fake_bearer_token"
    
    mock_user = MagicMock(
        id=12345, name="Test User", username="testuser",
        created_at=datetime(2022, 1, 1, tzinfo=timezone.utc),
        public_metrics={"followers_count": 100, "following_count": 50, "tweet_count": 10},
        description="A test user"
    )
    mock_tweet = MagicMock(spec=tweepy.Tweet)
    mock_tweet.id = 54321
    mock_tweet.author_id = 12345
    mock_tweet.text = "Hello world!"
    mock_tweet.created_at = datetime.now(timezone.utc) - timedelta(hours=1)
    mock_tweet.public_metrics = {}
    mock_tweet.attachments = None
    mock_tweet.in_reply_to_user_id = None
    
    client.get_user.return_value = MagicMock(data=mock_user)
    client.get_users_tweets.return_value = MagicMock(
        data=[mock_tweet], includes={}, meta={"next_token": None}
    )
    return client

@pytest.fixture
def mock_cache(mocker):
    return mocker.MagicMock()

def test_fetch_data_cache_miss(mock_tweepy_client, mock_cache):
    """Test fetch_data when no cache exists."""
    # Arrange
    mock_cache.load.return_value = None
    mock_cache.is_offline = False
    username = "testuser"

    # Act
    result = twitter_fetcher.fetch_data(
        client=mock_tweepy_client,
        username=username,
        cache=mock_cache,
        force_refresh=False,
        fetch_limit=50,
    )

    # Assert
    mock_cache.load.assert_called_once_with("twitter", username)
    mock_tweepy_client.get_user.assert_called_once()
    mock_tweepy_client.get_users_tweets.assert_called()
    mock_cache.save.assert_called_once()
    assert result is not None
    assert len(result["posts"]) == 1
    assert result["posts"][0]["text"] == "Hello world!"
    assert result["profile"]["username"] == "testuser"

def test_fetch_data_cache_hit_sufficient_items(mock_tweepy_client, mock_cache):
    """Test that API is not called when cache has enough items."""
    # Arrange
    cached_data = {
        "profile": {"id": "123"},
        "posts": [{"id": f"t{i}"} for i in range(50)],
    }
    mock_cache.load.return_value = cached_data
    username = "testuser"

    # Act
    twitter_fetcher.fetch_data(
        client=mock_tweepy_client, username=username, cache=mock_cache, fetch_limit=50
    )

    # Assert
    mock_tweepy_client.get_user.assert_not_called()
    mock_tweepy_client.get_users_tweets.assert_not_called()

def test_user_not_found(mock_tweepy_client, mock_cache):
    """Test that UserNotFoundError is raised for a non-existent user."""
    # Arrange
    mock_cache.load.return_value = None
    mock_cache.is_offline = False
    mock_tweepy_client.get_user.return_value = MagicMock(data=None)
    username = "nonexistent"

    # Act & Assert
    with pytest.raises(UserNotFoundError):
        twitter_fetcher.fetch_data(
            client=mock_tweepy_client,
            username=username,
            cache=mock_cache
        )