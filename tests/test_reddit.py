import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

from socialosintagent.platforms import reddit as reddit_fetcher

@pytest.fixture
def mock_praw_client(mocker):
    """Provides a mocked praw.Reddit client."""
    client = mocker.MagicMock()
    mock_redditor = MagicMock()
    mock_redditor.id = 'u123'
    mock_redditor.name = 'reddituser'
    mock_redditor.created_utc = datetime(2020, 1, 1).timestamp()
    mock_redditor.link_karma = 100
    mock_redditor.comment_karma = 1000

    mock_submission = MagicMock()
    mock_submission.id = 's1'
    mock_submission.author = 'reddituser'
    mock_submission.title = 'My first post'
    mock_submission.selftext = 'Hello Reddit!'
    mock_submission.created_utc = (datetime.now(timezone.utc).timestamp()) - 100
    mock_submission.permalink = '/r/testing/s1'
    mock_submission.subreddit = MagicMock(display_name='testing')
    mock_submission.is_self = True

    mock_comment = MagicMock()
    mock_comment.id = 'c1'
    mock_comment.author = 'reddituser'
    mock_comment.body = 'A test comment'
    mock_comment.created_utc = datetime.now(timezone.utc).timestamp()
    mock_comment.permalink = '/r/testing/s1/c1'
    mock_comment.subreddit = MagicMock(display_name='testing')

    client.redditor.return_value = mock_redditor
    mock_redditor.submissions.new.return_value = [mock_submission]
    mock_redditor.comments.new.return_value = [mock_comment]
    
    return client

@pytest.fixture
def mock_cache(mocker):
    """Provides mocked CacheManager."""
    mock_cache = mocker.MagicMock()
    mock_cache.is_offline = False
    mock_cache.load.return_value = None # Cache miss
    return mock_cache

def test_reddit_fetch_data_cache_miss(mock_praw_client, mock_cache, mocker):
    """Test fetching reddit data on a cache miss."""
    # Arrange
    mocker.patch('socialosintagent.platforms.reddit.download_media', return_value=None)

    # Act
    result = reddit_fetcher.fetch_data(
        client=mock_praw_client,
        username='reddituser',
        cache=mock_cache,
        fetch_limit=50
    )

    # Assert
    mock_praw_client.redditor.assert_called_with('reddituser')
    assert result is not None
    assert len(result['posts']) == 2 # 1 submission + 1 comment
    assert result['profile']['username'] == 'reddituser'
    
    # Check that both post types were normalized correctly
    submission_post = next(p for p in result['posts'] if p['type'] == 'submission')
    comment_post = next(p for p in result['posts'] if p['type'] == 'comment')

    assert 'My first post' in submission_post['text']
    assert 'A test comment' in comment_post['text']
    
    mock_cache.save.assert_called_once()