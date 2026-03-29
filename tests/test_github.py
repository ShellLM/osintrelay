import pytest
from unittest.mock import MagicMock, ANY
from datetime import datetime, timezone
import httpx

from socialosintagent.platforms import github as github_fetcher
from socialosintagent.exceptions import UserNotFoundError

@pytest.fixture
def mock_httpx_client(mocker):
    """Provides a mocked httpx.Client context manager."""
    mock_client_instance = mocker.MagicMock(spec=httpx.Client)

    profile_data = {
        "id": 1, 
        "login": "testuser", 
        "name": "Test User", 
        "created_at": "2022-01-01T00:00:00Z",
        "followers": 10, 
        "following": 5, 
        "public_repos": 3,
        "html_url": "https://github.com/testuser"
    }
    
    event_data = [{
        "id": "123", "type": "PushEvent", "actor": {"login": "testuser"},
        "repo": {"name": "test/repo"}, "payload": {"commits": []},
        "created_at": "2023-01-01T12:00:00Z"
    }]
    
    mock_request = MagicMock()
    
    def get_side_effect(url, params=None):
        if 'events/public' in url:
            # FIX: If page > 1, return empty list to stop the while loop
            if params and params.get("page", 1) > 1:
                return httpx.Response(200, json=[], request=mock_request)
            
            return httpx.Response(
                200,
                json=event_data,
                headers={'x-ratelimit-remaining': '4999'},
                request=mock_request
            )
        
        if '/users/' in url:
            return httpx.Response(
                200,
                json=profile_data,
                headers={'x-ratelimit-remaining': '5000'},
                request=mock_request
            )
            
        return httpx.Response(404, request=mock_request)

    mock_client_instance.get.side_effect = get_side_effect
    
    mock_context_manager = mocker.MagicMock()
    mock_context_manager.__enter__.return_value = mock_client_instance
    return mock_context_manager

@pytest.fixture
def mock_cache(mocker):
    mock_cache = mocker.MagicMock()
    mock_cache.is_offline = False
    mock_cache.load.return_value = None
    return mock_cache

def test_github_fetch_data_cache_miss(mock_httpx_client, mock_cache, monkeypatch, mocker):
    """Test fetching github data on a cache miss."""
    monkeypatch.setenv("GITHUB_TOKEN", "fake_token")
    mocker.patch('httpx.Client', return_value=mock_httpx_client)

    result = github_fetcher.fetch_data(
        username='testuser',
        cache=mock_cache,
        fetch_limit=50
    )

    assert result is not None
    assert result['profile']['username'] == 'testuser'
    assert len(result['posts']) == 1
    mock_cache.save.assert_called_once()

def test_github_user_not_found(mock_httpx_client, mock_cache, monkeypatch, mocker):
    """Test UserNotFoundError for a 404 response."""
    monkeypatch.setenv("GITHUB_TOKEN", "fake_token")
    
    mock_client_instance = mock_httpx_client.__enter__.return_value
    mock_client_instance.get.side_effect = httpx.HTTPStatusError(
        "Not Found", request=MagicMock(), response=httpx.Response(404)
    )
    mocker.patch('httpx.Client', return_value=mock_httpx_client)

    with pytest.raises(UserNotFoundError):
        github_fetcher.fetch_data(username='nonexistent', cache=mock_cache)