import json
from datetime import datetime, timedelta, timezone

import pytest

from socialosintagent.cache import CacheManager
from socialosintagent.utils import UserData

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory for testing."""
    cache_dir = tmp_path / "data"
    (cache_dir / "cache").mkdir(parents=True, exist_ok=True)
    return cache_dir

def test_save_and_load(temp_cache_dir):
    """Test that data can be saved and loaded correctly."""
    # Arrange
    cache = CacheManager(base_dir=temp_cache_dir, is_offline=False)
    platform = "test_platform"
    username = "test_user"
    data_to_save: UserData = {
        "profile": {"id": "123", "username": "test_user"}, 
        "posts": [{"id": "t1", "text": "hello"}]
    }

    # Act
    cache.save(platform, username, data_to_save)
    loaded_data = cache.load(platform, username)

    # Assert
    assert loaded_data is not None
    assert loaded_data["profile"]["id"] == "123"
    assert "timestamp" in loaded_data
    assert cache.get_cache_path(platform, username).exists()

def test_cache_expiry_returns_none_in_online_mode(temp_cache_dir):
    """Test that stale/expired cache returns None in online mode."""
    # Arrange
    cache = CacheManager(base_dir=temp_cache_dir, is_offline=False)
    platform = "test_platform"
    username = "expired_user"
    
    old_time = datetime.now(timezone.utc) - timedelta(hours=48)
    
    cache_path = cache.get_cache_path(platform, username)
    data_to_write: UserData = {
        "timestamp": old_time.isoformat(),
        "posts": [],
        "profile": {"id": "456"},
    }
    cache_path.write_text(json.dumps(data_to_write))

    # Act
    loaded_data_online = cache.load(platform, username) 

    # Assert
    assert loaded_data_online is None

def test_offline_mode_returns_stale_cache(temp_cache_dir):
    """Test that stale cache IS returned in offline mode."""
    # Arrange
    cache = CacheManager(base_dir=temp_cache_dir, is_offline=True)
    platform = "test_platform"
    username = "offline_user"
    
    old_time = datetime.now(timezone.utc) - timedelta(hours=48)
    
    cache_path = cache.get_cache_path(platform, username)
    data_to_write: UserData = {
        "timestamp": old_time.isoformat(),
        "posts": [{"id": "t1"}],
        "profile": {"id": "789"},
    }
    cache_path.write_text(json.dumps(data_to_write))

    # Act
    loaded_data = cache.load(platform, username)

    # Assert
    assert loaded_data is not None
    assert loaded_data["profile"]["id"] == "789"