from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from socialosintagent import utils

def test_sanitize_username_with_control_chars():
    """Test that Unicode control characters are stripped."""
    dirty_username = "user\u200bname"  # Contains a zero-width space
    clean_username = "username"
    assert utils.sanitize_username(dirty_username) == clean_username


def test_sanitize_username_no_change():
    """Test that a clean username is not modified."""
    clean_username = "user-123"
    assert utils.sanitize_username(clean_username) == clean_username


def test_extract_and_resolve_urls():
    """Test URL extraction regex."""
    text = "Check out my site at https://example.com/page?q=1 and also www.anothersite.net."
    urls = utils.extract_and_resolve_urls(text)
    assert "https://example.com/page?q=1" in urls
    assert "www.anothersite.net" in urls
    assert len(urls) == 2


def test_get_sort_key_from_iso_string():
    """Test parsing a standard ISO 8601 string."""
    item = {"created_at": "2023-01-01T12:00:00+00:00"}
    expected_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert utils.get_sort_key(item, "created_at") == expected_dt


def test_get_sort_key_from_timestamp():
    """Test parsing a Unix timestamp."""
    ts = 1672574400  # This is 2023-01-01 12:00:00 UTC
    item = {"created_utc": ts}
    expected_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert utils.get_sort_key(item, "created_utc") == expected_dt

# get_sort_key edge cases

class TestGetSortKey:
    def test_datetime_object_returned_directly(self):
        """A datetime value is returned as-is (with tz ensured)."""
        dt = datetime(2024, 7, 4, 15, 30, 0, tzinfo=timezone.utc)
        item = {"created_at": dt}
        assert utils.get_sort_key(item, "created_at") == dt

    def test_naive_datetime_gets_utc_attached(self):
        """A naive datetime (no tzinfo) gets UTC attached."""
        naive = datetime(2024, 1, 1, 0, 0, 0)
        item = {"created_at": naive}
        result = utils.get_sort_key(item, "created_at")
        assert result.tzinfo == timezone.utc
        assert result.replace(tzinfo=None) == naive

    def test_z_suffix_iso_string_is_parsed(self):
        """A 'Z'-terminated ISO string is handled correctly."""
        item = {"created_at": "2023-06-15T08:00:00Z"}
        expected = datetime(2023, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        assert utils.get_sort_key(item, "created_at") == expected

    def test_unparseable_string_returns_min_date(self):
        """A string that cannot be parsed returns datetime.min with UTC tz."""
        item = {"created_at": "not-a-date"}
        result = utils.get_sort_key(item, "created_at")
        assert result == datetime.min.replace(tzinfo=timezone.utc)

    def test_missing_key_returns_min_date(self):
        """A missing key returns datetime.min with UTC tz."""
        item = {"other_key": "value"}
        result = utils.get_sort_key(item, "created_at")
        assert result == datetime.min.replace(tzinfo=timezone.utc)

    def test_none_value_returns_min_date(self):
        """An explicit None value returns datetime.min with UTC tz."""
        item = {"created_at": None}
        result = utils.get_sort_key(item, "created_at")
        assert result == datetime.min.replace(tzinfo=timezone.utc)

# sanitize_username NFKC

class TestSanitizeUsername:
    def test_nfkc_ligature_is_normalised(self):
        """The NFKC ligature 'ﬁ' (U+FB01) is expanded to 'fi'."""
        # U+FB01 is the 'fi' ligature character
        dirty = "pro\ufb01le"  # "profile" with fi-ligature
        assert utils.sanitize_username(dirty) == "profile"

    def test_empty_string_stays_empty(self):
        assert utils.sanitize_username("") == ""

    def test_numeric_username_unchanged(self):
        assert utils.sanitize_username("12345") == "12345"

# extract_and_resolve_urls edge cases

class TestExtractAndResolveUrls:
    def test_empty_string_returns_empty_list(self):
        assert utils.extract_and_resolve_urls("") == []

    def test_none_returns_empty_list(self):
        assert utils.extract_and_resolve_urls(None) == []

    def test_plain_text_no_urls_returns_empty(self):
        assert utils.extract_and_resolve_urls("Just some plain text here.") == []

    def test_multiple_same_url_returns_each_occurrence(self):
        text = "see https://a.com and also https://a.com again"
        urls = utils.extract_and_resolve_urls(text)
        assert urls.count("https://a.com") == 2

# download_media security & offline tests

class TestDownloadMedia:
    def test_external_domain_blocked_for_twitter(self, tmp_path):
        """A non-CDN domain is blocked for Twitter when allow_external=False."""
        # twitter's safe CDNs are pbs.twimg.com and video.twimg.com
        external_url = "https://evil-server.example.com/photo.jpg"
        result = utils.download_media(
            base_dir=tmp_path,
            url=external_url,
            is_offline=False,
            platform="twitter",
            allow_external=False,
        )
        assert result is None

    def test_whitelisted_cdn_domain_is_not_blocked(self, tmp_path, mocker):
        """A CDN-whitelisted domain passes the security check (proceeds to download)."""
        whitelisted_url = "https://pbs.twimg.com/media/photo.jpg"

        # Mock the HTTP call so we don't actually download anything
        mock_resp = MagicMock()
        mock_resp.headers = {"content-type": "image/jpeg"}
        mock_resp.content = b"\xff\xd8\xff\xe0fake_jpeg_bytes"
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mocker.patch("socialosintagent.utils.httpx.Client", return_value=mock_client)

        # Ensure the media dir exists
        (tmp_path / "media").mkdir(exist_ok=True)

        result = utils.download_media(
            base_dir=tmp_path,
            url=whitelisted_url,
            is_offline=False,
            platform="twitter",
            allow_external=False,
        )
        # Should have attempted the download (result is a Path or None based on content-type)
        # The important thing is it was NOT blocked
        assert result is not None

    def test_offline_mode_skips_download_when_not_cached(self, tmp_path):
        """In offline mode, if the file isn't in the local media cache, None is returned."""
        result = utils.download_media(
            base_dir=tmp_path,
            url="https://pbs.twimg.com/media/missing.jpg",
            is_offline=True,
            platform="twitter",
        )
        assert result is None

    def test_allow_external_bypasses_cdn_check(self, tmp_path, mocker):
        """Setting allow_external=True skips the CDN whitelist entirely."""
        external_url = "https://my-personal-server.com/photo.jpg"

        mock_resp = MagicMock()
        mock_resp.headers = {"content-type": "image/png"}
        mock_resp.content = b"\x89PNGfake"
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mocker.patch("socialosintagent.utils.httpx.Client", return_value=mock_client)

        (tmp_path / "media").mkdir(exist_ok=True)

        result = utils.download_media(
            base_dir=tmp_path,
            url=external_url,
            is_offline=False,
            platform="twitter",
            allow_external=True,
        )
        # Not blocked — download was attempted
        assert result is not None