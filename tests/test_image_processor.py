"""
Tests for socialosintagent/image_processor.py

Covers:
ImageProcessor.preprocess_image()
  - Returns None for non-existent file
  - Returns None for unsupported extension
  - Returns None when PIL cannot identify the file (UnidentifiedImageError)
  - Returns a path for a valid RGB JPEG
  - Converts RGBA image to RGB (white background compositing)
  - Resizes an oversized image to max_dimension
  - Extracts first frame of an animated GIF

ImageProcessor.process_single_image()
  - Returns PREPROCESSING_FAILED when file does not exist
  - Returns UNSUPPORTED_FORMAT for a .txt file
  - Returns PREPROCESSING_FAILED when preprocess_image returns None
  - Returns SUCCESS with local_path when no analyze_func provided
  - Returns SUCCESS with analysis when analyze_func returns a result
  - Returns ANALYSIS_FAILED when analyze_func returns None
  - Returns ANALYSIS_FAILED when analyze_func raises a generic exception
  - Returns RATE_LIMITED and propagates when analyze_func raises RateLimitExceededError
  - Cleans up the .processed.jpg temp file after successful analysis
  - Cleans up the .processed.jpg temp file after failed analysis
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from socialosintagent.exceptions import RateLimitExceededError
from socialosintagent.image_processor import ImageProcessor, ProcessingStatus


# ── Helpers ─────────────────────────────────────────────────────────────────

def _write_jpeg(path: Path, size=(100, 100), mode="RGB") -> Path:
    """Write a minimal valid JPEG to *path* and return it."""
    img = Image.new(mode, size, color=(128, 64, 32))
    img.save(path, "JPEG")
    return path


def _write_png_rgba(path: Path, size=(80, 80)) -> Path:
    img = Image.new("RGBA", size, color=(200, 100, 50, 128))
    img.save(path, "PNG")
    return path


def _write_gif(path: Path, size=(60, 60)) -> Path:
    """Write a two-frame animated GIF."""
    frames = [Image.new("P", size, color=i) for i in (0, 128)]
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0, format="GIF")
    return path


# ── preprocess_image ─────────────────────────────────────────────────────────

class TestPreprocessImage:
    def test_returns_none_for_nonexistent_file(self, tmp_path):
        proc = ImageProcessor()
        result = proc.preprocess_image(tmp_path / "ghost.jpg")
        assert result is None

    def test_returns_none_for_unsupported_extension(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        proc = ImageProcessor()
        result = proc.preprocess_image(f)
        assert result is None

    def test_returns_none_for_unidentifiable_image(self, tmp_path):
        f = tmp_path / "corrupt.jpg"
        f.write_bytes(b"this is not an image")
        proc = ImageProcessor()
        result = proc.preprocess_image(f)
        assert result is None

    def test_valid_rgb_jpeg_returns_output_path(self, tmp_path):
        src = _write_jpeg(tmp_path / "photo.jpg")
        proc = ImageProcessor()
        out = proc.preprocess_image(src)
        assert out is not None
        assert out.exists()
        assert out.suffix == ".jpg"

    def test_rgba_png_converted_to_rgb(self, tmp_path):
        src = _write_png_rgba(tmp_path / "transparent.png")
        proc = ImageProcessor()
        out = proc.preprocess_image(src)
        assert out is not None
        with Image.open(out) as img:
            assert img.mode == "RGB"

    def test_oversized_image_is_resized(self, tmp_path):
        # Create an image larger than the 1536 default max_dimension
        src = tmp_path / "big.jpg"
        Image.new("RGB", (2000, 1500), color=(0, 0, 0)).save(src, "JPEG")
        proc = ImageProcessor(max_dimension=200)
        out = proc.preprocess_image(src)
        assert out is not None
        with Image.open(out) as img:
            assert max(img.size) <= 200

    def test_small_image_is_not_resized(self, tmp_path):
        src = _write_jpeg(tmp_path / "small.jpg", size=(50, 50))
        proc = ImageProcessor(max_dimension=200)
        out = proc.preprocess_image(src)
        assert out is not None
        with Image.open(out) as img:
            assert img.size == (50, 50)

    def test_animated_gif_first_frame_extracted(self, tmp_path):
        src = _write_gif(tmp_path / "anim.gif")
        proc = ImageProcessor()
        out = proc.preprocess_image(src)
        assert out is not None
        assert out.exists()

    def test_custom_output_path_respected(self, tmp_path):
        src = _write_jpeg(tmp_path / "in.jpg")
        custom_out = tmp_path / "custom_output.jpg"
        proc = ImageProcessor()
        out = proc.preprocess_image(src, output_path=custom_out)
        assert out == custom_out
        assert custom_out.exists()


# ── process_single_image ─────────────────────────────────────────────────────

class TestProcessSingleImage:
    def test_preprocessing_failed_for_nonexistent_file(self, tmp_path):
        proc = ImageProcessor()
        result = proc.process_single_image(tmp_path / "ghost.jpg")
        assert result.status == ProcessingStatus.PREPROCESSING_FAILED
        assert result.analysis is None

    def test_unsupported_format_for_wrong_extension(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("hello")
        proc = ImageProcessor()
        result = proc.process_single_image(f)
        assert result.status == ProcessingStatus.UNSUPPORTED_FORMAT

    def test_preprocessing_failed_when_preprocess_returns_none(self, tmp_path):
        src = _write_jpeg(tmp_path / "img.jpg")
        proc = ImageProcessor()
        with patch.object(proc, "preprocess_image", return_value=None):
            result = proc.process_single_image(src)
        assert result.status == ProcessingStatus.PREPROCESSING_FAILED

    def test_success_with_local_path_when_no_analyze_func(self, tmp_path):
        src = _write_jpeg(tmp_path / "img.jpg")
        proc = ImageProcessor()
        result = proc.process_single_image(src)
        assert result.status == ProcessingStatus.SUCCESS
        assert result.analysis is None
        assert result.local_path is not None

    def test_success_with_analysis_when_analyze_func_returns_string(self, tmp_path):
        src = _write_jpeg(tmp_path / "img.jpg")
        proc = ImageProcessor()
        analyze_func = MagicMock(return_value="An outdoor park scene.")
        result = proc.process_single_image(src, analyze_func=analyze_func, source_url="http://example.com/img.jpg", context="twitter user alice")
        assert result.status == ProcessingStatus.SUCCESS
        assert result.analysis == "An outdoor park scene."
        analyze_func.assert_called_once()

    def test_analysis_failed_when_analyze_func_returns_none(self, tmp_path):
        src = _write_jpeg(tmp_path / "img.jpg")
        proc = ImageProcessor()
        result = proc.process_single_image(src, analyze_func=lambda *a, **kw: None)
        assert result.status == ProcessingStatus.ANALYSIS_FAILED
        assert "no result" in result.error_message.lower()

    def test_analysis_failed_when_analyze_func_raises_generic_exception(self, tmp_path):
        src = _write_jpeg(tmp_path / "img.jpg")
        proc = ImageProcessor()
        def bad_analyze(*a, **kw):
            raise ValueError("model unavailable")
        result = proc.process_single_image(src, analyze_func=bad_analyze)
        assert result.status == ProcessingStatus.ANALYSIS_FAILED
        assert "model unavailable" in result.error_message

    def test_rate_limited_status_when_analyze_func_raises_rate_limit(self, tmp_path):
        src = _write_jpeg(tmp_path / "img.jpg")
        proc = ImageProcessor()
        def rate_limited_analyze(*a, **kw):
            raise RateLimitExceededError("Vision API rate limit")
        result = proc.process_single_image(src, analyze_func=rate_limited_analyze)
        assert result.status == ProcessingStatus.RATE_LIMITED

    def test_temp_processed_file_cleaned_up_after_success(self, tmp_path):
        src = _write_jpeg(tmp_path / "img.jpg")
        proc = ImageProcessor()
        analyze_func = MagicMock(return_value="scene description")
        proc.process_single_image(src, analyze_func=analyze_func)
        # The .processed.jpg should be deleted after analysis
        processed = src.with_suffix(".processed.jpg")
        assert not processed.exists()

    def test_temp_processed_file_cleaned_up_after_failure(self, tmp_path):
        src = _write_jpeg(tmp_path / "img.jpg")
        proc = ImageProcessor()
        proc.process_single_image(src, analyze_func=lambda *a, **kw: None)
        processed = src.with_suffix(".processed.jpg")
        assert not processed.exists()

    def test_source_url_passed_through_to_result(self, tmp_path):
        src = _write_jpeg(tmp_path / "img.jpg")
        proc = ImageProcessor()
        result = proc.process_single_image(src, source_url="https://cdn.example.com/photo.jpg")
        assert result.url == "https://cdn.example.com/photo.jpg"

    def test_fallback_url_is_file_path_string_when_no_source_url(self, tmp_path):
        src = _write_jpeg(tmp_path / "img.jpg")
        proc = ImageProcessor()
        result = proc.process_single_image(src)
        assert result.url == str(src)
