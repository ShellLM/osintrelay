"""
Improved image processing module with graceful error handling.

This module handles image preprocessing, downloading, and analysis with
resilience to failures - individual image failures won't crash the entire pipeline.
"""

import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from PIL import Image, UnidentifiedImageError

from .exceptions import RateLimitExceededError
from .utils import SUPPORTED_IMAGE_EXTENSIONS

logger = logging.getLogger("SocialOSINTAgent.image_processor")


class ProcessingStatus(Enum):
    """Status of image processing operations."""
    SUCCESS = "success"
    DOWNLOAD_FAILED = "download_failed"
    PREPROCESSING_FAILED = "preprocessing_failed"
    ANALYSIS_FAILED = "analysis_failed"
    UNSUPPORTED_FORMAT = "unsupported_format"
    RATE_LIMITED = "rate_limited"
    SKIPPED = "skipped"


@dataclass
class ImageProcessingResult:
    """Result of processing a single image."""
    url: str
    status: ProcessingStatus
    local_path: Optional[Path] = None
    analysis: Optional[str] = None
    error_message: Optional[str] = None


class ImageProcessor:
    """
    Handles all image processing operations with graceful error handling.
    
    Features:
    - Resilient downloading with retries
    - Safe image preprocessing (format conversion, resizing)
    - Batch processing with failure isolation
    - Detailed error reporting
    """

    def __init__(
        self,
        max_dimension: int = 1536,
        jpeg_quality: int = 85,
        request_timeout: float = 20.0,
    ):
        """
        Initialize the image processor.
        
        Args:
            max_dimension: Maximum width/height for resized images
            jpeg_quality: JPEG compression quality (1-100)
            request_timeout: HTTP request timeout in seconds
        """
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality
        self.request_timeout = request_timeout
        self.logger = logger

    def preprocess_image(
        self, file_path: Path, output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Safely preprocess an image file.
        
        Operations:
        - Convert to RGB (if needed)
        - Extract first frame (for animated images)
        - Resize to max dimension
        - Convert to JPEG
        
        Args:
            file_path: Path to the input image
            output_path: Optional output path (defaults to .processed.jpg)
            
        Returns:
            Path to processed image, or None on failure
        """
        if not file_path.exists():
            self.logger.warning(f"Image file does not exist: {file_path}")
            return None

        if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            self.logger.warning(f"Unsupported image format: {file_path.suffix}")
            return None

        # Default output path
        if output_path is None:
            output_path = file_path.with_suffix(".processed.jpg")

        try:
            with Image.open(file_path) as img:
                # Handle animated images (extract first frame)
                img_to_process = img
                if getattr(img, "is_animated", False):
                    self.logger.debug(f"Extracting first frame from animated image: {file_path}")
                    img.seek(0)
                    img_to_process = img.copy()

                # Convert to RGB if needed
                if img_to_process.mode in ('RGBA', 'LA') or (img_to_process.mode == 'P' and 'transparency' in img_to_process.info):
                    # Create a white background
                    background = Image.new('RGB', img_to_process.size, (255, 255, 255))
                    # Paste the image on top using alpha channel as mask
                    if img_to_process.mode == 'P':
                        img_to_process = img_to_process.convert('RGBA')
                    background.paste(img_to_process, mask=img_to_process.split()[3]) # 3 is alpha
                    img_to_process = background
                elif img_to_process.mode != "RGB":
                    img_to_process = img_to_process.convert("RGB")

                # Resize if too large
                original_size = img_to_process.size
                if max(img_to_process.size) > self.max_dimension:
                    img_to_process.thumbnail(
                        (self.max_dimension, self.max_dimension), 
                        Image.Resampling.LANCZOS
                    )
                    self.logger.debug(
                        f"Resized image from {original_size} to {img_to_process.size}"
                    )

                # Save as JPEG
                img_to_process.save(
                    output_path, "JPEG", quality=self.jpeg_quality, optimize=True
                )
                self.logger.debug(f"Preprocessed image saved to: {output_path}")
                return output_path

        except UnidentifiedImageError:
            self.logger.error(f"Cannot identify image file: {file_path}")
            return None
        except OSError as e:
            self.logger.error(f"OS error processing image {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error preprocessing image {file_path}: {e}",
                exc_info=True,
            )
            return None

    def encode_image_to_base64(self, file_path: Path) -> Optional[str]:
        """
        Encode an image file to base64.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Base64-encoded string, or None on failure
        """
        try:
            if not file_path.exists():
                self.logger.warning(f"File does not exist: {file_path}")
                return None

            with open(file_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        except Exception as e:
            self.logger.error(f"Error encoding image {file_path} to base64: {e}")
            return None

    def process_single_image(
        self,
        file_path: Path,
        analyze_func: Optional[callable] = None,
        source_url: Optional[str] = None,
        context: Optional[str] = None,
    ) -> ImageProcessingResult:
        """
        Process a single image through the full pipeline.
        
        Args:
            file_path: Path to the image file
            analyze_func: Optional function to analyze the preprocessed image
            source_url: Original URL of the image
            context: Context for analysis
            
        Returns:
            ImageProcessingResult with status and data
        """
        url = source_url or str(file_path)
        
        # Validate file exists
        if not file_path.exists():
            return ImageProcessingResult(
                url=url,
                status=ProcessingStatus.PREPROCESSING_FAILED,
                error_message="File does not exist",
            )

        # Validate format
        if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            return ImageProcessingResult(
                url=url,
                status=ProcessingStatus.UNSUPPORTED_FORMAT,
                error_message=f"Unsupported format: {file_path.suffix}",
            )

        # Preprocess
        processed_path = self.preprocess_image(file_path)
        if not processed_path:
            return ImageProcessingResult(
                url=url,
                status=ProcessingStatus.PREPROCESSING_FAILED,
                error_message="Image preprocessing failed",
            )

        # If no analysis function provided, just return preprocessed result
        if not analyze_func:
            return ImageProcessingResult(
                url=url,
                status=ProcessingStatus.SUCCESS,
                local_path=processed_path,
            )

        # Perform analysis
        try:
            analysis = analyze_func(processed_path, source_url=url, context=context)
            
            # Clean up temporary processed file
            if processed_path != file_path and processed_path.exists():
                processed_path.unlink()

            if analysis:
                return ImageProcessingResult(
                    url=url,
                    status=ProcessingStatus.SUCCESS,
                    local_path=file_path,
                    analysis=analysis,
                )
            else:
                return ImageProcessingResult(
                    url=url,
                    status=ProcessingStatus.ANALYSIS_FAILED,
                    local_path=file_path,
                    error_message="Analysis returned no result",
                )

        except RateLimitExceededError:
            # Clean up and propagate rate limit errors
            if processed_path != file_path and processed_path.exists():
                processed_path.unlink()
            return ImageProcessingResult(
                url=url,
                status=ProcessingStatus.RATE_LIMITED,
                local_path=file_path,
                error_message="Rate limit exceeded during analysis",
            )

        except Exception as e:
            # Clean up
            if processed_path != file_path and processed_path.exists():
                processed_path.unlink()
            
            self.logger.error(
                f"Error analyzing image {file_path}: {e}", 
                exc_info=True
            )
            return ImageProcessingResult(
                url=url,
                status=ProcessingStatus.ANALYSIS_FAILED,
                local_path=file_path,
                error_message=str(e),
            )