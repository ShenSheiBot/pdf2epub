"""
OCR-specific LLM client for handling Japanese text OCR with multiple model fallback.
"""

from typing import Union, List, Dict, Optional, Any
from loguru import logger
from tenacity import stop_after_attempt, wait_random_exponential
from .retry_utils import retry_with_logging
from .network_utils import (
    GeminiClient,
    AnthropicClient,
    is_transient_gemini_error,
    is_transient_anthropic_error
)
from google.genai.types import Part
import base64


class SafetyBlockError(Exception):
    """Raised when content is blocked for safety reasons."""
    def __init__(self, message: str, provider: str):
        self.provider = provider
        super().__init__(message)


class OCRClient:
    """
    OCR-specific LLM client that handles multiple providers transparently.
    Optimized for Japanese vertical text OCR with illustration detection.
    """
    
    # Standard OCR prompt for Japanese vertical text
    OCR_PROMPT = """You are processing a page from a Japanese light novel or manga that uses vertical text (縦書き).

Convert this page to markdown format following these guidelines:

1. **Text Direction**: This is vertical Japanese text that reads from right to left, top to bottom. Read and transcribe in the correct order.

2. **Text Formatting**:
   - Maintain paragraph breaks
   - Keep dialogue in quotation marks 「」
   - Preserve chapter titles or section headers as markdown headers (# or ##)
   - Include furigana (ruby text) in parentheses after the kanji

3. **Illustrations**: 
   - If you encounter an illustration, manga panel, or character art, simply write: [illustration]
   - Do not describe the illustration
   - Place [illustration] at the appropriate position in the text flow

4. **Output Format**:
   - Use clean markdown formatting
   - Each paragraph on its own line with blank line between paragraphs
   - Maintain original punctuation

Transcribe the entire page content accurately."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OCR client with configuration.
        
        Args:
            config: Configuration dict containing API keys and settings
        """
        self.config = config
        self._gemini_client = None
        self._anthropic_client = None
        # Track safety blocks per page to allow retrying on different pages
        self._safety_blocked_pages = {}  # {provider: set(page_nums)}
        
        # Initialize clients based on available API keys
        if config.get("google_api_key"):
            self._gemini_client = GeminiClient(
                config["google_api_key"],
                base_url=config.get("google_base_url")
            )
            
        if config.get("anthropic_api_key"):
            self._anthropic_client = AnthropicClient(
                api_key=config["anthropic_api_key"],
                base_url=config.get("anthropic_base_url")
            )
    
    def ocr_page(
        self,
        img_bytes: bytes,
        page_num: Optional[int] = None,
        model_configs: Optional[List[Dict]] = None,
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        OCR a page using configured models with automatic fallback.
        
        Args:
            img_bytes: Image bytes to OCR
            page_num: Page number for logging
            model_configs: List of model configurations to try in order
            custom_prompt: Optional custom prompt (defaults to standard OCR prompt)
            
        Returns:
            OCR text result
            
        Raises:
            Exception: If all models fail
        """
        # Use model configs from parameter or config file
        if model_configs is None:
            model_configs = self.config.get("ocr_models", [
                {"provider": "gemini", "model": "gemini-2.5-pro", "max_retries": 1},
                {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929", "max_retries": 2}
            ])
        
        prompt = custom_prompt if custom_prompt else self.OCR_PROMPT
        page_info = f"page {page_num}" if page_num else "page"
        operation_name = f"OCR {page_info}"
        
        last_error = None
        attempts_summary = []
        
        for model_config in model_configs:
            provider = model_config["provider"]
            model = model_config["model"]
            max_retries = model_config.get("max_retries", 1)
            
            # Skip if provider was blocked for this specific page
            if provider in self._safety_blocked_pages:
                if page_num and page_num in self._safety_blocked_pages[provider]:
                    logger.info(f"Skipping {provider} for {operation_name} (blocked on this page)")
                    attempts_summary.append(f"{provider}: skipped (safety on this page)")
                    continue
                # Log if provider has had safety issues on other pages but trying anyway
                elif self._safety_blocked_pages[provider]:
                    blocked_count = len(self._safety_blocked_pages[provider])
                    logger.debug(f"{provider} had safety blocks on {blocked_count} other page(s), trying anyway for {operation_name}")
            
            try:
                logger.info(f"Trying {provider} model {model} for {operation_name}")
                
                if provider == "gemini" and self._gemini_client:
                    response = self._ocr_with_gemini(
                        img_bytes=img_bytes,
                        prompt=prompt,
                        model=model,
                        max_retries=max_retries,
                        operation_name=operation_name
                    )
                    logger.success(f"Successfully OCR'd with {provider} for {operation_name}")
                    return response
                    
                elif provider == "anthropic" and self._anthropic_client:
                    response = self._ocr_with_anthropic(
                        img_bytes=img_bytes,
                        prompt=prompt,
                        model=model,
                        max_retries=max_retries,
                        operation_name=operation_name
                    )
                    logger.success(f"Successfully OCR'd with {provider} for {operation_name}")
                    return response
                    
                else:
                    logger.warning(f"Provider {provider} not available or not configured")
                    attempts_summary.append(f"{provider}: not configured")
                    continue
                    
            except SafetyBlockError as e:
                # Track which pages have safety blocks for this provider
                if provider not in self._safety_blocked_pages:
                    self._safety_blocked_pages[provider] = set()
                if page_num:
                    self._safety_blocked_pages[provider].add(page_num)
                    logger.warning(f"{provider} blocked for safety on page {page_num}: {e}")
                else:
                    logger.warning(f"{provider} blocked for safety for {operation_name}: {e}")
                attempts_summary.append(f"{provider}: safety blocked")
                last_error = e
                continue
                
            except Exception as e:
                logger.warning(f"{provider} failed for {operation_name}: {e}")
                attempts_summary.append(f"{provider}: failed")
                last_error = e
                continue
        
        # All models failed
        error_msg = f"All models failed for {operation_name}. Attempts: {', '.join(attempts_summary)}"
        logger.error(error_msg)
        if last_error:
            raise Exception(f"{error_msg}. Last error: {last_error}")
        else:
            raise Exception(error_msg)
    
    def _ocr_with_gemini(
        self,
        img_bytes: bytes,
        prompt: str,
        model: str,
        max_retries: int,
        operation_name: str
    ) -> str:
        """OCR with Gemini, handling retries internally."""
        
        # Prepare contents for Gemini API
        contents = [
            prompt,
            Part.from_bytes(
                mime_type="image/png",
                data=img_bytes
            ),
        ]
        
        # Configure generation with defaults
        temperature = 0.1  # Low temperature for accurate OCR
        generation_config = self._gemini_client.get_default_config(temperature)
        
        # Create retry decorator with specified attempts
        @retry_with_logging(
            operation_name=operation_name,
            retry_condition=self._is_retryable_gemini_error,
            wait_strategy=wait_random_exponential(multiplier=1, max=30),
            stop_strategy=stop_after_attempt(max_retries),
        )
        def ocr_with_retry():
            try:
                return self._gemini_client.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generation_config,
                    operation_name=operation_name
                )
            except Exception as e:
                # Check for safety block
                error_str = str(e).lower()
                if any(term in error_str for term in ['prohibited', 'safety', 'blocked']):
                    raise SafetyBlockError(str(e), "gemini")
                raise

        return ocr_with_retry()
    
    def _ocr_with_anthropic(
        self,
        img_bytes: bytes,
        prompt: str,
        model: str,
        max_retries: int,
        operation_name: str
    ) -> str:
        """OCR with Anthropic, handling retries internally."""
        
        # Convert image to base64 for Anthropic
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Create multi-part content for Anthropic
        multi_part_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_base64
                }
            }
        ]
        
        # Use defaults for temperature and max_tokens
        temperature = 0.1  # Low temperature for accurate OCR
        max_tokens = 8192  # Sufficient for most OCR tasks
        
        # Create retry decorator with specified attempts
        @retry_with_logging(
            operation_name=operation_name,
            retry_condition=self._is_retryable_anthropic_error,
            wait_strategy=wait_random_exponential(multiplier=1, max=30),
            stop_strategy=stop_after_attempt(max_retries),
        )
        def ocr_with_retry():
            try:
                return self._anthropic_client.generate_content(
                    prompt=multi_part_content,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    operation_name=operation_name
                )
            except Exception as e:
                # Check for safety block
                error_str = str(e).lower()
                if any(term in error_str for term in ['content_policy', 'unsafe', 'violation']):
                    raise SafetyBlockError(str(e), "anthropic")
                raise

        return ocr_with_retry()
    
    def _is_retryable_gemini_error(self, exception: Exception) -> bool:
        """Check if Gemini error should be retried (not safety blocks)."""
        if isinstance(exception, SafetyBlockError):
            return False
        return is_transient_gemini_error(exception)
    
    def _is_retryable_anthropic_error(self, exception: Exception) -> bool:
        """Check if Anthropic error should be retried (not safety blocks)."""
        if isinstance(exception, SafetyBlockError):
            return False
        return is_transient_anthropic_error(exception)
    
    def get_safety_stats(self) -> Dict[str, int]:
        """Get statistics about safety blocks per provider."""
        stats = {}
        for provider, blocked_pages in self._safety_blocked_pages.items():
            stats[provider] = len(blocked_pages)
        return stats
    
    def clear_safety_blocks(self, provider: Optional[str] = None):
        """Clear safety block tracking for a provider or all providers.
        
        Args:
            provider: Specific provider to clear, or None to clear all
        """
        if provider:
            if provider in self._safety_blocked_pages:
                self._safety_blocked_pages[provider].clear()
                logger.info(f"Cleared safety blocks for {provider}")
        else:
            self._safety_blocked_pages.clear()
            logger.info("Cleared all safety blocks")
    
