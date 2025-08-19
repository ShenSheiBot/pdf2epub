"""
Refactored network utilities using tenacity for cleaner retry logic.
"""

import base64
import httpx
from loguru import logger
from typing import Optional, Dict, Any, Union, List
from typing import Any as Any  # Explicit for backward compatibility
from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting
)
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception


# Define transient errors that should trigger retries
def is_transient_gemini_error(exception: Exception) -> bool:
    """Check if a Gemini API error is transient and should be retried."""
    # Don't retry content safety blocks - these should fail fast
    error_str = str(exception).lower()
    if any(term in error_str for term in ['prohibited', 'safety', 'blocked', 'harmful']):
        return False
    
    # Retry network and rate limit errors
    if isinstance(exception, (httpx.TimeoutException, httpx.ConnectError, ConnectionError)):
        return True
    
    # Check for specific error codes
    transient_keywords = [
        'rate_limit', '429', 'quota',
        'timeout', 'unavailable', '503',
        'internal', '500', '502', '504',
        'resource_exhausted', 'overloaded'
    ]
    
    return any(keyword in error_str for keyword in transient_keywords)


def is_transient_anthropic_error(exception: Exception) -> bool:
    """Check if an Anthropic API error is transient and should be retried."""
    error_str = str(exception).lower()
    
    # Don't retry content blocks
    if any(term in error_str for term in ['content_policy', 'unsafe', 'violation']):
        return False
    
    # Retry rate limits and server errors
    if '429' in error_str or 'rate' in error_str:
        return True
    
    if any(code in error_str for code in ['500', '502', '503', '504']):
        return True
    
    if isinstance(exception, (TimeoutError, ConnectionError, httpx.TimeoutException)):
        return True
    
    return False


class GeminiClient:
    """Wrapper for Gemini API with smart retry logic."""
    
    def __init__(self, api_key: str):
        """Initialize Gemini client."""
        from google import genai
        self.client = genai.Client(api_key=api_key)
    
    @retry(
        retry=retry_if_exception(is_transient_gemini_error),
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def generate_content(
        self,
        model: str,
        contents: Any,
        config: Optional[GenerateContentConfig] = None,
        operation_name: str = "Gemini API call"
    ) -> Any:
        """Generate content with automatic retry for transient errors."""
        logger.info(f"Calling Gemini API for {operation_name}")
        
        if config is None:
            config = self.get_default_config()
        
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
        
        if not response or not response.text:
            raise ValueError(f"Empty response from Gemini for {operation_name}")
        
        return response
    
    @retry(
        retry=retry_if_exception(is_transient_gemini_error),
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def generate_content_stream(
        self,
        model: str,
        contents: Any,
        config: Optional[GenerateContentConfig] = None,
        operation_name: str = "Gemini API call"
    ) -> str:
        """Generate content with streaming and automatic retry."""
        logger.info(f"Streaming from Gemini API for {operation_name}")
        
        if config is None:
            config = self.get_default_config()
        
        # Stream and aggregate response
        stream_response = self.client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config
        )
        
        aggregated_text = ""
        chunk_count = 0
        
        for chunk in stream_response:
            chunk_count += 1
            if hasattr(chunk, 'text') and chunk.text:
                aggregated_text += chunk.text
                
                # Log progress periodically
                if len(aggregated_text) % 2000 < 10:
                    estimated_tokens = len(aggregated_text) // 4
                    logger.debug(f"Streaming {operation_name}: ~{estimated_tokens} tokens")
            
            # Check for early termination
            if hasattr(chunk, 'candidates') and chunk.candidates:
                for candidate in chunk.candidates:
                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                        reason = str(candidate.finish_reason)
                        if any(term in reason for term in ['PROHIBITED', 'SAFETY', 'BLOCKED']):
                            raise ValueError(f"Content blocked: {reason}")
        
        if not aggregated_text:
            raise ValueError(f"Empty stream response for {operation_name}")
        
        logger.info(f"Streamed {len(aggregated_text)} chars ({chunk_count} chunks) for {operation_name}")
        return aggregated_text
    
    @staticmethod
    def get_default_config(temperature: float = 0.1) -> GenerateContentConfig:
        """Get default generation config."""
        return GenerateContentConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=20,
            candidate_count=1,
            max_output_tokens=65536,
            stop_sequences=None,
            safety_settings=[
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
            ],
        )


class AnthropicClient:
    """Wrapper for Anthropic API with smart retry logic."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """Initialize Anthropic client."""
        import anthropic
        if base_url:
            self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
    
    @retry(
        retry=retry_if_exception(is_transient_anthropic_error),
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def generate_content(
        self,
        prompt: Union[str, List[Dict]],
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8192,
        temperature: float = 0.1,
        operation_name: str = "Anthropic API call"
    ) -> str:
        """Generate content with automatic retry for transient errors."""
        logger.info(f"Calling Anthropic API for {operation_name}")
        
        # Process content for images
        content = self._process_content(prompt)
        
        # Create message with streaming
        stream = self.client.messages.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        # Aggregate streamed response
        response_text = ""
        chunk_count = 0
        for event in stream:
            if event.type == "content_block_delta":
                if hasattr(event.delta, 'text'):
                    response_text += event.delta.text
                    chunk_count += 1
                    
                    # Log progress periodically (similar to Gemini)
                    if len(response_text) % 2000 < 10:
                        estimated_tokens = len(response_text) // 4
                        logger.debug(f"Streaming {operation_name}: ~{estimated_tokens} tokens")
        
        if not response_text:
            raise ValueError(f"Empty response from Anthropic for {operation_name}")
        
        logger.info(f"Streamed {len(response_text)} chars ({chunk_count} chunks) from Anthropic for {operation_name}")
        return response_text
    
    def _process_content(self, prompt: Union[str, List[Dict]]) -> Union[str, List[Dict]]:
        """Process content to handle images properly."""
        if isinstance(prompt, str):
            return prompt
        
        if isinstance(prompt, list):
            processed = []
            for part in prompt:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        processed.append(part)
                    elif part.get("type") == "image":
                        # Check if already in Anthropic format
                        if "source" in part and isinstance(part["source"], dict):
                            # Already formatted correctly, just pass through
                            processed.append(part)
                        else:
                            # Convert image bytes to base64
                            image_data = part.get("data")
                            mime_type = part.get("mime_type", "image/png")
                            
                            if isinstance(image_data, bytes):
                                base64_data = base64.b64encode(image_data).decode('utf-8')
                            else:
                                base64_data = image_data
                            
                            processed.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": base64_data
                                }
                            })
                    else:
                        processed.append(part)
                else:
                    processed.append(part)
            return processed
        
        return prompt


def generate_with_fallback(
    prompt: Union[str, List[Dict]],
    gemini_client: Optional[GeminiClient] = None,
    anthropic_client: Optional[AnthropicClient] = None,
    gemini_model: str = "gemini-2.5-pro",
    anthropic_model: str = "claude-sonnet-4-20250514",
    operation_name: str = "API call",
    prefer_anthropic: bool = False
) -> str:
    """
    Generate content with fallback between Gemini and Anthropic.
    
    Args:
        prompt: The prompt (string or list of content parts)
        gemini_client: Optional Gemini client
        anthropic_client: Optional Anthropic client
        gemini_model: Gemini model to use
        anthropic_model: Anthropic model to use
        operation_name: Name for logging
        prefer_anthropic: If True, try Anthropic first
        
    Returns:
        Generated text
    """
    if prefer_anthropic and anthropic_client:
        try:
            return anthropic_client.generate_content(
                prompt=prompt,
                model=anthropic_model,
                operation_name=f"{operation_name} (Anthropic)"
            )
        except Exception as e:
            if not is_transient_anthropic_error(e):
                logger.warning(f"Anthropic failed with non-transient error: {e}")
                if gemini_client:
                    logger.info("Falling back to Gemini")
                else:
                    raise
    
    if gemini_client:
        try:
            # Convert prompt format for Gemini if needed
            if isinstance(prompt, list):
                # Gemini expects different format
                contents = []
                for part in prompt:
                    if isinstance(part, dict) and part.get("type") == "text":
                        contents.append({"text": part["text"]})
                    else:
                        contents.append(part)
            else:
                contents = prompt
            
            response = gemini_client.generate_content_stream(
                model=gemini_model,
                contents=contents,
                operation_name=f"{operation_name} (Gemini)"
            )
            return response
        except Exception as e:
            if not is_transient_gemini_error(e) and anthropic_client and not prefer_anthropic:
                logger.warning(f"Gemini failed with non-transient error: {e}")
                logger.info("Falling back to Anthropic")
                return anthropic_client.generate_content(
                    prompt=prompt,
                    model=anthropic_model,
                    operation_name=f"{operation_name} (Anthropic)"
                )
            raise
    
    if anthropic_client and not prefer_anthropic:
        return anthropic_client.generate_content(
            prompt=prompt,
            model=anthropic_model,
            operation_name=f"{operation_name} (Anthropic)"
        )
    
    raise ValueError("No API clients available")