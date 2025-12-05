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
    SafetySetting,
    ThinkingConfig
)
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception
from tenacity.stop import stop_base
from tenacity.wait import wait_base
import tiktoken


class stop_after_self_retries(stop_base):
    """Stop after `self.num_retries` attempts (defaults to 3 if missing)."""
    def __init__(self, attr: str = "num_retries", default: int = 3):
        self.attr = attr
        self.default = default

    def __call__(self, retry_state):
        # For bound instance methods, args[0] is `self`
        obj = retry_state.args[0] if retry_state.args else None
        num = getattr(obj, self.attr, self.default)
        return retry_state.attempt_number >= int(num)


class wait_exponential_with_self_max(wait_base):
    """Use Tenacity's wait_random_exponential with self.max_backoff_seconds."""
    def __init__(self, multiplier: float = 1, attr: str = "max_backoff_seconds", default: int = 30):
        self.multiplier = multiplier
        self.attr = attr
        self.default = default

    def __call__(self, retry_state):
        # For bound instance methods, args[0] is `self`
        obj = retry_state.args[0] if retry_state.args else None
        max_seconds = getattr(obj, self.attr, self.default)
        
        # Create and use Tenacity's wait_random_exponential
        wait_strategy = wait_random_exponential(multiplier=self.multiplier, max=max_seconds)
        return wait_strategy(retry_state)


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
        'resource_exhausted', 'overloaded',
        'disconnected', 'connection'
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


def is_transient_openai_error(exception: Exception) -> bool:
    """Check if an OpenAI API error is transient and should be retried."""
    error_str = str(exception).lower()
    
    # Don't retry content policy violations
    if any(term in error_str for term in ['content_policy', 'violation', 'refused']):
        return False
    
    # Retry rate limits and server errors
    if '429' in error_str or 'rate' in error_str:
        return True
    
    if any(code in error_str for code in ['500', '502', '503', '504']):
        return True
    
    if isinstance(exception, (TimeoutError, ConnectionError, httpx.TimeoutException)):
        return True
    
    # OpenAI specific errors
    if 'timeout' in error_str or 'connection' in error_str:
        return True
        
    return False


class GeminiClient:
    """Wrapper for Gemini API with smart retry logic."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, num_retries: int = 3, max_backoff_seconds: int = 30):
        """Initialize Gemini client.

        Args:
            api_key: Gemini API key
            base_url: Custom API endpoint (e.g., 'google.shenshei.fans')
            num_retries: Number of retries for transient errors
            max_backoff_seconds: Maximum backoff time between retries
        """
        from google import genai
        if base_url:
            # Use custom endpoint
            self.client = genai.Client(
                api_key=api_key,
                http_options={'base_url': base_url}
            )
        else:
            self.client = genai.Client(api_key=api_key)
        self.num_retries = num_retries
        self.max_backoff_seconds = max_backoff_seconds
        # Initialize tokenizer for accurate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    @retry(
        retry=retry_if_exception(is_transient_gemini_error),
        wait=wait_exponential_with_self_max(multiplier=1),
        stop=stop_after_self_retries(),
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
        wait=wait_exponential_with_self_max(multiplier=1),
        stop=stop_after_self_retries(),
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
        last_log_length = 0

        for chunk in stream_response:
            chunk_count += 1

            # Handle Gemini 3 response format with potential thought parts
            if hasattr(chunk, 'candidates') and chunk.candidates:
                for candidate in chunk.candidates:
                    # Check for early termination
                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                        reason = str(candidate.finish_reason)
                        if any(term in reason for term in ['PROHIBITED', 'SAFETY', 'BLOCKED']):
                            raise ValueError(f"Content blocked: {reason}")

                    # Extract text from content parts, filtering out thoughts
                    if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            # Skip thought parts (Gemini 3 thinking mode)
                            if hasattr(part, 'thought') and part.thought:
                                continue
                            if hasattr(part, 'text') and part.text:
                                aggregated_text += part.text
            elif hasattr(chunk, 'text') and chunk.text:
                # Fallback for simpler response format
                aggregated_text += chunk.text

            # Log progress periodically - every 500 tokens
            current_tokens = len(self.tokenizer.encode(aggregated_text))
            if current_tokens - last_log_length >= 500:
                logger.debug(f"Streaming {operation_name}: {current_tokens} tokens")
                last_log_length = current_tokens
        
        if not aggregated_text:
            logger.error(f"Empty stream: {chunk_count} chunks received for {operation_name}")
            if chunk_count > 0:
                logger.error(f"Last chunk had candidates: {hasattr(chunk, 'candidates')}")
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        logger.error(f"Candidate finish_reason: {getattr(candidate, 'finish_reason', 'N/A')}")
            raise ValueError(f"Empty stream response for {operation_name}")
        
        # Get final token count
        final_tokens = len(self.tokenizer.encode(aggregated_text))
        logger.info(f"Streamed {final_tokens} tokens ({chunk_count} chunks) for {operation_name}")
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
            # Configure thinking for Gemini 3 models (ignored by older models)
            # Note: Gemini 3 requires thinking mode, cannot disable it
            thinking_config=ThinkingConfig(
                thinking_budget=1024,  # Minimal thinking budget
                include_thoughts=False  # Don't include thought summaries in output
            ),
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
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, num_retries: int = 3, max_backoff_seconds: int = 30):
        """Initialize Anthropic client."""
        import anthropic
        if base_url:
            self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
        self.num_retries = num_retries
        self.max_backoff_seconds = max_backoff_seconds
        # Initialize tokenizer for accurate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    @retry(
        retry=retry_if_exception(is_transient_anthropic_error),
        wait=wait_exponential_with_self_max(multiplier=1),
        stop=stop_after_self_retries(),
        reraise=True
    )
    def generate_content(
        self,
        prompt: Union[str, List[Dict]],
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 8192,
        temperature: float = 0.1,
        operation_name: str = "Anthropic API call"
    ) -> str:
        """Generate content with automatic retry for transient errors.

        Args:
            prompt: Can be:
                - str: Simple text prompt
                - List[Dict]: Either content blocks or messages with roles
                  - Content blocks: [{"type": "text", "text": "..."}]
                  - Messages: [{"role": "user"|"assistant", "content": "..."}]
        """
        logger.info(f"Calling Anthropic API for {operation_name}")

        # Check if prompt is a list of messages with roles
        messages = None
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict) and "role" in prompt[0]:
            # It's a conversation history with roles
            messages = prompt
        else:
            # Process content for images and create single user message
            content = self._process_content(prompt)
            messages = [{"role": "user", "content": content}]

        # Create message with streaming
        stream = self.client.messages.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        # Aggregate streamed response
        response_text = ""
        chunk_count = 0
        last_log_length = 0
        for event in stream:
            if event.type == "content_block_delta":
                if hasattr(event.delta, 'text'):
                    response_text += event.delta.text
                    chunk_count += 1
                    
                    # Log progress periodically - every 500 tokens
                    current_tokens = len(self.tokenizer.encode(response_text))
                    if current_tokens - last_log_length >= 500:
                        logger.debug(f"Streaming {operation_name}: {current_tokens} tokens")
                        last_log_length = current_tokens
        
        if not response_text:
            raise ValueError(f"Empty response from Anthropic for {operation_name}")
        
        # Get final token count
        final_tokens = len(self.tokenizer.encode(response_text))
        logger.info(f"Streamed {final_tokens} tokens ({chunk_count} chunks) from Anthropic for {operation_name}")
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


class OpenAIClient:
    """Wrapper for OpenAI API with smart retry logic."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None, num_retries: int = 3, max_backoff_seconds: int = 30):
        """Initialize OpenAI client."""
        from openai import OpenAI
        
        # Set up client with optional base URL
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        
        # Store default model
        self.default_model = model or "gpt-4o"
        self.num_retries = num_retries
        self.max_backoff_seconds = max_backoff_seconds
        
        # Initialize tokenizer for accurate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    @retry(
        retry=retry_if_exception(is_transient_openai_error),
        wait=wait_exponential_with_self_max(multiplier=1),
        stop=stop_after_self_retries(),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"OpenAI API retry {retry_state.attempt_number} for "
            f"{retry_state.args[0] if retry_state.args else 'unknown operation'}: "
            f"{retry_state.outcome.exception()}"
        )
    )
    def generate_content(
        self,
        prompt: Union[str, List[Dict]],
        model: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.1,
        operation_name: str = "OpenAI API call"
    ) -> str:
        """Generate content with automatic retry for transient errors."""
        logger.info(f"Calling OpenAI API for {operation_name}")
        
        # Use provided model or default
        model_to_use = model or self.default_model
        
        # Process content for messages format
        messages = self._format_messages(prompt)
        
        # Create chat completion with streaming
        try:
            stream = self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
        except Exception as e:
            logger.error(f"Failed to create OpenAI stream for {operation_name}: {e}")
            raise
        
        # Aggregate streamed response
        response_text = ""
        chunk_count = 0
        last_log_tokens = 0
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
                chunk_count += 1
                
                # Log progress periodically - every 500 tokens
                current_tokens = len(self.tokenizer.encode(response_text))
                if current_tokens - last_log_tokens >= 500:
                    logger.debug(f"Streaming {operation_name}: {current_tokens} tokens")
                    last_log_tokens = current_tokens
        
        if not response_text:
            raise ValueError(f"Empty response from OpenAI for {operation_name}")
        
        # Get final token count
        final_tokens = len(self.tokenizer.encode(response_text))
        logger.info(f"Streamed {final_tokens} tokens ({chunk_count} chunks) from OpenAI for {operation_name}")
        return response_text
    
    def _format_messages(self, prompt: Union[str, List[Dict]]) -> List[Dict]:
        """Format prompt into OpenAI messages format."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        
        if isinstance(prompt, list):
            # Handle multi-part content
            content_parts = []
            for part in prompt:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        content_parts.append({
                            "type": "text",
                            "text": part["text"]
                        })
                    elif part.get("type") == "image":
                        # Handle image data
                        if "source" in part and isinstance(part["source"], dict):
                            # Already in structured format
                            base64_data = part["source"].get("data")
                            mime_type = part["source"].get("media_type", "image/png")
                        else:
                            # Simple format
                            image_data = part.get("data")
                            mime_type = part.get("mime_type", "image/png")
                            
                            if isinstance(image_data, bytes):
                                base64_data = base64.b64encode(image_data).decode('utf-8')
                            else:
                                base64_data = image_data
                        
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_data}"
                            }
                        })
                    else:
                        # Pass through other types
                        content_parts.append(part)
                else:
                    # String content
                    content_parts.append({
                        "type": "text",
                        "text": str(part)
                    })
            
            return [{"role": "user", "content": content_parts}]
        
        # Default case
        return [{"role": "user", "content": str(prompt)}]


def generate_with_fallback(
    prompt: Union[str, List[Dict]],
    gemini_client: Optional[GeminiClient] = None,
    anthropic_client: Optional[AnthropicClient] = None,
    gemini_model: str = "gemini-2.5-pro",
    anthropic_model: str = "claude-sonnet-4-5-20250929",
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
