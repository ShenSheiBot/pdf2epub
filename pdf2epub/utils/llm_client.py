"""
Unified LLM client interface for model-agnostic API calls.
Handles provider-specific logic and retry strategies internally.
"""

from typing import Union, List, Dict, Optional, Any
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception
from .network_utils import (
    GeminiClient, 
    AnthropicClient,
    OpenAIClient,
    is_transient_gemini_error,
    is_transient_anthropic_error,
    is_transient_openai_error
)


class SafetyBlockError(Exception):
    """Raised when content is blocked for safety reasons."""
    def __init__(self, message: str, provider: str):
        self.provider = provider
        super().__init__(message)


class LLMClient:
    """
    Unified LLM client that handles multiple providers transparently.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM client with configuration.
        
        Args:
            config: Configuration dict containing API keys and settings
        """
        self.config = config
        self._gemini_client = None
        self._anthropic_client = None
        self._openai_client = None
        # Track safety blocks per operation to allow retrying on different content
        self._safety_blocked_operations = {}  # {provider: set(operation_names)}
        
        # Initialize clients based on available API keys
        if config.get("google_api_key"):
            self._gemini_client = GeminiClient(config["google_api_key"])
            
        if config.get("anthropic_api_key"):
            self._anthropic_client = AnthropicClient(
                api_key=config["anthropic_api_key"],
                base_url=config.get("anthropic_base_url")
            )
        
        if config.get("openai_api_key"):
            self._openai_client = OpenAIClient(
                api_key=config["openai_api_key"],
                base_url=config.get("openai_base_url"),
                model=config.get("openai_model")
            )
    
    def generate(
        self,
        prompt: Union[str, List[Dict]],
        model_configs: Optional[List[Dict]] = None,
        operation_name: str = "LLM generation"
    ) -> str:
        """
        Generate content using configured models with automatic fallback.
        
        Args:
            prompt: The prompt (string or list of content parts)
            model_configs: List of model configurations to try in order
                         Each dict should have: provider, model, max_retries
            operation_name: Name for logging
            
        Returns:
            Generated text
            
        Raises:
            Exception: If all models fail
        """
        # Use model configs from parameter or config file
        if model_configs is None:
            model_configs = self.config.get("polish_models", [
                {"provider": "gemini", "model": "gemini-2.5-pro", "max_retries": 1},
                {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "max_retries": 2}
            ])
        
        last_error = None
        attempts_summary = []
        
        for model_config in model_configs:
            provider = model_config["provider"]
            model = model_config["model"]
            max_retries = model_config.get("max_retries", 1)
            
            # Skip if provider was blocked for this specific operation
            if provider in self._safety_blocked_operations:
                if operation_name in self._safety_blocked_operations[provider]:
                    logger.info(f"Skipping {provider} for {operation_name} (blocked on this operation)")
                    attempts_summary.append(f"{provider}: skipped (safety on this operation)")
                    continue
                # Log if provider has had safety issues on other operations but trying anyway
                elif self._safety_blocked_operations[provider]:
                    blocked_count = len(self._safety_blocked_operations[provider])
                    logger.debug(f"{provider} had safety blocks on {blocked_count} other operation(s), trying anyway for {operation_name}")
            
            try:
                logger.info(f"Trying {provider} model {model} for {operation_name}")
                
                if provider == "gemini" and self._gemini_client:
                    response = self._generate_with_gemini(
                        prompt=prompt,
                        model=model,
                        max_retries=max_retries,
                        operation_name=operation_name
                    )
                    logger.success(f"Successfully generated with {provider} for {operation_name}")
                    return response
                    
                elif provider == "anthropic" and self._anthropic_client:
                    response = self._generate_with_anthropic(
                        prompt=prompt,
                        model=model,
                        max_retries=max_retries,
                        operation_name=operation_name
                    )
                    logger.success(f"Successfully generated with {provider} for {operation_name}")
                    return response
                    
                elif provider == "openai" and self._openai_client:
                    response = self._generate_with_openai(
                        prompt=prompt,
                        model=model,
                        max_retries=max_retries,
                        operation_name=operation_name
                    )
                    logger.success(f"Successfully generated with {provider} for {operation_name}")
                    return response
                    
                else:
                    logger.warning(f"Provider {provider} not available or not configured")
                    attempts_summary.append(f"{provider}: not configured")
                    continue
                    
            except SafetyBlockError as e:
                # Track which operations have safety blocks for this provider
                if provider not in self._safety_blocked_operations:
                    self._safety_blocked_operations[provider] = set()
                self._safety_blocked_operations[provider].add(operation_name)
                logger.warning(f"{provider} blocked for safety on {operation_name}: {e}")
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
    
    def _generate_with_gemini(
        self,
        prompt: Union[str, List[Dict]],
        model: str,
        max_retries: int,
        operation_name: str
    ) -> str:
        """Generate content with Gemini, handling retries internally."""
        
        # Convert prompt format for Gemini
        if isinstance(prompt, list):
            # Convert from Anthropic-style format to Gemini format
            contents = []
            for part in prompt:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        contents.append({"text": part["text"]})
                    else:
                        contents.append(part)
                else:
                    contents.append(part)
        else:
            contents = prompt
        
        # Configure generation with defaults
        temperature = 0.1  # Low temperature for consistent results
        config = self._gemini_client.get_default_config(temperature)
        # Max output tokens is already set to 65536 in get_default_config
        
        # Create retry decorator with specified attempts
        @retry(
            retry=retry_if_exception(self._is_retryable_gemini_error),
            wait=wait_random_exponential(multiplier=1, max=30),
            stop=stop_after_attempt(max_retries),
            reraise=True
        )
        def generate_with_retry():
            try:
                return self._gemini_client.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=config,
                    operation_name=operation_name
                )
            except Exception as e:
                # Check for safety block
                error_str = str(e).lower()
                if any(term in error_str for term in ['prohibited', 'safety', 'blocked']):
                    raise SafetyBlockError(str(e), "gemini")
                raise
        
        return generate_with_retry()
    
    def _generate_with_anthropic(
        self,
        prompt: Union[str, List[Dict]],
        model: str,
        max_retries: int,
        operation_name: str
    ) -> str:
        """Generate content with Anthropic, handling retries internally."""
        
        # Use defaults for temperature and max_tokens
        temperature = 0.1  # Low temperature for consistent results
        max_tokens = 64000  # Claude Sonnet 4 max limit
        
        # Create retry decorator with specified attempts
        @retry(
            retry=retry_if_exception(self._is_retryable_anthropic_error),
            wait=wait_random_exponential(multiplier=1, max=30),
            stop=stop_after_attempt(max_retries),
            reraise=True
        )
        def generate_with_retry():
            try:
                return self._anthropic_client.generate_content(
                    prompt=prompt,
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
        
        return generate_with_retry()
    
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
    
    def _generate_with_openai(
        self,
        prompt: Union[str, List[Dict]],
        model: str,
        max_retries: int,
        operation_name: str
    ) -> str:
        """Generate content with OpenAI, handling retries internally."""
        
        # Use model-specific max tokens if available
        max_tokens = 8192  # Default for most OpenAI models
        if "gpt-4" in model.lower():
            max_tokens = 8192
        elif "gpt-3.5" in model.lower():
            max_tokens = 4096
        
        temperature = 0.1  # Low temperature for consistent results
        
        # Create retry decorator with specified attempts
        @retry(
            retry=retry_if_exception(self._is_retryable_openai_error),
            wait=wait_random_exponential(multiplier=1, max=30),
            stop=stop_after_attempt(max_retries),
            reraise=True
        )
        def generate_with_retry():
            try:
                return self._openai_client.generate_content(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    operation_name=operation_name
                )
            except Exception as e:
                # Check for safety/content policy blocks
                error_str = str(e).lower()
                if any(term in error_str for term in ['content_policy', 'refused', 'violation']):
                    raise SafetyBlockError(str(e), "openai")
                raise
        
        return generate_with_retry()
    
    def _is_retryable_openai_error(self, exception: Exception) -> bool:
        """Check if OpenAI error should be retried (not safety blocks)."""
        if isinstance(exception, SafetyBlockError):
            return False
        return is_transient_openai_error(exception)
    
    def get_safety_stats(self) -> Dict[str, int]:
        """Get statistics about safety blocks per provider."""
        stats = {}
        for provider, blocked_ops in self._safety_blocked_operations.items():
            stats[provider] = len(blocked_ops)
        return stats
    
    def clear_safety_blocks(self, provider: Optional[str] = None):
        """Clear safety block tracking for a provider or all providers.
        
        Args:
            provider: Specific provider to clear, or None to clear all
        """
        if provider:
            if provider in self._safety_blocked_operations:
                self._safety_blocked_operations[provider].clear()
                logger.info(f"Cleared safety blocks for {provider}")
        else:
            self._safety_blocked_operations.clear()
            logger.info("Cleared all safety blocks")