"""
Unified LLM client interface for model-agnostic API calls.
Handles provider-specific logic and retry strategies internally.
"""

from typing import Union, List, Dict, Optional, Any, Callable, Tuple
from loguru import logger
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_random_exponential, retry_if_exception
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
        # Cache for created clients by provider name
        self._clients: Dict[str, Any] = {}
        # Track safety blocks per operation to allow retrying on different content
        self._safety_blocked_operations = {}  # {provider: set(operation_names)}

        # Default retry settings
        self._num_retries = config.get("num_retries", 3)
        self._max_backoff_seconds = config.get("max_backoff_seconds", 30)

        # Legacy client references for backward compatibility
        self._gemini_client = None
        self._anthropic_client = None
        self._openai_client = None

        # Initialize legacy clients for backward compatibility
        self._init_legacy_clients()

    def _init_legacy_clients(self):
        """Initialize clients using legacy config keys for backward compatibility."""
        if self.config.get("google_api_key"):
            self._gemini_client = GeminiClient(
                api_key=self.config["google_api_key"],
                base_url=self.config.get("google_base_url"),
                num_retries=self._num_retries,
                max_backoff_seconds=self._max_backoff_seconds
            )
            self._clients["gemini"] = self._gemini_client

        if self.config.get("anthropic_api_key"):
            self._anthropic_client = AnthropicClient(
                api_key=self.config["anthropic_api_key"],
                base_url=self.config.get("anthropic_base_url"),
                num_retries=self._num_retries,
                max_backoff_seconds=self._max_backoff_seconds
            )
            self._clients["anthropic"] = self._anthropic_client

        if self.config.get("openai_api_key"):
            self._openai_client = OpenAIClient(
                api_key=self.config["openai_api_key"],
                base_url=self.config.get("openai_base_url"),
                model=self.config.get("openai_model"),
                num_retries=self._num_retries,
                max_backoff_seconds=self._max_backoff_seconds
            )
            self._clients["openai"] = self._openai_client
            # Also register for common OpenAI-compatible providers
            self._clients["deepseek"] = self._openai_client
            self._clients["poe"] = self._openai_client

    def _get_client(self, provider_name: str):
        """
        Get or create a client for the given provider.

        Args:
            provider_name: Name of the provider (e.g., "gemini", "deepseek", "anthropic-proxy")

        Returns:
            Client instance or None if provider not configured
        """
        # Check cache first
        if provider_name in self._clients:
            return self._clients[provider_name]

        # Try to get provider config from ConfigManager structure
        providers = self.config.get("credentials", {}).get("providers", {})

        if provider_name in providers:
            provider_config = providers[provider_name]
            client = self._create_client_from_provider(provider_name, provider_config)
            if client:
                self._clients[provider_name] = client
                return client

        # Fallback: infer from provider name for legacy compatibility
        return self._get_legacy_client(provider_name)

    def _create_client_from_provider(self, provider_name: str, provider_config: Dict) -> Any:
        """
        Create a client from provider configuration.

        Args:
            provider_name: Name of the provider
            provider_config: Provider configuration dict with type, api_key, etc.

        Returns:
            Client instance
        """
        provider_type = provider_config.get("type", self._infer_provider_type(provider_name))
        api_key = provider_config.get("api_key")
        base_url = provider_config.get("base_url")

        if not api_key:
            logger.warning(f"No API key found for provider '{provider_name}'")
            return None

        if provider_type == "google":
            return GeminiClient(
                api_key=api_key,
                base_url=base_url,
                num_retries=self._num_retries,
                max_backoff_seconds=self._max_backoff_seconds
            )
        elif provider_type == "anthropic":
            return AnthropicClient(
                api_key=api_key,
                base_url=base_url,
                num_retries=self._num_retries,
                max_backoff_seconds=self._max_backoff_seconds
            )
        elif provider_type == "openai":
            return OpenAIClient(
                api_key=api_key,
                base_url=base_url,
                model=provider_config.get("default_model"),
                num_retries=self._num_retries,
                max_backoff_seconds=self._max_backoff_seconds
            )
        else:
            logger.warning(f"Unknown provider type '{provider_type}' for '{provider_name}'")
            return None

    def _get_legacy_client(self, provider_name: str):
        """Get client using legacy provider name mapping."""
        name_lower = provider_name.lower()

        if "gemini" in name_lower or provider_name == "google":
            return self._gemini_client
        elif "anthropic" in name_lower or "claude" in name_lower:
            return self._anthropic_client
        elif any(x in name_lower for x in ["openai", "deepseek", "poe"]):
            return self._openai_client

        return None

    def _infer_provider_type(self, provider_name: str) -> str:
        """Infer provider type from provider name."""
        name_lower = provider_name.lower()

        if "gemini" in name_lower or "vertex" in name_lower:
            return "google"
        elif "anthropic" in name_lower or "claude" in name_lower:
            return "anthropic"
        else:
            return "openai"

    def _get_provider_type(self, provider_name: str) -> str:
        """Get the type of a provider."""
        # Check ConfigManager structure first
        providers = self.config.get("credentials", {}).get("providers", {})
        if provider_name in providers:
            return providers[provider_name].get("type", self._infer_provider_type(provider_name))

        # Fallback to inference
        return self._infer_provider_type(provider_name)
    
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
                {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929", "max_retries": 2}
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

            # Get client for this provider
            client = self._get_client(provider)
            if not client:
                logger.warning(f"No client available for provider '{provider}'")
                attempts_summary.append(f"{provider}: no client")
                continue

            try:
                logger.info(f"Trying {provider} model {model} for {operation_name}")

                # Determine provider type and call appropriate method
                provider_type = self._get_provider_type(provider)

                if provider_type == "google":
                    response = self._generate_with_gemini(
                        prompt=prompt,
                        model=model,
                        max_retries=max_retries,
                        operation_name=operation_name,
                        client=client
                    )
                    logger.success(f"Successfully generated with {provider} for {operation_name}")
                    return response

                elif provider_type == "anthropic":
                    response = self._generate_with_anthropic(
                        prompt=prompt,
                        model=model,
                        max_retries=max_retries,
                        operation_name=operation_name,
                        client=client
                    )
                    logger.success(f"Successfully generated with {provider} for {operation_name}")
                    return response

                elif provider_type == "openai":
                    response = self._generate_with_openai(
                        prompt=prompt,
                        model=model,
                        max_retries=max_retries,
                        operation_name=operation_name,
                        client=client
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

    def generate_with_validation(
        self,
        prompt: Union[str, List[Dict]],
        model_configs: Optional[List[Dict]] = None,
        validator: Optional[Callable[[str], Tuple[bool, str]]] = None,
        validation_strategy: Optional['ValidationStrategy'] = None,
        operation_name: str = "LLM generation"
    ) -> str:
        """
        Generate content with validation and retry logic.

        This method handles:
        1. API-level retries (for transient errors)
        2. Validation retries (when output doesn't meet criteria)
        3. Model fallback (trying alternative models)
        4. Best response selection (when all attempts fail)

        Args:
            prompt: The prompt (string or list of content parts)
            model_configs: List of model configurations with api_retries and validation_retries
            validator: Optional validation function that returns (is_valid, reason)
            validation_strategy: Strategy for handling validation failures
            operation_name: Name for logging

        Returns:
            Generated and validated text

        Raises:
            Exception: If all models and retries fail
        """
        # Import here to avoid circular dependency
        from ..processors.validation_strategy import ValidationStrategy

        # Use provided strategy or create default
        if validation_strategy is None:
            validation_strategy = ValidationStrategy()

        # Use model configs from parameter or config file
        if model_configs is None:
            model_configs = self.config.get("polish_models", [
                {"provider": "gemini", "model": "gemini-2.5-pro", "max_retries": 1},
                {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929", "max_retries": 2}
            ])

        validation_strategy.clear_attempts()
        last_error = None

        for model_idx, model_config in enumerate(model_configs):
            provider = model_config["provider"]
            model = model_config["model"]

            # Parse retry configuration with backward compatibility
            api_retries, validation_retries = validation_strategy.parse_model_config(model_config)

            # Skip if provider was blocked for this specific operation
            if provider in self._safety_blocked_operations:
                if operation_name in self._safety_blocked_operations[provider]:
                    logger.info(f"Skipping {provider} for {operation_name} (blocked on this operation)")
                    continue

            # Try validation retries for this specific model
            for val_attempt in range(validation_retries + 1):
                try:
                    logger.info(
                        f"Trying {provider} model {model} for {operation_name} "
                        f"(validation attempt {val_attempt + 1}/{validation_retries + 1})"
                    )

                    # Get client for this provider
                    client = self._get_client(provider)
                    if not client:
                        logger.warning(f"No client available for provider '{provider}'")
                        break  # Skip to next model

                    # Generate with API retries handled internally
                    provider_type = self._get_provider_type(provider)

                    if provider_type == "google":
                        response = self._generate_with_gemini(
                            prompt=prompt,
                            model=model,
                            max_retries=api_retries,
                            operation_name=operation_name,
                            client=client
                        )
                    elif provider_type == "anthropic":
                        response = self._generate_with_anthropic(
                            prompt=prompt,
                            model=model,
                            max_retries=api_retries,
                            operation_name=operation_name,
                            client=client
                        )
                    elif provider_type == "openai":
                        response = self._generate_with_openai(
                            prompt=prompt,
                            model=model,
                            max_retries=api_retries,
                            operation_name=operation_name,
                            client=client
                        )
                    else:
                        logger.warning(f"Unknown provider type '{provider_type}' for '{provider}'")
                        break  # Skip to next model

                    # Validate if validator provided
                    if validator:
                        is_valid, reason = validator(response)

                        # Save error response if validation failed
                        error_output_path = None
                        if not is_valid:
                            error_output_path = validation_strategy.save_error_response(
                                unit_key=operation_name,
                                attempt_number=val_attempt + 1,
                                response=response,
                                validation_reason=reason
                            )

                        validation_strategy.record_attempt(
                            response=response,
                            model_config=model_config,
                            is_valid=is_valid,
                            validation_reason=reason,
                            attempt_number=val_attempt + 1,
                            error_output_path=error_output_path
                        )

                        if is_valid:
                            logger.success(
                                f"Successfully generated and validated with {provider} "
                                f"for {operation_name}"
                            )
                            return response

                        # Check if should retry validation with same model
                        if validation_strategy.should_retry_validation(
                            model_idx, val_attempt, validation_retries, is_valid, reason
                        ):
                            continue  # Retry with same model
                        else:
                            break  # Move to next model
                    else:
                        # No validation needed
                        logger.success(f"Successfully generated with {provider} for {operation_name}")
                        return response

                except SafetyBlockError as e:
                    # Track which operations have safety blocks
                    if provider not in self._safety_blocked_operations:
                        self._safety_blocked_operations[provider] = set()
                    self._safety_blocked_operations[provider].add(operation_name)
                    logger.warning(f"Safety block from {provider} for {operation_name}: {str(e)}")
                    break  # Move to next model, don't retry safety blocks

                except Exception as e:
                    last_error = e
                    # If we get here, it means the retry logic exhausted (30 min timeout)
                    # or a non-retryable error occurred
                    logger.warning(
                        f"API error with {provider} model {model} after retries exhausted: {e}"
                    )
                    break  # Move to next model - validation_retries are for validation failures, not API errors

            # Check if should try next model
            all_attempts_failed = not any(
                a.is_valid for a in validation_strategy.current_attempts
                if a.model_config == model_config
            )

            if not validation_strategy.should_try_next_model(
                model_idx, len(model_configs), all_attempts_failed
            ):
                break  # Don't try next model

        # All models exhausted - apply fallback strategy
        logger.warning(validation_strategy.get_summary())

        best_response = validation_strategy.select_best_response()
        if best_response:
            return best_response

        # Complete failure - include validation summary and error output paths for debugging
        summary = validation_strategy.get_summary()

        # Collect error output paths from all attempts
        error_paths = [
            a.error_output_path for a in validation_strategy.current_attempts
            if a.error_output_path
        ]
        error_paths_str = ""
        if error_paths:
            error_paths_str = f"\n\nError outputs saved to:\n" + "\n".join(f"  - {p}" for p in error_paths)

        error_msg = f"All models failed validation for {operation_name}"
        if last_error:
            raise Exception(f"{error_msg}. Last error: {last_error}\n\nValidation summary:\n{summary}{error_paths_str}")
        else:
            raise Exception(f"{error_msg}\n\nValidation summary:\n{summary}{error_paths_str}")

    def _generate_with_gemini(
        self,
        prompt: Union[str, List[Dict]],
        model: str,
        max_retries: int,
        operation_name: str,
        client: GeminiClient = None
    ) -> str:
        """Generate content with Gemini, handling retries internally."""

        # Use provided client or fall back to legacy
        gemini_client = client or self._gemini_client
        if not gemini_client:
            raise ValueError("No Gemini client available")

        # Convert prompt format for Gemini
        contents = None
        if isinstance(prompt, list):
            # Check if it's a conversation history with roles
            if prompt and isinstance(prompt[0], dict) and "role" in prompt[0]:
                # Convert conversation history to Gemini format
                # Gemini uses a different format for conversations
                from google.genai.types import Content, Part
                contents = []
                for msg in prompt:
                    role = "user" if msg["role"] == "user" else "model"
                    if isinstance(msg["content"], str):
                        contents.append(Content(role=role, parts=[Part(text=msg["content"])]))
                    elif isinstance(msg["content"], list):
                        # Handle multi-part content
                        parts = []
                        for part in msg["content"]:
                            if isinstance(part, dict) and part.get("type") == "text":
                                parts.append(Part(text=part["text"]))
                        contents.append(Content(role=role, parts=parts))
            else:
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
        config = gemini_client.get_default_config(temperature)
        # Max output tokens is already set to 65536 in get_default_config

        # Get retry config: use time-based limit for network errors (5 min default)
        # max_wait_between_retries is the max backoff between retries
        max_wait_between = self.config.get('retry', {}).get('max_wait_seconds', 120)
        # max_retry_duration is the total time to keep retrying (5 min = 300s)
        max_retry_duration = self.config.get('retry', {}).get('max_retry_duration_seconds', 300)

        # Create retry decorator with time-based limit for network/rate-limit errors
        @retry(
            retry=retry_if_exception(self._is_retryable_gemini_error),
            wait=wait_random_exponential(multiplier=2, max=max_wait_between),
            stop=stop_after_delay(max_retry_duration),
            reraise=True,
            before_sleep=lambda rs: logger.info(
                f"Retrying {operation_name} in {rs.next_action.sleep:.1f}s "
                f"(attempt {rs.attempt_number}, elapsed {rs.seconds_since_start:.0f}s)"
            )
        )
        def generate_with_retry():
            try:
                return gemini_client.generate_content_stream(
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
        operation_name: str,
        client: AnthropicClient = None
    ) -> str:
        """Generate content with Anthropic, handling retries internally."""

        # Use provided client or fall back to legacy
        anthropic_client = client or self._anthropic_client
        if not anthropic_client:
            raise ValueError("No Anthropic client available")

        # Use defaults for temperature and max_tokens
        temperature = 0.1  # Low temperature for consistent results
        max_tokens = 64000  # Claude Sonnet 4 max limit

        # Get retry config: use time-based limit for network errors (5 min default)
        max_wait_between = self.config.get('retry', {}).get('max_wait_seconds', 120)
        max_retry_duration = self.config.get('retry', {}).get('max_retry_duration_seconds', 300)

        # Create retry decorator with time-based limit for network/rate-limit errors
        @retry(
            retry=retry_if_exception(self._is_retryable_anthropic_error),
            wait=wait_random_exponential(multiplier=2, max=max_wait_between),
            stop=stop_after_delay(max_retry_duration),
            reraise=True,
            before_sleep=lambda rs: logger.info(
                f"Retrying {operation_name} in {rs.next_action.sleep:.1f}s "
                f"(attempt {rs.attempt_number}, elapsed {rs.seconds_since_start:.0f}s)"
            )
        )
        def generate_with_retry():
            try:
                return anthropic_client.generate_content(
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
        operation_name: str,
        client: OpenAIClient = None
    ) -> str:
        """Generate content with OpenAI, handling retries internally."""

        # Use provided client or fall back to legacy
        openai_client = client or self._openai_client
        if not openai_client:
            raise ValueError("No OpenAI client available")

        # Use model-specific max tokens if available
        max_tokens = 8192  # Default for most OpenAI models
        if "gpt-4" in model.lower():
            max_tokens = 8192
        elif "gpt-3.5" in model.lower():
            max_tokens = 4096

        temperature = 0.1  # Low temperature for consistent results

        # Get retry config: use time-based limit for network errors (5 min default)
        max_wait_between = self.config.get('retry', {}).get('max_wait_seconds', 120)
        max_retry_duration = self.config.get('retry', {}).get('max_retry_duration_seconds', 300)

        # Create retry decorator with time-based limit for network/rate-limit errors
        @retry(
            retry=retry_if_exception(self._is_retryable_openai_error),
            wait=wait_random_exponential(multiplier=2, max=max_wait_between),
            stop=stop_after_delay(max_retry_duration),
            reraise=True,
            before_sleep=lambda rs: logger.info(
                f"Retrying {operation_name} in {rs.next_action.sleep:.1f}s "
                f"(attempt {rs.attempt_number}, elapsed {rs.seconds_since_start:.0f}s)"
            )
        )
        def generate_with_retry():
            try:
                return openai_client.generate_content(
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