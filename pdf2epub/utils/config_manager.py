"""
ConfigManager - Centralized configuration management with migration support.

Provides:
- Unified access to configuration
- Migration from old flat structure to new nested structure
- Backward compatibility
- Provider-based credential lookup
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger


class ConfigManager:
    """
    Centralized configuration manager with backward compatibility.

    Handles migration from old flat config structure to new nested structure
    while maintaining backward compatibility.
    """

    # Mapping from old flat keys to new nested paths
    # Format: "old_key" -> "new.nested.path"
    MIGRATION_MAP = {
        # Credentials - Google
        "google_api_key": "credentials.providers.gemini.api_key",
        "google_base_url": "credentials.providers.gemini.base_url",

        # Credentials - Anthropic
        "anthropic_api_key": "credentials.providers.anthropic.api_key",
        "anthropic_base_url": "credentials.providers.anthropic.base_url",
        "anthropic_model": "credentials.providers.anthropic.default_model",

        # Credentials - OpenAI (default instance)
        "openai_api_key": "credentials.providers.openai.api_key",
        "openai_base_url": "credentials.providers.openai.base_url",
        "openai_model": "credentials.providers.openai.default_model",

        # Credentials - Mistral (for OCR)
        "mistral_api_key": "credentials.providers.mistral.api_key",
        "mistral_base_url": "credentials.providers.mistral.base_url",

        # Credentials - Azure
        "azure_endpoint": "credentials.providers.azure.endpoint",
        "azure_key": "credentials.providers.azure.api_key",

        # OCR
        "ocr_backend": "ocr.backend",
        "ocr_vllm_models": "ocr.vllm_models",
        "vision_ocr_settings": "ocr.vision.settings",
        "service_account_key_path": "ocr.vision.service_account_key_path",
        "furigana_mode": "ocr.furigana_mode",

        # Translation
        "source_language": "translation.source_language",
        "target_language": "translation.target_language",
        "translation_models": "translation.models",
        "entity_extraction_model": "translation.entity_extraction_model",

        # Polish
        "polish_models": "polish.models",
        "polish_processing_mode": "polish.processing_mode",

        # Breakdown
        "breakdown_model": "breakdown.model",

        # Storage
        "s3_access_key_id": "storage.s3.access_key_id",
        "s3_secret_access_key": "storage.s3.secret_access_key",
        "s3_bucket_name": "storage.s3.bucket_name",
        "s3_endpoint": "storage.s3.endpoint",

        # General
        "max_concurrent_workers": "general.max_concurrent_workers",
        "num_retries": "retry.num_retries",
        "max_backoff_seconds": "retry.max_backoff_seconds",
    }

    # Provider type detection based on provider name patterns
    PROVIDER_TYPE_PATTERNS = {
        "gemini": "google",
        "vertex": "google",
        "claude": "anthropic",
        "anthropic": "anthropic",
        "azure": "azure",
        # Everything else defaults to "openai" (OpenAI-compatible)
    }

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize ConfigManager.

        Args:
            config_path: Path to the YAML config file
        """
        self.config_path = Path(config_path)
        self._raw_config: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}
        self._load_and_migrate()

    def _load_and_migrate(self):
        """Load config and migrate old structure to new."""
        # Load raw config
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._raw_config = yaml.safe_load(f) or {}
        else:
            logger.warning(f"Config file not found: {self.config_path}")
            self._raw_config = {}

        # Start with raw config
        self._config = self._raw_config.copy()

        # Migrate old flat keys to new nested structure
        self._migrate_config()

        # Ensure provider types are set
        self._ensure_provider_types()

    def _migrate_config(self):
        """Migrate old flat config keys to new nested structure."""
        for old_key, new_path in self.MIGRATION_MAP.items():
            if old_key in self._raw_config:
                value = self._raw_config[old_key]
                self._set_nested(new_path, value)
                logger.debug(f"Migrated {old_key} -> {new_path}")

    def _ensure_provider_types(self):
        """Ensure all providers have a type field set."""
        providers = self.get("credentials.providers", {})

        for provider_name, provider_config in providers.items():
            if isinstance(provider_config, dict) and "type" not in provider_config:
                # Infer type from provider name
                provider_type = self._infer_provider_type(provider_name)
                provider_config["type"] = provider_type
                logger.debug(f"Inferred type '{provider_type}' for provider '{provider_name}'")

    def _infer_provider_type(self, provider_name: str) -> str:
        """Infer provider type from provider name."""
        name_lower = provider_name.lower()

        for pattern, provider_type in self.PROVIDER_TYPE_PATTERNS.items():
            if pattern in name_lower:
                return provider_type

        # Default to openai (OpenAI-compatible)
        return "openai"

    def _set_nested(self, path: str, value: Any):
        """Set a value at a nested path."""
        keys = path.split(".")
        current = self._config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _get_nested(self, path: str, default: Any = None) -> Any:
        """Get a value from a nested path."""
        keys = path.split(".")
        current = self._config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by path.

        Supports both old flat keys and new nested paths.

        Args:
            path: Dot-separated path (e.g., "ocr.backend") or old flat key
            default: Default value if not found

        Returns:
            Configuration value
        """
        # First try the path directly
        value = self._get_nested(path, None)
        if value is not None:
            return value

        # Check if it's an old flat key that was migrated
        if path in self.MIGRATION_MAP:
            new_path = self.MIGRATION_MAP[path]
            value = self._get_nested(new_path, None)
            if value is not None:
                return value

        # Try as a simple key in raw config (for backward compatibility)
        if path in self._raw_config:
            return self._raw_config[path]

        return default

    def get_provider(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """
        Get provider configuration by name.

        Args:
            provider_name: Name of the provider (e.g., "deepseek", "anthropic-proxy")

        Returns:
            Provider configuration dict with type, api_key, base_url, etc.
        """
        providers = self.get("credentials.providers", {})

        if provider_name in providers:
            provider = providers[provider_name].copy()
            # Ensure type is set
            if "type" not in provider:
                provider["type"] = self._infer_provider_type(provider_name)
            return provider

        # Backward compatibility: try to construct from old flat keys
        return self._get_legacy_provider(provider_name)

    def _get_legacy_provider(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get provider config from legacy flat keys."""
        name_lower = provider_name.lower()

        if "gemini" in name_lower or provider_name == "google":
            api_key = self._raw_config.get("google_api_key")
            if api_key:
                return {
                    "type": "google",
                    "api_key": api_key
                }

        elif "anthropic" in name_lower or "claude" in name_lower:
            api_key = self._raw_config.get("anthropic_api_key")
            if api_key:
                return {
                    "type": "anthropic",
                    "api_key": api_key,
                    "base_url": self._raw_config.get("anthropic_base_url")
                }

        elif "openai" in name_lower or provider_name in ["deepseek", "poe"]:
            api_key = self._raw_config.get("openai_api_key")
            if api_key:
                return {
                    "type": "openai",
                    "api_key": api_key,
                    "base_url": self._raw_config.get("openai_base_url")
                }

        elif "azure" in name_lower:
            api_key = self._raw_config.get("azure_key")
            if api_key:
                return {
                    "type": "azure",
                    "api_key": api_key,
                    "endpoint": self._raw_config.get("azure_endpoint")
                }

        elif "mistral" in name_lower:
            api_key = self._raw_config.get("mistral_api_key")
            if api_key:
                return {
                    "type": "mistral",
                    "api_key": api_key,
                    "base_url": self._raw_config.get("mistral_base_url")
                }

        return None

    def get_provider_type(self, provider_name: str) -> Optional[str]:
        """
        Get the type of a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider type (google, anthropic, openai, azure, mistral)
        """
        provider = self.get_provider(provider_name)
        if provider:
            return provider.get("type")
        return None

    def get_model_configs(self, config_key: str) -> List[Dict[str, Any]]:
        """
        Get model configurations with resolved provider credentials.

        Args:
            config_key: Config key for models (e.g., "translation.models", "polish.models")

        Returns:
            List of model configs with provider info resolved
        """
        models = self.get(config_key, [])
        resolved = []

        for model_config in models:
            resolved_config = model_config.copy()
            provider_name = model_config.get("provider")

            if provider_name:
                provider = self.get_provider(provider_name)
                if provider:
                    # Add provider info to config
                    resolved_config["_provider_type"] = provider.get("type")
                    resolved_config["_api_key"] = provider.get("api_key")
                    resolved_config["_base_url"] = provider.get("base_url")
                else:
                    logger.warning(f"Provider '{provider_name}' not found in credentials")

            resolved.append(resolved_config)

        return resolved

    def as_dict(self) -> Dict[str, Any]:
        """
        Get the full migrated configuration as a dictionary.

        Returns:
            Full configuration dictionary
        """
        return self._config.copy()

    def as_legacy_dict(self) -> Dict[str, Any]:
        """
        Get configuration in legacy flat format for backward compatibility.

        This allows existing code to work without modification.

        Returns:
            Configuration in old flat format
        """
        legacy = self._raw_config.copy()

        # Also include new nested values at their old keys if they exist
        for old_key, new_path in self.MIGRATION_MAP.items():
            if old_key not in legacy:
                value = self._get_nested(new_path, None)
                if value is not None:
                    legacy[old_key] = value

        # Include other nested configs that code might access
        for key in ["validation_strategy", "model_output_limits", "splitting",
                    "retry", "polish", "translate", "ocr"]:
            if key in self._config and key not in legacy:
                legacy[key] = self._config[key]

        return legacy

    @property
    def title(self) -> str:
        """Get book title."""
        return self.get("title", "")

    @property
    def ocr_backend(self) -> str:
        """Get OCR backend."""
        return self.get("ocr.backend", "vision")

    @property
    def source_language(self) -> str:
        """Get source language."""
        return self.get("translation.source_language", "English")

    @property
    def target_language(self) -> str:
        """Get target language."""
        return self.get("translation.target_language", "Chinese")

    @property
    def max_concurrent_workers(self) -> int:
        """Get max concurrent workers."""
        return self.get("general.max_concurrent_workers", 8)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration with backward compatibility.

    This is a drop-in replacement for the old load_config function.
    Returns config in legacy format for existing code.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary in legacy format
    """
    manager = ConfigManager(config_path)
    return manager.as_legacy_dict()


def get_config_manager(config_path: str = "config.yaml") -> ConfigManager:
    """
    Get a ConfigManager instance.

    Args:
        config_path: Path to config file

    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)
