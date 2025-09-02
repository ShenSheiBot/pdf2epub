"""
Utility functions for model selection and configuration.
"""

from typing import List, Dict, Any, Optional


def get_cheapest_model_configs(
    config: Dict[str, Any],
    max_models: int = 3,
    include_providers: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Get model configurations ordered by cost (cheapest first).
    
    Args:
        config: Configuration dictionary containing API keys and settings
        max_models: Maximum number of models to return
        include_providers: List of providers to include (None = all available)
    
    Returns:
        List of model configurations ordered by cost
    """
    model_configs = []
    # Add Gemini Flash (third cheapest)
    if config.get('google_api_key'):
        if not include_providers or 'gemini' in include_providers:
            model_configs.append({
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "max_retries": 1
            })
    
    # Add regular OpenAI models if not DeepSeek
    if config.get('openai_api_key') and config.get('openai_base_url'):
        if 'deepseek' not in config.get('openai_base_url', '').lower():
            if not include_providers or 'openai' in include_providers:
                # Standard OpenAI models are more expensive
                model_configs.append({
                    "provider": "openai",
                    "model": config.get('openai_model', 'gpt-4o-mini'),
                    "max_retries": 1
                })
    
    return model_configs[:max_models]


def get_fast_model_configs(
    config: Dict[str, Any],
    max_models: int = 2
) -> List[Dict[str, Any]]:
    """
    Get fast model configurations for quick operations like validation.
    Prioritizes speed over cost.
    
    Args:
        config: Configuration dictionary
        max_models: Maximum number of models to return
    
    Returns:
        List of fast model configurations
    """
    model_configs = []
    
    # Gemini Flash is very fast
    if config.get('google_api_key'):
        model_configs.append({
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "max_retries": 1
        })
    
    # DeepSeek is fast if configured
    if config.get('openai_api_key') and config.get('openai_base_url'):
        if 'deepseek' in config.get('openai_base_url', '').lower():
            model_configs.append({
                "provider": "openai",
                "model": config.get('openai_model', 'deepseek-chat'),
                "max_retries": 1
            })
    
    # Claude Haiku is fast
    if config.get('anthropic_api_key'):
        model_configs.append({
            "provider": "anthropic",
            "model": "claude-3-5-haiku-20241022",
            "max_retries": 1
        })
    
    return model_configs[:max_models]


def get_quality_model_configs(
    config: Dict[str, Any],
    max_models: int = 3
) -> List[Dict[str, Any]]:
    """
    Get high-quality model configurations for important operations.
    Prioritizes quality over cost.
    
    Args:
        config: Configuration dictionary
        max_models: Maximum number of models to return
    
    Returns:
        List of high-quality model configurations
    """
    model_configs = []
    
    # Anthropic Sonnet for high quality
    if config.get('anthropic_api_key'):
        model_configs.append({
            "provider": "anthropic",
            "model": config.get('anthropic_model', 'claude-sonnet-4-20250514'),
            "max_retries": 2
        })
    
    # Gemini Pro for high quality
    if config.get('google_api_key'):
        model_configs.append({
            "provider": "gemini",
            "model": "gemini-2.5-pro",
            "max_retries": 2
        })
    
    # GPT-4o for high quality (if not using DeepSeek)
    if config.get('openai_api_key') and config.get('openai_base_url'):
        if 'deepseek' not in config.get('openai_base_url', '').lower():
            model_configs.append({
                "provider": "openai",
                "model": config.get('openai_model', 'gpt-4o'),
                "max_retries": 2
            })
    
    return model_configs[:max_models]
