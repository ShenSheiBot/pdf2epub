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




