"""OCR backend implementations for different services."""

from typing import Dict, Any, Callable, Tuple


def get_backend(backend_name: str) -> Tuple[Callable, Callable]:
    """
    Get the init_client and process_page functions for a backend.
    
    Args:
        backend_name: Name of the backend ('azure', 'vision', or 'vllm')
        
    Returns:
        Tuple of (init_client, process_page) functions
        
    Raises:
        ValueError: If backend_name is not recognized
    """
    if backend_name == 'azure':
        from .azure import init_client, process_page
    elif backend_name == 'vision':
        from .vision import init_client, process_page
    elif backend_name == 'vllm':
        from .vllm import init_client, process_page
    else:
        raise ValueError(f"Unknown backend: {backend_name}")
    
    return init_client, process_page


__all__ = ['get_backend']
