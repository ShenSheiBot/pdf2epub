"""
Common utility functions used across multiple modules.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Optional


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from config file.
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def load_book_structure(book_title: str) -> Optional[Dict]:
    """
    Load the book structure JSON file.
    
    Args:
        book_title: Title of the book
        
    Returns:
        Book structure dictionary or None if not found
    """
    structure_path = Path("output") / book_title / "book_structure.json"
    if structure_path.exists():
        with open(structure_path, "r", encoding="utf-8") as file:
            structure = json.load(file)
        return structure
    return None


def ensure_directory(directory_path: Path) -> None:
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        directory_path: Path to the directory
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for filesystem compatibility.
    
    Args:
        filename: The filename to sanitize
    
    Returns:
        Sanitized filename safe for filesystem use
    """
    return "".join(c for c in filename if c not in '<>:"/\\|?*')