import yaml
from typing import Optional


def get_config_value(config_path: str, key: str) -> Optional[str]:
    """
    Safely extract a value from YAML config file.
    
    Args:
        config_path: Path to the YAML config file
        key: The key to extract the value for
        
    Returns:
        The value for the given key or None if not found
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return str(config.get(key, '')).strip()
    except Exception as e:
        print(f"Error reading config value {key}: {e}")
        return None


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python config_parser.py <config_file> <key>")
        sys.exit(1)
        
    value = get_config_value(sys.argv[1], sys.argv[2])
    if value is not None:
        print(value)
