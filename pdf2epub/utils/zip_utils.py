#!/usr/bin/env python3
"""
Utility functions for creating password-protected ZIP files.
"""

import random
import string
import subprocess
from pathlib import Path
from loguru import logger


def generate_random_password(length=8):
    """Generate a random password with letters, numbers, and commas.
    
    Args:
        length: Length of password to generate (default 8)
        
    Returns:
        str: Random password containing uppercase, lowercase, numbers, and comma
    """
    # Create character pool: uppercase, lowercase, numbers, and comma
    characters = string.ascii_letters + string.digits + ','
    
    # Ensure at least one of each type (except comma which is optional)
    password = [
        random.choice(string.ascii_uppercase),
        random.choice(string.ascii_lowercase),
        random.choice(string.digits),
        random.choice(',')
    ]
    
    # Fill remaining characters
    for _ in range(length - 4):
        password.append(random.choice(characters))
    
    # Shuffle to randomize position
    random.shuffle(password)
    
    return ''.join(password)


def create_password_protected_zip(file_path, password=None):
    """Create a password-protected ZIP file with password in filename.
    
    Args:
        file_path: Path to the file to zip (e.g., EPUB file)
        password: Optional password to use. If None, generates random password.
        
    Returns:
        tuple: (zip_path, password) or (None, None) if failed
    """
    file_path = Path(file_path).absolute()
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None, None
    
    # Generate password if not provided
    if password is None:
        password = generate_random_password(8)
    
    # Create zip filename with password in title
    base_name = file_path.stem
    zip_filename = f"{base_name}_【密码：{password}】.zip"
    zip_path = file_path.parent / zip_filename
    
    # Create password-protected zip using command line zip
    try:
        # Build command with absolute paths
        cmd = [
            'zip', 
            '-j',  # Store just the file, not the path
            '-P', password,
            str(zip_path.absolute()),
            str(file_path.absolute())
        ]
        
        # Run command 
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            logger.success(f"Created password-protected ZIP: {zip_path}")
            logger.info(f"Password: {password}")
            return zip_path, password
        else:
            logger.error(f"Failed to create password-protected ZIP")
            logger.error(f"Return code: {result.returncode}")
            logger.error(f"Stderr: {result.stderr}")
            logger.error(f"Stdout: {result.stdout}")
            return None, None
            
    except FileNotFoundError:
        logger.error("'zip' command not found. Please install zip utility.")
        return None, None
    except Exception as e:
        logger.error(f"Error creating password-protected ZIP: {e}")
        return None, None