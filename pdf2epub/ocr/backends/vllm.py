#!/usr/bin/env python3
"""Vision Language Model (VLLM) OCR backend using Claude and Gemini."""

import os
import base64
from pathlib import Path
from typing import Optional, Dict, Any
import anthropic
from google import genai
from google.genai.types import Part
from PIL import Image
import io
import yaml
import numpy as np
from loguru import logger

from pdf2epub.utils.logging_config import configure_logging
from ..illustration_extractor import extract_illustrations

# Configure logger
logger = configure_logging()


class OCRClient:
    """Client for performing OCR using different vision models."""
    
    def __init__(self, model_type: str = "gemini", api_key: Optional[str] = None, config_path: str = "config.yaml"):
        """
        Initialize the OCR client.
        
        Args:
            model_type: Either "gemini" or "anthropic"
            api_key: API key for the model. If not provided, will look in config.yaml then env vars.
            config_path: Path to config.yaml file
        """
        self.model_type = model_type.lower()
        
        # Try to load config
        config = {}
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        if self.model_type == "gemini":
            # Try: provided key -> config.yaml -> env var
            key = api_key or config.get('google_api_key') or os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError("Gemini API key not provided in arguments, config.yaml, or GEMINI_API_KEY env var")
            
            # Initialize Gemini client
            self.client = genai.Client(api_key=key)
            
            # Use model from config if available
            self.model_name = config.get('model', 'gemini-1.5-flash')
            
        elif self.model_type == "anthropic":
            # Try: provided key -> config.yaml -> env var
            key = api_key or config.get('anthropic_api_key') or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("Anthropic API key not provided in arguments, config.yaml, or ANTHROPIC_API_KEY env var")
            
            # Check if custom base URL is provided in config
            base_url = config.get('anthropic_base_url')
            if base_url:
                self.client = anthropic.Anthropic(api_key=key, base_url=base_url)
            else:
                self.client = anthropic.Anthropic(api_key=key)
            
            # Store model name from config for later use
            self.anthropic_model = config.get('anthropic_model', 'claude-3-5-sonnet-20241022')
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'gemini' or 'anthropic'")
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def ocr_gemini(self, image: Image.Image, prompt: str) -> str:
        """Perform OCR using Gemini model."""
        # Convert image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        # Create image part
        image_part = Part.from_bytes(data=img_bytes, mime_type="image/png")
        
        # Generate content
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image_part]
        )
        
        return response.text
    
    def ocr_anthropic(self, image: Image.Image, prompt: str) -> str:
        """Perform OCR using Anthropic Claude model."""
        base64_image = self.image_to_base64(image)
        
        message = self.client.messages.create(
            model=self.anthropic_model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image
                            }
                        }
                    ]
                }
            ]
        )
        return message.content[0].text
    
    def ocr(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """
        Perform OCR on an image.
        
        Args:
            image: PIL Image to perform OCR on
            prompt: Custom prompt for the model. If not provided, uses default.
        """
        if prompt is None:
            prompt = """Extract all Japanese text from this image, preserving the exact layout and reading order.
For vertical Japanese text (read right-to-left), maintain the column structure.
Include all furigana (ruby text) inline with parentheses after the kanji.
Example format: 漢字(かんじ)

Please extract the complete text maintaining the original structure."""
        
        if self.model_type == "gemini":
            return self.ocr_gemini(image, prompt)
        else:
            return self.ocr_anthropic(image, prompt)
    


# Interface functions for ocr_chapters_jp.py compatibility
def init_client(config: Dict) -> OCRClient:
    """
    Initialize VLLM OCR client for use with ocr_chapters_jp.py.
    
    Args:
        config: Configuration dictionary from config.yaml
        
    Returns:
        OCRClient instance configured based on ocr_vllm_models in config
    """
    # Get VLLM model configuration
    vllm_models = config.get('ocr_vllm_models', [])
    if not vllm_models:
        raise ValueError("No ocr_vllm_models configured in config.yaml")
    
    # Use the first configured model
    model_config = vllm_models[0]
    provider = model_config.get('provider', 'anthropic')
    
    # Map provider to model type for OCRClient
    if provider == 'anthropic':
        model_type = 'anthropic'
        # Set the model name in config for OCRClient to use
        config['anthropic_model'] = model_config.get('model', 'claude-3-5-sonnet-20241022')
    elif provider == 'google' or provider == 'gemini':
        model_type = 'gemini'
        config['model'] = model_config.get('model', 'gemini-1.5-flash')
    else:
        raise ValueError(f"Unsupported VLLM provider: {provider}")
    
    return OCRClient(model_type=model_type, config_path='config.yaml')


def process_page(
    client: OCRClient,
    img_bytes: bytes,
    page_num: int,
    config: Dict,
    base_output_dir: Path = None,
    verbose: bool = False,
) -> Dict:
    """
    Process a single page using VLLM OCR.
    Interface function for ocr_chapters_jp.py.
    
    Args:
        client: OCRClient instance
        img_bytes: Image data as bytes
        page_num: Page number being processed
        config: Configuration dictionary
        base_output_dir: Base output directory (typically output/{book_title})
        verbose: If True, enables detailed logging
        
    Returns:
        Dictionary with:
            - text: Extracted text from the page
            - illustrations: List of detected illustrations
            - columns: Column classification data (empty for VLLM)
            - viz_data: Visualization data (empty for VLLM)
    """
    # Convert bytes to PIL Image
    img = Image.open(io.BytesIO(img_bytes))
    img_array = np.array(img)
    
    # Log if verbose
    if verbose:
        logger.info(f"Processing page {page_num} with VLLM OCR (provider: {client.model_type})")
    
    # Get custom prompt from config if available
    prompt = config.get('ocr_vllm_prompt', None)
    if prompt is None:
        prompt = """Extract all Japanese text from this image, preserving the exact layout and reading order.
For vertical Japanese text (read right-to-left), maintain the column structure.
Include all furigana (ruby text) inline with parentheses after the kanji.
Example format: 漢字(かんじ)

If the page contains an illustration or image (not just text), include "[illustration]" in your output at the appropriate position.

Please extract the complete text maintaining the original structure."""
    
    # Perform OCR
    try:
        text = client.ocr(img, prompt)
    except Exception as e:
        logger.error(f"VLLM OCR failed for page {page_num}: {e}")
        text = ""
    
    # Extract illustrations using the illustration extractor
    # For VLLM, we pass the OCR text as text_annotation so it can check for [illustration] markers
    illustrations = []
    if base_output_dir:
        try:
            illustrations = extract_illustrations(
                img_array=img_array,
                backend="vllm",
                text_annotation=text,  # Pass the OCR text to check for [illustration] markers
                config=config,
                page_num=page_num,
                output_dir=base_output_dir,
            )
        except Exception as e:
            logger.warning(f"Illustration extraction failed for page {page_num}: {e}")
            illustrations = []
    
    return {
        "text": text,
        "illustrations": illustrations,
        "columns": {},  # VLLM doesn't provide column data
        "viz_data": [],  # VLLM doesn't provide visualization data
    }


