#!/usr/bin/env python3
"""
OCR client for testing different vision models (Gemini and Anthropic).
"""

import os
import base64
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import fitz  # PyMuPDF
import anthropic
from google import genai
from google.genai.types import Part
from PIL import Image
import io
import yaml


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
    
    def pdf_page_to_image(self, pdf_path: str, page_num: int = 0) -> Image.Image:
        """Convert a PDF page to PIL Image."""
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for better quality
        img_data = pix.tobytes("png")
        doc.close()
        return Image.open(io.BytesIO(img_data))
    
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
    
    def ocr_pdf_page(self, pdf_path: str, page_num: int = 0, prompt: Optional[str] = None) -> str:
        """
        Perform OCR on a specific page of a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (0-indexed)
            prompt: Custom prompt for the model
        """
        image = self.pdf_page_to_image(pdf_path, page_num)
        return self.ocr(image, prompt)


# Test examples (same as in analyze_azure_reading_order.py)
EXAMPLES = {
    1: {
        'pdf_path': '/Users/kevinzhou/Github/pdf2epub/output/今さらですが、幼なじみを好きになってしまいました1/input_original.pdf',
        'page_num': 62,
        'description': 'Page with furigana (幼なじみ page 62)'
    },
    2: {
        'pdf_path': '/Users/kevinzhou/Github/pdf2epub/output/ゼルトリーク戦記_追放された皇子が美女たちを娶り帝国を統一するまで_ノベル/input_original.pdf',
        'page_num': 5,
        'description': 'Page without furigana (ゼルトリーク page 5)'
    },
    3: {
        'pdf_path': '/Users/kevinzhou/Github/pdf2epub/output/今さらですが、幼なじみを好きになってしまいました1/input_original.pdf',
        'page_num': 21,
        'description': 'Page with furigana (幼なじみ page 21)'
    },
    4: {
        'pdf_path': '/Users/kevinzhou/Github/pdf2epub/output/今さらですが、幼なじみを好きになってしまいました1/input_original.pdf',
        'page_num': 186,
        'description': 'Page with furigana (幼なじみ page 186)'
    },
    5: {
        'pdf_path': '/Users/kevinzhou/Github/pdf2epub/output/今さらですが、幼なじみを好きになってしまいました1/input_original.pdf',
        'page_num': 40,
        'description': 'Page with illustration (幼なじみ page 40)'
    },
    6: {
        'pdf_path': '/Users/kevinzhou/Github/pdf2epub/output/今さらですが、幼なじみを好きになってしまいました1/input_original.pdf',
        'page_num': 11,
        'description': 'Page with illustration (幼なじみ page 12)'
    }
}


def main():
    """Main function for testing OCR models."""
    parser = argparse.ArgumentParser(description='Test OCR with different vision models')
    parser.add_argument('--model', choices=['gemini', 'anthropic'], default='gemini',
                        help='Model to use for OCR')
    parser.add_argument('--example', type=int, choices=range(1, 7),
                        help='Example number to test (1-6)')
    parser.add_argument('--pdf', type=str,
                        help='Path to custom PDF file')
    parser.add_argument('--page', type=int, default=0,
                        help='Page number (0-indexed)')
    parser.add_argument('--prompt', type=str,
                        help='Custom prompt for the model')
    parser.add_argument('--api-key', type=str,
                        help='API key for the model')
    
    args = parser.parse_args()
    
    # Initialize OCR client
    try:
        client = OCRClient(model_type=args.model, api_key=args.api_key)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Determine what to process
    if args.example:
        example = EXAMPLES[args.example]
        pdf_path = example['pdf_path']
        page_num = example['page_num']
        print(f"\nUsing Example {args.example}: {example['description']}")
        print(f"PDF: {pdf_path}, Page: {page_num}")
    elif args.pdf:
        pdf_path = args.pdf
        page_num = args.page
        print(f"\nProcessing custom PDF: {pdf_path}, Page: {page_num}")
    else:
        print("Error: Please specify either --example or --pdf")
        return
    
    # Check if file exists
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        return
    
    # Perform OCR
    print(f"\nPerforming OCR with {args.model.upper()} model...")
    print("=" * 80)
    
    try:
        result = client.ocr_pdf_page(pdf_path, page_num, args.prompt)
        print("\nOCR Result:")
        print("-" * 80)
        print(result)
    except Exception as e:
        print(f"Error during OCR: {e}")
        return
    
    # Optionally save result
    output_file = f"ocr_result_{args.model}_{Path(pdf_path).stem}_p{page_num}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Model: {args.model.upper()}\n")
        f.write(f"PDF: {pdf_path}\n")
        f.write(f"Page: {page_num}\n")
        f.write("=" * 80 + "\n")
        f.write(result)
    print(f"\nResult saved to: {output_file}")


if __name__ == "__main__":
    main()
