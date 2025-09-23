#!/usr/bin/env python3
"""Improved Azure backend with paragraph-aware text reconstruction."""

import os
from pathlib import Path
import yaml
import io
import numpy as np
from PIL import Image
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import base64
from typing import Dict, Tuple, Any, List
from loguru import logger

from pdf2epub.utils.logging_config import configure_logging

# Configure logger
logger = configure_logging()


def init_client(config: Dict) -> DocumentIntelligenceClient:
    """Initialize the Azure Document Intelligence client."""
    azure_key = config.get('azure_di_key') or os.getenv('AZURE_DI_KEY')
    azure_endpoint = config.get('azure_di_endpoint') or os.getenv('AZURE_DI_ENDPOINT')

    if not azure_key or not azure_endpoint:
        raise ValueError("Azure credentials not found in config or environment variables")

    return DocumentIntelligenceClient(
        endpoint=azure_endpoint,
        credential=AzureKeyCredential(azure_key)
    )


def _call_azure_api_improved(client, img_bytes, use_layout=True, extract_figures=False, request_markdown=False):
    """Calls the Azure Document Intelligence API with improved settings."""
    from azure.core.exceptions import AzureError

    logger.info("Calling Azure Document Intelligence API (improved)...")
    try:
        model_id = "prebuilt-layout" if use_layout else "prebuilt-read"
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        # Prepare kwargs for the API call
        api_kwargs = {
            "model_id": model_id,
            "body": {"base64Source": img_base64},
            "locale": "ja-JP"
        }

        # Add features if using layout
        if use_layout:
            # Request styleFont for bold detection
            api_kwargs["features"] = ["languages", "styleFont"]

        # Request Markdown output for better paragraph structure
        if request_markdown:
            api_kwargs["output_content_format"] = "markdown"

        # Add figure extraction output if requested
        if extract_figures and use_layout:
            api_kwargs["output"] = ["figures"]

        # Make the API call
        poller = client.begin_analyze_document(**api_kwargs)
        result = poller.result()

        # Store the operation ID for potential figure downloads
        result._operation_id = poller.details.get('operation_location', '').split('/')[-1] if hasattr(poller, 'details') else None

        logger.success("Azure Document Intelligence analysis completed.")
        return result

    except AzureError as e:
        logger.error(f"Azure API call failed: {e}")
        raise


def process_page_improved(client: DocumentIntelligenceClient, img_bytes: bytes, page_num: int, config: Dict,
                          base_output_dir: Path = None, verbose: bool = False) -> Dict:
    """
    Process a single page using improved Azure Document Intelligence with paragraph grouping.
    """
    # Call Azure API with Markdown output for better structure
    result = _call_azure_api_improved(
        client,
        img_bytes,
        use_layout=True,
        extract_figures=config.get('use_azure_illustrations', False),
        request_markdown=True
    )

    # Get the markdown content directly if available
    if hasattr(result, 'content') and result.content:
        markdown_text = result.content

        # Process the markdown to ensure proper formatting
        lines = markdown_text.split('\n')
        processed_lines = []

        for line in lines:
            # Skip empty lines
            if not line.strip():
                processed_lines.append('')
                continue

            # Keep headers and lists as-is
            if line.startswith('#') or line.startswith('- ') or line.startswith('* '):
                processed_lines.append(line)
            else:
                # For regular text, don't break mid-sentence
                processed_lines.append(line)

        clean_text = '\n'.join(processed_lines)

        if verbose:
            print("\n" + "="*80)
            print("IMPROVED MARKDOWN OUTPUT:")
            print("="*80)
            print(clean_text)

        return {
            'text': clean_text,
            'illustrations': [],
            'raw_result': result
        }

    # Fallback to paragraph-based reconstruction if markdown not available
    return _reconstruct_from_paragraphs(result, verbose)


def _reconstruct_from_paragraphs(result, verbose=False):
    """Reconstruct text using paragraph information from Azure."""

    if not hasattr(result, 'paragraphs') or not result.paragraphs:
        # Fallback to simple content
        if hasattr(result, 'content'):
            return {'text': result.content, 'illustrations': []}
        return {'text': '', 'illustrations': []}

    output_lines = []

    # Group paragraphs by role
    title_paragraphs = []
    section_heading_paragraphs = []
    body_paragraphs = []

    for para in result.paragraphs:
        role = getattr(para, 'role', None)
        content = getattr(para, 'content', '')

        if role == 'title':
            title_paragraphs.append(content)
        elif role == 'sectionHeading':
            section_heading_paragraphs.append(content)
        else:
            body_paragraphs.append(content)

    # Output titles first with # markers
    for title in title_paragraphs:
        output_lines.append(f"# {title}")
        output_lines.append('')  # Add blank line after title

    # Output section headings with ## markers
    for heading in section_heading_paragraphs:
        output_lines.append(f"## {heading}")
        output_lines.append('')  # Add blank line after heading

    # Output body paragraphs
    for para in body_paragraphs:
        # Don't add line breaks within a paragraph
        # Just output the whole paragraph as one block
        output_lines.append(para)
        output_lines.append('')  # Add blank line between paragraphs

    clean_text = '\n'.join(output_lines).strip()

    if verbose:
        print("\n" + "="*80)
        print("PARAGRAPH-BASED RECONSTRUCTION:")
        print("="*80)
        print(f"Found {len(title_paragraphs)} titles")
        print(f"Found {len(section_heading_paragraphs)} section headings")
        print(f"Found {len(body_paragraphs)} body paragraphs")
        print("\nReconstructed text:")
        print("-"*40)
        print(clean_text)

    return {
        'text': clean_text,
        'illustrations': []
    }


# Make it compatible with existing interface
def process_page(client: DocumentIntelligenceClient, img_bytes: bytes, page_num: int, config: Dict,
                 base_output_dir: Path = None, verbose: bool = False) -> Dict:
    """Wrapper for compatibility with existing code."""
    return process_page_improved(client, img_bytes, page_num, config, base_output_dir, verbose)