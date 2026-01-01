"""
================================================================================
⚠️  DEPRECATED MODULE / 已弃用模块
================================================================================
This module is part of the LEGACY workflow and may be removed in future versions.
此模块属于旧版工作流，可能会在未来版本中移除。

RECOMMENDED workflow / 推荐的新工作流:
    pdf2epub ocr-pages -i <pdf>   # Page-level OCR
    pdf2epub refine               # Generate toc_tree.json
    pdf2epub polish
    pdf2epub build-epub

This module (breakdown.py) uses Gemini to analyze PDF structure and generates
book_structure.json. The new workflow uses ocr-pages + refine which generates
toc_tree.json with more accurate boundary detection.
================================================================================
"""

import yaml
from pathlib import Path
from google.genai.types import Part
from .utils.common import parse_llm_json
from .utils.network_utils import GeminiClient
from .utils.pdf_utils import add_page_number_patches, preprocess_pdf
import argparse
from loguru import logger
from .utils.logging_config import configure_logging

# Configure logger
logger = configure_logging()


def load_config(config_path="config.yaml"):
    """Load configuration from config file."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def analyze_pdf_structure(client: GeminiClient, pdf_path, book_title, config):
    """Use Gemini model to analyze the PDF structure from the full PDF."""
    prompt = f"""
    Analyze this book PDF with title "{book_title}" and provide a detailed breakdown of its structure.
    Include the following elements:
    1. Author name(s) - Extract from the cover page, title page, or copyright page
    2. Cover page (page number)
    3. Table of contents (page numbers)
    4. All chapters and subchapters as referenced in the table of contents
    5. Back cover page (page number)

    Important: Use the PDF page numbers (not the printed page numbers that might appear in the table of contents).
    Note that nearby chapters may overlap if there are no page breaks.
    Keep the original language for all titles and author names.

    Additionally, identify special chapter types:
    - If a chapter consists ONLY of footnotes, endnotes, or references for citations in other chapters, add a "type": "notes" field
    - If any chapter's notes are at the end of itself, then there should be NO note chapter.
    - A book contains at most one note chapter.
    - Abbreviations, Bibliography, Index, or Summary Table are NOT considered as notes. Only literal `Notes` with [1], [2], [3]... are considered as notes.
    - Regular content chapters should not have a "type" field

    Also analyze the content characteristics:
    - **language**: The primary language of the book content (e.g., "english", "japanese", "chinese", "french", "german", etc.)
    - **is_vertical_text**: true if the PDF contains vertical text layout (縦書き), false otherwise
    - **has_footnotes**: true if the content has footnotes, endnotes, or citations with superscript numbers (like ¹²³, [1], $^{{1}}$), false otherwise

    Return the result in the following JSON structure:
    {{
        "author": string,  # Author name(s) as they appear in the book. If multiple authors, separate with commas. If no author found, use "Unknown Author"
        "language": string,  # Primary language of the content (e.g., "english", "japanese", "chinese")
        "is_vertical_text": boolean,  # true if vertical text layout, false otherwise
        "has_footnotes": boolean,  # true if content has footnotes/citations, false otherwise
        "cover_page": {{
            "page_number": int
        }},
        "table_of_contents": {{
            "start_page": int,
            "end_page": int,
            "entries": [
                {{
                    "title": string,
                    "page_number": int,
                    "level": int  # 1 for main chapter, 2 for subchapter, etc.
                }}
            ]
        }},
        "chapters": [
            {{
                "title": string,
                "start_page": int,
                "end_page": int,
                "level": int,
                "type": string,  # Optional: "notes" for footnote/reference chapters
                "subchapters": [
                    {{
                        "title": string,
                        "start_page": int,
                        "end_page": int,
                        "level": int
                    }}
                ]
            }}
        ],
        "back_cover": {{
            "page_number": int
        }}
    }}

    Example of a notes chapter:
    {{
        "title": "Notes",
        "start_page": 250,
        "end_page": 275,
        "level": 1,
        "type": "notes"
    }}
    """

    # Read the PDF file as binary
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()

    # Create parts for the multimodal input - text and PDF data
    parts = [
        prompt,
        Part.from_bytes(data=pdf_data, mime_type="application/pdf"),
    ]

    # Get model from config
    model = config.get("breakdown_model", config.get("model", "gemini-2.5-pro"))
    
    # Get default config from client and modify for JSON response  
    generation_config = client.get_default_config(temperature=0.1)
    generation_config.response_mime_type = "application/json"
    
    # Generate content using the new API
    response_text = client.generate_content_stream(
        model=model,
        contents=parts,
        config=generation_config,
        operation_name="PDF structure analysis"
    )
    
    logger.debug(f"API response length: {len(response_text)} chars")

    # Parse the response as JSON
    return parse_llm_json(response_text, operation_name="PDF breakdown")


def detect_front_and_back_matter(structure, pdf_path):
    """
    Automatically detect front matter and back matter based on page gaps.
    Front matter: pages between cover and first chapter
    Back matter: pages between last chapter and back cover
    
    Args:
        structure: The book structure dict from LLM analysis
        pdf_path: Path to the PDF file to get total page count
    
    Returns:
        Updated structure with front_matter and back_matter sections added
    """
    # Get total page count from PDF
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
    except Exception as e:
        logger.error(f"Failed to get page count from PDF: {e}")
        return structure
    
    # Detect front matter
    cover_page = structure.get('cover_page', {}).get('page_number', 1)
    first_chapter_start = None
    
    if structure.get('chapters'):
        first_chapter_start = structure['chapters'][0].get('start_page')
    
    if first_chapter_start and first_chapter_start > cover_page + 1:
        # There are pages between cover and first chapter
        front_matter_start = cover_page + 1
        front_matter_end = first_chapter_start - 1
        
        structure['front_matter'] = {
            'start_page': front_matter_start,
            'end_page': front_matter_end
        }
        logger.info(f"Detected front matter: pages {front_matter_start}-{front_matter_end}")
    
    # Detect back matter
    last_chapter_end = None
    back_cover_page = structure.get('back_cover', {}).get('page_number')
    
    if structure.get('chapters'):
        last_chapter = structure['chapters'][-1]
        last_chapter_end = last_chapter.get('end_page')
    
    if last_chapter_end:
        # Determine the end of back matter
        if back_cover_page and back_cover_page > last_chapter_end + 1:
            # There are pages between last chapter and back cover
            back_matter_start = last_chapter_end + 1
            back_matter_end = back_cover_page - 1
        elif last_chapter_end < total_pages:
            # No back cover specified, but there are pages after last chapter
            back_matter_start = last_chapter_end + 1
            back_matter_end = total_pages
        else:
            back_matter_start = None
            back_matter_end = None
        
        if back_matter_start and back_matter_end and back_matter_start <= back_matter_end:
            structure['back_matter'] = {
                'start_page': back_matter_start,
                'end_page': back_matter_end
            }
            logger.info(f"Detected back matter: pages {back_matter_start}-{back_matter_end}")
    
    return structure


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process a PDF book for structure analysis.')
    parser.add_argument('-i', '--input', required=True, help='Path to input PDF file')
    parser.add_argument('-c', '--config', default='config.yaml', help='Path to config file (default: config.yaml)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)

    # Get provider config for breakdown
    breakdown_config = config.get("breakdown", {})
    provider_name = breakdown_config.get("provider", "gemini")
    providers = config.get("credentials", {}).get("providers", {})
    provider_config = providers.get(provider_name, {})

    api_key = provider_config.get("api_key") or config.get("google_api_key")
    base_url = provider_config.get("base_url") or config.get("google_base_url")
    vertexai = provider_config.get("vertexai", False)
    extra_headers = provider_config.get("extra_headers")
    
    # Get input PDF path
    input_pdf = Path(args.input)
    
    # Get book title from config instead of PDF filename
    book_title = config.get("title")
    if not book_title:
        # Fallback to PDF filename if title not in config
        book_title = input_pdf.stem
        logger.warning(f"No title found in config, using PDF filename: {book_title}")
    
    # Define output directory based on config title
    output_dir = Path("output") / Path(book_title)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if API key exists
    if not api_key:
        raise ValueError("Google API key not found in config.yaml")

    # Setup Gemini API
    client = GeminiClient(
        api_key=api_key,
        base_url=base_url,
        vertexai=vertexai,
        extra_headers=extra_headers
    )

    # Preprocess and get the PDF path to use
    processed_pdf = preprocess_pdf(input_pdf, output_dir)
    
    # Analyze PDF structure using Gemini
    logger.info(f"Analyzing PDF structure for '{book_title}'...")
    try:
        structure = analyze_pdf_structure(client, processed_pdf, book_title, config)
        
        # Add book title to the structure
        structure['book_title'] = book_title
        
        # Detect front matter and back matter automatically
        structure = detect_front_and_back_matter(structure, processed_pdf)
        
        # Validate and fix the structure (remove overlaps and add missing pages)
        from pdf2epub.utils.structure_validator import validate_structure
        
        # Get total pages from PDF for missing page detection
        import fitz
        with fitz.open(processed_pdf) as pdf:
            total_pages = len(pdf)
        
        logger.info("Validating and fixing book structure...")
        structure = validate_structure(structure, fix=True, total_pages=total_pages)

        # Save the structured output to the output directory
        output_file = output_dir / "book_structure.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(structure, f, ensure_ascii=False, indent=2)

        logger.success(f"Book structure analysis completed and saved to {output_file}")

        # Print a summary
        logger.info("\nStructure Summary:")
        logger.info(
            f"Cover page: {structure.get('cover_page', {}).get('page_number', 'Not found')}"
        )
        
        # Print front matter if detected
        if 'front_matter' in structure:
            logger.info(
                f"Front matter: Pages {structure['front_matter']['start_page']}-"
                f"{structure['front_matter']['end_page']}"
            )
        
        logger.info(
            f"Table of contents: Pages {structure.get('table_of_contents', {}).get('start_page', 'N/A')}-"
            f"{structure.get('table_of_contents', {}).get('end_page', 'N/A')}"
        )
        logger.info(f"Total chapters: {len(structure.get('chapters', []))}")
        
        # Print back matter if detected
        if 'back_matter' in structure:
            logger.info(
                f"Back matter: Pages {structure['back_matter']['start_page']}-"
                f"{structure['back_matter']['end_page']}"
            )
        
        logger.info(
            f"Back cover: {structure.get('back_cover', {}).get('page_number', 'Not found')}"
        )

    except Exception as e:
        logger.error(f"Error analyzing PDF structure: {e}")
        raise


if __name__ == "__main__":
    main()
