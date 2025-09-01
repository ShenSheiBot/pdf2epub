import json
import yaml
import shutil
from pathlib import Path
from google.genai.types import Part
from utils.network_utils import GeminiClient
from pdf_compressor import compress_pdf
import argparse
from loguru import logger
from utils.logging_config import configure_logging
import fitz  # PyMuPDF for PDF manipulation

# Configure logger
logger = configure_logging()


def load_config(config_path="config.yaml"):
    """Load configuration from config file."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def add_page_number_patches(pdf_path, output_path=None):
    """
    Add white patches with actual PDF page numbers in the corners of each page.
    This helps the LLM avoid being misled by printed page numbers in the book.
    
    Args:
        pdf_path: Path to the input PDF
        output_path: Path for the output PDF (if None, overwrites input)
    
    Returns:
        Path to the patched PDF
    """
    if output_path is None:
        output_path = pdf_path
    
    try:
        doc = fitz.open(pdf_path)
        
        total_pages = len(doc)
        for page_num, page in enumerate(doc, 1):
            # Show progress every 10 pages or at the end
            if page_num % 10 == 0 or page_num == total_pages:
                logger.info(f"Processing page {page_num}/{total_pages}...")
            
            # Get page dimensions
            rect = page.rect
            width = rect.width
            height = rect.height
            
            # Define patch size relative to page dimensions
            # Use ~10% of width and ~5% of height as base, with minimum sizes
            patch_width = max(120, int(width * 0.15))  # 15% of page width, min 120px
            patch_height = max(50, int(height * 0.05))  # 5% of page height, min 50px
            # Scale font size based on patch height
            font_size = max(16, int(patch_height * 0.32))  # ~32% of patch height, min 16pt
            
            # Corner positions: top-left, top-right, bottom-left, bottom-right
            # Adjusted margins to ensure better coverage
            corner_positions = [
                (5, 5),  # top-left
                (width - patch_width - 5, 5),  # top-right
                (5, height - patch_height - 5),  # bottom-left
                (width - patch_width - 5, height - patch_height - 5)  # bottom-right
            ]
            
            for x, y in corner_positions:
                white_rect = fitz.Rect(x, y, x + patch_width, y + patch_height)
                
                # Draw white filled rectangle
                page.draw_rect(white_rect, color=(1, 1, 1), fill=(1, 1, 1))
                
                # Add the black border
                page.draw_rect(white_rect, color=(0, 0, 0), width=1, fill=None)
                
                # Add the text on top
                text_point = fitz.Point(x + 15, y + patch_height / 2 + 6)
                page.insert_text(
                    text_point,
                    f"PDF Page: {page_num}",  # More descriptive label
                    fontsize=font_size,
                    color=(0, 0, 0),
                    fontname="helv"  # Explicitly specify font
                )
        
        # Get page count before closing
        page_count = len(doc)
        
        # Save the modified PDF
        doc.save(output_path)
        doc.close()
        
        logger.info(f"Added page number patches to {page_count} pages")
        return Path(output_path)
        
    except Exception as e:
        logger.error(f"Failed to add page number patches: {e}")
        # Return original path if patching fails
        return Path(pdf_path)


def preprocess_pdf(input_pdf, output_dir):
    """
    Preprocess PDF: add page number patches and compress if necessary.
    Keeps original as input_original.pdf and creates processed version as input.pdf.
    Returns the path to the PDF that should be used.
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define paths
    processed_pdf = output_dir / "input.pdf"
    original_pdf = output_dir / "input_original.pdf"

    # If already processed, just return the processed file
    if processed_pdf.exists() and original_pdf.exists():
        logger.info("Using existing preprocessed PDF")
        return processed_pdf

    # First time processing - save the original
    if not original_pdf.exists():
        shutil.copy2(input_pdf, original_pdf)
        logger.info(f"Saved original PDF as: {original_pdf}")

    # Add page number patches to create input.pdf
    logger.info("Adding page number patches to PDF...")

    # Get file size in MB
    file_size_mb = processed_pdf.stat().st_size / (1024 * 1024)

    # If file size > 45MB, compress it
    if file_size_mb > 45:
        logger.warning(f"PDF file size ({file_size_mb:.2f}MB) exceeds 45MB. Compressing...")

        # Start with moderate compression settings
        compression_settings = [
            # (dpi, quality, grayscale)
            (150, 60, False),  # Medium compression
            (120, 40, False),  # Higher compression
            (100, 30, True),   # Aggressive compression with grayscale
            (80, 25, True),    # Very aggressive
            (72, 20, True),    # Ultra aggressive (minimum readable)
            (60, 15, True)     # Extreme compression (last resort)
        ]

        # Try compression with increasingly aggressive settings until size is under limit
        current_size_mb = file_size_mb
        for dpi, quality, grayscale in compression_settings:
            # Create a fresh temporary file for each compression attempt
            temp_output = output_dir / f"compressed_temp_{dpi}_{quality}.pdf"
            try:
                logger.info(f"Trying compression with DPI={dpi}, quality={quality}, grayscale={grayscale}...")
                logger.info(f"Current file size before compression: {current_size_mb:.2f}MB")
                
                success, stats = compress_pdf(
                    str(processed_pdf), 
                    str(temp_output), 
                    dpi=dpi, 
                    quality=quality, 
                    grayscale=grayscale
                )

                if success:
                    compressed_size_mb = stats["output_size_mb"]
                    logger.info(
                        f"Compression result: {compressed_size_mb:.2f}MB ({stats['saved_percentage']:.1f}% reduction)"
                    )

                    # If compression was successful and reduced size, use the compressed file
                    if compressed_size_mb < current_size_mb:
                        # Replace the processed file with our compressed version
                        if temp_output.exists():
                            shutil.move(str(temp_output), str(processed_pdf))
                            # Update current_size_mb for next iteration
                            current_size_mb = compressed_size_mb
                            logger.info(f"Replaced file with compressed version: {current_size_mb:.2f}MB")
                        else:
                            logger.error("Compressed temp file doesn't exist!")
                    else:
                        logger.warning("Compression did not reduce file size. Keeping original.")
                        # Clean up unused temp file
                        if temp_output.exists():
                            temp_output.unlink()

                    # If we're under 45MB, we're done
                    if current_size_mb <= 45:
                        logger.info(f"File size {current_size_mb:.2f}MB is now under 45MB limit. Stopping compression.")
                        break
                else:
                    logger.warning(f"Compression failed with DPI={dpi}, quality={quality}, grayscale={grayscale}")
                    # Clean up temp file if it exists
                    if temp_output.exists():
                        temp_output.unlink()

            except Exception as e:
                logger.error(f"Compression attempt failed: {e}")
                # Clean up temp file if it exists
                if temp_output.exists():
                    temp_output.unlink()

        # Update the original variable for final check
        file_size_mb = current_size_mb

        # Check final file size
        final_size_mb = processed_pdf.stat().st_size / (1024 * 1024)
        if final_size_mb > 45:
            logger.warning(f"PDF is still {final_size_mb:.2f}MB (larger than 45MB) after compression")

    return processed_pdf


def analyze_pdf_structure(client: GeminiClient, pdf_path, book_title, config):
    """Use Gemini model to analyze the PDF structure from the full PDF."""
    prompt = f"""
    Analyze this book PDF with title "{book_title}" and provide a detailed breakdown of its structure.
    Include the following elements:
    1. Cover page (page number)
    2. Table of contents (page numbers)
    3. All chapters and subchapters as referenced in the table of contents
    4. Back cover page (page number)
    
    Important: Use the PDF page numbers (not the printed page numbers that might appear in the table of contents).
    Note that nearby chapters may overlap if there are no page breaks.
    Keep the original language for all titles.
    
    Return the result in the following JSON structure:
    {{
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
    model = config.get("model", "gemini-2.5-pro")
    
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
    return json.loads(response_text)


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
    api_key = config.get("google_api_key")
    
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
    client = GeminiClient(api_key)

    # Preprocess and get the PDF path to use
    processed_pdf = preprocess_pdf(input_pdf, output_dir)
    
    # Analyze PDF structure using Gemini
    logger.info(f"Analyzing PDF structure for '{book_title}'...")
    try:
        structure = analyze_pdf_structure(client, processed_pdf, book_title, config)
        
        # Detect front matter and back matter automatically
        structure = detect_front_and_back_matter(structure, processed_pdf)

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
