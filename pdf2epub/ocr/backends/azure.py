#!/usr/bin/env python3
"""Azure AI Document Intelligence OCR backend for Japanese text extraction."""

import os
from pathlib import Path
import yaml
import io
import numpy as np
from PIL import Image
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    DocumentContentFormat,
    DocumentAnalysisFeature
)
from azure.core.credentials import AzureKeyCredential
import base64
from typing import Dict, Tuple, Any, List, Optional
from dataclasses import dataclass
from loguru import logger

from pdf2epub.utils.logging_config import configure_logging
from ..illustration_extractor import extract_illustrations

# Configure logger
logger = configure_logging()


# Data structures for span-based furigana mapping
@dataclass
class Span:
    """Represents a character range in the content string."""
    start: int  # offset
    end: int    # offset + length


@dataclass
class RubyPair:
    """Represents a furigana-base text pair with their spans."""
    rb_spans: List[Span]  # main text (kanji) word spans
    rt_spans: List[Span]  # furigana word spans
    rt_text: str          # furigana text for convenience


@dataclass
class Edit:
    """Represents a text edit operation."""
    start: int
    end: int
    text: str  # replacement; insertion if start==end, deletion if text==""


def _extract_azure_figures(client: DocumentIntelligenceClient, azure_result, page_num: int, base_output_dir: Path = None, config: Dict = None) -> List[Dict]:
    """Extract figures using Azure's native figure detection."""
    illustrations = []
    
    if not hasattr(azure_result, 'figures') or not azure_result.figures:
        return illustrations
    
    # Set up output directory for images
    if base_output_dir:
        images_dir = base_output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
    else:
        images_dir = None
    
    # Get operation ID for downloading figure images
    operation_id = getattr(azure_result, '_operation_id', None)
    model_id = getattr(azure_result, 'model_id', 'prebuilt-layout')
    
    if not operation_id:
        logger.warning("No operation ID found for downloading figure images")
        # Even without operation ID, we can still report figure locations
        for idx, figure in enumerate(azure_result.figures):
            for region in (figure.bounding_regions or []):
                # Azure returns page 1 for single-page analysis, so we don't need to check
                illustrations.append({
                    'path': None,  # No image file available
                    'caption': figure.caption.content if hasattr(figure, 'caption') and figure.caption else None,
                    'page': page_num,
                    'figure_id': figure.id,
                    'polygon': region.polygon,  # Store bounding region
                    'placement': 'end'  # Default placement for Azure figures
                })
        return illustrations
    
    # Get page dimensions 
    page_width = None
    page_height = None
    
    # Try to get from Azure pages info first
    if hasattr(azure_result, 'pages') and azure_result.pages:
        page = azure_result.pages[0]  # Single page analysis
        if hasattr(page, 'width'):
            page_width = page.width
        if hasattr(page, 'height'):
            page_height = page.height
            
    # Fallback: estimate from all content bounding boxes
    if not page_width or not page_height:
        max_x = 0
        max_y = 0
        
        # Check all figures
        for fig in (azure_result.figures or []):
            for region in (fig.bounding_regions or []):
                if region.polygon:
                    # Flat list [x1, y1, x2, y2, ...]
                    for i in range(0, len(region.polygon), 2):
                        max_x = max(max_x, region.polygon[i])
                        if i + 1 < len(region.polygon):
                            max_y = max(max_y, region.polygon[i + 1])

        # Check all lines if available
        if hasattr(azure_result, 'pages') and azure_result.pages:
            for page in azure_result.pages:
                if hasattr(page, 'lines'):
                    for line in (page.lines or []):
                        if hasattr(line, 'polygon') and line.polygon:
                            # Flat list [x1, y1, x2, y2, ...]
                            for i in range(0, len(line.polygon), 2):
                                max_x = max(max_x, line.polygon[i])
                                if i + 1 < len(line.polygon):
                                    max_y = max(max_y, line.polygon[i + 1])

        # Use max coordinates as page size estimate (with small margin)
        if max_x > 0 and max_y > 0:
            page_width = max_x * 1.05
            page_height = max_y * 1.05
    
    for idx, figure in enumerate(azure_result.figures):
        try:
            # Extract bounding region for this page
            # Azure returns page_number as 1 for single-page analysis
            for region in (figure.bounding_regions or []):
                
                # Calculate figure dimensions from bounding polygon
                if region.polygon:
                    # Azure returns polygon as flat list of coordinates [x1, y1, x2, y2, ...]
                    xs = [region.polygon[i] for i in range(0, len(region.polygon), 2)]
                    ys = [region.polygon[i] for i in range(1, len(region.polygon), 2)]
                    
                    fig_width = max(xs) - min(xs)
                    fig_height = max(ys) - min(ys)
                    
                    # Check if figure is too small (less than 3% of page area)
                    if page_width and page_height:
                        fig_area = fig_width * fig_height
                        page_area = page_width * page_height
                        area_ratio = fig_area / page_area
                        
                        # Skip if too small (likely page number or decoration)
                        min_area_ratio = config.get('azure_min_figure_area_ratio', 0.03) if config else 0.03  # Default 3%
                        if area_ratio < min_area_ratio:
                            logger.info(f"Skipping small figure {idx} on page {page_num}: {fig_width:.0f}x{fig_height:.0f} ({area_ratio:.1%} of page)")
                            continue
                
                # Since we're analyzing one page at a time, region.page_number is always 1
                # So we should process all figures found on this single-page analysis
                
                # Download the cropped figure image
                if figure.id and images_dir:
                    figure_filename = f"page_{page_num}_figure_{idx + 1}.png"
                    figure_path = images_dir / figure_filename
                    
                    try:
                        # Download the figure using the SDK method
                        stream = client.get_analyze_result_figure(
                            model_id=model_id,
                            result_id=operation_id,
                            figure_id=figure.id
                        )
                        
                        with open(figure_path, "wb") as f:
                            for chunk in stream:
                                f.write(chunk)
                        
                        logger.success(f"Saved Azure figure to {figure_path}")
                        
                        # Add to illustrations list
                        # Use relative path for markdown (../images/ from ocr_markdown folder)
                        relative_path = f"../images/{figure_filename}"
                        illustrations.append({
                            'path': relative_path,
                            'caption': figure.caption.content if hasattr(figure, 'caption') and figure.caption else None,
                            'page': page_num,
                            'figure_id': figure.id,
                            'placement': 'end'  # Default placement for Azure figures
                        })
                    except Exception as e:
                        logger.error(f"Failed to download figure {figure.id}: {e}")
                        # Add without path for tracking
                        illustrations.append({
                            'path': None,
                            'caption': figure.caption.content if hasattr(figure, 'caption') and figure.caption else None,
                            'page': page_num,
                            'figure_id': figure.id,
                            'placement': 'end',  # Default placement for Azure figures
                            'error': str(e)
                        })
        except Exception as e:
            logger.error(f"Error processing figure {idx}: {e}")
    
    return illustrations


# Interface functions for ocr_chapters_jp.py
def init_client(config: Dict) -> DocumentIntelligenceClient:
    """Initialize Azure Document Intelligence client for use with ocr_chapters_jp.py."""
    azure_endpoint = config.get('azure_endpoint', os.getenv('AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT'))
    azure_key = config.get('azure_key', os.getenv('AZURE_DOCUMENT_INTELLIGENCE_KEY'))
    
    if not azure_endpoint or not azure_key:
        raise ValueError(
            "Azure credentials not found. Please set in config.yaml:\n"
            "  azure_endpoint: Your Azure Document Intelligence endpoint\n"
            "  azure_key: Your Azure Document Intelligence API key"
        )
    
    return DocumentIntelligenceClient(
        endpoint=azure_endpoint,
        credential=AzureKeyCredential(azure_key)
    )


def process_page(client: DocumentIntelligenceClient, img_bytes: bytes, page_num: int, config: Dict, base_output_dir: Path = None, verbose: bool = False) -> Dict:
    """
    Process a single page using Azure Document Intelligence.
    Interface function for ocr_chapters_jp.py.
    
    Args:
        base_output_dir: Base output directory (typically output/{book_title})
                        The function will create 'images' subdirectory under this
        verbose: If True, enables detailed logging from the analysis function.
    
    Returns:
        Dictionary with:
            - text: Clean markdown text with furigana
            - illustrations: List of figure data with paths
            - columns: Column classification data (optional)
            - viz_data: Data needed for visualization (for testing/debugging)
    """
    # Set up the images directory under the base output directory
    if base_output_dir:
        images_dir = base_output_dir / "images"
    else:
        images_dir = None
    
    # Check if we should use Azure's native figure extraction
    use_azure_illustrations = config.get('use_azure_illustrations', False)
    
    # Call analyze_azure_ocr directly with figure extraction if enabled
    # Use 'prebuilt-layout' if we need figures, otherwise 'prebuilt-read' for cost savings
    # Always request markdown when using layout model
    use_markdown = True  # Always use markdown for better formatting
    clean_text, azure_result, all_lines_data = analyze_azure_ocr(
        img_bytes=img_bytes,
        page_num=page_num,
        output_dir=images_dir,
        config=config,
        client=client,
        use_layout=use_azure_illustrations or use_markdown,  # Use layout for markdown
        verbose=verbose,
        extract_figures=use_azure_illustrations,
        use_markdown=use_markdown
    )
    
    # Check if Azure returned figures
    if use_azure_illustrations:
        if hasattr(azure_result, 'figures') and azure_result.figures:
            # Use Azure's native figure detection
            illustrations = _extract_azure_figures(client, azure_result, page_num, base_output_dir, config)
        else:
            illustrations = []
    else:
        # Use custom illustration extraction as fallback
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(img)
        
        illustrations = extract_illustrations(
            img_array=img_array,
            backend="azure",
            text_annotation=azure_result,  # Pass the Azure result for text region detection
            config=config,
            page_num=page_num,
            output_dir=base_output_dir if base_output_dir else None
        )
    
    return {
        'text': clean_text if clean_text is not None else "",
        'illustrations': illustrations if illustrations else [],
        'columns': {},
        'viz_data': all_lines_data if all_lines_data is not None else []
    }


def _call_azure_api(client, img_bytes, use_layout=True, extract_figures=False, use_markdown=True):
    """Calls the Azure Document Intelligence API and returns the result."""
    from azure.core.exceptions import AzureError

    logger.info("Calling Azure Document Intelligence API...")
    try:
        # Markdown only works with layout model
        model_id = "prebuilt-layout" if (use_layout or use_markdown) else "prebuilt-read"
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        # Prepare kwargs for the API call
        api_kwargs = {
            "model_id": model_id,
            "body": {"base64Source": img_base64},
            "locale": "ja-JP",
            "string_index_type": "textElements"  # Makes spans easy to slice
        }

        # Add features if using layout
        if model_id == "prebuilt-layout":
            features = [DocumentAnalysisFeature.LANGUAGES]
            # Add style font feature for bold/italic spans
            if use_markdown:
                features.append(DocumentAnalysisFeature.STYLE_FONT)
            api_kwargs["features"] = features

        # Add markdown output format if requested (requires Layout model)
        if use_markdown and model_id == "prebuilt-layout":
            api_kwargs["output_content_format"] = DocumentContentFormat.MARKDOWN  # snake_case!

        # Add figure extraction output if requested
        if extract_figures and model_id == "prebuilt-layout":
            api_kwargs["output"] = ["figures"]

        poller = client.begin_analyze_document(**api_kwargs)
        
        # Store the operation ID before getting result for figure extraction
        if extract_figures:
            try:
                # According to Azure SDK docs, operation_id should be in details
                operation_id = poller.details.get("operation_id") if hasattr(poller, 'details') else None
            except Exception as e:
                logger.warning(f"Error accessing poller details: {e}")
                operation_id = None
        
        result = poller.result()
        
        # Store the operation ID and model ID for figure extraction
        if extract_figures:
            result.model_id = model_id
            result._operation_id = operation_id
            if operation_id:
                logger.info(f"Operation ID for figures: {operation_id}")
            else:
                logger.warning("No operation ID found - figure images cannot be downloaded")
        
        logger.success("Azure Document Intelligence analysis completed.")
        return result
    except AzureError as e:
        logger.error(f"Azure API error: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during Azure analysis: {e}")
        raise


def _print_verbose_azure_summary(result, model_id, verbose=False):
    """Prints a high-level summary of the Azure Document Intelligence result."""
    if not verbose:
        return

    print("\n" + "=" * 100)
    print(f"AZURE DOCUMENT INTELLIGENCE RESULT ({model_id})")
    print("=" * 100)

    if hasattr(result, 'languages') and result.languages:
        print("\nDetected languages:")
        for lang in result.languages:
            print(f"  - {lang.locale}: {lang.confidence:.2%} confidence")

    if hasattr(result, 'content') and result.content:
        print("\nExtracted text:")
        print("-" * 100)
        print(result.content)
        print("-" * 100)


def _print_verbose_structure_analysis(result, use_layout=True, verbose=False):
    """Prints the document structure analysis (pages, tables, paragraphs)."""
    if not verbose:
        return

    pages = getattr(result, "pages", []) or []
    print("\n" + "=" * 100)
    print("DOCUMENT STRUCTURE ANALYSIS")
    print("=" * 100)

    if pages:
        for page_idx, page in enumerate(pages):
            print(f"\nPage {page_idx + 1}:")
            print(f"  Dimensions: {page.width} x {page.height} {page.unit}")
            print(f"  Angle: {page.angle if page.angle else 0}°")
            if page.lines:
                print(f"  Lines: {len(page.lines)}")
                for line_idx, line in enumerate(page.lines[:5]):
                    print(f"    Line {line_idx + 1}: {line.content[:50]}...")
            if page.words:
                print(f"  Words: {len(page.words)}")
            if page.selection_marks:
                print(f"  Selection marks: {len(page.selection_marks)}")

    if use_layout and hasattr(result, 'tables') and result.tables:
        print(f"\nTables detected: {len(result.tables)}")
        for table_idx, table in enumerate(result.tables):
            print(f"  Table {table_idx + 1}: {table.row_count} rows x {table.column_count} columns")

    if use_layout and hasattr(result, 'paragraphs') and result.paragraphs:
        print(f"\nParagraphs detected: {len(result.paragraphs)}")
        for para_idx, para in enumerate(result.paragraphs[:3]):
            role = para.role if hasattr(para, 'role') and para.role else "text"
            print(f"  Paragraph {para_idx + 1} ({role}): {para.content[:50]}...")


def _extract_line_data(result, img_height, verbose=False):
    """Extracts and processes line information from the Azure result."""
    
    def words_for_line(line, page_words):
        """Get words that belong to a line based on span overlap"""
        line_ranges = [(s.offset, s.offset + s.length) for s in (getattr(line, "spans", []) or [])]
        
        def overlaps(word):
            for s in (getattr(word, "spans", []) or [getattr(word, "span", None)]):
                if not s: 
                    continue
                ws, we = s.offset, s.offset + s.length
                for ls, le in line_ranges:
                    if ws < le and we > ls: 
                        return True
            return False
        
        words = [w for w in (page_words or []) if overlaps(w)]
        words.sort(key=lambda w: getattr(getattr(w, "spans", [getattr(w, "span", None)])[0] if (getattr(w, "spans", []) or [getattr(w, "span", None)]) else None, "offset", 10**12))
        return words

    all_lines_data = []
    horizontal_body_lines = []
    pages = getattr(result, "pages", []) or []
    
    upper_threshold = img_height * 0.15
    lower_threshold = img_height * 0.85

    if verbose:
        print("\n" + "=" * 100)
        print("LINE-BASED ANALYSIS (Vision API Logic)")
        print("=" * 100)

    for page in pages:
        if not page.lines:
            continue
            
        if verbose:
            print(f"\nProcessing {len(page.lines)} lines...")
            if page.lines:
                first_line = page.lines[0]
                print(f"Debug - First line structure:")
                print(f"  Has polygon: {hasattr(first_line, 'polygon')}")
                if hasattr(first_line, 'polygon') and first_line.polygon:
                    print(f"  Polygon length: {len(first_line.polygon)}")
                    print(f"  Polygon sample: {first_line.polygon[:min(8, len(first_line.polygon))]}")
                print(f"  Has words: {hasattr(first_line, 'words')}")
                if hasattr(first_line, 'words') and first_line.words:
                    print(f"  Word count: {len(first_line.words)}")
            
        for line_idx, line in enumerate(page.lines):
            if hasattr(line, 'polygon') and line.polygon and len(line.polygon) >= 8:
                try:
                    x_coords = [line.polygon[i] for i in range(0, len(line.polygon), 2)]
                    y_coords = [line.polygon[i] for i in range(1, len(line.polygon), 2)]
                except IndexError:
                    if verbose:
                        print(f"Line {line_idx}: polygon length = {len(line.polygon)}, polygon = {line.polygon[:8]}")
                    continue
            elif hasattr(line, 'words') and line.words:
                x_coords = []
                y_coords = []
                for word in line.words:
                    if word.polygon and len(word.polygon) >= 8:
                        x_coords.extend([word.polygon[i] for i in range(0, len(word.polygon), 2)])
                        y_coords.extend([word.polygon[i] for i in range(1, len(word.polygon), 2)])
            else:
                continue
            
            if not (x_coords and y_coords):
                continue
                
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            width, height = x_max - x_min, y_max - y_min
            
            orientation = 'VERTICAL' if height > width * 1.5 or len(line.content) <= 2 else 'HORIZONTAL'
            
            words_data = []
            line_words = words_for_line(line, getattr(page, 'words', []))
            for word in line_words:
                if hasattr(word, 'content') and hasattr(word, 'polygon') and len(word.polygon) >= 8:
                    w_x = [word.polygon[i] for i in range(0, len(word.polygon), 2)]
                    w_y = [word.polygon[i] for i in range(1, len(word.polygon), 2)]

                    # Extract span information from word
                    word_spans = []
                    if hasattr(word, 'spans') and word.spans:
                        for s in word.spans:
                            if hasattr(s, 'offset') and hasattr(s, 'length'):
                                word_spans.append(Span(start=s.offset, end=s.offset + s.length))
                    elif hasattr(word, 'span') and word.span:
                        s = word.span
                        if hasattr(s, 'offset') and hasattr(s, 'length'):
                            word_spans.append(Span(start=s.offset, end=s.offset + s.length))

                    words_data.append({
                        'text': word.content, 'y_min': min(w_y), 'y_max': max(w_y),
                        'y_center': (min(w_y) + max(w_y)) / 2, 'x_min': min(w_x),
                        'x_max': max(w_x), 'x_center': (min(w_x) + max(w_x)) / 2,
                        'spans': word_spans,  # Add span information
                        'azure_word': word    # Keep reference to original word
                    })
            
            line_data = {
                'idx': line_idx, 'text': line.content, 'x_min': x_min, 'x_max': x_max,
                'y_min': y_min, 'y_max': y_max, 'x_avg': (x_min + x_max) / 2,
                'y_avg': (y_min + y_max) / 2, 'width': width, 'height': height,
                'orientation': orientation, 'words': words_data
            }
            all_lines_data.append(line_data)

            if orientation == 'HORIZONTAL' and not (line_data['y_avg'] < upper_threshold or line_data['y_avg'] > lower_threshold):
                horizontal_body_lines.append(line_data)
                if verbose:
                    logger.info(f"  Horizontal body text at y={line_data['y_avg']:.1f}: {line.content[:50]}")

    return all_lines_data, horizontal_body_lines


def _classify_lines_by_width(all_lines_data, img_width, verbose=False):
    """Analyzes vertical line widths to classify them as main text or furigana."""
    vertical_line_widths = [
        line['width'] for line in all_lines_data
        if line['orientation'] == 'VERTICAL' and not line['text'].strip().isdigit()
    ]

    if not vertical_line_widths:
        return [], [], 30, {}

    sorted_widths = sorted(vertical_line_widths)
    num_bins = max(20, int(0.003 * img_width))
    hist, bin_edges = np.histogram(sorted_widths, bins=num_bins)
    
    # Find gaps to determine threshold
    gaps = []
    gap_start = None
    for i, count in enumerate(hist):
        if count == 0:
            if gap_start is None: 
                gap_start = bin_edges[i]
        else:
            if gap_start is not None:
                gaps.append({'start': gap_start, 'end': bin_edges[i], 'size': (bin_edges[i] - gap_start)})
                gap_start = None
    
    if gaps:
        best_gap = max(gaps, key=lambda g: g['size'])
        dynamic_threshold = (best_gap['start'] + best_gap['end']) / 2
    else:
        dynamic_threshold = np.median(sorted_widths)
    
    # Classify lines
    furigana_lines = []
    main_text_lines = []
    for line in all_lines_data:
        if line['orientation'] == 'VERTICAL' and not line['text'].strip().isdigit():
            if line['width'] < dynamic_threshold:
                line['classification'] = 'FURIGANA'
                furigana_lines.append(line)
                if verbose and ' ' in line['text']:
                    logger.info(f"  Classified as FURIGANA (has spaces): '{line['text']}' width={line['width']}, threshold={dynamic_threshold}")
            else:
                line['classification'] = 'MAIN'
                main_text_lines.append(line)
    
    # Ratio check
    if furigana_lines and main_text_lines:
        avg_furi_width = np.mean([f['width'] for f in furigana_lines])
        avg_main_width = np.mean([m['width'] for m in main_text_lines])
        if (avg_furi_width / avg_main_width) > 0.7:
            if verbose:
                logger.warning("Furigana width is >70% of main text; reclassifying all as main text.")
            for line in furigana_lines:
                line['classification'] = 'MAIN'
            main_text_lines.extend(furigana_lines)
            furigana_lines = []
            
    hist_data = {'hist': hist, 'bin_edges': bin_edges, 'min': sorted_widths[0], 'max': sorted_widths[-1]}
    return main_text_lines, furigana_lines, dynamic_threshold, hist_data


def _print_verbose_histogram_analysis(main_lines, furi_lines, threshold, hist_data, verbose=False):
    """Prints the histogram and classification statistics for furigana detection."""
    if not verbose or not hist_data:
        return

    print("\n" + "=" * 100)
    print("HISTOGRAM-BASED FURIGANA DETECTION")
    print("=" * 100)

    print(f"\nAnalyzing {len(main_lines) + len(furi_lines)} vertical text lines")
    print(f"✓ Selected threshold: {threshold:.1f}px")

    # Display text-based histogram
    hist, bin_edges = hist_data['hist'], hist_data['bin_edges']
    max_count = max(hist) if hist.any() else 1
    print("\nWidth (px)  Count  Histogram")
    print("-" * 80)
    for i in range(len(hist)):
        start, end, count = bin_edges[i], bin_edges[i + 1], hist[i]
        classification = "FURIGANA" if end < threshold else "MAIN"
        if start < threshold < end: 
            classification = "← THRESHOLD"
        bar = "█" * int(count * 50 / max_count)
        print(f"{start:5.1f}-{end:5.1f}  {count:5d}  {bar:50s} [{classification}]")

    if furi_lines and main_lines:
        furi_max = max(l['width'] for l in furi_lines)
        main_min = min(l['width'] for l in main_lines)
        if furi_max < main_min:
            print(f"\n✓ Good separation: Furigana max ({furi_max:.1f}px) < Main min ({main_min:.1f}px)")
        else:
            print(f"\n⚠ Some overlap: Furigana max ({furi_max:.1f}px) ≥ Main min ({main_min:.1f}px)")


def _group_furigana_words(furigana_lines, main_text_lines, threshold, verbose=False):
    """Groups adjacent words within furigana lines."""
    if not furigana_lines:
        return []
        
    if main_text_lines:
        main_widths = [l['width'] for l in main_text_lines if l['width'] > 0]
        median_main_width = np.median(main_widths) if main_widths else threshold
        max_gap = median_main_width * 1.0
    else:
        max_gap = threshold

    grouped_furigana_words = []
    for furi_line in furigana_lines:
        words = sorted(furi_line.get('words', []), key=lambda w: w['y_center'])
        if not words: 
            continue

        groups = []
        current_group = [words[0]]
        for i in range(1, len(words)):
            word, prev_word = words[i], current_group[-1]
            y_gap = word['y_min'] - prev_word['y_max']
            
            if verbose and furi_line['idx'] == 9:
                logger.info(f"    Gap: {prev_word['y_max']:.1f} to {word['y_min']:.1f} = {y_gap:.1f} (max_gap={max_gap:.1f})")
            
            if 0 <= y_gap <= max_gap:
                current_group.append(word)
            else:
                groups.append(current_group)
                current_group = [word]
        if current_group:
            groups.append(current_group)

        for group in groups:
            # Ensure each word in the group has span information
            # (Words come from line data which should have spans)
            grouped_furigana_words.append({
                'words': group,
                'text': ''.join(w['text'] for w in group),
                'x_avg': np.mean([w['x_center'] for w in group]),
                'y_min': min(w['y_min'] for w in group),
                'y_max': max(w['y_max'] for w in group),
                'y_avg': np.mean([w['y_center'] for w in group]),
                'line_idx': furi_line['idx']
            })
    
    if verbose:
        print(f"\nGrouped furigana words:")
        for group in grouped_furigana_words:
            print(f"  Group: '{group['text']}' from line {group['line_idx']}")
    
    return grouped_furigana_words


def _match_furigana_to_main_text(main_text_lines, grouped_furigana_words, furigana_lines=None):
    """Matches furigana groups to words in main text lines, allowing cross-line matching.

    Returns:
        tuple: (line_reconstructions dict, used_furigana_groups set, ruby_pairs list)
    """
    if not main_text_lines or not grouped_furigana_words:
        return {}, set(), []

    line_reconstructions = {}
    used_furigana_groups = set()
    ruby_pairs = []  # List of RubyPair objects for span-based editing

    # Process each main text line
    for line in main_text_lines:
        if not line.get('words'):
            continue

        matched_groups = []

        # Try to match each furigana group with words in this line
        for furi_group in grouped_furigana_words:
            if id(furi_group) in used_furigana_groups:
                continue

            # Check if furigana could match this line based on X position
            # For vertical text, furigana should be to the right of main text
            x_dist_min = furi_group['x_avg'] - line['x_max']

            # Get the furigana width from the original line it came from
            furi_width = 50  # Default fallback
            if furigana_lines:
                for furi_line in furigana_lines:
                    if furi_line['idx'] == furi_group.get('line_idx'):
                        furi_width = furi_line['width']
                        break

            # Allow some overlap or distance up to the furigana line width
            if not (-furi_width * 0.5 <= x_dist_min <= furi_width):
                continue

            # Find all words in this line that have meaningful Y overlap with the furigana group
            overlapping_words = []
            for idx, main_word in enumerate(line['words']):
                y_overlap_start = max(furi_group['y_min'], main_word['y_min'])
                y_overlap_end = min(furi_group['y_max'], main_word['y_max'])
                y_overlap = max(0, y_overlap_end - y_overlap_start)

                # Only include words with meaningful overlap (>10% of word height)
                # to avoid OCR precision issues
                word_height = main_word['y_max'] - main_word['y_min']
                if y_overlap > word_height * 0.1:
                    overlapping_words.append((idx, main_word, y_overlap))

            if not overlapping_words:
                continue

            # Sort by index to maintain reading order
            overlapping_words.sort(key=lambda x: x[0])

            # Find the minimal contiguous range of words that covers all furigana
            # Check how many furigana words have overlap with the selected range
            # Be lenient due to OCR precision issues - require 80% coverage
            furi_with_overlap = 0
            total_furi_words = len(furi_group['words'])

            for furi_word in furi_group['words']:
                has_overlap = False
                for _, main_word, _ in overlapping_words:
                    y_overlap = max(0, min(furi_word['y_max'], main_word['y_max']) -
                                    max(furi_word['y_min'], main_word['y_min']))
                    if y_overlap > 0:
                        has_overlap = True
                        break
                if has_overlap:
                    furi_with_overlap += 1

            # Require at least 80% of furigana words to have overlap (allows for OCR precision issues)
            coverage_ratio = furi_with_overlap / total_furi_words if total_furi_words > 0 else 0
            if coverage_ratio < 0.8:
                continue

            # Find contiguous sequences within overlapping words
            contiguous_groups = []
            current_group = [overlapping_words[0]]

            for i in range(1, len(overlapping_words)):
                if overlapping_words[i][0] == current_group[-1][0] + 1:
                    current_group.append(overlapping_words[i])
                else:
                    contiguous_groups.append(current_group)
                    current_group = [overlapping_words[i]]

            if current_group:
                contiguous_groups.append(current_group)

            # Use the longest contiguous group that covers the furigana
            best_group = None
            for group in contiguous_groups:
                group_y_min = min(w[1]['y_min'] for w in group)
                group_y_max = max(w[1]['y_max'] for w in group)

                # Check if this group covers most furigana characters (80% threshold)
                # Be lenient due to OCR precision issues
                furi_covered = 0
                for furi_word in furi_group['words']:
                    # Check if this furigana word has meaningful overlap with the group
                    y_overlap = max(0, min(furi_word['y_max'], group_y_max) -
                                   max(furi_word['y_min'], group_y_min))
                    furi_height = furi_word['y_max'] - furi_word['y_min']

                    # Count as covered if at least 10% overlap
                    if y_overlap >= furi_height * 0.1:
                        furi_covered += 1

                # Accept if 80% of furigana is covered (same as earlier check)
                coverage = furi_covered / len(furi_group['words']) if furi_group['words'] else 0
                if coverage >= 0.8:
                    if best_group is None or len(group) > len(best_group):
                        best_group = group

            if best_group:
                matched_main_words = [w[1] for w in best_group]

                # Create RubyPair with spans
                rb_spans = []
                for word in matched_main_words:
                    if 'spans' in word and word['spans']:
                        rb_spans.extend(word['spans'])

                rt_spans = []
                # Get spans from furigana words
                for furi_word in furi_group['words']:
                    if 'spans' in furi_word and furi_word['spans']:
                        rt_spans.extend(furi_word['spans'])

                if rb_spans and rt_spans:
                    ruby_pair = RubyPair(
                        rb_spans=rb_spans,
                        rt_spans=rt_spans,
                        rt_text=furi_group['text']
                    )
                    ruby_pairs.append(ruby_pair)

                matched_groups.append({
                    'furigana_group': furi_group,
                    'matched_main_words': matched_main_words,
                    'start_idx': best_group[0][0],
                    'end_idx': best_group[-1][0]
                })
                used_furigana_groups.add(id(furi_group))

        # Reconstruct the line with furigana annotations
        if matched_groups:
            reconstructed_parts = []
            word_idx = 0
            for group in sorted(matched_groups, key=lambda g: g['start_idx']):
                # Add text before the furigana match
                reconstructed_parts.append(''.join(w['text'] for w in line['words'][word_idx:group['start_idx']]))
                # Add the matched text with furigana
                main_text = ''.join(w['text'] for w in group['matched_main_words'])
                furigana_text = group['furigana_group']['text']
                reconstructed_parts.append(f"{main_text}({furigana_text})")
                word_idx = group['end_idx'] + 1
            # Add remaining text
            reconstructed_parts.append(''.join(w['text'] for w in line['words'][word_idx:]))
            line_reconstructions[line['idx']] = ''.join(reconstructed_parts)

    return line_reconstructions, used_furigana_groups, ruby_pairs


def _assemble_reconstructed_text(main_text_lines, horizontal_body_lines, grouped_furigana_words, 
                                 line_reconstructions, used_furigana_groups):
    """Assembles all text pieces into a final list sorted by reading order."""
    reconstructed_lines = []
    
    for line in horizontal_body_lines:
        reconstructed_lines.append({'text': line['text'], 'x': line['x_avg'], 'y': line['y_avg'], 'type': 'horizontal_body'})
    
    for line in main_text_lines:
        text = line_reconstructions.get(line['idx'], line['text'])
        line_type = 'main_with_furigana' if line['idx'] in line_reconstructions else 'main'
        reconstructed_lines.append({'text': text, 'x': line['x_avg'], 'y': line['y_avg'], 'type': line_type})
        
    for furi_group in grouped_furigana_words:
        if id(furi_group) not in used_furigana_groups:
            reconstructed_lines.append({'text': furi_group['text'], 'x': furi_group['x_avg'], 'y': furi_group['y_avg'], 'type': 'standalone_furigana'})
            
    reconstructed_lines.sort(key=lambda l: (-l['x'], l['y']))
    return reconstructed_lines


def _format_clean_output(reconstructed_lines, verbose=False):
    """Formats the final text into a clean string and optionally prints it."""
    if verbose:
        print("\n" + "=" * 100)
        print("FINAL RECONSTRUCTED TEXT (in reading order)")
        print("=" * 100)

    output_lines = []
    horizontal_elements = []
    vertical_elements = []
    
    for line in reconstructed_lines:
        if line['type'] == 'horizontal_body':
            horizontal_elements.append(line)
        else:
            vertical_elements.append(line)
    
    # Output horizontal text first
    if horizontal_elements:
        horizontal_elements.sort(key=lambda e: e['y'])
        for elem in horizontal_elements:
            output_lines.append(elem['text'])
            if verbose:
                print(f"  {elem['text']}")
    
    # Then output vertical text
    if vertical_elements:
        if horizontal_elements and output_lines:
            output_lines.append("")
        
        vertical_elements.sort(key=lambda e: (-e['x'], e['y']))
        
        # Group by approximate column
        columns = {}
        for elem in vertical_elements:
            col_key = round(elem['x'] / 100) * 100
            if col_key not in columns:
                columns[col_key] = []
            columns[col_key].append(elem)
        
        for col_x in sorted(columns.keys(), reverse=True):
            elements = columns[col_x]
            for elem in elements:
                output_lines.append(elem['text'])
                if verbose:
                    print(f"  {elem['text']}")
    
    return '\n'.join(output_lines)


# Markdown editing helper functions
def apply_edits(content: str, edits: List[Edit]) -> str:
    """Apply non-overlapping edits safely by sorting them in descending order."""
    for e in sorted(edits, key=lambda x: (x.start, x.end), reverse=True):
        content = content[:e.start] + e.text + content[e.end:]
    return content


def build_reflow_edits(result) -> List[Edit]:
    """Build edits to reflow text by removing unnecessary line breaks within paragraphs."""
    import re

    edits = []
    content = result.content if hasattr(result, 'content') else ""

    if not content:
        return edits

    # Roles to skip when reflowing
    SKIP_ROLES = {"title", "sectionHeading", "pageHeader", "pageFooter", "pageNumber", "footnote", "formulaBlock"}

    # 1) Remove headers/footers/page numbers completely
    for p in (getattr(result, "paragraphs", []) or []):
        role = getattr(p, "role", None)
        if role in {"pageHeader", "pageFooter", "pageNumber"}:
            for s in (p.spans or []):
                if hasattr(s, 'offset') and hasattr(s, 'length'):
                    edits.append(Edit(start=s.offset, end=s.offset + s.length, text=""))

    # 2) Unwrap lines inside normal paragraphs
    # Get table cell spans to avoid reflowing table content
    blocked_ranges = []
    for t in (getattr(result, "tables", []) or []):
        for cell in (getattr(t, "cells", []) or []):
            for s in (getattr(cell, "spans", []) or []):
                if hasattr(s, 'offset') and hasattr(s, 'length'):
                    blocked_ranges.append((s.offset, s.offset + s.length))

    for p in (getattr(result, "paragraphs", []) or []):
        role = getattr(p, "role", None)
        if role in SKIP_ROLES:
            continue

        for s in (p.spans or []):
            if not (hasattr(s, 'offset') and hasattr(s, 'length')):
                continue

            seg_start = s.offset
            seg_end = s.offset + s.length

            # Skip if this overlaps with table content
            if any(not (seg_end <= start or seg_start >= end) for start, end in blocked_ranges):
                continue

            seg_text = content[seg_start:seg_end]

            # Remove single newlines (not double newlines which are paragraph breaks)
            # For Japanese text, we don't want spaces between lines
            unwrapped = re.sub(r'(?<!\n)\n(?!\n)', '', seg_text)

            # Also clean up any stray double newlines within the same paragraph
            unwrapped = re.sub(r'\n{2,}', '\n', unwrapped)

            if unwrapped != seg_text:
                edits.append(Edit(start=seg_start, end=seg_end, text=unwrapped))

    return edits


def build_delete_furigana_edits(pairs: List[RubyPair]) -> List[Edit]:
    """Build edits to delete all furigana text."""
    edits = []
    for pair in pairs:
        for span in pair.rt_spans:
            edits.append(Edit(start=span.start, end=span.end, text=""))
    return edits


def build_attach_furigana_edits(pairs: List[RubyPair]) -> List[Edit]:
    """Build edits to attach furigana as (かな) after base text."""
    edits = []
    for pair in pairs:
        # Delete furigana spans
        for span in pair.rt_spans:
            edits.append(Edit(start=span.start, end=span.end, text=""))

        # Insert (furigana) after base text
        if pair.rb_spans:
            # Get the end position of the last base span
            insert_pos = max(span.end for span in pair.rb_spans)
            edits.append(Edit(start=insert_pos, end=insert_pos, text=f"({pair.rt_text})"))

    return edits


def analyze_azure_ocr(img_bytes, page_num=1, output_dir=None, config=None, client=None, use_layout=True, verbose=False, extract_figures=False, use_markdown=True) -> Tuple[str, Any, List[Dict]]:
    """
    Analyzes Azure Document Intelligence OCR with Japanese text from image bytes.
    This function orchestrates the process: API call -> Data Extraction -> Analysis -> Reconstruction.

    Args:
        use_markdown: If True, returns markdown-formatted content with furigana handling

    Returns:
        A tuple containing:
        - clean_text (str): The reconstructed text (markdown if use_markdown=True).
        - result (Any): The raw Azure result object.
        - all_lines_data (List[Dict]): The processed line data for visualization.
    """
    # 1. Setup
    if config is None:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    
    if output_dir is None:
        book_title = config.get('title', 'book')
        output_dir = Path("output") / book_title / "images"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if client is None:
        client = init_client(config)
    
    img = Image.open(io.BytesIO(img_bytes))
    logger.info(f"Processing page {page_num}: {img.width}x{img.height}px")
    
    # 2. API Call
    try:
        result = _call_azure_api(client, img_bytes, use_layout, extract_figures, use_markdown)
    except Exception:
        return None, None, []
        
    # 3. Verbose Summary (only prints if verbose=True)
    model_id = "prebuilt-layout" if use_layout else "prebuilt-read"
    _print_verbose_azure_summary(result, model_id, verbose)
    _print_verbose_structure_analysis(result, use_layout, verbose)

    # 4. Data Extraction
    all_lines_data, horizontal_body_lines = _extract_line_data(result, img.height, verbose)
    
    # 5. Furigana Analysis & Classification
    main_text_lines, furigana_lines, threshold, hist_data = _classify_lines_by_width(all_lines_data, img.width, verbose)
    _print_verbose_histogram_analysis(main_text_lines, furigana_lines, threshold, hist_data, verbose)
    
    # 6. Furigana Grouping
    grouped_furigana_words = _group_furigana_words(furigana_lines, main_text_lines, threshold, verbose)

    # 7. Furigana Matching
    line_reconstructions, used_furigana, ruby_pairs = _match_furigana_to_main_text(main_text_lines, grouped_furigana_words, furigana_lines)

    if verbose:
        logger.info(f"Created {len(ruby_pairs)} ruby pairs")
        for rp in ruby_pairs:
            logger.info(f"  Ruby pair: {rp.rt_text}")

    # 8. Check if we should use markdown with furigana handling
    if use_markdown and use_layout and hasattr(result, 'content'):
        # Use markdown content with furigana handling
        markdown_text = result.content
        edits = []

        # First: Apply furigana edits based on config
        furigana_mode = config.get('furigana_mode', 'attach') if config else 'attach'

        if ruby_pairs:  # Only apply edits if we found furigana pairs
            if furigana_mode == 'delete':
                edits.extend(build_delete_furigana_edits(ruby_pairs))
            elif furigana_mode == 'attach':
                edits.extend(build_attach_furigana_edits(ruby_pairs))
            else:
                # Default to attach mode
                edits.extend(build_attach_furigana_edits(ruby_pairs))

        # Apply furigana edits first
        if edits:
            markdown_text = apply_edits(markdown_text, edits)

        # Then apply reflow as a simple regex operation after furigana is handled
        # This avoids span confusion
        import re
        # Remove page headers/footers
        markdown_text = re.sub(r'<!-- Page(?:Header|Footer)="[^"]*" -->\n?', '', markdown_text)

        # Join lines that were incorrectly split within sentences
        # For Japanese text, remove single newlines (keep paragraph breaks)
        markdown_text = re.sub(r'(?<!\n)\n(?!\n)', '', markdown_text)

        # Final cleanup
        import re
        # Remove any remaining HTML comments (shouldn't be necessary after reflow_edits)
        markdown_text = re.sub(r'<!-- [^>]+ -->\n?', '', markdown_text)
        # Remove multiple consecutive blank lines
        markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
        # Remove leading/trailing whitespace
        markdown_text = markdown_text.strip()

        return markdown_text, result, all_lines_data
    else:
        # Fallback to original text assembly (for non-markdown or when markdown not available)
        reconstructed_lines = _assemble_reconstructed_text(main_text_lines, horizontal_body_lines,
                                                           grouped_furigana_words, line_reconstructions, used_furigana)

        # 9. Final Output Formatting
        clean_text = _format_clean_output(reconstructed_lines, verbose)

        return clean_text, result, all_lines_data
