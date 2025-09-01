#!/usr/bin/env python3
"""Analyze Google Cloud Vision OCR with Japanese text and visualize bounding boxes."""

import sys
import os
from pathlib import Path
import yaml
import io
import numpy as np
from PIL import Image
from google.cloud import vision_v1 as vision
from google.oauth2 import service_account
from typing import Dict, Tuple, Any, List
from collections import defaultdict

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger

try:
    from .utils.logging_config import configure_logging
    from .illustration_extractor import extract_illustrations
except ImportError:
    # When running as a script
    from utils.logging_config import configure_logging

    try:
        from illustration_extractor import extract_illustrations
    except ImportError:
        # If running from src directory
        sys.path.insert(0, str(Path(__file__).parent))
        from illustration_extractor import extract_illustrations

# Configure logger
logger = configure_logging()


# Interface functions for ocr_chapters_jp.py
def init_client(config: Dict) -> vision.ImageAnnotatorClient:
    """Initialize Google Cloud Vision client for use with ocr_chapters_jp.py."""
    # Look for credentials in config, then environment variables
    key_path = config.get(
        "service_account_key_path", os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )

    if not key_path:
        raise ValueError(
            "Google Cloud credentials not found. Please set in config.yaml:\n"
            "  service_account_key_path: /path/to/your/sa-keys.json\n"
            "or set the GOOGLE_APPLICATION_CREDENTIALS environment variable."
        )

    if not Path(key_path).is_file():
        raise FileNotFoundError(f"Service account key file not found at: {key_path}")

    credentials = service_account.Credentials.from_service_account_file(key_path)
    return vision.ImageAnnotatorClient(credentials=credentials)


def process_page(
    client: vision.ImageAnnotatorClient,
    img_bytes: bytes,
    page_num: int,
    config: Dict,
    base_output_dir: Path = None,
    verbose: bool = False,
) -> Dict:
    """
    Process a single page using Google Cloud Vision.
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

    # Call analyze_vision_ocr directly
    clean_text, vision_result, all_lines_data = analyze_vision_ocr(
        img_bytes=img_bytes,
        page_num=page_num,
        output_dir=images_dir,
        config=config,
        client=client,
        verbose=verbose,
    )

    # Use custom illustration extraction
    img = Image.open(io.BytesIO(img_bytes))
    img_array = np.array(img)

    illustrations = extract_illustrations(
        img_array=img_array,
        backend="vision",
        text_annotation=vision_result,  # Pass the Vision result for text region detection
        config=config,
        page_num=page_num,
        output_dir=base_output_dir if base_output_dir else None,
    )

    # Return viz_data in the result dictionary
    return {
        "text": clean_text if clean_text is not None else "",
        "illustrations": illustrations if illustrations else [],
        "columns": {},  # Vision backend doesn't populate this yet
        "viz_data": all_lines_data if all_lines_data is not None else [],
    }


def _call_vision_api(client, img_bytes):
    """Calls the Google Cloud Vision API and returns the annotation."""
    from google.api_core.exceptions import GoogleAPICallError

    logger.info("Calling Google Cloud Vision API...")
    try:
        image = vision.Image(content=img_bytes)
        image_context = vision.ImageContext(
            language_hints=["ja"],
            text_detection_params=vision.TextDetectionParams(
                enable_text_detection_confidence_score=True
            ),
        )
        response = client.document_text_detection(
            image=image, image_context=image_context
        )
        if response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")

        logger.success("Google Cloud Vision analysis completed.")
        return response.full_text_annotation
    except GoogleAPICallError as e:
        logger.error(f"Google Vision API error: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during Vision analysis: {e}")
        raise


def _extract_line_data(
    annotation: vision.AnnotateFileResponse, img_height: int, verbose: bool = False
) -> Tuple[List[Dict], List[Dict]]:
    """Extracts and processes line information from the Vision result, preserving character data."""
    all_lines_data = []
    horizontal_body_lines = []
    line_idx_counter = 0

    upper_threshold = img_height * 0.15
    lower_threshold = img_height * 0.85

    if not annotation or not annotation.pages:
        return [], []

    def process_word_list_into_line(word_list, orientation):
        nonlocal line_idx_counter
        if not word_list:
            return None

        line_text, line_x_coords, line_y_coords, words_data = "", [], [], []

        for word in word_list:
            word_text_segment = "".join([s.text for s in word.symbols])
            line_text += word_text_segment

            w_vertices = word.bounding_box.vertices
            w_x, w_y = [v.x for v in w_vertices], [v.y for v in w_vertices]
            line_x_coords.extend(w_x)
            line_y_coords.extend(w_y)

            # --- KEY CHANGE: Extract and store character-level data ---
            char_data = []
            for symbol in word.symbols:
                s_vertices = symbol.bounding_box.vertices
                s_x = [v.x for v in s_vertices]
                s_y = [v.y for v in s_vertices]
                char_data.append(
                    {
                        "text": symbol.text,
                        "x_min": min(s_x),
                        "x_max": max(s_x),
                        "x_center": (min(s_x) + max(s_x)) / 2,
                        "y_min": min(s_y),
                        "y_max": max(s_y),
                        "y_center": (min(s_y) + max(s_y)) / 2,
                    }
                )

            words_data.append(
                {
                    "text": word_text_segment,
                    "y_min": min(w_y),
                    "y_max": max(w_y),
                    "y_center": (min(w_y) + max(w_y)) / 2,
                    "x_min": min(w_x),
                    "x_max": max(w_x),
                    "x_center": (min(w_x) + max(w_x)) / 2,
                    "chars": char_data,  # Store the character data
                }
            )

        if not line_x_coords or not line_y_coords:
            return None

        x_min, x_max = min(line_x_coords), max(line_x_coords)
        y_min, y_max = min(line_y_coords), max(line_y_coords)

        line_data = {
            "idx": line_idx_counter,
            "text": line_text,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "x_avg": (x_min + x_max) / 2,
            "y_avg": (y_min + y_max) / 2,
            "width": x_max - x_min,
            "height": y_max - y_min,
            "orientation": orientation,
            "words": words_data,
        }
        line_idx_counter += 1
        return line_data

    for page in annotation.pages:
        for block in page.blocks:
            if block.block_type != vision.Block.BlockType.TEXT:
                continue
            b_vertices = block.bounding_box.vertices
            b_x, b_y = [v.x for v in b_vertices], [v.y for v in b_vertices]
            block_width, block_height = max(b_x) - min(b_x), max(b_y) - min(b_y)
            block_orientation = (
                "HORIZONTAL" if block_width > block_height * 1.5 else "VERTICAL"
            )

            for paragraph in block.paragraphs:
                current_line_words = []
                for word in paragraph.words:
                    current_line_words.append(word)
                    has_eol = any(
                        s.property
                        and s.property.detected_break
                        and s.property.detected_break.type_ == 3
                        for s in word.symbols
                    )
                    if has_eol:
                        if line_data := process_word_list_into_line(
                            current_line_words, block_orientation
                        ):
                            all_lines_data.append(line_data)
                        current_line_words = []
                if current_line_words:
                    if line_data := process_word_list_into_line(
                        current_line_words, block_orientation
                    ):
                        all_lines_data.append(line_data)

    for line in all_lines_data:
        if line["orientation"] == "HORIZONTAL" and not (
            line["y_avg"] < upper_threshold or line["y_avg"] > lower_threshold
        ):
            horizontal_body_lines.append(line)
            if verbose:
                logger.info(
                    f"  Horizontal body text at y={line['y_avg']:.1f}: {line['text'][:50]}"
                )

    return all_lines_data, horizontal_body_lines


def _is_kanji(char):
    """Check if a character is a CJK Unified Ideograph (a Kanji)."""
    return "\u4e00" <= char <= "\u9fff"


def _classify_lines_by_width(all_lines_data, img_width, verbose=False):
    """Analyzes vertical line widths to classify them as main text or furigana."""
    vertical_line_widths = [
        line["width"]
        for line in all_lines_data
        if line["orientation"] == "VERTICAL" and not line["text"].strip().isdigit()
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
                gaps.append(
                    {
                        "start": gap_start,
                        "end": bin_edges[i],
                        "size": (bin_edges[i] - gap_start),
                    }
                )
                gap_start = None

    if gaps:
        best_gap = max(gaps, key=lambda g: g["size"])
        dynamic_threshold = (best_gap["start"] + best_gap["end"]) / 2
    else:
        dynamic_threshold = np.median(sorted_widths)

    # Classify lines
    furigana_lines = []
    main_text_lines = []
    for line in all_lines_data:
        if line["orientation"] == "VERTICAL" and not line["text"].strip().isdigit():
            if line["width"] < dynamic_threshold:
                line["classification"] = "FURIGANA"
                furigana_lines.append(line)
                if verbose and " " in line["text"]:
                    logger.info(
                        f"  Classified as FURIGANA (has spaces): '{line['text']}' width={line['width']}, threshold={dynamic_threshold}"
                    )
            else:
                line["classification"] = "MAIN"
                main_text_lines.append(line)

    # Ratio check
    if furigana_lines and main_text_lines:
        avg_furi_width = np.mean([f["width"] for f in furigana_lines])
        avg_main_width = np.mean([m["width"] for m in main_text_lines])
        if (avg_furi_width / avg_main_width) > 0.7:
            if verbose:
                logger.warning(
                    "Furigana width is >70% of main text; reclassifying all as main text."
                )
            for line in furigana_lines:
                line["classification"] = "MAIN"
            main_text_lines.extend(furigana_lines)
            furigana_lines = []

    hist_data = {
        "hist": hist,
        "bin_edges": bin_edges,
        "min": sorted_widths[0] if sorted_widths else 0,
        "max": sorted_widths[-1] if sorted_widths else 0,
    }
    return main_text_lines, furigana_lines, dynamic_threshold, hist_data


def _print_verbose_histogram_analysis(
    main_lines, furi_lines, threshold, hist_data, verbose=False
):
    """Prints the histogram and classification statistics for furigana detection."""
    if not verbose or not hist_data:
        return

    print("\n" + "=" * 100)
    print("HISTOGRAM-BASED FURIGANA DETECTION")
    print("=" * 100)

    print(f"\nAnalyzing {len(main_lines) + len(furi_lines)} vertical text lines")
    print(f"✓ Selected threshold: {threshold:.1f}px")

    # Display text-based histogram
    hist, bin_edges = hist_data["hist"], hist_data["bin_edges"]
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
        furi_max = max(l["width"] for l in furi_lines)
        main_min = min(l["width"] for l in main_lines)
        if furi_max < main_min:
            print(
                f"\n✓ Good separation: Furigana max ({furi_max:.1f}px) < Main min ({main_min:.1f}px)"
            )
        else:
            print(
                f"\n⚠ Some overlap: Furigana max ({furi_max:.1f}px) ≥ Main min ({main_min:.1f}px)"
            )


def _group_furigana_words(furigana_lines, main_text_lines, threshold, verbose=False):
    """Groups adjacent words within furigana lines."""
    if not furigana_lines:
        return []

    if main_text_lines:
        main_widths = [l["width"] for l in main_text_lines if l["width"] > 0]
        median_main_width = np.median(main_widths) if main_widths else threshold
        max_gap = median_main_width * 1.0
    else:
        max_gap = threshold

    grouped_furigana_words = []
    for furi_line in furigana_lines:
        words = sorted(furi_line.get("words", []), key=lambda w: w["y_center"])
        if not words:
            continue

        groups = []
        current_group = [words[0]]
        for i in range(1, len(words)):
            word, prev_word = words[i], current_group[-1]
            y_gap = word["y_min"] - prev_word["y_max"]

            if verbose:
                logger.info(
                    f"    Gap check line {furi_line['idx']}: {prev_word['y_max']:.1f} to {word['y_min']:.1f} = {y_gap:.1f} (max_gap={max_gap:.1f})"
                )

            if 0 <= y_gap <= max_gap:
                current_group.append(word)
            else:
                groups.append(current_group)
                current_group = [word]
        if current_group:
            groups.append(current_group)

        for group in groups:
            grouped_furigana_words.append(
                {
                    "words": group,
                    "text": "".join(w["text"] for w in group),
                    "x_avg": np.mean([w["x_center"] for w in group]),
                    "y_min": min(w["y_min"] for w in group),
                    "y_max": max(w["y_max"] for w in group),
                    "y_avg": np.mean([w["y_center"] for w in group]),
                    "line_idx": furi_line["idx"],
                }
            )

    if verbose:
        print(f"\nGrouped furigana words:")
        for group in grouped_furigana_words:
            print(f"  Group: '{group['text']}' from line {group['line_idx']}")

    return grouped_furigana_words


def _match_furigana_to_main_text(
    main_text_lines, grouped_furigana_words, verbose=False
):
    """
    Matches furigana groups to spans of characters (kanji or otherwise) within main text words.
    """
    if not main_text_lines or not grouped_furigana_words:
        return {}, set()

    main_widths = [l["width"] for l in main_text_lines if l["width"] > 0]
    median_main_width = np.median(main_widths) if main_widths else 50
    max_furigana_distance = median_main_width * 1.5

    # This dictionary stores furigana for character SPANS.
    # Format: { line_idx: { word_idx: { (start_char_idx, end_char_idx): "furigana_text" } } }
    span_furigana_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    used_furigana_groups = set()

    # Step 1: For each furigana group, find its best parent WORD based on Y-overlap.
    for furi_group in grouped_furigana_words:
        best_match = {"max_overlap": 0, "line": None, "word_idx": -1}
        
        # Debug: Check what words are near this furigana
        if verbose and "こんしん" in furi_group["text"]:
            logger.info(f"  DEBUG: Looking for parent word for furigana 'こんしん' at x={furi_group['x_avg']:.1f}, y_range=[{furi_group['y_min']:.1f}, {furi_group['y_max']:.1f}]")
        
        for line in main_text_lines:
            for word_idx, word in enumerate(line["words"]):
                x_dist = furi_group["x_avg"] - word["x_center"]
                
                # Furigana should be to the right of the main text
                if not (0 < x_dist < max_furigana_distance):
                    continue

                # Calculate Y-axis overlap
                y_overlap = max(
                    0,
                    min(furi_group["y_max"], word["y_max"])
                    - max(furi_group["y_min"], word["y_min"]),
                )
                
                # Debug: Show words with y-overlap
                if verbose and "こんしん" in furi_group["text"] and y_overlap > 0:
                    logger.info(f"    Word '{word['text']}' has y_overlap={y_overlap:.1f} (y_range=[{word['y_min']:.1f}, {word['y_max']:.1f}])")
                
                # Choose the word with maximum Y-overlap
                if y_overlap > best_match["max_overlap"]:
                    best_match.update(
                        {"max_overlap": y_overlap, "line": line, "word_idx": word_idx}
                    )

        # Step 2: If a parent word is found, identify the SPAN of characters it covers.
        if best_match["line"]:
            line = best_match["line"]
            word_idx = best_match["word_idx"]
            word = line["words"][word_idx]
            
            # Debug logging for specific cases
            if verbose and "こんしん" in furi_group["text"]:
                logger.info(f"  DEBUG: Furigana 'こんしん' matched to word '{word['text']}' "
                          f"at word_idx={word_idx}, y_overlap={best_match['max_overlap']:.1f}")

            # --- KEY CHANGE: Fallback to ALL characters if no kanji are present ---
            has_kanji = any(_is_kanji(c["text"]) for c in word["chars"])
            target_chars = (
                [c for c in word["chars"] if _is_kanji(c["text"])]
                if has_kanji
                else word["chars"]
            )

            # Find all target characters that vertically overlap with the furigana
            # AND are positioned to the left of the furigana (furigana should be to the right)
            overlapping_chars = []
            for i, c in enumerate(word["chars"]):
                if c not in target_chars:
                    continue
                    
                # Check vertical overlap
                y_overlap = max(
                    0,
                    min(furi_group["y_max"], c["y_max"])
                    - max(furi_group["y_min"], c["y_min"]),
                )
                if y_overlap <= 0:
                    continue
                    
                # Check that furigana is to the right of this character
                # (furigana x should be greater than character x)
                char_x_center = (c.get("x_min", 0) + c.get("x_max", 0)) / 2 if "x_min" in c else c.get("x_center", 0)
                x_dist_to_char = furi_group["x_avg"] - char_x_center
                
                # Debug logging for specific cases
                if verbose and "渾" in word["text"] and "こんしん" in furi_group["text"]:
                    logger.info(f"    DEBUG: Matching furigana 'こんしん' to char '{c['text']}': "
                              f"x_dist={x_dist_to_char:.1f}, y_overlap={y_overlap:.1f}, "
                              f"char_x={char_x_center:.1f}, furi_x={furi_group['x_avg']:.1f}")
                
                # Furigana should be to the right (positive x_dist) and within reasonable distance
                if 0 < x_dist_to_char < max_furigana_distance:
                    overlapping_chars.append((i, c))

            if overlapping_chars:
                # The span is the min and max index of the overlapping characters
                start_char_idx = min(i for i, c in overlapping_chars)
                end_char_idx = max(i for i, c in overlapping_chars)
                
                # Debug logging for specific cases
                if verbose and "こんしん" in furi_group["text"]:
                    char_text = "".join(c["text"] for i, c in overlapping_chars)
                    logger.info(f"    DEBUG: Furigana 'こんしん' overlaps with chars '{char_text}' "
                              f"at indices {start_char_idx}-{end_char_idx} in word '{word['text']}'")

                # Store the furigana group to be processed later
                span_furigana_map[line["idx"]][word_idx][
                    (start_char_idx, end_char_idx)
                ].append(furi_group)
                used_furigana_groups.add(id(furi_group))

    # Step 3: Process the map to merge furigana text and reconstruct lines.
    line_reconstructions = {}
    for line in main_text_lines:
        reconstructed_words = []
        for word_idx, word in enumerate(line["words"]):
            if word_idx not in span_furigana_map[line["idx"]]:
                reconstructed_words.append(word["text"])
                continue

            # --- KEY CHANGE: Reconstruct word using character spans ---
            word_spans = span_furigana_map[line["idx"]][word_idx]

            # Combine furigana text for each span, sorted by vertical position
            final_span_map = {}
            for span, furi_groups in word_spans.items():
                furi_groups.sort(key=lambda g: g["y_avg"])
                final_span_map[span] = "".join(g["text"] for g in furi_groups)

            sorted_spans = sorted(final_span_map.keys())
            
            # Debug logging for specific cases
            if verbose and "渾身" in word["text"]:
                logger.info(f"    DEBUG: Reconstructing word '{word['text']}' with spans: {sorted_spans}")
                for idx, c in enumerate(word["chars"]):
                    logger.info(f"      char[{idx}]: '{c['text']}'")

            reconstructed_word = ""
            current_char_idx = 0
            for start, end in sorted_spans:
                # Add characters before the current span
                reconstructed_word += "".join(
                    c["text"] for c in word["chars"][current_char_idx:start]
                )

                # Add the characters within the span and their combined furigana
                span_text = "".join(c["text"] for c in word["chars"][start : end + 1])
                furigana_text = final_span_map[(start, end)]
                reconstructed_word += f"{span_text}({furigana_text})"

                current_char_idx = end + 1

            # Add any remaining characters after the last span
            reconstructed_word += "".join(
                c["text"] for c in word["chars"][current_char_idx:]
            )
            
            # Debug logging for specific cases
            if verbose and "渾身" in word["text"]:
                logger.info(f"    DEBUG: Reconstructed word with furigana: '{reconstructed_word}' "
                          f"(original: '{word['text']}')")
            
            reconstructed_words.append(reconstructed_word)

        line_reconstructions[line["idx"]] = "".join(reconstructed_words)

    return line_reconstructions, used_furigana_groups


def _assemble_reconstructed_text(
    main_text_lines,
    horizontal_body_lines,
    grouped_furigana_words,
    line_reconstructions,
    used_furigana_groups,
):
    """Assembles all text pieces into a final list sorted by reading order."""
    reconstructed_lines = []

    for line in horizontal_body_lines:
        reconstructed_lines.append(
            {
                "text": line["text"],
                "x": line["x_avg"],
                "y": line["y_avg"],
                "type": "horizontal_body",
            }
        )

    for line in main_text_lines:
        text = line_reconstructions.get(line["idx"], line["text"])
        line_type = (
            "main_with_furigana" if line["idx"] in line_reconstructions else "main"
        )
        reconstructed_lines.append(
            {"text": text, "x": line["x_avg"], "y": line["y_avg"], "type": line_type}
        )

    for furi_group in grouped_furigana_words:
        if id(furi_group) not in used_furigana_groups:
            reconstructed_lines.append(
                {
                    "text": furi_group["text"],
                    "x": furi_group["x_avg"],
                    "y": furi_group["y_avg"],
                    "type": "standalone_furigana",
                }
            )

    reconstructed_lines.sort(key=lambda l: (-l["x"], l["y"]))
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
        if line["type"] == "horizontal_body":
            horizontal_elements.append(line)
        else:
            vertical_elements.append(line)

    # Output horizontal text first
    if horizontal_elements:
        horizontal_elements.sort(key=lambda e: e["y"])
        for elem in horizontal_elements:
            output_lines.append(elem["text"])
            if verbose:
                print(f"  {elem['text']}")

    # Then output vertical text
    if vertical_elements:
        if horizontal_elements and output_lines:
            output_lines.append("")

        vertical_elements.sort(key=lambda e: (-e["x"], e["y"]))

        # Group by approximate column
        columns = {}
        for elem in vertical_elements:
            col_key = round(elem["x"] / 100) * 100
            if col_key not in columns:
                columns[col_key] = []
            columns[col_key].append(elem)

        for col_x in sorted(columns.keys(), reverse=True):
            elements = columns[col_x]
            for elem in elements:
                output_lines.append(elem["text"])
                if verbose:
                    print(f"  {elem['text']}")

    return "\n".join(output_lines)


def analyze_vision_ocr(
    img_bytes, page_num=1, output_dir=None, config=None, client=None, verbose=False
) -> Tuple[str, Any, List[Dict]]:
    """
    Analyzes Google Vision OCR with Japanese text from image bytes.
    This function orchestrates the process: API call -> Data Extraction -> Analysis -> Reconstruction.

    Returns:
        A tuple containing:
        - clean_text (str): The reconstructed text.
        - result (Any): The raw Vision result object.
        - all_lines_data (List[Dict]): The processed line data for visualization.
    """
    # 1. Setup
    if config is None:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

    if output_dir is None:
        book_title = config.get("title", "book")
        output_dir = Path("output") / book_title / "images"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if client is None:
        client = init_client(config)

    img = Image.open(io.BytesIO(img_bytes))
    logger.info(f"Processing page {page_num}: {img.width}x{img.height}px")

    # 2. API Call
    try:
        result = _call_vision_api(client, img_bytes)
    except Exception:
        return None, None, []

    # 3. Data Extraction
    all_lines_data, horizontal_body_lines = _extract_line_data(
        result, img.height, verbose
    )

    # 4. Furigana Analysis & Classification
    main_text_lines, furigana_lines, threshold, hist_data = _classify_lines_by_width(
        all_lines_data, img.width, verbose
    )
    _print_verbose_histogram_analysis(
        main_text_lines, furigana_lines, threshold, hist_data, verbose
    )

    # 5. Furigana Grouping
    grouped_furigana_words = _group_furigana_words(
        furigana_lines, main_text_lines, threshold, verbose
    )

    # 6. Furigana Matching
    line_reconstructions, used_furigana = _match_furigana_to_main_text(
        main_text_lines, grouped_furigana_words, verbose
    )

    # 7. Text Assembly
    reconstructed_lines = _assemble_reconstructed_text(
        main_text_lines,
        horizontal_body_lines,
        grouped_furigana_words,
        line_reconstructions,
        used_furigana,
    )

    # 8. Final Output Formatting
    clean_text = _format_clean_output(reconstructed_lines, verbose)

    # Return all_lines_data for visualization
    return clean_text, result, all_lines_data
