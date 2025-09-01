#!/usr/bin/env python3
"""Module for extracting illustrations from pages by detecting non-text regions."""

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from loguru import logger
from typing import List, Tuple, Dict, Any, Optional


def inject_illustrations_into_text(text: str, illustrations: List[Dict]) -> str:
    """Inject illustration references into text based on their placement."""
    if not illustrations:
        return text

    lines = text.split("\n")
    result_lines = []

    # Group illustrations by placement
    above_illustrations = [ill for ill in illustrations if ill["placement"] == "above"]
    below_illustrations = [ill for ill in illustrations if ill["placement"] == "below"]
    between_illustrations = [
        ill for ill in illustrations if ill["placement"] == "between"
    ]
    end_illustrations = [ill for ill in illustrations if ill["placement"] == "end"]

    # Add above illustrations at the beginning
    for ill in above_illustrations:
        # Use the 'path' key
        result_lines.append(f"![Image]({ill['path']})")
        result_lines.append("")  # Add blank line

    # Add text lines
    for i, line in enumerate(lines):
        result_lines.append(line)

        # Add between illustrations at logical breaks (empty lines or between columns)
        if i < len(lines) - 1 and line.strip() == "" and between_illustrations:
            # Insert one between illustration at empty lines
            ill = between_illustrations.pop(0)
            # Use the 'path' key
            result_lines.append(f"![Image]({ill['path']})")
            result_lines.append("")

    # Add below illustrations before the end
    for ill in below_illustrations:
        result_lines.append("")
        # Use the 'path' key
        result_lines.append(f"![Image]({ill['path']})")

    # Add end illustrations at the very end
    for ill in end_illustrations:
        result_lines.append("")
        # Use the 'path' key
        result_lines.append(f"![Image]({ill['path']})")

    return "\n".join(result_lines)


def white_out_text_regions_vision(
    img_array: np.ndarray, annotation: Any, padding: int = 25
) -> np.ndarray:
    """
    White out text regions for Vision API annotations.

    Args:
        img_array: Original image as numpy array
        annotation: Google Cloud Vision API annotation object
        padding: Padding around text regions in pixels

    Returns:
        Image array with text regions whited out
    """
    img_copy = img_array.copy()
    img_pil = Image.fromarray(img_copy)
    draw = ImageDraw.Draw(img_pil)

    for page in annotation.pages:
        for block in page.blocks:
            if block.block_type == 1:  # TEXT block
                if block.bounding_box and block.bounding_box.vertices:
                    vertices = block.bounding_box.vertices
                    x_coords = [v.x if hasattr(v, "x") else 0 for v in vertices]
                    y_coords = [v.y if hasattr(v, "y") else 0 for v in vertices]

                    if x_coords and y_coords:
                        x_min = max(0, min(x_coords) - padding)
                        y_min = max(0, min(y_coords) - padding)
                        x_max = min(img_array.shape[1] - 1, max(x_coords) + padding)
                        y_max = min(img_array.shape[0] - 1, max(y_coords) + padding)

                        draw.rectangle([(x_min, y_min), (x_max, y_max)], fill="white")

    return np.array(img_pil)


def white_out_text_regions_azure(
    img_array: np.ndarray, pages: List[Any], padding: int = 25
) -> np.ndarray:
    """
    White out text regions for Azure Document Intelligence annotations.

    Args:
        img_array: Original image as numpy array
        pages: List of Azure page objects with lines
        padding: Padding around text regions in pixels

    Returns:
        Image array with text regions whited out
    """
    img_copy = img_array.copy()
    img_pil = Image.fromarray(img_copy)
    draw = ImageDraw.Draw(img_pil)

    for page in pages:
        if hasattr(page, "lines") and page.lines:
            for line in page.lines:
                if hasattr(line, "polygon") and line.polygon and len(line.polygon) >= 8:
                    # Azure polygon format: [x1, y1, x2, y2, x3, y3, x4, y4, ...]
                    x_coords = [line.polygon[i] for i in range(0, len(line.polygon), 2)]
                    y_coords = [line.polygon[i] for i in range(1, len(line.polygon), 2)]

                    if x_coords and y_coords:
                        x_min = max(0, min(x_coords) - padding)
                        y_min = max(0, min(y_coords) - padding)
                        x_max = min(img_array.shape[1] - 1, max(x_coords) + padding)
                        y_max = min(img_array.shape[0] - 1, max(y_coords) + padding)

                        draw.rectangle([(x_min, y_min), (x_max, y_max)], fill="white")

    return np.array(img_pil)


def trim_white_borders(
    img_array: np.ndarray, step_percent: float = 0.005, min_black_pixels: int = 200
) -> Tuple[int, int, int, int]:
    """
    Progressively trim white borders to find core illustration area.

    Args:
        img_array: Image array (whitened)
        step_percent: Step size as percentage of image dimension
        min_black_pixels: Minimum number of non-white pixels to consider as content

    Returns:
        Tuple of (top, bottom, left, right) bounds
    """
    h, w = img_array.shape[:2]

    # Convert to grayscale for analysis
    if len(img_array.shape) == 3:
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        gray = img_array

    # Start with full image bounds
    top, bottom, left, right = 0, h - 1, 0, w - 1

    # Progressive trimming from each edge
    step_h = max(1, int(h * step_percent))
    step_w = max(1, int(w * step_percent))

    # Trim from top
    for y in range(0, h // 2, step_h):
        stripe = gray[y : y + step_h, :]
        if np.sum(stripe < 240) > min_black_pixels:
            top = y
            break

    # Trim from bottom
    for y in range(h - 1, h // 2, -step_h):
        stripe = gray[y - step_h : y, :]
        if np.sum(stripe < 240) > min_black_pixels:
            bottom = y
            break

    # Trim from left
    for x in range(0, w // 2, step_w):
        stripe = gray[:, x : x + step_w]
        if np.sum(stripe < 240) > min_black_pixels:
            left = x
            break

    # Trim from right
    for x in range(w - 1, w // 2, -step_w):
        stripe = gray[:, x - step_w : x]
        if np.sum(stripe < 240) > min_black_pixels:
            right = x
            break

    return top, bottom, left, right


def determine_placement(
    ill_x_center: float, ill_width: float, vertical_columns: List[Dict], img_width: int
) -> str:
    """
    Determine where to place the illustration in the markdown based on text column positions.

    Args:
        ill_x_center: X-coordinate of illustration center
        ill_width: Width of illustration
        vertical_columns: List of text column dictionaries with x_min, x_max, type
        img_width: Width of the full image

    Returns:
        Placement string: "above", "below", "between", or "end"
    """
    main_columns_x = []
    for col in vertical_columns:
        if col.get("type") == "MAIN":
            main_columns_x.append(col["x_min"])
            main_columns_x.append(col["x_max"])

    placement = "end"  # Default placement

    if main_columns_x:
        min_text_x = min(main_columns_x)
        max_text_x = max(main_columns_x)

        # Rule 1: Check for large gaps between column groups
        column_groups = {}
        for col in vertical_columns:
            if col.get("type") == "MAIN":
                group_key = round(col["x_min"] / 100) * 100
                if group_key not in column_groups:
                    column_groups[group_key] = []
                column_groups[group_key].append(col)

        if len(column_groups) >= 2:
            sorted_groups = sorted(column_groups.keys())
            for i in range(len(sorted_groups) - 1):
                gap_start = max(
                    [col["x_max"] for col in column_groups[sorted_groups[i]]]
                )
                gap_end = min(
                    [col["x_min"] for col in column_groups[sorted_groups[i + 1]]]
                )
                gap = gap_end - gap_start

                if gap > ill_width + 50:  # 50px margin
                    placement = "between"
                    break
            else:
                # Check rules 2 and 3
                if max_text_x < img_width * 0.5:
                    placement = "above"
                elif min_text_x > img_width * 0.5:
                    placement = "below"
        else:
            # Single column group
            if max_text_x < img_width * 0.5:
                placement = "above"
            elif min_text_x > img_width * 0.5:
                placement = "below"

    return placement


def extract_vertical_columns_from_azure(result: Any) -> List[Dict]:
    """
    Extract vertical column information from Azure result.

    Args:
        result: Azure Document Intelligence result object

    Returns:
        List of column dictionaries with x_min, x_max, y_min, y_max, type
    """
    vertical_columns = []
    pages = getattr(result, "pages", []) or []

    if pages:
        for page in pages:
            if hasattr(page, "lines") and page.lines:
                for line in page.lines:
                    if (
                        hasattr(line, "polygon")
                        and line.polygon
                        and len(line.polygon) >= 8
                    ):
                        x_coords = [
                            line.polygon[i] for i in range(0, len(line.polygon), 2)
                        ]
                        y_coords = [
                            line.polygon[i] for i in range(1, len(line.polygon), 2)
                        ]
                        if x_coords and y_coords:
                            # Determine if line is vertical based on aspect ratio
                            width = max(x_coords) - min(x_coords)
                            height = max(y_coords) - min(y_coords)
                            if height > width * 1.5:  # Likely vertical text
                                vertical_columns.append(
                                    {
                                        "x_min": min(x_coords),
                                        "x_max": max(x_coords),
                                        "y_min": min(y_coords),
                                        "y_max": max(y_coords),
                                        "type": "MAIN",
                                    }
                                )

    return vertical_columns


def extract_vertical_columns_from_vision(annotation: Any) -> List[Dict]:
    """
    Extract vertical column information from Vision API annotation.

    Args:
        annotation: Google Cloud Vision API annotation object

    Returns:
        List of column dictionaries with x_min, x_max, y_min, y_max, type
    """
    vertical_columns = []

    if annotation and annotation.pages:
        for page in annotation.pages:
            for block in page.blocks:
                if block.block_type == 1:  # TEXT block
                    if block.bounding_box and block.bounding_box.vertices:
                        vertices = block.bounding_box.vertices
                        x_coords = [v.x if hasattr(v, "x") else 0 for v in vertices]
                        y_coords = [v.y if hasattr(v, "y") else 0 for v in vertices]

                        if x_coords and y_coords:
                            # Determine if block is vertical based on aspect ratio
                            width = max(x_coords) - min(x_coords)
                            height = max(y_coords) - min(y_coords)
                            if height > width * 1.5:  # Likely vertical text
                                vertical_columns.append(
                                    {
                                        "x_min": min(x_coords),
                                        "x_max": max(x_coords),
                                        "y_min": min(y_coords),
                                        "y_max": max(y_coords),
                                        "type": "MAIN",
                                    }
                                )

    return vertical_columns


def extract_illustrations(
    img_array: np.ndarray,
    backend: str,
    text_annotation: Any,
    config: Dict,
    page_num: int,
    output_dir: Path,
    chapter_num: Optional[int] = None,
) -> List[Dict]:
    """
    Extract illustrations from the page by detecting non-text regions.

    Args:
        img_array: Original image as numpy array
        backend: "azure" or "vision" to determine how to process text regions
        text_annotation: Text annotation object (Azure result or Vision annotation)
        config: Configuration dictionary
        page_num: Page number (1-indexed)
        output_dir: Base output directory (typically output/{book_title})
        chapter_num: Optional chapter number for naming

    Returns:
        List of illustration dictionaries with path, bbox, and placement
    """
    settings = config.get("vision_ocr_settings", {})
    padding = settings.get("illustration_padding", 25)
    trim_step = settings.get("trim_step_percent", 0.005)
    min_black = settings.get("min_black_pixels", 200)

    # Extract vertical columns based on backend
    if backend == "azure":
        vertical_columns = extract_vertical_columns_from_azure(text_annotation)
        pages = getattr(text_annotation, "pages", []) or []

        # White out text regions
        if not pages:
            logger.debug(
                f"Page {page_num}: No text detected, using original image for illustration detection"
            )
            whitened_img = img_array
        else:
            whitened_img = white_out_text_regions_azure(img_array, pages, padding)
    elif backend == "vision":
        vertical_columns = extract_vertical_columns_from_vision(text_annotation)

        # White out text regions
        if not text_annotation or not text_annotation.pages:
            logger.debug(
                f"Page {page_num}: No text detected, using original image for illustration detection"
            )
            whitened_img = img_array
        else:
            whitened_img = white_out_text_regions_vision(
                img_array, text_annotation, padding
            )
    elif backend == "vllm":
        # VLLM detects illustrations and marks them in the text output
        # text_annotation here is actually the OCR text output string
        logger.debug(
            f"Page {page_num}: VLLM backend - checking for [illustration] markers in text"
        )
        
        # Check if the text contains [illustration] marker
        if text_annotation and "[illustration]" in text_annotation.lower():
            # Save the entire page as an illustration
            logger.info(f"Page {page_num}: VLLM detected illustration on page")
            
            # Save the full page image
            img_pil = Image.fromarray(img_array)
            images_dir = output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            
            if chapter_num is not None:
                filename = f"ch{chapter_num:03d}_p{page_num:03d}_illustration.png"
            else:
                filename = f"p{page_num:03d}_illustration.png"
            
            img_path = images_dir / filename
            img_pil.save(img_path, "PNG", optimize=True)
            
            return [{
                "path": f"../images/{filename}",
                "page": page_num,
                "placement": "full_page",
                "bounds": [0, 0, img_array.shape[1], img_array.shape[0]]
            }]
        else:
            return []
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Check if there's significant non-white content remaining
    gray = np.dot(whitened_img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    non_white_pixels = np.sum(gray < 240)
    total_pixels = gray.shape[0] * gray.shape[1]
    non_white_ratio = non_white_pixels / total_pixels

    illustrations = []

    if non_white_ratio > 0.01:  # More than 1% non-white suggests illustration
        logger.debug(
            f"Page {page_num}: Detected {non_white_ratio * 100:.2f}% non-white pixels, checking for illustration"
        )

        # Trim white borders to find illustration bounds
        top, bottom, left, right = trim_white_borders(
            whitened_img, trim_step, min_black
        )

        # Validate the bounds
        if right > left and bottom > top:
            # Determine placement
            placement = determine_placement(
                (left + right) / 2, right - left, vertical_columns, img_array.shape[1]
            )

            # Add padding to the illustration bounds
            ill_padding = 10
            crop_x_min = max(0, left - ill_padding)
            crop_y_min = max(0, top - ill_padding)
            crop_x_max = min(img_array.shape[1], right + ill_padding)
            crop_y_max = min(img_array.shape[0], bottom + ill_padding)

            # Extract from ORIGINAL image
            illustration = img_array[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

            # Save illustration if output_dir is provided
            if output_dir:
                images_dir = output_dir / "images"
                images_dir.mkdir(parents=True, exist_ok=True)

                # Create filename
                if chapter_num is not None:
                    illust_path = (
                        images_dir / f"chapter_{chapter_num}_page_{page_num}.png"
                    )
                else:
                    illust_path = images_dir / f"page_{page_num}_illustration.png"

                Image.fromarray(illustration).save(illust_path)
                illust_filename = illust_path.name  # Get the filename from the path

                # Use standardized keys: 'path' and 'bbox'
                illustrations.append(
                    {
                        "path": f"../images/{illust_filename}",
                        "bbox": (
                            left,
                            top,
                            right,
                            bottom,
                        ),  # Standard (x_min, y_min, x_max, y_max) format
                        "placement": placement,
                    }
                )

                logger.info(
                    f"Page {page_num}: Found illustration at bbox: ({left}, {top}, {right}, {bottom}), placement: {placement}"
                )
            else:
                logger.warning(
                    f"Page {page_num}: Found illustration but output_dir was not provided. Cannot save."
                )

    return illustrations
