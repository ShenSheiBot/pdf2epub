# Create new file: src/utils/visualize.py

from PIL import Image, ImageDraw
from pathlib import Path
from typing import List, Dict
from loguru import logger


def save_bounding_box_visualization(
    img_bytes: bytes,
    all_lines_data: List[Dict],
    output_path: Path,
):
    """
    Creates and saves an image with bounding boxes drawn for classified text lines.

    Args:
        img_bytes: The original image in bytes.
        all_lines_data: A list of dictionaries, where each dict represents a line
                        and must contain 'x_min', 'y_min', 'x_max', 'y_max',
                        and optionally 'classification' ('MAIN', 'FURIGANA') and
                        'orientation' ('HORIZONTAL').
        output_path: The path to save the visualization image (e.g., "viz.png").
    """
    try:
        import io
        img = Image.open(io.BytesIO(img_bytes))
        draw = ImageDraw.Draw(img)

        # Define colors for different classifications
        colors = {
            'MAIN': (255, 0, 0),        # Red
            'FURIGANA': (0, 255, 0),    # Green
            'HORIZONTAL': (0, 0, 255),  # Blue
            'UNCLASSIFIED': (128, 128, 128),  # Gray
        }
        
        logger.info(f"Generating bounding box visualization for {len(all_lines_data)} lines...")

        for line in all_lines_data:
            orientation = line.get('orientation', '')
            classification = line.get('classification', 'UNCLASSIFIED')
            
            # Determine color and line width
            if orientation == 'HORIZONTAL':
                color = colors['HORIZONTAL']
                width = 3
            elif classification == 'FURIGANA':
                color = colors['FURIGANA']
                width = 2
            elif classification == 'MAIN':
                color = colors['MAIN']
                width = 2
            else:
                color = colors['UNCLASSIFIED']
                width = 1

            # Get bounding box coordinates
            x_min = line.get('x_min')
            y_min = line.get('y_min')
            x_max = line.get('x_max')
            y_max = line.get('y_max')

            if all(v is not None for v in [x_min, y_min, x_max, y_max]):
                draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=width)
        
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        logger.success(f"Saved bounding box visualization to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")
