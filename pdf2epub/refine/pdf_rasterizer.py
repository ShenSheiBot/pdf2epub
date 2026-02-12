"""
PDF rasterization utilities for API compatibility.

When Gemini API rejects certain PDF structures (503 error),
this module provides JBIG2 rasterization as fallback.
"""

import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm


JBIG2_DPI_LEVELS = [150, 120]


def _otsu_threshold(img: Image.Image) -> int:
    """
    Calculate optimal binarization threshold using Otsu's method.

    Args:
        img: Grayscale PIL Image

    Returns:
        Optimal threshold value (0-255)
    """
    arr = np.array(img)
    hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 256))
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)

    sum_bg, weight_bg, max_var, threshold = 0, 0, 0, 128  # default 128
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t

    return threshold


def _binarize_image(img: Image.Image) -> Image.Image:
    """
    Convert grayscale image to binary using Otsu's adaptive threshold.

    Args:
        img: Grayscale PIL Image

    Returns:
        Binary (1-bit) PIL Image
    """
    threshold = _otsu_threshold(img)
    return img.point(lambda x: 255 if x > threshold else 0, '1')


def check_jbig2_available() -> bool:
    """Check if jbig2 command is available."""
    import shutil
    return shutil.which('jbig2') is not None


def _jbig2_to_pdf(sym_path: str, page_files: List[str]) -> bytes:
    """
    Convert JBIG2 encoded files to PDF.

    This is a Python implementation of jbig2topdf.py from jbig2enc.
    Based on https://github.com/agl/jbig2enc (Apache 2.0 license).

    Args:
        sym_path: Path to symbol table file (.sym)
        page_files: List of page file paths (.0000, .0001, etc.)

    Returns:
        PDF bytes
    """
    class Ref:
        def __init__(self, x: int):
            self.x = x
        def __str__(self) -> str:
            return f"{self.x} 0 R"

    class Dict:
        def __init__(self, values: dict = None):
            self.d = (values or {}).copy()
        def __str__(self) -> str:
            entries = [f"/{key} {value}" for key, value in self.d.items()]
            return f"<< {' '.join(entries)} >>\n"

    class Obj:
        next_id = 1
        def __init__(self, d: dict = None, stream: str = None):
            if d is None:
                d = {}
            if stream is not None:
                d["Length"] = str(len(stream))
            self.d = Dict(d)
            self.stream = stream
            self.id = Obj.next_id
            Obj.next_id += 1
        def __str__(self) -> str:
            result = [str(self.d)]
            if self.stream is not None:
                result.append(f"stream\n{self.stream}\nendstream\n")
            result.append("endobj\n")
            return "".join(result)

    class Doc:
        def __init__(self):
            self.objs = []
            self.pages = []
        def add_object(self, obj: Obj) -> Obj:
            self.objs.append(obj)
            return obj
        def __str__(self) -> str:
            output = []
            offsets = []
            current_offset = 0
            def add_line(line: str):
                nonlocal current_offset
                output.append(line)
                current_offset += len(line) + 1
            add_line("%PDF-1.4")
            for obj in self.objs:
                offsets.append(current_offset)
                add_line(f"{obj.id} 0 obj")
                add_line(str(obj))
            xref_start = current_offset
            add_line("xref")
            add_line(f"0 {len(offsets) + 1}")
            add_line("0000000000 65535 f ")
            for offset in offsets:
                add_line(f"{offset:010} 00000 n ")
            add_line("trailer")
            add_line(f"<< /Size {len(offsets) + 1}\n/Root 1 0 R >>")
            add_line("startxref")
            add_line(str(xref_start))
            add_line("%%EOF")
            return "\n".join(output)

    def ref(x: int) -> str:
        return f"{x} 0 R"

    # Reset object ID counter
    Obj.next_id = 1

    doc = Doc()
    dpi = 72

    # Add catalog and outlines objects
    doc.add_object(Obj({"Type": "/Catalog", "Outlines": ref(2), "Pages": ref(3)}))
    doc.add_object(Obj({"Type": "/Outlines", "Count": "0"}))
    pages_obj = Obj({"Type": "/Pages"})
    doc.add_object(pages_obj)

    # Read symbol table if it exists
    symd = None
    if sym_path and Path(sym_path).exists():
        sym_data = Path(sym_path).read_bytes()
        symd = doc.add_object(Obj({}, sym_data.decode("latin1")))

    page_objs = []
    page_files.sort()

    for p in page_files:
        contents = Path(p).read_bytes()
        try:
            width, height, xres, yres = struct.unpack(">IIII", contents[11:27])
        except struct.error:
            logger.warning(f"Error unpacking page file: {p}")
            continue

        xres = xres or dpi
        yres = yres or dpi

        lexicon = {
            "Type": "/XObject",
            "Subtype": "/Image",
            "Width": str(width),
            "Height": str(height),
            "ColorSpace": "/DeviceGray",
            "BitsPerComponent": "1",
            "Filter": "/JBIG2Decode",
        }
        if symd:
            lexicon["DecodeParms"] = f"<< /JBIG2Globals {symd.id} 0 R >>"

        xobj = doc.add_object(Obj(lexicon, contents.decode("latin1")))
        contents_obj = doc.add_object(Obj(
            {},
            f"q {float(width * 72) / xres} 0 0 {float(height * 72) / yres} 0 0 cm /Im1 Do Q"
        ))
        resources_obj = doc.add_object(Obj(
            {"ProcSet": "[/PDF /ImageB]", "XObject": f"<< /Im1 {xobj.id} 0 R >>"}
        ))
        page_obj = doc.add_object(Obj({
            "Type": "/Page",
            "Parent": "3 0 R",
            "MediaBox": f"[ 0 0 {float(width * 72) / xres} {float(height * 72) / yres} ]",
            "Contents": ref(contents_obj.id),
            "Resources": ref(resources_obj.id),
        }))
        page_objs.append(page_obj)

    pages_obj.d.d["Count"] = str(len(page_objs))
    pages_obj.d.d["Kids"] = "[" + " ".join([ref(x.id) for x in page_objs]) + "]"

    return str(doc).encode("latin1")


def rasterize_pdf_jbig2(
    pdf_path: Path,
    output_path: Path,
    pages: Optional[List[int]] = None,
    dpi: int = 150
) -> Tuple[bool, Dict]:
    """
    Rasterize PDF pages to JBIG2 compressed PDF using per-page mode.

    Each page is individually rendered, binarized with Otsu's method,
    and compressed with jbig2 in per-page mode (no symbol table).

    Args:
        pdf_path: Input PDF path
        output_path: Output PDF path
        pages: Pages to process (1-indexed), None for all
        dpi: Render resolution

    Returns:
        (success, stats_dict)
    """
    if not check_jbig2_available():
        logger.warning("jbig2 not available")
        return False, {}

    doc = fitz.open(pdf_path)
    page_indices = [p - 1 for p in pages] if pages else list(range(len(doc)))

    with tempfile.TemporaryDirectory() as tmpdir:
        jbig2_files = []

        for i, page_idx in enumerate(tqdm(page_indices, desc=f"JBIG2 {dpi}dpi", unit="page")):
            page = doc[page_idx]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            img = Image.frombytes('L', [pix.width, pix.height], pix.samples)
            img_bw = _binarize_image(img)

            # Save as PBM for jbig2 input
            pbm_path = f'{tmpdir}/page_{i:04d}.pbm'
            img_bw.save(pbm_path)

            # Compress single page with jbig2 per-page mode (no -s, no -b)
            jbig2_path = f'{tmpdir}/page_{i:04d}.jbig2'
            result = subprocess.run(
                ['jbig2', '-p', pbm_path],
                capture_output=True
            )
            if result.returncode != 0:
                logger.error(f"jbig2 failed on page {page_idx + 1}: {result.stderr.decode()}")
                doc.close()
                return False, {}

            # jbig2 -p writes to stdout
            with open(jbig2_path, 'wb') as f:
                f.write(result.stdout)
            jbig2_files.append(jbig2_path)

        doc.close()

        if not jbig2_files:
            logger.error("No JBIG2 pages produced")
            return False, {}

        # Assemble PDF (no symbol table in per-page mode)
        pdf_bytes = _jbig2_to_pdf(sym_path=None, page_files=jbig2_files)

        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)

    output_size = output_path.stat().st_size
    return True, {
        'output_size_mb': output_size / 1024 / 1024,
        'page_count': len(page_indices),
        'dpi': dpi,
        'method': 'jbig2'
    }


def rasterize_to_limit(
    pdf_path: Path,
    output_path: Path,
    pages: Optional[List[int]] = None,
    target_mb: float = 30.0
) -> Tuple[bool, Dict]:
    """
    Try JBIG2 rasterization at decreasing DPI until file is below target size.

    Returns (False, {}) if jbig2 is unavailable or all DPI levels exceed target.
    """
    for dpi in JBIG2_DPI_LEVELS:
        success, stats = rasterize_pdf_jbig2(pdf_path, output_path, pages, dpi)
        if not success:
            return False, {}
        if stats['output_size_mb'] <= target_mb:
            logger.info(f"Rasterized at {dpi} DPI: {stats['output_size_mb']:.1f} MB")
            return True, stats
        logger.warning(
            f"{dpi} DPI produced {stats['output_size_mb']:.1f} MB, "
            f"trying lower DPI..."
        )

    logger.error(f"Failed to rasterize below {target_mb} MB even at lowest DPI")
    return False, {}
