"""
PDF rasterization utilities for API compatibility.

When Gemini API rejects certain PDF structures (503 error),
this module provides JBIG2 rasterization as fallback.
"""

import glob
import io
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
from loguru import logger
from PIL import Image


JBIG2_DPI_LEVELS = [150, 120, 100]


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
    Rasterize PDF pages to JBIG2 compressed PDF.

    Args:
        pdf_path: Input PDF path
        output_path: Output PDF path
        pages: Pages to process (1-indexed), None for all
        dpi: Render resolution

    Returns:
        (success, stats_dict)
    """
    if not check_jbig2_available():
        logger.warning("jbig2 not available, falling back to CCITT G4")
        return rasterize_pdf_ccitt(pdf_path, output_path, pages, dpi)

    doc = fitz.open(pdf_path)
    page_indices = [p - 1 for p in pages] if pages else list(range(len(doc)))

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Render each page to binary PBM
        pbm_files = []
        for i, page_idx in enumerate(page_indices):
            page = doc[page_idx]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            img = Image.frombytes('L', [pix.width, pix.height], pix.samples)
            img_bw = img.convert('1')
            pbm_path = f'{tmpdir}/page_{i:04d}.pbm'
            img_bw.save(pbm_path)
            pbm_files.append(pbm_path)

        doc.close()

        # 2. Compress with jbig2
        output_base = f'{tmpdir}/output'
        result = subprocess.run(
            ['jbig2', '-s', '-p', '-b', output_base] + pbm_files,
            capture_output=True
        )
        if result.returncode != 0:
            logger.error(f"jbig2 failed: {result.stderr.decode()}")
            return False, {}

        # 3. Assemble PDF using built-in converter
        sym_path = f'{output_base}.sym'
        page_files = sorted(glob.glob(f'{output_base}.[0-9]*'))

        if not page_files:
            logger.error("jbig2 produced no output files")
            return False, {}

        pdf_bytes = _jbig2_to_pdf(sym_path, page_files)

        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)

    output_size = output_path.stat().st_size
    return True, {
        'output_size_mb': output_size / 1024 / 1024,
        'page_count': len(page_indices),
        'dpi': dpi,
        'method': 'jbig2'
    }


def rasterize_pdf_ccitt(
    pdf_path: Path,
    output_path: Path,
    pages: Optional[List[int]] = None,
    dpi: int = 150
) -> Tuple[bool, Dict]:
    """
    Fallback: Use CCITT G4 compression (pure Python, no external dependencies).
    Note: Files will be ~8x larger than JBIG2.
    """
    try:
        doc = fitz.open(pdf_path)
        page_indices = [p - 1 for p in pages] if pages else list(range(len(doc)))

        new_doc = fitz.open()
        for page_idx in page_indices:
            page = doc[page_idx]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            img = Image.frombytes('L', [pix.width, pix.height], pix.samples)
            img_bw = img.convert('1')

            tiff_buffer = io.BytesIO()
            img_bw.save(tiff_buffer, format='TIFF', compression='group4')
            tiff_data = tiff_buffer.getvalue()

            img_doc = fitz.open('tiff', tiff_data)
            rect = img_doc[0].rect
            new_page = new_doc.new_page(width=rect.width, height=rect.height)
            new_page.insert_image(rect, stream=tiff_data)
            img_doc.close()

        new_doc.save(str(output_path))
        new_doc.close()
        doc.close()

        output_size = output_path.stat().st_size
        return True, {
            'output_size_mb': output_size / 1024 / 1024,
            'page_count': len(page_indices),
            'dpi': dpi,
            'method': 'ccitt_g4'
        }
    except Exception as e:
        logger.error(f"CCITT rasterization failed: {e}")
        return False, {}


def rasterize_to_limit(
    pdf_path: Path,
    output_path: Path,
    pages: Optional[List[int]] = None,
    target_mb: float = 30.0
) -> Tuple[bool, Dict]:
    """
    Progressive degradation until file is below target size.
    """
    for dpi in JBIG2_DPI_LEVELS:
        success, stats = rasterize_pdf_jbig2(pdf_path, output_path, pages, dpi)
        if not success:
            continue
        if stats['output_size_mb'] <= target_mb:
            logger.info(f"Rasterized at {dpi} DPI: {stats['output_size_mb']:.1f} MB")
            return True, stats
        logger.warning(
            f"{dpi} DPI produced {stats['output_size_mb']:.1f} MB, "
            f"trying lower DPI..."
        )

    logger.error(f"Failed to rasterize below {target_mb} MB even at lowest DPI")
    return False, {}
