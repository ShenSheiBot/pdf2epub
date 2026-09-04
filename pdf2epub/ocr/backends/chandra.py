"""Chandra 2 OCR client and lossless layout materializer.

The GPU model is deliberately kept behind its OpenAI-compatible vLLM service.
This module only renders a page, sends one image request, and turns Chandra's
raw layout HTML into three durable views:

* Markdown for the existing refine/polish pipeline;
* HTML retaining block labels and bounding boxes;
* ordered JSON blocks plus the exact raw model response.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz
import six
from bs4 import BeautifulSoup, Tag
from markdownify import MarkdownConverter, re_whitespace
from openai import OpenAI
from PIL import Image

from ..artifacts import OCRPageResult


ALLOWED_TAGS = [
    "math", "br", "i", "b", "u", "del", "sup", "sub", "table", "tr",
    "td", "p", "th", "div", "pre", "h1", "h2", "h3", "h4", "h5",
    "ul", "ol", "li", "input", "a", "span", "img", "hr", "tbody",
    "small", "caption", "strong", "thead", "big", "code", "chem",
]
ALLOWED_ATTRIBUTES = [
    "class", "colspan", "rowspan", "display", "checked", "type", "border",
    "value", "style", "href", "alt", "align", "data-bbox", "data-label",
]

PROMPT_ENDING = f"""
Only use these tags {ALLOWED_TAGS}, and these attributes {ALLOWED_ATTRIBUTES}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Footnotes: Mark each genuine inline footnote reference as <sup class="footnote-ref">key</sup>. At the start of every definition inside a Footnote block, mark its key as <sup class="footnote-def">key</sup>. Never use either class for exponents, ordinals, or other ordinary superscripts.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a concise description in the alt attribute of an <img> tag and do not fill out the src property. Do not repeat or expand that description as visible text inside the image/figure div. Put only captions that are actually printed on the page in a separate Caption block. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags. Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret. Reading order should be correct and natural.
""".strip()

OCR_LAYOUT_PROMPT = f"""
OCR this image to HTML, arranged as layout blocks. Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format. Bboxes are normalized 0-1000. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure
- Chemical-Block
- Diagram
- Bibliography
- Blank-Page

{PROMPT_ENDING}
""".strip()


def _scale_to_fit(
    image: Image.Image,
    max_pixels: int = 3072 * 2048,
    min_pixels: int = 1792 * 28,
    grid_size: int = 28,
) -> Image.Image:
    """Match Chandra's official grid-aligned image preprocessing."""
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Cannot OCR an empty image")
    original_ratio = width / height
    current_pixels = width * height
    scale = 1.0
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5

    width_blocks = max(1, round(width * scale / grid_size))
    height_blocks = max(1, round(height * scale / grid_size))
    while width_blocks * height_blocks * grid_size * grid_size > max_pixels:
        if width_blocks == 1:
            height_blocks -= 1
        elif height_blocks == 1:
            width_blocks -= 1
        else:
            lose_width = abs((width_blocks - 1) / height_blocks - original_ratio)
            lose_height = abs(width_blocks / (height_blocks - 1) - original_ratio)
            if lose_width < lose_height:
                width_blocks -= 1
            else:
                height_blocks -= 1

    new_size = (width_blocks * grid_size, height_blocks * grid_size)
    if new_size == image.size:
        return image
    return image.resize(new_size, Image.Resampling.LANCZOS)


def render_pdf_page(pdf_bytes: bytes, *, dpi: int = 192, min_dimension: int = 1024) -> Image.Image:
    """Render a single PDF page using Chandra's native resolution policy."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        if len(document) != 1:
            raise ValueError(f"Chandra expects one PDF page, got {len(document)}")
        page = document[0]
        width, height = page.rect.width, page.rect.height
        base_scale = dpi / 72.0
        short_pixels = min(width, height) * base_scale
        if short_pixels < min_dimension:
            base_scale *= min_dimension / short_pixels
        pixmap = page.get_pixmap(matrix=fitz.Matrix(base_scale, base_scale), alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
        image.load()
        return image


def _detect_repeat_token(
    text: str,
    base_max_repeats: int = 4,
    window_size: int = 500,
    scaling_factor: float = 3.0,
) -> bool:
    """Detect the suffix repetition failure seen by Chandra on RTX 4090."""
    for sequence_length in range(1, min(window_size // 2, len(text)) + 1):
        candidate = text[-sequence_length:]
        allowed = int(base_max_repeats * (1 + scaling_factor / sequence_length))
        repeats = 0
        position = len(text) - sequence_length
        while position >= 0 and text[position : position + sequence_length] == candidate:
            repeats += 1
            position -= sequence_length
        if repeats > allowed:
            return True
    return False


def _parse_normalized_bbox(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    try:
        bbox = [int(part) for part in value.split()]
    except (TypeError, ValueError):
        return None
    if len(bbox) != 4:
        return None
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        return None
    return bbox


def _bbox_to_pixels(bbox: Iterable[int], size: Tuple[int, int]) -> List[int]:
    x0, y0, x1, y1 = bbox
    width, height = size
    return [
        max(0, min(width, int(x0 * width / 1000))),
        max(0, min(height, int(y0 * height / 1000))),
        max(0, min(width, int(x1 * width / 1000))),
        max(0, min(height, int(y1 * height / 1000))),
    ]


def _atomic_save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        image.save(temporary_name, format="PNG")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


@dataclass
class MaterializedLayout:
    html: str
    blocks: List[Dict[str, Any]]
    assets: List[Dict[str, Any]]


def materialize_layout(
    raw_html: str,
    page_image: Image.Image,
    images_dir: Optional[Path],
    page_number: int,
) -> MaterializedLayout:
    """Retain all blocks and recursively materialize every valid image bbox."""
    soup = BeautifulSoup(raw_html, "html.parser")
    top_level_divs = soup.find_all("div", recursive=False)
    blocks: List[Dict[str, Any]] = []
    assets: List[Dict[str, Any]] = []
    image_index = 0

    for order, div in enumerate(top_level_divs):
        label = div.get("data-label") or "block"
        normalized_bbox = _parse_normalized_bbox(div.get("data-bbox"))
        pixel_bbox = _bbox_to_pixels(normalized_bbox, page_image.size) if normalized_bbox else None
        blocks.append(
            {
                "order": order,
                "label": label,
                "bbox": normalized_bbox,
                "bbox_px": pixel_bbox,
                "html": str(div.decode_contents()),
            }
        )

        image_tags = list(div.find_all("img"))
        if label in {"Image", "Figure"} and not image_tags:
            image_tags = [soup.new_tag("img")]
            div.append(image_tags[0])

        for image_tag in image_tags:
            own_bbox = _parse_normalized_bbox(image_tag.get("data-bbox"))
            chosen_bbox = own_bbox or normalized_bbox
            if chosen_bbox is None:
                continue
            crop_bbox = _bbox_to_pixels(chosen_bbox, page_image.size)
            if crop_bbox[2] <= crop_bbox[0] or crop_bbox[3] <= crop_bbox[1]:
                continue
            image_index += 1
            filename = f"page_{page_number:03d}_img_{image_index:03d}.png"
            relative_path = f"../images/{filename}"
            if images_dir is not None:
                _atomic_save_image(page_image.crop(crop_bbox), images_dir / filename)
            image_tag["src"] = relative_path
            image_tag.attrs.pop("data-bbox", None)
            assets.append(
                {
                    "name": filename,
                    "path": relative_path,
                    "bbox": chosen_bbox,
                    "bbox_px": crop_bbox,
                    "block_order": order,
                    "nested": own_bbox is not None,
                    "alt": image_tag.get("alt", ""),
                }
            )

    return MaterializedLayout(html="".join(str(div) for div in top_level_divs), blocks=blocks, assets=assets)


class _ChandraMarkdownConverter(MarkdownConverter):
    def convert_math(self, element, text, parent_tags):
        if element.get("display") == "block":
            return f"\n\n$${text.strip()}$$\n\n"
        return f"${text.strip()}$"

    def convert_table(self, element, text, parent_tags):
        return "\n\n" + str(element) + "\n\n"

    def process_text(self, element, parent_tags=None):
        text = six.text_type(element) or ""
        if not element.find_parent("pre"):
            text = re_whitespace.sub(" ", text)
        if not element.find_parent(["pre", "code", "kbd", "samp", "math"]):
            text = self.escape(text, parent_tags)
        if element.parent.name == "li" and (
            not element.next_sibling or getattr(element.next_sibling, "name", None) in ["ul", "ol"]
        ):
            text = text.rstrip()
        return text


_MARKDOWN_FOOTNOTE_KEY_RE = re.compile(r"^\w+$")


def _superscript_key(element: Tag) -> Optional[str]:
    """Return a key that the existing Markdown footnote pipeline can parse."""
    key = "".join(element.stripped_strings)
    if _MARKDOWN_FOOTNOTE_KEY_RE.fullmatch(key):
        return key
    return None


def _leading_footnote_marker(block: Tag) -> Optional[Tuple[Tag, str]]:
    """Find a Footnote block's leading superscript without guessing from prose."""
    for descendant in block.descendants:
        if isinstance(descendant, Tag):
            if descendant.name == "sup":
                key = _superscript_key(descendant)
                return (descendant, key) if key is not None else None
            continue
        if str(descendant).strip():
            return None
    return None


def _is_leading_marker(block: Tag, marker: Tag) -> bool:
    """Return whether ``marker`` precedes all visible text in its block."""
    for descendant in block.descendants:
        if descendant is marker:
            return True
        if not isinstance(descendant, Tag) and str(descendant).strip():
            return False
    return False


def _has_footnote_class(marker: Tag, class_name: str) -> bool:
    return class_name in (marker.get("class") or [])


def _materialize_markdown_footnotes(soup: BeautifulSoup) -> Dict[str, str]:
    """Express Chandra's layout labels using the established Qwen syntax.

    Chandra identifies definition blocks structurally and marks genuine body
    references explicitly. The rest of the pipeline understands ``[^N]``
    references and ``[^N]:`` definitions. Ordinary unmarked superscripts remain
    untouched, even when their text happens to equal a footnote key.
    """
    footnote_blocks = soup.find_all("div", attrs={"data-label": "Footnote"})
    definition_markers: List[Tuple[Tag, str, bool]] = []
    for block in footnote_blocks:
        marked_definitions = []
        for marker in block.find_all("sup"):
            if not _has_footnote_class(marker, "footnote-def"):
                continue
            key = _superscript_key(marker)
            if key is not None:
                marked_definitions.append((marker, key))
        # Compatibility with already-produced Chandra layouts: a leading
        # superscript inside an explicit Footnote block is unambiguous. Include
        # it even when later definitions use the new explicit class.
        leading = _leading_footnote_marker(block)
        marked_ids = {id(marker) for marker, _key in marked_definitions}
        if leading is not None and id(leading[0]) not in marked_ids:
            definition_markers.append((leading[0], leading[1], True))
        definition_markers.extend(
            (marker, key, _is_leading_marker(block, marker))
            for marker, key in marked_definitions
        )

    replacements: Dict[str, str] = {}
    for index, (marker, key, is_leading) in enumerate(definition_markers, 1):
        placeholder = f"PDF2EPUBFOOTNOTEDEF{index:06d}TOKEN"
        prefix = "" if is_leading else "\n\n"
        replacements[placeholder] = f"{prefix}[^{key}]:"
        marker.replace_with(placeholder)

    # A Markdown definition is one logical line in the existing scanner. Keep
    # visual line breaks inside Chandra's definition block from detaching its
    # continuation text; additional definitions receive their own line through
    # the placeholders above.
    for block in footnote_blocks:
        for line_break in block.find_all("br"):
            line_break.replace_with(" ")
        for index, paragraph in enumerate(list(block.find_all("p"))):
            if index:
                paragraph.insert_before(" ")
            paragraph.unwrap()

    reference_index = 0
    for marker in list(soup.find_all("sup")):
        if marker.find_parent("div", attrs={"data-label": "Footnote"}) is not None:
            continue
        if not _has_footnote_class(marker, "footnote-ref"):
            continue
        key = _superscript_key(marker)
        if key is not None:
            reference_index += 1
            placeholder = f"PDF2EPUBFOOTNOTEREF{reference_index:06d}TOKEN"
            replacements[placeholder] = f"[^{key}]"
            marker.replace_with(placeholder)
    return replacements


def layout_to_markdown(html: str, *, include_headers_footers: bool) -> str:
    """Create the compatibility Markdown view without altering stored HTML."""
    soup = BeautifulSoup(html, "html.parser")
    footnote_replacements = _materialize_markdown_footnotes(soup)
    # Chandra's image descriptions are useful in the retained HTML/JSON
    # evidence, but they are not source text. Exclude them from the Markdown
    # that is polished, translated, and published. Real printed captions are
    # emitted as separate Caption blocks and remain untouched.
    for image in soup.find_all("img"):
        image["alt"] = ""
    for div in soup.find_all("div", recursive=False):
        label = div.get("data-label")
        if not include_headers_footers and label in {"Page-Header", "Page-Footer"}:
            div.decompose()
        elif label in {"Image", "Figure"}:
            for text_node in div.find_all(string=True):
                text_node.extract()
    converter = _ChandraMarkdownConverter(
        heading_style="ATX",
        bullets="-",
        escape_misc=False,
        escape_underscores=True,
        escape_asterisks=True,
        escape_dollars=True,
        sub_symbol="<sub>",
        sup_symbol="<sup>",
    )
    markdown = converter.convert(str(soup)).strip()
    for placeholder, replacement in footnote_replacements.items():
        markdown = markdown.replace(placeholder, replacement)
    markdown = re.sub(r"^(\[\^\w+\]:)[ \t]*", r"\1 ", markdown, flags=re.MULTILINE)
    return markdown


def collect_caption_markdown(pages_dir: Path) -> List[str]:
    """Load exact printed-caption Markdown from retained Chandra sidecars."""
    fragments: List[str] = []
    for artifact_path in sorted(Path(pages_dir).glob("page_*.ocr.json")):
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        for block in artifact.get("blocks", []):
            if block.get("label") != "Caption" or not isinstance(
                block.get("html"), str
            ):
                continue
            fragment = layout_to_markdown(
                block["html"], include_headers_footers=True
            ).strip()
            if fragment:
                fragments.append(fragment)
    return fragments


class ChandraClient:
    def __init__(self, config: Dict[str, Any]):
        ocr_config = config.get("ocr", {})
        backend_config = ocr_config.get("backends", {}).get("chandra", {})
        self.base_url = backend_config.get("base_url", "http://127.0.0.1:8100/v1")
        self.model = backend_config.get("model", "chandra")
        self.model_revision = backend_config.get("model_revision")
        self.max_output_tokens = int(backend_config.get("max_output_tokens", 12384))
        self.max_retries = int(backend_config.get("max_retries", ocr_config.get("max_retries", 6)))
        self.initial_backoff = float(backend_config.get("initial_backoff", ocr_config.get("initial_backoff", 2.0)))
        self.request_timeout = float(backend_config.get("request_timeout", 300.0))
        self.include_headers_footers = bool(backend_config.get("include_headers_footers", False))
        self.dpi = int(backend_config.get("dpi", 192))
        self.min_dimension = int(backend_config.get("min_dimension", 1024))
        access_client_id_env = backend_config.get(
            "access_client_id_env", "CHANDRA_CF_ACCESS_CLIENT_ID"
        )
        access_client_secret_env = backend_config.get(
            "access_client_secret_env", "CHANDRA_CF_ACCESS_CLIENT_SECRET"
        )
        access_client_id = os.environ.get(access_client_id_env)
        access_client_secret = os.environ.get(access_client_secret_env)
        if bool(access_client_id) != bool(access_client_secret):
            raise ValueError(
                "Chandra Cloudflare Access requires both "
                f"{access_client_id_env} and {access_client_secret_env}"
            )
        default_headers = None
        if access_client_id and access_client_secret:
            default_headers = {
                "CF-Access-Client-Id": access_client_id,
                "CF-Access-Client-Secret": access_client_secret,
            }
        self.client = OpenAI(
            api_key=backend_config.get("api_key", "EMPTY"),
            base_url=self.base_url,
            timeout=self.request_timeout,
            # Page-level retries below preserve attempt count, backoff, and
            # progress semantics.  Layering the SDK's implicit retries on top
            # would multiply the configured timeout without visibility.
            max_retries=0,
            default_headers=default_headers,
        )

    def _request(
        self,
        model_image: Image.Image,
        *,
        temperature: float,
        top_p: float,
    ) -> tuple[str, int, Optional[str]]:
        buffer = io.BytesIO()
        model_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                        {"type": "text", "text": OCR_LAYOUT_PROMPT},
                    ],
                }
            ],
            max_tokens=self.max_output_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        choice = response.choices[0]
        raw = choice.message.content or ""
        token_count = response.usage.completion_tokens if response.usage else 0
        return raw, token_count, choice.finish_reason

    def process_pdf_page(
        self,
        pdf_bytes: bytes,
        *,
        page_number: int,
        images_dir: Optional[Path],
        image_counter: int,
    ) -> OCRPageResult:
        page_image = render_pdf_page(pdf_bytes, dpi=self.dpi, min_dimension=self.min_dimension)
        model_image = _scale_to_fit(page_image)
        last_error: Optional[Exception] = None
        raw = ""
        token_count = 0
        finish_reason: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                temperature = min(0.2 * attempt, 0.8)
                top_p = 0.1 if attempt == 0 else 0.95
                raw, token_count, finish_reason = self._request(
                    model_image,
                    temperature=temperature,
                    top_p=top_p,
                )
                if finish_reason != "stop":
                    raise ValueError(
                        f"Chandra response did not finish normally: finish_reason={finish_reason!r}"
                    )
                if not raw.strip():
                    raise ValueError("Chandra returned an empty response")
                if not BeautifulSoup(raw, "html.parser").find_all("div", recursive=False):
                    raise ValueError("Chandra response has no top-level layout blocks")
                if _detect_repeat_token(raw) or (len(raw) > 50 and _detect_repeat_token(raw[:-50])):
                    raise ValueError("Chandra response ended in a repeated sequence")
                break
            except Exception as error:
                last_error = error
                if attempt + 1 >= self.max_retries:
                    raise RuntimeError(f"Chandra OCR failed after {self.max_retries} attempts: {error}") from error
                time.sleep(self.initial_backoff * (attempt + 1))
        else:  # pragma: no cover - loop always returns or raises
            raise RuntimeError(f"Chandra OCR failed: {last_error}")

        materialized = materialize_layout(raw, page_image, images_dir, page_number)
        markdown = layout_to_markdown(
            materialized.html,
            include_headers_footers=self.include_headers_footers,
        )
        image_records = [
            {"filename": asset["name"], "format": "png", "bbox": asset["bbox_px"]}
            for asset in materialized.assets
        ]
        return OCRPageResult(
            markdown=markdown,
            html=materialized.html,
            raw_html=raw,
            blocks=materialized.blocks,
            page_box=[0, 0, page_image.width, page_image.height],
            model_input_size=[model_image.width, model_image.height],
            token_count=token_count,
            backend="chandra",
            model=self.model,
            model_revision=self.model_revision,
            images=image_records,
            assets=materialized.assets,
            image_counter=image_counter + len(materialized.assets),
        )


_CLIENTS: Dict[Tuple[str, str], ChandraClient] = {}


def get_client(config: Dict[str, Any]) -> ChandraClient:
    backend_config = config.get("ocr", {}).get("backends", {}).get("chandra", {})
    key = (
        backend_config.get("base_url", "http://127.0.0.1:8100/v1"),
        backend_config.get("model", "chandra"),
    )
    if key not in _CLIENTS:
        _CLIENTS[key] = ChandraClient(config)
    return _CLIENTS[key]


def process_pdf_page(
    pdf_bytes: bytes,
    config: Dict[str, Any],
    *,
    page_number: int,
    images_dir: Optional[Path],
    image_counter: int,
) -> OCRPageResult:
    return get_client(config).process_pdf_page(
        pdf_bytes,
        page_number=page_number,
        images_dir=images_dir,
        image_counter=image_counter,
    )
