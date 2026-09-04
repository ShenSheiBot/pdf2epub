"""Render LaTeX formulas as self-contained inline SVG for EPUB 2."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"
ID_MARKER = "PDF2EPUBMATHID-"
RENDERER_VERSION = "latex-svg-v4"
TOOL_TIMEOUT_SECONDS = 60

_ALLOWED_SVG_ATTRIBUTES = {
    "svg": {
        "class",
        "fill",
        "height",
        "preserveAspectRatio",
        "version",
        "viewBox",
        "width",
    },
    "g": {"id", "transform"},
    "path": {"d", "id", "transform"},
    "rect": {"height", "transform", "width", "x", "y"},
}
_SAFE_SVG_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_SAFE_TRANSFORM_RE = re.compile(
    r"^(?:(?:matrix|translate|scale|rotate|skewX|skewY)|[0-9eE+.,()\s-])+$"
)
_STANDALONE_DISPLAY_RE = re.compile(
    r"\A\s*\\begin\{(?P<environment>"
    r"equation\*?|align\*?|alignat\*?|flalign\*?|gather\*?|multline\*?|"
    r"displaymath|eqnarray\*?"
    r")\}.*\\end\{(?P=environment)\}\s*\Z",
    flags=re.DOTALL,
)


def _wrap_formula_for_tex(source: str, display: bool) -> str:
    """Place a formula in the right TeX math context without nesting displays."""
    if display and _STANDALONE_DISPLAY_RE.fullmatch(source):
        return source
    if display:
        return rf"\[\displaystyle {source}\]"
    return rf"\({source}\)"


class MathRenderingError(RuntimeError):
    """Raised when a formula cannot be rendered without losing information."""


class LatexSvgRenderer:
    """Use XeLaTeX and dvisvgm to produce theme-adaptive inline SVG.

    Cached SVG templates contain an ID marker. Each insertion replaces that
    marker with a fresh XML ID prefix, so repeated formulas in one XHTML file
    cannot collide.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self._counter = 0
        self.render_count = 0
        self.cache_hit_count = 0
        self._tools: tuple[str, str, Path] | None = None

    def render(self, latex_source: str, display: bool = False) -> str:
        """Return one self-contained SVG fragment for ``latex_source``."""
        source = latex_source.strip()
        if not source:
            raise MathRenderingError("Cannot render an empty formula")

        digest = hashlib.sha256(
            f"{RENDERER_VERSION}\0{int(display)}\0{source}".encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{digest}.svg"

        if cache_path.exists():
            svg_template = cache_path.read_text(encoding="utf-8")
            self.cache_hit_count += 1
        else:
            svg_template = self._compile(source, display, digest)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.cache_dir,
                prefix=f".{digest}.",
                suffix=".tmp",
                delete=False,
            ) as cache_file:
                cache_file.write(svg_template)
                temporary_path = Path(cache_file.name)
            os.replace(temporary_path, cache_path)

        self._counter += 1
        self.render_count += 1
        instance_prefix = f"math-{digest[:12]}-{self._counter}-"
        return svg_template.replace(ID_MARKER, instance_prefix)

    def _resolve_tools(self) -> tuple[str, str, Path]:
        if self._tools is not None:
            return self._tools

        xelatex = shutil.which("xelatex")
        dvisvgm = shutil.which("dvisvgm")
        kpsewhich = shutil.which("kpsewhich")
        missing = [
            name
            for name, path in (
                ("xelatex", xelatex),
                ("dvisvgm", dvisvgm),
                ("kpsewhich", kpsewhich),
            )
            if path is None
        ]
        if missing:
            raise MathRenderingError(
                "EPUB formula rendering requires TeX Live tools: "
                + ", ".join(missing)
            )

        try:
            font_probe = subprocess.run(
                [kpsewhich, "FandolSong-Regular.otf"],
                check=False,
                capture_output=True,
                text=True,
                timeout=TOOL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise MathRenderingError("kpsewhich timed out while locating the CJK font") from error
        font_path = Path(font_probe.stdout.strip())
        if font_probe.returncode != 0 or not font_path.is_file():
            raise MathRenderingError(
                "XeLaTeX could not locate FandolSong-Regular.otf for CJK formulas"
            )

        self._tools = (xelatex, dvisvgm, font_path)
        return self._tools

    def _compile(self, source: str, display: bool, digest: str) -> str:
        xelatex, dvisvgm, font_path = self._resolve_tools()
        formula = _wrap_formula_for_tex(source, display)
        tex_source = "\n".join(
            [
                r"\documentclass[preview,border=1pt]{standalone}",
                r"\usepackage{amsmath,amssymb,mathtools}",
                r"\usepackage{fontspec}",
                r"\usepackage{xeCJK}",
                (
                    rf"\setCJKmainfont[Path={font_path.parent.as_posix()}/]"
                    rf"{{{font_path.name}}}"
                ),
                r"\begin{document}",
                formula,
                r"\end{document}",
                "",
            ]
        )

        with tempfile.TemporaryDirectory(prefix="pdf2epub-math-") as work_dir_string:
            work_dir = Path(work_dir_string)
            tex_path = work_dir / "formula.tex"
            tex_path.write_text(tex_source, encoding="utf-8")

            tex_environment = os.environ.copy()
            tex_environment.update(
                {
                    "openin_any": "p",
                    "openout_any": "p",
                    "shell_escape": "f",
                }
            )
            try:
                latex_result = subprocess.run(
                    [
                        xelatex,
                        "-no-pdf",
                        "-no-shell-escape",
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        tex_path.name,
                    ],
                    cwd=work_dir,
                    env=tex_environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=TOOL_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as error:
                raise MathRenderingError(
                    f"XeLaTeX timed out for formula {digest[:12]}"
                ) from error
            xdv_path = work_dir / "formula.xdv"
            if latex_result.returncode != 0 or not xdv_path.exists():
                raise MathRenderingError(
                    f"XeLaTeX failed for formula {digest[:12]}: "
                    f"{self._error_tail(latex_result.stdout, latex_result.stderr)}"
                )

            svg_path = work_dir / "formula.svg"
            try:
                svg_result = subprocess.run(
                    [
                        dvisvgm,
                        "--no-fonts",
                        "--exact-bbox",
                        "--page=1",
                        f"--output={svg_path.name}",
                        xdv_path.name,
                    ],
                    cwd=work_dir,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=TOOL_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as error:
                raise MathRenderingError(
                    f"dvisvgm timed out for formula {digest[:12]}"
                ) from error
            if svg_result.returncode != 0 or not svg_path.exists():
                raise MathRenderingError(
                    f"dvisvgm failed for formula {digest[:12]}: "
                    f"{self._error_tail(svg_result.stdout, svg_result.stderr)}"
                )

            return self._prepare_svg(svg_path.read_text(encoding="utf-8"), display)

    @staticmethod
    def _error_tail(stdout: str, stderr: str) -> str:
        lines = [
            line.strip()
            for line in (stdout + "\n" + stderr).splitlines()
            if line.strip()
        ]
        return " | ".join(lines[-8:]) or "no diagnostic output"

    @staticmethod
    def _prepare_svg(svg_text: str, display: bool) -> str:
        try:
            root = ET.fromstring(svg_text)
        except ET.ParseError as error:
            raise MathRenderingError(f"dvisvgm returned invalid SVG: {error}") from error

        if root.tag != f"{{{SVG_NAMESPACE}}}svg":
            raise MathRenderingError("dvisvgm output did not contain an SVG root")

        root.set(
            "class",
            "math-svg math-svg-display" if display else "math-svg math-svg-inline",
        )
        root.set("fill", "currentColor")
        root.set("preserveAspectRatio", "xMidYMid meet")

        # Calibre accepts SVG in EPUB 2 but its conversion/rendering path can
        # leave dvisvgm's glyph indirection blank. Expand every glyph reference
        # to a directly positioned path before the fragment enters the book.
        elements_by_id = {
            element.get("id"): element
            for element in root.iter()
            if element.get("id")
        }
        use_tag = f"{{{SVG_NAMESPACE}}}use"
        defs_tag = f"{{{SVG_NAMESPACE}}}defs"
        for parent in list(root.iter()):
            for child_index, child in enumerate(list(parent)):
                if child.tag != use_tag:
                    continue
                href = child.get(f"{{{XLINK_NAMESPACE}}}href") or child.get("href")
                target = elements_by_id.get(
                    href[1:] if href and href.startswith("#") else ""
                )
                if target is None or target.tag != f"{{{SVG_NAMESPACE}}}path":
                    raise MathRenderingError(
                        "dvisvgm SVG contained an unsupported glyph reference"
                    )

                replacement = ET.Element(target.tag, {"d": target.get("d", "")})
                x = child.get("x", "0")
                y = child.get("y", "0")
                replacement.set("transform", f"translate({x} {y})")
                replacement.tail = child.tail
                parent.remove(child)
                parent.insert(child_index, replacement)

        for parent in list(root.iter()):
            for child in list(parent):
                if child.tag == defs_tag:
                    parent.remove(child)

        # TeX source is untrusted OCR/model output. dvisvgm raw specials can
        # emit executable SVG, so enforce the renderer's path-only contract
        # before embedding the fragment in an EPUB.
        for element in root.iter():
            if not element.tag.startswith(f"{{{SVG_NAMESPACE}}}"):
                raise MathRenderingError("dvisvgm SVG contained a foreign namespace")
            local_name = element.tag.rsplit("}", 1)[-1]
            if local_name not in _ALLOWED_SVG_ATTRIBUTES:
                raise MathRenderingError(
                    f"dvisvgm SVG contained an unsupported <{local_name}> element"
                )
            if element is not root and local_name == "svg":
                raise MathRenderingError("dvisvgm SVG contained a nested SVG root")
            if element.text is not None and element.text.strip():
                raise MathRenderingError("dvisvgm SVG contained unexpected text")
            if element.tail is not None and element.tail.strip():
                raise MathRenderingError("dvisvgm SVG contained unexpected tail text")
            allowed_attributes = _ALLOWED_SVG_ATTRIBUTES[local_name]
            for attribute_name, attribute_value in element.attrib.items():
                if attribute_name not in allowed_attributes:
                    raise MathRenderingError(
                        "dvisvgm SVG contained an unsupported attribute: "
                        f"{local_name}.{attribute_name}"
                    )
                if attribute_name == "id" and not _SAFE_SVG_ID_RE.fullmatch(
                    attribute_value
                ):
                    raise MathRenderingError("dvisvgm SVG contained an invalid ID")
                if attribute_name == "transform" and not _SAFE_TRANSFORM_RE.fullmatch(
                    attribute_value
                ):
                    raise MathRenderingError("dvisvgm SVG contained an unsafe transform")

        for element in root.iter():
            if element.text is not None and not element.text.strip():
                element.text = None
            if element.tail is not None and not element.tail.strip():
                element.tail = None
            element_id = element.get("id")
            if element_id:
                element.set("id", ID_MARKER + element_id)
            for attribute_name, attribute_value in list(element.attrib.items()):
                if attribute_name in (
                    "href",
                    f"{{{XLINK_NAMESPACE}}}href",
                ) and attribute_value.startswith("#"):
                    element.set(
                        attribute_name,
                        "#" + ID_MARKER + attribute_value[1:],
                    )
                elif "url(#" in attribute_value:
                    element.set(
                        attribute_name,
                        re.sub(
                            r"url\(#([^)]+)\)",
                            rf"url(#{ID_MARKER}\1)",
                            attribute_value,
                        ),
                    )

        ET.register_namespace("", SVG_NAMESPACE)
        ET.register_namespace("xlink", XLINK_NAMESPACE)
        return ET.tostring(root, encoding="unicode", short_empty_elements=True)
