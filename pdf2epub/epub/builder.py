"""
EPUB builder for creating EPUB file structure.

This module handles all EPUB-specific file generation including:
- Structural files (content.opf, toc.ncx, etc.)
- Stylesheets
- Final EPUB packaging
"""

import os
import html
import uuid
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger


class EpubBuilder:
    """Handles creation of all EPUB structural files and final packaging."""
    
    def __init__(self, config, converter=None):
        """
        Initialize the EpubBuilder.
        
        Args:
            config: EpubConfig instance with all settings
            converter: Optional ContentConverter instance for file naming helpers
        """
        self.config = config
        self.converter = converter
    
    def create_cover_html(self, cover_image_filename: str, output_path: Path) -> bool:
        """
        Create the cover HTML page.
        
        Args:
            cover_image_filename: Name of the cover image file
            output_path: Path to save the cover HTML
            
        Returns:
            True if successful
        """
        try:
            # Get the language from config
            language = self.config.language if hasattr(self.config, 'language') else 'en'
            lang_class = f"lang-{language}"
            
            cover_html = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{language}" lang="{language}">
<head>
    <title>Cover</title>
    <link rel="stylesheet" href="../stylesheet.css" type="text/css"/>
</head>
<body class="{lang_class}">
    <div class="cover">
        <img src="../images/{cover_image_filename}" alt="{html.escape(self.config.book_title)} Cover" />
    </div>
</body>
</html>"""
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cover_html)
            
            logger.success(f"Created cover HTML: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create cover HTML: {e}")
            return False
    
    def create_stylesheet(self, output_path: Path) -> bool:
        """
        Create the EPUB stylesheet.
        
        Args:
            output_path: Path to save the stylesheet
            
        Returns:
            True if successful
        """
        try:
            # Load stylesheet from resources directory
            resources_dir = Path(__file__).parent / "resources"
            stylesheet_path = resources_dir / "stylesheet.css"
            
            if stylesheet_path.exists():
                # Use the elegant stylesheet from resources
                with open(stylesheet_path, "r", encoding="utf-8") as f:
                    stylesheet = f.read()
                logger.info(f"Loaded stylesheet from {stylesheet_path}")
            else:
                # Fallback to a minimal default stylesheet if file not found
                logger.warning(f"Stylesheet not found at {stylesheet_path}, using default")
                stylesheet = """/* Default EPUB Stylesheet */
@namespace h "http://www.w3.org/1999/xhtml";

body {
    font-family: Georgia, serif;
    line-height: 1.6;
    margin: 1em;
}

h1 { font-size: 2em; margin: 1em 0; page-break-before: always; }
h2 { font-size: 1.5em; margin: 0.8em 0; }
h3 { font-size: 1.2em; margin: 0.6em 0; }
p { margin: 0.8em 0; }
a { text-decoration: none; }
.toc { page-break-after: always; }
.cover { text-align: center; page-break-after: always; }
"""
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(stylesheet)
            
            logger.success(f"Created stylesheet: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create stylesheet: {e}")
            return False
    
    def create_toc_ncx(self, structure: Dict[str, Any], output_path: Path, 
                       parts_info: Optional[Dict] = None, chapters_with_parts: Optional[set] = None,
                       subchapter_locations: Optional[Dict] = None) -> bool:
        """
        Create the NCX navigation file for EPUB2 compatibility.
        
        Args:
            structure: Book structure dictionary
            output_path: Path to save the NCX file
            
        Returns:
            True if successful
        """
        try:
            # Generate unique ID
            uid = str(uuid.uuid4())
            
            # Start NCX document
            ncx = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE ncx PUBLIC "-//NISO//DTD ncx 2005-1//EN" "http://www.daisy.org/z3986/2005/ncx-2005-1.dtd">
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
    <head>
        <meta name="dtb:uid" content="{uid}"/>
        <meta name="dtb:depth" content="2"/>
        <meta name="dtb:totalPageCount" content="0"/>
        <meta name="dtb:maxPageNumber" content="0"/>
    </head>
    <docTitle>
        <text>{html.escape(self.config.book_title)}</text>
    </docTitle>
    <navMap>
"""
            
            # Start with TOC like original (no Cover entry)
            play_order = 1
            ncx += f"""        <navPoint id="navpoint-toc" playOrder="{play_order}">
            <navLabel>
                <text>Table of Contents</text>
            </navLabel>
            <content src="text/toc.html"/>
        </navPoint>
"""
            play_order += 1
            
            # Add chapters
            for i, chapter in enumerate(structure.get("chapters", []), 1):
                chapter_index = chapter.get('index', i)
                
                # Use converter's function to get proper filename for multi-part chapters
                if self.converter:
                    chapter_file = self.converter.get_chapter_html_filename(
                        chapter_index, parts_info, chapters_with_parts
                    )
                else:
                    chapter_file = f"chapter_{chapter_index}.html"
                
                chapter_title = chapter.get("title", f"Chapter {i}")
                
                ncx += f"""        <navPoint id="navpoint-chapter-{i}" playOrder="{play_order}">
            <navLabel>
                <text>{html.escape(chapter_title)}</text>
            </navLabel>
            <content src="text/{chapter_file}"/>
"""
                
                # Add subchapters if they exist
                if "subchapters" in chapter:
                    for j, subchapter in enumerate(chapter["subchapters"], 1):
                        play_order += 1
                        sub_title = subchapter.get("title", f"Section {j}")
                        anchor = f"{chapter_index}-{j}"
                        
                        # Use converter's function to get proper file for subchapter location
                        if self.converter and subchapter_locations:
                            subchapter_file = self.converter.get_subchapter_html_file(
                                chapter_index, j, subchapter_locations, chapter_file, sub_title
                            )
                        else:
                            subchapter_file = chapter_file
                        
                        ncx += f"""            <navPoint id="navpoint-sub-{i}-{j}" playOrder="{play_order}">
                <navLabel>
                    <text>{html.escape(sub_title)}</text>
                </navLabel>
                <content src="text/{subchapter_file}#{anchor}"/>
            </navPoint>
"""
                
                ncx += """        </navPoint>
"""
                play_order += 1
            
            # Add back matter if exists
            if structure.get("back_matter"):
                ncx += f"""        <navPoint id="navpoint-backmatter" playOrder="{play_order}">
            <navLabel>
                <text>Back Matter</text>
            </navLabel>
            <content src="text/back_matter.html"/>
        </navPoint>
"""
            
            ncx += """    </navMap>
</ncx>"""
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(ncx)
            
            logger.success(f"Created NCX TOC: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create NCX: {e}")
            return False
    
    def create_toc_html(self, structure: Dict[str, Any], output_path: Path,
                        parts_info: Optional[Dict] = None, chapters_with_parts: Optional[set] = None,
                        subchapter_locations: Optional[Dict] = None) -> bool:
        """
        Create an HTML table of contents page.
        
        Args:
            structure: Book structure dictionary
            output_path: Path to save the TOC HTML
            
        Returns:
            True if successful
        """
        try:
            # Get the language from config
            language = self.config.language if hasattr(self.config, 'language') else 'en'
            lang_class = f"lang-{language}"
            
            toc_html = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{language}" lang="{language}">
<head>
    <title>Table of Contents</title>
    <link rel="stylesheet" type="text/css" href="../stylesheet.css"/>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
</head>
<body class="{lang_class}">
    <div class="toc">
        <h1>Table of Contents</h1>
        <ul>
"""
            
            # Add front matter if exists
            if structure.get("front_matter"):
                toc_html += """            <li><a href="front_matter.html">Front Matter</a></li>
"""
            
            # Add chapters
            for i, chapter in enumerate(structure.get("chapters", []), 1):
                chapter_index = chapter.get('index', i)
                
                # Use converter's function to get proper filename for multi-part chapters
                if self.converter:
                    chapter_file = self.converter.get_chapter_html_filename(
                        chapter_index, parts_info, chapters_with_parts
                    )
                else:
                    chapter_file = f"chapter_{chapter_index}.html"
                
                chapter_title = chapter.get("title", f"Chapter {i}")
                
                toc_html += f"""            <li>
                <a href="{chapter_file}">{html.escape(chapter_title)}</a>
"""
                
                # Add subchapters if they exist
                if "subchapters" in chapter and chapter["subchapters"]:
                    toc_html += """                <ul>
"""
                    for j, subchapter in enumerate(chapter["subchapters"], 1):
                        sub_title = subchapter.get("title", f"Section {j}")
                        anchor = f"{chapter_index}-{j}"
                        
                        # Use converter's function to get proper file for subchapter location
                        if self.converter and subchapter_locations:
                            subchapter_file = self.converter.get_subchapter_html_file(
                                chapter_index, j, subchapter_locations, chapter_file, sub_title
                            )
                        else:
                            subchapter_file = chapter_file
                        
                        toc_html += f"""                    <li><a href="{subchapter_file}#{anchor}">{html.escape(sub_title)}</a></li>
"""
                    toc_html += """                </ul>
"""
                
                toc_html += """            </li>
"""
            
            # Add back matter if exists
            if structure.get("back_matter"):
                toc_html += """            <li><a href="back_matter.html">Back Matter</a></li>
"""
            
            toc_html += """        </ul>
    </div>
</body>
</html>"""
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(toc_html)
            
            logger.success(f"Created HTML TOC: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create HTML TOC: {e}")
            return False
    
    def create_container_xml(self, output_path: Path) -> bool:
        """
        Create the container.xml file.
        
        Args:
            output_path: Path to save the container.xml
            
        Returns:
            True if successful
        """
        try:
            container_xml = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>"""
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(container_xml)
            
            logger.success(f"Created container.xml: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create container.xml: {e}")
            return False
    
    def create_mimetype(self, output_path: Path) -> bool:
        """
        Create the mimetype file.
        
        Args:
            output_path: Path to save the mimetype file
            
        Returns:
            True if successful
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("application/epub+zip")
            
            logger.success(f"Created mimetype: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create mimetype: {e}")
            return False
    
    def create_content_opf(self, structure: Dict[str, Any], epub_dir: Path, 
                           output_path: Path, cover_image: Optional[str] = None,
                           all_html_files: Optional[List[str]] = None) -> bool:
        """
        Create the content.opf file with manifest and spine.
        
        Args:
            structure: Book structure dictionary
            epub_dir: EPUB directory containing all files
            output_path: Path to save the content.opf
            cover_image: Cover image filename
            all_html_files: List of all HTML files created (for multi-part chapters)
            
        Returns:
            True if successful
        """
        try:
            # Generate unique ID
            uid = str(uuid.uuid4())
            
            # Get current date
            date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Start OPF document
            opf = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="BookId">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:opf="http://www.idpf.org/2007/opf">
        <dc:title>{html.escape(self.config.book_title)}</dc:title>
        <dc:creator opf:role="aut">{html.escape(self.config.author)}</dc:creator>
        <dc:language>{self.config.language}</dc:language>
        <dc:identifier id="BookId" opf:scheme="UUID">{uid}</dc:identifier>
        <dc:date>{date}</dc:date>
        <meta name="generator" content="PDF2EPUB v3"/>
        <meta name="cover" content="cover-image"/>
    </metadata>
    
    <manifest>
        <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
        <item id="stylesheet" href="stylesheet.css" media-type="text/css"/>
        <item id="cover" href="text/cover.html" media-type="application/xhtml+xml"/>
        <item id="toc" href="text/toc.html" media-type="application/xhtml+xml"/>
"""
            
            # Add cover image to manifest
            if cover_image:
                opf += f"""        <item id="cover-image" href="images/{cover_image}" media-type="image/jpeg"/>
"""
            
            # Add HTML files to manifest
            if all_html_files:
                # Use provided list of HTML files (for proper ordering with multi-part chapters)
                for html_file in all_html_files:
                    item_id = html_file.replace(".html", "").replace(" ", "_").replace("-", "_")
                    opf += f"""        <item id="{item_id}" href="text/{html_file}" media-type="application/xhtml+xml"/>
"""
            else:
                # Fallback to structure-based manifest
                # Add front matter if exists
                if structure.get("front_matter"):
                    opf += """        <item id="front_matter" href="text/front_matter.html" media-type="application/xhtml+xml"/>
"""
                
                # Add chapters to manifest
                for i, chapter in enumerate(structure.get("chapters", []), 1):
                    chapter_id = f"chapter_{chapter.get('index', i)}"
                    chapter_file = f"{chapter_id}.html"
                    opf += f"""        <item id="{chapter_id}" href="text/{chapter_file}" media-type="application/xhtml+xml"/>
"""
                
                # Add back matter if exists
                if structure.get("back_matter"):
                    opf += """        <item id="back_matter" href="text/back_matter.html" media-type="application/xhtml+xml"/>
"""
            
            # Add images to manifest
            images_dir = epub_dir / "images"
            if images_dir.exists():
                for img_file in sorted(images_dir.iterdir()):
                    if img_file.is_file() and img_file.name != cover_image:
                        img_id = img_file.stem.replace(" ", "_").replace("-", "_")
                        media_type = self._get_image_media_type(img_file.suffix)
                        if media_type:
                            opf += f"""        <item id="img_{img_id}" href="images/{img_file.name}" media-type="{media_type}"/>
"""
            
            opf += """    </manifest>
    
    <spine toc="ncx">
        <itemref idref="cover" linear="no"/>
        <itemref idref="toc"/>
"""
            
            # Build spine based on all_html_files if provided
            if all_html_files:
                # Use the exact order of HTML files (important for multi-part chapters)
                for html_file in all_html_files:
                    if html_file != "cover.html" and html_file != "toc.html":  # Already added
                        item_id = html_file.replace(".html", "").replace(" ", "_").replace("-", "_")
                        opf += f"""        <itemref idref="{item_id}"/>
"""
            else:
                # Fallback to structure-based spine
                # Add front matter to spine
                if structure.get("front_matter"):
                    opf += """        <itemref idref="front_matter"/>
"""
                
                # Add chapters to spine
                for i, chapter in enumerate(structure.get("chapters", []), 1):
                    chapter_id = f"chapter_{chapter.get('index', i)}"
                    opf += f"""        <itemref idref="{chapter_id}"/>
"""
                
                # Add back matter to spine
                if structure.get("back_matter"):
                    opf += """        <itemref idref="back_matter"/>
"""
            
            opf += """    </spine>
</package>"""
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(opf)
            
            logger.success(f"Created content.opf: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create content.opf: {e}")
            return False
    
    def _get_image_media_type(self, suffix: str) -> Optional[str]:
        """Get the media type for an image file extension."""
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".webp": "image/webp"
        }
        return media_types.get(suffix.lower())
    
    def create_epub(self, epub_dir: Path, output_path: Path) -> bool:
        """
        Create the final EPUB file by zipping the directory.
        
        Args:
            epub_dir: Directory containing all EPUB files
            output_path: Path for the output EPUB file
            
        Returns:
            True if successful
        """
        try:
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as epub:
                # Add mimetype first (uncompressed as per EPUB spec)
                mimetype_path = epub_dir / "mimetype"
                if mimetype_path.exists():
                    epub.write(mimetype_path, "mimetype", compress_type=zipfile.ZIP_STORED)
                
                # Add all other files
                for root, dirs, files in os.walk(epub_dir):
                    for file in files:
                        if file != "mimetype":  # Skip mimetype as we already added it
                            file_path = Path(root) / file
                            arcname = file_path.relative_to(epub_dir)
                            epub.write(file_path, arcname)
            
            logger.success(f"Created EPUB: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create EPUB: {e}")
            return False
