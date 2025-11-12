"""
Semantic Enrichment Pipeline for EPUB HTML

Implements the complete pipeline:
1. Playwright: Analyze CSS styles and convert visual headings to semantic <h2>, <h3> tags
2. Trafilatura: Clean and linearize HTML
3. Markdownify: Convert to Markdown (handled by custom_converters.py)

This module focuses on Stage 1 (Playwright) and Stage 2 (Trafilatura).
"""

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import trafilatura
from loguru import logger
from typing import Optional
import re


class SemanticEnricher:
    """
    Enriches HTML with semantic headings based on visual styles.

    Uses Playwright to analyze computed styles and inject semantic <h1>-<h6> tags.
    """

    # JavaScript to inject semantic headings based on visual styles
    SEMANTIC_INJECTION_SCRIPT = """
    (() => {
        // Configuration: thresholds for heading detection
        const config = {
            // Font size thresholds (in pixels)
            h2MinSize: 18,
            h3MinSize: 16,
            h4MinSize: 14,

            // Font weight threshold
            boldMinWeight: 600,

            // Classes that indicate headings
            headingClasses: ['part-n', 'part-tit', 'chapter-title', 'section-title', 'titre'],

            // Tags to check
            checkTags: ['p', 'div', 'span']
        };

        let headingsFound = 0;

        // Helper: Get computed font size in pixels
        function getFontSize(element) {
            const computed = window.getComputedStyle(element);
            return parseFloat(computed.fontSize);
        }

        // Helper: Get computed font weight
        function getFontWeight(element) {
            const computed = window.getComputedStyle(element);
            const weight = computed.fontWeight;
            return weight === 'bold' ? 700 : parseInt(weight) || 400;
        }

        // Helper: Check if element has heading-like class
        function hasHeadingClass(element) {
            if (!element.className) return false;
            const classes = element.className.toLowerCase().split(/\s+/);
            return config.headingClasses.some(hc => classes.some(c => c.includes(hc)));
        }

        // Helper: Determine heading level based on style
        function determineHeadingLevel(element) {
            const fontSize = getFontSize(element);
            const fontWeight = getFontWeight(element);
            const isBold = fontWeight >= config.boldMinWeight;
            const hasClass = hasHeadingClass(element);

            // Special handling for known classes
            if (hasClass) {
                const classes = element.className.toLowerCase();
                if (classes.includes('part')) return 2;  // Parts are h2
                if (classes.includes('chapter')) return 3; // Chapters are h3
                if (classes.includes('section')) return 4; // Sections are h4
            }

            // Font size based detection
            if (fontSize >= config.h2MinSize) {
                return isBold ? 2 : 3;
            } else if (fontSize >= config.h3MinSize) {
                return isBold ? 3 : 4;
            } else if (fontSize >= config.h4MinSize && isBold) {
                return 4;
            }

            return null; // Not a heading
        }

        // Helper: Check if element looks like a heading
        function isVisualHeading(element) {
            // Skip if already a heading
            if (/^H[1-6]$/.test(element.tagName)) return false;

            // Skip if too long (headings are usually short)
            const text = element.textContent.trim();
            if (text.length > 200) return false;
            if (text.length === 0) return false;

            // Check if has heading class
            if (hasHeadingClass(element)) return true;

            // Check font size and weight
            const fontSize = getFontSize(element);
            const fontWeight = getFontWeight(element);

            return fontSize >= config.h4MinSize && fontWeight >= config.boldMinWeight;
        }

        // Main: Process all elements
        const elements = document.querySelectorAll(config.checkTags.join(','));

        elements.forEach(element => {
            if (isVisualHeading(element)) {
                const level = determineHeadingLevel(element);
                if (level) {
                    // Create new heading element
                    const heading = document.createElement('h' + level);
                    heading.innerHTML = element.innerHTML;

                    // Copy attributes
                    Array.from(element.attributes).forEach(attr => {
                        heading.setAttribute(attr.name, attr.value);
                    });

                    // Replace original element
                    element.parentNode.replaceChild(heading, element);
                    headingsFound++;
                }
            }
        });

        return {
            headingsFound: headingsFound,
            success: true
        };
    })();
    """

    def __init__(self, headless: bool = True):
        """
        Initialize semantic enricher.

        Args:
            headless: Run browser in headless mode
        """
        self.headless = headless
        self.playwright = None
        self.browser = None

    def __enter__(self):
        """Context manager entry."""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def enrich_html(self, html_content: str, base_url: str = "about:blank") -> str:
        """
        Enrich HTML with semantic headings based on visual styles.

        Args:
            html_content: Raw HTML content
            base_url: Base URL for resolving relative paths

        Returns:
            Semantically enriched HTML
        """
        if not self.browser:
            raise RuntimeError("SemanticEnricher must be used as context manager")

        # Create a new page
        page = self.browser.new_page()

        try:
            # Set content
            page.set_content(html_content, wait_until="domcontentloaded")

            # Execute semantic injection script
            result = page.evaluate(self.SEMANTIC_INJECTION_SCRIPT)

            if result['success']:
                logger.debug(f"Semantic enrichment: {result['headingsFound']} headings injected")
            else:
                logger.warning("Semantic enrichment failed")

            # Get enriched HTML
            enriched_html = page.content()

            return enriched_html

        finally:
            page.close()


def process_with_trafilatura(html_content: str, favor_recall: bool = True) -> Optional[str]:
    """
    Process HTML with Trafilatura for cleaning and linearization.

    Args:
        html_content: HTML content (ideally semantically enriched)
        favor_recall: Favor completeness over precision

    Returns:
        Cleaned HTML or None if extraction failed
    """
    try:
        # Trafilatura extraction
        # output_format='html' keeps HTML tags for further processing
        extracted = trafilatura.extract(
            html_content,
            output_format='html',
            include_comments=False,
            include_tables=True,
            include_images=True,
            include_links=True,
            favor_recall=favor_recall,  # Prefer completeness
            deduplicate=True,
        )

        if extracted:
            logger.debug(f"Trafilatura extracted {len(extracted)} chars")
            return extracted
        else:
            logger.warning("Trafilatura extraction returned None, using original HTML")
            return html_content

    except Exception as e:
        logger.error(f"Trafilatura processing failed: {e}")
        return html_content


def semantic_pipeline(
    html_content: str,
    use_playwright: bool = True,
    use_trafilatura: bool = True,
    enricher: Optional[SemanticEnricher] = None
) -> str:
    """
    Complete semantic enrichment pipeline.

    Stage 1 (Playwright): Inject semantic headings based on visual styles
    Stage 2 (Trafilatura): Clean and linearize HTML

    Args:
        html_content: Raw HTML
        use_playwright: Enable Playwright semantic enrichment
        use_trafilatura: Enable Trafilatura cleaning
        enricher: Reusable SemanticEnricher instance (for batch processing)

    Returns:
        Processed HTML ready for markdown conversion
    """
    result = html_content

    # Stage 1: Playwright semantic enrichment
    if use_playwright:
        if enricher:
            # Reuse existing enricher
            result = enricher.enrich_html(result)
        else:
            # Create temporary enricher
            with SemanticEnricher() as temp_enricher:
                result = temp_enricher.enrich_html(result)

    # Stage 2: Trafilatura cleaning
    if use_trafilatura:
        result = process_with_trafilatura(result)

    return result
