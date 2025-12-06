"""
CSS Matching Oracle using cssselect2

Provides ground-truth CSS selector matching for verified DOM transformations.
Uses element-only index-path keys for stable element identification across subtree replacements.
"""

from typing import Dict, Set, FrozenSet, List, Optional, Tuple
from dataclasses import dataclass
from lxml import etree
import tinycss2
import cssselect2
from loguru import logger


# XHTML namespace
XHTML_NS = "http://www.w3.org/1999/xhtml"
# EPUB OPS namespace
EPUB_NS = "http://www.idpf.org/2007/ops"
# Default namespaces for CSS selector matching
DEFAULT_NAMESPACES = {None: XHTML_NS, 'epub': EPUB_NS}


# Type alias for index path (stable element identifier, element-only indexing)
IndexPath = Tuple[int, ...]


@dataclass
class SelectorPayload:
    """Payload attached to each compiled selector."""
    selector_id: int
    selector_text: str


def _element_index(parent: etree._Element, child: etree._Element) -> int:
    """
    Get element-only index of child within parent.

    Ignores comments, processing instructions, etc.
    Only counts actual element nodes.
    """
    i = 0
    for node in parent:
        if isinstance(node.tag, str):  # Element node
            if node is child:
                return i
            i += 1
    raise ValueError(f"Child not found in parent")


def _get_element_children(parent: etree._Element) -> List[etree._Element]:
    """Get only element children (skip comments, PI, etc.)."""
    return [n for n in parent if isinstance(n.tag, str)]


class StylesheetOracle:
    """
    Parses CSS and provides ground-truth selector matching via cssselect2.
    """

    def __init__(self, css_content: str = ""):
        self.css_content = css_content
        self.matcher = cssselect2.Matcher()
        self.selector_count = 0
        self.failed_selectors: List[str] = []
        self.id_to_selector: Dict[int, str] = {}  # For debug: selector_id -> selector_text
        self.is_ready = False
        # Namespaces for CSS selector matching (can be extended by @namespace rules)
        self.namespaces: Dict[Optional[str], str] = dict(DEFAULT_NAMESPACES)

        if css_content:
            self._parse_and_compile()

    def _parse_and_compile(self) -> None:
        """Parse CSS with tinycss2, compile selectors with cssselect2."""
        try:
            rules = tinycss2.parse_stylesheet(
                self.css_content,
                skip_whitespace=True,
                skip_comments=True
            )
            # First pass: extract @namespace rules
            self._extract_namespaces(rules)
            # Second pass: compile selectors
            self._process_rules(rules)
            self.is_ready = True
            logger.debug(
                f"StylesheetOracle: {self.selector_count} selectors, "
                f"{len(self.failed_selectors)} failed"
            )
        except Exception as e:
            logger.warning(f"Failed to parse CSS: {e}")
            self.is_ready = False

    def _extract_namespaces(self, rules) -> None:
        """Extract @namespace rules and add to namespaces dict."""
        for rule in rules:
            if rule.type == 'at-rule' and rule.lower_at_keyword == 'namespace':
                # Parse @namespace prelude: either "url" or prefix "url"
                # URL can be: string token, url token (unquoted), or url() function block
                prelude = [t for t in rule.prelude
                           if t.type not in ('whitespace', 'comment')]

                def _get_url(tok) -> Optional[str]:
                    """Extract URL from string, url token, or url() function."""
                    if tok.type == 'string':
                        return tok.value
                    if tok.type == 'url':
                        # Unquoted: url(http://...)
                        return tok.value
                    if tok.type == 'function' and tok.lower_name == 'url':
                        # Quoted: url("http://...")
                        # Extract string from function arguments
                        for arg in tok.arguments:
                            if arg.type == 'string':
                                return arg.value
                    return None

                if len(prelude) == 1:
                    # Default namespace: @namespace "url"; or @namespace url("...");
                    url = _get_url(prelude[0])
                    if url:
                        self.namespaces[None] = url
                elif len(prelude) == 2 and prelude[0].type == 'ident':
                    # Prefixed namespace: @namespace prefix "url"; or @namespace prefix url("...");
                    prefix = prelude[0].value
                    url = _get_url(prelude[1])
                    if url:
                        self.namespaces[prefix] = url

    def _process_rules(self, rules) -> None:
        """Recursively process CSS rules, including @media/@supports."""
        for rule in rules:
            if rule.type == 'qualified-rule':
                self._compile_rule(rule)
            elif rule.type == 'at-rule' and rule.content:
                nested = tinycss2.parse_rule_list(rule.content, skip_whitespace=True)
                self._process_rules(nested)

    def _compile_rule(self, rule) -> None:
        """Compile a single CSS rule's selectors."""
        # Use serialized text for stability across cssselect2 versions
        selector_text = tinycss2.serialize(rule.prelude).strip()

        try:
            compiled_selectors = cssselect2.compile_selector_list(
                selector_text,
                namespaces=self.namespaces
            )

            for compiled in compiled_selectors:
                payload = SelectorPayload(
                    selector_id=self.selector_count,
                    selector_text=selector_text,
                )
                self.matcher.add_selector(compiled, payload)
                self.id_to_selector[self.selector_count] = selector_text
                self.selector_count += 1

        except cssselect2.SelectorError as e:
            self.failed_selectors.append(selector_text)
            logger.debug(f"Failed to compile selector '{selector_text}': {e}")

    def get_selector_text(self, selector_id: int) -> str:
        """Get selector text for a given selector ID (for debug logging)."""
        text = self.id_to_selector.get(selector_id, f"#{selector_id}")
        # Truncate long selectors for readability
        if len(text) > 40:
            return text[:37] + "..."
        return text

    def compute_region_signatures(
        self,
        region_root: etree._Element,
        wrapper_cache: Optional[Dict[etree._Element, 'cssselect2.ElementWrapper']] = None
    ) -> Dict[IndexPath, FrozenSet[int]]:
        """
        Compute selector signatures for a region subtree.

        Uses element-only index-path keys for stable identification across subtree replacements.

        Args:
            region_root: Root of the region to analyze
            wrapper_cache: Optional pre-built wrapper map for performance

        Returns:
            Dict mapping index paths to selector signatures
        """
        signatures: Dict[IndexPath, FrozenSet[int]] = {}

        if not self.is_ready:
            return signatures

        # Build or use wrapper map
        if wrapper_cache is None:
            # We need to wrap from the document root for proper selector matching
            doc_root = region_root
            while doc_root.getparent() is not None:
                doc_root = doc_root.getparent()

            wrapped_root = cssselect2.ElementWrapper.from_xml_root(doc_root)

            wrapper_map: Dict[etree._Element, cssselect2.ElementWrapper] = {}
            for wrapper in wrapped_root.iter_subtree():
                wrapper_map[wrapper.etree_element] = wrapper
        else:
            wrapper_map = wrapper_cache

        # Only compute signatures for element nodes in the region
        for elem in region_root.iter():
            if not isinstance(elem.tag, str):
                continue  # Skip comments, PI, etc.

            if elem not in wrapper_map:
                continue

            wrapper = wrapper_map[elem]
            matches = self.matcher.match(wrapper)
            sig = frozenset(match[3].selector_id for match in matches)

            # Compute relative path from region_root using element-only indexing
            rel_path = self._get_relative_path(elem, region_root)
            signatures[rel_path] = sig

        return signatures

    def _get_path_from_root(
        self,
        elem: etree._Element,
        root: etree._Element
    ) -> IndexPath:
        """Get element-only index path from root to element."""
        path = []
        current = elem
        while current is not root and current is not None:
            parent = current.getparent()
            if parent is None:
                break
            idx = _element_index(parent, current)
            path.append(idx)
            current = parent
        return tuple(reversed(path))

    def _get_relative_path(
        self,
        elem: etree._Element,
        region_root: etree._Element
    ) -> IndexPath:
        """Get element-only index path relative to region_root."""
        if elem is region_root:
            return ()
        return self._get_path_from_root(elem, region_root)

    def get_element_at_path(
        self,
        region_root: etree._Element,
        path: IndexPath
    ) -> Optional[etree._Element]:
        """Navigate to element at given path from region_root (element-only indexing)."""
        current = region_root
        for idx in path:
            children = _get_element_children(current)
            if idx >= len(children):
                return None
            current = children[idx]
        return current


class RegionSnapshot:
    """
    Captures selector match state for a region subtree.

    Uses element-only index-path keys for stable comparison after subtree replacements.
    """

    def __init__(
        self,
        oracle: StylesheetOracle,
        region_root: etree._Element,
        wrapper_cache: Optional[Dict[etree._Element, 'cssselect2.ElementWrapper']] = None
    ):
        """
        Create a snapshot of selector matches for region.

        Args:
            oracle: The StylesheetOracle to use
            region_root: Root of the region to snapshot
            wrapper_cache: Optional pre-built wrapper map for performance
        """
        self.oracle = oracle
        self.region_root = region_root
        self._signatures: Dict[IndexPath, FrozenSet[int]] = {}

        if oracle.is_ready:
            self._signatures = oracle.compute_region_signatures(region_root, wrapper_cache)

    def signature(self, path: IndexPath) -> FrozenSet[int]:
        """Get signature at path."""
        return self._signatures.get(path, frozenset())

    def all_signatures(self) -> Dict[IndexPath, FrozenSet[int]]:
        """Get all signatures."""
        return dict(self._signatures)

    def paths(self) -> Set[IndexPath]:
        """Get all paths in this snapshot."""
        return set(self._signatures.keys())

    @staticmethod
    def create_fresh(
        oracle: StylesheetOracle,
        region_root: etree._Element,
        wrapper_cache: Optional[Dict[etree._Element, 'cssselect2.ElementWrapper']] = None
    ) -> 'RegionSnapshot':
        """Create a fresh snapshot (after DOM modification)."""
        return RegionSnapshot(oracle, region_root, wrapper_cache)


def backup_subtree(elem: etree._Element) -> bytes:
    """Backup a subtree for rollback (excludes tail to avoid parse errors)."""
    # with_tail=False prevents "Extra content at end of document" when elem.tail exists
    return etree.tostring(elem, with_tail=False)


def restore_subtree(
    parent: etree._Element,
    index: int,
    backup: bytes,
    old_elem: etree._Element
) -> etree._Element:
    """
    Restore a subtree from backup.

    Args:
        parent: Parent element
        index: Index where the element should be
        backup: Serialized backup (without tail, from backup_subtree)
        old_elem: The old element to remove (if still attached)

    Returns:
        The restored element
    """
    # Save tail before removing (backup excludes tail)
    original_tail = old_elem.tail

    # Remove old element if still in tree
    if old_elem.getparent() is parent:
        parent.remove(old_elem)

    # Parse and insert restored element
    restored = etree.fromstring(backup)
    restored.tail = original_tail  # Restore the tail
    parent.insert(index, restored)
    return restored


def restore_in_place(elem: etree._Element, backup: bytes) -> None:
    """
    Restore an element's contents in-place from backup.

    Used when elem is the root and cannot be replaced via parent.
    Clears elem and copies all content from the parsed backup.

    Args:
        elem: The element to restore (will be modified in-place)
        backup: Serialized backup bytes (without tail, from backup_subtree)
    """
    # Save original tail (backup excludes tail)
    original_tail = elem.tail

    restored = etree.fromstring(backup)

    # Clear current element
    elem.clear()  # Removes all children and attributes

    # Copy attributes
    elem.attrib.update(restored.attrib)

    # Copy text
    elem.text = restored.text

    # Copy children
    for child in list(restored):
        elem.append(child)

    # Restore original tail (not from backup since backup excludes tail)
    elem.tail = original_tail
