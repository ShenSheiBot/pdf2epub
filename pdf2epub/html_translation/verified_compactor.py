"""
Verified HTML DOM Compactor

Uses cssselect2 oracle to verify each transformation preserves CSS selector
matching semantics. Uses subtree backup/restore for correct rollback and
element-only index-path keys for stable comparison across DOM modifications.

Invariants:
- Phase1 (unwrap): DISABLED - low value, complex path mapping needed
- Phase2 (merge): After merge, all elements except parent (deleted) and child
  (at new position) must have exactly the same signatures at their new paths.
  Child constraint: child_before ⊆ child_after ⊆ (parent_before ∪ child_before)
"""

from typing import Tuple, Dict, FrozenSet
from lxml import etree
import re
import cssselect2
from loguru import logger

from .oracle import (
    StylesheetOracle,
    RegionSnapshot,
    IndexPath,
    backup_subtree,
    restore_subtree,
    restore_in_place,
    XHTML_NS,
    _get_element_children,
)

# EPUB OPS namespace (for epub:type etc.)
EPUB_NS = "http://www.idpf.org/2007/ops"


def localname(tag: str) -> str:
    """Extract local name from potentially namespaced tag."""
    if tag.startswith('{'):
        return tag.split('}', 1)[1]
    return tag


def _build_wrapper_cache(doc_root: etree._Element) -> Dict[etree._Element, cssselect2.ElementWrapper]:
    """Build wrapper map for entire document (for reuse across snapshots)."""
    wrapped_root = cssselect2.ElementWrapper.from_xml_root(doc_root)
    wrapper_map: Dict[etree._Element, cssselect2.ElementWrapper] = {}
    for wrapper in wrapped_root.iter_subtree():
        wrapper_map[wrapper.etree_element] = wrapper
    return wrapper_map


class VerifiedCompactor:
    """
    CSS-verified HTML compactor.

    Every transformation is verified against the CSS oracle before being committed.
    Uses subtree backup/restore for correct rollback.
    """

    # Tags that can potentially be unwrapped/merged
    MERGEABLE_TAGS = {'span', 'div'}

    # For merge, id/name can be migrated (so not in this set)
    # Also allow xml:lang, xml:space, epub:type migration
    UNMERGEABLE_ATTRS_MERGE = {
        'href', 'src',
        'onclick', 'onload',
        'data-', 'aria-', 'role',
    }

    # Namespaced attributes that CAN be migrated during merge
    MIGRATABLE_NS_ATTRS = {
        '{http://www.w3.org/XML/1998/namespace}lang',  # xml:lang
        '{http://www.w3.org/XML/1998/namespace}space',  # xml:space
        '{http://www.idpf.org/2007/ops}type',  # epub:type
    }

    def __init__(self, css_content: str = "", conservative_mode: bool = False):
        """
        Initialize compactor with CSS content.

        Args:
            css_content: The CSS stylesheet content
            conservative_mode: If True, disable all transformations when oracle
                             has failed selectors. Default False (fail-open).
        """
        self.oracle = StylesheetOracle(css_content)
        self.conservative_mode = conservative_mode
        self._warned_partial_oracle = False  # Only warn once per instance

    def _has_unmergeable_attrs(self, element: etree._Element) -> bool:
        """Check if element has attributes that prevent merge."""
        for attr in element.attrib:
            # Handle namespaced attributes
            if attr.startswith('{'):
                # Allow known migratable ns attrs
                if attr in self.MIGRATABLE_NS_ATTRS:
                    continue
                # Unknown namespaced attr blocks transformation
                return True

            # Handle prefix notation (xml:lang etc) - shouldn't happen with lxml
            if ':' in attr:
                return True

            attr_local = localname(attr)
            if attr_local in self.UNMERGEABLE_ATTRS_MERGE:
                return True
            # Prefix matches (data-*, aria-*)
            for prefix in self.UNMERGEABLE_ATTRS_MERGE:
                if prefix.endswith('-') and attr_local.startswith(prefix):
                    return True
        return False

    def _can_attempt_merge(
        self,
        parent: etree._Element,
        child: etree._Element
    ) -> bool:
        """
        Check structural prerequisites for merge (not CSS verification).

        CSS verification is done separately via oracle.
        """
        parent_tag = localname(parent.tag)
        child_tag = localname(child.tag)

        # Must be mergeable tags
        if parent_tag not in self.MERGEABLE_TAGS:
            return False
        if child_tag not in self.MERGEABLE_TAGS:
            return False

        # Same tag only (div→div, span→span)
        # Mixing block/inline changes layout even without CSS
        if parent_tag != child_tag:
            return False

        # Parent must have ONLY element children (no comment/PI)
        # Otherwise merge would "swallow" those non-element nodes
        for node in parent:
            if not isinstance(node.tag, str):
                return False

        # Reject non-whitespace text that would change visible content
        # Whitespace-only text is allowed - oracle verification will catch
        # any :empty breakage via signature comparison
        if parent.text and parent.text.strip():
            return False
        if child.tail and child.tail.strip():
            return False

        # Check unmergeable attributes
        if self._has_unmergeable_attrs(parent):
            return False
        if self._has_unmergeable_attrs(child):
            return False

        # id/name conflict check: can migrate if only one has it
        parent_id = parent.get('id') or parent.get('name')
        child_id = child.get('id') or child.get('name')
        if parent_id and child_id:
            return False

        # Check for conflicting migratable ns attrs
        for ns_attr in self.MIGRATABLE_NS_ATTRS:
            if parent.get(ns_attr) and child.get(ns_attr):
                if parent.get(ns_attr) != child.get(ns_attr):
                    return False  # Conflicting values

        return True

    def _do_merge(
        self,
        parent: etree._Element,
        child: etree._Element
    ) -> bool:
        """
        Actually perform the merge operation.

        Merges parent's attributes into child, then replaces parent with child.
        Returns True if successful.
        """
        grandparent = parent.getparent()
        if grandparent is None:
            return False

        # Use RAW index for insert (includes comment/PI)
        index = list(grandparent).index(parent)

        # Merge attributes: class, style, others
        self._merge_attrs(child, parent)

        # Handle parent.text → prepend to child.text
        if parent.text:
            child.text = (parent.text or '') + (child.text or '')

        # Handle child.tail: preserve original + parent's tail
        orig_tail = child.tail or ''
        parent_tail = parent.tail or ''
        child.tail = orig_tail + parent_tail

        # Replace parent with child
        grandparent.remove(parent)
        grandparent.insert(index, child)

        return True

    def _merge_attrs(
        self,
        target: etree._Element,
        source: etree._Element
    ) -> None:
        """
        Merge attributes from source (parent) into target (child).

        - class: combine, deduplicate
        - style: source first, then target (target/inner overrides)
        - id/name: migrate if target doesn't have
        - namespaced attrs: migrate if allowed and no conflict
        - other: copy if not present
        """
        # Merge class
        target_classes = target.get('class', '').split()
        source_classes = source.get('class', '').split()
        combined = []
        seen = set()
        for cls in target_classes + source_classes:
            if cls and cls not in seen:
                combined.append(cls)
                seen.add(cls)
        if combined:
            target.set('class', ' '.join(combined))
        elif 'class' in target.attrib:
            del target.attrib['class']

        # Merge style (source is outer, target is inner - inner wins)
        target_style = target.get('style', '').strip()
        source_style = source.get('style', '').strip()
        if source_style or target_style:
            if source_style and target_style:
                if not source_style.endswith(';'):
                    source_style += ';'
                combined_style = source_style + ' ' + target_style
            else:
                combined_style = source_style or target_style
            target.set('style', combined_style)

        # Merge other attributes (including id/name migration)
        for attr, value in source.attrib.items():
            attr_local = localname(attr)
            if attr_local not in ('class', 'style') and attr not in target.attrib:
                target.set(attr, value)

    def _format_sig_diff(self, before: FrozenSet[int], after: FrozenSet[int]) -> str:
        """Format signature difference for debug logging."""
        added = after - before
        removed = before - after
        parts = []
        if added:
            added_text = ', '.join(self.oracle.get_selector_text(sid) for sid in sorted(added))
            parts.append(f"+[{added_text}]")
        if removed:
            removed_text = ', '.join(self.oracle.get_selector_text(sid) for sid in sorted(removed))
            parts.append(f"-[{removed_text}]")
        return ' '.join(parts) if parts else "(no change)"

    def _verify_merge(
        self,
        before_sigs: Dict[IndexPath, FrozenSet[int]],
        after_sigs: Dict[IndexPath, FrozenSet[int]],
        parent_path: IndexPath,
        child_path_before: IndexPath
    ) -> bool:
        """
        Verify that merging doesn't break CSS selector matches.

        Strict invariant: All elements except:
        - parent (deleted)
        - child (moved to parent's position)

        must have EXACTLY the same signature at their corresponding path.

        Child constraint (conservative):
        - child_before ⊆ child_after ⊆ (parent_before ∪ child_before)
        - Child must keep all its original matches AND not gain new ones
        - This prevents both "losing styling" and "gaining unexpected styling"

        The path transformation for merge is:
        - parent_path -> deleted
        - child_path (= parent_path + (0,)) -> parent_path
        - child's descendants: prefix parent_path+(0,) -> parent_path
        - siblings after parent in grandparent: unchanged
        """
        # Get parent and child before signatures
        parent_sig_before = before_sigs.get(parent_path, frozenset())
        child_sig_before = before_sigs.get(child_path_before, frozenset())
        allowed_child_sig = parent_sig_before | child_sig_before

        # Paths that are deleted
        deleted_paths = {parent_path}

        # Build expected path mapping for child's descendants
        child_prefix = child_path_before  # = parent_path + (0,)
        new_child_prefix = parent_path

        # Check child's new signature (at parent_path after merge)
        child_sig_after = after_sigs.get(parent_path, frozenset())

        # Conservative constraint: child_before ⊆ child_after ⊆ allowed
        # Child must not lose its original matches
        if not child_sig_before <= child_sig_after:
            lost = child_sig_before - child_sig_after
            lost_text = ', '.join(self.oracle.get_selector_text(sid) for sid in sorted(lost))
            logger.debug(f"Merge verification failed: child lost matches. Lost: [{lost_text}]")
            return False

        # Child must not gain unexpected matches
        if not child_sig_after <= allowed_child_sig:
            gained = child_sig_after - allowed_child_sig
            gained_text = ', '.join(self.oracle.get_selector_text(sid) for sid in sorted(gained))
            logger.debug(f"Merge verification failed: child gained new matches. Gained: [{gained_text}]")
            return False

        # Check all before paths (except parent and child itself)
        for before_path, before_sig in before_sigs.items():
            if before_path in deleted_paths:
                continue  # parent is deleted, skip

            if before_path == child_path_before:
                # Child itself - already checked above
                continue

            # Compute expected after_path
            if len(before_path) > len(child_prefix) and before_path[:len(child_prefix)] == child_prefix:
                # Descendant of child - path prefix changes
                subpath = before_path[len(child_prefix):]
                after_path = new_child_prefix + subpath
            else:
                # Other elements - path unchanged
                after_path = before_path

            # Check signature
            after_sig = after_sigs.get(after_path, frozenset())
            if before_sig != after_sig:
                logger.debug(
                    f"Merge verification failed: path {before_path}->{after_path} "
                    f"sig changed. Diff: {self._format_sig_diff(before_sig, after_sig)}"
                )
                return False

        # Also check that no unexpected new paths appeared in after
        expected_after_paths = set()
        for before_path in before_sigs:
            if before_path in deleted_paths:
                continue
            if before_path == child_path_before:
                expected_after_paths.add(parent_path)  # child moves here
            elif len(before_path) > len(child_prefix) and before_path[:len(child_prefix)] == child_prefix:
                subpath = before_path[len(child_prefix):]
                expected_after_paths.add(new_child_prefix + subpath)
            else:
                expected_after_paths.add(before_path)

        for after_path in after_sigs:
            if after_path not in expected_after_paths:
                if after_path == parent_path:
                    continue  # This is expected - child moved here
                logger.debug(f"Merge verification failed: unexpected new path {after_path}")
                return False

        return True

    def _phase2_merge(self, root: etree._Element) -> int:
        """
        Phase 2: Merge single-child nesting chains.

        Uses oracle verification with subtree backup/restore for rollback.
        Returns number of merges performed.
        """
        if not self.oracle.is_ready:
            return 0

        if self.conservative_mode and self.oracle.failed_selectors:
            logger.debug("Phase2 disabled: oracle has failed selectors in conservative mode")
            return 0

        # Warn once about partial oracle (fail-open mode)
        if self.oracle.failed_selectors and not self._warned_partial_oracle:
            logger.warning(
                f"Partial oracle: {len(self.oracle.failed_selectors)} selectors failed to compile, "
                f"proceeding with {self.oracle.selector_count} valid selectors"
            )
            self._warned_partial_oracle = True

        count = 0
        # Track paths of candidates that failed verification in CURRENT tree state
        # Cleared after each successful merge (tree structure changes)
        failed_paths: set[IndexPath] = set()

        while True:
            restart = False

            # Build wrapper cache for this scan iteration (performance optimization)
            doc_root = root
            while doc_root.getparent() is not None:
                doc_root = doc_root.getparent()
            wrapper_cache = _build_wrapper_cache(doc_root)

            for element in list(root.iter()):
                if element is root:
                    continue
                if not isinstance(element.tag, str):
                    continue

                # Must have exactly one element child (and no non-element children)
                element_children = _get_element_children(element)
                if len(element_children) != 1:
                    continue

                child = element_children[0]

                if not self._can_attempt_merge(element, child):
                    continue

                grandparent = element.getparent()
                if grandparent is None:
                    continue

                # Compute stable path from root (for failed_paths tracking)
                candidate_path_from_root = self.oracle._get_relative_path(element, root)
                if candidate_path_from_root in failed_paths:
                    continue  # Skip previously failed candidate in current tree state

                # Compute paths before modification (relative to grandparent for verification)
                parent_path = self.oracle._get_relative_path(element, grandparent)
                child_path = parent_path + (0,)

                # Take before snapshot (use wrapper cache)
                before_snap = RegionSnapshot(self.oracle, grandparent, wrapper_cache)
                before_sigs = before_snap.all_signatures()

                # Backup for rollback
                gp_backup = backup_subtree(grandparent)

                # Check if grandparent is root (no great-grandparent)
                great_grandparent = grandparent.getparent()
                is_root_level = (great_grandparent is None)

                # Use RAW index for restore
                gp_index_raw = None
                if not is_root_level:
                    gp_index_raw = list(great_grandparent).index(grandparent)

                # Perform the merge
                if not self._do_merge(element, child):
                    continue

                # Build fresh wrapper cache for after snapshot (DOM changed)
                doc_root_after = grandparent
                while doc_root_after.getparent() is not None:
                    doc_root_after = doc_root_after.getparent()
                wrapper_cache_after = _build_wrapper_cache(doc_root_after)

                # Take after snapshot with fresh cache
                after_snap = RegionSnapshot(self.oracle, grandparent, wrapper_cache_after)
                after_sigs = after_snap.all_signatures()

                # Verify with path-aligned comparison
                if self._verify_merge(before_sigs, after_sigs, parent_path, child_path):
                    count += 1
                    # Tree structure changed - clear failed_paths (same path may now be different element)
                    failed_paths.clear()
                    restart = True
                    break  # Restart with fresh iteration
                else:
                    # Rollback
                    if is_root_level:
                        restore_in_place(grandparent, gp_backup)
                    else:
                        restore_subtree(great_grandparent, gp_index_raw, gp_backup, grandparent)
                    # Mark this candidate as failed in current tree state
                    failed_paths.add(candidate_path_from_root)
                    # Restart scan with fresh element list (rollback created new objects)
                    restart = True
                    break

            if restart:
                continue
            break

        return count

    def compact(self, html: str) -> str:
        """
        Compact HTML using verified transformations.

        Args:
            html: HTML string to compact

        Returns:
            Compacted HTML string
        """
        if not html or not html.strip():
            return html

        try:
            wrapped = f'<div xmlns="{XHTML_NS}" xmlns:epub="{EPUB_NS}">{html}</div>'
            root = etree.fromstring(wrapped.encode('utf-8'))
        except etree.XMLSyntaxError as e:
            logger.warning(f"Invalid XHTML, skipping compaction: {e}")
            return html
        except Exception as e:
            logger.warning(f"Failed to parse HTML: {e}")
            return html

        # Apply Phase 2 only (Phase 1 disabled - low value)
        try:
            self._phase2_merge(root)
        except Exception as e:
            logger.warning(f"Compaction failed, returning original: {e}")
            return html

        # Serialize
        parts = []
        if root.text:
            parts.append(root.text)
        for child in root:
            child_str = etree.tostring(child, encoding='unicode')
            # Strip namespace declarations added by wrapper
            child_str = re.sub(r'\s+xmlns="http://www\.w3\.org/1999/xhtml"', '', child_str)
            child_str = re.sub(r'\s+xmlns:epub="http://www\.idpf\.org/2007/ops"', '', child_str)
            parts.append(child_str)

        return ''.join(parts)

    def compact_with_stats(self, html: str) -> Tuple[str, Dict]:
        """
        Compact HTML and return statistics.
        """
        if not html or not html.strip():
            return html, {'phase1': 0, 'phase2': 0, 'total': 0}

        try:
            wrapped = f'<div xmlns="{XHTML_NS}" xmlns:epub="{EPUB_NS}">{html}</div>'
            root = etree.fromstring(wrapped.encode('utf-8'))
        except etree.XMLSyntaxError as e:
            logger.warning(f"Invalid XHTML, skipping compaction: {e}")
            return html, {'error': str(e), 'skipped': True}
        except Exception as e:
            logger.warning(f"Failed to parse HTML: {e}")
            return html, {'error': str(e)}

        before_count = len([el for el in root.iter() if el is not root and isinstance(el.tag, str)])

        phase1_count = 0
        phase2_count = self._phase2_merge(root)

        after_count = len([el for el in root.iter() if el is not root and isinstance(el.tag, str)])

        # Serialize
        parts = []
        if root.text:
            parts.append(root.text)
        for child in root:
            child_str = etree.tostring(child, encoding='unicode')
            # Strip namespace declarations added by wrapper
            child_str = re.sub(r'\s+xmlns="http://www\.w3\.org/1999/xhtml"', '', child_str)
            child_str = re.sub(r'\s+xmlns:epub="http://www\.idpf\.org/2007/ops"', '', child_str)
            parts.append(child_str)

        result = ''.join(parts)

        stats = {
            'phase1': phase1_count,
            'phase2': phase2_count,
            'total': phase1_count + phase2_count,
            'elements_before': before_count,
            'elements_after': after_count,
            'reduction': before_count - after_count,
            'oracle_ready': self.oracle.is_ready,
            'selectors_compiled': self.oracle.selector_count,
            'selectors_failed': len(self.oracle.failed_selectors),
        }

        return result, stats
