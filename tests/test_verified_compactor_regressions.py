import pytest
from lxml import etree

# ⚠️ 改成你工程里真实的 import 路径：
# 例如：from pdf2epub.html_translation.oracle import ...
from pdf2epub.html_translation.oracle import (
    backup_subtree,
    restore_subtree,
    StylesheetOracle,
)
from pdf2epub.html_translation.verified_compactor import VerifiedCompactor, XHTML_NS


def test_restore_subtree_should_not_crash_when_region_root_has_nonwhitespace_tail_text():
    """
    现象：backup_subtree(elem) 默认会把 elem.tail 一起序列化。
    如果 elem.tail 是非空白字符（混合内容里完全可能出现），etree.fromstring(backup) 会直接 XMLSyntaxError。
    这个测试"期望不崩"，你的当前实现会崩 -> FAIL。
    """
    parent = etree.fromstring(
        f'<div xmlns="{XHTML_NS}"><section/>X<section/></div>'.encode("utf-8")
    )
    victim = parent[0]  # <section/>
    assert victim.tail == "X"  # 确认是非空白 tail

    b = backup_subtree(victim)

    # 期望 restore_subtree 能工作（不抛异常）
    # 你当前实现会在 etree.fromstring(b) 处爆：Extra content at the end of the document
    restore_subtree(parent, 0, b, victim)


def test_compact_with_stats_should_not_skip_fragment_with_epub_type():
    """
    现象：你把 fragment 包到 <div xmlns="xhtml">...</div> 里 strict parse。
    fragment 如果含 epub:type 之类前缀属性，而 wrapper 没声明 xmlns:epub，就会 XMLSyntaxError -> skipped=True。
    这个测试"期望不 skip"，当前实现会 skip -> FAIL。
    """
    css = ".a{color:red;}"
    comp = VerifiedCompactor(css, conservative_mode=True)

    frag = '<span epub:type="pagebreak" class="a">x</span>'
    out, stats = comp.compact_with_stats(frag)

    assert stats.get("skipped") is not True


def test_oracle_should_honor_css_namespace_rule_and_compile_epub_prefixed_selector():
    """
    现象：EPUB CSS 很常见 @namespace epub "..."; 然后用 *[epub|type~="..."]。
    oracle 不处理 @namespace 的话，cssselect2 会把 'epub' 视为未知前缀 -> 编译失败。
    这个测试"期望能编译"，当前实现会 failed_selectors 非空/selector_count=0 -> FAIL。
    """
    css = (
        '@namespace epub "http://www.idpf.org/2007/ops"; '
        '*[epub|type~="pagebreak"] { color: red; }'
    )
    oracle = StylesheetOracle(css)

    assert oracle.selector_count >= 1
    assert oracle.failed_selectors == []


def test_conservative_mode_should_not_disable_all_merges_due_to_one_failed_selector():
    """
    现象：只要 oracle.failed_selectors 非空，你 Phase2 直接 return 0。
    这会导致"哪怕有大量可证明安全的 merge，也全部被禁用"。
    这个测试构造一个 trivially mergeable span 链，期望 reduction>0；当前实现会 reduction==0 -> FAIL。
    """
    css = (
        ".a{color:red;}"
        # 这条故意触发 failed_selectors（未知 namespace prefix）
        '*[epub|type~="pagebreak"]{color:blue;}'
    )
    comp = VerifiedCompactor(css, conservative_mode=True)

    html = '<span class="a"><span>t</span></span>'
    out, stats = comp.compact_with_stats(html)

    # 期望至少做一次 merge
    assert stats["reduction"] > 0
