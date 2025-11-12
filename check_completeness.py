#!/usr/bin/env python3
"""
检查OCR、polish和翻译各阶段的字数完整性
用于确保处理过程中没有内容截断
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Tuple
import re


def load_config() -> Dict:
    """从config.yaml加载配置"""
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found")

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def sanitize_title(title: str) -> str:
    """清理书名，移除不适合作为文件名的字符"""
    sanitized = re.sub(r'[<>:"/\\|?*]', '', title)
    sanitized = sanitized.strip()
    return sanitized


def find_output_dir(title: str) -> Path:
    """根据书名查找输出目录"""
    output_base = Path("output")

    if not output_base.exists():
        raise FileNotFoundError("output directory not found")

    # 尝试多种可能的目录名格式
    possible_names = [
        title,
        sanitize_title(title),
        title.replace(' ', '_'),
        title.replace(' ', ' ')[:50],  # 截断长标题
    ]

    for dir_name in output_base.iterdir():
        if dir_name.is_dir():
            dir_name_str = dir_name.name
            for possible in possible_names:
                if possible.lower() in dir_name_str.lower() or dir_name_str.lower() in possible.lower():
                    return dir_name

    # 如果没找到，列出所有目录让用户选择
    print("Could not automatically find output directory. Available directories:")
    dirs = [d for d in output_base.iterdir() if d.is_dir()]
    for i, d in enumerate(dirs, 1):
        print(f"{i}. {d.name}")

    choice = input("Enter directory number: ")
    return dirs[int(choice) - 1]


def count_words(file_path: Path, is_chinese: bool = False) -> Tuple[int, int]:
    """
    统计文件字数
    返回: (字符数, 单词数/中文字数)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 字符数（不含空白）
        char_count = len(content.replace(' ', '').replace('\n', '').replace('\t', ''))

        if is_chinese:
            # 中文字数：统计中文字符
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
            return char_count, chinese_chars
        else:
            # 英文单词数
            words = re.findall(r'\b\w+\b', content)
            return char_count, len(words)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0, 0


def get_chapter_files(directory: Path) -> Dict[str, Path]:
    """获取目录中所有章节文件，返回 {文件名: 路径} 字典"""
    if not directory.exists():
        return {}

    files = {}
    for file_path in directory.glob("*.md"):
        if file_path.name not in ['front_matter.md', 'back_matter.md']:
            files[file_path.name] = file_path

    return files


def get_base_chapter_name(filename: str) -> str:
    """获取章节基础名称（去除.part后缀）"""
    return re.sub(r'\.part\d+\.md$', '.md', filename)


def analyze_directory(output_dir: Path, source_lang: str, target_lang: str):
    """分析输出目录中各阶段的文件完整性"""

    ocr_dir = output_dir / "ocr_markdown"
    polish_dir = output_dir / "polished_markdown"
    translated_dir = output_dir / "translated"

    is_chinese_target = target_lang.lower() in ['chinese', 'zh', '中文']

    # 获取所有章节文件
    ocr_files = get_chapter_files(ocr_dir)
    polish_files = get_chapter_files(polish_dir)
    translated_files = get_chapter_files(translated_dir)

    # 构建章节组：按照base name分组
    chapter_groups = {}
    for filename in set(list(ocr_files.keys()) + list(polish_files.keys()) + list(translated_files.keys())):
        base_name = get_base_chapter_name(filename)
        if base_name not in chapter_groups:
            chapter_groups[base_name] = {
                'ocr': [],
                'polish': [],
                'translated': []
            }

        # 根据文件来源分类
        if filename in ocr_files:
            chapter_groups[base_name]['ocr'].append(filename)
        if filename in polish_files:
            chapter_groups[base_name]['polish'].append(filename)
        if filename in translated_files:
            chapter_groups[base_name]['translated'].append(filename)

    # 获取所有章节名（排序）
    all_chapters = sorted(chapter_groups.keys())

    print(f"\n{'='*100}")
    print(f"📚 Book: {output_dir.name}")
    print(f"📂 Output Directory: {output_dir}")
    print(f"🌐 Source → Target: {source_lang} → {target_lang}")
    print(f"{'='*100}\n")

    # 统计数据
    stats = []
    warnings = []

    print(f"{'Chapter':<30} {'OCR':<20} {'Polish':<20} {'Translated':<20} {'Status':<15}")
    print(f"{'-'*30} {'-'*20} {'-'*20} {'-'*20} {'-'*15}")

    for base_chapter in all_chapters:
        group = chapter_groups[base_chapter]

        # 统计每个阶段的总字数（包括所有part）
        ocr_words_total = 0
        polish_words_total = 0
        trans_words_total = 0

        for ocr_file in group['ocr']:
            _, words = count_words(ocr_files[ocr_file], False)
            ocr_words_total += words

        for polish_file in group['polish']:
            _, words = count_words(polish_files[polish_file], False)
            polish_words_total += words

        for trans_file in group['translated']:
            _, words = count_words(translated_files[trans_file], is_chinese_target)
            trans_words_total += words

        # 状态判断
        status = "✅ OK"

        # 检查是否存在各阶段文件
        if not group['ocr']:
            status = "🔴 NO OCR"
            warnings.append(f"{base_chapter}: Missing OCR files")
        elif not group['polish']:
            status = "🔴 NO POLISH"
            warnings.append(f"{base_chapter}: Missing polish files")
        elif not group['translated']:
            status = "🔴 NO TRANS"
            warnings.append(f"{base_chapter}: Missing translation files")
        else:
            # 检查字数异常
            # OCR -> Polish: 预期减少10-30%（去除页眉页脚等）
            if polish_words_total > 0 and polish_words_total < ocr_words_total * 0.5:
                status = "🟡 POLISH SHORT"
                warnings.append(f"{base_chapter}: Polish suspiciously short ({polish_words_total} vs {ocr_words_total} words)")

            # Polish -> Translated: 中文预期是英文的0.4-0.8倍
            if is_chinese_target and trans_words_total > 0:
                expected_ratio_min = 0.3
                expected_ratio_max = 1.0
                actual_ratio = trans_words_total / polish_words_total if polish_words_total > 0 else 0

                if actual_ratio < expected_ratio_min:
                    status = "🔴 TRANS SHORT"
                    warnings.append(f"{base_chapter}: Translation suspiciously short (ratio: {actual_ratio:.2f})")
                elif actual_ratio > expected_ratio_max:
                    status = "🟡 TRANS LONG"

            # 检查是否内容为空
            if trans_words_total < 100 and trans_words_total > 0:
                status = "🟡 VERY SHORT"
                warnings.append(f"{base_chapter}: Translation appears very short ({trans_words_total} words/chars)")

        # 格式化输出（显示part数量）
        ocr_str = f"{ocr_words_total:,} words" if ocr_words_total > 0 else "-"
        if len(group['ocr']) > 1:
            ocr_str += f" ({len(group['ocr'])}p)"

        polish_str = f"{polish_words_total:,} words" if polish_words_total > 0 else "-"
        if len(group['polish']) > 1:
            polish_str += f" ({len(group['polish'])}p)"

        if is_chinese_target:
            trans_str = f"{trans_words_total:,} 字" if trans_words_total > 0 else "-"
        else:
            trans_str = f"{trans_words_total:,} words" if trans_words_total > 0 else "-"
        if len(group['translated']) > 1:
            trans_str += f" ({len(group['translated'])}p)"

        print(f"{base_chapter:<30} {ocr_str:<20} {polish_str:<20} {trans_str:<20} {status:<15}")

        stats.append({
            'chapter': base_chapter,
            'ocr_words': ocr_words_total,
            'polish_words': polish_words_total,
            'trans_words': trans_words_total,
            'status': status,
            'ocr_parts': len(group['ocr']),
            'polish_parts': len(group['polish']),
            'trans_parts': len(group['translated'])
        })

    # 汇总统计
    print(f"\n{'='*100}")
    print("📊 SUMMARY")
    print(f"{'='*100}")

    total_ocr = sum(s['ocr_words'] for s in stats)
    total_polish = sum(s['polish_words'] for s in stats)
    total_trans = sum(s['trans_words'] for s in stats)

    complete_chapters = len([s for s in stats if '✅' in s['status']])
    chapters_with_issues = len([s for s in stats if '🔴' in s['status']])

    print(f"Total chapters: {len(all_chapters)}")
    print(f"  ✅ Complete: {complete_chapters}")
    print(f"  🔴 With issues: {chapters_with_issues}")
    print()
    print(f"Total files:")
    print(f"  OCR files: {len(ocr_files)}")
    print(f"  Polish files: {len(polish_files)}")
    print(f"  Translated files: {len(translated_files)}")
    print()

    if is_chinese_target:
        print(f"Total OCR words: {total_ocr:,}")
        print(f"Total Polish words: {total_polish:,}")
        print(f"Total Translation chars: {total_trans:,} 字")
        if total_polish > 0:
            print(f"Translation ratio: {total_trans/total_polish:.2f} (Chinese chars / English words)")
    else:
        print(f"Total OCR words: {total_ocr:,}")
        print(f"Total Polish words: {total_polish:,}")
        print(f"Total Translation words: {total_trans:,}")
        if total_polish > 0:
            print(f"Polish/OCR ratio: {total_polish/total_ocr:.2%}")
            print(f"Trans/Polish ratio: {total_trans/total_polish:.2%}")

    # 输出警告
    if warnings:
        print(f"\n{'='*100}")
        print("⚠️  WARNINGS")
        print(f"{'='*100}")
        for warning in warnings:
            print(f"  • {warning}")
    else:
        print(f"\n✅ No warnings - all chapters appear complete!")

    print(f"\n{'='*100}\n")

    return stats, warnings


def main():
    """主函数"""
    try:
        # 加载配置
        config = load_config()
        title = config.get('title', '')
        source_lang = config.get('source_language', 'English')
        target_lang = config.get('target_language', 'Chinese')

        if not title:
            print("Error: No title found in config.yaml")
            return

        print(f"Looking for output directory for: {title}")

        # 查找输出目录
        output_dir = find_output_dir(title)
        print(f"Found: {output_dir}")

        # 分析完整性
        stats, warnings = analyze_directory(output_dir, source_lang, target_lang)

        # 返回状态码
        if warnings:
            exit(1)
        else:
            exit(0)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
