"""AZW3/MOBI 转 EPUB 工具，使用 mobi 包"""
import re
import shutil
import zipfile
from pathlib import Path

from loguru import logger

CONVERTIBLE_FORMATS = {".azw3", ".mobi", ".azw"}


def _fix_unescaped_ampersands(content: str) -> str:
    """修复 XML 中未转义的 & 字符"""
    # 匹配 & 后面不是有效实体引用的情况
    # 有效实体: &amp; &lt; &gt; &quot; &apos; &#123; &#x1F;
    pattern = r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)'
    return re.sub(pattern, '&amp;', content)


def _sanitize_epub(epub_path: Path) -> None:
    """修复 mobi 包生成的 EPUB 中的 XML 问题"""
    import tempfile

    # 需要修复的文件扩展名
    xml_extensions = {'.ncx', '.opf', '.xhtml', '.html', '.xml'}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        fixed_epub = tmpdir / 'fixed.epub'

        with zipfile.ZipFile(epub_path, 'r') as zf_in:
            with zipfile.ZipFile(fixed_epub, 'w', zipfile.ZIP_DEFLATED) as zf_out:
                for item in zf_in.infolist():
                    content = zf_in.read(item.filename)

                    # 检查是否是 XML 文件
                    if any(item.filename.lower().endswith(ext) for ext in xml_extensions):
                        try:
                            text = content.decode('utf-8')
                            fixed_text = _fix_unescaped_ampersands(text)
                            if fixed_text != text:
                                logger.debug(f"Fixed unescaped ampersands in {item.filename}")
                            content = fixed_text.encode('utf-8')
                        except UnicodeDecodeError:
                            pass  # 不是文本文件，跳过

                    # mimetype 必须不压缩且在第一个
                    if item.filename == 'mimetype':
                        zf_out.writestr(item, content, compress_type=zipfile.ZIP_STORED)
                    else:
                        zf_out.writestr(item, content)

        # 替换原文件
        shutil.move(str(fixed_epub), str(epub_path))


def needs_conversion(file_path: Path) -> bool:
    """检查文件是否需要转换为 EPUB"""
    return Path(file_path).suffix.lower() in CONVERTIBLE_FORMATS


def convert_to_epub(input_path: Path, output_dir: Path) -> tuple[Path, bool]:
    """
    转换 azw3/mobi 为 epub

    Args:
        input_path: 输入文件路径 (azw3/mobi)
        output_dir: 输出目录

    Returns:
        (epub_path, was_converted): EPUB 文件路径和是否进行了转换

    Raises:
        FileNotFoundError: 输入文件不存在
        RuntimeError: 转换失败
    """
    import mobi

    input_path = Path(input_path).absolute()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not needs_conversion(input_path):
        return input_path, False

    logger.info(f"Converting {input_path.suffix} to EPUB using mobi package...")

    # mobi.extract 返回 (临时目录, 提取文件路径)
    tempdir, extracted_path = mobi.extract(str(input_path))
    extracted_path = Path(extracted_path)

    # 确定输出路径
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if extracted_path.suffix.lower() == ".epub":
            # 保存为 input.epub，方便后续 build-html-epub 自动找到
            epub_path = output_dir / "input.epub"
            shutil.copy2(extracted_path, epub_path)

            # 修复 mobi 包生成的 EPUB 中可能存在的 XML 问题
            _sanitize_epub(epub_path)

            logger.success(f"Converted to EPUB: {epub_path}")
            return epub_path, True
        else:
            # 提取的是 html 或其他格式
            raise RuntimeError(
                f"mobi package extracted {extracted_path.suffix} instead of EPUB. "
                f"This usually happens with older mobi7 format files. "
                f"Please install Calibre and use 'ebook-convert' for this file."
            )
    finally:
        # 清理临时目录
        shutil.rmtree(tempdir, ignore_errors=True)
