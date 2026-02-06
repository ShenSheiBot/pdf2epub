"""
Translation processor for markdown content.

This processor translates markdown content from one language to another
while preserving formatting and structure.
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, TYPE_CHECKING

from loguru import logger
from .base import BaseMarkdownProcessor
from ..chapter_identity import ChapterIdentity

if TYPE_CHECKING:
    from ..core._protocol import ProcessContext


class TranslateProcessor(BaseMarkdownProcessor):
    """Processor for translating markdown content."""

    def __init__(
        self,
        config: Dict,
        book_title: str,
        source_language: str = "Japanese",
        target_language: str = "Chinese",
        max_workers: int = 4,
        resume: bool = False,
        translation_models: Optional[list] = None,
        use_entities: Optional[bool] = None,
        use_longest_on_failure: bool = False,
        book_structure: Optional[Dict] = None
    ):
        """
        Initialize the translation processor.

        Args:
            config: Configuration dictionary
            book_title: Title of the book being processed
            source_language: Source language for translation
            target_language: Target language for translation
            max_workers: Maximum number of concurrent workers
            resume: Whether to resume from previous progress
            translation_models: Optional override for model configurations
            use_entities: Whether to use extracted entities for consistency
                        (None = auto-detect, True = force use, False = force disable)
            use_longest_on_failure: If True, use longest response when all attempts fail validation
            book_structure: Optional book structure from breakdown/refine phase
        """
        super().__init__(
            config=config,
            book_title=book_title,
            input_dir="polished_markdown",
            output_dir="translated",
            max_workers=max_workers,
            resume=resume,
            use_longest_on_failure=use_longest_on_failure
        )

        self.book_structure = book_structure or {}
        self.source_language = source_language
        self.target_language = target_language

        # Set default translation models if not provided
        # Use translation.models from config, same as regular translate
        self.translation_models = translation_models or config.get('translation', {}).get('models') or [
            {"provider": "gemini", "model": "gemini-2.5-pro", "api_retries": 2, "validation_retries": 2},
            {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929", "api_retries": 2, "validation_retries": 1}
        ]

        # Auto-detect entities if use_entities is None, otherwise use explicit value
        if use_entities is None:
            # Auto-detect: use entities if the file exists
            entities_file = Path("output") / self.book_title / "translation_entities.json"
            if entities_file.exists():
                logger.info("Auto-detected translation entities file, will use for consistency")
                self.entities = self._load_entities()
            else:
                self.entities = None
        elif use_entities:
            # Explicitly requested to use entities
            self.entities = self._load_entities()
        else:
            # Explicitly requested NOT to use entities
            self.entities = None

    def get_model_configs(self) -> List[Dict]:
        """Get the model configurations for translation."""
        return self.translation_models

    def build_prompt(self, content: str, context: "ProcessContext") -> Any:
        """
        Build the translation prompt with optional conversation history.

        Args:
            content: Content to translate
            context: Processing context with file info, part info, previous context

        Returns:
            Multi-part content for LLM (may include conversation history)
        """
        part_idx = context.part_index
        total_parts = context.total_parts

        # Build previous_context dict if context injection is available
        previous_context = None
        if context.has_previous_context:
            previous_context = {
                'original': context.previous_original,
                'processed': context.previous_processed
            }

        # Create the translation prompt
        prompt = self._create_translation_prompt()

        # Create multi-part content for the LLM
        if previous_context and part_idx and part_idx > 1:
            # Use conversation history for terminology/style consistency
            prev_user_content = [
                {"type": "text", "text": prompt + f"\n\nContent to translate (part {part_idx-1}/{total_parts}):"},
                {"type": "text", "text": previous_context['original']}
            ]
            current_user_content = [
                {"type": "text", "text": f"Now translate part {part_idx}/{total_parts} (continuation). Maintain consistent terminology and style with the previous translation.\n\nContent to translate:"},
                {"type": "text", "text": content}
            ]
            return [
                {"role": "user", "content": prev_user_content},
                {"role": "assistant", "content": previous_context['processed']},
                {"role": "user", "content": current_user_content}
            ]
        else:
            # No previous context, use standard format
            return [
                {"type": "text", "text": prompt},
                {"type": "text", "text": content}
            ]

    def clean_response(self, response: str) -> str:
        """
        Clean LLM response before validation.

        Performs all cleaning that was previously done before validation:
        1. Markdown cleanup (remove code blocks)
        2. Japanese artifact cleanup (if Japanese to Chinese)
        3. Footnote colon correction

        Args:
            response: Raw LLM response

        Returns:
            Fully cleaned response (ready for validation)
        """
        # First do standard markdown cleanup
        cleaned = self.clean_markdown_response(response)

        # Clean Japanese artifacts if translating Japanese to Chinese
        if self.source_language.lower() in ["japanese", "日本語"] and self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
            cleaned = self._clean_japanese_artifacts(cleaned)

        # Correct footnote colons
        return self._correct_footnote_colons(cleaned)

    def post_process(self, result: str, context: "ProcessContext") -> str:
        """
        Post-process the translated result.

        All cleaning is done in clean_response() before validation,
        so post_process just returns the result unchanged.

        Args:
            result: Already cleaned LLM response
            context: Processing context

        Returns:
            Result unchanged
        """
        return result

    def _load_entities(self) -> Optional[Dict]:
        """Load translation entities from JSON file."""
        entities_file = Path("output") / self.book_title / "translation_entities.json"
        if entities_file.exists():
            try:
                with open(entities_file, "r", encoding="utf-8") as f:
                    entities = json.load(f)
                logger.info(f"Loaded translation entities from {entities_file}")
                return entities
            except Exception as e:
                logger.warning(f"Failed to load entities: {e}")
                return None
        else:
            logger.warning(f"Entity file not found: {entities_file}")
            logger.info("Run 'extract-entities' command first to generate entity reference")
            return None

    def _correct_footnote_colons(self, text: str) -> str:
        """
        Correct full-width colons in footnote definitions to standard ASCII colons.

        Targets patterns like [^1]： and replaces the full-width colon with a standard colon.
        This is a common issue when LLMs translate footnotes, especially to languages
        that use full-width punctuation.

        Args:
            text: The translated text to correct

        Returns:
            Text with corrected footnote colons
        """
        # Pattern to find footnote references followed by full-width colon
        # (\[\^.+?\]) captures the footnote reference (e.g., [^1], [^note])
        # ： matches the full-width colon that needs to be replaced
        pattern = r'(\[\^.+?\])：'
        replacement = r'\1: '

        corrected_text = re.sub(pattern, replacement, text)
        return corrected_text

    def _clean_japanese_artifacts(self, text: str) -> str:
        """
        Clean up Japanese artifacts in Chinese translation.
        Specifically removes っ when it appears around punctuation.
        """

        # Define Japanese and Chinese punctuation marks
        punctuation = [
            '。', '、', '，', '！', '？', '…', '～', '—', '「', '」', '『', '』',
            '（', '）', '【', '】', '・', '：', '；', '"', '"', ''', '''
        ]

        # Remove っ before punctuation
        for p in punctuation:
            text = text.replace(f'っ{p}', p)

        # Remove っ after punctuation
        for p in punctuation:
            text = text.replace(f'{p}っ', p)

        # Remove standalone っ surrounded by spaces or punctuation
        # Pattern: punctuation/space + っ + punctuation/space
        pattern = r'([。、，！？…～—「」『』（）【】・：；\s])っ([。、，！？…～—「」『』（）【】・：；\s])'
        text = re.sub(pattern, r'\1\2', text)

        # Remove っ at the end of quoted speech before closing quotes
        text = re.sub(r'っ([」』])', r'\1', text)

        return text

    def _create_translation_prompt(self) -> str:
        """Create the prompt for translation."""
        # Use Chinese prompt if target language is Chinese
        if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
            prompt = f"""你是一位专业的学术和文学文本翻译专家。

请将以下markdown内容从{self.source_language}翻译成简体中文。

重要要求：
1. **保留所有markdown格式**：保持标题(#, ##, ###)、强调(*斜体*, **粗体**)、列表、引用、代码块等格式不变
2. **图片链接保持不变**：不要翻译或修改图片路径，如 ![...](../images/xxx.png) 或 <img src="..." />
3. **脚注处理**：
   - 保持脚注格式 [^1], [^2] 等不变
   - 不要添加原文中不存在的脚注
4. **维持文档结构**：保持相同的段落分隔、章节划分和整体布局
5. **学术文本**：使用准确的中文学术术语
6. **文学文本**：保留原文的风格和语调
7. **不要添加说明**：只返回翻译后的markdown内容，不要添加任何解释或评论
8. **必须输出简体中文**：请确保翻译结果是简体中文，不要返回英文或其他语言"""
        else:
            prompt = f"""You are a professional translator specializing in academic and literary texts.

Translate the following markdown content from {self.source_language} to {self.target_language}.

IMPORTANT REQUIREMENTS:
1. **Preserve ALL markdown formatting**: Keep headers (#, ##, ###), emphasis (*italic*, **bold**), lists, quotes, code blocks, etc.
2. **Keep image links unchanged**: Do not translate or modify image paths like ![...](../images/xxx.png) or <img src="..." />
3. **Don't touch footnote**:
   - Keep the footnote format [^1], [^2], etc. unchanged
   - Don't add footnote that does not exist in the original text
4. **Maintain document structure**: Keep the same paragraph breaks, section divisions, and overall layout
5. **For academic texts**: Use appropriate academic terminology in the target language
6. **For literary texts**: Preserve the style and tone of the original
7. **Do NOT add explanations**: Return ONLY the translated markdown, no explanations or comments"""

        # Add specific rules for Japanese to Chinese translation
        if self.source_language.lower() in ["japanese", "日本語"] and self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
            # Add rules in Chinese since the prompt is in Chinese
            if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
                prompt += """

日译中特殊规则：
9. **删除不必要的注音**：不要包含类似 谦逊（けんそん）的日文读音标注，中文读者不需要日文读音
10. **正确处理日文语气词**：
   - 删除或调整句尾的っ
   - 「っ，呜，呜嗯っ……呜，呜」应该翻译为「呜，呜嗯……呜，呜」
   - 不要直接将っ翻译成中文字符
11. **自然的中文表达**：确保译文符合中文表达习惯，没有日文语言痕迹"""
            else:
                prompt += """

SPECIFIC RULES FOR JAPANESE TO CHINESE:
8. **Remove unnecessary ruby annotations**: Do NOT include pronunciation guides like 谦逊（けんそん）. Chinese readers don't need Japanese readings.
9. **Handle Japanese particles properly**:
   - Remove or adapt っ at the end of sentences/exclamations
   - 「っ，呜，呜嗯っ……呜，呜」 should become 「呜，呜嗯……呜，呜」
   - Do not literally translate っ as a character
10. **Natural Chinese expression**: Ensure the translation reads naturally in Chinese without Japanese linguistic artifacts"""

        # Add entity reference if available
        if self.entities:
            prompt += self._create_entity_reference_section()

        # Final instruction in appropriate language
        if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
            prompt += "\n\n请翻译以下内容："
        else:
            prompt += "\n\nTranslate the following content:"
        return prompt

    def _create_entity_reference_section(self) -> str:
        """Create the entity reference section for the prompt."""
        if not self.entities:
            return ""

        # Use Chinese headers if target language is Chinese
        if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
            reference = "\n\n**翻译一致性参考：**\n"
            reference += "请使用以下既定译名以保持一致性：\n"
            reference += "重要：请始终使用提供的人物名称及其昵称的翻译。\n\n"
        else:
            reference = "\n\n**TRANSLATION CONSISTENCY REFERENCE:**\n"
            reference += "Use these established translations for consistency:\n"
            reference += "IMPORTANT: Always use the provided translations for character names AND their nicknames.\n\n"

        # Add characters
        if "characters" in self.entities and self.entities["characters"]:
            if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
                reference += "**人物：**\n"
            else:
                reference += "**Characters:**\n"
            for char in self.entities["characters"]:
                reference += f"- {char['japanese']}"
                if char.get('reading'):
                    reference += f" ({char['reading']})"
                reference += f" → {char['chinese']}"
                if char.get('gender'):
                    reference += f" [{char['gender']}]"
                reference += "\n"

                # Add nicknames if present
                if char.get('nicknames'):
                    for nickname in char['nicknames']:
                        if isinstance(nickname, dict):
                            reference += f"  • {nickname.get('japanese', '')} → {nickname.get('chinese', '')}\n"
                        elif isinstance(nickname, str):
                            # Old format compatibility - just the Japanese nickname
                            reference += f"  • {nickname} (needs translation)\n"
            reference += "\n"

        # Add places
        if "places" in self.entities and self.entities["places"]:
            if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
                reference += "**地点：**\n"
            else:
                reference += "**Places:**\n"
            for place in self.entities["places"]:
                reference += f"- {place['japanese']} → {place['chinese']}\n"
            reference += "\n"

        # Add important terms
        if "terms" in self.entities and self.entities["terms"]:
            if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
                reference += "**专有名词：**\n"
            else:
                reference += "**Special Terms:**\n"
            for term in self.entities["terms"]:
                reference += f"- {term['japanese']} → {term['chinese']}\n"
            reference += "\n"

        # Add organizations
        if "organizations" in self.entities and self.entities["organizations"]:
            if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
                reference += "**组织：**\n"
            else:
                reference += "**Organizations:**\n"
            for org in self.entities["organizations"]:
                reference += f"- {org['japanese']} → {org['chinese']}\n"
            reference += "\n"

        # Add races if present
        if "races" in self.entities and self.entities["races"]:
            if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
                reference += "**种族/物种：**\n"
            else:
                reference += "**Races/Species:**\n"
            for race in self.entities["races"]:
                reference += f"- {race['japanese']} → {race['chinese']}"
                if race.get('chinese_plural'):
                    reference += f" (plural: {race['chinese_plural']})"
                reference += "\n"
            reference += "\n"

        # Add items if present
        if "items" in self.entities and self.entities["items"]:
            if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
                reference += "**物品：**\n"
            else:
                reference += "**Items:**\n"
            for item in self.entities["items"]:
                reference += f"- {item['japanese']} → {item['chinese']}\n"
            reference += "\n"

        if self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
            reference += "**重要提示：** 请在整个文本中保持这些译名的一致性。\n"
        else:
            reference += "**IMPORTANT:** Maintain these translations consistently throughout the text.\n"

        return reference
