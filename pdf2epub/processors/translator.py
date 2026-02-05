"""
Translation processor for markdown content.

This processor translates markdown content from one language to another
while preserving formatting and structure.
"""

import json
import random
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

from loguru import logger
from .base import BaseMarkdownProcessor
from ..utils.common import parse_llm_json
from .utils.truncation import NGramTruncationDetector
from .utils.split_manager import SplitManager
from .tracker import ProcessingTracker
from ..chapter_identity import ChapterIdentity


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

        # Get validation settings from config
        validation_config = config.get('validation_strategy', {})
        self.validate_chinese = validation_config.get('validate_chinese_translation', True)
        
        # Set default translation models if not provided
        # Use translation.models from config, same as regular translate
        self.translation_models = translation_models or config.get('translation', {}).get('models') or [
            {"provider": "gemini", "model": "gemini-2.5-pro", "api_retries": 2, "validation_retries": 2},
            {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929", "api_retries": 2, "validation_retries": 1}
        ]
        
        # Initialize truncation detector
        translate_config = config.get('translation', {})

        # Use NGramTruncationDetector for fast screening (same as polish)
        # Agent verification will handle accurate detection in Phase 2
        self.truncation_detector = NGramTruncationDetector(
            min_unique_preserved_ratio=0.60,
            allow_deduplication=True
        )
        
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

        # Initialize ProcessingTracker for audit trail
        tracker_path = self.output_dir / "processing_tracker.json"
        self.processing_tracker = ProcessingTracker(tracker_path, "TranslateProcessor")

        # Initialize SplitManager for dynamic splitting
        # Translation often needs larger parts since translated text expands
        splitting_config = config.get('splitting', {})
        self.split_manager = SplitManager(
            tracker=self.processing_tracker,
            output_dir=self.output_dir,
            default_max_tokens=self.get_max_tokens_per_part(),
            max_resplits=splitting_config.get('max_resplits', 3),
            consecutive_failures_threshold=splitting_config.get('consecutive_failures_threshold', 2)
        )

        # Enable batch validation mode (validate after all files processed)
        self.validation_mode = "batch"
        self.auto_save = False
        self._agent_verifier = None

    @property
    def agent_verifier(self):
        """Lazy initialization of agent verifier for batch validation."""
        if self._agent_verifier is None:
            from .utils.agent_verifier import TranslationVerificationAgent
            from .utils.verification_tools import VerificationTools

            tools = VerificationTools(units={})
            self._agent_verifier = TranslationVerificationAgent(tools)
        return self._agent_verifier

    def get_operation_name(self, file_name: str) -> str:
        """Get the operation name for logging."""
        return f"Translate {file_name}"

    def get_model_configs(self) -> List[Dict]:
        """Get the model configurations for translation."""
        return self.translation_models

    def build_prompt(self, content: str, unit_key: str, **context) -> List[Dict]:
        """
        Build the translation prompt with optional conversation history.

        Args:
            content: Content to translate
            unit_key: Unit identifier for tracking
            **context: Context including file_name, part_idx, total_parts, previous_context

        Returns:
            Multi-part content for LLM (may include conversation history)
        """
        part_idx = context.get('part_idx')
        total_parts = context.get('total_parts')
        previous_context = context.get('previous_context')

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

    def post_process(self, result: str, **context) -> str:
        """
        Post-process the translated result.

        All cleaning is done in clean_response() before validation,
        so post_process just returns the result unchanged.

        Args:
            result: Already cleaned LLM response
            **context: Processing context

        Returns:
            Result unchanged
        """
        return result

    def get_context_for_next_part(self, content: str, result: str, **context) -> Optional[Dict]:
        """
        Get context to inject into the next part's build_prompt.

        Always provides context for terminology/style consistency.

        Args:
            content: Original content of this part
            result: Processed result of this part
            **context: Processing context

        Returns:
            Context dict for next part
        """
        return {"original": content, "processed": result}

    def get_split_strategy(self) -> str:
        """
        Get splitting strategy for this processor.

        Returns:
            Strategy name: 'markdown'
        """
        return 'markdown'

    def validate_output(
        self,
        original: str,
        processed: str,
        file_name: str
    ) -> Tuple[bool, str]:
        """
        Validate the translated output using LLM-based truncation detection and target language validation.

        Args:
            original: Original content
            processed: Translated content
            file_name: Name of the file

        Returns:
            Tuple of (is_valid, reason)
        """
        # Skip truncation validation for front and back matter files
        # These files often have non-standard structures that can trigger false positives
        base_name = Path(file_name).stem.lower()
        skip_truncation = 'front_matter' in base_name or 'back_matter' in base_name

        if skip_truncation:
            logger.info(f"Skipping truncation validation for {file_name} (front/back matter)")
        else:
            # Check for truncation
            is_truncated, truncation_reason, details = self.truncation_detector.detect(
                original=original,
                processed=processed,
                source_language=self.source_language,
                target_language=self.target_language
            )

            # Log the truncation check summary
            summary = self.truncation_detector.get_summary(is_truncated, truncation_reason, details)
            if is_truncated:
                logger.warning(f"{file_name} translation truncation detected:\n{summary}")
                return False, truncation_reason
            else:
                logger.info(f"{file_name} translation validated:\n{summary}")

        # Then validate Chinese content if target language is Chinese and validation is enabled
        if self.validate_chinese and self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
            is_valid_chinese, chinese_validation_msg = self._validate_chinese_translation(processed)
            if not is_valid_chinese:
                logger.error(f"Chinese validation failed for {file_name}: {chinese_validation_msg}")
                logger.warning("LLM returned English or non-Chinese content when Chinese was requested")
                return False, f"Chinese validation failed: {chinese_validation_msg}"
            else:
                logger.debug(f"Chinese translation validated for {file_name}: {chinese_validation_msg}")

        return True, "Validation passed"

    def _load_entities(self) -> Optional[Dict]:
        """Load translation entities from JSON file."""
        import json
        from pathlib import Path
        
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
    
    def _validate_chinese_translation(self, text: str) -> Tuple[bool, str]:
        """
        Validate that the translation contains Chinese characters.

        Randomly samples 5 sections of 500 characters each and checks if they contain Chinese.
        Considers valid if at least 4 out of 5 sections contain Chinese.

        Args:
            text: The translated text to validate

        Returns:
            Tuple of (is_valid, reason)
        """
        # Remove markdown formatting and whitespace for better sampling
        clean_text = re.sub(r'[#\*\[\]\(\)!`\n\s]+', '', text)

        window_size = 500  # Extended window for better sampling

        if len(clean_text) < window_size:
            # Text too short, check if it has any Chinese characters at all
            has_chinese = bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', clean_text))
            if not has_chinese:
                return False, "Text contains no Chinese characters"
            return True, "Text contains Chinese characters (short text)"

        # Sample 5 random positions
        sample_size = min(5, len(clean_text) // window_size)
        if sample_size == 0:
            sample_size = 1

        sections_checked = []
        sections_with_chinese = 0

        for _ in range(sample_size):
            # Get random starting position
            max_start = len(clean_text) - window_size
            start = random.randint(0, max(0, max_start))
            end = min(start + window_size, len(clean_text))
            section = clean_text[start:end]

            # Check if this section contains Chinese characters
            # Unicode ranges for Chinese characters:
            # - Common CJK Unified Ideographs: U+4E00-U+9FFF
            # - CJK Extension A: U+3400-U+4DBF
            has_chinese = bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', section))

            sections_checked.append({
                'position': f"chars {start}-{end}",
                'sample': section[:30] + "..." if len(section) > 30 else section,
                'has_chinese': has_chinese
            })

            if has_chinese:
                sections_with_chinese += 1

        # Check if enough sampled sections contain Chinese (4 out of 5 is acceptable)
        min_required = max(1, sample_size - 1) if sample_size > 1 else 1  # At least 4/5, or all if less than 5

        if sections_with_chinese == 0:
            details = "\n".join([f"  - {s['position']}: No Chinese found in '{s['sample']}'"
                                for s in sections_checked])
            return False, f"No Chinese characters found in any of {sample_size} sampled sections:\n{details}"
        elif sections_with_chinese < min_required:
            details = "\n".join([f"  - {s['position']}: {'✓' if s['has_chinese'] else '✗'} Chinese in '{s['sample']}'"
                                for s in sections_checked])
            return False, f"Only {sections_with_chinese}/{sample_size} sections contain Chinese (need at least {min_required}):\n{details}"

        return True, f"{sections_with_chinese}/{sample_size} sampled sections contain Chinese characters"

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

    def _build_toc_reference_json(self) -> List[Dict]:
        """
        Build TOC reference JSON including book_title and all chapter titles.

        Extracts titles from translated markdown files as references.

        Returns:
            List of dicts with id, level, original, and reference fields
        """
        result = []

        # Load toc_tree.json
        toc_tree_path = self.output_dir.parent / "toc_tree.json"
        if not toc_tree_path.exists():
            logger.error(f"toc_tree.json not found at {toc_tree_path}")
            return result

        with open(toc_tree_path, 'r', encoding='utf-8') as f:
            toc_tree = json.load(f)

        # Add book_title as first item (level 0)
        book_title = toc_tree.get('book_title', '')
        result.append({
            "id": "book_title",
            "level": 0,
            "original": book_title,
            "reference": ""  # No reference for book title
        })

        # Build mapping from chapter_id to translated title
        title_references = self._extract_titles_from_translated()

        # Recursively process chapters
        chapter_counter = [0]  # Use list for mutable counter in nested function

        def process_chapters(chapters: List[Dict], parent_id: str = ""):
            for chapter in chapters:
                chapter_counter[0] += 1

                # Generate chapter_id based on structure
                if parent_id:
                    chapter_id = f"{parent_id}.{chapter_counter[0]}"
                else:
                    chapter_id = f"chapter_{chapter_counter[0]}"

                original_title = chapter.get('title', '')
                level = chapter.get('level', 1)

                # Try to find reference from translated files
                # Match by looking for files that correspond to this chapter
                reference = title_references.get(chapter_id, "")

                result.append({
                    "id": chapter_id,
                    "level": level,
                    "original": original_title,
                    "reference": reference
                })

                # Process children
                if 'children' in chapter:
                    old_counter = chapter_counter[0]
                    chapter_counter[0] = 0
                    process_chapters(chapter['children'], chapter_id)
                    chapter_counter[0] = old_counter

        process_chapters(toc_tree.get('chapters', []))

        return result

    def _extract_titles_from_translated(self) -> Dict[str, str]:
        """
        Extract first # heading from each translated markdown file.

        Returns:
            Dict mapping chapter_id to translated title
        """
        title_map = {}

        # Get all markdown files in translated directory
        translated_dir = self.output_dir
        if not translated_dir.exists():
            return title_map

        for md_file in sorted(translated_dir.glob("*.md")):
            # Parse filename to get chapter_id
            stem = md_file.stem
            identity = ChapterIdentity.parse(stem)

            # Only use part1 or non-part files
            if identity.part and identity.part > 1:
                continue

            # Read file and extract first # heading
            try:
                content = md_file.read_text(encoding='utf-8')

                # Find first # heading (not ##)
                match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                if match:
                    title = match.group(1).strip()
                    # Use base_name as key (e.g., "chapter_3.1")
                    title_map[identity.base_name] = title
            except Exception as e:
                logger.warning(f"Failed to extract title from {md_file}: {e}")

        return title_map

    def _translate_toc_batch(self, reference_json: List[Dict]) -> List[Dict]:
        """
        Translate entire TOC in one batch using LLM.

        Args:
            reference_json: List of dicts with id, level, original, reference

        Returns:
            List of dicts with id and translated fields
        """
        if not reference_json:
            return []

        # Build prompt
        input_json = json.dumps(reference_json, ensure_ascii=False, indent=2)

        prompt = f"""翻译以下书籍目录结构从{self.source_language}到简体中文。

**要求**：
1. 参考"reference"字段保持术语一致（如果有参考的话）
2. 统一全书序号格式（如发现"1."和"一、"混用，请统一为同一种格式）
3. 全部使用简体中文（不要繁体）
4. 可修正OCR导致的明显错误
5. 保持原文中的序号格式（如原文是"1."则保持阿拉伯数字）

**输入**：
```json
{input_json}
```

**返回格式**（只返回JSON数组，不要其他内容）：
```json
[
  {{"id": "book_title", "translated": "翻译后的书名"}},
  {{"id": "chapter_1", "translated": "翻译后的标题"}},
  ...
]
```
"""

        try:
            # Call LLM
            response = self.llm_client.generate(
                prompt=prompt,
                model_configs=self.translation_models,
                operation_name="TOC batch translation"
            )

            # Parse response - extract JSON from possible markdown code block
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code block
                lines = response.split('\n')
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                response = '\n'.join(json_lines)

            translations = parse_llm_json(response, operation_name="TOC translation")

            # Validate
            if len(translations) != len(reference_json):
                logger.warning(
                    f"Translation count mismatch: expected {len(reference_json)}, "
                    f"got {len(translations)}"
                )

            return translations

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse TOC translation response: {e}")
            logger.error(f"Response was: {response[:500]}...")
            return []
        except Exception as e:
            logger.error(f"TOC batch translation failed: {e}")
            return []

    def _save_toc_tree_translated(self, translations: List[Dict]):
        """
        Save toc_tree_translated.json with translated titles.

        Args:
            translations: List of dicts with id and translated fields
        """
        # Load original toc_tree
        toc_tree_path = self.output_dir.parent / "toc_tree.json"
        if not toc_tree_path.exists():
            logger.error(f"toc_tree.json not found at {toc_tree_path}")
            return

        with open(toc_tree_path, 'r', encoding='utf-8') as f:
            toc_tree = json.load(f)

        # Build translation lookup
        trans_map = {t['id']: t['translated'] for t in translations if 'id' in t and 'translated' in t}

        # Update book_title
        if 'book_title' in trans_map:
            toc_tree['book_title'] = trans_map['book_title']

        # Update language to target
        toc_tree['language'] = self.target_language.lower()
        toc_tree['source_language'] = self.source_language.lower()

        # Recursively update chapter titles
        chapter_counter = [0]

        def update_chapters(chapters: List[Dict], parent_id: str = ""):
            for chapter in chapters:
                chapter_counter[0] += 1

                if parent_id:
                    chapter_id = f"{parent_id}.{chapter_counter[0]}"
                else:
                    chapter_id = f"chapter_{chapter_counter[0]}"

                # Store original title
                if 'original_title' not in chapter:
                    chapter['original_title'] = chapter.get('title', '')

                # Update with translation
                if chapter_id in trans_map:
                    chapter['title'] = trans_map[chapter_id]

                # Process children
                if 'children' in chapter:
                    old_counter = chapter_counter[0]
                    chapter_counter[0] = 0
                    update_chapters(chapter['children'], chapter_id)
                    chapter_counter[0] = old_counter

        update_chapters(toc_tree.get('chapters', []))

        # Save translated toc_tree
        output_path = self.output_dir.parent / "toc_tree_translated.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(toc_tree, f, ensure_ascii=False, indent=2)

        logger.success(f"Saved translated TOC to {output_path}")

    def _get_original_content(self, file_key: str) -> str:
        """
        Get original content for a given file key.

        Args:
            file_key: File identifier (e.g., "chapter_3.part2")

        Returns:
            Original content string
        """
        # Get input file path
        if file_key.endswith('.md'):
            input_path = self.input_dir / file_key
        else:
            input_path = self.input_dir / f"{file_key}.md"

        if not input_path.exists():
            # Try without .md extension in case it's in splits dir
            input_path = self.input_dir / "splits" / f"{file_key}.md"

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        return input_path.read_text(encoding='utf-8')

    def _save_result(self, file_key: str, content: str) -> None:
        """
        Save translation result to output directory.

        Args:
            file_key: File identifier
            content: Translated content
        """
        # Determine output path
        output_path = self.output_dir / "splits" / f"{file_key}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        output_path.write_text(content, encoding='utf-8')
        logger.debug(f"Saved translation result: {output_path}")

    def _batch_validate_and_save(
        self,
        all_units: list,
        completed_results: Dict[str, str],
        failed_ids: List[str]
    ) -> tuple:
        """
        Validate and save all completed translation results using two-phase validation.

        Phase 1: LLMTruncationDetector screening (fast)
        Phase 2: Agent verification for suspicious files (accurate)

        Args:
            all_units: List of all work units
            completed_results: Dict mapping unit_key to translated content
            failed_ids: List of unit keys that failed during processing

        Returns:
            Tuple of (validated_results dict, updated_failed_ids list)
        """
        from .utils.verification_tools import VerificationFile
        from .utils.agent_verifier import verify_batch

        if not completed_results:
            logger.info("No completed results to validate")
            return {}, failed_ids

        # Separate newly processed from already-saved units
        newly_processed = {}
        already_saved = {}

        for unit in all_units:
            unit_id = unit.id
            if unit_id not in completed_results:
                continue

            if not unit.output_path.exists():
                newly_processed[unit_id] = completed_results[unit_id]
            else:
                already_saved[unit_id] = completed_results[unit_id]
                logger.debug(f"Skipping validation for {unit_id} (already saved)")

        logger.info(f"Validating {len(newly_processed)} newly processed units, skipping {len(already_saved)} already saved")

        if not newly_processed:
            return completed_results, failed_ids

        # Phase 1: LLMTruncationDetector screening
        logger.info(f"Phase 1: LLMTruncationDetector screening for {len(newly_processed)} units")

        passed = {}
        suspicious = {}

        for unit_key, translated in newly_processed.items():
            try:
                original = self._get_original_content(unit_key)

                # Use LLMTruncationDetector (same as inline validation)
                is_truncated, reason, details = self.truncation_detector.detect(
                    original=original,
                    processed=translated,
                    source_language=self.source_language,
                    target_language=self.target_language
                )

                if not is_truncated:
                    passed[unit_key] = translated
                    logger.debug(f"{unit_key}: LLM detector passed - {reason}")
                else:
                    suspicious[unit_key] = VerificationFile(
                        key=unit_key,
                        original=original,
                        processed=translated
                    )
                    logger.debug(f"{unit_key}: LLM detector flagged as suspicious - {reason}")

            except Exception as e:
                logger.error(f"Error in LLM screening for {unit_key}: {e}")
                suspicious[unit_key] = VerificationFile(
                    key=unit_key,
                    original=self._get_original_content(unit_key),
                    processed=translated
                )

        logger.info(f"{len(passed)} units passed LLM screening, {len(suspicious)} suspicious")

        # Phase 2: Agent verification for suspicious files
        validated_by_agent = {}
        still_failed = []

        if suspicious:
            logger.info(f"Phase 2: Agent verification for {len(suspicious)} suspicious units")

            try:
                verification_results = verify_batch(
                    files=suspicious,
                    task_type="translate"
                )

                for result in verification_results:
                    if result.status == "complete":
                        validated_by_agent[result.file_key] = suspicious[result.file_key].processed
                        logger.info(f"{result.file_key}: Agent verified complete - {result.reason}")
                    else:
                        still_failed.append(result.file_key)
                        logger.warning(f"{result.file_key}: Agent confirmed truncation - {result.reason}")

            except Exception as e:
                logger.error(f"Agent verification failed: {e}")
                # Fall back to saving all suspicious files anyway (don't lose translated content)
                logger.warning(f"Saving {len(suspicious)} files despite verification failure")
                for unit_key, vfile in suspicious.items():
                    validated_by_agent[unit_key] = vfile.processed
                    logger.info(f"{unit_key}: Saved despite verification error (content may need review)")

        # Also save files that agent marked as truncated (don't lose translated content)
        if still_failed and suspicious:
            logger.warning(f"Saving {len(still_failed)} files that failed verification (may be incomplete)")
            for unit_key in still_failed:
                if unit_key in suspicious and unit_key not in validated_by_agent:
                    validated_by_agent[unit_key] = suspicious[unit_key].processed
                    logger.info(f"{unit_key}: Saved with truncation warning")

        # Combine validated results
        all_validated = {**passed, **validated_by_agent, **already_saved}

        # Save validated units
        logger.info(f"Saving {len(passed) + len(validated_by_agent)} validated units")

        for unit_key, content in {**passed, **validated_by_agent}.items():
            try:
                self._save_result(unit_key, content)
            except Exception as e:
                logger.error(f"Failed to save {unit_key}: {e}")
                still_failed.append(unit_key)

        # Aggregate multi-part files
        self._aggregate_validated_files(all_validated)

        # Update failed_ids (convert to set, merge, keep as set)
        updated_failed_ids = failed_ids.union(still_failed)

        return all_validated, updated_failed_ids

    def _aggregate_validated_files(self, validated_results: Dict[str, str]) -> None:
        """
        Aggregate multi-part translated files into single files.

        Args:
            validated_results: Dict of validated translations
        """
        # Group by base file name
        from collections import defaultdict
        base_files = defaultdict(list)

        for unit_key in validated_results.keys():
            # Extract base name (e.g., "chapter_3" from "chapter_3.part2")
            if '.part' in unit_key:
                base_name = unit_key.split('.part')[0]
                base_files[base_name].append(unit_key)

        # Aggregate each multi-part file
        for base_name, parts in base_files.items():
            if len(parts) <= 1:
                continue

            # Sort parts by part number
            sorted_parts = sorted(parts, key=lambda x: int(x.split('.part')[1]))

            # Concatenate content
            aggregated = []
            for part_key in sorted_parts:
                part_path = self.output_dir / "splits" / f"{part_key}.md"
                if part_path.exists():
                    aggregated.append(part_path.read_text(encoding='utf-8'))

            # Write aggregated file
            if aggregated:
                output_path = self.output_dir / f"{base_name}.md"
                output_path.write_text('\n\n'.join(aggregated), encoding='utf-8')
                logger.info(f"Aggregated {len(sorted_parts)} parts into {base_name}.md")

    def process_all_files(self) -> Dict[str, Any]:
        """
        Process all markdown files and then translate TOC.

        Overrides base class to add TOC translation at the end.

        Returns:
            Summary statistics
        """
        # Process all chapter files first
        result = super().process_all_files()

        # After all chapters are translated, translate the TOC
        try:
            logger.info("Translating TOC titles...")

            # Build reference JSON from translated files
            reference_json = self._build_toc_reference_json()

            if reference_json:
                # Batch translate all titles
                translations = self._translate_toc_batch(reference_json)

                if translations:
                    # Save translated TOC
                    self._save_toc_tree_translated(translations)
                    logger.success("TOC translation completed")
                else:
                    logger.error("TOC batch translation returned no results")
            else:
                logger.warning("No TOC reference data found, skipping TOC translation")

        except Exception as e:
            logger.error(f"TOC translation failed: {e}")
            # Don't fail the entire process for TOC translation failure
            result['toc_translation_error'] = str(e)

        return result
