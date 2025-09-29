"""
Translation processor for markdown content.

This processor translates markdown content from one language to another
while preserving formatting and structure.
"""

from typing import Dict, Optional, Tuple
from pathlib import Path
from loguru import logger
import re
import random

from .base import BaseMarkdownProcessor
from .utils.truncation import LLMTruncationDetector


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
        use_longest_on_failure: bool = False
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
        
        self.source_language = source_language
        self.target_language = target_language

        # Get validation settings from config
        validation_config = config.get('validation_strategy', {})
        self.validate_chinese = validation_config.get('validate_chinese_translation', True)
        
        # Set default translation models if not provided
        self.translation_models = translation_models or [
            {"provider": "gemini", "model": "gemini-2.5-pro", "max_retries": 2},
            {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "max_retries": 2}
        ]
        
        # Initialize truncation detector
        self.truncation_detector = LLMTruncationDetector(
            llm_client=self.llm_client,
            num_lines=3
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
        
        # Store language info in progress
        if self.progress.get("target_language") != target_language:
            if self.progress.get("target_language") is not None:
                logger.warning(
                    f"Target language changed from {self.progress['target_language']} "
                    f"to {target_language}"
                )
                if resume:
                    logger.warning("Resuming with different target language may produce mixed results")
            self.progress["target_language"] = target_language
            self.progress["source_language"] = source_language
            self.save_progress()
    
    def get_progress_filename(self) -> str:
        """Get the name for the progress file."""
        return "translation_progress"
    
    def get_progress_key(self) -> str:
        """Get the key used in progress tracking."""
        return "translations"
    
    def get_operation_name(self, file_name: str) -> str:
        """Get the operation name for logging."""
        return f"Translate {file_name}"
    
    def _detect_part_files(self, file_name: str) -> list:
        """Detect if there are part files for this file."""
        base_name = Path(file_name).stem
        part_files = sorted(self.input_dir.glob(f"{base_name}.part*.md"))
        return part_files
    
    def _translate_part(self, content: str, file_name: str, part_idx: int = None, total_parts: int = None) -> str:
        """Translate a single part of content using the new centralized retry logic."""
        # Create the translation prompt
        prompt = self._create_translation_prompt()

        # Create multi-part content for the LLM
        multi_part_content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": content}
        ]

        # Generate operation name
        if part_idx and total_parts:
            operation_name = f"{self.get_operation_name(file_name)} part {part_idx}/{total_parts}"
        else:
            operation_name = self.get_operation_name(file_name)

        # Define validator function that includes all our translation validations
        def validator(response: str) -> Tuple[bool, str]:
            # Clean the response first
            cleaned = self.clean_markdown_response(response)

            # Apply language-specific post-processing for validation
            if self.source_language.lower() in ["japanese", "日本語"] and self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
                cleaned = self._clean_japanese_artifacts(cleaned)

            # Correct footnote colons
            cleaned = self._correct_footnote_colons(cleaned)

            # Validate using existing method (checks for truncation)
            is_valid, reason = self.validate_output(
                original=content,
                processed=cleaned,
                file_name=f"{file_name} part {part_idx}/{total_parts}" if part_idx and total_parts else file_name
            )

            return is_valid, reason

        try:
            # Use the new generate_with_validation method
            # All retry logic is now handled within the LLMClient
            # Create a new ValidationStrategy instance for thread safety
            from ..processors.validation_strategy import ValidationStrategy
            validation_config = self.config.get('validation_strategy', {})
            validation_config['use_longest_on_failure'] = self.use_longest_on_failure
            thread_local_strategy = ValidationStrategy(validation_config)

            translated_content = self.llm_client.generate_with_validation(
                prompt=multi_part_content,
                model_configs=self.translation_models,
                validator=validator,
                validation_strategy=thread_local_strategy,
                operation_name=operation_name
            )

            # Clean and post-process the final response
            translated_content = self.clean_markdown_response(translated_content)

            # Apply Japanese to Chinese specific post-processing
            if self.source_language.lower() in ["japanese", "日本語"] and self.target_language.lower() in ["chinese", "中文", "chinese simplified", "简体中文"]:
                translated_content = self._clean_japanese_artifacts(translated_content)

            # Correct footnote colon syntax for all translations
            translated_content = self._correct_footnote_colons(translated_content)

            return translated_content

        except Exception as e:
            logger.error(f"Failed to translate {operation_name}: {e}")
            raise
    
    def process_content(
        self,
        content: str,
        file_name: str,
        **kwargs
    ) -> str:
        """
        Process markdown content by translating it.
        
        Args:
            content: The markdown content to translate
            file_name: Name of the file being processed
            **kwargs: Additional arguments
        
        Returns:
            Translated markdown content
        """
        # If this is already a part file, don't look for more parts
        if '.part' in Path(file_name).stem:
            # This is already a part file, translate it directly
            return self._translate_part(content, file_name)
        
        # Check if there are part files (from polisher)
        part_files = self._detect_part_files(file_name)
        
        if part_files:
            logger.info(f"Found {len(part_files)} part files for {file_name}, translating separately")
            translated_parts = []
            output_dir = Path(self.output_dir)
            base_name = Path(file_name).stem

            for part_idx, part_file in enumerate(part_files, 1):
                # Check if this part was already translated (for resume)
                translated_part_file = output_dir / f"{base_name}.part{part_idx}.md"

                if self.resume and translated_part_file.exists():
                    # Part already translated, load it
                    logger.info(f"Skipping {file_name} part {part_idx}/{len(part_files)} (already translated)")
                    with open(translated_part_file, 'r', encoding='utf-8') as f:
                        translated_part = f.read()
                    translated_parts.append(translated_part)
                    continue

                # Read part content
                with open(part_file, 'r', encoding='utf-8') as f:
                    part_content = f.read()

                # Translate the part
                translated_part = self._translate_part(
                    content=part_content,
                    file_name=file_name,
                    part_idx=part_idx,
                    total_parts=len(part_files)
                )

                # Validate this part
                is_valid, reason = self.validate_output(
                    original=part_content,
                    processed=translated_part,
                    file_name=f"{file_name} part {part_idx}/{len(part_files)}"
                )

                if not is_valid:
                    logger.warning(f"Part {part_idx}/{len(part_files)} validation failed: {reason}")

                translated_parts.append(translated_part)

                # Save translated part file
                with open(translated_part_file, 'w', encoding='utf-8') as f:
                    f.write(translated_part)
                logger.debug(f"Saved translated part: {translated_part_file.name}")

            # Combine all parts
            combined = "\n\n".join(translated_parts)
            return combined
        else:
            # No parts, translate as single file
            return self._translate_part(content, file_name)
    
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
    
    def process_file(
        self,
        input_path,
        output_path,
        **kwargs
    ) -> bool:
        """
        Process a single markdown file for translation.
        
        Override to add language information to progress tracking.
        """
        success = super().process_file(input_path, output_path, **kwargs)
        
        if success:
            # Update progress with language info
            file_key = str(input_path.stem)
            progress_key = self.get_progress_key()
            if file_key in self.progress[progress_key]:
                self.progress[progress_key][file_key].update({
                    "source_language": self.source_language,
                    "target_language": self.target_language
                })
                self.save_progress()
        
        return success
    
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
2. **图片链接保持不变**：不要翻译或修改图片路径，如 ![...](../images/xxx.png)
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
2. **Keep image links unchanged**: Do not translate or modify image paths like ![...](../images/xxx.png)
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
