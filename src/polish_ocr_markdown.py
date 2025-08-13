import json
import yaml
import argparse
import re
import regex  # For fuzzy matching
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from utils.logging_config import configure_logging
from utils.llm_client import LLMClient
from utils.network_utils import GeminiClient
from truncation_detector import (
    detect_truncation, 
    get_truncation_summary,
    count_tokens
)

# Configure logger
logger = configure_logging()


def fuzzy_find_sentence(haystack: str, needle: str, max_edits: int = 3) -> Optional[Tuple[int, int, str]]:
    """
    Find a sentence in text with fuzzy matching, allowing for small differences.
    
    Args:
        haystack: The text to search in
        needle: The sentence to find
        max_edits: Maximum number of character edits allowed
    
    Returns:
        Tuple of (start_pos, end_pos, matched_text) or None if not found
    """
    # First try exact match
    exact_pos = haystack.find(needle)
    if exact_pos != -1:
        return (exact_pos, exact_pos + len(needle), needle)
    
    # Try fuzzy match with regex library
    try:
        # Allow up to max_edits character differences (substitutions, insertions, deletions)
        pattern = f'(?b)({regex.escape(needle)}){{e<={max_edits}}}'
        match = regex.search(pattern, haystack)
        if match:
            return (match.start(), match.end(), match.group(0))
    except Exception as e:
        logger.debug(f"Fuzzy matching failed: {e}")
    
    # Try to find with common escape variations
    variations = [
        needle.replace('&', r'\&'),  # Escaped ampersand
        needle.replace(r'\&', '&'),  # Unescaped ampersand
        needle.replace('"', r'\"'),  # Escaped quotes
        needle.replace(r'\"', '"'),  # Unescaped quotes
        needle.replace("'", r"\'"),  # Escaped single quotes
        needle.replace(r"\'", "'"),  # Unescaped single quotes
    ]
    
    for variant in variations:
        pos = haystack.find(variant)
        if pos != -1:
            return (pos, pos + len(variant), variant)
    
    return None


def load_config(config_path="config.yaml"):
    """Load configuration from config file."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def load_book_structure(book_title):
    """Load the book structure JSON file."""
    structure_path = Path("output") / Path(book_title) / "book_structure.json"
    if structure_path.exists():
        with open(structure_path, "r", encoding="utf-8") as file:
            structure = json.load(file)
        return structure
    return None


def load_or_create_progress(progress_file: Path, markdown_files: List[Path]) -> Dict:
    """Load existing progress or create new progress tracking."""
    if progress_file.exists():
        with open(progress_file, "r") as f:
            progress = json.load(f)
            # Migrate old format to new format if needed
            if "parts_info" not in progress:
                progress["parts_info"] = {}
                # Migrate from old parts_polished format
                if "parts_polished" in progress:
                    for key in progress["parts_polished"]:
                        progress["parts_info"][str(key)] = {"completed": True}
            return progress
    
    # Create new progress
    progress = {
        "parts_info": {},
        "total_files": len(markdown_files)
    }
    return progress


def save_progress(progress_file: Path, progress: Dict):
    """Save progress to file."""
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def post_process_markdown(markdown: str) -> str:
    """
    Post-process the polished markdown to clean up any issues.
    
    Args:
        markdown: The polished markdown text
    
    Returns:
        Cleaned markdown text
    """
    # Remove any leading/trailing whitespace
    markdown = markdown.strip()
    
    # Fix common markdown issues
    # 1. Ensure headers have space after #
    markdown = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', markdown, flags=re.MULTILINE)
    
    # 2. Ensure blank lines around headers
    markdown = re.sub(r'([^\n])\n(#{1,6} )', r'\1\n\n\2', markdown)
    markdown = re.sub(r'(#{1,6} [^\n]+)\n([^\n#])', r'\1\n\n\2', markdown)
    
    # 3. Remove excessive blank lines (more than 2)
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)
    
    # 4. Ensure images have blank lines around them
    markdown = re.sub(r'([^\n])\n(!\[)', r'\1\n\n\2', markdown)
    markdown = re.sub(r'(!\[[^\]]*\]\([^\)]*\))\n([^\n])', r'\1\n\n\2', markdown)
    
    return markdown


def split_content_simple(content: str, max_tokens: int) -> List[str]:
    """
    Simple content splitter that divides content into roughly equal parts.
    
    Args:
        content: The content to split
        max_tokens: Maximum tokens per part
    
    Returns:
        List of content parts
    """
    # Estimate total tokens (rough approximation: 1 token ≈ 4 chars)
    estimated_tokens = len(content) // 4
    
    if estimated_tokens <= max_tokens:
        return [content]
    
    # Calculate number of parts needed
    num_parts = (estimated_tokens // max_tokens) + 1
    
    # Split by paragraphs
    paragraphs = content.split('\n\n')
    
    # Distribute paragraphs evenly
    parts = []
    paras_per_part = len(paragraphs) // num_parts
    
    for i in range(num_parts):
        start_idx = i * paras_per_part
        if i == num_parts - 1:
            # Last part gets all remaining paragraphs
            part = '\n\n'.join(paragraphs[start_idx:])
        else:
            end_idx = start_idx + paras_per_part
            part = '\n\n'.join(paragraphs[start_idx:end_idx])
        
        if part:
            parts.append(part)
    
    return parts if parts else [content]


def split_content_intelligently(content: str, max_tokens: int, gemini_client: GeminiClient) -> List[str]:
    """
    Use LLM to intelligently split content at natural boundaries.
    
    Args:
        content: The content to split
        max_tokens: Maximum tokens per part (used as guideline)
        gemini_client: Gemini client for split detection
    
    Returns:
        List of content parts
    """
    # Estimate total tokens
    estimated_tokens = len(content) // 4
    
    if estimated_tokens <= max_tokens:
        return [content]
    
    # Calculate number of parts needed
    num_parts = max(2, (estimated_tokens // max_tokens) + 1)
    
    logger.info(f"Content has ~{estimated_tokens:,} tokens, splitting into {num_parts} parts")
    
    # Ask LLM to identify good split points with comprehensive rules
    total_tokens = estimated_tokens
    split_prompt = f"""You are helping split a long academic chapter into smaller parts for processing.

The chapter has approximately {total_tokens:,} tokens. While we'd prefer parts under {max_tokens:,} tokens, 
the MOST IMPORTANT criteria are semantic completeness and avoiding citation conflicts.

CRITICAL SPLITTING RULES (in order of priority):

1. **Keep Citations with their Notes/References**:
   - IMPORTANT: Footnotes can appear in two ways:
     a) **Inline footnotes**: Definition appears immediately after citation in the text flow
     b) **End-of-section footnotes**: Definitions collected at the end under "Notes" or "References"
   - For inline footnotes: NEVER split between a citation and its nearby definition
   - For end-of-section footnotes: Keep the entire section WITH its Notes/References in the same part
   - If footnotes are inline, do NOT use them as split points - keep reading until you find a section boundary
   - Split AFTER a complete section with all its footnotes (whether inline or at end)

2. **No Duplicate Citations**:
   - Each footnote number (e.g., [^1], $^1$, ¹) must appear ONLY in one part
   - Both the citation [^1] and its definition [^1]: must be in the SAME part
   - Never split between a citation and its corresponding footnote definition
   - If footnotes are inline (definition immediately follows citation), keep them together as a unit
   - If you see patterns like [^1], [^2], [^3] in text, ensure ALL of them and their definitions stay together

3. **Section Integrity**:
   - Split at major section boundaries (look for ## or ### headings)
   - Keep entire sections together when possible
   - If footnotes are inline within a section, the entire section must stay together
   - Only split at points where NO citations span across the boundary

4. **No Cross-References Between Parts**:
   - A citation in one part should NEVER refer to a footnote definition in another part
   - For inline footnotes, this means keeping the citation and its immediate definition together
   - For end-of-section footnotes, this means keeping the entire section with its footnotes
   - Never have orphaned citations or orphaned footnote definitions

5. **Token Limits** (lowest priority):
   - Aim for parts under {max_tokens:,} tokens if possible
   - But it's OK to exceed this if needed to maintain semantic integrity
   - Aim for roughly {num_parts} parts, but adjust based on content structure

Scan the chapter and identify:
- Whether footnotes are inline (definitions immediately after citations) or collected at section ends
- If inline: Find section boundaries where no footnotes are actively being defined
- If at section ends: Identify which sections have citations and where their Notes/References are
- Natural boundaries where no citations span across
- Major section boundaries that don't break citation-reference pairs

IMPORTANT: If you detect inline footnotes (e.g., [^1] followed shortly by [^1]: definition in the main text flow),
do NOT split near these footnotes. Instead, find major section breaks or topic changes as split points.

Return a JSON array of the EXACT final sentences that mark the end of each part (except the last one).
Choose split points that respect the above priorities. For inline footnotes, split at section boundaries.
For end-of-section footnotes, split AFTER the Notes/References section ends.

Example response:
["[^5]: Johnson, 2019, p. 45.", "This concludes the historical overview."]

Here is the content to analyze:

{content}"""

    try:
        # Use gemini-2.5-pro for the splitting task
        config = gemini_client.get_default_config(temperature=0.1)
        config.response_mime_type = "application/json"
        
        response = gemini_client.generate_content(
            model="gemini-2.5-pro",  # Always use pro for splitting
            contents=split_prompt,
            config=config,
            operation_name="Split chapter"
        )
        
        if not response or not response.text:
            logger.warning("Failed to get LLM split suggestions, falling back to simple split")
            return split_content_simple(content, max_tokens)
        
        # Parse the JSON response
        try:
            split_sentences = json.loads(response.text)
            if not isinstance(split_sentences, list):
                raise ValueError("Response is not a list")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM split response: {e}, falling back to simple split")
            return split_content_simple(content, max_tokens)
        
        # Find the split points in the content
        parts = []
        start_pos = 0
        
        for split_sentence in split_sentences:
            # Use fuzzy matching to find the sentence
            match_result = fuzzy_find_sentence(content, split_sentence)
            
            if match_result:
                end_pos = match_result[1]
                # Include the sentence in the current part
                part = content[start_pos:end_pos].strip()
                if part:
                    parts.append(part)
                start_pos = end_pos
                logger.debug(f"Found split point: '{match_result[2][:50]}...'")
            else:
                logger.warning(f"Could not find split sentence: '{split_sentence[:50]}...'")
        
        # Add the remaining content as the last part
        if start_pos < len(content):
            last_part = content[start_pos:].strip()
            if last_part:
                parts.append(last_part)
        
        # Validate the split
        if len(parts) < 2:
            logger.warning("LLM split resulted in too few parts, falling back to simple split")
            return split_content_simple(content, max_tokens)
        
        # Log part sizes
        for i, part in enumerate(parts, 1):
            part_tokens = len(part) // 4
            logger.info(f"Part {i}/{len(parts)}: ~{part_tokens:,} tokens")
        
        return parts
        
    except Exception as e:
        logger.warning(f"Intelligent split failed: {e}, falling back to simple split")
        return split_content_simple(content, max_tokens)


def polish_markdown_part(
    part_content: str,
    chapter_index: int,
    chapter_title: str,
    part_idx: int,
    total_parts: int,
    output_path: Path,
    llm_client: LLMClient,
    book_title: str,
    skip_truncation_check: bool = False,
    polish_models: Optional[List[Dict]] = None
) -> Tuple[int, str, bool]:
    """
    Polish a single part of markdown content using configured models.
    
    Args:
        part_content: The markdown content to polish
        chapter_index: Chapter number/index
        chapter_title: Title of the chapter
        part_idx: Part number (1-indexed)
        total_parts: Total number of parts for this chapter
        output_path: Path to save the polished content
        llm_client: Unified LLM client
        book_title: Title of the book
        skip_truncation_check: Whether to skip truncation detection
        polish_models: Optional override for model configurations
    
    Returns:
        Tuple of (part_idx, polished_content, success)
    """
    # Create the comprehensive polish prompt
    book_info = f" from the book titled \"{book_title}\"" if book_title else ""
    
    prompt_text = f"""You are an expert document editor. Polish this OCR-extracted markdown content from the chapter "{chapter_title}"{book_info}.

Your tasks:
1. **Remove page headers/footers**: Delete any repeated page numbers, headers, or footers that appear throughout the text (e.g., "Page 123", repeated instances of "{book_title}" as headers/footers, author names before and after page separators).{' Be especially careful to remove occurrences of "' + book_title + '" that appear as page headers or footers.' if book_title else ''}

2. **Handle separators and page breaks**:
   - Remove horizontal separators (---, ***, ___) that were used to separate pages
   - Join sentences that were artificially broken by page boundaries
   - Keep paragraph breaks that are semantically meaningful
   - Remove excessive blank lines (more than 2 consecutive)

3. **Adjust heading hierarchy**:
   - The main chapter title should be # (H1)
   - Subchapter titles should be ## (H2)
   - Sub-sections should be ### (H3), and so on
   - Ensure consistent heading hierarchy throughout

4. **For Japanese text**:
   - Keep ruby text/furigana annotations like: 一人(ひとり), 今更(いまさら), 幼馴染(おさななじみ)
   - Preserve the parentheses format for readings

5. **Fix footnotes**:
   - Convert citation formats (like ${{ }}^{{1}}$, ${{1}}$, [1], ¹, etc.) to standard markdown format [^1]
   - Convert footnote definition formats: change "1{{ }}^{{1}} content" or "${{ }}^{{1}}$ content" to "[^1]: content"
   - CRITICAL: Only include footnotes that ACTUALLY EXIST in the source text below
   - DO NOT create, invent, or add any footnotes that are not present in the source
   - DO NOT try to "complete" or "fill in" missing footnotes
   - Keep the footnote content EXACTLY as it appears - do not modify, shorten, or rewrite the text
   - Keep ALL footnotes you find in the source, even if you don't see their citations
   - Keep the original footnote numbers as they appear (don't renumber)
   - NEVER create new footnotes, NEVER add placeholder content, NEVER modify the footnote text

6. **Organize References and Notes**:
   - ONLY add a "## Notes" section if there are actual footnotes in the source text
   - ONLY add a "## References" section if there are actual bibliographic references in the source text
   - DO NOT add these sections if they don't exist in the source material
   - If footnotes exist: Move ALL footnotes to the end under a "## Notes" heading
   - If bibliographic references exist: Organize them under a "## References" heading AFTER the Notes section
   - Format references properly when they exist:
     - For books: Author(s). (Year). *Title in italics*. Publisher.
     - For articles: Author(s). (Year). "Article title." *Journal Name in italics*, Volume(Issue), pages.
     - For web sources: Author/Organization. (Year). "Title." Website. URL
   - Sort references alphabetically by author's last name
   - Ensure consistent formatting across all references
   - The final structure (when applicable) should be: Main Content → ## Notes (if exists) → ## References (if exists)

Additional requirements:
- Preserve all meaningful content, images, and tables
- Keep markdown formatting for emphasis (*italic*, **bold**)
- Maintain proper markdown syntax for lists, quotes, and code blocks
- Fix obvious OCR errors (e.g., "tlie" → "the", "I1" → "Il")
- Join hyphenated words at line breaks (e.g., "exam-\nple" → "example")
- Preserve the original language (don't translate)"""
    
    # Add context-specific instructions for multi-part chapters
    if total_parts > 1:
        prompt_text += f"""

CONTEXT: This is part {part_idx} of {total_parts} of a multi-part chapter."""
        
        # For part 2 and later, limit heading levels
        if part_idx > 1:
            prompt_text += """

IMPORTANT HEADING RULE FOR CONTINUATION PARTS:
- Since this is a continuation (part 2 or later), your MAXIMUM heading level is ## (H2)
- Do NOT use # (H1) anywhere in your output
- If you see # (H1) headings in the source, convert them to ## (H2)
- Use ## for main sections, ### for subsections, #### for sub-subsections, etc.
- This ensures proper document hierarchy since the chapter title (H1) was already in part 1"""
    
    prompt_text += """

IMPORTANT: Return ONLY the polished markdown content for the CURRENT PART. Do not add any explanations or wrap in code blocks.
IMPORTANT: Don't remove any TABLES or IMAGES unless duplicated.
IMPORTANT: Only include footnotes that are ACTUALLY PRESENT in the source text below. Do not create or invent any footnotes.

Polish the following content:"""

    # Create multi-part content for the LLM
    multi_part_content = [
        {"type": "text", "text": prompt_text},
        {"type": "text", "text": part_content}
    ]
    
    # Try generation with all configured models - use consistent naming
    if total_parts > 1:
        operation_name = f"{chapter_title} part {part_idx}/{total_parts}"
    else:
        operation_name = f"{chapter_title}"
    
    max_attempts = 3
    all_attempts = []
    
    for attempt in range(max_attempts):
        try:
            # Generate polished content
            polished_content = llm_client.generate(
                prompt=multi_part_content,
                model_configs=polish_models,
                operation_name=operation_name
            )
            
            # Store this attempt
            response_tokens = count_tokens(polished_content)
            all_attempts.append((polished_content, response_tokens))
            logger.debug(f"Generated response: {response_tokens} tokens")
            
            # Use smart truncation detection
            if not skip_truncation_check:
                is_truncated, reason, details = detect_truncation(
                    part_content,
                    polished_content,
                    min_token_ratio=0.60,
                    min_unique_preserved_ratio=0.60,
                    allow_deduplication=True
                )
                
                # Log detailed analysis
                summary = get_truncation_summary(is_truncated, reason, details)
                if is_truncated:
                    logger.warning(f"{chapter_title} part {part_idx}/{total_parts} truncation analysis:\n{summary}")
                    if attempt < max_attempts - 1:
                        logger.info(f"Retrying {chapter_title} part {part_idx}/{total_parts} (attempt {attempt + 2}/{max_attempts})...")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        # This is the final attempt and it's truncated
                        # Don't save it yet, let it fall through to the fallback logic
                        logger.warning(f"Final attempt ({attempt + 1}/{max_attempts}) for {chapter_title} part {part_idx}/{total_parts} is truncated")
                        raise Exception(f"Truncation detected on final attempt: {reason}")
                else:
                    logger.info(f"{chapter_title} part {part_idx}/{total_parts} processed successfully:\n{summary}")
            else:
                # Even when skipping truncation check, we want to track attempts for later use
                logger.info(f"Skipping truncation check for {chapter_title} part {part_idx}/{total_parts} (attempt {attempt + 1}/{max_attempts})")
            
            # Post-process
            polished_content = post_process_markdown(polished_content)
            
            # Remove any markdown code blocks if the LLM wrapped the response
            if polished_content.startswith('```'):
                lines = polished_content.split('\n')
                start_idx = 1 if lines[0].startswith('```') else 0
                end_idx = len(lines) - 1 if lines[-1] == '```' else len(lines)
                polished_content = '\n'.join(lines[start_idx:end_idx])
            
            # Save the part file
            if total_parts > 1:
                part_output_path = output_path.with_suffix(f'.part{part_idx}.md')
                logger.success(f"Successfully polished {chapter_title} part {part_idx}/{total_parts}")
            else:
                part_output_path = output_path
                logger.success(f"Successfully polished {chapter_title}")
            
            # Write the file
            with open(part_output_path, 'w', encoding='utf-8') as f:
                f.write(polished_content)
            logger.info(f"Saved to {part_output_path.name}")
            
            # Verify the file was written correctly
            if not part_output_path.exists():
                raise IOError(f"File {part_output_path} not found after writing")
                
            return part_idx, polished_content, True
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{max_attempts} failed for {chapter_title} part {part_idx}/{total_parts}: {e}")
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                # On final failure, use longest response if available
                if all_attempts:
                    all_attempts.sort(key=lambda x: x[1], reverse=True)
                    best_response, best_tokens = all_attempts[0]
                    
                    logger.warning(f"All {max_attempts} attempts failed for {chapter_title} part {part_idx}/{total_parts}")
                    logger.warning(f"Using longest response ({best_tokens:,} tokens) as fallback for {chapter_title} part {part_idx}/{total_parts}")
                    
                    # Post-process the best response
                    best_response = post_process_markdown(best_response)
                    
                    # Save the file
                    if total_parts > 1:
                        part_output_path = output_path.with_suffix(f'.part{part_idx}.md')
                    else:
                        part_output_path = output_path
                    
                    with open(part_output_path, 'w', encoding='utf-8') as f:
                        f.write(best_response)
                    
                    logger.warning(f"Saved fallback response to {part_output_path.name}")
                    return part_idx, best_response, True
                
                # No attempts succeeded at all
                return part_idx, "", False


def process_markdown_file(
    markdown_path: Path,
    output_path: Path,
    llm_client: LLMClient,
    gemini_client: GeminiClient,
    book_title: str,
    polish_models: Optional[List[Dict]] = None
) -> bool:
    """
    Process a single markdown file, splitting if necessary.
    
    Args:
        markdown_path: Path to input markdown file
        output_path: Path to output polished markdown
        llm_client: Unified LLM client for polishing
        gemini_client: Gemini client for intelligent splitting
        book_title: Title of the book
        polish_models: Optional override for model configurations
    
    Returns:
        True if successful, False otherwise
    """
    # Extract chapter info from filename
    filename = markdown_path.stem
    
    # Determine chapter type
    if filename == "front_matter":
        chapter_index = 0
        chapter_title = "Front Matter"
    elif filename == "back_matter":
        chapter_index = 999
        chapter_title = "Back Matter"
    else:
        # Extract chapter number
        match = re.search(r'chapter_(\d+)', filename)
        if match:
            chapter_index = int(match.group(1))
            chapter_title = f"Chapter {chapter_index}"
        else:
            chapter_index = 0
            chapter_title = filename
    
    logger.info(f"Processing {chapter_title} from {markdown_path.name}")
    
    # Read the content
    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if not content.strip():
        logger.warning(f"File {markdown_path.name} is empty, skipping")
        return False
    
    # Dynamically determine max_tokens_per_part based on models
    # Default to conservative limit unless we're certain no limited models are used
    max_tokens_per_part = 10000
    
    # Only increase limit if we can verify no limited-context models are being used
    logger.debug(f"polish_models value: {polish_models}")
    if polish_models:
        # Check if any model contains indicators of smaller/faster models
        # Use more specific patterns to avoid false positives
        limited_model_patterns = [
            "flash",  # e.g., gemini-1.5-flash
            "haiku",  # e.g., claude-3-haiku
            "-mini",  # e.g., gpt-4-mini (but not gemini since it doesn't have -mini)
        ]
        
        # Debug: log the models being checked
        model_names = [model_config.get("model", "") for model_config in polish_models]
        logger.debug(f"Checking models for limited context indicators: {model_names}")
        
        has_limited_model = any(
            any(pattern in model_config.get("model", "").lower() for pattern in limited_model_patterns)
            for model_config in polish_models
        )
        
        if not has_limited_model:
            max_tokens_per_part = 30000
            logger.info(f"Using max_tokens_per_part=30000 (no limited-context models detected)")
        else:
            # Log which model triggered the limited context
            for model_config in polish_models:
                model_name = model_config.get("model", "").lower()
                for pattern in limited_model_patterns:
                    if pattern in model_name:
                        logger.debug(f"Limited context triggered by '{pattern}' in model '{model_config.get('model', '')}'")
            logger.info(f"Using max_tokens_per_part=10000 (limited-context model detected)")
    else:
        # Can't determine models, use conservative limit
        logger.info(f"Using max_tokens_per_part=10000 (default conservative limit)")
    
    # Check content size and split if necessary
    estimated_tokens = len(content) // 4
    
    if estimated_tokens > max_tokens_per_part:
        logger.info(f"Content has ~{estimated_tokens:,} tokens, splitting into parts")
        parts = split_content_intelligently(content, max_tokens_per_part, gemini_client)
    else:
        parts = [content]
    
    logger.info(f"Processing {len(parts)} part(s) for {chapter_title}")
    
    # Process each part
    all_parts_success = True
    polished_parts = []
    
    for part_idx, part_content in enumerate(parts, 1):
        _, polished_content, success = polish_markdown_part(
            part_content=part_content,
            chapter_index=chapter_index,
            chapter_title=chapter_title,
            part_idx=part_idx,
            total_parts=len(parts),
            output_path=output_path,
            llm_client=llm_client,
            book_title=book_title,
            skip_truncation_check=False,  # Always check truncation
            polish_models=polish_models
        )
        
        if success:
            polished_parts.append(polished_content)
        else:
            all_parts_success = False
            logger.error(f"Failed to process part {part_idx} of {chapter_title}")
    
    # If multiple parts, combine them
    if len(parts) > 1 and all_parts_success:
        combined_content = "\n\n".join(polished_parts)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_content)
        logger.success(f"Combined {len(parts)} parts into {output_path.name}")
        
        # Clean up part files
        for part_idx in range(1, len(parts) + 1):
            part_file = output_path.with_suffix(f'.part{part_idx}.md')
            if part_file.exists():
                part_file.unlink()
    
    return all_parts_success


def main():
    parser = argparse.ArgumentParser(description="Polish OCR-extracted markdown files")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume from previous progress")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of concurrent workers")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    book_title = config.get("title")
    
    if not book_title:
        logger.error("No title found in config.yaml")
        return
    
    # Initialize clients
    llm_client = LLMClient(config)
    
    # Initialize Gemini client for splitting (if available)
    gemini_client = None
    if config.get("google_api_key"):
        gemini_client = GeminiClient(config["google_api_key"])
    
    # Setup directories
    ocr_dir = Path("output") / book_title / "ocr_markdown"
    polished_dir = Path("output") / book_title / "polished_markdown"
    polished_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all markdown files to process
    markdown_files = sorted(ocr_dir.glob("*.md"))
    
    # Separate into categories
    chapter_files = []
    front_matter = None
    back_matter = None
    
    for f in markdown_files:
        if f.stem == "front_matter":
            front_matter = f
        elif f.stem == "back_matter":
            back_matter = f
        else:
            chapter_files.append(f)
    
    # Combine all files in processing order
    all_files = []
    if front_matter:
        all_files.append(front_matter)
    all_files.extend(chapter_files)
    if back_matter:
        all_files.append(back_matter)
    
    if not all_files:
        logger.error(f"No markdown files found in {ocr_dir}")
        return
    
    logger.info(f"Found {len(all_files)} markdown files to polish")
    
    # Setup progress tracking
    progress_file = polished_dir / "polish_progress.json"
    progress = load_or_create_progress(progress_file, all_files)
    
    # Get model configs
    polish_models = config.get("polish_models")
    
    # Process files with thread pool
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        
        for markdown_path in all_files:
            # Check if already processed
            file_key = str(markdown_path.stem)
            
            if args.resume and file_key in progress["parts_info"]:
                if progress["parts_info"][file_key].get("completed", False):
                    logger.info(f"Skipping {markdown_path.name} (already processed)")
                    continue
            
            # Submit task
            output_path = polished_dir / markdown_path.name
            future = executor.submit(
                process_markdown_file,
                markdown_path,
                output_path,
                llm_client,
                gemini_client,
                book_title,
                polish_models
            )
            futures.append((future, file_key))
        
        # Process completed tasks
        for future, file_key in futures:
            try:
                success = future.result()
                if success:
                    progress["parts_info"][file_key] = {"completed": True}
                    save_progress(progress_file, progress)
                else:
                    logger.error(f"Failed to process {file_key}")
                    progress["parts_info"][file_key] = {"completed": False}
                    save_progress(progress_file, progress)
            except Exception as e:
                logger.error(f"Error processing {file_key}: {e}")
                progress["parts_info"][file_key] = {"completed": False, "error": str(e)}
                save_progress(progress_file, progress)
    
    # Final summary
    completed = sum(1 for info in progress["parts_info"].values() if info.get("completed", False))
    total = len(all_files)
    
    logger.info(f"\n=== Polish Summary ===")
    logger.info(f"Completed: {completed}/{total} files")
    
    if completed < total:
        failed = [k for k, v in progress["parts_info"].items() if not v.get("completed", False)]
        logger.warning(f"Failed files: {', '.join(failed)}")
    else:
        logger.success("All files polished successfully!")
    
    # Log safety block statistics
    safety_stats = llm_client.get_safety_stats()
    if safety_stats:
        logger.info("\n=== Safety Block Statistics ===")
        for provider, blocked_count in safety_stats.items():
            logger.info(f"{provider}: {blocked_count} operations blocked for safety")


if __name__ == "__main__":
    main()