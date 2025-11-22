#!/usr/bin/env python3
"""
Extract entities (characters, places, terms) from PDF for translation consistency.

This module analyzes a PDF (typically Japanese light novel or manga) to extract
all named entities that need consistent translation across chapters.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from google.genai.types import Part
from loguru import logger
from .utils.logging_config import configure_logging
from .utils.network_utils import GeminiClient
from .utils.common import load_config

# Configure logger
logger = configure_logging()


def create_entity_extraction_prompt(book_title: str, language_pair: tuple = ("Japanese", "Chinese")) -> str:
    """
    Create the prompt for entity extraction based on language pair.
    
    Args:
        book_title: Title of the book
        language_pair: Tuple of (source_language, target_language)
    
    Returns:
        The formatted prompt string
    """
    source_lang, target_lang = language_pair
    
    if source_lang == "Japanese" and target_lang == "Chinese":
        return f"""Analyze this Japanese book "{book_title}" and extract ALL important entities that need consistent translation to Chinese.

IMPORTANT: Scan the ENTIRE book thoroughly. Look for:

1. **Characters (人物)**: 
   - ALL named characters (major and minor)
   - Include their full names, nicknames, titles
   - Note gender, role, and brief description
   - Include family relationships

2. **Places (地点)**:
   - Cities, countries, regions
   - Buildings, landmarks, shops
   - Fantasy locations, dungeons, worlds

3. **Organizations (组织)**:
   - Guilds, companies, schools
   - Government bodies, military units
   - Clubs, teams, factions

4. **Special Terms (特殊用语)**:
   - Magic systems, skills, spells
   - Titles, ranks, positions
   - Weapons, items, artifacts
   - Currency, measurements
   - Food, drinks, cultural items
   - Special concepts unique to this world

5. **Races/Species (种族)** (if fantasy):
   - Elves, demons, dragons, etc.
   - Include both singular and plural forms

For each entity, provide:
- Original Japanese (with kanji and kana)
- Reading in hiragana (if applicable)
- Romaji romanization
- Suggested Chinese translation
- Brief description in Chinese
- Category type
- For characters: Include nicknames with their Chinese translations

Return as JSON with this structure:
{{
  "metadata": {{
    "book_title": "{book_title}",
    "total_pages_analyzed": <number>,
    "extraction_complete": true
  }},
  "characters": [
    {{
      "japanese": "緋奈",
      "reading": "ひな",
      "romaji": "Hina",
      "chinese": "绯奈",
      "gender": "female",
      "description": "主人公，精灵少女",
      "role": "protagonist",
      "relationships": ["妹妹: 露娜"],
      "nicknames": [
        {{"japanese": "ひなちゃん", "chinese": "小绯奈"}},
        {{"japanese": "ヒナ様", "chinese": "绯奈大人"}}
      ]
    }}
  ],
  "places": [
    {{
      "japanese": "王都エルフィン",
      "reading": "おうとえるふぃん",
      "romaji": "Outo Erufin",
      "chinese": "王都艾尔芬",
      "description": "精灵王国的首都",
      "type": "city"
    }}
  ],
  "organizations": [
    {{
      "japanese": "冒険者ギルド",
      "reading": "ぼうけんしゃぎるど",
      "romaji": "Boukensha Girudo",
      "chinese": "冒险者公会",
      "description": "管理冒险者的组织"
    }}
  ],
  "terms": [
    {{
      "japanese": "転生",
      "reading": "てんせい",
      "romaji": "tensei",
      "chinese": "转生",
      "description": "重生到异世界",
      "category": "concept"
    }},
    {{
      "japanese": "魔法",
      "reading": "まほう",
      "romaji": "mahou",
      "chinese": "魔法",
      "description": "魔法",
      "category": "magic"
    }}
  ],
  "races": [
    {{
      "japanese": "エルフ",
      "reading": "えるふ",
      "romaji": "erufu",
      "chinese": "精灵",
      "chinese_plural": "精灵们",
      "description": "长耳朵的魔法种族"
    }}
  ],
  "items": [
    {{
      "japanese": "聖剣",
      "reading": "せいけん",
      "romaji": "seiken",
      "chinese": "圣剑",
      "description": "传说中的神圣武器",
      "type": "weapon"
    }}
  ]
}}

IMPORTANT NOTES:
1. Be EXHAUSTIVE - include every named entity you find
2. Maintain consistency in translation style
3. For Chinese translations, prefer commonly used terms in light novel translations
4. Include page numbers where entities first appear if clearly visible
5. Scan the ENTIRE PDF, not just the beginning"""
    
    else:
        # Generic prompt for other language pairs
        return f"""Analyze this book "{book_title}" and extract all important named entities for translation consistency from {source_lang} to {target_lang}.

Extract:
1. Character names (with gender and description)
2. Place names (cities, locations)
3. Organization names
4. Special terminology
5. Important items

Return as JSON with appropriate translations."""


def extract_entities_from_pdf(
    pdf_path: Path,
    book_title: str,
    gemini_client: GeminiClient,
    config: Dict,
    language_pair: tuple = ("Japanese", "Chinese")
) -> Dict:
    """
    Extract entities from PDF using Gemini.
    
    Args:
        pdf_path: Path to the PDF file
        book_title: Title of the book
        gemini_client: Initialized Gemini client
        config: Configuration dictionary
        language_pair: Tuple of (source_language, target_language)
    
    Returns:
        Dictionary containing extracted entities
    """
    logger.info(f"Extracting entities from '{book_title}'...")
    logger.info(f"Language pair: {language_pair[0]} → {language_pair[1]}")
    
    # Create the extraction prompt
    prompt = create_entity_extraction_prompt(book_title, language_pair)
    
    # Read the PDF file
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    
    # Create multimodal input
    parts = [
        prompt,
        Part.from_bytes(data=pdf_data, mime_type="application/pdf"),
    ]
    
    # Get model from config
    model = config.get("entity_extraction_model", config.get("model", "gemini-2.5-flash"))
    logger.info(f"Using model: {model}")
    
    # Configure generation
    generation_config = gemini_client.get_default_config(temperature=0.1)
    generation_config.response_mime_type = "application/json"
    
    # Generate content
    try:
        response_text = gemini_client.generate_content_stream(
            model=model,
            contents=parts,
            config=generation_config,
            operation_name="Entity extraction"
        )
        
        # Parse JSON response
        entities = json.loads(response_text)
        
        # Log summary
        logger.success("Entity extraction completed!")
        if "characters" in entities:
            logger.info(f"Found {len(entities['characters'])} characters")
        if "places" in entities:
            logger.info(f"Found {len(entities['places'])} places")
        if "terms" in entities:
            logger.info(f"Found {len(entities['terms'])} special terms")
        if "organizations" in entities:
            logger.info(f"Found {len(entities['organizations'])} organizations")
        if "races" in entities:
            logger.info(f"Found {len(entities['races'])} races/species")
        if "items" in entities:
            logger.info(f"Found {len(entities['items'])} items")
        
        return entities
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Response text: {response_text[:500]}")
        raise
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        raise


def save_entities(entities: Dict, output_dir: Path):
    """
    Save extracted entities to JSON file.
    
    Args:
        entities: Dictionary of extracted entities
        output_dir: Output directory path
    """
    output_file = output_dir / "translation_entities.json"
    
    # Add metadata if not present
    if "metadata" not in entities:
        entities["metadata"] = {}
    
    entities["metadata"]["extraction_timestamp"] = json.dumps(
        entities["metadata"].get("extraction_timestamp", ""),
        default=str
    )
    
    # Save to file with nice formatting
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    
    logger.success(f"Entities saved to: {output_file}")
    
    # Also create a simplified reference sheet for quick lookup
    reference_file = output_dir / "translation_reference.txt"
    create_reference_sheet(entities, reference_file)


def create_reference_sheet(entities: Dict, output_path: Path):
    """
    Create a human-readable reference sheet from entities.
    
    Args:
        entities: Dictionary of extracted entities
        output_path: Path for the reference file
    """
    lines = []
    lines.append("=" * 60)
    lines.append("TRANSLATION REFERENCE SHEET")
    lines.append("=" * 60)
    lines.append("")
    
    # Characters
    if "characters" in entities and entities["characters"]:
        lines.append("【CHARACTERS / 人物】")
        lines.append("-" * 40)
        for char in entities["characters"]:
            line = f"{char['japanese']}"
            if char.get('reading'):
                line += f" ({char['reading']})"
            line += f" → {char['chinese']}"
            if char.get('gender'):
                line += f" [{char['gender']}]"
            if char.get('description'):
                line += f" - {char['description']}"
            lines.append(line)
        lines.append("")
    
    # Places
    if "places" in entities and entities["places"]:
        lines.append("【PLACES / 地点】")
        lines.append("-" * 40)
        for place in entities["places"]:
            line = f"{place['japanese']}"
            if place.get('reading'):
                line += f" ({place['reading']})"
            line += f" → {place['chinese']}"
            if place.get('description'):
                line += f" - {place['description']}"
            lines.append(line)
        lines.append("")
    
    # Organizations
    if "organizations" in entities and entities["organizations"]:
        lines.append("【ORGANIZATIONS / 组织】")
        lines.append("-" * 40)
        for org in entities["organizations"]:
            line = f"{org['japanese']}"
            if org.get('reading'):
                line += f" ({org['reading']})"
            line += f" → {org['chinese']}"
            if org.get('description'):
                line += f" - {org['description']}"
            lines.append(line)
        lines.append("")
    
    # Terms
    if "terms" in entities and entities["terms"]:
        lines.append("【SPECIAL TERMS / 特殊用语】")
        lines.append("-" * 40)
        for term in entities["terms"]:
            line = f"{term['japanese']}"
            if term.get('reading'):
                line += f" ({term['reading']})"
            line += f" → {term['chinese']}"
            if term.get('category'):
                line += f" [{term['category']}]"
            if term.get('description'):
                line += f" - {term['description']}"
            lines.append(line)
        lines.append("")
    
    # Races
    if "races" in entities and entities["races"]:
        lines.append("【RACES/SPECIES / 种族】")
        lines.append("-" * 40)
        for race in entities["races"]:
            line = f"{race['japanese']}"
            if race.get('reading'):
                line += f" ({race['reading']})"
            line += f" → {race['chinese']}"
            if race.get('chinese_plural'):
                line += f" (pl: {race['chinese_plural']})"
            if race.get('description'):
                line += f" - {race['description']}"
            lines.append(line)
        lines.append("")
    
    # Items
    if "items" in entities and entities["items"]:
        lines.append("【ITEMS / 物品】")
        lines.append("-" * 40)
        for item in entities["items"]:
            line = f"{item['japanese']}"
            if item.get('reading'):
                line += f" ({item['reading']})"
            line += f" → {item['chinese']}"
            if item.get('type'):
                line += f" [{item['type']}]"
            if item.get('description'):
                line += f" - {item['description']}"
            lines.append(line)
        lines.append("")
    
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Reference sheet created: {output_path}")


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Extract entities from PDF for translation consistency"
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input PDF file")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--source-lang",
        default="Japanese",
        help="Source language (default: Japanese)"
    )
    parser.add_argument(
        "--target-lang",
        default="Chinese",
        help="Target language (default: Chinese)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    book_title = config.get("title")
    
    if not book_title:
        # Use filename as fallback
        book_title = Path(args.input).stem
        logger.warning(f"No title in config, using: {book_title}")
    
    # Get API key
    api_key = config.get("google_api_key")
    base_url = config.get("google_base_url")
    if not api_key:
        logger.error("Google API key not found in config.yaml")
        return 1

    # Initialize Gemini client
    gemini_client = GeminiClient(api_key, base_url=base_url)
    
    # Setup output directory
    output_dir = Path("output") / book_title
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get PDF path
    pdf_path = Path(args.input)
    if not pdf_path.exists():
        # Try to find in output directory
        alt_path = output_dir / "input_original.pdf"
        if alt_path.exists():
            pdf_path = alt_path
            logger.info(f"Using PDF from output directory: {pdf_path}")
        else:
            logger.error(f"PDF not found: {args.input}")
            return 1
    
    # Extract entities
    try:
        entities = extract_entities_from_pdf(
            pdf_path=pdf_path,
            book_title=book_title,
            gemini_client=gemini_client,
            config=config,
            language_pair=(args.source_lang, args.target_lang)
        )
        
        # Save results
        save_entities(entities, output_dir)
        
        logger.success("Entity extraction completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
