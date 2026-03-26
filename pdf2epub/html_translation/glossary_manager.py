"""
Glossary Manager: Long-term and short-term glossary memory for novel translation.

Provides:
- Long-term store: keyed by full Japanese name, with aliases for recall
- Short-term memory: previous chapter's glossary for continuity
- Recall: exact string match of keys + aliases against chapter text
- Extract: post-translation glossary generation via LLM (cache hit on same prefix)
- Dedup: LLM-based pronoun/generic cleaning + duplicate merging with persistent state
- Version history: tracks which chapter updated each entry

Independent of any specific translation flow — can be reused by PDF/EPUB pipelines.
"""

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..utils.common import parse_llm_json

logger = logging.getLogger(__name__)

GLOSSARY_EXTRACT_PROMPT = """\
你是轻小说术语表管理器。根据以下已完成的翻译，提取/更新术语表。

输出JSON数组，每个条目：
- key: 日文全名（越完整越好，如"宮崎薰"而非"宮崎"）
- zh_name: 中文翻译名
- aliases: 该角色的其他称呼，每个alias包含日文原文和对应中文翻译，格式：[{{"ja": "日文称呼", "zh": "中文翻译"}}]
- description: 简要描述，包括：性别、身份、与主角关系、当前状态

规则：
- 只收录重要角色、地名、专有名词
- key 必须是专有名词（人名、地名、组织名等），不能是纯粹的代词（俺、僕、彼女）
- aliases 可以包括：姓、名、昵称、以及该角色的专属描述性称呼（如"猫头鹰少女"、"金发少女"等绰号）
- aliases 不要包括纯粹的人称代词（俺、私、彼女等）
- 每个alias的zh翻译必须与本章译文中的实际用词一致
- 不收录章节标题、书名、出版社等元信息
- description 控制在一两句话
- 如果和已有术语表有冲突，以本章翻译为准

{existing_section}\
本章原文：
{source_text}

本章译文：
{translated_text}"""

# Patterns for chapter titles / metadata (should not be glossary entries)
_METADATA_PATTERNS = [
    re.compile(r"第[一二三四五六七八九十百千\d]+[章節巻話]"),  # 第一章, 第2話
    re.compile(r"^(プロローグ|エピローグ|Prologue|Epilogue|あとがき|序章|終章)$", re.I),
]


def _normalize_aliases(aliases: list) -> List[Dict]:
    """Normalize aliases to structured format [{ja, zh}].

    Handles backward compatibility: plain strings become {"ja": str, "zh": ""}.
    """
    result = []
    for a in aliases:
        if isinstance(a, dict) and "ja" in a:
            result.append({"ja": a["ja"], "zh": a.get("zh", "")})
        elif isinstance(a, str) and a:
            result.append({"ja": a, "zh": ""})
    return result


def _alias_ja_keys(aliases: list) -> List[str]:
    """Extract Japanese keys from aliases (handles both formats)."""
    result = []
    for a in aliases:
        if isinstance(a, dict) and "ja" in a:
            result.append(a["ja"])
        elif isinstance(a, str) and a:
            result.append(a)
    return result


def _is_metadata_key(key: str) -> bool:
    """Check if a glossary key is a chapter title or metadata term."""
    for pat in _METADATA_PATTERNS:
        if pat.search(key):
            return True
    return False


GLOSSARY_COMPRESS_PROMPT = """\
以下术语表条目过多，请精简到最重要的条目，控制总长度在 {max_tokens} tokens 以内。
保留所有主要角色和关键术语，删除次要/一次性角色。
输出格式和输入相同（JSON数组）。

{glossary_json}"""

PRONOUN_IDENTIFY_PROMPT = """\
以下是术语表中出现的所有别名（aliases）。请识别其中哪些是代词或泛称。

判定标准：这个词本身是否是单独的通用词，不含任何专有成分？
- "国王" → 泛称（任何国家都有国王）
- "ヘタリア王国の王" → 不是泛称（含有专有名词）
- "勇者" → 泛称
- "炎の勇者" → 不是泛称（专属称号）
- "田中先生" → 不是泛称（人名+敬称）
- "先生" → 泛称

只输出代词和泛称，其余不要列出。
输出JSON数组：[{{"alias": "俺", "type": "pronoun"}}, {{"alias": "国王", "type": "generic_title"}}, ...]

别名列表（格式：别名 ← 所属条目）：
{aliases}"""

DESC_COMPRESS_PROMPT = """\
以下角色描述过长，请压缩到{max_chars}字以内，保留最重要的信息（性别、身份、与主角关系），去掉具体章节剧情细节。

角色：{key}（{zh_name}）
当前描述：{description}

只输出压缩后的描述文字，不要输出其他内容。"""

# Max chars for a single entry's description before triggering compression
DESC_COMPRESS_THRESHOLD = 200

DEDUP_PROMPT = """\
以上是本章术语表提取结果。现在请对术语表进行去重和代词/泛称清理。

以下条目可能存在重复或包含非专属代词/泛称别名：

{grouped_entries}

任务：
1. 同一组中，如果实际上是同一个角色/实体，请合并为一个条目（选择最完整的key，另一个key变为alias）
2. 对于标注了代词/泛称的alias，根据原文判断是否为该角色专属称呼。如果确实是该角色在本作中的专属称呼（例如全书只有这一个角色使用"俺"），请保留。只删除确实非专属的。倾向于保留——误保留的代价远小于误删除。
3. 如果key是同一个角色的不同写法（如"穂"和"穗"），合并

输出处理后的JSON数组（和之前格式相同：key, zh_name, aliases, description）。只输出被修改过的条目。未修改的条目不要输出。如果没有任何修改，输出空数组[]。\
{error_context}"""


def _make_sorted_pair(a: str, b: str) -> Tuple[str, str]:
    """Make a sorted pair for consistent lookup."""
    return (min(a, b), max(a, b))


class GlossaryManager:
    """Manages long-term and short-term glossary memory for novel translation."""

    def __init__(
        self,
        output_dir: Path,
        llm_client,
        model_configs: List[Dict],
        max_tokens: int = 1000,
        extract_retries: int = 2,
        dedup_retries: int = 2,
    ):
        """
        Args:
            output_dir: Directory for persisting glossary files.
            llm_client: LLMClient instance for LLM calls.
            model_configs: Model configs for LLM calls.
            max_tokens: Max token budget for per-chapter glossary output.
            extract_retries: Max retries for glossary extraction.
            dedup_retries: Max retries for dedup Call 2.
        """
        self.output_dir = output_dir
        self.llm_client = llm_client
        self.model_configs = model_configs
        self.max_tokens = max_tokens
        self.extract_retries = extract_retries
        self.dedup_retries = dedup_retries

        self.store_path = output_dir / "glossary_store.json"
        self.prev_chapter_path = output_dir / "glossary_prev.txt"
        self.dedup_state_path = output_dir / "dedup_state.json"
        self.log_dir = output_dir / "logs" / "glossary"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.store: Dict[str, dict] = {}
        self.prev_chapter: str = ""

        # Dedup state — persisted across chapters
        self.dedup_state: Dict = {
            "confirmed_not_dup": [],   # list of [key1, key2] sorted pairs
            "exclusive_aliases": {},   # {key: [aliases]} — confirmed exclusive
            "removed_aliases": {},     # {key: [aliases]} — blacklist, never re-add
            "known_pronouns": [],      # Call 1 results: confirmed pronoun/generic aliases
            "checked_aliases": [],     # All aliases ever sent to Call 1 (pronoun + non-pronoun)
        }

    def load(self):
        """Load store, prev_chapter, and dedup_state from disk. Tolerant of missing files."""
        if self.store_path.exists():
            try:
                self.store = json.loads(self.store_path.read_text(encoding="utf-8"))
                logger.info(f"Loaded glossary store: {len(self.store)} entries")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load glossary store: {e}")
                self.store = {}

        if self.prev_chapter_path.exists():
            self.prev_chapter = self.prev_chapter_path.read_text(encoding="utf-8")

        if self.dedup_state_path.exists():
            try:
                self.dedup_state = json.loads(self.dedup_state_path.read_text(encoding="utf-8"))
                logger.info(
                    f"Loaded dedup state: {len(self.dedup_state.get('confirmed_not_dup', []))} confirmed pairs, "
                    f"{len(self.dedup_state.get('known_pronouns', []))} known pronouns"
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load dedup state: {e}")

    def load_initial_glossary(self, path: Path):
        """Load an initial glossary file (from a previous volume or manual creation).

        Accepts either:
        - JSON file matching glossary_store.json format
        - Plain text with lines like: "日文名 → 中文名：描述"
        """
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return

        # Try JSON first
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                # glossary_store.json format
                for key, entry in data.items():
                    if key not in self.store:
                        self.store[key] = entry
                logger.info(f"Loaded {len(data)} entries from initial glossary (JSON)")
                return
        except json.JSONDecodeError:
            pass

        # Plain text: "key → zh_name：description"
        count = 0
        for line in content.splitlines():
            line = line.strip()
            if not line or "→" not in line:
                continue
            parts = line.split("→", 1)
            key = parts[0].strip()
            rest = parts[1].strip()
            zh_name = rest.split("：", 1)[0].strip() if "：" in rest else rest.split(":", 1)[0].strip()
            desc = rest.split("：", 1)[1].strip() if "：" in rest else (rest.split(":", 1)[1].strip() if ":" in rest else "")
            if key and key not in self.store:
                self.store[key] = {
                    "zh_name": zh_name,
                    "aliases": [],
                    "description": desc,
                    "updated_by": "initial",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "history": [],
                }
                count += 1
        if count:
            logger.info(f"Loaded {count} entries from initial glossary (text)")

    def save(self):
        """Persist store, prev_chapter, and dedup_state to disk."""
        self.store_path.write_text(
            json.dumps(self.store, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.prev_chapter_path.write_text(self.prev_chapter, encoding="utf-8")

        # Serialize dedup_state (convert sets to lists for JSON)
        state_serializable = {
            "confirmed_not_dup": self.dedup_state.get("confirmed_not_dup", []),
            "exclusive_aliases": {
                k: sorted(v) if isinstance(v, set) else v
                for k, v in self.dedup_state.get("exclusive_aliases", {}).items()
            },
            "removed_aliases": {
                k: sorted(v) if isinstance(v, set) else v
                for k, v in self.dedup_state.get("removed_aliases", {}).items()
            },
            "known_pronouns": sorted(set(self.dedup_state.get("known_pronouns", []))),
            "checked_aliases": sorted(set(self.dedup_state.get("checked_aliases", []))),
        }
        self.dedup_state_path.write_text(
            json.dumps(state_serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def recall(self, source_text: str) -> str:
        """Recall relevant glossary entries for a chapter.

        Exact string match all keys + aliases against source_text.
        Returns formatted glossary string for prompt injection.
        """
        if not self.store:
            return self.prev_chapter

        matched = []
        for key, entry in self.store.items():
            # Check if key appears in source
            if key in source_text:
                matched.append((key, entry))
                continue
            # Check aliases (by Japanese key)
            for ja_key in _alias_ja_keys(entry.get("aliases", [])):
                if ja_key in source_text:
                    matched.append((key, entry))
                    break

        # Format matched entries with structured alias mapping
        lines = []
        for key, entry in matched:
            zh = entry.get("zh_name", "")
            desc = entry.get("description", "")
            lines.append(f"{key} → {zh}：{desc}")
            for a in _normalize_aliases(entry.get("aliases", [])):
                if a["zh"]:
                    lines.append(f"  {a['ja']} → {a['zh']}")
                else:
                    lines.append(f"  {a['ja']}")

        recalled = "\n".join(lines)

        # Combine with short-term memory
        parts = []
        if recalled:
            parts.append(f"术语表：\n{recalled}")
        if self.prev_chapter:
            parts.append(f"上一章术语：\n{self.prev_chapter}")

        return "\n\n".join(parts)

    # ─── Dedup helpers ───

    def _get_confirmed_pairs(self) -> Set[Tuple[str, str]]:
        """Get confirmed_not_dup as a set of sorted tuples for O(1) lookup."""
        return {
            _make_sorted_pair(p[0], p[1])
            for p in self.dedup_state.get("confirmed_not_dup", [])
            if isinstance(p, (list, tuple)) and len(p) == 2
        }

    def _get_exclusive_aliases(self) -> Dict[str, Set[str]]:
        """Get exclusive_aliases as {key: set(aliases)}."""
        return {
            k: set(v) if isinstance(v, list) else v
            for k, v in self.dedup_state.get("exclusive_aliases", {}).items()
        }

    def _get_removed_aliases(self) -> Dict[str, Set[str]]:
        """Get removed_aliases as {key: set(aliases)}."""
        return {
            k: set(v) if isinstance(v, list) else v
            for k, v in self.dedup_state.get("removed_aliases", {}).items()
        }

    def _get_known_pronouns(self) -> Set[str]:
        """Get known_pronouns as a set."""
        return set(self.dedup_state.get("known_pronouns", []))

    def _dedup_entries(
        self,
        entries: List[Dict],
        chapter_id: str,
        extraction_prompt: str,
        extraction_response: str,
    ) -> List[Dict]:
        """Dedup and clean entries via two LLM calls with persistent state.

        Call 1 (lightweight): identify pronouns/generics from NEW aliases only.
        Call 2 (cache-friendly): grouped entries → merge duplicates, remove non-exclusive aliases.

        Uses dedup_state to skip already-resolved groups and cache pronoun knowledge.
        Infers new state from Call 2 input/output diff.
        """
        # Collect all alias ja keys from new entries + existing store
        all_ja_aliases = set()
        for entry in entries:
            all_ja_aliases.update(_alias_ja_keys(entry.get("aliases", [])))
        for store_entry in self.store.values():
            all_ja_aliases.update(_alias_ja_keys(store_entry.get("aliases", [])))

        if not all_ja_aliases:
            return entries

        # Build combined entry pool: new entries + existing store entries
        entry_pool = {}  # key → entry dict (with "key" field)
        for key, se in self.store.items():
            entry_pool[key] = {"key": key, **{k: v for k, v in se.items() if k not in ("updated_by", "timestamp", "history")}}
        for entry in entries:
            key = entry.get("key", "").strip()
            if key:
                entry_pool[key] = entry

        # Build alias (ja) → [keys] mapping
        alias_to_keys = defaultdict(set)
        for key, entry in entry_pool.items():
            for ja_key in _alias_ja_keys(entry.get("aliases", [])):
                alias_to_keys[ja_key].add(key)

        # Find shared aliases (appear in 2+ entries)
        shared_aliases = {a: keys for a, keys in alias_to_keys.items() if len(keys) >= 2}

        # NOTE: Call 1 (pronoun identification) and removed_aliases are disabled.
        # Current LLM quality is high enough that extraction rarely includes pure pronouns.
        # More importantly, removing shared aliases (like 母さん) from recall is counterproductive:
        # it's better to recall multiple entries sharing an alias than to recall none.
        # If future books show pronoun pollution in extraction, re-enable Call 1 here.
        # See: PRONOUN_IDENTIFY_PROMPT, _get_known_pronouns(), _get_removed_aliases()
        flagged_aliases = set()
        pronoun_entries = {}

        # --- Filter shared alias groups using confirmed_not_dup ---
        confirmed_pairs = self._get_confirmed_pairs()

        filtered_shared = {}
        for alias, keys in shared_aliases.items():
            # Check if all pairs in this group are confirmed not-dup
            group_resolved = True
            keys_list = sorted(keys)
            for i in range(len(keys_list)):
                for j in range(i + 1, len(keys_list)):
                    if _make_sorted_pair(keys_list[i], keys_list[j]) not in confirmed_pairs:
                        group_resolved = False
                        break
                if not group_resolved:
                    break

            if not group_resolved:
                filtered_shared[alias] = keys

        # If no shared aliases to dedup, return as-is
        if not filtered_shared:
            logger.info(f"  Glossary dedup: no shared aliases to resolve for {chapter_id}")
            return entries

        # --- Build grouped entries text for Call 2 ---
        grouped_text = self._build_dedup_groups(entry_pool, filtered_shared, pronoun_entries, flagged_aliases)

        # --- Call 2: Dedup with retry ---
        dedup_result = self._call_dedup_with_retry(
            extraction_prompt, extraction_response, grouped_text,
            chapter_id, entry_pool, filtered_shared, pronoun_entries, flagged_aliases,
        )
        if dedup_result is None:
            return entries  # All retries failed

        # Save dedup log
        dedup_log = self.log_dir / f"{chapter_id}_dedup.json"
        dedup_log.write_text(
            json.dumps({"dedup_result": dedup_result, "groups": grouped_text}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # --- Infer state from diff ---
        self._infer_dedup_state(filtered_shared, pronoun_entries, dedup_result, entry_pool)

        if not dedup_result:
            # Empty result = no modifications. All entries in groups are confirmed as-is.
            logger.info(f"  Glossary dedup: no modifications for {chapter_id}")
            return entries

        # --- Apply dedup results ---
        return self._apply_dedup_results(entries, dedup_result)

    def _call_dedup_with_retry(
        self,
        extraction_prompt: str,
        extraction_response: str,
        grouped_text: str,
        chapter_id: str,
        entry_pool: Dict,
        shared_aliases: Dict,
        pronoun_entries: Dict,
        flagged_aliases: Set,
    ) -> Optional[List[Dict]]:
        """Call 2 with retry on catastrophic failures."""
        error_context = ""

        for attempt in range(self.dedup_retries + 1):
            try:
                messages = [
                    {"role": "user", "content": extraction_prompt},
                    {"role": "assistant", "content": extraction_response},
                    {"role": "user", "content": DEDUP_PROMPT.format(
                        grouped_entries=grouped_text,
                        error_context=error_context,
                    )},
                ]
                dedup_response = self.llm_client.generate(
                    prompt=messages,
                    model_configs=self.model_configs,
                    operation_name=f"Glossary dedup {chapter_id} (attempt {attempt + 1})",
                )
                dedup_result = parse_llm_json(
                    dedup_response,
                    save_dir=self.log_dir,
                    operation_name=f"Glossary dedup {chapter_id}",
                )
                if not isinstance(dedup_result, list):
                    dedup_result = []
            except Exception as e:
                logger.warning(f"Glossary dedup failed for {chapter_id} (attempt {attempt + 1}): {e}")
                if attempt < self.dedup_retries:
                    continue
                return None

            # Catastrophic checks
            error = self._check_catastrophic(dedup_result, entry_pool, shared_aliases, pronoun_entries, flagged_aliases)
            if error:
                logger.warning(f"  Glossary dedup catastrophic failure (attempt {attempt + 1}): {error}")
                if attempt < self.dedup_retries:
                    error_context = f"\n\n注意：上次输出有问题：{error}。请修正。"
                    continue
                logger.warning(f"  Glossary dedup: all {self.dedup_retries + 1} attempts failed, using original entries")
                return None

            logger.info(f"  Glossary dedup: {len(dedup_result)} entries modified")
            return dedup_result

        return None

    def _check_catastrophic(
        self,
        dedup_result: List[Dict],
        entry_pool: Dict,
        shared_aliases: Dict,
        pronoun_entries: Dict,
        flagged_aliases: Set,
    ) -> Optional[str]:
        """Check for catastrophic dedup failures. Returns error string or None."""
        if not dedup_result:
            return None  # Empty = no modifications, always valid

        dedup_by_key = {e["key"]: e for e in dedup_result if isinstance(e, dict) and "key" in e}
        all_dedup_aliases = set()
        for e in dedup_result:
            if isinstance(e, dict):
                all_dedup_aliases.update(_alias_ja_keys(e.get("aliases", [])))

        # Check 1: new keys must exist in entry pool or be an alias of an entry pool key
        all_pool_keys = set(entry_pool.keys())
        all_pool_aliases = set()
        for entry in entry_pool.values():
            all_pool_aliases.update(_alias_ja_keys(entry.get("aliases", [])))

        for key in dedup_by_key:
            if key not in all_pool_keys and key not in all_pool_aliases:
                return f"输出了未知的key \"{key}\"，不在任何已有条目中"

        # Check 2: for each shared alias group, the alias should still exist in at least
        # one result entry (unless it was a flagged pronoun that got removed from all)
        for alias, keys in shared_aliases.items():
            if alias in flagged_aliases:
                continue  # OK for flagged aliases to be fully removed
            # Check if alias still exists in any entry (modified or original)
            alias_survives = False
            for key in keys:
                if key in dedup_by_key:
                    if alias in _alias_ja_keys(dedup_by_key[key].get("aliases", [])):
                        alias_survives = True
                        break
                else:
                    # Entry not modified — alias still present in original
                    alias_survives = True
                    break
            if not alias_survives:
                return f"alias \"{alias}\" 从所有条目中消失了，但它不是代词/泛称"

        return None

    def _infer_dedup_state(
        self,
        shared_aliases: Dict[str, set],
        pronoun_entries: Dict[str, List[str]],
        dedup_result: List[Dict],
        entry_pool: Dict,
    ):
        """Infer confirmed_not_dup, exclusive_aliases, removed_aliases from Call 2 diff."""
        dedup_by_key = {e["key"]: e for e in dedup_result if isinstance(e, dict) and "key" in e}

        # Collect all alias ja keys in dedup results (to detect absorbed entries)
        all_result_aliases = set()
        for e in dedup_result:
            if isinstance(e, dict):
                all_result_aliases.update(_alias_ja_keys(e.get("aliases", [])))

        confirmed = self._get_confirmed_pairs()
        exclusive = self._get_exclusive_aliases()
        removed = self._get_removed_aliases()

        # 1. Infer confirmed_not_dup from shared alias groups
        for alias, keys in shared_aliases.items():
            # Check which keys survived (not absorbed)
            surviving = set()
            for key in keys:
                if key in dedup_by_key:
                    surviving.add(key)  # Modified but still primary
                elif key not in all_result_aliases:
                    surviving.add(key)  # Not modified, not absorbed — still exists
                # else: key became an alias in result — it was absorbed (merged)

            # All surviving pairs are confirmed not-dup
            surviving_list = sorted(surviving)
            for i in range(len(surviving_list)):
                for j in range(i + 1, len(surviving_list)):
                    confirmed.add(_make_sorted_pair(surviving_list[i], surviving_list[j]))

        # NOTE: exclusive/removed alias inference disabled (see comment in _dedup_entries)
        # Kept for reference in case pronoun filtering is re-enabled for other books.

        # Save back
        self.dedup_state["confirmed_not_dup"] = sorted(
            [list(p) for p in confirmed], key=lambda x: (x[0], x[1])
        )
        self.dedup_state["exclusive_aliases"] = {
            k: sorted(v) for k, v in exclusive.items() if v
        }
        self.dedup_state["removed_aliases"] = {
            k: sorted(v) for k, v in removed.items() if v
        }

    def _build_dedup_groups(
        self,
        entry_pool: Dict[str, Dict],
        shared_aliases: Dict[str, set],
        pronoun_entries: Dict[str, List[str]],
        flagged_aliases: set,
    ) -> str:
        """Build the grouped entries text for the dedup prompt."""
        sections = []
        covered_keys = set()  # track which entries are already in a group

        # Group 1: entries sharing the same alias
        for alias, keys in sorted(shared_aliases.items()):
            lines = [f'alias为"{alias}"的:']
            for key in sorted(keys):
                entry = entry_pool.get(key, {})
                entry_json = json.dumps(entry, ensure_ascii=False)
                note = ""
                if alias in flagged_aliases:
                    note = f"（备注：alias里有\"{alias}\"，如果非该角色专属则删除）"
                lines.append(f"  {entry_json}{note}")
                covered_keys.add(key)
            sections.append("\n".join(lines))

        # Group 2: entries with pronoun aliases but not already covered by shared alias groups
        pronoun_only = []
        for key, flagged in sorted(pronoun_entries.items()):
            if key in covered_keys:
                continue
            entry = entry_pool.get(key, {})
            entry_json = json.dumps(entry, ensure_ascii=False)
            flagged_str = "、".join(f'"{a}"' for a in flagged)
            pronoun_only.append(f"  {entry_json}（备注：alias里有{flagged_str}，如果非该角色专属则删除）")
            covered_keys.add(key)

        if pronoun_only:
            sections.append("alias不相同，但包含代词/泛称的:\n" + "\n".join(pronoun_only))

        return "\n\n".join(sections)

    def _apply_dedup_results(self, entries: List[Dict], dedup_result: List[Dict]) -> List[Dict]:
        """Apply dedup results to the entries list and update store for absorbed entries."""
        dedup_by_key = {e["key"]: e for e in dedup_result if isinstance(e, dict) and "key" in e}

        all_dedup_aliases = set()
        for de in dedup_result:
            if isinstance(de, dict):
                all_dedup_aliases.update(_alias_ja_keys(de.get("aliases", [])))

        # Update store entries modified by dedup (store-only entries, not in new entries)
        new_keys = {e.get("key", "").strip() for e in entries}
        for key, dedup_entry in dedup_by_key.items():
            if key in self.store and key not in new_keys:
                old = self.store[key]
                old["aliases"] = [a for a in dedup_entry.get("aliases", []) if a]
                if dedup_entry.get("zh_name"):
                    old["zh_name"] = dedup_entry["zh_name"]
                logger.info(f"  Glossary dedup: updated store entry '{key}' aliases → {old['aliases']}")

        # Remove absorbed entries from store
        for store_key in list(self.store.keys()):
            if store_key in all_dedup_aliases and store_key not in dedup_by_key:
                logger.info(f"  Glossary dedup: removing absorbed store entry '{store_key}'")
                del self.store[store_key]

        # Update entries list
        updated_entries = []
        seen_keys = set()
        for entry in entries:
            key = entry.get("key", "").strip()
            if key in dedup_by_key:
                updated_entries.append(dedup_by_key[key])
                seen_keys.add(key)
            elif key in all_dedup_aliases:
                logger.info(f"  Glossary dedup: dropping absorbed entry '{key}'")
                continue
            else:
                updated_entries.append(entry)
                seen_keys.add(key)

        # Add dedup results for keys not in original entries
        for key, de in dedup_by_key.items():
            if key not in seen_keys:
                updated_entries.append(de)

        return updated_entries

    def _compress_descriptions(self, chapter_id: str):
        """Compress any store descriptions that exceed DESC_COMPRESS_THRESHOLD."""
        for key, entry in self.store.items():
            desc = entry.get("description", "")
            if len(desc) <= DESC_COMPRESS_THRESHOLD:
                continue
            zh_name = entry.get("zh_name", "")
            try:
                compressed = self.llm_client.generate(
                    prompt=DESC_COMPRESS_PROMPT.format(
                        max_chars=DESC_COMPRESS_THRESHOLD,
                        key=key,
                        zh_name=zh_name,
                        description=desc,
                    ),
                    model_configs=self.model_configs,
                    operation_name=f"Glossary compress desc {key} ({chapter_id})",
                )
                compressed = compressed.strip()
                if compressed and len(compressed) < len(desc):
                    logger.info(f"  Glossary: compressed '{key}' description {len(desc)} → {len(compressed)} chars")
                    entry["description"] = compressed
            except Exception as e:
                logger.warning(f"Description compression failed for '{key}': {e}")

    def extract_and_update(
        self, source_text: str, translated_text: str, chapter_id: str
    ) -> str:
        """Generate glossary from completed translation, update store.

        Uses cache hit: same source_text prefix as the translation call.
        """
        # Build existing entries section for context
        existing_section = ""
        if self.store:
            existing_lines = []
            for key, entry in self.store.items():
                zh = entry.get("zh_name", "")
                aliases = _normalize_aliases(entry.get("aliases", []))
                desc = entry.get("description", "")
                alias_strs = [f"{a['ja']}→{a['zh']}" if a["zh"] else a["ja"] for a in aliases]
                existing_lines.append(
                    f"{key} → {zh} (aliases: {', '.join(alias_strs)}): {desc}"
                )
            existing_section = (
                "已有术语表（供参考，可更新）：\n"
                + "\n".join(existing_lines)
                + "\n\n"
            )

        prompt = GLOSSARY_EXTRACT_PROMPT.format(
            existing_section=existing_section,
            source_text=source_text,
            translated_text=translated_text,
        )

        # Extraction with retry
        entries = None
        for attempt in range(self.extract_retries + 1):
            try:
                response = self.llm_client.generate(
                    prompt=prompt,
                    model_configs=self.model_configs,
                    operation_name=f"Glossary extract {chapter_id} (attempt {attempt + 1})",
                )
                parsed = parse_llm_json(
                    response,
                    save_dir=self.log_dir,
                    operation_name=f"Glossary extract {chapter_id}",
                )
                if isinstance(parsed, list):
                    entries = parsed
                    break
                else:
                    logger.warning(f"Glossary extraction returned non-list for {chapter_id} (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Glossary extraction failed for {chapter_id} (attempt {attempt + 1}): {e}")

        if entries is None:
            logger.warning(f"Glossary extraction failed after {self.extract_retries + 1} attempts for {chapter_id}")
            return ""

        # Dedup: LLM-based pronoun cleaning + duplicate merging (skip if no entries)
        if entries:
            entries = self._dedup_entries(entries, chapter_id, prompt, response)

        # Update store
        # NOTE: removed_aliases blacklist disabled (see comment in _dedup_entries)
        now = datetime.now(timezone.utc).isoformat()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            key = entry.get("key", "").strip()
            if not key:
                continue
            if _is_metadata_key(key):
                logger.debug(f"  Glossary: skipping metadata key '{key}'")
                continue

            zh_name = entry.get("zh_name", "")
            aliases = _normalize_aliases(entry.get("aliases", []))

            description = entry.get("description", "")

            if key in self.store:
                old = self.store[key]
                # Merge aliases by ja key (new zh overwrites old)
                old_aliases = _normalize_aliases(old.get("aliases", []))
                old_by_ja = {a["ja"]: a for a in old_aliases}
                for a in aliases:
                    old_by_ja[a["ja"]] = a  # New alias overwrites
                merged_aliases = sorted(old_by_ja.values(), key=lambda a: a["ja"])

                # Push old description to history if changed
                if old.get("description") != description:
                    history = old.get("history", [])
                    history.append({
                        "description": old.get("description", ""),
                        "updated_by": old.get("updated_by", ""),
                        "timestamp": old.get("timestamp", ""),
                    })
                    self.store[key] = {
                        "zh_name": zh_name or old.get("zh_name", ""),
                        "aliases": merged_aliases,
                        "description": description,
                        "updated_by": chapter_id,
                        "timestamp": now,
                        "history": history,
                    }
                else:
                    # Description unchanged — still update aliases and zh_name
                    if merged_aliases != old_aliases:
                        self.store[key]["aliases"] = merged_aliases
                    if zh_name and zh_name != old.get("zh_name"):
                        self.store[key]["zh_name"] = zh_name
            else:
                self.store[key] = {
                    "zh_name": zh_name,
                    "aliases": aliases,
                    "description": description,
                    "updated_by": chapter_id,
                    "timestamp": now,
                    "history": [],
                }

        # Compress oversized descriptions
        self._compress_descriptions(chapter_id)

        # Build prev_chapter summary, bounded by max_tokens
        chapter_lines = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            key = entry.get("key", "")
            zh = entry.get("zh_name", "")
            desc = entry.get("description", "")
            if key:
                chapter_lines.append(f"{key} → {zh}：{desc}")

        prev_text = "\n".join(chapter_lines)
        # Truncate if exceeds max_tokens (simple char-based estimate: ~1.5 chars/token for CJK)
        max_chars = self.max_tokens * 2
        if len(prev_text) > max_chars:
            prev_text = prev_text[:max_chars]
            # Cut at last newline to avoid partial entry
            last_nl = prev_text.rfind("\n")
            if last_nl > 0:
                prev_text = prev_text[:last_nl]
            logger.info(f"  Glossary prev_chapter truncated to ~{self.max_tokens} tokens")
        self.prev_chapter = prev_text

        # Save per-chapter log
        log_file = self.log_dir / f"{chapter_id}.json"
        log_file.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Persist
        self.save()

        logger.info(
            f"  Glossary: {len(entries)} entries extracted, "
            f"{len(self.store)} total in store"
        )
        return self.prev_chapter
