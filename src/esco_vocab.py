"""
======================================================================
AI RESUME ANALYZER - SHARED ESCO SKILL VOCABULARY
======================================================================

One definition of "what counts as a skill" for the whole project:

    Step 5  (05_extract_skills.py)   offline skill extraction
    Step 13 (13_explainable_rag_...) command-line analysis
    Step 14 (14_app.py)              Streamlit app
    job_ranker.py                    job skill evidence

What counts as a skill
----------------------
ONLY the ESCO *skills* table (``skills_en.csv`` / its cleaned copy
``esco_skills_clean.csv``), and ONLY two of its fields:

    preferredLabel   the skill's name
    altLabels        its synonyms

Nothing else in the ESCO download is a skill. The folder
``data/external`` also holds occupations, ISCO groups, skill groups,
hierarchies and relation tables. Their ``preferredLabel`` / ``title`` /
``name`` / ``conceptUri`` columns contain occupation names ("software
developer"), group names, and URL strings - not skills. The previous
Streamlit app globbed every CSV there and treated all of those columns
as skills. This module never does that:

  * only a recognised skills file is read (explicit candidate paths);
  * rows are restricted to ESCO ``KnowledgeSkillCompetence`` concepts
    when the ``conceptType`` column exists (or to ``/esco/skill/`` URIs
    otherwise);
  * URIs are identifiers, never matching text;
  * hidden labels (misspellings, search aids) are not used;
  * single-character terms are ignored because they are ambiguous
    ("c", "r", "a" match grades, initials and bullets).

Matching is token-based (a trie over normalised tokens), identical to
the matcher Step 5 always used, so offline and online skill sets agree.
======================================================================
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# The ONLY files a skill vocabulary may be built from, in priority order.
# The processed copy is what the offline pipeline used; the raw ESCO
# download is the fallback for people who only have data/external.
ESCO_SKILLS_FILE_CANDIDATES: List[Path] = [
    DATA_DIR / "processed" / "esco_skills_clean.csv",
    DATA_DIR / "external" / "skills_en.csv",
]

SKILL_CONCEPT_TYPE = "knowledgeskillcompetence"
SKILL_URI_MARKER = "/esco/skill/"

MIN_TERM_CHARS = 2


# ======================================================================
# TEXT NORMALISATION (identical to the matcher used by Step 5)
# ======================================================================

def normalize_column_name(name: Any) -> str:
    """conceptUri / concept_uri / Concept URI -> "concepturi"."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def find_column(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = True,
) -> Optional[str]:

    column_map: Dict[str, str] = {}

    for column in df.columns:
        column_map.setdefault(normalize_column_name(column), column)

    for candidate in candidates:

        key = normalize_column_name(candidate)

        if key in column_map:
            return column_map[key]

    if required:
        raise ValueError(
            f"Could not find required column. Candidates: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def clean_text(value: Any) -> str:

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_text(text: str) -> str:
    """
    Normalise text for matching. Keeps letters, digits, + # . so that
    c++, c#, .net and node.js survive.
    """

    text = text.lower()
    text = re.sub(r"[/_|\\\-]+", " ", text)
    text = re.sub(r"[^a-z0-9+#. ]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text: str) -> List[str]:
    """
    Normalised tokens.

    Trailing dots are stripped so a skill at the end of a sentence still
    matches ("... strong project management." -> "management"). Dots
    inside a token are kept (node.js, e.g.) and so is a leading dot
    (.net).
    """

    text = normalize_text(text)

    if not text:
        return []

    tokens = []

    for token in text.split():

        token = token.rstrip(".")

        if token:
            tokens.append(token)

    return tokens


# ======================================================================
# TRIE
# ======================================================================

class SkillTrieNode:

    __slots__ = ("children", "skill_ids")

    def __init__(self) -> None:
        self.children: Dict[str, "SkillTrieNode"] = {}
        self.skill_ids: List[int] = []


class SkillTrie:
    """Token-based multi-word matcher ("machine learning" -> machine -> learning)."""

    def __init__(self) -> None:
        self.root = SkillTrieNode()

    def add(self, tokens: List[str], skill_id: int) -> None:

        if not tokens:
            return

        node = self.root

        for token in tokens:

            if token not in node.children:
                node.children[token] = SkillTrieNode()

            node = node.children[token]

        if skill_id not in node.skill_ids:
            node.skill_ids.append(skill_id)

    def find(self, tokens: List[str]) -> List[int]:

        matches: List[int] = []
        token_count = len(tokens)

        for start in range(token_count):

            node = self.root
            position = start

            while position < token_count:

                token = tokens[position]

                if token not in node.children:
                    break

                node = node.children[token]

                if node.skill_ids:
                    matches.extend(node.skill_ids)

                position += 1

        return matches


# ======================================================================
# SKILL-ROW SELECTION
# ======================================================================

def select_skill_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Keep only rows that are ESCO skills. Raises ValueError when the
    table clearly is not an ESCO skills table.
    """

    uri_column = find_column(
        df,
        ["conceptUri", "concept_uri", "uri", "skillUri", "skill_uri"],
    )

    type_column = find_column(
        df,
        ["conceptType", "concept_type"],
        required=False,
    )

    stats: Dict[str, Any] = {
        "input_rows": int(len(df)),
        "filter": None,
        "dropped_rows": 0,
    }

    if type_column is not None:

        is_skill = (
            df[type_column]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(SKILL_CONCEPT_TYPE)
        )

        stats["filter"] = f"conceptType == {SKILL_CONCEPT_TYPE}"

        if not is_skill.any():
            raise ValueError(
                "This table has a conceptType column but no "
                "KnowledgeSkillCompetence rows, so it is not the ESCO "
                "skills table."
            )

    else:

        is_skill = (
            df[uri_column]
            .astype(str)
            .str.contains(SKILL_URI_MARKER, regex=False)
        )

        if is_skill.any():
            stats["filter"] = f"conceptUri contains {SKILL_URI_MARKER}"
        else:
            # Cannot verify the table type; keep everything but say so.
            is_skill = pd.Series(True, index=df.index)
            stats["filter"] = "none (could not verify rows are skills)"
            logger.warning(
                "ESCO skills table has neither conceptType nor /esco/skill/ "
                "URIs; using all rows."
            )

    selected = df[is_skill]

    stats["dropped_rows"] = int(len(df) - len(selected))

    return selected, stats


# ======================================================================
# VOCABULARY
# ======================================================================

def _split_alt_labels(value: Any) -> List[str]:

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    # ESCO separates alternative labels with newlines.
    return [
        clean_text(part)
        for part in re.split(r"[\n\r|;]+", str(value))
        if clean_text(part)
    ]


class ESCOVocabulary:
    """
    ESCO skills, indexed for fast phrase matching.

    ``skills`` is a list of dicts::

        {"skill_id": int, "concept_uri": str, "preferred_label": str}
    """

    def __init__(self, min_term_chars: int = MIN_TERM_CHARS) -> None:

        self.min_term_chars = min_term_chars

        self.skills: List[Dict[str, Any]] = []
        self.trie = SkillTrie()

        # normalised phrase -> skill_id (first claimant wins, as in Step 5)
        self.term_index: Dict[str, int] = {}

        self.uri_label: Dict[str, str] = {}

        self.term_count = 0
        self.skill_count = 0
        self.skipped_short_terms = 0
        self.dropped_rows = 0

        self.source: str = ""
        self.source_signature: str = ""
        self.row_filter: Optional[str] = None

    # ------------------------------------------------------------------

    def _add_term(self, text: str, skill_id: int) -> None:

        tokens = tokenize(text)

        if not tokens:
            return

        key = " ".join(tokens)

        if len(key) < self.min_term_chars:
            self.skipped_short_terms += 1
            return

        if key in self.term_index:
            return

        self.trie.add(tokens, skill_id)
        self.term_index[key] = skill_id

    # ------------------------------------------------------------------

    def build_from_dataframe(
        self,
        esco_df: pd.DataFrame,
        *,
        restrict_to_skill_rows: bool = True,
    ) -> None:

        logger.info("=" * 70)
        logger.info("Loading ESCO skill vocabulary")
        logger.info("=" * 70)

        logger.info(
            "ESCO skills shape: %s rows x %s columns",
            esco_df.shape[0],
            esco_df.shape[1],
        )

        if restrict_to_skill_rows:

            esco_df, stats = select_skill_rows(esco_df)

            self.dropped_rows = stats["dropped_rows"]
            self.row_filter = stats["filter"]

            if self.dropped_rows:
                logger.warning(
                    "Ignored %d non-skill ESCO rows (%s).",
                    self.dropped_rows,
                    stats["filter"],
                )

        uri_column = find_column(
            esco_df,
            ["conceptUri", "concept_uri", "uri", "skillUri", "skill_uri"],
        )

        label_column = find_column(
            esco_df,
            [
                "preferredLabel",
                "preferred_label",
                "skillLabel",
                "skill_label",
                "label",
                "name",
            ],
        )

        alt_column = find_column(
            esco_df,
            [
                "altLabels",
                "alt_labels",
                "alternativeLabels",
                "alternative_labels",
            ],
            required=False,
        )

        records: List[Tuple[int, str, List[str]]] = []

        for _, row in esco_df.iterrows():

            label = clean_text(row.get(label_column, ""))

            if not label:
                continue

            uri = clean_text(row.get(uri_column, ""))

            skill_id = len(self.skills)

            self.skills.append(
                {
                    "skill_id": skill_id,
                    "concept_uri": uri,
                    "preferred_label": label,
                }
            )

            if uri:
                self.uri_label.setdefault(uri, label)

            alternatives = (
                _split_alt_labels(row.get(alt_column))
                if alt_column
                else []
            )

            records.append((skill_id, label, alternatives))

        # Preferred labels first, so a skill's own name is never shadowed
        # by another skill that lists the same words as a synonym.
        for skill_id, label, _ in records:
            self._add_term(label, skill_id)

        for skill_id, _, alternatives in records:
            for alternative in alternatives:
                self._add_term(alternative, skill_id)

        self.skill_count = len(self.skills)
        self.term_count = len(self.term_index)

        logger.info("Unique ESCO skills: %s", self.skill_count)
        logger.info("Unique matching terms: %s", self.term_count)

    # ------------------------------------------------------------------

    def lookup_term(self, text: str) -> Optional[int]:
        """Exact (normalised) preferred/alternative-label lookup."""

        key = " ".join(tokenize(text))

        return self.term_index.get(key)

    def label_for(self, uri: str) -> str:
        return self.uri_label.get(uri) or uri.rsplit("/", 1)[-1]

    def extract(self, text: str) -> List[Dict[str, Any]]:
        return extract_skills_from_text(text, self)

    @property
    def fingerprint(self) -> str:
        return (
            f"{self.source_signature}|{self.skill_count}|{self.term_count}"
            f"|min{self.min_term_chars}"
        )


# ======================================================================
# EXTRACTION
# ======================================================================

def _deduplicate(skill_ids: List[int]) -> List[int]:

    seen = set()
    result = []

    for skill_id in skill_ids:

        if skill_id not in seen:
            seen.add(skill_id)
            result.append(skill_id)

    return result


def extract_skills_from_text(
    text: str,
    vocabulary: ESCOVocabulary,
) -> List[Dict[str, Any]]:
    """
    Skills mentioned in ``text``, de-duplicated, ordered by label.

    Each item: ``{"skill_id", "concept_uri", "preferred_label"}``.
    """

    tokens = tokenize(text)

    if not tokens:
        return []

    results = []

    for skill_id in _deduplicate(vocabulary.trie.find(tokens)):

        skill = vocabulary.skills[skill_id]

        results.append(
            {
                "skill_id": skill_id,
                "concept_uri": skill["concept_uri"],
                "preferred_label": skill["preferred_label"],
            }
        )

    results.sort(key=lambda item: item["preferred_label"].lower())

    return results


# ======================================================================
# LOADING
# ======================================================================

def find_esco_skills_file(
    candidates: Optional[List[Path]] = None,
) -> Optional[Path]:
    """First existing ESCO skills file, or None."""

    for path in candidates or ESCO_SKILLS_FILE_CANDIDATES:
        if Path(path).is_file():
            return Path(path)

    return None


def load_esco_vocabulary(
    path: Optional[Path] = None,
    *,
    min_term_chars: int = MIN_TERM_CHARS,
) -> ESCOVocabulary:
    """
    Build the skill vocabulary from the ESCO skills table.

    Raises FileNotFoundError when no skills file exists and ValueError
    when the file is not an ESCO skills table.
    """

    skills_path = Path(path) if path else find_esco_skills_file()

    if skills_path is None or not skills_path.is_file():

        searched = "\n  ".join(str(p) for p in ESCO_SKILLS_FILE_CANDIDATES)

        raise FileNotFoundError(
            "ESCO skills file not found. Expected one of:\n  "
            f"{searched}\n"
            "(download ESCO from https://esco.ec.europa.eu and place "
            "skills_en.csv in data/external/)."
        )

    frame = pd.read_csv(
        skills_path,
        low_memory=False,
        dtype=str,
        keep_default_na=False,
    )

    vocabulary = ESCOVocabulary(min_term_chars=min_term_chars)

    vocabulary.build_from_dataframe(frame)

    stat = skills_path.stat()

    vocabulary.source = str(skills_path)
    vocabulary.source_signature = (
        f"{skills_path.name}:{stat.st_size}:{int(stat.st_mtime)}"
    )

    if vocabulary.skill_count == 0:
        raise ValueError(f"No skills found in {skills_path}.")

    return vocabulary
