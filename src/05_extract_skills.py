"""
AI RESUME ANALYZER
STEP 5 - SKILL EXTRACTION

Purpose
-------
Extract skills from processed resumes and job descriptions using the
ESCO skill vocabulary.

Input
-----
data/processed/
    resumes_text_processed.csv
    jobs_text_processed.csv
    esco_skills_clean.csv

Output
------
data/processed/
    resumes_skills_extracted.csv
    jobs_skills_extracted.csv

reports/
    skill_extraction_report.json

Design principles
-----------------
- No hardcoded absolute paths
- Automatically detects normalized/original column names
- Preserves ESCO identifiers and human-readable labels
- Avoids huge regular-expression construction
- Uses token-based trie matching for efficiency
"""

from __future__ import annotations

import ast
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

RESUME_INPUT = PROCESSED_DIR / "resumes_text_processed.csv"
JOB_INPUT = PROCESSED_DIR / "jobs_text_processed.csv"
ESCO_SKILLS_INPUT = PROCESSED_DIR / "esco_skills_clean.csv"

RESUME_OUTPUT = PROCESSED_DIR / "resumes_skills_extracted.csv"
JOB_OUTPUT = PROCESSED_DIR / "jobs_skills_extracted.csv"

REPORT_OUTPUT = REPORTS_DIR / "skill_extraction_report.json"


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def normalize_column_name(name: Any) -> str:
    """
    Normalize a column name so that different naming conventions match.

    Examples:
        conceptUri      -> concepturi
        concept_uri     -> concepturi
        Concept URI     -> concepturi
        preferredLabel  -> preferredlabel
        preferred_label -> preferredlabel
    """
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def build_column_map(df: pd.DataFrame) -> Dict[str, str]:
    """
    Map normalized column names to actual dataframe column names.
    """
    mapping = {}

    for column in df.columns:
        normalized = normalize_column_name(column)

        if normalized not in mapping:
            mapping[normalized] = column

    return mapping


def find_column(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = True
) -> str | None:
    """
    Find a dataframe column using normalized candidate names.
    """

    column_map = build_column_map(df)

    for candidate in candidates:
        normalized = normalize_column_name(candidate)

        if normalized in column_map:
            return column_map[normalized]

    if required:
        raise ValueError(
            f"Could not find required column.\n"
            f"Candidates: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def clean_text(value: Any) -> str:
    """
    Convert a value into clean searchable text.
    """
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    text = str(value)

    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_text(text: str) -> str:
    """
    Normalize text for matching.

    Human-readable source columns remain untouched in the saved data.
    This normalized version is used only for matching.
    """

    text = text.lower()

    # Normalize common separators to spaces.
    text = re.sub(r"[/_|\\\-]+", " ", text)

    # Keep letters, numbers and spaces.
    text = re.sub(r"[^a-z0-9+#. ]+", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text: str) -> List[str]:
    """
    Tokenize normalized text.

    Keeps:
        c++
        c#
        .net
        node.js
        python
        sql
    """

    text = normalize_text(text)

    if not text:
        return []

    return text.split()


# ============================================================================
# SKILL TRIE
# ============================================================================

class SkillTrieNode:
    """
    Node in a token-based trie.
    """

    __slots__ = ("children", "skill_ids")

    def __init__(self):
        self.children: Dict[str, SkillTrieNode] = {}
        self.skill_ids: List[int] = []


class SkillTrie:
    """
    Efficient multi-word skill matcher.

    Instead of creating tens of thousands of regular expressions,
    skills are stored as token sequences.

    Example:

        "machine learning"

    becomes:

        machine -> learning

    The resume is then scanned token-by-token.
    """

    def __init__(self):
        self.root = SkillTrieNode()

    def add(
        self,
        tokens: List[str],
        skill_id: int
    ) -> None:

        if not tokens:
            return

        node = self.root

        for token in tokens:

            if token not in node.children:
                node.children[token] = SkillTrieNode()

            node = node.children[token]

        if skill_id not in node.skill_ids:
            node.skill_ids.append(skill_id)

    def find(
        self,
        tokens: List[str]
    ) -> List[int]:

        matches = []

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


# ============================================================================
# ESCO VOCABULARY
# ============================================================================

class ESCOVocabulary:

    def __init__(self):
        self.skills: List[Dict[str, Any]] = []
        self.trie = SkillTrie()

        self.term_count = 0
        self.skill_count = 0

    def build_from_dataframe(
        self,
        esco_df: pd.DataFrame
    ) -> None:

        logger.info("=" * 70)
        logger.info("Loading ESCO skill vocabulary")
        logger.info("=" * 70)

        logger.info(
            "ESCO skills shape: %s rows × %s columns",
            esco_df.shape[0],
            esco_df.shape[1]
        )

        # ------------------------------------------------------------
        # Automatically detect columns
        # ------------------------------------------------------------

        uri_column = find_column(
            esco_df,
            [
                "conceptUri",
                "concept_uri",
                "uri",
                "skillUri",
                "skill_uri"
            ]
        )

        label_column = find_column(
            esco_df,
            [
                "preferredLabel",
                "preferred_label",
                "skillLabel",
                "skill_label",
                "label",
                "name"
            ]
        )

        alt_column = find_column(
            esco_df,
            [
                "altLabels",
                "alt_labels",
                "alternativeLabels",
                "alternative_labels"
            ],
            required=False
        )

        description_column = find_column(
            esco_df,
            [
                "description"
            ],
            required=False
        )

        logger.info("Detected ESCO URI column: %s", uri_column)
        logger.info("Detected ESCO label column: %s", label_column)

        if alt_column:
            logger.info("Detected alternative labels column: %s", alt_column)

        # ------------------------------------------------------------
        # Build vocabulary
        # ------------------------------------------------------------

        seen_terms = set()

        for _, row in esco_df.iterrows():

            uri = clean_text(row.get(uri_column, ""))
            label = clean_text(row.get(label_column, ""))

            if not label:
                continue

            skill_id = len(self.skills)

            description = ""

            if description_column:
                description = clean_text(
                    row.get(description_column, "")
                )

            skill_record = {
                "skill_id": skill_id,
                "concept_uri": uri,
                "preferred_label": label,
                "description": description
            }

            self.skills.append(skill_record)

            # --------------------------------------------------------
            # Preferred label
            # --------------------------------------------------------

            label_tokens = tokenize(label)

            if label_tokens:

                key = " ".join(label_tokens)

                if key not in seen_terms:

                    self.trie.add(
                        label_tokens,
                        skill_id
                    )

                    seen_terms.add(key)

            # --------------------------------------------------------
            # Alternative labels
            # --------------------------------------------------------

            if alt_column:

                alt_value = row.get(alt_column)

                if alt_value is not None and not pd.isna(alt_value):

                    alt_text = str(alt_value)

                    # ESCO commonly separates alternatives by newline.
                    alternatives = re.split(
                        r"[\n\r|;]+",
                        alt_text
                    )

                    for alternative in alternatives:

                        alternative = clean_text(
                            alternative
                        )

                        if not alternative:
                            continue

                        alt_tokens = tokenize(
                            alternative
                        )

                        if not alt_tokens:
                            continue

                        key = " ".join(alt_tokens)

                        if key in seen_terms:
                            continue

                        self.trie.add(
                            alt_tokens,
                            skill_id
                        )

                        seen_terms.add(key)

        self.skill_count = len(self.skills)
        self.term_count = len(seen_terms)

        logger.info(
            "Unique ESCO skills: %s",
            self.skill_count
        )

        logger.info(
            "Unique matching terms: %s",
            self.term_count
        )


# ============================================================================
# SKILL EXTRACTION
# ============================================================================

def deduplicate_skill_ids(
    skill_ids: List[int]
) -> List[int]:

    seen = set()
    result = []

    for skill_id in skill_ids:

        if skill_id not in seen:

            seen.add(skill_id)
            result.append(skill_id)

    return result


def extract_skills_from_text(
    text: str,
    vocabulary: ESCOVocabulary
) -> List[Dict[str, Any]]:

    tokens = tokenize(text)

    if not tokens:
        return []

    skill_ids = vocabulary.trie.find(tokens)

    skill_ids = deduplicate_skill_ids(
        skill_ids
    )

    results = []

    for skill_id in skill_ids:

        skill = vocabulary.skills[skill_id]

        results.append({
            "skill_id": skill_id,
            "concept_uri": skill["concept_uri"],
            "preferred_label": skill["preferred_label"]
        })

    # Human-readable ordering.
    results.sort(
        key=lambda x: x["preferred_label"].lower()
    )

    return results


# ============================================================================
# SERIALIZATION
# ============================================================================

def skills_to_json(
    skills: List[Dict[str, Any]]
) -> str:

    return json.dumps(
        skills,
        ensure_ascii=False
    )


def labels_to_json(
    skills: List[Dict[str, Any]]
) -> str:

    labels = [
        skill["preferred_label"]
        for skill in skills
    ]

    return json.dumps(
        labels,
        ensure_ascii=False
    )


# ============================================================================
# RESUME PROCESSING
# ============================================================================

def process_resumes(
    vocabulary: ESCOVocabulary
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    logger.info("")
    logger.info("=" * 70)
    logger.info("Extracting skills from resumes")
    logger.info("=" * 70)

    if not RESUME_INPUT.exists():

        raise FileNotFoundError(
            f"Resume input not found: {RESUME_INPUT}"
        )

    df = pd.read_csv(
        RESUME_INPUT,
        low_memory=False
    )

    logger.info(
        "Resume dataset: %s rows × %s columns",
        df.shape[0],
        df.shape[1]
    )

    text_column = find_column(
        df,
        [
            "resume_str",
            "resume_text",
            "resume",
            "text",
            "content"
        ]
    )

    logger.info(
        "Detected resume text column: %s",
        text_column
    )

    extracted = []

    total_skills = 0
    resumes_with_skills = 0

    start_time = time.time()

    total_rows = len(df)

    for index, text in enumerate(
        df[text_column].fillna("").astype(str)
    ):

        skills = extract_skills_from_text(
            text,
            vocabulary
        )

        extracted.append(
            skills
        )

        total_skills += len(skills)

        if skills:
            resumes_with_skills += 1

        # Progress every 100 resumes.
        if (index + 1) % 100 == 0 or index + 1 == total_rows:

            elapsed = time.time() - start_time

            rate = (
                (index + 1) / elapsed
                if elapsed > 0
                else 0
            )

            logger.info(
                "Resume progress: %s/%s | "
                "with skills: %s | "
                "speed: %.1f resumes/sec",
                index + 1,
                total_rows,
                resumes_with_skills,
                rate
            )

    df["extracted_skills"] = [
        skills_to_json(skills)
        for skills in extracted
    ]

    df["skill_labels"] = [
        labels_to_json(skills)
        for skills in extracted
    ]

    df["skill_count"] = [
        len(skills)
        for skills in extracted
    ]

    df.to_csv(
        RESUME_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    elapsed = time.time() - start_time

    stats = {
        "input_rows": int(len(df)),
        "output_rows": int(len(df)),
        "rows_with_skills": int(resumes_with_skills),
        "rows_without_skills": int(
            len(df) - resumes_with_skills
        ),
        "total_extracted_skill_mentions": int(
            total_skills
        ),
        "average_skills_per_resume": (
            float(total_skills / len(df))
            if len(df) > 0
            else 0.0
        ),
        "processing_seconds": round(
            elapsed,
            3
        ),
        "output_file": str(
            RESUME_OUTPUT
        )
    }

    logger.info(
        "Saved: %s",
        RESUME_OUTPUT
    )

    return df, stats


# ============================================================================
# JOB PROCESSING
# ============================================================================

def process_jobs(
    vocabulary: ESCOVocabulary
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    logger.info("")
    logger.info("=" * 70)
    logger.info("Extracting skills from jobs")
    logger.info("=" * 70)

    if not JOB_INPUT.exists():

        raise FileNotFoundError(
            f"Job input not found: {JOB_INPUT}"
        )

    df = pd.read_csv(
        JOB_INPUT,
        low_memory=False
    )

    logger.info(
        "Job dataset: %s rows × %s columns",
        df.shape[0],
        df.shape[1]
    )

    description_column = find_column(
        df,
        [
            "job_description",
            "jobDescription",
            "description",
            "job_text",
            "text"
        ]
    )

    skill_column = find_column(
        df,
        [
            "job_skill_set",
            "jobSkillSet",
            "skill_set",
            "skills",
            "skill_list"
        ],
        required=False
    )

    title_column = find_column(
        df,
        [
            "job_title",
            "jobTitle",
            "title"
        ],
        required=False
    )

    logger.info(
        "Detected job description column: %s",
        description_column
    )

    if title_column:
        logger.info(
            "Detected job title column: %s",
            title_column
        )

    extracted = []

    total_skills = 0
    jobs_with_skills = 0

    start_time = time.time()

    total_rows = len(df)

    for index, row in df.iterrows():

        description = clean_text(
            row.get(description_column, "")
        )

        title = ""

        if title_column:
            title = clean_text(
                row.get(title_column, "")
            )

        # Include title because job titles often contain useful
        # explicit skills such as "Python Developer".
        combined_text = (
            f"{title} {description}"
        ).strip()

        skills = extract_skills_from_text(
            combined_text,
            vocabulary
        )

        extracted.append(
            skills
        )

        total_skills += len(skills)

        if skills:
            jobs_with_skills += 1

        if (index + 1) % 100 == 0 or index + 1 == total_rows:

            elapsed = time.time() - start_time

            rate = (
                (index + 1) / elapsed
                if elapsed > 0
                else 0
            )

            logger.info(
                "Job progress: %s/%s | "
                "with skills: %s | "
                "speed: %.1f jobs/sec",
                index + 1,
                total_rows,
                jobs_with_skills,
                rate
            )

    df["esco_extracted_skills"] = [
        skills_to_json(skills)
        for skills in extracted
    ]

    df["esco_skill_labels"] = [
        labels_to_json(skills)
        for skills in extracted
    ]

    df["esco_skill_count"] = [
        len(skills)
        for skills in extracted
    ]

    df.to_csv(
        JOB_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    elapsed = time.time() - start_time

    stats = {
        "input_rows": int(len(df)),
        "output_rows": int(len(df)),
        "rows_with_skills": int(jobs_with_skills),
        "rows_without_skills": int(
            len(df) - jobs_with_skills
        ),
        "total_extracted_skill_mentions": int(
            total_skills
        ),
        "average_skills_per_job": (
            float(total_skills / len(df))
            if len(df) > 0
            else 0.0
        ),
        "processing_seconds": round(
            elapsed,
            3
        ),
        "output_file": str(
            JOB_OUTPUT
        )
    }

    logger.info(
        "Saved: %s",
        JOB_OUTPUT
    )

    return df, stats


# ============================================================================
# REPORT
# ============================================================================

def save_report(
    vocabulary: ESCOVocabulary,
    resume_stats: Dict[str, Any],
    job_stats: Dict[str, Any]
) -> None:

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "step": 5,
        "name": "Skill Extraction",
        "status": "completed",

        "input_files": {
            "esco_skills": str(
                ESCO_SKILLS_INPUT
            ),
            "resumes": str(
                RESUME_INPUT
            ),
            "jobs": str(
                JOB_INPUT
            )
        },

        "output_files": {
            "resumes": str(
                RESUME_OUTPUT
            ),
            "jobs": str(
                JOB_OUTPUT
            )
        },

        "vocabulary": {
            "esco_skill_count": vocabulary.skill_count,
            "matching_term_count": vocabulary.term_count
        },

        "resumes": resume_stats,

        "jobs": job_stats,

        "method": {
            "matching": "token_based_trie",
            "case_sensitive": False,
            "uses_preferred_labels": True,
            "uses_alternative_labels": True,
            "embeddings": False,
            "llm": False,
            "rag": False
        }
    }

    with open(
        REPORT_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    logger.info(
        "Report saved: %s",
        REPORT_OUTPUT
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    logger.info("=" * 70)
    logger.info("AI RESUME ANALYZER - STEP 5 SKILL EXTRACTION")
    logger.info("=" * 70)

    # ------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------

    required_files = [
        RESUME_INPUT,
        JOB_INPUT,
        ESCO_SKILLS_INPUT
    ]

    for path in required_files:

        if not path.exists():

            logger.error(
                "Required input file not found: %s",
                path
            )

            sys.exit(1)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Load ESCO
    # ------------------------------------------------------------

    esco_df = pd.read_csv(
        ESCO_SKILLS_INPUT,
        low_memory=False
    )

    vocabulary = ESCOVocabulary()

    vocabulary.build_from_dataframe(
        esco_df
    )

    # ------------------------------------------------------------
    # Process resumes
    # ------------------------------------------------------------

    _, resume_stats = process_resumes(
        vocabulary
    )

    # ------------------------------------------------------------
    # Process jobs
    # ------------------------------------------------------------

    _, job_stats = process_jobs(
        vocabulary
    )

    # ------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------

    save_report(
        vocabulary,
        resume_stats,
        job_stats
    )

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 5 COMPLETE")
    logger.info("=" * 70)

    logger.info(
        "Resume output: %s",
        RESUME_OUTPUT
    )

    logger.info(
        "Job output: %s",
        JOB_OUTPUT
    )

    logger.info(
        "Report: %s",
        REPORT_OUTPUT
    )

    logger.info(
        "ESCO skills: %s",
        vocabulary.skill_count
    )

    logger.info(
        "Matching terms: %s",
        vocabulary.term_count
    )

    logger.info("=" * 70)


if __name__ == "__main__":
    main()