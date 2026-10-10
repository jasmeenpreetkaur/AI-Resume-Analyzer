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
- The vocabulary, tokenizer and matcher come from esco_vocab.py, the same
  module used by Step 13 and the Streamlit app, so a skill means the same
  thing offline and online
- Only the ESCO skills table is used (preferredLabel + altLabels of
  KnowledgeSkillCompetence rows); occupations, skill groups and relation
  tables are never treated as skills
- No hardcoded absolute paths
- Automatically detects normalized/original column names
- Preserves ESCO identifiers and human-readable labels
- Avoids huge regular-expression construction
- Uses token-based trie matching for efficiency
"""

from __future__ import annotations

import json
import logging
import sys
import time
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
# SHARED SKILL VOCABULARY
# ============================================================================
#
# The skill vocabulary, tokenizer and phrase matcher live in
# esco_vocab.py so that this offline step, Step 13 and the Streamlit app
# all agree on what a skill is and how it is found in text. Only the ESCO
# skills table (preferredLabel + altLabels of KnowledgeSkillCompetence
# rows) is used - never occupations, skill groups or other ESCO tables.

from esco_vocab import (  # noqa: E402
    ESCOVocabulary,
    clean_text,
    extract_skills_from_text,
    find_column,
    load_esco_vocabulary,
    normalize_column_name,
    normalize_text,
    tokenize,
)


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
            "source_file": vocabulary.source,
            "row_filter": vocabulary.row_filter,
            "non_skill_rows_ignored": vocabulary.dropped_rows,
            "esco_skill_count": vocabulary.skill_count,
            "matching_term_count": vocabulary.term_count,
            "single_character_terms_skipped": (
                vocabulary.skipped_short_terms
            )
        },

        "resumes": resume_stats,

        "jobs": job_stats,

        "method": {
            "matching": "token_based_trie",
            "case_sensitive": False,
            "vocabulary_module": "esco_vocab.py (shared with steps 13 and 14)",
            "skills_table_only": True,
            "uses_preferred_labels": True,
            "uses_alternative_labels": True,
            "uses_hidden_labels": False,
            "uses_descriptions": False,
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

    # Skills table only (preferredLabel + altLabels of skill rows).
    vocabulary = load_esco_vocabulary(
        ESCO_SKILLS_INPUT
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