"""
AI Resume Analyzer
Step 4: Prepare Resume and Job Text Data

Purpose:
- Load cleaned resume and job datasets.
- Normalize text safely.
- Parse job_skill_set into structured lists.
- Create unified text fields.
- Generate a processing report.

This step does NOT:
- extract skills
- map skills to ESCO
- generate embeddings
- use an LLM
- perform resume-job matching
- train ML models
"""

from pathlib import Path
import ast
import json
import logging
import re
from typing import Any

import pandas as pd


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

RESUME_INPUT = PROCESSED_DIR / "resumes_clean.csv"
JOB_INPUT = PROCESSED_DIR / "jobs_clean.csv"

RESUME_OUTPUT = PROCESSED_DIR / "resumes_text_processed.csv"
JOB_OUTPUT = PROCESSED_DIR / "jobs_text_processed.csv"

REPORT_OUTPUT = REPORTS_DIR / "text_processing_report.json"


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names so the rest of the pipeline can work
    consistently with cleaned datasets.
    """

    df = df.copy()

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    return df


def normalize_text(value: Any) -> str:
    """
    Normalize text while preserving meaningful content.

    Operations:
    - null -> empty string
    - convert to string
    - remove obvious HTML tags
    - normalize line breaks
    - collapse excessive whitespace
    - strip surrounding whitespace

    Text is NOT blindly lowercased.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value)

    # Remove obvious HTML tags.
    if "<" in text and ">" in text:
        text = re.sub(r"<[^>]+>", " ", text)

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize excessive whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def parse_skill_list(value: Any) -> list[str]:
    """
    Safely parse job_skill_set.

    Supported formats:
    - Python list
    - tuple
    - string representation of a list
    - JSON-like list
    - single string

    Invalid values are returned as an empty list.
    """

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    if isinstance(value, list):
        raw_items = value

    elif isinstance(value, tuple):
        raw_items = list(value)

    else:
        text = str(value).strip()

        if not text:
            return []

        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, list):
                raw_items = parsed

            elif isinstance(parsed, tuple):
                raw_items = list(parsed)

            elif isinstance(parsed, str):
                raw_items = [parsed]

            else:
                raw_items = []

        except (ValueError, SyntaxError):
            # Treat a non-list string as one skill entry.
            raw_items = [text]

    cleaned_items = []

    for item in raw_items:

        if item is None:
            continue

        try:
            if pd.isna(item):
                continue
        except (TypeError, ValueError):
            pass

        item_text = normalize_text(item)

        if item_text:
            cleaned_items.append(item_text)

    return cleaned_items


def build_job_text(
    title: str,
    description: str,
    skills: list[str]
) -> str:
    """
    Build a unified job text representation.
    """

    sections = []

    if title:
        sections.append(f"Job Title: {title}")

    if description:
        sections.append(
            f"Job Description: {description}"
        )

    if skills:
        sections.append(
            "Required Skills: " + ", ".join(skills)
        )

    return "\n".join(sections).strip()


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str
) -> None:
    """
    Validate required columns after column-name normalization.
    """

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Available columns: {list(dataframe.columns)}"
        )


# ============================================================================
# RESUME PROCESSING
# ============================================================================

def process_resumes() -> tuple[pd.DataFrame, dict]:

    logger.info("")
    logger.info("=" * 70)
    logger.info("Processing resumes")
    logger.info("=" * 70)

    if not RESUME_INPUT.exists():
        raise FileNotFoundError(
            f"Resume dataset not found: {RESUME_INPUT}"
        )

    df = pd.read_csv(RESUME_INPUT)

    original_rows = len(df)

    # Normalize column names.
    df = normalize_column_names(df)

    logger.info(
        "Original shape: %d rows × %d columns",
        df.shape[0],
        df.shape[1]
    )

    logger.info(
        "Detected columns: %s",
        list(df.columns)
    )

    validate_required_columns(
        df,
        ["resume_str"],
        "Resume dataset"
    )

    # ------------------------------------------------------------------------
    # Create output
    # ------------------------------------------------------------------------

    output = pd.DataFrame()

    # Preserve source ID.
    if "id" in df.columns:
        output["resume_id"] = df["id"]

    elif "resume_id" in df.columns:
        output["resume_id"] = df["resume_id"]

    else:
        output["resume_id"] = range(1, len(df) + 1)

    # Preserve category.
    if "category" in df.columns:
        output["category"] = df["category"].apply(
            normalize_text
        )
    else:
        output["category"] = ""

    # Process resume text.
    output["resume_text"] = df["resume_str"].apply(
        normalize_text
    )

    # Preserve HTML if available.
    if "resume_html" in df.columns:
        output["resume_html"] = df["resume_html"].apply(
            normalize_text
        )

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    text_lengths = output["resume_text"].str.len()

    empty_text_count = int(
        output["resume_text"].str.strip().eq("").sum()
    )

    output.to_csv(
        RESUME_OUTPUT,
        index=False,
        encoding="utf-8"
    )

    logger.info(
        "Cleaned shape: %d rows × %d columns",
        output.shape[0],
        output.shape[1]
    )

    logger.info(
        "Empty resume texts: %d",
        empty_text_count
    )

    logger.info(
        "Saved: %s",
        RESUME_OUTPUT
    )

    statistics = {
        "original_rows": original_rows,
        "output_rows": len(output),
        "empty_text_count": empty_text_count,
        "text_length": {
            "min": (
                int(text_lengths.min())
                if len(text_lengths)
                else 0
            ),
            "max": (
                int(text_lengths.max())
                if len(text_lengths)
                else 0
            ),
            "mean": (
                round(float(text_lengths.mean()), 2)
                if len(text_lengths)
                else 0
            )
        }
    }

    return output, statistics


# ============================================================================
# JOB PROCESSING
# ============================================================================

def process_jobs() -> tuple[pd.DataFrame, dict]:

    logger.info("")
    logger.info("=" * 70)
    logger.info("Processing jobs")
    logger.info("=" * 70)

    if not JOB_INPUT.exists():
        raise FileNotFoundError(
            f"Job dataset not found: {JOB_INPUT}"
        )

    df = pd.read_csv(JOB_INPUT)

    original_rows = len(df)

    # Normalize column names.
    df = normalize_column_names(df)

    logger.info(
        "Original shape: %d rows × %d columns",
        df.shape[0],
        df.shape[1]
    )

    logger.info(
        "Detected columns: %s",
        list(df.columns)
    )

    validate_required_columns(
        df,
        [
            "job_title",
            "job_description",
            "job_skill_set"
        ],
        "Job dataset"
    )

    # ------------------------------------------------------------------------
    # Create output
    # ------------------------------------------------------------------------

    output = pd.DataFrame()

    # Preserve ID.
    if "job_id" in df.columns:
        output["job_id"] = df["job_id"]

    else:
        output["job_id"] = range(1, len(df) + 1)

    # Preserve category.
    if "category" in df.columns:
        output["category"] = df["category"].apply(
            normalize_text
        )

    # Process title.
    output["job_title"] = df["job_title"].apply(
        normalize_text
    )

    # Process description.
    output["job_description"] = df["job_description"].apply(
        normalize_text
    )

    # ------------------------------------------------------------------------
    # Parse skills
    # ------------------------------------------------------------------------

    parsed_skills = df["job_skill_set"].apply(
        parse_skill_list
    )

    output["job_skills"] = parsed_skills.apply(
        lambda skills: json.dumps(
            skills,
            ensure_ascii=False
        )
    )

    # ------------------------------------------------------------------------
    # Unified job text
    # ------------------------------------------------------------------------

    output["job_text"] = [
        build_job_text(
            title,
            description,
            skills
        )
        for title, description, skills
        in zip(
            output["job_title"],
            output["job_description"],
            parsed_skills
        )
    ]

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    empty_title_count = int(
        output["job_title"].str.strip().eq("").sum()
    )

    empty_description_count = int(
        output["job_description"].str.strip().eq("").sum()
    )

    empty_job_text_count = int(
        output["job_text"].str.strip().eq("").sum()
    )

    successfully_parsed = int(
        parsed_skills.apply(
            lambda skills:
            isinstance(skills, list) and len(skills) > 0
        ).sum()
    )

    empty_or_malformed = (
        len(parsed_skills) - successfully_parsed
    )

    job_text_lengths = output["job_text"].str.len()

    output.to_csv(
        JOB_OUTPUT,
        index=False,
        encoding="utf-8"
    )

    logger.info(
        "Cleaned shape: %d rows × %d columns",
        output.shape[0],
        output.shape[1]
    )

    logger.info(
        "Successfully parsed skill lists: %d",
        successfully_parsed
    )

    logger.info(
        "Empty/malformed skill lists: %d",
        empty_or_malformed
    )

    logger.info(
        "Saved: %s",
        JOB_OUTPUT
    )

    statistics = {
        "original_rows": original_rows,
        "output_rows": len(output),
        "empty_title_count": empty_title_count,
        "empty_description_count": empty_description_count,
        "empty_job_text_count": empty_job_text_count,
        "successfully_parsed_skill_lists": (
            successfully_parsed
        ),
        "empty_or_malformed_skill_lists": (
            empty_or_malformed
        ),
        "text_length": {
            "min": (
                int(job_text_lengths.min())
                if len(job_text_lengths)
                else 0
            ),
            "max": (
                int(job_text_lengths.max())
                if len(job_text_lengths)
                else 0
            ),
            "mean": (
                round(float(job_text_lengths.mean()), 2)
                if len(job_text_lengths)
                else 0
            )
        }
    }

    return output, statistics


# ============================================================================
# REPORT
# ============================================================================

def create_report(
    resume_stats: dict,
    job_stats: dict
) -> dict:

    return {
        "step": "Step 4 - Text Data Preparation",
        "status": "completed",

        "inputs": {
            "resumes": str(RESUME_INPUT),
            "jobs": str(JOB_INPUT)
        },

        "outputs": {
            "resumes": str(RESUME_OUTPUT),
            "jobs": str(JOB_OUTPUT)
        },

        "resumes": resume_stats,

        "jobs": job_stats,

        "transformations": [
            "Normalized column names",
            "Normalized excessive whitespace",
            "Normalized line breaks",
            "Removed obvious HTML tags",
            "Handled null and empty text safely",
            "Preserved meaningful capitalization",
            "Parsed job_skill_set into structured lists",
            "Created unified job_text representation",
            "Preserved source identifiers",
            "Preserved categories"
        ],

        "not_performed": [
            "Skill extraction",
            "ESCO mapping",
            "Embeddings",
            "LLM processing",
            "Resume-job matching",
            "Machine learning training"
        ]
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    logger.info("=" * 70)
    logger.info(
        "AI RESUME ANALYZER - STEP 4 TEXT PREPARATION"
    )
    logger.info("=" * 70)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        _, resume_stats = process_resumes()

        _, job_stats = process_jobs()

        report = create_report(
            resume_stats,
            job_stats
        )

        with open(
            REPORT_OUTPUT,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False
            )

        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 4 COMPLETE")
        logger.info("=" * 70)

        logger.info(
            "Processed resumes: %s",
            RESUME_OUTPUT
        )

        logger.info(
            "Processed jobs: %s",
            JOB_OUTPUT
        )

        logger.info(
            "Processing report: %s",
            REPORT_OUTPUT
        )

        logger.info("=" * 70)

    except Exception as exc:

        logger.error("")
        logger.error("=" * 70)
        logger.error("STEP 4 FAILED")
        logger.error("=" * 70)
        logger.error(str(exc))

        raise


if __name__ == "__main__":
    main()