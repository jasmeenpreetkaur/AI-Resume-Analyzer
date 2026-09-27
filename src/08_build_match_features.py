"""
AI Resume Analyzer - Step 8
Build and Validate Resume-Job Match Features

Input:
    data/processed/resume_job_matching.csv

Output:
    data/processed/resume_job_features.csv
    reports/match_features_report.json

Step 8 uses the deterministic matching features already produced
by Step 7 and adds additional useful features.

No:
    - embeddings
    - LLM processing
    - RAG
    - ML training
    - hardcoded category values
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

INPUT_FILE = PROCESSED_DIR / "resume_job_matching.csv"
OUTPUT_FILE = PROCESSED_DIR / "resume_job_features.csv"
REPORT_FILE = REPORTS_DIR / "match_features_report.json"


# =====================================================================
# LOGGING
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


# =====================================================================
# COLUMN UTILITIES
# =====================================================================

def normalize_column_name(value: Any) -> str:
    """Normalize column names for flexible detection."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    """
    Find a column without depending on capitalization or punctuation.
    """

    normalized_columns = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized_candidate = normalize_column_name(candidate)

        if normalized_candidate in normalized_columns:
            return normalized_columns[normalized_candidate]

    if required:
        raise ValueError(
            "Required column not found.\n"
            f"Candidates: {candidates}\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    return None


# =====================================================================
# COLUMN DETECTION
# =====================================================================

def detect_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """
    Detect the columns actually produced by Step 7.

    Step 8 intentionally does NOT require raw resume/job skill lists,
    because Step 7 has already converted them into matching features.
    """

    return {
        "resume_id": find_column(
            df,
            [
                "resume_id",
            ],
        ),

        "job_id": find_column(
            df,
            [
                "job_id",
            ],
        ),

        "resume_category": find_column(
            df,
            [
                "resume_category",
                "category_resume",
            ],
        ),

        "job_category": find_column(
            df,
            [
                "job_category",
                "category_job",
            ],
        ),

        "job_title": find_column(
            df,
            [
                "job_title",
                "title",
            ],
            required=False,
        ),

        "shared_skill_count": find_column(
            df,
            [
                "shared_skill_count",
                "common_skill_count",
            ],
        ),

        "resume_skill_count": find_column(
            df,
            [
                "resume_skill_count",
            ],
        ),

        "job_skill_count": find_column(
            df,
            [
                "job_skill_count",
            ],
        ),

        "jaccard_similarity": find_column(
            df,
            [
                "jaccard_similarity",
                "skill_jaccard_similarity",
            ],
        ),

        "resume_skill_coverage": find_column(
            df,
            [
                "resume_skill_coverage",
            ],
        ),

        "job_skill_coverage": find_column(
            df,
            [
                "job_skill_coverage",
            ],
        ),

        "shared_skills": find_column(
            df,
            [
                "shared_esco_skill_uris",
                "shared_skill_uris",
                "shared_skills",
            ],
            required=False,
        ),
    }


# =====================================================================
# VALUE HELPERS
# =====================================================================

def safe_numeric(
    series: pd.Series,
) -> pd.Series:
    """Convert values to numeric safely."""

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def normalize_category(
    series: pd.Series,
) -> pd.Series:
    """Normalize categories only for comparison."""

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )


def parse_shared_skill_count(
    value: Any,
) -> int:
    """
    Determine number of shared skills from a list-like field.

    Used only when the explicit shared_skill_count is unavailable
    or needs validation.
    """

    if value is None:
        return 0

    if isinstance(value, float) and np.isnan(value):
        return 0

    if isinstance(value, (list, tuple, set)):
        return len(value)

    text = str(value).strip()

    if not text or text in {"[]", "{}", "nan", "None"}:
        return 0

    # URI lists and Python-list strings.
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()

        if not inner:
            return 0

        parts = [
            item.strip()
            for item in inner.split(",")
            if item.strip()
        ]

        return len(parts)

    # Semicolon-separated fallback.
    if ";" in text:
        return len(
            [
                item
                for item in text.split(";")
                if item.strip()
            ]
        )

    # A single skill.
    return 1


# =====================================================================
# FEATURE BUILDING
# =====================================================================

def build_features(
    df: pd.DataFrame,
    columns: dict[str, str | None],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Validate Step 7 features and add additional deterministic features.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("Preparing matching features")
    logger.info("=" * 70)

    shared_col = columns["shared_skill_count"]
    resume_count_col = columns["resume_skill_count"]
    job_count_col = columns["job_skill_count"]
    jaccard_col = columns["jaccard_similarity"]
    resume_cov_col = columns["resume_skill_coverage"]
    job_cov_col = columns["job_skill_coverage"]

    # ---------------------------------------------------------------
    # Convert existing numeric columns
    # ---------------------------------------------------------------

    numeric_columns = [
        shared_col,
        resume_count_col,
        job_count_col,
        jaccard_col,
        resume_cov_col,
        job_cov_col,
    ]

    for column in numeric_columns:
        df[column] = safe_numeric(df[column])

    # ---------------------------------------------------------------
    # Existing feature aliases
    # ---------------------------------------------------------------

    # Keep Step 7 names untouched, but provide consistent ML-friendly
    # feature names for downstream stages.

    df["feature_shared_skill_count"] = df[shared_col]
    df["feature_resume_skill_count"] = df[resume_count_col]
    df["feature_job_skill_count"] = df[job_count_col]
    df["feature_jaccard_similarity"] = df[jaccard_col]
    df["feature_resume_skill_coverage"] = df[resume_cov_col]
    df["feature_job_skill_coverage"] = df[job_cov_col]

    # ---------------------------------------------------------------
    # Skill count difference
    # ---------------------------------------------------------------

    df["skill_count_difference"] = (
        df[resume_count_col]
        - df[job_count_col]
    ).abs()

    # ---------------------------------------------------------------
    # Minimum skill coverage
    # ---------------------------------------------------------------

    minimum_counts = pd.concat(
        [
            df[resume_count_col],
            df[job_count_col],
        ],
        axis=1,
    ).min(axis=1)

    df["minimum_skill_coverage"] = np.where(
        minimum_counts > 0,
        df[shared_col] / minimum_counts,
        0.0,
    )

    # ---------------------------------------------------------------
    # Maximum skill coverage
    # ---------------------------------------------------------------

    maximum_counts = pd.concat(
        [
            df[resume_count_col],
            df[job_count_col],
        ],
        axis=1,
    ).max(axis=1)

    df["maximum_skill_overlap_ratio"] = np.where(
        maximum_counts > 0,
        df[shared_col] / maximum_counts,
        0.0,
    )

    # ---------------------------------------------------------------
    # Shared-skill indicator
    # ---------------------------------------------------------------

    df["has_shared_skills"] = (
        df[shared_col] > 0
    ).astype("int8")

    # ---------------------------------------------------------------
    # Category features
    # ---------------------------------------------------------------

    resume_category_col = columns["resume_category"]
    job_category_col = columns["job_category"]

    resume_categories = normalize_category(
        df[resume_category_col]
    )

    job_categories = normalize_category(
        df[job_category_col]
    )

    category_available = (
        resume_categories.ne("")
        & job_categories.ne("")
    )

    df["category_available"] = (
        category_available
    ).astype("int8")

    df["category_match"] = (
        category_available
        & resume_categories.eq(job_categories)
    ).astype("int8")

    # ---------------------------------------------------------------
    # Job title availability
    # ---------------------------------------------------------------

    title_col = columns["job_title"]

    if title_col is not None:

        title_available = (
            df[title_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )

        df["job_title_available"] = (
            title_available
        ).astype("int8")

    else:

        df["job_title_available"] = 0

    # ---------------------------------------------------------------
    # Shared URI availability
    # ---------------------------------------------------------------

    shared_skills_col = columns["shared_skills"]

    if shared_skills_col is not None:

        shared_available = (
            df[shared_skills_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )

        df["shared_skill_uris_available"] = (
            shared_available
        ).astype("int8")

    else:

        df["shared_skill_uris_available"] = (
            df[shared_col] > 0
        ).astype("int8")

    # ---------------------------------------------------------------
    # Consistency validation against shared URI list
    # ---------------------------------------------------------------

    if shared_skills_col is not None:

        calculated_shared_counts = (
            df[shared_skills_col]
            .map(parse_shared_skill_count)
        )

        df["shared_skill_count_consistent"] = (
            calculated_shared_counts
            == df[shared_col].fillna(0)
        ).astype("int8")

    else:

        df["shared_skill_count_consistent"] = 1

    feature_columns = [
        "feature_shared_skill_count",
        "feature_resume_skill_count",
        "feature_job_skill_count",
        "feature_jaccard_similarity",
        "feature_resume_skill_coverage",
        "feature_job_skill_coverage",
        "skill_count_difference",
        "minimum_skill_coverage",
        "maximum_skill_overlap_ratio",
        "has_shared_skills",
        "category_available",
        "category_match",
        "job_title_available",
        "shared_skill_uris_available",
        "shared_skill_count_consistent",
    ]

    report = {
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "rows_with_shared_skills": int(
            df["has_shared_skills"].sum()
        ),
        "rows_without_shared_skills": int(
            (df["has_shared_skills"] == 0).sum()
        ),
        "category_information_available": int(
            df["category_available"].sum()
        ),
        "category_matches": int(
            df["category_match"].sum()
        ),
        "shared_skill_count_consistency": {
            "consistent": int(
                df["shared_skill_count_consistent"].sum()
            ),
            "inconsistent": int(
                (
                    df["shared_skill_count_consistent"] == 0
                ).sum()
            ),
        },
    }

    return df, report


# =====================================================================
# VALIDATION
# =====================================================================

def validate_features(
    df: pd.DataFrame,
    columns: dict[str, str | None],
    original_rows: int,
) -> dict[str, Any]:

    logger.info("")
    logger.info("=" * 70)
    logger.info("Validating feature dataset")
    logger.info("=" * 70)

    required_features = [
        "feature_shared_skill_count",
        "feature_resume_skill_count",
        "feature_job_skill_count",
        "feature_jaccard_similarity",
        "feature_resume_skill_coverage",
        "feature_job_skill_coverage",
        "skill_count_difference",
        "minimum_skill_coverage",
        "maximum_skill_overlap_ratio",
        "has_shared_skills",
        "category_available",
        "category_match",
    ]

    missing_features = [
        column
        for column in required_features
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}"
        )

    # ---------------------------------------------------------------
    # Row count
    # ---------------------------------------------------------------

    row_count_ok = (
        len(df) == original_rows
    )

    if not row_count_ok:
        raise ValueError(
            "Row count changed unexpectedly. "
            f"Input={original_rows}, Output={len(df)}"
        )

    # ---------------------------------------------------------------
    # Duplicate resume-job pairs
    # ---------------------------------------------------------------

    resume_id_col = columns["resume_id"]
    job_id_col = columns["job_id"]

    duplicate_pairs = int(
        df.duplicated(
            subset=[
                resume_id_col,
                job_id_col,
            ]
        ).sum()
    )

    # ---------------------------------------------------------------
    # Negative values
    # ---------------------------------------------------------------

    count_features = [
        "feature_shared_skill_count",
        "feature_resume_skill_count",
        "feature_job_skill_count",
        "skill_count_difference",
    ]

    negative_counts = {}

    for column in count_features:

        negative_counts[column] = int(
            (
                df[column] < 0
            ).sum()
        )

    # ---------------------------------------------------------------
    # Shared skill constraints
    # ---------------------------------------------------------------

    invalid_shared = (
        (
            df["feature_shared_skill_count"]
            > df["feature_resume_skill_count"]
        )
        |
        (
            df["feature_shared_skill_count"]
            > df["feature_job_skill_count"]
        )
    )

    invalid_shared_count = int(
        invalid_shared.sum()
    )

    # ---------------------------------------------------------------
    # Similarity range
    # ---------------------------------------------------------------

    bounded_features = [
        "feature_jaccard_similarity",
        "feature_resume_skill_coverage",
        "feature_job_skill_coverage",
        "minimum_skill_coverage",
        "maximum_skill_overlap_ratio",
    ]

    range_errors = {}

    for column in bounded_features:

        invalid = (
            (df[column] < 0)
            |
            (df[column] > 1)
            |
            (~np.isfinite(df[column]))
        )

        range_errors[column] = int(
            invalid.sum()
        )

    # ---------------------------------------------------------------
    # Missing numerical values
    # ---------------------------------------------------------------

    numerical_features = (
        count_features
        + bounded_features
    )

    missing_values = {
        column: int(
            df[column].isna().sum()
        )
        for column in numerical_features
    }

    # ---------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------

    statistics = {}

    for column in numerical_features:

        series = df[column]

        statistics[column] = {
            "mean": (
                float(series.mean())
                if not series.empty
                else None
            ),
            "median": (
                float(series.median())
                if not series.empty
                else None
            ),
            "min": (
                float(series.min())
                if not series.empty
                else None
            ),
            "max": (
                float(series.max())
                if not series.empty
                else None
            ),
        }

    # ---------------------------------------------------------------
    # Validation result
    # ---------------------------------------------------------------

    validation_passed = (
        row_count_ok
        and duplicate_pairs == 0
        and invalid_shared_count == 0
        and all(
            value == 0
            for value in negative_counts.values()
        )
        and all(
            value == 0
            for value in range_errors.values()
        )
        and all(
            value == 0
            for value in missing_values.values()
        )
    )

    logger.info(
        "Rows: %d",
        len(df),
    )

    logger.info(
        "Duplicate resume-job pairs: %d",
        duplicate_pairs,
    )

    logger.info(
        "Invalid shared-skill rows: %d",
        invalid_shared_count,
    )

    logger.info(
        "Rows with shared skills: %d",
        int(df["has_shared_skills"].sum()),
    )

    logger.info(
        "Average shared skills: %.2f",
        df["feature_shared_skill_count"].mean(),
    )

    logger.info(
        "Average Jaccard similarity: %.4f",
        df["feature_jaccard_similarity"].mean(),
    )

    logger.info(
        "Average job skill coverage: %.4f",
        df["feature_job_skill_coverage"].mean(),
    )

    logger.info(
        "Validation: %s",
        "PASSED" if validation_passed else "FAILED",
    )

    if not validation_passed:
        raise ValueError(
            "Feature validation failed."
        )

    return {
        "validation_passed": validation_passed,
        "row_count": int(len(df)),
        "duplicate_resume_job_pairs": duplicate_pairs,
        "invalid_shared_skill_rows": invalid_shared_count,
        "negative_values": negative_counts,
        "range_errors": range_errors,
        "missing_values": missing_values,
        "statistics": statistics,
    }


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    logger.info("=" * 70)
    logger.info("AI RESUME ANALYZER - STEP 8 MATCH FEATURES")
    logger.info("=" * 70)

    # ---------------------------------------------------------------
    # Check input
    # ---------------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("Loading Step 7 matching dataset")
    logger.info("=" * 70)

    df = pd.read_csv(
        INPUT_FILE
    )

    logger.info(
        "Input shape: %d rows × %d columns",
        df.shape[0],
        df.shape[1],
    )

    logger.info(
        "Input columns: %s",
        list(df.columns),
    )

    original_rows = len(df)

    if original_rows == 0:

        raise ValueError(
            "Step 7 matching dataset contains zero rows."
        )

    # ---------------------------------------------------------------
    # Detect columns
    # ---------------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("Detecting columns")
    logger.info("=" * 70)

    columns = detect_columns(df)

    for key, value in columns.items():

        logger.info(
            "%s: %s",
            key,
            value,
        )

    # ---------------------------------------------------------------
    # Build features
    # ---------------------------------------------------------------

    featured_df, feature_report = build_features(
        df,
        columns,
    )

    # ---------------------------------------------------------------
    # Validate
    # ---------------------------------------------------------------

    validation_report = validate_features(
        featured_df,
        columns,
        original_rows,
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("Saving feature dataset")
    logger.info("=" * 70)

    featured_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    logger.info(
        "Saved: %s",
        OUTPUT_FILE,
    )

    logger.info(
        "Output shape: %d rows × %d columns",
        featured_df.shape[0],
        featured_df.shape[1],
    )

    # ---------------------------------------------------------------
    # Missing-value summary
    # ---------------------------------------------------------------

    missing_summary = {
        column: int(
            featured_df[column].isna().sum()
        )
        for column in featured_df.columns
    }

    # ---------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------

    report = {
        "step": "8",
        "step_name": "match_features",
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "input_row_count": int(original_rows),
        "output_row_count": int(len(featured_df)),
        "input_column_count": int(df.shape[1]),
        "output_column_count": int(featured_df.shape[1]),
        "detected_columns": columns,
        "features": feature_report,
        "missing_value_summary": missing_summary,
        "validation": validation_report,
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Report saved: %s",
        REPORT_FILE,
    )

    # ---------------------------------------------------------------
    # Complete
    # ---------------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 8 COMPLETE")
    logger.info("=" * 70)

    logger.info(
        "Matching pairs processed: %d",
        original_rows,
    )

    logger.info(
        "Feature dataset: %s",
        OUTPUT_FILE,
    )

    logger.info(
        "Report: %s",
        REPORT_FILE,
    )

    logger.info("=" * 70)


if __name__ == "__main__":
    main()