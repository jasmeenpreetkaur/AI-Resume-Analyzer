"""
AI Resume Analyzer
Step 4: Prepare Resume and Job Text Data

Purpose:
- Load cleaned resume and job datasets.
- Use the PDF parser output (Step 3, resumes_parsed.jsonl) as the resume
  text whenever a PDF was parsed successfully; fall back to the CSV text
  (Resume_str) only for resumes with no usable parsed PDF.
- Normalize text safely.
- Parse job_skill_set into structured lists.
- Create unified text fields.
- Generate a processing report.

Why the parsed PDFs are the primary resume text
-----------------------------------------------
The Streamlit app and Step 13 read uploaded PDFs with the same shared
extractor Step 3 uses (``pdf_text``, with OCR for scanned pages). If the
offline pipeline were built from the CSV text instead, the models and
signals would be calibrated on a different extraction than the one real
users' resumes go through, and Step 3's output (including OCR results)
would never be used by anything.

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

# Step 3 output (03_resume_parser.py)
PARSED_RESUMES_INPUT = (
    PROCESSED_DIR / "resume_parsed" / "resumes_parsed.jsonl"
)

# Resumes that exist as parsed PDFs but have no row in resumes_clean.csv
# are still part of the corpus.
INCLUDE_PDF_ONLY_RESUMES = True

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

def normalize_id(value: Any) -> str:
    """
    Normalise IDs only for joining the CSV and the parsed PDFs
    (1234, "1234", 1234.0 and "1234.0" are the same resume).
    """

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if text.endswith(".0"):
        head = text[:-2]

        if head.isdigit():
            return head

    return text


def count_words(text: str) -> int:
    return len(text.split())


def load_parsed_resumes() -> dict[str, dict]:
    """
    Step 3 records keyed by normalised resume_id. Empty when Step 3 has
    not been run.
    """

    if not PARSED_RESUMES_INPUT.exists():
        logger.warning(
            "Parsed PDF resumes not found (%s). Run Step 3 "
            "(03_resume_parser.py) first; using CSV text only.",
            PARSED_RESUMES_INPUT
        )
        return {}

    records: dict[str, dict] = {}
    malformed = 0

    with PARSED_RESUMES_INPUT.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue

            resume_id = normalize_id(record.get("resume_id"))

            if resume_id:
                records[resume_id] = record

    if malformed:
        logger.warning(
            "Skipped %d malformed lines in %s",
            malformed,
            PARSED_RESUMES_INPUT
        )

    logger.info(
        "Parsed PDF resumes loaded: %d",
        len(records)
    )

    return records


def choose_resume_text(
    csv_text: str,
    parsed: dict | None
) -> tuple[str, str]:
    """
    Pick the resume text and say where it came from.

    pdf_parser   usable parsed PDF (text layer or OCR)
    csv          no parsed PDF for this resume
    csv_fallback the PDF could not be parsed (or parsed to far less
                 text than the CSV, i.e. a low-quality extraction)
    none         neither source has text
    """

    if parsed is None:
        return (csv_text, "csv") if csv_text else ("", "none")

    parsed_text = (
        normalize_text(parsed.get("raw_text", ""))
        if parsed.get("extraction_status") == "success"
        else ""
    )

    if not parsed_text:
        return (
            (csv_text, "csv_fallback")
            if csv_text
            else ("", "none")
        )

    if (
        parsed.get("extraction_quality") == "low"
        and csv_text
        and count_words(csv_text) > count_words(parsed_text)
    ):
        return csv_text, "csv_fallback"

    return parsed_text, "pdf_parser"


def process_resumes() -> tuple[pd.DataFrame, dict]:

    logger.info("")
    logger.info("=" * 70)
    logger.info("Processing resumes")
    logger.info("=" * 70)

    parsed_records = load_parsed_resumes()

    csv_exists = RESUME_INPUT.exists()

    if not csv_exists and not parsed_records:
        raise FileNotFoundError(
            f"Resume dataset not found: {RESUME_INPUT} "
            f"(and no parsed PDFs at {PARSED_RESUMES_INPUT})"
        )

    rows: list[dict] = []
    original_rows = 0
    seen_ids: set[str] = set()
    category_disagreements = 0

    # ------------------------------------------------------------------------
    # CSV rows (text from the parsed PDF where available)
    # ------------------------------------------------------------------------

    if csv_exists:

        df = pd.read_csv(RESUME_INPUT)

        original_rows = len(df)

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

        for position, source in enumerate(
            df.to_dict("records"),
            start=1
        ):

            if "id" in source:
                raw_id = source["id"]
            elif "resume_id" in source:
                raw_id = source["resume_id"]
            else:
                raw_id = position

            key = normalize_id(raw_id)

            parsed = parsed_records.get(key)

            csv_text = normalize_text(source.get("resume_str"))

            text, text_source = choose_resume_text(
                csv_text,
                parsed
            )

            csv_category = (
                normalize_text(source.get("category"))
                if "category" in source
                else ""
            )

            parsed_category = (
                normalize_text(parsed.get("category"))
                if parsed
                else ""
            )

            if (
                csv_category
                and parsed_category
                and csv_category.lower() != parsed_category.lower()
            ):
                category_disagreements += 1

            row = {
                "resume_id": raw_id,
                "category": csv_category or parsed_category,
                "resume_text": text,
                "text_source": text_source,
                "pdf_extraction_quality": (
                    parsed.get("extraction_quality", "")
                    if parsed
                    else ""
                ),
                "ocr_used": bool(parsed and parsed.get("ocr_used")),
            }

            if "resume_html" in source:
                row["resume_html"] = normalize_text(
                    source.get("resume_html")
                )

            rows.append(row)
            seen_ids.add(key)

    # ------------------------------------------------------------------------
    # Parsed PDFs that have no CSV row
    # ------------------------------------------------------------------------

    pdf_only = 0

    if INCLUDE_PDF_ONLY_RESUMES:

        for key, parsed in parsed_records.items():

            if key in seen_ids:
                continue

            text, text_source = choose_resume_text("", parsed)

            # Keep PDF-only resumes even when extraction returns no text.
            # This can happen when OCR is unavailable for scanned PDFs.
            rows.append(
                {
                    "resume_id": parsed.get("resume_id"),
                    "category": normalize_text(parsed.get("category")),
                    "resume_text": text,
                    "text_source": text_source,
                    "pdf_extraction_quality": parsed.get(
                        "extraction_quality",
                        ""
                    ),
                    "ocr_used": bool(parsed.get("ocr_used")),
                }
            )

            pdf_only += 1

    output = pd.DataFrame(rows)

    # IDs that are all digits stay numeric, as they were when they came
    # straight from the CSV.
    numeric_ids = pd.to_numeric(
        output["resume_id"],
        errors="coerce"
    )

    if len(output) and not numeric_ids.isna().any():
        output["resume_id"] = numeric_ids.astype("int64")

    if "resume_html" in output.columns:
        output["resume_html"] = output["resume_html"].fillna("")

    if not original_rows:
        original_rows = len(output)

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    text_lengths = output["resume_text"].str.len()

    empty_text_count = int(
        output["resume_text"].str.strip().eq("").sum()
    )

    source_counts = {
        str(name): int(count)
        for name, count in output["text_source"]
        .value_counts()
        .items()
    }

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
        "Resume text sources: %s",
        source_counts
    )

    if not parsed_records:
        logger.warning(
            "No parsed PDFs were used: every resume text came from the CSV."
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
        "text_source_counts": source_counts,
        "parsed_pdf_input": (
            str(PARSED_RESUMES_INPUT)
            if parsed_records
            else None
        ),
        "parsed_pdf_records": len(parsed_records),
        "pdf_only_resumes_added": pdf_only,
        "resumes_read_with_ocr": int(output["ocr_used"].sum()),
        "category_disagreements_csv_vs_pdf_folder": (
            category_disagreements
        ),
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
            "parsed_pdf_resumes": str(PARSED_RESUMES_INPUT),
            "jobs": str(JOB_INPUT)
        },

        "outputs": {
            "resumes": str(RESUME_OUTPUT),
            "jobs": str(JOB_OUTPUT)
        },

        "resumes": resume_stats,

        "jobs": job_stats,

        "transformations": [
            "Used Step 3 parsed PDF text as resume text "
            "(CSV text only as fallback)",
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