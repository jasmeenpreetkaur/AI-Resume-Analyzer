from pathlib import Path
import ast
import json
import logging
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# DATASET PATHS
# ============================================================

DATASETS = {
    "resumes": {
        "input": DATA_DIR / "resume_dataset" / "Resume" / "Resume.csv",
        "output": PROCESSED_DIR / "resumes_clean.csv"
    },

    "jobs": {
        "input": DATA_DIR / "jobs" / "all_job_post.csv",
        "output": PROCESSED_DIR / "jobs_clean.csv"
    },

    "esco_skills": {
        "input": DATA_DIR / "external" / "skills_en.csv",
        "output": PROCESSED_DIR / "esco_skills_clean.csv"
    },

    "esco_occupations": {
        "input": DATA_DIR / "external" / "occupations_en.csv",
        "output": PROCESSED_DIR / "esco_occupations_clean.csv"
    },

    "esco_occupation_skill_relations": {
        "input": DATA_DIR / "external" / "occupationSkillRelations_en.csv",
        "output": PROCESSED_DIR / "esco_occupation_skill_relations_clean.csv"
    },

    "esco_skill_hierarchy": {
        "input": DATA_DIR / "external" / "skillsHierarchy_en.csv",
        "output": PROCESSED_DIR / "esco_skill_hierarchy_clean.csv"
    },

    "esco_skill_skill_relations": {
        "input": DATA_DIR / "external" / "skillSkillRelations_en.csv",
        "output": PROCESSED_DIR / "esco_skill_skill_relations_clean.csv"
    }
}


# ============================================================
# COLUMN NAME NORMALIZATION
# ============================================================

def normalize_column_name(column):
    """
    Convert column names to consistent snake_case.
    """

    column = str(column).strip()
    column = column.replace("-", "_")
    column = column.replace(" ", "_")

    return column.lower()


def normalize_columns(df):
    """
    Normalize all dataframe column names.
    """

    df = df.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]

    return df


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):
    """
    Clean unnecessary whitespace while preserving the
    actual capitalization and content.
    """

    if pd.isna(value):
        return value

    if not isinstance(value, str):
        return value

    # Normalize line breaks
    value = value.replace("\r\n", "\n")
    value = value.replace("\r", "\n")

    # Remove trailing spaces from each line
    lines = [line.strip() for line in value.split("\n")]

    # Remove excessive blank lines
    cleaned_lines = []

    previous_blank = False

    for line in lines:

        if line == "":
            if not previous_blank:
                cleaned_lines.append(line)

            previous_blank = True

        else:
            cleaned_lines.append(line)
            previous_blank = False

    return "\n".join(cleaned_lines).strip()


def clean_text_columns(df):
    """
    Apply text cleaning to object/string columns.
    """

    df = df.copy()

    for column in df.columns:

        if pd.api.types.is_object_dtype(df[column]):

            df[column] = df[column].apply(clean_text)

    return df


# ============================================================
# MISSING VALUE REPORT
# ============================================================

def missing_value_summary(df):

    summary = {}

    for column in df.columns:

        missing_count = int(df[column].isna().sum())

        if missing_count > 0:

            summary[column] = {
                "missing_count": missing_count,
                "missing_percentage": round(
                    (missing_count / len(df)) * 100, 2
                ) if len(df) > 0 else 0
            }

    return summary


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def remove_duplicates(df):

    original_count = len(df)

    duplicate_count = int(df.duplicated().sum())

    df = df.drop_duplicates().reset_index(drop=True)

    return df, duplicate_count


# ============================================================
# JOB SKILL SET PARSER
# ============================================================

def parse_job_skill_set(value):

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    value = str(value).strip()

    if not value:
        return []

    try:

        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):

            cleaned_skills = []

            for skill in parsed:

                if skill is None:
                    continue

                skill = clean_text(skill)

                if skill:
                    cleaned_skills.append(skill)

            return cleaned_skills

    except (ValueError, SyntaxError):

        logger.warning(
            "Could not parse job_skill_set value: %s",
            value[:100]
        )

    # Fallback
    return [value]


# ============================================================
# RESUME CLEANING
# ============================================================

def clean_resumes(df):

    transformations = []

    df = normalize_columns(df)

    transformations.append("Normalized column names to snake_case.")

    df = clean_text_columns(df)

    transformations.append(
        "Normalized whitespace in text fields while preserving capitalization."
    )

    if "category" in df.columns:
        df["category"] = df["category"].apply(clean_text)

    df, duplicates = remove_duplicates(df)

    transformations.append(
        f"Removed {duplicates} exact duplicate rows."
    )

    return df, transformations, duplicates


# ============================================================
# JOB CLEANING
# ============================================================

def clean_jobs(df):

    transformations = []

    df = normalize_columns(df)

    transformations.append("Normalized column names to snake_case.")

    df = clean_text_columns(df)

    transformations.append(
        "Normalized whitespace in job text fields."
    )

    if "job_skill_set" in df.columns:

        df["job_skill_set"] = df["job_skill_set"].apply(
            parse_job_skill_set
        )

        # Store as JSON string because CSV does not have
        # a native Python list datatype.
        df["job_skill_set"] = df["job_skill_set"].apply(
            lambda skills: json.dumps(
                skills,
                ensure_ascii=False
            )
        )

        transformations.append(
            "Safely parsed job_skill_set using ast.literal_eval "
            "and stored it as JSON-compatible text."
        )

    df, duplicates = remove_duplicates(df)

    transformations.append(
        f"Removed {duplicates} exact duplicate rows."
    )

    return df, transformations, duplicates


# ============================================================
# GENERIC ESCO CLEANING
# ============================================================

def clean_esco(df):

    transformations = []

    df = normalize_columns(df)

    transformations.append(
        "Normalized column names to snake_case."
    )

    df = clean_text_columns(df)

    transformations.append(
        "Normalized whitespace in ESCO text fields."
    )

    df, duplicates = remove_duplicates(df)

    transformations.append(
        f"Removed {duplicates} exact duplicate rows."
    )

    return df, transformations, duplicates


# ============================================================
# PDF DIRECTORY ANALYSIS
# ============================================================

def analyze_pdf_directory():

    pdf_root = DATA_DIR / "resume_dataset" / "data"

    result = {
        "pdf_root": str(pdf_root),
        "total_pdfs": 0,
        "categories": {},
        "category_detection_method": "Immediate parent folder of each PDF"
    }

    if not pdf_root.exists():

        result["error"] = "PDF directory does not exist."

        return result

    pdf_files = list(pdf_root.rglob("*.pdf"))

    result["total_pdfs"] = len(pdf_files)

    category_counts = {}

    for pdf in pdf_files:

        category = pdf.parent.name.upper()

        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

    result["categories"] = dict(
        sorted(
            category_counts.items(),
            key=lambda item: item[0]
        )
    )

    return result


# ============================================================
# CLEAN ONE DATASET
# ============================================================

def clean_dataset(name, config):

    input_path = config["input"]
    output_path = config["output"]

    logger.info("")
    logger.info("=" * 70)
    logger.info("Processing: %s", name)
    logger.info("=" * 70)

    if not input_path.exists():

        logger.error(
            "Input file not found: %s",
            input_path
        )

        return {
            "status": "failed",
            "error": f"Input file not found: {input_path}"
        }

    df = pd.read_csv(input_path)

    original_rows = len(df)

    logger.info(
        "Original shape: %s rows × %s columns",
        df.shape[0],
        df.shape[1]
    )

    original_missing = missing_value_summary(df)

    if name == "resumes":

        cleaned_df, transformations, duplicates = clean_resumes(df)

    elif name == "jobs":

        cleaned_df, transformations, duplicates = clean_jobs(df)

    else:

        cleaned_df, transformations, duplicates = clean_esco(df)

    cleaned_missing = missing_value_summary(cleaned_df)

    cleaned_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    # Reload to validate output
    validation_df = pd.read_csv(output_path)

    validation_passed = (
        len(validation_df) == len(cleaned_df)
        and list(validation_df.columns)
        == list(cleaned_df.columns)
    )

    logger.info(
        "Cleaned shape: %s rows × %s columns",
        cleaned_df.shape[0],
        cleaned_df.shape[1]
    )

    logger.info(
        "Duplicates removed: %s",
        duplicates
    )

    logger.info(
        "Saved: %s",
        output_path
    )

    logger.info(
        "Validation: %s",
        "PASSED" if validation_passed else "FAILED"
    )

    return {
        "status": "success" if validation_passed else "failed",
        "input_path": str(input_path.relative_to(PROJECT_ROOT)),
        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
        "original_rows": original_rows,
        "cleaned_rows": len(cleaned_df),
        "rows_removed": original_rows - len(cleaned_df),
        "original_columns": list(df.columns),
        "cleaned_columns": list(cleaned_df.columns),
        "duplicate_rows_removed": duplicates,
        "original_missing_values": original_missing,
        "cleaned_missing_values": cleaned_missing,
        "transformations": transformations,
        "validation_passed": validation_passed
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("AI RESUME ANALYZER - STEP 3 DATA CLEANING")
    print("=" * 70)

    report = {
        "step": "Step 3 - Data Cleaning",
        "status": "started",
        "datasets": {},
        "pdf_analysis": {}
    }

    # --------------------------------------------------------
    # Clean datasets
    # --------------------------------------------------------

    for name, config in DATASETS.items():

        result = clean_dataset(name, config)

        report["datasets"][name] = result

    # --------------------------------------------------------
    # Analyze PDFs
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PDF DIRECTORY ANALYSIS")
    print("=" * 70)

    pdf_analysis = analyze_pdf_directory()

    report["pdf_analysis"] = pdf_analysis

    print(
        f"\nTotal PDFs found: "
        f"{pdf_analysis.get('total_pdfs', 0)}"
    )

    if pdf_analysis.get("categories"):

        print("\nPDF categories:")

        for category, count in pdf_analysis["categories"].items():

            print(f"  {category}: {count}")

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    dataset_statuses = [
        result.get("status") == "success"
        for result in report["datasets"].values()
    ]

    all_successful = all(dataset_statuses)

    if pdf_analysis.get("total_pdfs", 0) == 2484:

        pdf_status = "PASSED"

    else:

        pdf_status = "CHECK_REQUIRED"

    report["status"] = (
        "completed"
        if all_successful
        else "completed_with_errors"
    )

    report["pdf_validation"] = pdf_status

    report_path = REPORTS_DIR / "cleaning_report.json"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE")
    print("=" * 70)

    print(
        f"\nCleaning report:"
        f"\n{report_path}"
    )

    print("\nProcessed datasets:")

    for name, result in report["datasets"].items():

        status = result.get("status", "unknown")

        # Use ASCII status markers so the script also works with Windows
        # consoles using cp1252 instead of UTF-8. Unicode status glyphs can
        # raise UnicodeEncodeError and abort otherwise successful pipelines.
        marker = "OK" if status == "success" else "ERROR"
        print(f"  [{marker}] {name}: {status}")

    print(
        f"\nPDF validation: {pdf_status}"
    )

    print("\nProcessed data location:")
    print(PROCESSED_DIR)


if __name__ == "__main__":
    main()