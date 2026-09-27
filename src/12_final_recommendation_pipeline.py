"""
======================================================================
AI RESUME ANALYZER - STEP 12 FINAL RECOMMENDATION PIPELINE
======================================================================

Purpose:
    Convert the validated Step 10 hybrid matching scores into a
    user-facing job recommendation dataset.

IMPORTANT:
- Uses Step 10 hybrid_match_score as the authoritative ranking signal.
- Does NOT retrain any model.
- Does NOT create pseudo-labels.
- Does NOT create hiring labels.
- Does NOT claim probability of getting hired.
- Does NOT use Step 9 evidence_score as a ranking feature.
- Uses actual resume/job information for recommendation explanations.
- Produces transparent match evidence and skill-gap information.
- Produces resume-wise ranked job recommendations.
======================================================================
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ======================================================================
# PATH CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEP10_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "step10_matching_predictions.csv"
)

RESUME_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "resumes_esco_mapped.csv"
)

JOB_PATH = (
    PROJECT_ROOT
    / "data"
    / "jobs"
    / "all_job_post.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "recommendations"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_OUTPUT_PATH = OUTPUT_DIR / "final_job_recommendations.csv"
TOP10_OUTPUT_PATH = OUTPUT_DIR / "top10_job_recommendations.csv"
REPORT_PATH = REPORT_DIR / "final_recommendation_report.json"


# ======================================================================
# CONFIGURATION
# ======================================================================

TOP_K = 10

# Minimum number of recommendations to retain when available.
MIN_RECOMMENDATIONS = 5

# We do not call this a probability.
SCORE_DISPLAY_SCALE = 100.0

# Maximum number of characters used when storing descriptions.
MAX_DESCRIPTION_LENGTH = 5000


# ======================================================================
# LOGGING
# ======================================================================

def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def info(message: str) -> None:
    print(message)


# ======================================================================
# GENERAL HELPERS
# ======================================================================

def normalize_id(value) -> str:
    """
    Normalize IDs only for matching across source files.

    IDs are kept as strings to avoid accidental numeric conversion such
    as 001 -> 1.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        try:
            number = float(text)
            if number.is_integer():
                return str(int(number))
        except Exception:
            pass

    return text


def clean_text(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_skill(skill: str) -> str:
    skill = clean_text(skill).lower()

    skill = skill.replace("&", " and ")
    skill = re.sub(r"[^a-z0-9+#.\- ]+", " ", skill)
    skill = re.sub(r"\s+", " ", skill)

    return skill.strip()


def truncate_text(text: str, max_length: int) -> str:
    text = clean_text(text)

    if len(text) <= max_length:
        return text

    return text[: max_length - 3].rstrip() + "..."


def first_existing_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = True,
    description: str = "",
) -> Optional[str]:

    columns_lower = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in columns_lower:
            return columns_lower[candidate.lower()]

    if required:
        raise ValueError(
            f"Required column not found for {description}. "
            f"Tried: {list(candidates)}"
        )

    return None


def find_skill_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "skills",
        "skill",
        "resume_skills",
        "job_skills",
        "ESCO_skills",
        "esco_skills",
        "skill_list",
        "skills_list",
        "extracted_skills",
        "matched_skills",
    ]

    return first_existing_column(
        df,
        candidates,
        required=False,
        description="skills",
    )


def parse_skill_value(value) -> List[str]:
    """
    Handles common representations:
        - Python list
        - JSON list
        - semicolon-separated
        - comma-separated
        - pipe-separated
        - plain text
    """

    if value is None or pd.isna(value):
        return []

    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)

    else:
        text = str(value).strip()

        if not text:
            return []

        # Try JSON representation first.
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)

                if isinstance(parsed, list):
                    raw_values = parsed
                else:
                    raw_values = [text]

            except Exception:
                raw_values = re.split(r"[;,|]", text)

        else:
            raw_values = re.split(r"[;,|]", text)

    cleaned = []

    for item in raw_values:
        item = normalize_skill(str(item))

        if item:
            cleaned.append(item)

    return sorted(set(cleaned))


def extract_skills_from_text(text: str) -> List[str]:
    """
    Fallback skill extraction.

    This is intentionally conservative. It is NOT used to create labels.
    It only provides an explanation when explicit skill columns are absent.
    """

    text = normalize_skill(text)

    if not text:
        return []

    known_skills = [
        "python",
        "java",
        "c++",
        "c#",
        "javascript",
        "typescript",
        "sql",
        "mysql",
        "postgresql",
        "mongodb",
        "html",
        "css",
        "react",
        "angular",
        "node.js",
        "nodejs",
        "django",
        "flask",
        "fastapi",
        "spring",
        "tensorflow",
        "pytorch",
        "keras",
        "scikit-learn",
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "natural language processing",
        "nlp",
        "computer vision",
        "data analysis",
        "data science",
        "pandas",
        "numpy",
        "opencv",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "git",
        "github",
        "linux",
        "excel",
        "power bi",
        "tableau",
        "communication",
        "leadership",
        "project management",
        "marketing",
        "recruitment",
        "human resources",
        "customer service",
        "sales",
        "accounting",
        "finance",
    ]

    found = []

    for skill in known_skills:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text):
            found.append(skill)

    return sorted(set(found))


# ======================================================================
# COLUMN DETECTION
# ======================================================================

def detect_step10_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:

    mapping = {}

    mapping["resume_id"] = first_existing_column(
        df,
        ["resume_id", "Resume_ID", "resumeId"],
        description="Step 10 resume ID",
    )

    mapping["job_id"] = first_existing_column(
        df,
        ["job_id", "Job_ID", "jobId"],
        description="Step 10 job ID",
    )

    mapping["hybrid_score"] = first_existing_column(
        df,
        [
            "hybrid_match_score",
            "hybrid_score",
            "match_score",
            "final_match_score",
        ],
        description="Step 10 hybrid score",
    )

    mapping["semantic_similarity"] = first_existing_column(
        df,
        ["semantic_similarity"],
        required=False,
        description="semantic similarity",
    )

    mapping["title_similarity"] = first_existing_column(
        df,
        ["title_similarity"],
        required=False,
        description="title similarity",
    )

    mapping["skill_match_signal"] = first_existing_column(
        df,
        ["skill_match_signal"],
        required=False,
        description="skill match signal",
    )

    mapping["category_match_signal"] = first_existing_column(
        df,
        ["category_match_signal"],
        required=False,
        description="category match signal",
    )

    mapping["job_skill_coverage"] = first_existing_column(
        df,
        ["job_skill_coverage"],
        required=False,
        description="job skill coverage",
    )

    mapping["resume_skill_coverage"] = first_existing_column(
        df,
        ["resume_skill_coverage"],
        required=False,
        description="resume skill coverage",
    )

    return mapping


# ======================================================================
# SOURCE LOADING
# ======================================================================

section("AI RESUME ANALYZER - STEP 12 FINAL RECOMMENDATION PIPELINE")

section("Loading Step 10 matching predictions")

if not STEP10_PATH.exists():
    raise FileNotFoundError(
        f"Step 10 predictions not found:\n{STEP10_PATH}"
    )

step10 = pd.read_csv(STEP10_PATH)

info(f"Input shape: {step10.shape}")


section("Detecting Step 10 columns")

columns = detect_step10_columns(step10)

for key, value in columns.items():
    info(f"{key}: {value}")


section("Validating Step 10 recommendation source")

if step10.empty:
    raise ValueError("Step 10 prediction dataset is empty.")

resume_col = columns["resume_id"]
job_col = columns["job_id"]
score_col = columns["hybrid_score"]

step10[resume_col] = step10[resume_col].map(normalize_id)
step10[job_col] = step10[job_col].map(normalize_id)

step10[score_col] = pd.to_numeric(
    step10[score_col],
    errors="coerce",
)

if step10[resume_col].eq("").any():
    raise ValueError("Missing resume IDs detected.")

if step10[job_col].eq("").any():
    raise ValueError("Missing job IDs detected.")

if step10[score_col].isna().any():
    raise ValueError("Missing hybrid match scores detected.")

if step10[[resume_col, job_col]].duplicated().any():
    raise ValueError(
        "Duplicate resume-job pairs detected in Step 10."
    )

info(f"Rows: {len(step10):,}")
info(f"Unique resumes: {step10[resume_col].nunique():,}")
info(f"Unique jobs: {step10[job_col].nunique():,}")
info(f"Duplicate pairs: {step10[[resume_col, job_col]].duplicated().sum()}")
info(f"Hybrid score mean: {step10[score_col].mean():.4f}")
info(f"Hybrid score median: {step10[score_col].median():.4f}")


# ======================================================================
# LOAD ACTUAL RESUME DATA
# ======================================================================

section("Loading actual resume data")

if not RESUME_PATH.exists():
    raise FileNotFoundError(
        f"Resume source not found:\n{RESUME_PATH}"
    )

resumes = pd.read_csv(RESUME_PATH)

info(f"Resume source: {RESUME_PATH}")
info(f"Resume records: {len(resumes):,}")

resume_id_source = first_existing_column(
    resumes,
    ["resume_id", "Resume_ID", "resumeId", "id"],
    description="resume source ID",
)

resumes[resume_id_source] = resumes[resume_id_source].map(
    normalize_id
)

resume_skill_col = find_skill_column(resumes)

resume_text_col = first_existing_column(
    resumes,
    [
        "resume_text",
        "Resume_str",
        "resume",
        "text",
        "Resume",
        "cv_text",
    ],
    required=False,
    description="resume text",
)

info(f"Resume ID column: {resume_id_source}")
info(f"Resume skill column: {resume_skill_col}")
info(f"Resume text column: {resume_text_col}")


# ======================================================================
# LOAD ACTUAL JOB DATA
# ======================================================================

section("Loading actual job data")

if not JOB_PATH.exists():
    raise FileNotFoundError(
        f"Job source not found:\n{JOB_PATH}"
    )

jobs = pd.read_csv(JOB_PATH)

info(f"Job source: {JOB_PATH}")
info(f"Job records: {len(jobs):,}")

job_id_source = first_existing_column(
    jobs,
    ["job_id", "Job_ID", "jobId", "id"],
    description="job source ID",
)

jobs[job_id_source] = jobs[job_id_source].map(normalize_id)

job_title_col = first_existing_column(
    jobs,
    [
        "job_title",
        "title",
        "Job Title",
        "jobtitle",
        "position",
    ],
    required=False,
    description="job title",
)

job_description_col = first_existing_column(
    jobs,
    [
        "job_description",
        "description",
        "Job Description",
        "jobdescription",
        "details",
    ],
    required=False,
    description="job description",
)

job_company_col = first_existing_column(
    jobs,
    [
        "company",
        "company_name",
        "Company",
        "employer",
        "organization",
    ],
    required=False,
    description="company",
)

job_location_col = first_existing_column(
    jobs,
    [
        "location",
        "job_location",
        "Location",
        "city",
        "work_location",
    ],
    required=False,
    description="location",
)

job_skill_col = find_skill_column(jobs)

info(f"Job ID column: {job_id_source}")
info(f"Job title column: {job_title_col}")
info(f"Job description column: {job_description_col}")
info(f"Job company column: {job_company_col}")
info(f"Job location column: {job_location_col}")
info(f"Job skill column: {job_skill_col}")


# ======================================================================
# BUILD SOURCE LOOKUPS
# ======================================================================

section("Building actual resume/job lookups")

resume_lookup = (
    resumes
    .drop_duplicates(subset=[resume_id_source], keep="first")
    .set_index(resume_id_source)
)

job_lookup = (
    jobs
    .drop_duplicates(subset=[job_id_source], keep="first")
    .set_index(job_id_source)
)

missing_resume_records = 0
missing_job_records = 0

for resume_id in step10[resume_col].unique():
    if resume_id not in resume_lookup.index:
        missing_resume_records += 1

for job_id in step10[job_col].unique():
    if job_id not in job_lookup.index:
        missing_job_records += 1

info(f"Missing resume records: {missing_resume_records}")
info(f"Missing job records: {missing_job_records}")

if missing_resume_records:
    warnings.warn(
        "Some Step 10 resume IDs are missing from the resume source."
    )

if missing_job_records:
    warnings.warn(
        "Some Step 10 job IDs are missing from the job source."
    )


# ======================================================================
# PREPARE SKILL LOOKUPS
# ======================================================================

section("Preparing skill evidence")

resume_skills_lookup: Dict[str, set] = {}
job_skills_lookup: Dict[str, set] = {}

for resume_id, row in resume_lookup.iterrows():

    skills = []

    if resume_skill_col is not None:
        skills = parse_skill_value(row.get(resume_skill_col))

    if not skills and resume_text_col is not None:
        skills = extract_skills_from_text(
            row.get(resume_text_col, "")
        )

    resume_skills_lookup[normalize_id(resume_id)] = set(skills)


for job_id, row in job_lookup.iterrows():

    skills = []

    if job_skill_col is not None:
        skills = parse_skill_value(row.get(job_skill_col))

    if not skills:

        title = (
            row.get(job_title_col, "")
            if job_title_col is not None
            else ""
        )

        description = (
            row.get(job_description_col, "")
            if job_description_col is not None
            else ""
        )

        skills = extract_skills_from_text(
            f"{title} {description}"
        )

    job_skills_lookup[normalize_id(job_id)] = set(skills)


# ======================================================================
# MERGE SOURCE INFORMATION
# ======================================================================

section("Joining actual job and resume information")

result = step10.copy()

result = result.rename(
    columns={
        resume_col: "resume_id",
        job_col: "job_id",
        score_col: "hybrid_match_score",
    }
)

result["resume_id"] = result["resume_id"].map(normalize_id)
result["job_id"] = result["job_id"].map(normalize_id)

# Remove possible stale audit-only columns from previous processing.
# We preserve them only if they already exist in Step 10.
# They are never used to generate the recommendation score.


def get_source_value(
    lookup: pd.DataFrame,
    key: str,
    column: Optional[str],
) -> str:

    if column is None:
        return ""

    if key not in lookup.index:
        return ""

    return clean_text(lookup.loc[key, column])


result["job_title"] = [
    get_source_value(job_lookup, jid, job_title_col)
    for jid in result["job_id"]
]

result["company"] = [
    get_source_value(job_lookup, jid, job_company_col)
    for jid in result["job_id"]
]

result["location"] = [
    get_source_value(job_lookup, jid, job_location_col)
    for jid in result["job_id"]
]

result["job_description"] = [
    truncate_text(
        get_source_value(
            job_lookup,
            jid,
            job_description_col,
        ),
        MAX_DESCRIPTION_LENGTH,
    )
    for jid in result["job_id"]
]


# ======================================================================
# MATCHED SKILLS AND SKILL GAPS
# ======================================================================

section("Calculating transparent matched skills and skill gaps")


def calculate_skill_evidence(
    resume_id: str,
    job_id: str,
) -> Tuple[List[str], List[str]]:

    resume_skills = resume_skills_lookup.get(
        resume_id,
        set(),
    )

    job_skills = job_skills_lookup.get(
        job_id,
        set(),
    )

    matched = sorted(resume_skills.intersection(job_skills))
    missing = sorted(job_skills.difference(resume_skills))

    return matched, missing


matched_skills_values = []
missing_skills_values = []

for resume_id, job_id in zip(
    result["resume_id"],
    result["job_id"],
):

    matched, missing = calculate_skill_evidence(
        resume_id,
        job_id,
    )

    matched_skills_values.append(matched)
    missing_skills_values.append(missing)


result["matched_skills"] = [
    ", ".join(skills)
    for skills in matched_skills_values
]

result["skill_gaps"] = [
    ", ".join(skills)
    for skills in missing_skills_values
]

result["matched_skill_count"] = [
    len(skills)
    for skills in matched_skills_values
]

result["skill_gap_count"] = [
    len(skills)
    for skills in missing_skills_values
]


# ======================================================================
# SCORE PRESENTATION
# ======================================================================

section("Preparing user-facing match scores")

# Step 10 produces a continuous matching score.
# This is NOT a probability.
#
# We therefore transform the score into a presentation scale while
# explicitly calling it a Match Score.

raw_scores = pd.to_numeric(
    result["hybrid_match_score"],
    errors="coerce",
)

if raw_scores.isna().any():
    raise ValueError(
        "Invalid hybrid scores found while preparing recommendations."
    )

score_min = float(raw_scores.min())
score_max = float(raw_scores.max())

if score_max > score_min:

    result["match_score"] = (
        (raw_scores - score_min)
        / (score_max - score_min)
        * SCORE_DISPLAY_SCALE
    )

else:
    result["match_score"] = 0.0

result["match_score"] = result["match_score"].clip(
    lower=0.0,
    upper=100.0,
)

result["match_score"] = result["match_score"].round(2)

result["score_type"] = "Match Score"


# ======================================================================
# TRANSPARENT MATCH EXPLANATION
# ======================================================================

section("Generating transparent recommendation explanations")


def numeric_signal(row: pd.Series, column: str) -> Optional[float]:

    if column not in row.index:
        return None

    value = pd.to_numeric(
        row[column],
        errors="coerce",
    )

    if pd.isna(value):
        return None

    return float(value)


def build_explanation(row: pd.Series) -> str:

    parts = []

    semantic = numeric_signal(
        row,
        columns.get("semantic_similarity", ""),
    )

    title_similarity = numeric_signal(
        row,
        columns.get("title_similarity", ""),
    )

    skill_signal = numeric_signal(
        row,
        columns.get("skill_match_signal", ""),
    )

    category_signal = numeric_signal(
        row,
        columns.get("category_match_signal", ""),
    )

    matched = clean_text(row.get("matched_skills", ""))
    gaps = clean_text(row.get("skill_gaps", ""))

    if semantic is not None:
        parts.append(
            f"semantic similarity={semantic:.3f}"
        )

    if title_similarity is not None:
        parts.append(
            f"title similarity={title_similarity:.3f}"
        )

    if skill_signal is not None:
        parts.append(
            f"skill signal={skill_signal:.3f}"
        )

    if category_signal is not None:
        parts.append(
            f"category signal={category_signal:.3f}"
        )

    evidence = (
        "Matching evidence: "
        + ", ".join(parts)
        if parts
        else "Matching evidence is based on the validated Step 10 hybrid score."
    )

    if matched:
        evidence += (
            f" Matched skills: {matched}."
        )

    if gaps:
        evidence += (
            f" Potential skill gaps: {gaps}."
        )

    return evidence


result["match_explanation"] = [
    build_explanation(row)
    for _, row in result.iterrows()
]


# ======================================================================
# RESUME-WISE RANKING
# ======================================================================

section("Generating resume-wise job rankings")

result = result.sort_values(
    by=[
        "resume_id",
        "hybrid_match_score",
        "job_id",
    ],
    ascending=[
        True,
        False,
        True,
    ],
    kind="mergesort",
).reset_index(drop=True)

result["recommendation_rank"] = (
    result
    .groupby("resume_id", sort=False)
    .cumcount()
    + 1
)

result["is_top_10"] = (
    result["recommendation_rank"] <= TOP_K
)


# ======================================================================
# TOP-K VALIDATION
# ======================================================================

section("Validating final recommendation rankings")

resume_counts = result.groupby("resume_id").size()

info(
    f"Unique resumes ranked: {result['resume_id'].nunique():,}"
)

info(
    f"Minimum jobs/resume: {resume_counts.min():,}"
)

info(
    f"Maximum jobs/resume: {resume_counts.max():,}"
)

info(
    f"Mean jobs/resume: {resume_counts.mean():.2f}"
)

if result["recommendation_rank"].isna().any():
    raise ValueError(
        "Recommendation ranks contain missing values."
    )

if result["recommendation_rank"].min() != 1:
    raise ValueError(
        "Recommendation ranking does not start at rank 1."
    )


# ======================================================================
# TOP-10 DATASET
# ======================================================================

top10 = result[
    result["recommendation_rank"] <= TOP_K
].copy()

if top10.empty:
    raise ValueError(
        "Top-10 recommendation dataset is empty."
    )

top10_count = top10["resume_id"].nunique()

info(
    f"Resumes with recommendations: {top10_count:,}"
)

info(
    f"Top-{TOP_K} recommendation rows: {len(top10):,}"
)


# ======================================================================
# FINAL OUTPUT COLUMN ORDER
# ======================================================================

preferred_columns = [
    "resume_id",
    "job_id",
    "recommendation_rank",
    "is_top_10",
    "match_score",
    "hybrid_match_score",
    "score_type",
    "job_title",
    "company",
    "location",
    "matched_skills",
    "matched_skill_count",
    "skill_gaps",
    "skill_gap_count",
    "match_explanation",
    "job_description",
]

# Include available Step 10 diagnostic signals after the main
# recommendation fields.
diagnostic_columns = [
    "semantic_similarity",
    "title_similarity",
    "skill_match_signal",
    "category_match_signal",
    "job_skill_coverage",
    "resume_skill_coverage",
    "jaccard_similarity",
    "text_containment",
    "maximum_skill_overlap_ratio",
    "shared_skill_count",
]

final_columns = []

for column in preferred_columns + diagnostic_columns:

    if column in result.columns and column not in final_columns:
        final_columns.append(column)

final_output = result[final_columns].copy()

top10_output = top10[
    [
        column
        for column in final_columns
        if column in top10.columns
    ]
].copy()


# ======================================================================
# FINAL VALIDATION
# ======================================================================

section("Final recommendation validation")

if len(final_output) != len(step10):
    raise ValueError(
        "Final recommendation row count does not match Step 10."
    )

if final_output[
    ["resume_id", "job_id"]
].duplicated().any():
    raise ValueError(
        "Duplicate resume-job pairs found in final output."
    )

if final_output["match_score"].isna().any():
    raise ValueError(
        "Missing user-facing match scores."
    )

if (
    (final_output["match_score"] < 0)
    | (final_output["match_score"] > 100)
).any():
    raise ValueError(
        "Match scores outside the 0-100 presentation range."
    )

if final_output["recommendation_rank"].isna().any():
    raise ValueError(
        "Missing recommendation ranks."
    )

if not final_output["is_top_10"].equals(
    final_output["recommendation_rank"] <= TOP_K
):
    raise ValueError(
        "Top-10 flag is inconsistent with recommendation rank."
    )

info("Row count validation: PASSED")
info("Duplicate pair validation: PASSED")
info("Match score validation: PASSED")
info("Ranking validation: PASSED")
info("Top-10 flag validation: PASSED")


# ======================================================================
# SAVE OUTPUTS
# ======================================================================

section("Saving final recommendation dataset")

final_output.to_csv(
    FINAL_OUTPUT_PATH,
    index=False,
)

info(f"Saved: {FINAL_OUTPUT_PATH}")
info(f"Output shape: {final_output.shape}")


section("Saving Top-10 recommendation dataset")

top10_output.to_csv(
    TOP10_OUTPUT_PATH,
    index=False,
)

info(f"Saved: {TOP10_OUTPUT_PATH}")
info(f"Output shape: {top10_output.shape}")


# ======================================================================
# REPORT
# ======================================================================

section("Building Step 12 recommendation report")

top_score = float(
    final_output["hybrid_match_score"].max()
)

mean_score = float(
    final_output["hybrid_match_score"].mean()
)

median_score = float(
    final_output["hybrid_match_score"].median()
)

top10_mean_score = float(
    top10_output["hybrid_match_score"].mean()
)

report = {
    "step": 12,
    "name": "Final Recommendation Pipeline",
    "status": "COMPLETE",

    "input": {
        "step10_predictions": str(STEP10_PATH),
        "rows": int(len(step10)),
        "unique_resumes": int(
            step10["resume_id"].nunique()
        ),
        "unique_jobs": int(
            step10["job_id"].nunique()
        ),
    },

    "recommendation_method": {
        "ranking_signal": "Step 10 hybrid_match_score",
        "model_retrained": False,
        "new_model_trained": False,
        "pseudo_labels_created": False,
        "ground_truth_hiring_labels": False,
        "hiring_probability_claimed": False,
        "score_interpretation": (
            "continuous match score, not probability of hiring"
        ),
    },

    "score_presentation": {
        "display_name": "Match Score",
        "display_scale": "0-100",
        "normalization": "min-max presentation scaling",
        "raw_score_preserved": True,
    },

    "recommendation": {
        "top_k": TOP_K,
        "minimum_recommendations": MIN_RECOMMENDATIONS,
        "resume_wise_ranking": True,
        "ranking_direction": "descending hybrid match score",
    },

    "explainability": {
        "actual_job_title": True,
        "actual_job_description": True,
        "matched_skills": True,
        "skill_gaps": True,
        "semantic_similarity": (
            "preserved from Step 10"
        ),
        "title_similarity": (
            "preserved from Step 10"
        ),
        "skill_match_signal": (
            "preserved from Step 10"
        ),
        "category_match_signal": (
            "preserved from Step 10"
        ),
    },

    "statistics": {
        "mean_raw_hybrid_score": mean_score,
        "median_raw_hybrid_score": median_score,
        "maximum_raw_hybrid_score": top_score,
        "mean_top10_raw_score": top10_mean_score,
        "top10_rows": int(len(top10_output)),
    },

    "validation": {
        "row_count_match": True,
        "duplicate_pairs": False,
        "missing_match_scores": False,
        "ranking_validated": True,
        "top10_flag_validated": True,
    },

    "outputs": {
        "all_recommendations": str(FINAL_OUTPUT_PATH),
        "top10_recommendations": str(TOP10_OUTPUT_PATH),
        "report": str(REPORT_PATH),
    },

    "next_step": (
        "Step 13 - explainable LLM/RAG recommendation layer"
    ),
}

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        report,
        file,
        indent=2,
        ensure_ascii=False,
    )

info(f"Report saved: {REPORT_PATH}")


# ======================================================================
# FINAL SUMMARY
# ======================================================================

section("STEP 12 COMPLETE")

info(
    f"Total resume-job pairs: {len(final_output):,}"
)

info(
    f"Unique resumes: {final_output['resume_id'].nunique():,}"
)

info(
    f"Unique jobs: {final_output['job_id'].nunique():,}"
)

info(
    f"Top-{TOP_K} recommendation rows: {len(top10_output):,}"
)

info(
    f"Mean raw hybrid score: {mean_score:.4f}"
)

info(
    f"Median raw hybrid score: {median_score:.4f}"
)

info(
    f"Mean Top-{TOP_K} raw hybrid score: {top10_mean_score:.4f}"
)

info("Ranking signal: Step 10 hybrid_match_score")
info("Model retrained: NO")
info("Pseudo-labels: NO")
info("Ground-truth hiring labels: NO")
info("Hiring probability claimed: NO")
info("Matched skills: YES")
info("Skill gaps: YES")
info("Actual job information: YES")

info(f"All recommendations: {FINAL_OUTPUT_PATH}")
info(f"Top-{TOP_K} recommendations: {TOP10_OUTPUT_PATH}")
info(f"Report: {REPORT_PATH}")

info("")
info("NEXT: STEP 13 - EXPLAINABLE LLM/RAG RECOMMENDATION LAYER")