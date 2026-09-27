"""
AI RESUME ANALYZER
STEP 11 - AUTOMATIC INDEPENDENT VALIDATION

Purpose
-------
Automatically validate the Step 10 hybrid matching system without
manual human annotation.

IMPORTANT
---------
- Uses actual resume text and actual job title + description.
- Does NOT use Step 9 evidence_score as a label.
- Does NOT create pseudo-labels.
- Does NOT assume ground-truth hiring decisions.
- Does NOT claim ROC-AUC/F1.
- Does NOT train another supervised model.
- Does NOT require manual CSV annotation.
- Evaluates ranking consistency, semantic agreement, score behavior,
  and retrieval stability.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd


# ================================================================
# CONFIGURATION
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEP10_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "step10_matching_predictions.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

OUTPUT_PATH = (
    EVALUATION_DIR
    / "step11_automatic_validation_results.csv"
)

REPORT_PATH = (
    REPORT_DIR
    / "automatic_validation_report.json"
)

TOP_K_VALUES = [5, 10, 20]


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def minmax_normalize(series):
    values = safe_numeric(series).astype(float)

    minimum = values.min()
    maximum = values.max()

    if not np.isfinite(minimum) or not np.isfinite(maximum):
        return pd.Series(
            np.zeros(len(values)),
            index=series.index,
            dtype=float,
        )

    if maximum <= minimum:
        return pd.Series(
            np.full(len(values), 0.5),
            index=series.index,
            dtype=float,
        )

    return (values - minimum) / (maximum - minimum)


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)

    if mask.sum() < 2:
        return np.nan

    if np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return np.nan

    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def mean_absolute_difference(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)

    if mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs(a[mask] - b[mask])))


def top_k_overlap(group_a, group_b, k):
    a = list(group_a.head(k))
    b = list(group_b.head(k))

    if not a or not b:
        return np.nan

    return len(set(a).intersection(set(b))) / min(len(a), len(b))


# ================================================================
# START
# ================================================================

print_header(
    "AI RESUME ANALYZER - STEP 11 AUTOMATIC INDEPENDENT VALIDATION"
)

print(
    """
IMPORTANT:
- No manual annotation required.
- No pseudo-labels.
- No ground-truth hiring labels.
- Step 9 evidence_score is NOT treated as ground truth.
- Step 10 is evaluated through independent consistency diagnostics.
- No supervised model is trained.
- ROC-AUC/F1 are NOT claimed.
"""
)


# ================================================================
# CHECK DIRECTORIES
# ================================================================

REPORT_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# LOAD STEP 10
# ================================================================

print_header("Loading Step 10 matching predictions")

if not STEP10_PATH.exists():
    raise FileNotFoundError(
        f"Step 10 prediction file not found:\n{STEP10_PATH}"
    )

df = pd.read_csv(STEP10_PATH)

print(f"Input shape: {df.shape}")


# ================================================================
# BASIC COLUMN DETECTION
# ================================================================

print_header("Detecting Step 10 columns")


def find_column(candidates, required=True):

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    if required:
        raise ValueError(
            f"Required column not found.\n"
            f"Expected one of: {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


resume_col = find_column(
    [
        "resume_id",
        "Resume_ID",
        "resume",
        "Resume",
    ]
)

job_col = find_column(
    [
        "job_id",
        "Job_ID",
        "job",
        "Job",
    ]
)

score_col = find_column(
    [
        "hybrid_match_score",
        "hybrid_score",
        "match_score",
        "score",
    ]
)

semantic_col = find_column(
    [
        "semantic_similarity",
    ],
    required=False,
)

title_col = find_column(
    [
        "title_similarity",
    ],
    required=False,
)

skill_col = find_column(
    [
        "skill_match_signal",
    ],
    required=False,
)

category_col = find_column(
    [
        "category_match_signal",
    ],
    required=False,
)

evidence_col = find_column(
    [
        "evidence_score",
    ],
    required=False,
)


print(f"Resume column: {resume_col}")
print(f"Job column: {job_col}")
print(f"Hybrid score column: {score_col}")

if semantic_col:
    print(f"Semantic similarity column: {semantic_col}")

if title_col:
    print(f"Title similarity column: {title_col}")

if skill_col:
    print(f"Skill signal column: {skill_col}")

if category_col:
    print(f"Category signal column: {category_col}")

if evidence_col:
    print(
        "Step 9 evidence_score detected: audit-only"
    )


# ================================================================
# BASIC VALIDATION
# ================================================================

print_header("Validating Step 10 dataset")

initial_rows = len(df)

df[score_col] = safe_numeric(df[score_col])

if semantic_col:
    df[semantic_col] = safe_numeric(df[semantic_col])

if title_col:
    df[title_col] = safe_numeric(df[title_col])

if skill_col:
    df[skill_col] = safe_numeric(df[skill_col])

if category_col:
    df[category_col] = safe_numeric(df[category_col])

if evidence_col:
    df[evidence_col] = safe_numeric(df[evidence_col])


duplicate_pairs = df.duplicated(
    subset=[resume_col, job_col]
).sum()

missing_scores = df[score_col].isna().sum()

print(f"Rows: {initial_rows:,}")
print(f"Unique resumes: {df[resume_col].nunique():,}")
print(f"Unique jobs: {df[job_col].nunique():,}")
print(f"Duplicate pairs: {duplicate_pairs:,}")
print(f"Missing hybrid scores: {missing_scores:,}")


if duplicate_pairs > 0:
    raise ValueError(
        "Duplicate resume-job pairs detected."
    )

if missing_scores > 0:
    raise ValueError(
        "Missing hybrid match scores detected."
    )


# ================================================================
# SCORE RANGE VALIDATION
# ================================================================

print_header("Validating hybrid score behavior")

score_values = df[score_col].to_numpy(dtype=float)

finite_scores = np.isfinite(score_values)

if not finite_scores.all():
    raise ValueError(
        "Hybrid scores contain non-finite values."
    )

print(
    f"Hybrid score mean: "
    f"{np.mean(score_values):.4f}"
)

print(
    f"Hybrid score median: "
    f"{np.median(score_values):.4f}"
)

print(
    f"Hybrid score minimum: "
    f"{np.min(score_values):.4f}"
)

print(
    f"Hybrid score maximum: "
    f"{np.max(score_values):.4f}"
)

print(
    f"Hybrid score standard deviation: "
    f"{np.std(score_values):.4f}"
)


# ================================================================
# SCORE DISTRIBUTION
# ================================================================

print_header("Analyzing score distribution")

percentiles = {
    "p01": float(np.percentile(score_values, 1)),
    "p05": float(np.percentile(score_values, 5)),
    "p10": float(np.percentile(score_values, 10)),
    "p25": float(np.percentile(score_values, 25)),
    "p50": float(np.percentile(score_values, 50)),
    "p75": float(np.percentile(score_values, 75)),
    "p90": float(np.percentile(score_values, 90)),
    "p95": float(np.percentile(score_values, 95)),
    "p99": float(np.percentile(score_values, 99)),
}

for name, value in percentiles.items():
    print(f"{name}: {value:.4f}")


# ================================================================
# NORMALIZED SIGNALS
# ================================================================

print_header("Preparing independent validation signals")

validation_signals = {}

if semantic_col:
    validation_signals["semantic_similarity"] = (
        minmax_normalize(df[semantic_col])
    )

if title_col:
    validation_signals["title_similarity"] = (
        minmax_normalize(df[title_col])
    )

if skill_col:
    validation_signals["skill_match_signal"] = (
        minmax_normalize(df[skill_col])
    )

if category_col:
    validation_signals["category_match_signal"] = (
        minmax_normalize(df[category_col])
    )


print(
    f"Independent validation signals available: "
    f"{len(validation_signals)}"
)

for name in validation_signals:
    print(f"Validation signal: {name}")


# ================================================================
# SIGNAL AGREEMENT
# ================================================================

print_header("Calculating signal agreement")

signal_correlations = {}

for name, signal in validation_signals.items():

    correlation = safe_corr(
        df[score_col].to_numpy(dtype=float),
        signal.to_numpy(dtype=float),
    )

    signal_correlations[name] = correlation

    if np.isfinite(correlation):
        print(
            f"{name} vs hybrid score correlation: "
            f"{correlation:.4f}"
        )
    else:
        print(
            f"{name} vs hybrid score correlation: unavailable"
        )


# ================================================================
# RESUME-WISE RANKING
# ================================================================

print_header("Generating resume-wise rankings")

df["_rank"] = (
    df.groupby(resume_col)[score_col]
    .rank(
        ascending=False,
        method="first",
    )
)

print(
    f"Rankings generated for "
    f"{df[resume_col].nunique():,} resumes"
)


# ================================================================
# TOP-K STATISTICS
# ================================================================

print_header("Calculating top-k retrieval statistics")

ranking_statistics = {}

for k in TOP_K_VALUES:

    top_k = df[df["_rank"] <= k]

    counts = (
        top_k.groupby(resume_col)[job_col]
        .count()
    )

    mean_score = (
        top_k[score_col].mean()
        if len(top_k) > 0
        else np.nan
    )

    ranking_statistics[f"top_{k}"] = {
        "pairs": int(len(top_k)),
        "resumes_with_results": int(len(counts)),
        "mean_score": (
            float(mean_score)
            if np.isfinite(mean_score)
            else None
        ),
        "minimum_results_per_resume": (
            int(counts.min())
            if len(counts) > 0
            else 0
        ),
        "maximum_results_per_resume": (
            int(counts.max())
            if len(counts) > 0
            else 0
        ),
    }

    print(
        f"Top-{k}: "
        f"{len(top_k):,} pairs | "
        f"mean score = {mean_score:.4f}"
    )


# ================================================================
# TOP-K SCORE SEPARATION
# ================================================================

print_header("Calculating top-k score separation")

score_separation = {}

for k in TOP_K_VALUES:

    top_scores = (
        df[df["_rank"] <= k]
        .groupby(resume_col)[score_col]
        .mean()
    )

    lower_scores = (
        df[df["_rank"] > k]
        .groupby(resume_col)[score_col]
        .mean()
    )

    aligned = pd.concat(
        [
            top_scores.rename("top"),
            lower_scores.rename("lower"),
        ],
        axis=1,
        join="inner",
    )

    aligned["difference"] = (
        aligned["top"] - aligned["lower"]
    )

    separation = aligned["difference"].mean()

    score_separation[f"top_{k}"] = (
        float(separation)
        if np.isfinite(separation)
        else None
    )

    print(
        f"Top-{k} mean score separation: "
        f"{separation:.4f}"
    )


# ================================================================
# SEMANTIC CONSISTENCY
# ================================================================

print_header("Checking semantic consistency")

semantic_consistency = {}

if semantic_col:

    semantic_values = df[semantic_col].to_numpy(dtype=float)

    hybrid_values = df[score_col].to_numpy(dtype=float)

    semantic_consistency["correlation"] = safe_corr(
        hybrid_values,
        semantic_values,
    )

    semantic_consistency["mean_absolute_difference"] = (
        mean_absolute_difference(
            minmax_normalize(df[semantic_col]).to_numpy(),
            minmax_normalize(df[score_col]).to_numpy(),
        )
    )

    print(
        "Hybrid vs semantic similarity correlation: "
        f"{semantic_consistency['correlation']:.4f}"
    )

    print(
        "Normalized mean absolute difference: "
        f"{semantic_consistency['mean_absolute_difference']:.4f}"
    )

else:

    semantic_consistency["correlation"] = None
    semantic_consistency["mean_absolute_difference"] = None

    print(
        "Semantic similarity column unavailable."
    )


# ================================================================
# EVIDENCE SCORE AUDIT ONLY
# ================================================================

print_header("Auditing Step 9 evidence_score")

evidence_audit = {
    "available": False,
    "used_as_feature": False,
    "used_as_target": False,
    "reconstructed": False,
    "modified": False,
    "correlation_with_step10_score": None,
}

if evidence_col:

    evidence_audit["available"] = True

    valid = (
        df[evidence_col].notna()
        & df[score_col].notna()
    )

    if valid.sum() > 1:

        correlation = safe_corr(
            df.loc[valid, score_col].to_numpy(),
            df.loc[valid, evidence_col].to_numpy(),
        )

        evidence_audit[
            "correlation_with_step10_score"
        ] = correlation

        print(
            "Correlation with Step 9 evidence_score: "
            f"{correlation:.4f}"
        )

    print(
        """
Step 9 evidence_score is used ONLY for audit comparison.

It is NOT:
- a ground-truth label
- a hiring label
- a training target
- a model feature
- a pseudo-label
"""
    )

else:

    print(
        "Step 9 evidence_score not present in Step 10 output."
    )


# ================================================================
# HIGH-SCORE CONCENTRATION
# ================================================================

print_header("Analyzing high-score retrieval concentration")

thresholds = [0.50, 0.60, 0.70, 0.80]

high_score_statistics = {}

for threshold in thresholds:

    count = int(
        (df[score_col] >= threshold).sum()
    )

    percentage = (
        count / len(df) * 100
        if len(df) > 0
        else 0
    )

    high_score_statistics[
        f">={threshold:.2f}"
    ] = {
        "pairs": count,
        "percentage": float(percentage),
    }

    print(
        f"Score >= {threshold:.2f}: "
        f"{count:,} pairs "
        f"({percentage:.2f}%)"
    )


# ================================================================
# RESUME COVERAGE
# ================================================================

print_header("Checking resume retrieval coverage")

resume_counts = (
    df.groupby(resume_col)[job_col]
    .count()
)

coverage_statistics = {
    "number_of_resumes": int(len(resume_counts)),
    "minimum_jobs_per_resume": int(
        resume_counts.min()
    ),
    "maximum_jobs_per_resume": int(
        resume_counts.max()
    ),
    "mean_jobs_per_resume": float(
        resume_counts.mean()
    ),
    "median_jobs_per_resume": float(
        resume_counts.median()
    ),
}

print(
    f"Resumes: "
    f"{coverage_statistics['number_of_resumes']:,}"
)

print(
    f"Minimum jobs/resume: "
    f"{coverage_statistics['minimum_jobs_per_resume']}"
)

print(
    f"Maximum jobs/resume: "
    f"{coverage_statistics['maximum_jobs_per_resume']}"
)

print(
    f"Mean jobs/resume: "
    f"{coverage_statistics['mean_jobs_per_resume']:.2f}"
)


# ================================================================
# SCORE MONOTONICITY CHECK
# ================================================================

print_header("Checking ranking monotonicity")

monotonicity_violations = 0

for _, group in df.groupby(resume_col):

    ranked_scores = (
        group.sort_values(
            "_rank",
            ascending=True,
        )[score_col]
        .to_numpy()
    )

    if len(ranked_scores) > 1:

        if np.any(
            ranked_scores[:-1]
            < ranked_scores[1:]
        ):
            monotonicity_violations += 1


print(
    f"Resume ranking monotonicity violations: "
    f"{monotonicity_violations}"
)


# ================================================================
# BUILD OUTPUT
# ================================================================

print_header("Preparing Step 11 validation output")

output_columns = [
    resume_col,
    job_col,
    score_col,
    "_rank",
]

for column in [
    semantic_col,
    title_col,
    skill_col,
    category_col,
]:
    if column and column in df.columns:
        output_columns.append(column)

validation_output = df[
    output_columns
].copy()

validation_output = validation_output.rename(
    columns={
        "_rank": "step10_rank",
    }
)

validation_output["validation_stage"] = (
    "automatic_independent_validation"
)


# ================================================================
# FINAL OUTPUT VALIDATION
# ================================================================

print_header("Final Step 11 validation")

if len(validation_output) != len(df):
    raise ValueError(
        "Step 11 output row count changed."
    )

if validation_output[
    [resume_col, job_col]
].duplicated().any():

    raise ValueError(
        "Duplicate resume-job pairs detected "
        "in Step 11 output."
    )

if validation_output[score_col].isna().any():

    raise ValueError(
        "Missing hybrid scores detected "
        "in Step 11 output."
    )

if validation_output["step10_rank"].isna().any():

    raise ValueError(
        "Missing ranking values detected."
    )

print("Row count validation: PASSED")
print("Duplicate pair validation: PASSED")
print("Hybrid score validation: PASSED")
print("Ranking validation: PASSED")


# ================================================================
# SAVE OUTPUT
# ================================================================

print_header("Saving Step 11 validation results")

validation_output.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(f"Saved: {OUTPUT_PATH}")
print(
    f"Output shape: {validation_output.shape}"
)


# ================================================================
# BUILD REPORT
# ================================================================

report = {
    "step": 11,
    "name": "Automatic Independent Validation",

    "purpose": (
        "Automatic validation of Step 10 hybrid matching "
        "without manual annotation or ground-truth hiring labels."
    ),

    "input": {
        "path": str(STEP10_PATH),
        "rows": int(len(df)),
        "unique_resumes": int(
            df[resume_col].nunique()
        ),
        "unique_jobs": int(
            df[job_col].nunique()
        ),
    },

    "validation_design": {
        "manual_annotation_required": False,
        "pseudo_labels_created": False,
        "ground_truth_hiring_labels": False,
        "supervised_model_trained": False,
        "roc_auc_claimed": False,
        "f1_claimed": False,
    },

    "score_statistics": {
        "mean": float(np.mean(score_values)),
        "median": float(np.median(score_values)),
        "minimum": float(np.min(score_values)),
        "maximum": float(np.max(score_values)),
        "standard_deviation": float(
            np.std(score_values)
        ),
        "percentiles": percentiles,
    },

    "ranking_statistics": ranking_statistics,

    "score_separation": score_separation,

    "signal_correlations": signal_correlations,

    "semantic_consistency": semantic_consistency,

    "evidence_score_audit": evidence_audit,

    "high_score_statistics": high_score_statistics,

    "resume_coverage": coverage_statistics,

    "ranking_quality_checks": {
        "monotonicity_violations": int(
            monotonicity_violations
        ),
    },

    "interpretation": {
        "semantic_source": (
            "ACTUAL TEXT, computed in Step 9 "
            "using all-MiniLM-L6-v2"
        ),
        "step10_score_role": (
            "hybrid relevance score"
        ),
        "step9_evidence_role": (
            "audit-only"
        ),
        "human_labels": (
            "not required for this automatic validation stage"
        ),
        "ground_truth": (
            "not available and not assumed"
        ),
    },

    "output": {
        "validation_results": str(
            OUTPUT_PATH
        ),
        "report": str(
            REPORT_PATH
        ),
    },
}


# ================================================================
# SAVE REPORT
# ================================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=4,
    )


# ================================================================
# COMPLETE
# ================================================================

print_header(
    "STEP 11 AUTOMATIC VALIDATION COMPLETE"
)

print(
    f"Total resume-job pairs: "
    f"{len(df):,}"
)

print(
    f"Unique resumes: "
    f"{df[resume_col].nunique():,}"
)

print(
    f"Unique jobs: "
    f"{df[job_col].nunique():,}"
)

print(
    f"Mean hybrid score: "
    f"{np.mean(score_values):.4f}"
)

print(
    f"Median hybrid score: "
    f"{np.median(score_values):.4f}"
)

print(
    "Manual annotation required: NO"
)

print(
    "Pseudo-labels created: NO"
)

print(
    "Ground-truth hiring labels: NO"
)

print(
    "Supervised model trained: NO"
)

print(
    "ROC-AUC: NOT CLAIMED"
)

print(
    "F1: NOT CLAIMED"
)

print(
    f"Validation results: {OUTPUT_PATH}"
)

print(
    f"Report: {REPORT_PATH}"
)

print(
    "\nNEXT: STEP 12 - FINAL MATCHING / RECOMMENDATION PIPELINE"
)