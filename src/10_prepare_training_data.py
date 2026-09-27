"""
======================================================================
AI RESUME ANALYZER - STEP 10 FINAL HYBRID MATCHING MODEL
======================================================================

PURPOSE
-------
Generate a transparent hybrid resume-job matching score and
resume-wise job rankings from the corrected Step 9 dataset.

IMPORTANT
---------
- Uses corrected Step 9 REAL semantic similarity.
- Step 9 semantic similarity was generated from ACTUAL resume text.
- Step 9 semantic similarity was generated from ACTUAL job text.
- Raw semantic text columns are NOT required in Step 10.
- Step 9 evidence_score is preserved ONLY for audit.
- Step 9 evidence_score is NOT a model feature.
- Step 9 evidence_score is NOT a training target.
- No pseudo-labels.
- No ground-truth hiring labels.
- No ROC-AUC/F1.
- No artificial supervised target.
- No supervised model is trained.
- Final score is a transparent weighted hybrid score.
- Resume-wise ranking is deterministic and reproducible.
- Independent human validation follows Step 10.

======================================================================
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path(
    r"C:\Users\jasme\Desktop\ai resume analyzer"
)

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "resume_job_matching_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "step10_matching_predictions.csv"
)

REPORT_PATH = (
    REPORT_DIR
    / "hybrid_matching_model_report.json"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================================
# CONFIGURATION
# =====================================================================

TOP_K_VALUES = [5, 10, 20]

#
# Explicit hybrid matching weights.
#
# These are NOT learned weights.
# They are transparent design parameters.
#
WEIGHTS = {
    "semantic_similarity": 0.30,
    "skill_match_signal": 0.20,
    "title_similarity": 0.15,
    "category_match_signal": 0.10,
    "job_skill_coverage": 0.10,
    "resume_skill_coverage": 0.05,
    "jaccard_similarity": 0.04,
    "text_containment": 0.03,
    "maximum_skill_overlap_ratio": 0.02,
    "shared_skill_count_signal": 0.01,
}

WEIGHT_SUM = sum(
    WEIGHTS.values()
)

if not np.isclose(
    WEIGHT_SUM,
    1.0
):
    raise ValueError(
        "Hybrid weights must sum to 1.0. "
        f"Current sum = {WEIGHT_SUM}"
    )


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def require_columns(
    dataframe,
    columns,
    section_name
):
    missing = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{section_name}: missing required columns: {missing}"
        )


def numeric_column(
    dataframe,
    column
):
    values = pd.to_numeric(
        dataframe[column],
        errors="coerce"
    )

    if values.isna().any():
        count = int(
            values.isna().sum()
        )

        raise ValueError(
            f"Column '{column}' contains "
            f"{count} missing/non-numeric values."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"Column '{column}' contains non-finite values."
        )

    return values.astype(float)


def clip01(series):
    return (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .clip(
            lower=0.0,
            upper=1.0
        )
    )


# =====================================================================
# START
# =====================================================================

print_header(
    "AI RESUME ANALYZER - STEP 10 FINAL HYBRID MATCHING MODEL"
)

print("""
IMPORTANT:
- Uses corrected Step 9 REAL semantic similarity.
- Semantic similarity was computed from actual text in Step 9.
- Raw semantic text columns are NOT required here.
- Step 9 evidence_score is audit-only.
- No pseudo-labels.
- No ground-truth hiring labels.
- No supervised training.
- No ROC-AUC/F1.
""")


# =====================================================================
# LOAD STEP 9 DATASET
# =====================================================================

print_header(
    "Loading corrected Step 9 semantic matching dataset"
)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Step 9 dataset not found:\n{INPUT_PATH}"
    )

df = pd.read_csv(
    INPUT_PATH
)

print(
    f"Input shape: {df.shape}"
)


# =====================================================================
# VALIDATE STEP 9 STRUCTURE
# =====================================================================

print_header(
    "Validating Step 9 dataset"
)

required_columns = [
    "resume_id",
    "job_id",

    "semantic_similarity",
    "evidence_score",

    "shared_skill_count",
    "resume_skill_count",
    "job_skill_count",
    "jaccard_similarity",
    "resume_skill_coverage",
    "job_skill_coverage",
    "skill_count_difference",
    "maximum_skill_overlap_ratio",
    "category_match_signal",
    "text_containment",
    "title_similarity",
    "skill_match_signal",
]

require_columns(
    df,
    required_columns,
    "Step 9 validation"
)

duplicate_pairs = int(
    df.duplicated(
        subset=[
            "resume_id",
            "job_id"
        ]
    ).sum()
)

if duplicate_pairs > 0:
    raise ValueError(
        f"Duplicate resume-job pairs detected: "
        f"{duplicate_pairs}"
    )

print(
    f"Rows: {len(df):,}"
)

print(
    f"Unique resumes: "
    f"{df['resume_id'].nunique():,}"
)

print(
    f"Unique jobs: "
    f"{df['job_id'].nunique():,}"
)

print(
    f"Duplicate resume-job pairs: "
    f"{duplicate_pairs}"
)


# =====================================================================
# VALIDATE REAL SEMANTIC SIMILARITY
# =====================================================================

print_header(
    "Validating REAL semantic similarity"
)

df["semantic_similarity"] = numeric_column(
    df,
    "semantic_similarity"
)

semantic_missing = int(
    df["semantic_similarity"]
    .isna()
    .sum()
)

if semantic_missing > 0:
    raise ValueError(
        f"Missing semantic similarity: "
        f"{semantic_missing}"
    )

print(
    f"Semantic similarity mean: "
    f"{df['semantic_similarity'].mean():.4f}"
)

print(
    f"Semantic similarity median: "
    f"{df['semantic_similarity'].median():.4f}"
)

print(
    f"Semantic similarity minimum: "
    f"{df['semantic_similarity'].min():.4f}"
)

print(
    f"Semantic similarity maximum: "
    f"{df['semantic_similarity'].max():.4f}"
)

print(
    f"Missing semantic similarity: "
    f"{semantic_missing}"
)


# =====================================================================
# IMPORTANT CORRECTION
# =====================================================================

print_header(
    "Validating semantic source"
)

print("""
Step 9 did not save the raw semantic text columns in the final CSV.

Therefore Step 10 does NOT require:
    resume_semantic_text
    job_semantic_text

This is intentional.

The authoritative semantic signal for Step 10 is:
    semantic_similarity

Step 9 already computed that similarity using:
    ACTUAL resume text
    ACTUAL job title + description

Step 10 therefore consumes the validated semantic similarity rather
than attempting to reload text that was not stored in the Step 9 CSV.
""")

print(
    "Semantic source: Step 9 ACTUAL TEXT"
)

print(
    "Raw semantic text required in Step 10: NO"
)

print(
    "Semantic similarity available: YES"
)


# =====================================================================
# AUDIT STEP 9 EVIDENCE SCORE
# =====================================================================

print_header(
    "Auditing Step 9 evidence_score"
)

df["evidence_score"] = numeric_column(
    df,
    "evidence_score"
)

evidence_missing = int(
    df["evidence_score"]
    .isna()
    .sum()
)

if evidence_missing > 0:
    raise ValueError(
        f"Missing Step 9 evidence_score: "
        f"{evidence_missing}"
    )

print(
    f"Evidence score mean: "
    f"{df['evidence_score'].mean():.4f}"
)

print(
    f"Evidence score median: "
    f"{df['evidence_score'].median():.4f}"
)

print(
    f"Evidence score minimum: "
    f"{df['evidence_score'].min():.4f}"
)

print(
    f"Evidence score maximum: "
    f"{df['evidence_score'].max():.4f}"
)

print("\nEvidence score role:")
print("Preserved: YES")
print("Used as hybrid feature: NO")
print("Used as training target: NO")
print("Reconstructed: NO")
print("Modified: NO")


# =====================================================================
# PREPARE MATCHING SIGNALS
# =====================================================================

print_header(
    "Preparing transparent matching signals"
)

signal_columns = [
    "semantic_similarity",
    "skill_match_signal",
    "title_similarity",
    "category_match_signal",
    "job_skill_coverage",
    "resume_skill_coverage",
    "jaccard_similarity",
    "text_containment",
    "maximum_skill_overlap_ratio",
    "shared_skill_count",
]

for column in signal_columns:

    df[column] = numeric_column(
        df,
        column
    )

    print(
        f"Validated signal: {column}"
    )


# =====================================================================
# NORMALIZE SIGNALS
# =====================================================================

print_header(
    "Normalizing matching signals"
)

#
# Semantic similarity is already the direct Sentence Transformer
# similarity signal generated in Step 9.
#
df["_semantic_signal"] = clip01(
    df["semantic_similarity"]
)

df["_skill_signal"] = clip01(
    df["skill_match_signal"]
)

df["_title_signal"] = clip01(
    df["title_similarity"]
)

df["_category_signal"] = clip01(
    df["category_match_signal"]
)

df["_job_coverage_signal"] = clip01(
    df["job_skill_coverage"]
)

df["_resume_coverage_signal"] = clip01(
    df["resume_skill_coverage"]
)

df["_jaccard_signal"] = clip01(
    df["jaccard_similarity"]
)

df["_containment_signal"] = clip01(
    df["text_containment"]
)

df["_overlap_signal"] = clip01(
    df["maximum_skill_overlap_ratio"]
)


# =====================================================================
# SHARED SKILL SIGNAL
# =====================================================================

shared_skill = (
    df["shared_skill_count"]
    .clip(lower=0)
)

maximum_shared_skill = max(
    float(shared_skill.max()),
    1.0
)

df["_shared_skill_signal"] = (
    np.log1p(shared_skill)
    / np.log1p(maximum_shared_skill)
)

df["_shared_skill_signal"] = (
    df["_shared_skill_signal"]
    .replace(
        [np.inf, -np.inf],
        0.0
    )
    .fillna(0.0)
    .clip(
        0.0,
        1.0
    )
)


# =====================================================================
# CHECK NORMALIZED SIGNALS
# =====================================================================

normalized_columns = [
    "_semantic_signal",
    "_skill_signal",
    "_title_signal",
    "_category_signal",
    "_job_coverage_signal",
    "_resume_coverage_signal",
    "_jaccard_signal",
    "_containment_signal",
    "_overlap_signal",
    "_shared_skill_signal",
]

missing_normalized = int(
    df[normalized_columns]
    .isna()
    .sum()
    .sum()
)

if missing_normalized > 0:
    raise ValueError(
        f"Missing normalized signals: "
        f"{missing_normalized}"
    )

print(
    f"Normalized signals: "
    f"{len(normalized_columns)}"
)

print(
    f"Missing normalized values: "
    f"{missing_normalized}"
)


# =====================================================================
# CALCULATE HYBRID SCORE
# =====================================================================

print_header(
    "Calculating transparent hybrid relevance score"
)

#
# This is intentionally NOT machine learning.
#
# No training data.
# No target.
# No pseudo-label.
# No evidence-score leakage.
#
df["hybrid_match_score"] = (

    WEIGHTS["semantic_similarity"]
    * df["_semantic_signal"]

    + WEIGHTS["skill_match_signal"]
    * df["_skill_signal"]

    + WEIGHTS["title_similarity"]
    * df["_title_signal"]

    + WEIGHTS["category_match_signal"]
    * df["_category_signal"]

    + WEIGHTS["job_skill_coverage"]
    * df["_job_coverage_signal"]

    + WEIGHTS["resume_skill_coverage"]
    * df["_resume_coverage_signal"]

    + WEIGHTS["jaccard_similarity"]
    * df["_jaccard_signal"]

    + WEIGHTS["text_containment"]
    * df["_containment_signal"]

    + WEIGHTS["maximum_skill_overlap_ratio"]
    * df["_overlap_signal"]

    + WEIGHTS["shared_skill_count_signal"]
    * df["_shared_skill_signal"]
)

df["hybrid_match_score"] = (
    df["hybrid_match_score"]
    .clip(
        0.0,
        1.0
    )
)

print(
    "Scoring method: "
    "explicit weighted hybrid aggregation"
)

print(
    "Supervised model: NONE"
)

print(
    "Training target: NONE"
)

print(
    "Pseudo-label target: NONE"
)

print(
    f"Hybrid score mean: "
    f"{df['hybrid_match_score'].mean():.4f}"
)

print(
    f"Hybrid score median: "
    f"{df['hybrid_match_score'].median():.4f}"
)

print(
    f"Hybrid score minimum: "
    f"{df['hybrid_match_score'].min():.4f}"
)

print(
    f"Hybrid score maximum: "
    f"{df['hybrid_match_score'].max():.4f}"
)


# =====================================================================
# RESUME-WISE JOB RANKING
# =====================================================================

print_header(
    "Generating resume-wise job rankings"
)

df = df.sort_values(
    by=[
        "resume_id",
        "hybrid_match_score",
        "semantic_similarity",
        "title_similarity",
        "skill_match_signal",
        "job_id",
    ],
    ascending=[
        True,
        False,
        False,
        False,
        False,
        True,
    ],
    kind="mergesort",
)

df["job_rank"] = (
    df.groupby(
        "resume_id",
        sort=False
    )
    .cumcount()
    + 1
)

df["top_5"] = (
    df["job_rank"] <= 5
)

df["top_10"] = (
    df["job_rank"] <= 10
)

df["top_20"] = (
    df["job_rank"] <= 20
)

print(
    "Resume-wise rankings generated."
)


# =====================================================================
# VALIDATE RANKINGS
# =====================================================================

print_header(
    "Validating resume-wise rankings"
)

rank_validation = (
    df.groupby(
        "resume_id"
    )["job_rank"]
    .agg(
        [
            "min",
            "max",
            "count"
        ]
    )
)

if not (
    rank_validation["min"] == 1
).all():

    raise ValueError(
        "Some resume rankings do not start at rank 1."
    )

if not (
    rank_validation["max"]
    == rank_validation["count"]
).all():

    raise ValueError(
        "Ranking sequence contains gaps."
    )

print(
    f"Validated resume rankings: "
    f"{len(rank_validation):,}"
)

print(
    f"Minimum jobs per resume: "
    f"{rank_validation['count'].min():,}"
)

print(
    f"Maximum jobs per resume: "
    f"{rank_validation['count'].max():,}"
)


# =====================================================================
# TOP-K STATISTICS
# =====================================================================

print_header(
    "Calculating top-k ranking statistics"
)

top_k_statistics = {}

for k in TOP_K_VALUES:

    top_k = df[
        df["job_rank"] <= k
    ]

    top_k_statistics[str(k)] = {
        "rows": int(
            len(top_k)
        ),
        "resume_count": int(
            top_k["resume_id"].nunique()
        ),
        "mean_score": float(
            top_k["hybrid_match_score"].mean()
        ),
        "median_score": float(
            top_k["hybrid_match_score"].median()
        ),
    }

    print(
        f"Top-{k}: "
        f"{len(top_k):,} pairs | "
        f"mean score = "
        f"{top_k['hybrid_match_score'].mean():.4f}"
    )


# =====================================================================
# STEP 9 AUDIT COMPARISON
# =====================================================================

print_header(
    "Comparing hybrid score with Step 9 evidence_score"
)

audit_correlation = float(
    df[
        [
            "hybrid_match_score",
            "evidence_score"
        ]
    ]
    .corr()
    .iloc[0, 1]
)

audit_mae = float(
    np.mean(
        np.abs(
            df["hybrid_match_score"]
            - df["evidence_score"]
        )
    )
)

audit_rmse = float(
    np.sqrt(
        np.mean(
            (
                df["hybrid_match_score"]
                - df["evidence_score"]
            ) ** 2
        )
    )
)

print(
    f"Correlation with Step 9 evidence_score: "
    f"{audit_correlation:.4f}"
)

print(
    f"Mean absolute difference: "
    f"{audit_mae:.4f}"
)

print(
    f"RMSE difference: "
    f"{audit_rmse:.4f}"
)

print("""
This is an AUDIT comparison only.

Step 9 evidence_score was NOT:
- used as a feature
- used as a target
- reconstructed
- modified
""")


# =====================================================================
# SIGNAL CONTRIBUTION SUMMARY
# =====================================================================

print_header(
    "Calculating signal contribution summary"
)

signal_map = {
    "semantic_similarity": "_semantic_signal",
    "skill_match_signal": "_skill_signal",
    "title_similarity": "_title_signal",
    "category_match_signal": "_category_signal",
    "job_skill_coverage": "_job_coverage_signal",
    "resume_skill_coverage": "_resume_coverage_signal",
    "jaccard_similarity": "_jaccard_signal",
    "text_containment": "_containment_signal",
    "maximum_skill_overlap_ratio": "_overlap_signal",
    "shared_skill_count_signal": "_shared_skill_signal",
}

signal_contributions = {}

for name, weight in WEIGHTS.items():

    signal_column = signal_map[name]

    mean_signal = float(
        df[signal_column].mean()
    )

    mean_contribution = (
        weight
        * mean_signal
    )

    signal_contributions[name] = {
        "weight": float(weight),
        "mean_signal": mean_signal,
        "mean_contribution": float(
            mean_contribution
        ),
    }

    print(
        f"{name}: "
        f"weight={weight:.3f}, "
        f"mean contribution="
        f"{mean_contribution:.4f}"
    )


# =====================================================================
# PREPARE FINAL OUTPUT
# =====================================================================

print_header(
    "Preparing Step 10 output"
)

output_columns = [
    "resume_id",
    "job_id",

    "semantic_similarity",

    "shared_skill_count",
    "resume_skill_count",
    "job_skill_count",

    "jaccard_similarity",
    "resume_skill_coverage",
    "job_skill_coverage",
    "skill_count_difference",
    "maximum_skill_overlap_ratio",

    "category_match_signal",
    "text_containment",
    "title_similarity",
    "skill_match_signal",

    "evidence_score",

    "hybrid_match_score",
    "job_rank",

    "top_5",
    "top_10",
    "top_20",
]

predictions = df[
    output_columns
].copy()

print(
    f"Output shape: "
    f"{predictions.shape}"
)


# =====================================================================
# FINAL VALIDATION
# =====================================================================

print_header(
    "Final output validation"
)

if len(predictions) != len(df):
    raise ValueError(
        "Output row count differs from input."
    )

if predictions[
    [
        "resume_id",
        "job_id"
    ]
].duplicated().any():

    raise ValueError(
        "Duplicate resume-job pairs exist in output."
    )

if predictions[
    "hybrid_match_score"
].isna().any():

    raise ValueError(
        "Hybrid match score contains missing values."
    )

if not predictions[
    "hybrid_match_score"
].between(
    0.0,
    1.0
).all():

    raise ValueError(
        "Hybrid score contains values outside [0,1]."
    )

print(
    "Row count validation: PASSED"
)

print(
    "Duplicate pair validation: PASSED"
)

print(
    "Hybrid score validation: PASSED"
)


# =====================================================================
# SAVE PREDICTIONS
# =====================================================================

print_header(
    "Saving Step 10 matching predictions"
)

predictions.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"Saved: {OUTPUT_PATH}"
)

print(
    f"Output shape: {predictions.shape}"
)


# =====================================================================
# SAVE REPORT
# =====================================================================

print_header(
    "Saving Step 10 report"
)

report = {

    "step": 10,

    "step_name":
        "Final Hybrid Matching and Ranking",

    "input": {
        "path": str(INPUT_PATH),
        "rows": int(len(df)),
        "unique_resumes": int(
            df["resume_id"].nunique()
        ),
        "unique_jobs": int(
            df["job_id"].nunique()
        ),
        "duplicate_resume_job_pairs":
            duplicate_pairs,
    },

    "semantic_matching": {

        "source":
            "Step 9 actual resume text + actual job title and description",

        "model":
            "all-MiniLM-L6-v2",

        "raw_semantic_text_required":
            False,

        "semantic_similarity_mean":
            float(
                df["semantic_similarity"].mean()
            ),

        "semantic_similarity_median":
            float(
                df["semantic_similarity"].median()
            ),

        "semantic_similarity_min":
            float(
                df["semantic_similarity"].min()
            ),

        "semantic_similarity_max":
            float(
                df["semantic_similarity"].max()
            ),

        "missing_semantic_similarity":
            semantic_missing,
    },

    "step9_evidence_score": {

        "preserved_for_audit":
            True,

        "used_as_feature":
            False,

        "used_as_training_target":
            False,

        "reconstructed":
            False,

        "modified":
            False,

        "mean":
            float(
                df["evidence_score"].mean()
            ),

        "median":
            float(
                df["evidence_score"].median()
            ),

        "minimum":
            float(
                df["evidence_score"].min()
            ),

        "maximum":
            float(
                df["evidence_score"].max()
            ),
    },

    "model": {

        "supervised_model":
            False,

        "estimator":
            None,

        "training_target":
            None,

        "pseudo_labels":
            False,

        "ground_truth_labels":
            False,

        "evidence_score_leakage":
            False,
    },

    "hybrid_scoring": {

        "method":
            "explicit weighted signal aggregation",

        "weights":
            WEIGHTS,

        "weight_sum":
            float(WEIGHT_SUM),

        "mean_score":
            float(
                df["hybrid_match_score"].mean()
            ),

        "median_score":
            float(
                df["hybrid_match_score"].median()
            ),

        "minimum_score":
            float(
                df["hybrid_match_score"].min()
            ),

        "maximum_score":
            float(
                df["hybrid_match_score"].max()
            ),
    },

    "ranking": {

        "ranking_unit":
            "resume",

        "ranking_direction":
            "descending hybrid_match_score",

        "top_k_statistics":
            top_k_statistics,

        "ranking_validation":
            "passed",
    },

    "audit_comparison": {

        "step9_evidence_correlation":
            audit_correlation,

        "mean_absolute_difference":
            audit_mae,

        "rmse_difference":
            audit_rmse,

        "interpretation":
            "Audit comparison only; not a ground-truth metric.",
    },

    "signal_contributions":
        signal_contributions,

    "evaluation": {

        "roc_auc":
            None,

        "f1":
            None,

        "ground_truth_hiring_metric":
            None,

        "next_stage":
            "Independent human validation",
    },

    "output": {

        "predictions_path":
            str(OUTPUT_PATH),

        "report_path":
            str(REPORT_PATH),

        "rows":
            int(len(predictions)),

        "columns":
            int(len(predictions.columns)),
    },
}


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=2,
        ensure_ascii=False
    )

print(
    f"Report saved: {REPORT_PATH}"
)


# =====================================================================
# FINAL SUMMARY
# =====================================================================

print_header(
    "STEP 10 COMPLETE"
)

print(
    f"Total resume-job pairs: "
    f"{len(df):,}"
)

print(
    f"Unique resumes: "
    f"{df['resume_id'].nunique():,}"
)

print(
    f"Unique jobs: "
    f"{df['job_id'].nunique():,}"
)

print(
    "Semantic source: ACTUAL TEXT "
    "(computed in Step 9)"
)

print(
    "Semantic model: all-MiniLM-L6-v2"
)

print(
    "Raw semantic text required in Step 10: NO"
)

print(
    "Step 9 evidence score used as feature: NO"
)

print(
    "Step 9 evidence score used as target: NO"
)

print(
    "Step 9 evidence score reconstructed: NO"
)

print(
    "Step 9 evidence score modified: NO"
)

print(
    "Pseudo-labels used: NO"
)

print(
    "Ground-truth hiring labels used: NO"
)

print(
    "Supervised ML model trained: NO"
)

print(
    "Evidence-score leakage: PREVENTED"
)

print(
    f"Mean hybrid match score: "
    f"{df['hybrid_match_score'].mean():.4f}"
)

print(
    f"Median hybrid match score: "
    f"{df['hybrid_match_score'].median():.4f}"
)

print(
    f"Minimum hybrid match score: "
    f"{df['hybrid_match_score'].min():.4f}"
)

print(
    f"Maximum hybrid match score: "
    f"{df['hybrid_match_score'].max():.4f}"
)

print(
    f"Predictions: {OUTPUT_PATH}"
)

print(
    f"Report: {REPORT_PATH}"
)

print(
    "\nNEXT: INDEPENDENT HUMAN VALIDATION"
)