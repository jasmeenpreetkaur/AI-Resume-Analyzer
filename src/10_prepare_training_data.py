"""
======================================================================
AI RESUME ANALYZER - STEP 10 FINAL HYBRID MATCHING
======================================================================

PURPOSE
-------
Score every resume-job pair in the Step 9 dataset and produce
resume-wise job rankings.

ONE SCORING DEFINITION
----------------------
The score is NOT computed here. It comes from matching_core.MatchScorer,
the same object the Streamlit app (14_app.py) and the command-line
analysis (13_explainable_rag_recommendation.py) use:

  * models/match_regressor.joblib present and valid (written by
    10b_train_match_model.py)  ->  the trained
    HistGradientBoostingRegressor scores the pairs;
  * otherwise -> the transparent weighted formula whose weights are
    defined once in matching_core.WEIGHTS.

Every output row records which one produced it (scoring_method) and also
keeps the formula score (formula_match_score) so the two can be compared.
Offline rankings, the app and the CLI therefore can no longer disagree
because of a different formula or different weights.

IMPORTANT
---------
- This step trains nothing. Training is Step 10b.
- Uses corrected Step 9 REAL semantic similarity (actual resume text,
  actual job title + description).
- Step 9 evidence_score is preserved ONLY for audit. It is not a model
  feature and not a training target.
- Category agreement is used by the formula when both categories are
  known; a missing category is not treated as a mismatch (the weight is
  redistributed), exactly as the app does for uploaded resumes.
- No ground-truth hiring labels exist, so no ROC-AUC/F1 is reported here.
- Resume-wise ranking is deterministic and reproducible.
- Independent human validation follows Step 10.

======================================================================
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

import matching_core as mc


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
# The weights are defined ONCE, in matching_core. They are design
# parameters (not learned) and are the fallback scorer.
#
WEIGHTS = mc.WEIGHTS

WEIGHT_SUM = sum(
    WEIGHTS.values()
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
    "AI RESUME ANALYZER - STEP 10 FINAL HYBRID MATCHING"
)

print("""
IMPORTANT:
- Uses corrected Step 9 REAL semantic similarity.
- Semantic similarity was computed from actual text in Step 9.
- Raw semantic text columns are NOT required here.
- Scoring comes from matching_core.MatchScorer (shared with the app).
- Step 9 evidence_score is audit-only.
- No pseudo-labels.
- No ground-truth hiring labels.
- Nothing is trained in this step (see 10b_train_match_model.py).
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
# NORMALIZE SIGNALS (shared definition)
# =====================================================================

print_header(
    "Normalizing matching signals"
)

#
# matching_core.signal_frame is the one place that clips signals to
# [0, 1] and derives the shared-skill signal. The shared-skill signal
# uses a FIXED cap (matching_core.SHARED_SKILL_CAP), not the maximum of
# this dataset, so the same pair scores the same no matter what other
# pairs were loaded - and the app can reproduce it for one resume.
#
signals = mc.signal_frame(df)

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

for signal_name, column_name in signal_map.items():
    df[column_name] = signals[signal_name].to_numpy()

normalized_columns = list(signal_map.values())

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

if "category_available" in df.columns:
    category_known = (
        pd.to_numeric(df["category_available"], errors="coerce")
        .fillna(0)
        .astype(bool)
    )
else:
    category_known = pd.Series(
        "category_match_signal" in df.columns,
        index=df.index,
    )
    print(
        "NOTE: Step 9 output has no category_available column "
        "(older Step 9 run); treating every category as known."
    )

print(
    f"Normalized signals: "
    f"{len(normalized_columns)}"
)

print(
    f"Missing normalized values: "
    f"{missing_normalized}"
)

print(
    f"Pairs with a known category on both sides: "
    f"{int(category_known.sum()):,} of {len(df):,}"
)


# =====================================================================
# CALCULATE HYBRID SCORE
# =====================================================================

print_header(
    "Scoring resume-job pairs"
)

scorer = mc.MatchScorer.load()

#
# The transparent formula is always computed so it can be audited and
# compared with the trained model.
#
df["formula_match_score"] = mc.formula_score(df)

score_result = scorer.score(df)

df["hybrid_match_score"] = np.clip(
    score_result.scores,
    0.0,
    1.0
)

df["scoring_method"] = score_result.method

print(
    f"Scoring method: {score_result.method}"
)

print(
    f"Scorer: {scorer.description}"
)

if scorer.uses_model:

    print(
        f"Model file: {scorer.model_path}"
    )

    print(
        "Training target: "
        f"{scorer.metadata.get('target', 'unknown')}"
    )

else:

    print(
        "No trained model in use "
        "(run 10b_train_match_model.py to create one)."
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

#
# Tie-break with the formula as it runs online (no category term), so a
# trained model that gives many pairs the same score still yields the
# same ordering the app would show.
#
df["_tiebreak_formula"] = mc.formula_score(
    df[list(mc.MODEL_FEATURES)]
)

df = df.sort_values(
    by=[
        "resume_id",
        "hybrid_match_score",
        "_tiebreak_formula",
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
    "Comparing scores with Step 9 evidence_score"
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

formula_gap = float(
    np.max(
        np.abs(
            df["formula_match_score"]
            - df["evidence_score"]
        )
    )
)

step9_uses_same_formula = bool(
    formula_gap < 1e-6
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

print(
    f"Max |formula_match_score - Step 9 evidence_score|: "
    f"{formula_gap:.6f}"
)

if step9_uses_same_formula:

    print(
        "Step 9 and Step 10 use the same formula: YES"
    )

else:

    print(
        "WARNING: Step 9 evidence_score does not equal the shared "
        "formula. The Step 9 CSV was probably produced by an older "
        "version of Step 9 with different weights - re-run Step 9."
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

signal_contributions = {}

# These are contributions to the transparent FORMULA. When a trained
# model is the active scorer they describe the formula, not the model.

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
    "category_available",
    "text_containment",
    "title_similarity",
    "skill_match_signal",

    "evidence_score",

    "hybrid_match_score",
    "formula_match_score",
    "scoring_method",
    "scorer_fingerprint",
    "claim_label",
    "job_rank",

    "top_5",
    "top_10",
    "top_20",
]

output_columns = [
    column
    for column in output_columns
    if column in df.columns
]

predictions = df[
    output_columns
].copy()
predictions["scorer_fingerprint"] = scorer.scorer_fingerprint
predictions["claim_label"] = scorer.claim_label

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

    "scoring": {

        "source":
            "matching_core.MatchScorer (shared with steps 13 and 14)",

        "scoring_method":
            score_result.method,

        "description":
            scorer.description,

        "claim_label": scorer.claim_label,
        "claim_description": scorer.claim_description,
        "claim": scorer.claim.to_dict(),
        "suitability_wording_allowed": scorer.suitability_wording_allowed,
        "scorer_fingerprint": scorer.scorer_fingerprint,
        "model_trained_at": (
            scorer.metadata.get("trained_at") if scorer.uses_model else None
        ),

        "trained_model_in_use":
            bool(scorer.uses_model),

        "model_file":
            scorer.model_path,

        "fallback_reason":
            scorer.fallback_reason or None,

        "formula_weights":
            WEIGHTS,

        "category_weight_redistributed_when_unknown":
            True,

        "shared_skill_signal_cap":
            mc.SHARED_SKILL_CAP,
    },

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
            mc.SEMANTIC_MODEL_NAME,

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

        "equals_shared_formula":
            step9_uses_same_formula,

        "max_abs_gap_to_formula":
            formula_gap,

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
            bool(scorer.uses_model),

        "trained_in_this_step":
            False,

        "estimator":
            (
                type(scorer.estimator).__name__
                if scorer.uses_model
                else None
            ),

        "training_target":
            (
                scorer.metadata.get("target")
                if scorer.uses_model
                else None
            ),

        "model_features":
            (
                list(scorer.feature_names)
                if scorer.uses_model
                else None
            ),

        "model_trained_at":
            (
                scorer.metadata.get("trained_at")
                if scorer.uses_model
                else None
            ),

        "pseudo_labels":
            False,

        "ground_truth_labels":
            False,

        "evidence_score_leakage":
            False,
    },

    "hybrid_scoring": {

        "method":
            (
                "trained model prediction"
                if scorer.uses_model
                else "explicit weighted signal aggregation"
            ),

        "formula_mean_score":
            float(
                df["formula_match_score"].mean()
            ),

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
            "descending hybrid_match_score "
            "(ties: online formula score, semantic, title, skill signal, "
            "job_id)",

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
    f"Semantic model: {mc.SEMANTIC_MODEL_NAME}"
)

print(
    f"Scoring method: {score_result.method}"
)

print(
    f"Scorer: {scorer.description}"
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
    "Model trained in this step: NO"
)

print(
    "Trained model used for scoring: "
    + ("YES" if scorer.uses_model else "NO (formula)")
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