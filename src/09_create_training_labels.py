"""
AI RESUME ANALYZER - STEP 9
REAL SEMANTIC + STRUCTURED HYBRID MATCHING DATASET

This version:
- Uses ACTUAL resume text (Step 4 output: parsed PDF text when available)
- Uses ACTUAL job title + job description
- Does NOT use constructed text representations
- Does NOT create pseudo-labels
- Does NOT create training labels
- Computes semantic similarity using Sentence Transformers
- Preserves existing structured skill features
- Computes every signal through matching_core.py, the same definitions
  used by Step 10 (scoring), Step 13 (CLI) and the Streamlit app, so the
  offline dataset and the online app never disagree about a formula

evidence_score is the transparent weighted formula from matching_core
(weights live in ONE place). It is an audit score, not a learned label.
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import matching_core as mc

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_FILE = str(PROCESSED_DIR / "resume_job_features.csv")

RESUME_FILE = str(PROCESSED_DIR / "resumes_text_processed.csv")

JOB_FILE = str(PROCESSED_DIR / "jobs_text_processed.csv")

OUTPUT_FILE = str(PROCESSED_DIR / "resume_job_matching_dataset.csv")

REPORT_FILE = str(
    PROJECT_ROOT / "reports" / "semantic_matching_dataset_report.json"
)

MODEL_NAME = mc.SEMANTIC_MODEL_NAME

BATCH_SIZE = 64


# ============================================================
# LOGGING
# ============================================================

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# TEXT / LEXICAL SIGNALS (shared definitions)
# ============================================================

clean_text = mc.clean_text

calculate_title_similarity = mc.title_similarity

calculate_text_containment = mc.text_containment


def safe_float(value, default=0.0):
    try:
        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return default

        return value

    except Exception:
        return default


# ============================================================
# MAIN
# ============================================================

def main():

    section("AI RESUME ANALYZER - STEP 9 REAL SEMANTIC MATCHING DATASET")

    print("IMPORTANT:")
    print("- Actual resume text WILL be used.")
    print("- Actual job title + description WILL be used.")
    print("- No constructed semantic representations.")
    print("- No pseudo-labels.")
    print("- No ground-truth labels.")
    print("- Semantic similarity uses Sentence Transformers.")

    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)

    # --------------------------------------------------------
    # LOAD FEATURE DATASET
    # --------------------------------------------------------

    section("Loading existing structured matching features")

    features = pd.read_csv(FEATURE_FILE)

    print(f"Input shape: {features.shape[0]:,} rows × {features.shape[1]} columns")

    required_feature_columns = [
        "resume_id",
        "job_id",
        "resume_category",
        "job_category",
        "job_title",
        "shared_skill_count",
        "resume_skill_count",
        "job_skill_count",
        "jaccard_similarity",
        "resume_skill_coverage",
        "job_skill_coverage",
    ]

    missing = [
        col for col in required_feature_columns
        if col not in features.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}"
        )

    # --------------------------------------------------------
    # VALIDATE FEATURE DATASET
    # --------------------------------------------------------

    section("Validating structured matching dataset")

    print(f"Rows: {len(features):,}")
    print(f"Unique resumes: {features['resume_id'].nunique():,}")
    print(f"Unique jobs: {features['job_id'].nunique():,}")

    duplicate_pairs = features.duplicated(
        subset=["resume_id", "job_id"]
    ).sum()

    print(f"Duplicate resume-job pairs: {duplicate_pairs:,}")

    if duplicate_pairs > 0:
        raise ValueError(
            "Duplicate resume-job pairs detected."
        )

    # --------------------------------------------------------
    # LOAD REAL RESUME TEXT
    # --------------------------------------------------------

    section("Loading ACTUAL resume text")

    resumes = pd.read_csv(RESUME_FILE)

    required_resume_columns = [
        "resume_id",
        "resume_text"
    ]

    missing = [
        col for col in required_resume_columns
        if col not in resumes.columns
    ]

    if missing:
        raise ValueError(
            f"Missing resume columns: {missing}"
        )

    resumes = resumes[
        ["resume_id", "resume_text"]
    ].copy()

    resumes["resume_text"] = resumes["resume_text"].fillna("").map(clean_text)

    resumes = resumes.drop_duplicates(
        subset=["resume_id"]
    )

    print(f"Resume records: {len(resumes):,}")
    print(
        "Non-empty resume texts:",
        (resumes["resume_text"].str.len() > 0).sum()
    )

    # --------------------------------------------------------
    # LOAD REAL JOB TEXT
    # --------------------------------------------------------

    section("Loading ACTUAL job descriptions")

    jobs = pd.read_csv(JOB_FILE)

    required_job_columns = [
        "job_id",
        "job_title",
        "job_description"
    ]

    missing = [
        col for col in required_job_columns
        if col not in jobs.columns
    ]

    if missing:
        raise ValueError(
            f"Missing job columns: {missing}"
        )

    jobs = jobs[
        [
            "job_id",
            "job_title",
            "job_description"
        ]
    ].copy()

    jobs["job_title"] = jobs["job_title"].fillna("").map(clean_text)
    jobs["job_description"] = (
        jobs["job_description"]
        .fillna("")
        .map(clean_text)
    )

    jobs = jobs.drop_duplicates(
        subset=["job_id"]
    )

    # --------------------------------------------------------
    # BUILD ACTUAL JOB SEMANTIC TEXT
    # --------------------------------------------------------

    # Same construction the online app and Step 13 use.
    jobs["semantic_job_text"] = [
        mc.build_job_text(title, description)
        for title, description
        in zip(jobs["job_title"], jobs["job_description"])
    ]

    print(f"Job records: {len(jobs):,}")
    print(
        "Non-empty job texts:",
        (jobs["semantic_job_text"].str.len() > 0).sum()
    )

    # --------------------------------------------------------
    # BUILD LOOKUP TABLES
    # --------------------------------------------------------

    section("Building resume/job text lookup")

    resume_lookup = dict(
        zip(
            resumes["resume_id"],
            resumes["resume_text"]
        )
    )

    job_lookup = dict(
        zip(
            jobs["job_id"],
            jobs["semantic_job_text"]
        )
    )

    # --------------------------------------------------------
    # VERIFY COVERAGE
    # --------------------------------------------------------

    features["resume_text"] = features["resume_id"].map(
        resume_lookup
    )

    features["job_semantic_text"] = features["job_id"].map(
        job_lookup
    )

    missing_resume_text = (
        features["resume_text"]
        .isna()
        .sum()
    )

    missing_job_text = (
        features["job_semantic_text"]
        .isna()
        .sum()
    )

    print(
        f"Pairs missing resume text: {missing_resume_text:,}"
    )

    print(
        f"Pairs missing job text: {missing_job_text:,}"
    )

    if missing_resume_text > 0:
        print(
            "WARNING: Some resume IDs do not have corresponding text."
        )

    if missing_job_text > 0:
        print(
            "WARNING: Some job IDs do not have corresponding text."
        )

    features["resume_text"] = (
        features["resume_text"]
        .fillna("")
        .map(clean_text)
    )

    features["job_semantic_text"] = (
        features["job_semantic_text"]
        .fillna("")
        .map(clean_text)
    )

    # --------------------------------------------------------
    # LOAD SENTENCE TRANSFORMER
    # --------------------------------------------------------

    section("Loading semantic embedding model")

    print(f"Model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Sentence Transformer loaded.")

    # --------------------------------------------------------
    # UNIQUE TEXTS
    # --------------------------------------------------------

    section("Preparing unique texts for semantic encoding")

    unique_resume_ids = features["resume_id"].unique()
    unique_job_ids = features["job_id"].unique()

    resume_texts = [
        resume_lookup.get(resume_id, "")
        for resume_id in unique_resume_ids
    ]

    job_texts = [
        job_lookup.get(job_id, "")
        for job_id in unique_job_ids
    ]

    print(
        f"Unique resumes to encode: {len(resume_texts):,}"
    )

    print(
        f"Unique jobs to encode: {len(job_texts):,}"
    )

    # --------------------------------------------------------
    # ENCODE RESUMES
    # --------------------------------------------------------

    section("Encoding ACTUAL resume text")

    resume_embeddings = model.encode(
        resume_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print(
        "Resume embedding matrix:",
        resume_embeddings.shape
    )

    # --------------------------------------------------------
    # ENCODE JOBS
    # --------------------------------------------------------

    section("Encoding ACTUAL job title + description")

    job_embeddings = model.encode(
        job_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print(
        "Job embedding matrix:",
        job_embeddings.shape
    )

    # --------------------------------------------------------
    # BUILD EMBEDDING LOOKUPS
    # --------------------------------------------------------

    resume_embedding_lookup = {
        resume_id: embedding
        for resume_id, embedding
        in zip(unique_resume_ids, resume_embeddings)
    }

    job_embedding_lookup = {
        job_id: embedding
        for job_id, embedding
        in zip(unique_job_ids, job_embeddings)
    }

    # --------------------------------------------------------
    # CALCULATE SEMANTIC SIMILARITY
    # --------------------------------------------------------

    section("Calculating REAL semantic similarity")

    semantic_scores = np.zeros(
        len(features),
        dtype=np.float32
    )

    for start in range(0, len(features), 10000):

        end = min(
            start + 10000,
            len(features)
        )

        batch = features.iloc[start:end]

        resume_matrix = np.vstack([
            resume_embedding_lookup[rid]
            for rid in batch["resume_id"]
        ])

        job_matrix = np.vstack([
            job_embedding_lookup[jid]
            for jid in batch["job_id"]
        ])

        # Embeddings are normalized, so dot product = cosine similarity
        scores = np.sum(
            resume_matrix * job_matrix,
            axis=1
        )

        semantic_scores[start:end] = scores

        print(
            f"Processed pairs: {end:,}/{len(features):,}"
        )

    # --------------------------------------------------------
    # NORMALIZE SEMANTIC SCORE
    # --------------------------------------------------------

    # Cosine similarity can technically range [-1, 1].
    # For matching features we map it into [0, 1].

    features["semantic_similarity"] = mc.semantic_signal(
        semantic_scores
    )

    # --------------------------------------------------------
    # TEXT CONTAINMENT
    # --------------------------------------------------------

    section("Calculating lexical text containment")

    features["text_containment"] = [
        calculate_text_containment(
            resume_text,
            job_text
        )
        for resume_text, job_text
        in zip(
            features["resume_text"],
            features["job_semantic_text"]
        )
    ]

    # --------------------------------------------------------
    # TITLE SIMILARITY
    # --------------------------------------------------------

    section("Calculating title similarity")

    features["title_similarity"] = [
        calculate_title_similarity(
            job_title,
            resume_text
        )
        for job_title, resume_text
        in zip(
            features["job_title"],
            features["resume_text"]
        )
    ]

    # --------------------------------------------------------
    # SKILL MATCH SIGNAL
    # --------------------------------------------------------

    section("Building structured skill signal")

    features["skill_match_signal"] = mc.skill_match_signal(
        features["jaccard_similarity"].clip(0, 1),
        features["resume_skill_coverage"].clip(0, 1),
        features["job_skill_coverage"].clip(0, 1),
    )

    # --------------------------------------------------------
    # CATEGORY SIGNAL
    # --------------------------------------------------------

    # Unified definition (matching_core.category_signal_series):
    # case/separator-insensitive, and two MISSING categories are NOT a
    # match (the old code compared "" == "" and called it a match).
    # category_available / category_match are rewritten here so Step 10
    # and the report read one consistent definition.
    category_signal, category_available = mc.category_signal_series(
        features["resume_category"],
        features["job_category"],
    )

    features["category_match_signal"] = category_signal.astype(int)
    features["category_available"] = category_available.astype(int)
    features["category_match"] = features["category_match_signal"]

    print(
        "Pairs with a usable category on both sides:",
        f"{int(category_available.sum()):,}"
        f" of {len(features):,}"
    )

    # --------------------------------------------------------
    # HYBRID EVIDENCE SCORE
    # --------------------------------------------------------

    section("Building transparent hybrid evidence score")

    # This is NOT a learned ground-truth label. It is the transparent
    # weighted formula defined once in matching_core.WEIGHTS (the same
    # weights Step 10, Step 13 and the app use). When a pair has no
    # usable category the category weight is redistributed instead of
    # silently scoring the pair lower.

    features["evidence_score"] = mc.formula_score(features)

    print("Formula weights:", json.dumps(mc.WEIGHTS))

    # --------------------------------------------------------
    # SAVE FINAL DATASET
    # --------------------------------------------------------

    section("Preparing final semantic matching dataset")

    output_columns = [
        "resume_id",
        "job_id",
        "resume_category",
        "job_category",
        "job_title",

        "shared_skill_count",
        "resume_skill_count",
        "job_skill_count",

        "jaccard_similarity",
        "resume_skill_coverage",
        "job_skill_coverage",

        "shared_esco_skill_uris",

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

        # ACTUAL text
        "resume_text",
        "job_semantic_text",

        # Semantic / lexical signals
        "semantic_similarity",
        "text_containment",
        "title_similarity",
        "skill_match_signal",
        "category_match_signal",
        "evidence_score",
    ]

    # Keep only columns that actually exist
    output_columns = [
        col for col in output_columns
        if col in features.columns
    ]

    final_dataset = features[output_columns].copy()

    # Rename actual text fields clearly
    final_dataset = final_dataset.rename(
        columns={
            "resume_text":
                "resume_text_representation",

            "job_semantic_text":
                "job_text_representation"
        }
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    section("Validating Step 9 dataset")

    print(
        f"Rows: {len(final_dataset):,}"
    )

    print(
        f"Columns: {len(final_dataset.columns)}"
    )

    print(
        "Unique resumes:",
        final_dataset["resume_id"].nunique()
    )

    print(
        "Unique jobs:",
        final_dataset["job_id"].nunique()
    )

    print(
        "Semantic similarity mean:",
        round(
            final_dataset["semantic_similarity"].mean(),
            4
        )
    )

    print(
        "Semantic similarity median:",
        round(
            final_dataset["semantic_similarity"].median(),
            4
        )
    )

    print(
        "Semantic similarity min:",
        round(
            final_dataset["semantic_similarity"].min(),
            4
        )
    )

    print(
        "Semantic similarity max:",
        round(
            final_dataset["semantic_similarity"].max(),
            4
        )
    )

    print(
        "Evidence score mean:",
        round(
            final_dataset["evidence_score"].mean(),
            4
        )
    )

    missing_semantic = (
        final_dataset["semantic_similarity"]
        .isna()
        .sum()
    )

    print(
        "Missing semantic similarity:",
        missing_semantic
    )

    if missing_semantic > 0:
        raise ValueError(
            "Missing semantic similarity values detected."
        )

    # --------------------------------------------------------
    # CHECK ACTUAL TEXT
    # --------------------------------------------------------

    section("Checking semantic text source")

    sample_resume = (
        final_dataset["resume_text_representation"]
        .iloc[0]
    )

    sample_job = (
        final_dataset["job_text_representation"]
        .iloc[0]
    )

    print("\nResume semantic text sample:")
    print(sample_resume[:500])

    print("\nJob semantic text sample:")
    print(sample_job[:500])

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    section("Saving semantic matching dataset")

    final_dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print(
        f"Output shape: {final_dataset.shape}"
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    section("Saving Step 9 report")

    report = {
        "step": 9,
        "description":
            "Real semantic + structured hybrid matching dataset",

        "input_files": {
            "structured_features": FEATURE_FILE,
            "resume_text": RESUME_FILE,
            "job_text": JOB_FILE
        },

        "output_file": OUTPUT_FILE,

        "rows": int(len(final_dataset)),

        "unique_resumes":
            int(final_dataset["resume_id"].nunique()),

        "unique_jobs":
            int(final_dataset["job_id"].nunique()),

        "semantic_model": MODEL_NAME,

        "semantic_text_source": {
            "resume":
                "actual resume_text",
            "job":
                "actual job_title + job_description"
        },

        "semantic_similarity": {
            "mean":
                float(
                    final_dataset["semantic_similarity"].mean()
                ),
            "median":
                float(
                    final_dataset["semantic_similarity"].median()
                ),
            "minimum":
                float(
                    final_dataset["semantic_similarity"].min()
                ),
            "maximum":
                float(
                    final_dataset["semantic_similarity"].max()
                )
        },

        "evidence_score": {
            "definition": "matching_core.formula_score",
            "weights": mc.WEIGHTS,
            "category_weight_redistributed_when_unknown": True,
            "mean":
                float(
                    final_dataset["evidence_score"].mean()
                ),
            "median":
                float(
                    final_dataset["evidence_score"].median()
                )
        },

        "category": {
            "pairs_with_category_on_both_sides":
                int(final_dataset["category_available"].sum()),
            "pairs_with_matching_category":
                int(final_dataset["category_match"].sum())
        },

        "pseudo_labels_created": False,

        "ground_truth_labels_used": False,

        "training_labels_created": False,

        "semantic_representation": "REAL_TEXT",

        "notes": [
            "Resume semantic text comes from resume_text.",
            "Job semantic text comes from job_title + job_description.",
            "Embeddings are computed once per unique resume/job.",
            "Pairwise semantic similarity uses cosine similarity.",
            "Structured ESCO skill signals are preserved.",
            "All signals are computed by matching_core.py.",
            "Missing categories are not counted as a category match.",
            "No pseudo-labels are generated."
        ]
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print(
        f"Report saved: {REPORT_FILE}"
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    section("STEP 9 COMPLETE")

    print(
        f"Total resume-job pairs: {len(final_dataset):,}"
    )

    print(
        f"Unique resumes: {final_dataset['resume_id'].nunique():,}"
    )

    print(
        f"Unique jobs: {final_dataset['job_id'].nunique():,}"
    )

    print(
        "Semantic representation: ACTUAL TEXT"
    )

    print(
        "Resume source: resume_text"
    )

    print(
        "Job source: job_title + job_description"
    )

    print(
        "Pseudo-labels: NOT CREATED"
    )

    print(
        "Ground-truth labels: NOT USED"
    )

    print(
        f"Dataset: {OUTPUT_FILE}"
    )

    print(
        f"Report: {REPORT_FILE}"
    )

    print(
        "\nNext: rerun Step 10 using the corrected Step 9 dataset."
    )


if __name__ == "__main__":
    main()