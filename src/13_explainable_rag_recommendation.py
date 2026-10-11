"""
======================================================================
AI RESUME ANALYZER - STEP 13
USER-FACING EXPLAINABLE RAG RECOMMENDATION PIPELINE
======================================================================

PURPOSE
-------
Command-line version of the uploaded-resume analysis. It uses the SAME
shared modules as the Streamlit app (14_app.py), so a resume gets the
same skills, the same scores and the same ranking from either:

    pdf_text.py        PDF text extraction with OCR fallback for
                       scanned / image-only resumes
    esco_vocab.py      skill vocabulary = ESCO skills table only
    job_ranker.py      embeddings, skill/lexical signals, ranking
    matching_core.py   the ONE scoring definition: trained
                       HistGradientBoostingRegressor when
                       models/match_regressor.joblib exists (Step 10b),
                       otherwise the transparent weighted formula

USAGE
-----
python src/13_explainable_rag_recommendation.py [path/to/resume.pdf]

Default resume: data/test_resume.pdf

PIPELINE
--------
PDF Resume
    -> Text extraction (OCR fallback if the PDF is scanned)
    -> ESCO skill extraction
    -> Resume semantic embedding
    -> Job dataset + cached job embeddings (fingerprinted cache)
    -> Shared hybrid scoring (model or formula) and ranking
    -> Top-K jobs
    -> Groq explanation (explanation only; fallback if unavailable)
    -> Recommendation files

IMPORTANT
---------
- User uploads ONE resume.
- No manual annotation. No pseudo-labels. No ground-truth hiring labels.
- No hiring probability.
- LLM does NOT determine ranking and does NOT create the score.
- LLM only explains already-computed matching evidence.
- Skill gaps come from ESCO skills found in the job text/skill set.
- Groq failure does NOT stop the pipeline.

OUTPUT
------
data/recommendations/
    test_resume_recommendations.csv
    test_resume_top10.csv
    test_resume_analysis.json

reports/
    step13_user_resume_analysis_report.json

CACHE
-----
data/cache/job_embeddings.npy (+ .meta.json)
data/cache/job_skill_index.json
======================================================================
"""

import os
import re
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import matching_core as mc
import candidate_retrieval
import pdf_text
from esco_vocab import load_esco_vocabulary
from job_ranker import (
    build_job_corpus,
    rank_resume,
    resume_skill_uris,
    skill_breakdown,
)

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = PROJECT_ROOT

DEFAULT_RESUME_PATH = (
    PROJECT_ROOT
    / "data"
    / "test_resume.pdf"
)

JOB_PATH = (
    PROJECT_ROOT
    / "data"
    / "jobs"
    / "all_job_post.csv"
)

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "recommendations"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

JOB_EMBEDDING_CACHE = (
    CACHE_DIR
    / "job_embeddings.npy"
)

OUTPUT_ALL = (
    OUTPUT_DIR
    / "test_resume_recommendations.csv"
)

OUTPUT_TOP10 = (
    OUTPUT_DIR
    / "test_resume_top10.csv"
)

OUTPUT_ANALYSIS = (
    OUTPUT_DIR
    / "test_resume_analysis.json"
)

REPORT_PATH = (
    REPORT_DIR
    / "step13_user_resume_analysis_report.json"
)

SEMANTIC_MODEL_NAME = (
    mc.SEMANTIC_MODEL_NAME
)

TOP_K = 10

JOB_EMBED_BATCH_SIZE = 32

MAX_RESUME_CHARS_FOR_LLM = 6000

MAX_JOB_CHARS_FOR_LLM = 5000


# ======================================================================
# LOAD ENVIRONMENT
# ======================================================================

load_dotenv(
    PROJECT_ROOT / ".env"
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)


# ======================================================================
# OPTIONAL GROQ CLIENT
# ======================================================================

groq_client = None

if GROQ_API_KEY:

    try:

        from groq import Groq

        groq_client = Groq(
            api_key=GROQ_API_KEY
        )

        print(
            f"Groq explanations enabled: "
            f"{GROQ_MODEL}"
        )

    except Exception as exc:

        print(
            "WARNING: Groq initialization failed."
        )

        print(
            f"Reason: {exc}"
        )

        print(
            "Using deterministic fallback explanations."
        )

else:

    print(
        "GROQ_API_KEY not found."
    )

    print(
        "Using deterministic fallback explanations."
    )


# ======================================================================
# HELPERS
# ======================================================================

def print_header(title):

    print(
        "\n"
        + "=" * 70
    )

    print(title)

    print(
        "=" * 70
    )


def normalize_text(value):

    if value is None:
        return ""

    try:

        if pd.isna(value):
            return ""

    except Exception:
        pass

    text = str(value)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def safe_float(
    value,
    default=0.0
):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


def find_column(
    df,
    candidates,
    required=True
):

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        key = (
            str(candidate)
            .strip()
            .lower()
        )

        if key in normalized:

            return normalized[key]

    if required:

        raise ValueError(
            f"Required column not found.\n"
            f"Tried: {candidates}\n"
            f"Available columns: "
            f"{list(df.columns)}"
        )

    return None


def parse_list_value(value):

    if value is None:
        return []

    try:

        if pd.isna(value):
            return []

    except Exception:
        pass

    if isinstance(
        value,
        list
    ):

        return [
            normalize_text(x)
            for x in value
            if normalize_text(x)
        ]

    text = normalize_text(
        value
    )

    if not text:
        return []

    if (
        text.startswith("[")
        and
        text.endswith("]")
    ):

        try:

            parsed = json.loads(
                text
            )

            if isinstance(
                parsed,
                list
            ):

                return [
                    normalize_text(x)
                    for x in parsed
                    if normalize_text(x)
                ]

        except Exception:
            pass

    parts = re.split(
        r"\s*[;,|]\s*",
        text
    )

    return [
        normalize_text(x)
        for x in parts
        if normalize_text(x)
    ]


def clean_list(values):

    result = []

    seen = set()

    for value in values:

        value = normalize_text(
            value
        )

        if not value:
            continue

        key = value.lower()

        if key not in seen:

            seen.add(key)

            result.append(
                value
            )

    return result


# ======================================================================
# FALLBACK EXPLANATION
# ======================================================================

def fallback_explanation(
    job_title,
    score,
    semantic_similarity,
    matched_skills,
    skill_gaps
):

    matched = clean_list(
        matched_skills
    )

    gaps = clean_list(
        skill_gaps
    )

    if matched:

        matched_text = (
            ", ".join(
                matched[:8]
            )
        )

        why = (
            f"The resume shows relevant evidence for "
            f"the {job_title} role, including "
            f"{matched_text}. "
        )

    else:

        why = (
            f"The resume has text-level relevance "
            f"to the {job_title} role, although "
            f"no explicit overlapping skills were "
            f"identified by the skill matcher. "
        )

    why += (
        f"The semantic similarity signal is "
        f"{semantic_similarity:.3f}."
    )

    if gaps:

        gap_text = ", ".join(
            gaps[:8]
        )

        summary = (
            f"Potential skill gaps include "
            f"{gap_text}."
        )

    else:

        summary = (
            "No explicit skill gaps were "
            "identified from the available "
            "skill vocabulary."
        )

    return {
        "why_match": why,
        "matched_evidence": matched[:8],
        "skill_gaps": gaps[:8],
        "recommendation_summary": summary,
        "limitations": (
            f"The hybrid score is {score:.3f} and "
            "represents matching evidence, not "
            "a probability of hiring."
        ),
        "explanation_source": (
            "Deterministic evidence-based fallback"
        )
    }


# ======================================================================
# GROQ EXPLANATION
# ======================================================================

def generate_groq_explanation(
    resume_text,
    job_title,
    job_description,
    score,
    semantic_similarity,
    matched_skills,
    skill_gaps
):

    if groq_client is None:

        return fallback_explanation(
            job_title,
            score,
            semantic_similarity,
            matched_skills,
            skill_gaps
        )

    matched_text = ", ".join(
        clean_list(
            matched_skills
        )[:12]
    )

    gaps_text = ", ".join(
        clean_list(
            skill_gaps
        )[:12]
    )

    prompt = f"""
You are an explainability assistant for a resume-job
matching system.

You MUST ONLY explain the existing recommendation.

Do not:
- change the ranking
- create another score
- invent experience
- invent skills
- invent qualifications
- claim hiring probability
- claim the candidate will get the job
- introduce information not supplied below

Existing matching score:
{score:.4f}

Semantic similarity:
{semantic_similarity:.4f}

Matched skills:
{matched_text}

Potential skill gaps:
{gaps_text}

Resume:
{resume_text[:MAX_RESUME_CHARS_FOR_LLM]}

Job title:
{job_title}

Job description:
{job_description[:MAX_JOB_CHARS_FOR_LLM]}

Return ONLY valid JSON:

{{
  "why_match": "2-4 concise evidence-based sentences",
  "matched_evidence": ["evidence 1", "evidence 2"],
  "skill_gaps": ["gap 1", "gap 2"],
  "recommendation_summary": "1-2 concise sentences",
  "limitations": "One sentence stating that this is matching evidence, not hiring probability."
}}

Every statement must be supported by the supplied information.
"""

    try:

        response = groq_client.chat.completions.create(

            model=GROQ_MODEL,

            temperature=0.1,

            max_tokens=600,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate strictly "
                        "evidence-grounded "
                        "resume-job explanations."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        content = re.sub(
            r"^```json\s*",
            "",
            content,
            flags=re.IGNORECASE
        )

        content = re.sub(
            r"\s*```$",
            "",
            content
        )

        parsed = json.loads(
            content
        )

        if not isinstance(
            parsed,
            dict
        ):

            raise ValueError(
                "Groq returned non-object JSON."
            )

        return {
            "why_match": normalize_text(
                parsed.get(
                    "why_match",
                    ""
                )
            ),
            "matched_evidence": clean_list(
                parsed.get(
                    "matched_evidence",
                    []
                )
            ),
            "skill_gaps": clean_list(
                parsed.get(
                    "skill_gaps",
                    []
                )
            ),
            "recommendation_summary":
                normalize_text(
                    parsed.get(
                        "recommendation_summary",
                        ""
                    )
                ),
            "limitations": normalize_text(
                parsed.get(
                    "limitations",
                    ""
                )
            ),
            "explanation_source": (
                "Groq LLM + actual resume/job evidence"
            )
        }

    except Exception as exc:

        print(
            f"Groq explanation failed for "
            f"'{job_title}'."
        )

        print(
            f"Reason: {exc}"
        )

        print(
            "Using deterministic fallback "
            "for this recommendation."
        )

        fallback = fallback_explanation(
            job_title,
            score,
            semantic_similarity,
            matched_skills,
            skill_gaps
        )

        fallback["llm_error"] = str(
            exc
        )

        return fallback


# ======================================================================
# MAIN
# ======================================================================

print_header(
    "INITIALIZING RESUME ANALYZER"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# SCORER (trained model if present, otherwise the shared formula)
# ======================================================================

scorer = mc.MatchScorer.load()

print(
    f"Scoring: {scorer.description}"
)


# ======================================================================
# RESUME INPUT
# ======================================================================

resume_path = (
    Path(sys.argv[1]).expanduser()
    if len(sys.argv) > 1
    else DEFAULT_RESUME_PATH
)

print(
    f"Resume file: {resume_path}"
)

if not resume_path.exists():

    raise FileNotFoundError(
        f"Resume not found:\n{resume_path}"
    )


print(
    "Extracting resume text..."
)

ocr_status = pdf_text.ocr_engine_status()

print(
    "OCR engine for scanned PDFs: "
    + (
        ocr_status["selected"]
        if ocr_status["selected"]
        else "none installed"
    )
)

extraction = pdf_text.extract_pdf_text(
    resume_path
)

if not extraction.usable:

    # Scanned PDF without OCR, empty PDF, unreadable file: say exactly
    # what is wrong instead of continuing with an empty resume.
    raise SystemExit(
        "Could not read the resume: "
        + extraction.user_message()
    )

resume_text = extraction.text

if extraction.warnings:

    for warning in extraction.warnings:

        print(
            f"WARNING: {warning}"
        )

if extraction.ocr_used:

    print(
        f"OCR used on pages {extraction.ocr_pages} "
        f"with {extraction.ocr_engine} "
        f"(mean confidence "
        f"{extraction.mean_ocr_confidence})"
    )

print(
    f"Extracted resume characters: "
    f"{len(resume_text):,}"
)


# ======================================================================
# LOAD JOBS
# ======================================================================

print_header(
    "LOADING JOB DATASET"
)

if not JOB_PATH.exists():

    raise FileNotFoundError(
        f"Job dataset not found:\n{JOB_PATH}"
    )

jobs = pd.read_csv(
    JOB_PATH,
    low_memory=False
)

print(
    f"Jobs loaded: {len(jobs):,}"
)


# ======================================================================
# LOAD SEMANTIC MODEL
# ======================================================================

print_header(
    "LOADING SEMANTIC MODEL"
)

print(
    f"Model: {SEMANTIC_MODEL_NAME}"
)

model = SentenceTransformer(
    SEMANTIC_MODEL_NAME
)


def encode(texts):

    return np.asarray(
        model.encode(
            list(texts),
            batch_size=JOB_EMBED_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        ),
        dtype=np.float32
    )


# ======================================================================
# SKILL VOCABULARY (ESCO skills table only)
# ======================================================================

print_header(
    "LOADING ESCO SKILL VOCABULARY"
)

try:

    vocabulary = load_esco_vocabulary()

    print(
        f"ESCO skills file: {vocabulary.source}"
    )

    print(
        f"ESCO skills: {vocabulary.skill_count:,} | "
        f"matching terms: {vocabulary.term_count:,}"
    )

except (FileNotFoundError, ValueError) as exc:

    vocabulary = None

    print(
        f"WARNING: {exc}"
    )

    print(
        "Skill matching is disabled; jobs are ranked on "
        "semantic and lexical similarity only."
    )


# ======================================================================
# JOB CORPUS (embeddings + job skill sets, fingerprinted caches)
# ======================================================================

print_header(
    "PREPARING JOB EMBEDDINGS AND SKILLS"
)

corpus = build_job_corpus(
    jobs,
    encode=encode,
    model_name=SEMANTIC_MODEL_NAME,
    vocabulary=vocabulary,
    cache_dir=CACHE_DIR,
    progress=lambda message, fraction: print(
        f"  {message} ({fraction:.0%})"
    ),
)

for note in corpus.notes:

    print(
        f"NOTE: {note}"
    )

job_columns = corpus.columns

job_title_col = job_columns["title"]
job_description_col = job_columns["description"]
job_company_col = job_columns["company"]
job_location_col = job_columns["location"]


# ======================================================================
# RESUME SKILLS
# ======================================================================

print_header(
    "EXTRACTING RESUME SKILLS"
)

resume_skill_map = resume_skill_uris(
    resume_text,
    vocabulary
)

resume_skills = sorted(
    resume_skill_map.values(),
    key=str.lower
)

print(
    f"Skills detected: "
    f"{len(resume_skills):,}"
)

print(
    "Detected skills:"
)

print(
    ", ".join(
        resume_skills
    )
)


# ======================================================================
# RESUME EMBEDDING
# ======================================================================

print_header(
    "GENERATING RESUME SEMANTIC EMBEDDING"
)

resume_embedding = model.encode(
    [resume_text],
    convert_to_numpy=True,
    normalize_embeddings=True
)[0]

resume_embedding = np.asarray(
    resume_embedding,
    dtype=np.float32
)

print(
    "Resume embedding generated successfully."
)

print(
    f"Embedding dimension: "
    f"{resume_embedding.shape[0]}"
)


# ======================================================================
# SCORE AND RANK JOBS (shared scoring definition)
# ======================================================================

print_header(
    "SCORING AND RANKING JOBS"
)

ranked = rank_resume(
    corpus,
    resume_text=resume_text,
    resume_embedding=resume_embedding,
    resume_skills=resume_skill_map.keys(),
    scorer=scorer
)

scoring_method = (
    ranked["scoring_method"].iloc[0]
    if len(ranked)
    else scorer.method
)

print(
    f"Scoring method: {scoring_method}"
)

results = []

for _, scored in ranked.iterrows():

    row_index = int(
        scored["row_index"]
    )

    row = corpus.frame.iloc[
        row_index
    ]

    matched_skills, skill_gaps = skill_breakdown(
        corpus,
        row_index,
        resume_skill_map
    )

    results.append(
        {
            "job_id": scored["job_id"],

            "job_title": normalize_text(
                row[job_title_col]
            ),

            "job_description": normalize_text(
                row[job_description_col]
            ),

            "company": (
                normalize_text(
                    row[job_company_col]
                )
                if job_company_col
                else ""
            ),

            "location": (
                normalize_text(
                    row[job_location_col]
                )
                if job_location_col
                else ""
            ),

            "semantic_similarity":
                float(scored["semantic_similarity"]),

            "skill_match_signal":
                float(scored["skill_match_signal"]),

            "title_similarity":
                float(scored["title_similarity"]),

            "text_containment":
                float(scored["text_containment"]),

            "shared_skill_count":
                int(scored["shared_skill_count"]),

            "matched_skills":
                json.dumps(
                    matched_skills,
                    ensure_ascii=False
                ),

            "skill_gaps":
                json.dumps(
                    skill_gaps,
                    ensure_ascii=False
                ),

            "hybrid_match_score":
                float(scored["hybrid_match_score"]),

            "scoring_method":
                scored["scoring_method"],

            "recommendation_rank":
                int(scored["recommendation_rank"]),
        }
    )


# ======================================================================
# RANK JOBS
# ======================================================================

print_header(
    "RANKING JOBS"
)

recommendations = pd.DataFrame(
    results
)

# rank_jobs already returned the jobs best-first with a deterministic
# tie-break; recommendation_rank is its rank column.
recommendations[
    "is_top10"
] = (
    recommendations[
        "recommendation_rank"
    ]
    <= TOP_K
)


# ======================================================================
# TOP-K
# ======================================================================

top10 = (
    recommendations
    .head(TOP_K)
    .copy()
)


print(
    f"Top-{TOP_K} jobs selected."
)

print(
    f"Top score: "
    f"{top10['hybrid_match_score'].max():.4f}"
)

print(
    f"Top-{TOP_K} mean score: "
    f"{top10['hybrid_match_score'].mean():.4f}"
)


# ======================================================================
# EXPLANATIONS
# ======================================================================

print_header(
    "GENERATING TOP-K EXPLANATIONS"
)

explanations = []

for _, row in top10.iterrows():

    matched_skills = parse_list_value(
        row["matched_skills"]
    )

    skill_gaps = parse_list_value(
        row["skill_gaps"]
    )

    explanation = (
        generate_groq_explanation(
            resume_text=resume_text,
            job_title=row["job_title"],
            job_description=row[
                "job_description"
            ],
            score=safe_float(
                row[
                    "hybrid_match_score"
                ]
            ),
            semantic_similarity=safe_float(
                row[
                    "semantic_similarity"
                ]
            ),
            matched_skills=matched_skills,
            skill_gaps=skill_gaps
        )
    )

    explanations.append(
        explanation
    )


# ======================================================================
# ATTACH EXPLANATIONS
# ======================================================================

top10[
    "why_match"
] = [
    x.get(
        "why_match",
        ""
    )
    for x in explanations
]

top10[
    "rag_matched_evidence"
] = [
    json.dumps(
        x.get(
            "matched_evidence",
            []
        ),
        ensure_ascii=False
    )
    for x in explanations
]

top10[
    "llm_skill_gaps"
] = [
    json.dumps(
        x.get(
            "skill_gaps",
            []
        ),
        ensure_ascii=False
    )
    for x in explanations
]

top10[
    "recommendation_summary"
] = [
    x.get(
        "recommendation_summary",
        ""
    )
    for x in explanations
]

top10[
    "limitations"
] = [
    x.get(
        "limitations",
        ""
    )
    for x in explanations
]

top10[
    "explanation_source"
] = [
    x.get(
        "explanation_source",
        "fallback"
    )
    for x in explanations
]

top10[
    "llm_error"
] = [
    x.get(
        "llm_error",
        ""
    )
    for x in explanations
]


# ======================================================================
# SAVE ALL RECOMMENDATIONS
# ======================================================================

print_header(
    "SAVING RECOMMENDATIONS"
)

recommendations.to_csv(
    OUTPUT_ALL,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"All recommendations saved:\n"
    f"{OUTPUT_ALL}"
)


# ======================================================================
# SAVE TOP 10
# ======================================================================

top10.to_csv(
    OUTPUT_TOP10,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"Top-{TOP_K} recommendations saved:\n"
    f"{OUTPUT_TOP10}"
)


# ======================================================================
# USER-FACING JSON
# ======================================================================

top_jobs_for_json = []

for _, row in top10.iterrows():

    top_jobs_for_json.append(
        {
            "rank": int(
                row[
                    "recommendation_rank"
                ]
            ),

            "job_id": str(
                row[
                    "job_id"
                ]
            ),

            "job_title": row[
                "job_title"
            ],

            "company": row[
                "company"
            ],

            "location": row[
                "location"
            ],

            "match_score": round(
                safe_float(
                    row[
                        "hybrid_match_score"
                    ]
                ),
                4
            ),

            "scoring_method":
                str(
                    row[
                        "scoring_method"
                    ]
                ),

            "semantic_similarity": round(
                safe_float(
                    row[
                        "semantic_similarity"
                    ]
                ),
                4
            ),

            "matched_skills":
                parse_list_value(
                    row[
                        "matched_skills"
                    ]
                ),

            "skill_gaps":
                parse_list_value(
                    row[
                        "skill_gaps"
                    ]
                ),

            "why_match": row[
                "why_match"
            ],

            "recommendation_summary":
                row[
                    "recommendation_summary"
                ],

            "limitations": row[
                "limitations"
            ],

            "explanation_source":
                row[
                    "explanation_source"
                ]
        }
    )


analysis = {
    "project": "AI Resume Analyzer",

    "step": 13,

    "status": "COMPLETE",

    "input": {
        "resume_file": str(
            resume_path
        ),

        "resume_characters":
            len(resume_text),

        "detected_skills":
            len(resume_skills),

        "skill_list":
            resume_skills,

        "pdf_extraction":
            extraction.to_record()
    },

    "job_dataset": {
        "source": str(
            JOB_PATH
        ),

        "total_jobs":
            int(
                len(jobs)
            )
    },

    "semantic_model": {
        "name":
            SEMANTIC_MODEL_NAME,

        "embedding_dimension":
            int(
                resume_embedding.shape[0]
            ),

        "job_embeddings_cache":
            str(JOB_EMBEDDING_CACHE)
    },

    "skill_vocabulary": {
        "source":
            vocabulary.source if vocabulary else None,

        "skills_table_only":
            True,

        "skill_count":
            vocabulary.skill_count if vocabulary else 0
    },

    "matching": {
        "ranking_signal":
            "hybrid_match_score from matching_core.MatchScorer",

        "scoring_method":
            scoring_method,

        "scorer":
            scorer.description,

        "claim_label": scorer.claim_label,
        "claim_description": scorer.claim_description,
        "claim": scorer.claim.to_dict(),
        "suitability_wording_allowed": scorer.suitability_wording_allowed,
        "scorer_fingerprint": scorer.scorer_fingerprint,
        "candidate_policy": "semantic_top_k_union_lexical_top_k_union_esco_skill_top_k",
        "candidate_policy_fingerprint": candidate_retrieval.DEFAULT_POLICY.fingerprint,
        "candidates_scored": int(len(recommendations)),

        "trained_model_in_use":
            bool(scorer.uses_model),

        "formula_weights":
            mc.WEIGHTS,

        "category_weight_redistributed":
            True,

        "ranking_changed_by_llm":
            False,

        "score_created_by_llm":
            False
    },

    "recommendations": {
        "total_ranked_jobs":
            int(
                len(recommendations)
            ),

        "top_k":
            TOP_K,

        "top_k_mean_score":
            float(
                top10[
                    "hybrid_match_score"
                ].mean()
            ),

        "top_jobs":
            top_jobs_for_json
    },

    "llm": {
        "provider":
            "Groq"
            if groq_client
            else "deterministic fallback",

        "model":
            GROQ_MODEL
            if groq_client
            else None,

        "used_only_for_explanation":
            True,

        "used_for_ranking":
            False,

        "used_for_scoring":
            False
    },

    "methodology": {
        "pseudo_labels":
            False,

        "ground_truth_hiring_labels":
            False,

        "hiring_probability_claimed":
            False,

        "manual_annotation_required":
            not scorer.suitability_wording_allowed
    },

    "outputs": {
        "all_recommendations":
            str(
                OUTPUT_ALL
            ),

        "top10_recommendations":
            str(
                OUTPUT_TOP10
            ),

        "analysis":
            str(
                OUTPUT_ANALYSIS
            ),

        "report":
            str(
                REPORT_PATH
            )
    }
}


with open(
    OUTPUT_ANALYSIS,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        analysis,
        file,
        indent=2,
        ensure_ascii=False
    )


# ======================================================================
# REPORT
# ======================================================================

report = {
    "step": 13,

    "name":
        "User-Facing Explainable Resume Analyzer",

    "status":
        "COMPLETE",

    "resume":
        str(resume_path),

    "resume_characters":
        len(resume_text),

    "detected_skills":
        len(resume_skills),

    "jobs_evaluated":
        len(jobs),

    "top_k":
        TOP_K,

    "semantic_model":
        SEMANTIC_MODEL_NAME,

    "job_embedding_cache":
        str(
            JOB_EMBEDDING_CACHE
        ),

    "job_corpus_fingerprint":
        corpus.fingerprint,

    "ranking": {
        "method":
            scoring_method,

        "scorer":
            scorer.description,

        "claim_label": scorer.claim_label,
        "claim_description": scorer.claim_description,
        "claim": scorer.claim.to_dict(),
        "suitability_wording_allowed": scorer.suitability_wording_allowed,
        "scorer_fingerprint": scorer.scorer_fingerprint,
        "candidate_policy": "semantic_top_k_union_lexical_top_k_union_esco_skill_top_k",
        "candidate_policy_fingerprint": candidate_retrieval.DEFAULT_POLICY.fingerprint,

        "trained_model_in_use":
            bool(scorer.uses_model),

        "formula_weights_if_no_model":
            mc.WEIGHTS,

        "shared_with_streamlit_app":
            True
    },

    "pdf_extraction":
        extraction.to_record(),

    "llm": {
        "provider":
            "Groq"
            if groq_client
            else "fallback",

        "model":
            GROQ_MODEL
            if groq_client
            else None,

        "ranking":
            False,

        "scoring":
            False,

        "explanation":
            True
    },

    "validation": {
        "jobs_ranked":
            len(
                recommendations
            ),

        "top10_rows":
            len(
                top10
            ),

        "duplicate_job_ids":
            int(
                recommendations[
                    "job_id"
                ]
                .duplicated()
                .sum()
            ),

        "empty_explanations":
            int(
                top10[
                    "why_match"
                ]
                .fillna("")
                .str.strip()
                .eq("")
                .sum()
            )
    },

    "outputs": {
        "all":
            str(
                OUTPUT_ALL
            ),

        "top10":
            str(
                OUTPUT_TOP10
            ),

        "analysis":
            str(
                OUTPUT_ANALYSIS
            )
    }
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


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print_header(
    "STEP 13 COMPLETE"
)

print(
    f"Resume: {resume_path}"
)

print(
    f"Resume characters: "
    f"{len(resume_text):,}"
)

print(
    f"Skills detected: "
    f"{len(resume_skills):,}"
)

print(
    f"Jobs evaluated: "
    f"{len(jobs):,}"
)

print(
    f"Top-{TOP_K} recommendations: "
    f"{len(top10):,}"
)

print(
    f"Top score: "
    f"{top10['hybrid_match_score'].max():.4f}"
)

print(
    f"Mean Top-{TOP_K} score: "
    f"{top10['hybrid_match_score'].mean():.4f}"
)

print(
    f"Scoring method: {scoring_method}"
)

print(
    "LLM used for ranking: NO"
)

print(
    "LLM used for scoring: NO"
)

print(
    "LLM used for explanation: "
    + (
        "YES"
        if groq_client
        else "NO - deterministic fallback"
    )
)

print(
    "Human relevance validation passed: "
    + ("YES" if scorer.suitability_wording_allowed else "NO - results remain experimental similarity")
)

print(
    "Pseudo-labels: NO"
)

print(
    "Ground-truth hiring labels: NO"
)

print(
    "Hiring probability claimed: NO"
)

print(
    f"\nAll recommendations:\n"
    f"{OUTPUT_ALL}"
)

print(
    f"\nTop-{TOP_K} recommendations:\n"
    f"{OUTPUT_TOP10}"
)

print(
    f"\nUser-facing analysis:\n"
    f"{OUTPUT_ANALYSIS}"
)

print(
    f"\nReport:\n"
    f"{REPORT_PATH}"
)

print(
    "\nNEXT: STEP 14 - USER-FACING WEB APPLICATION"
)