"""
======================================================================
AI RESUME ANALYZER - STEP 13
USER-FACING EXPLAINABLE RAG RECOMMENDATION PIPELINE
======================================================================

PURPOSE
-------
This step is the actual uploaded-resume analysis pipeline.

USER INPUT
----------
data/test_resume.pdf

PIPELINE
--------
PDF Resume
    ↓
Resume Text Extraction
    ↓
Skill Extraction
    ↓
Resume Semantic Embedding
    ↓
Actual Job Dataset
    ↓
Cached Job Embeddings
    ↓
Semantic Similarity
    ↓
Skill Matching
    ↓
Hybrid Recommendation Score
    ↓
Job Ranking
    ↓
Top-K Jobs
    ↓
Groq LLM Explanation
    ↓
User-Facing Recommendation Output

IMPORTANT
---------
- User uploads ONE resume.
- No manual annotation.
- No pseudo-labels.
- No ground-truth hiring labels.
- No hiring probability.
- LLM does NOT determine ranking.
- LLM does NOT create the matching score.
- LLM only explains already-computed matching evidence.
- Actual resume text is used.
- Actual job title and description are used.
- Skill gaps are calculated from actual available evidence.
- Groq failure does NOT stop the pipeline.
- Deterministic fallback explanations are used if Groq is unavailable.

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
data/cache/job_embeddings.npy

DEFAULT MODEL
-------------
all-MiniLM-L6-v2

GROQ MODEL
----------
openai/gpt-oss-20b
======================================================================
"""

import os
import re
import json
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = Path(
    r"C:\Users\jasme\Desktop\ai resume analyzer"
)

DEFAULT_RESUME_PATH = (
    BASE_DIR
    / "data"
    / "test_resume.pdf"
)

JOB_PATH = (
    BASE_DIR
    / "data"
    / "jobs"
    / "all_job_post.csv"
)

CACHE_DIR = (
    BASE_DIR
    / "data"
    / "cache"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "recommendations"
)

REPORT_DIR = (
    BASE_DIR
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
    "all-MiniLM-L6-v2"
)

TOP_K = 10

JOB_EMBED_BATCH_SIZE = 32

MAX_RESUME_CHARS_FOR_LLM = 6000

MAX_JOB_CHARS_FOR_LLM = 5000

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)


# ======================================================================
# LOAD ENVIRONMENT
# ======================================================================

load_dotenv(
    BASE_DIR / ".env"
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


def normalize_signal(
    value,
    minimum=0.0,
    maximum=1.0
):

    value = safe_float(
        value
    )

    if maximum <= minimum:
        return 0.0

    value = (
        value - minimum
    ) / (
        maximum - minimum
    )

    return float(
        np.clip(
            value,
            0.0,
            1.0
        )
    )


def tokenize_text(text):

    text = normalize_text(
        text
    ).lower()

    tokens = re.findall(
        r"[a-zA-Z][a-zA-Z0-9+#.\-]*",
        text
    )

    return set(tokens)


def extract_skills_from_text(
    text,
    vocabulary
):

    text_lower = normalize_text(
        text
    ).lower()

    detected = []

    for skill in vocabulary:

        skill_clean = normalize_text(
            skill
        )

        if not skill_clean:
            continue

        skill_lower = (
            skill_clean.lower()
        )

        pattern = (
            r"(?<![a-zA-Z0-9])"
            + re.escape(skill_lower)
            + r"(?![a-zA-Z0-9])"
        )

        if re.search(
            pattern,
            text_lower
        ):

            detected.append(
                skill_clean
            )

    return clean_list(
        detected
    )


# ======================================================================
# PDF EXTRACTION
# ======================================================================

def extract_pdf_text(
    pdf_path
):

    try:

        import pymupdf

        document = pymupdf.open(
            str(pdf_path)
        )

        pages = []

        for page in document:

            pages.append(
                page.get_text()
            )

        document.close()

        return normalize_text(
            "\n".join(pages)
        )

    except ImportError:

        try:

            import fitz

            document = fitz.open(
                str(pdf_path)
            )

            pages = []

            for page in document:

                pages.append(
                    page.get_text()
                )

            document.close()

            return normalize_text(
                "\n".join(pages)
            )

        except Exception as exc:

            raise RuntimeError(
                f"Could not extract PDF text: {exc}"
            )

    except Exception as exc:

        raise RuntimeError(
            f"Could not extract PDF text: {exc}"
        )


# ======================================================================
# JOB TEXT
# ======================================================================

def build_job_text(
    title,
    description
):

    title = normalize_text(
        title
    )

    description = normalize_text(
        description
    )

    return (
        title
        + ". "
        + description
    ).strip()


# ======================================================================
# JOB SKILL VOCABULARY
# ======================================================================

def build_skill_vocabulary(
    jobs,
    title_col,
    description_col
):

    print_header(
        "Building skill vocabulary"
    )

    # A practical built-in vocabulary.
    # This supplements terms already present in the job dataset.

    base_skills = [
        "python",
        "java",
        "c",
        "c++",
        "c#",
        "javascript",
        "typescript",
        "html",
        "css",
        "sql",
        "mysql",
        "postgresql",
        "mongodb",
        "oracle",
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow",
        "pytorch",
        "keras",
        "opencv",
        "yolo",
        "machine learning",
        "deep learning",
        "computer vision",
        "natural language processing",
        "nlp",
        "generative ai",
        "genai",
        "llm",
        "large language model",
        "rag",
        "retrieval augmented generation",
        "transformers",
        "hugging face",
        "github",
        "git",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "api",
        "rest api",
        "flask",
        "django",
        "fastapi",
        "react",
        "node.js",
        "node",
        "express",
        "angular",
        "vue",
        "data analysis",
        "data science",
        "statistics",
        "power bi",
        "tableau",
        "excel",
        "communication",
        "problem solving",
        "teamwork",
        "leadership",
        "project management",
        "agile",
        "scrum"
    ]

    vocabulary = set(
        base_skills
    )

    for _, row in jobs.iterrows():

        text = build_job_text(
            row[title_col],
            row[description_col]
        ).lower()

        for skill in base_skills:

            if skill.lower() in text:

                vocabulary.add(
                    skill
                )

    vocabulary = sorted(
        vocabulary,
        key=lambda x: (
            -len(x),
            x.lower()
        )
    )

    print(
        f"Skill vocabulary size: "
        f"{len(vocabulary):,}"
    )

    return vocabulary


# ======================================================================
# JOB EMBEDDINGS
# ======================================================================

def create_or_load_job_embeddings(
    jobs,
    job_texts,
    model
):

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    expected_count = len(
        job_texts
    )

    if JOB_EMBEDDING_CACHE.exists():

        try:

            cached = np.load(
                JOB_EMBEDDING_CACHE
            )

            if (
                cached.ndim == 2
                and
                cached.shape[0]
                == expected_count
                and
                cached.shape[1]
                == 384
            ):

                print(
                    "Using cached job embeddings."
                )

                print(
                    f"Shape: {cached.shape}"
                )

                return cached

            print(
                "Existing job embedding cache "
                "does not match current dataset."
            )

        except Exception as exc:

            print(
                "Could not load job embedding cache."
            )

            print(
                f"Reason: {exc}"
            )

    print_header(
        "Creating job embedding cache"
    )

    print(
        f"Embedding {expected_count:,} jobs..."
    )

    embeddings = model.encode(
        job_texts,
        batch_size=JOB_EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    np.save(
        JOB_EMBEDDING_CACHE,
        embeddings
    )

    print(
        "Job embeddings cached at:"
    )

    print(
        JOB_EMBEDDING_CACHE
    )

    return embeddings


# ======================================================================
# SKILL MATCHING
# ======================================================================

def calculate_skill_match(
    resume_skills,
    job_text,
    vocabulary
):

    resume_set = {
        x.lower()
        for x in resume_skills
    }

    job_skills = extract_skills_from_text(
        job_text,
        vocabulary
    )

    job_set = {
        x.lower()
        for x in job_skills
    }

    matched = sorted(
        resume_set.intersection(
            job_set
        )
    )

    gaps = sorted(
        job_set.difference(
            resume_set
        )
    )

    if job_set:

        coverage = (
            len(matched)
            /
            len(job_set)
        )

    else:

        coverage = 0.0

    return (
        matched,
        gaps,
        float(
            np.clip(
                coverage,
                0.0,
                1.0
            )
        ),
        job_skills
    )


# ======================================================================
# TITLE SIMILARITY
# ======================================================================

def calculate_title_signal(
    resume_text,
    job_title
):

    resume_tokens = tokenize_text(
        resume_text
    )

    title_tokens = tokenize_text(
        job_title
    )

    if not title_tokens:

        return 0.0

    overlap = (
        resume_tokens
        &
        title_tokens
    )

    return float(
        len(overlap)
        /
        len(title_tokens)
    )


# ======================================================================
# HYBRID SCORE
# ======================================================================

def calculate_hybrid_score(
    semantic_similarity,
    skill_signal,
    title_signal,
    category_signal
):

    semantic = normalize_signal(
        semantic_similarity,
        -1.0,
        1.0
    )

    skill = normalize_signal(
        skill_signal
    )

    title = normalize_signal(
        title_signal
    )

    category = normalize_signal(
        category_signal
    )

    # Transparent weighted aggregation.
    #
    # Semantic similarity is the strongest general signal.
    # Skill matching provides explicit evidence.
    # Title similarity adds role-level relevance.
    # Category signal is retained when available.

    score = (
        0.50 * semantic
        +
        0.30 * skill
        +
        0.15 * title
        +
        0.05 * category
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


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


# ======================================================================
# RESUME INPUT
# ======================================================================

resume_path = DEFAULT_RESUME_PATH

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

resume_text = extract_pdf_text(
    resume_path
)

if not resume_text:

    raise ValueError(
        "No text could be extracted from the PDF."
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


job_id_col = find_column(
    jobs,
    ["job_id"]
)

job_title_col = find_column(
    jobs,
    [
        "job_title",
        "title",
        "job"
    ]
)

job_description_col = find_column(
    jobs,
    [
        "job_description",
        "description",
        "job_desc"
    ]
)

job_company_col = find_column(
    jobs,
    [
        "company",
        "company_name",
        "employer"
    ],
    required=False
)

job_location_col = find_column(
    jobs,
    [
        "location",
        "job_location",
        "city"
    ],
    required=False
)


# ======================================================================
# BUILD JOB TEXT
# ======================================================================

job_texts = []

for _, row in jobs.iterrows():

    job_texts.append(
        build_job_text(
            row[job_title_col],
            row[job_description_col]
        )
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


# ======================================================================
# JOB EMBEDDINGS
# ======================================================================

job_embeddings = (
    create_or_load_job_embeddings(
        jobs,
        job_texts,
        model
    )
)


# ======================================================================
# SKILL VOCABULARY
# ======================================================================

skill_vocabulary = (
    build_skill_vocabulary(
        jobs,
        job_title_col,
        job_description_col
    )
)


# ======================================================================
# RESUME SKILLS
# ======================================================================

print_header(
    "EXTRACTING RESUME SKILLS"
)

resume_skills = (
    extract_skills_from_text(
        resume_text,
        skill_vocabulary
    )
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
# SEMANTIC SIMILARITY
# ======================================================================

print_header(
    "CALCULATING SEMANTIC JOB SIMILARITY"
)

semantic_scores = (
    job_embeddings
    @
    resume_embedding
)

semantic_scores = np.asarray(
    semantic_scores,
    dtype=np.float32
)


# ======================================================================
# MATCH JOBS
# ======================================================================

print_header(
    "CALCULATING SKILL MATCHES AND GAPS"
)

results = []

for index, row in jobs.iterrows():

    job_id = row[
        job_id_col
    ]

    job_title = normalize_text(
        row[job_title_col]
    )

    job_description = normalize_text(
        row[job_description_col]
    )

    job_text = build_job_text(
        job_title,
        job_description
    )

    semantic_similarity = (
        safe_float(
            semantic_scores[index]
        )
    )

    matched_skills, skill_gaps, skill_coverage, job_skills = (
        calculate_skill_match(
            resume_skills,
            job_text,
            skill_vocabulary
        )
    )

    title_signal = (
        calculate_title_signal(
            resume_text,
            job_title
        )
    )

    # Category signal is intentionally neutral when the
    # job dataset does not provide a reliable category field.
    category_signal = 0.0

    hybrid_score = (
        calculate_hybrid_score(
            semantic_similarity,
            skill_coverage,
            title_signal,
            category_signal
        )
    )

    results.append(
        {
            "job_id": job_id,
            "job_title": job_title,
            "job_description": job_description,

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
                semantic_similarity,

            "skill_match_signal":
                skill_coverage,

            "title_similarity":
                title_signal,

            "category_match_signal":
                category_signal,

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
                hybrid_score
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

recommendations = (
    recommendations
    .sort_values(
        "hybrid_match_score",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

recommendations[
    "recommendation_rank"
] = (
    np.arange(
        1,
        len(recommendations) + 1
    )
)

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
            resume_skills
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

        "job_embeddings_cached":
            True
    },

    "matching": {
        "ranking_signal":
            "transparent hybrid score",

        "semantic_weight":
            0.50,

        "skill_weight":
            0.30,

        "title_weight":
            0.15,

        "category_weight":
            0.05,

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
            False
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

    "ranking": {
        "method":
            "explicit weighted hybrid aggregation",

        "semantic_weight":
            0.50,

        "skill_weight":
            0.30,

        "title_weight":
            0.15,

        "category_weight":
            0.05
    },

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
    "Manual annotation required: NO"
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