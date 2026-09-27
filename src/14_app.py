"""
======================================================================
AI RESUME ANALYZER - STREAMLIT APPLICATION
======================================================================

USER FLOW
---------
Upload Resume
    ↓
Extract Resume Text
    ↓
Load External Skill Vocabulary
    ↓
Extract Resume Skills
    ↓
Generate Resume Semantic Embedding
    ↓
Load Cached Job Embeddings
    ↓
Load / Build Cached Job Skill Evidence
    ↓
Calculate Semantic Similarity
    ↓
Calculate Skill Matches / Gaps
    ↓
Hybrid Ranking
    ↓
Top-K Jobs
    ↓
Groq Explanation for Top-K only
    ↓
User-Friendly Resume Analysis

IMPORTANT
---------
This application does NOT require:
    resume_analyzer_backend.py

It directly uses:
    data/jobs/all_job_post.csv
    data/cache/job_embeddings.npy
    data/external/*

Caches:
    data/cache/job_embeddings.npy
    data/cache/job_skill_evidence.json

Run:
    python -m streamlit run src/14_app.py
======================================================================
"""

from __future__ import annotations

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

JOB_DATA_PATH = DATA_DIR / "jobs" / "all_job_post.csv"

CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

JOB_EMBEDDING_CACHE = CACHE_DIR / "job_embeddings.npy"
JOB_SKILL_CACHE = CACHE_DIR / "job_skill_evidence.json"

EXTERNAL_DIR = DATA_DIR / "external"

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


# ======================================================================
# CONFIGURATION
# ======================================================================

MODEL_NAME = "all-MiniLM-L6-v2"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

DEFAULT_TOP_K = 5


# ======================================================================
# PAGE CONFIG
# ======================================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================================
# CSS
# ======================================================================

st.markdown(
    """
<style>

.main-title {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #9ca3af;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}

.section-title {
    font-size: 1.7rem;
    font-weight: 700;
    margin-top: 1.5rem;
}

.skill {
    display: inline-block;
    padding: 6px 12px;
    margin: 4px;
    border-radius: 20px;
    background: #172554;
    color: #bfdbfe;
    font-size: 0.9rem;
}

.gap {
    display: inline-block;
    padding: 6px 12px;
    margin: 4px;
    border-radius: 20px;
    background: #451a03;
    color: #fed7aa;
    font-size: 0.9rem;
}

.job-card {
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
}

.explanation {
    background: #111827;
    border-left: 4px solid #60a5fa;
    padding: 15px;
    border-radius: 8px;
}

.small-muted {
    color: #9ca3af;
    font-size: 0.85rem;
}

</style>
""",
    unsafe_allow_html=True,
)


# ======================================================================
# UTILITY
# ======================================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_text(text: str) -> str:
    """
    Normalize text for skill matching.
    """
    text = clean_text(text).lower()

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("&", " and ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_skill(skill: str) -> str:
    skill = normalize_text(skill)

    skill = skill.replace("node js", "node.js")
    skill = skill.replace("scikit learn", "scikit-learn")
    skill = skill.replace("machine-learning", "machine learning")
    skill = skill.replace("deep-learning", "deep learning")
    skill = skill.replace("natural-language-processing", "natural language processing")

    return skill.strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def score_percent(value: float) -> str:
    value = max(0.0, min(1.0, safe_float(value)))
    return f"{value * 100:.1f}%"


# ======================================================================
# PDF EXTRACTION
# ======================================================================

@st.cache_data(show_spinner=False)
def extract_pdf_text(file_bytes: bytes) -> str:

    document = pymupdf.open(
        stream=file_bytes,
        filetype="pdf"
    )

    pages = []

    for page in document:
        text = page.get_text("text")

        if text:
            pages.append(text)

    document.close()

    text = "\n".join(pages)

    return text.strip()


# ======================================================================
# JOB DATASET
# ======================================================================

@st.cache_data(show_spinner=False)
def load_jobs() -> pd.DataFrame:

    if not JOB_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Job dataset not found:\n{JOB_DATA_PATH}"
        )

    df = pd.read_csv(
        JOB_DATA_PATH,
        low_memory=False
    )

    df = df.fillna("")

    return df


# ======================================================================
# COLUMN DETECTION
# ======================================================================

def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = False
) -> str | None:

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        candidate = candidate.lower()

        if candidate in normalized:
            return normalized[candidate]

    for col in df.columns:

        col_lower = str(col).lower()

        for candidate in candidates:

            if candidate.lower() in col_lower:
                return col

    if required:
        raise ValueError(
            f"Could not find required column.\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def identify_job_columns(df: pd.DataFrame):

    title_col = find_column(
        df,
        [
            "job_title",
            "title",
            "jobtitle",
            "position",
            "job"
        ],
        required=True
    )

    description_col = find_column(
        df,
        [
            "job_description",
            "description",
            "jobdescription",
            "details",
            "text"
        ],
        required=True
    )

    company_col = find_column(
        df,
        [
            "company",
            "company_name",
            "employer"
        ]
    )

    location_col = find_column(
        df,
        [
            "location",
            "job_location",
            "city"
        ]
    )

    id_col = find_column(
        df,
        [
            "job_id",
            "id",
            "jobid"
        ]
    )

    return (
        id_col,
        title_col,
        description_col,
        company_col,
        location_col
    )


# ======================================================================
# JOB TEXT
# ======================================================================

def build_job_text(
    row: pd.Series,
    title_col: str,
    description_col: str
) -> str:

    title = clean_text(row[title_col])
    description = clean_text(row[description_col])

    return f"{title}. {description}".strip()


# ======================================================================
# SEMANTIC MODEL
# ======================================================================

@st.cache_resource(show_spinner=False)
def load_semantic_model():

    return SentenceTransformer(
        MODEL_NAME
    )


# ======================================================================
# JOB EMBEDDINGS
# ======================================================================

def create_job_embeddings(
    jobs: pd.DataFrame,
    title_col: str,
    description_col: str,
    model: SentenceTransformer
):

    texts = [
        build_job_text(
            row,
            title_col,
            description_col
        )
        for _, row in jobs.iterrows()
    ]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embeddings.astype(
        np.float32
    )


def load_job_embeddings(
    jobs: pd.DataFrame,
    title_col: str,
    description_col: str,
    model: SentenceTransformer
):

    if JOB_EMBEDDING_CACHE.exists():

        try:

            embeddings = np.load(
                JOB_EMBEDDING_CACHE
            )

            if (
                embeddings.shape[0] == len(jobs)
                and embeddings.shape[1] == 384
            ):
                return embeddings

        except Exception:
            pass

    embeddings = create_job_embeddings(
        jobs,
        title_col,
        description_col,
        model
    )

    np.save(
        JOB_EMBEDDING_CACHE,
        embeddings
    )

    return embeddings


# ======================================================================
# EXTERNAL SKILL VOCABULARY
# ======================================================================

def locate_external_csvs() -> list[Path]:

    if not EXTERNAL_DIR.exists():
        return []

    return list(
        EXTERNAL_DIR.rglob("*.csv")
    )


def find_skill_columns(
    df: pd.DataFrame
) -> list[str]:

    candidates = [
        "preferredLabel",
        "preferred_label",
        "skill",
        "skill_name",
        "skillName",
        "name",
        "label",
        "title",
        "conceptUri",
    ]

    result = []

    for col in df.columns:

        col_lower = str(col).lower()

        for candidate in candidates:

            if candidate.lower() == col_lower:
                result.append(col)
                break

    return result


@st.cache_data(show_spinner=False)
def load_external_skill_vocabulary():

    csv_files = locate_external_csvs()

    if not csv_files:
        return []

    skills = set()

    for csv_file in csv_files:

        try:

            df = pd.read_csv(
                csv_file,
                low_memory=False
            )

        except Exception:
            continue

        columns = find_skill_columns(
            df
        )

        for col in columns:

            for value in df[col].tolist():

                value = clean_text(value)

                if not value:
                    continue

                normalized = normalize_skill(
                    value
                )

                # Avoid useless one-character entries
                # except known programming language C.
                if len(normalized) == 1:
                    if normalized != "c":
                        continue

                if len(normalized) < 2:
                    continue

                skills.add(
                    normalized
                )

    return sorted(
        skills,
        key=lambda x: (-len(x), x)
    )


# ======================================================================
# SKILL MATCHING
# ======================================================================

def create_skill_pattern(
    skills: list[str]
):

    usable = []

    for skill in skills:

        escaped = re.escape(
            normalize_skill(skill)
        )

        usable.append(
            escaped
        )

    if not usable:
        return None

    # Longest first prevents short skills from stealing matches
    usable.sort(
        key=len,
        reverse=True
    )

    pattern = (
        r"(?<!\w)(?:"
        + "|".join(usable)
        + r")(?!\w)"
    )

    try:
        return re.compile(
            pattern,
            re.IGNORECASE
        )
    except re.error:
        return None


@st.cache_resource(show_spinner=False)
def get_skill_pattern(
    vocabulary_tuple: tuple[str, ...]
):

    return create_skill_pattern(
        list(vocabulary_tuple)
    )


def extract_resume_skills(
    resume_text: str,
    vocabulary: list[str]
):

    if not resume_text or not vocabulary:
        return []

    pattern = get_skill_pattern(
        tuple(vocabulary)
    )

    if pattern is None:
        return []

    normalized_resume = normalize_text(
        resume_text
    )

    found = set()

    for match in pattern.finditer(
        normalized_resume
    ):

        skill = normalize_skill(
            match.group(0)
        )

        found.add(
            skill
        )

    return sorted(
        found
    )


# ======================================================================
# JOB SKILL EVIDENCE
# ======================================================================

def build_job_skill_evidence_once(
    jobs: pd.DataFrame,
    title_col: str,
    description_col: str,
    vocabulary: list[str]
):

    """
    IMPORTANT:
    This function is called ONLY when the cache does not exist.

    It creates skill evidence for all jobs in one preprocessing pass.
    """

    pattern = create_skill_pattern(
        vocabulary
    )

    evidence = {}

    if pattern is None:
        return evidence

    for index, row in jobs.iterrows():

        job_text = normalize_text(
            build_job_text(
                row,
                title_col,
                description_col
            )
        )

        matched = set()

        for match in pattern.finditer(
            job_text
        ):

            skill = normalize_skill(
                match.group(0)
            )

            matched.add(
                skill
            )

        evidence[str(index)] = sorted(
            matched
        )

    return evidence


def load_or_build_job_skill_evidence(
    jobs: pd.DataFrame,
    title_col: str,
    description_col: str,
    vocabulary: list[str]
):

    # --------------------------------------------------------------
    # FAST PATH
    # --------------------------------------------------------------

    if JOB_SKILL_CACHE.exists():

        try:

            with open(
                JOB_SKILL_CACHE,
                "r",
                encoding="utf-8"
            ) as f:

                cached = json.load(f)

            if (
                cached.get("job_count")
                == len(jobs)
                and cached.get("skill_count")
                == len(vocabulary)
            ):

                return cached["evidence"]

        except Exception:
            pass

    # --------------------------------------------------------------
    # FIRST-TIME BUILD
    # --------------------------------------------------------------

    progress = st.progress(
        0
    )

    status = st.empty()

    status.info(
        "Preparing job skill evidence for the first run..."
    )

    evidence = {}

    pattern = create_skill_pattern(
        vocabulary
    )

    total = len(jobs)

    if pattern is None:

        return {}

    for position, (_, row) in enumerate(
        jobs.iterrows()
    ):

        job_text = normalize_text(
            build_job_text(
                row,
                title_col,
                description_col
            )
        )

        matched = set()

        for match in pattern.finditer(
            job_text
        ):

            matched.add(
                normalize_skill(
                    match.group(0)
                )
            )

        evidence[str(position)] = sorted(
            matched
        )

        if position % 25 == 0:

            progress.progress(
                min(
                    1.0,
                    (position + 1) / total
                )
            )

    progress.progress(
        1.0
    )

    # --------------------------------------------------------------
    # SAVE CACHE
    # --------------------------------------------------------------

    payload = {
        "job_count": len(jobs),
        "skill_count": len(vocabulary),
        "evidence": evidence
    }

    with open(
        JOB_SKILL_CACHE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            payload,
            f,
            ensure_ascii=False
        )

    status.success(
        "Job skill evidence cached successfully."
    )

    return evidence


# ======================================================================
# RESUME EMBEDDING
# ======================================================================

def generate_resume_embedding(
    resume_text: str,
    model: SentenceTransformer
):

    embedding = model.encode(
        [resume_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embedding.astype(
        np.float32
    )


# ======================================================================
# SKILL MATCH SIGNAL
# ======================================================================

def calculate_skill_signal(
    resume_skills: set[str],
    job_skills: set[str]
):

    if not job_skills:
        return 0.0

    matched = resume_skills.intersection(
        job_skills
    )

    return len(matched) / len(
        job_skills
    )


def calculate_skill_gap(
    resume_skills: set[str],
    job_skills: set[str]
):

    return sorted(
        job_skills - resume_skills
    )


def calculate_skill_matches(
    resume_skills: set[str],
    job_skills: set[str]
):

    return sorted(
        resume_skills.intersection(
            job_skills
        )
    )


# ======================================================================
# HYBRID RANKING
# ======================================================================

def calculate_rankings(
    jobs: pd.DataFrame,
    job_embeddings: np.ndarray,
    resume_embedding: np.ndarray,
    job_skill_evidence: dict,
    resume_skills: list[str],
    title_col: str
):

    resume_skill_set = {
        normalize_skill(x)
        for x in resume_skills
    }

    semantic_scores = cosine_similarity(
        resume_embedding,
        job_embeddings
    )[0]

    results = []

    for index, row in jobs.iterrows():

        job_skills = set(
            job_skill_evidence.get(
                str(index),
                []
            )
        )

        matched = calculate_skill_matches(
            resume_skill_set,
            job_skills
        )

        gaps = calculate_skill_gap(
            resume_skill_set,
            job_skills
        )

        skill_signal = calculate_skill_signal(
            resume_skill_set,
            job_skills
        )

        semantic = safe_float(
            semantic_scores[index]
        )

        # Keep semantic similarity as the main signal.
        # Skill evidence provides an additional interpretable signal.
        hybrid = (
            0.75 * semantic
            + 0.25 * skill_signal
        )

        results.append(
            {
                "row_index": int(index),
                "job_title": clean_text(
                    row[title_col]
                ),
                "semantic_similarity": semantic,
                "skill_match_signal": skill_signal,
                "hybrid_match_score": hybrid,
                "matched_skills": matched,
                "skill_gaps": gaps,
            }
        )

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        "hybrid_match_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    return result_df


# ======================================================================
# GROQ
# ======================================================================

@st.cache_resource(show_spinner=False)
def get_groq_client():

    if not GROQ_API_KEY:
        return None

    try:

        from groq import Groq

        return Groq(
            api_key=GROQ_API_KEY
        )

    except Exception:
        return None


def generate_user_friendly_explanation(
    client,
    job_title: str,
    semantic_score: float,
    skill_score: float,
    matched_skills: list[str],
    skill_gaps: list[str]
):

    if client is None:
        return deterministic_explanation(
            job_title,
            matched_skills,
            skill_gaps
        )

    matched_text = ", ".join(
        matched_skills[:12]
    ) or "No direct skill matches identified"

    gaps_text = ", ".join(
        skill_gaps[:12]
    ) or "No major skill gaps identified"

    prompt = f"""
You are an AI resume analyst.

Explain a job recommendation to a job seeker in clear,
natural, professional English.

DO NOT return JSON.
DO NOT use markdown tables.
DO NOT discuss hiring probability.
DO NOT claim that the person will get the job.

Job title:
{job_title}

Semantic resume-job similarity:
{semantic_score:.3f}

Skill matching signal:
{skill_score:.3f}

Skills found in both resume and job:
{matched_text}

Skills requested by the job but not detected in the resume:
{gaps_text}

Write exactly three short sections:

Why this role matches:
Explain the strongest evidence.

Skills you already have:
Mention the most relevant matching skills.

What you could improve:
Mention the most important missing skills, if any.

Keep the explanation concise and useful to the user.
"""

    try:

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You explain resume-job matches. "
                        "Always respond in normal human-readable English."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=500,
        )

        text = response.choices[0].message.content

        if text:
            return text.strip()

    except Exception as exc:

        return (
            deterministic_explanation(
                job_title,
                matched_skills,
                skill_gaps
            )
            + f"\n\n"
              f"*LLM explanation unavailable; evidence-based "
              f"fallback used.*"
        )

    return deterministic_explanation(
        job_title,
        matched_skills,
        skill_gaps
    )


def deterministic_explanation(
    job_title: str,
    matched_skills: list[str],
    skill_gaps: list[str]
):

    matched_text = ", ".join(
        matched_skills[:6]
    )

    gap_text = ", ".join(
        skill_gaps[:6]
    )

    if matched_text:

        why = (
            f"Your resume matches this {job_title} role "
            f"through relevant skills such as {matched_text}."
        )

    else:

        why = (
            f"The semantic content of your resume has "
            f"some alignment with the {job_title} role, "
            f"although few direct skill overlaps were detected."
        )

    if matched_text:

        skills = (
            f"You already demonstrate relevant evidence in "
            f"{matched_text}."
        )

    else:

        skills = (
            "No strong direct skill overlap was identified "
            "from the available external skill vocabulary."
        )

    if gap_text:

        improvement = (
            f"Skills that appear relevant to this role but were "
            f"not detected in your resume include {gap_text}."
        )

    else:

        improvement = (
            "No major skill gaps were identified from the "
            "available job skill evidence."
        )

    return (
        f"**Why this role matches**\n"
        f"{why}\n\n"
        f"**Skills you already have**\n"
        f"{skills}\n\n"
        f"**What you could improve**\n"
        f"{improvement}"
    )


# ======================================================================
# UI HELPERS
# ======================================================================

def render_skills(
    skills: list[str],
    gap: bool = False
):

    if not skills:

        st.caption(
            "None identified."
        )

        return

    css_class = "gap" if gap else "skill"

    html = "".join(
        f'<span class="{css_class}">{skill}</span>'
        for skill in skills[:20]
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ======================================================================
# MAIN ANALYSIS
# ======================================================================

def analyze_resume(
    resume_bytes: bytes,
    top_k: int
):

    # --------------------------------------------------------------
    # 1. TEXT
    # --------------------------------------------------------------

    st.write(
        "📄 Extracting resume text..."
    )

    resume_text = extract_pdf_text(
        resume_bytes
    )

    if not resume_text:

        raise ValueError(
            "Could not extract text from this PDF."
        )

    st.success(
        f"Extracted {len(resume_text):,} characters"
    )

    # --------------------------------------------------------------
    # 2. JOB DATA
    # --------------------------------------------------------------

    st.write(
        "📊 Loading job dataset..."
    )

    jobs = load_jobs()

    (
        id_col,
        title_col,
        description_col,
        company_col,
        location_col
    ) = identify_job_columns(
        jobs
    )

    st.success(
        f"Loaded {len(jobs):,} job postings"
    )

    # --------------------------------------------------------------
    # 3. MODEL
    # --------------------------------------------------------------

    st.write(
        "🧠 Loading semantic model..."
    )

    model = load_semantic_model()

    # --------------------------------------------------------------
    # 4. JOB EMBEDDINGS
    # --------------------------------------------------------------

    st.write(
        "⚡ Loading cached job embeddings..."
    )

    job_embeddings = load_job_embeddings(
        jobs,
        title_col,
        description_col,
        model
    )

    st.success(
        f"Job embedding matrix: {job_embeddings.shape}"
    )

    # --------------------------------------------------------------
    # 5. EXTERNAL VOCABULARY
    # --------------------------------------------------------------

    st.write(
        "📚 Loading external skill vocabulary..."
    )

    vocabulary = load_external_skill_vocabulary()

    if not vocabulary:

        st.warning(
            "No external skill vocabulary was found."
        )

    else:

        st.success(
            f"Skill vocabulary: {len(vocabulary):,}"
        )

    # --------------------------------------------------------------
    # 6. RESUME SKILLS
    # --------------------------------------------------------------

    st.write(
        "🔎 Extracting resume skills..."
    )

    resume_skills = extract_resume_skills(
        resume_text,
        vocabulary
    )

    st.success(
        f"Detected {len(resume_skills)} skills"
    )

    # --------------------------------------------------------------
    # 7. RESUME EMBEDDING
    # --------------------------------------------------------------

    st.write(
        "🧠 Generating resume semantic embedding..."
    )

    resume_embedding = generate_resume_embedding(
        resume_text,
        model
    )

    # --------------------------------------------------------------
    # 8. JOB SKILL EVIDENCE
    # --------------------------------------------------------------

    st.write(
        "🔍 Loading job skill evidence..."
    )

    job_skill_evidence = (
        load_or_build_job_skill_evidence(
            jobs,
            title_col,
            description_col,
            vocabulary
        )
    )

    # --------------------------------------------------------------
    # 9. RANKING
    # --------------------------------------------------------------

    st.write(
        "📈 Calculating job similarity and ranking..."
    )

    ranked = calculate_rankings(
        jobs,
        job_embeddings,
        resume_embedding,
        job_skill_evidence,
        resume_skills,
        title_col
    )

    top = ranked.head(
        top_k
    ).copy()

    # --------------------------------------------------------------
    # 10. ADD JOB METADATA
    # --------------------------------------------------------------

    for idx in top.index:

        row_index = int(
            top.loc[idx, "row_index"]
        )

        row = jobs.iloc[
            row_index
        ]

        top.loc[idx, "company"] = (
            clean_text(
                row[company_col]
            )
            if company_col
            else ""
        )

        top.loc[idx, "location"] = (
            clean_text(
                row[location_col]
            )
            if location_col
            else ""
        )

        top.loc[idx, "description"] = (
            clean_text(
                row[description_col]
            )
        )

    # --------------------------------------------------------------
    # 11. GROQ EXPLANATIONS
    # --------------------------------------------------------------

    client = get_groq_client()

    explanations = []

    progress = st.progress(
        0
    )

    status = st.empty()

    for i, (_, row) in enumerate(
        top.iterrows()
    ):

        status.write(
            f"💬 Generating explanation {i + 1}/{len(top)}..."
        )

        explanation = (
            generate_user_friendly_explanation(
                client,
                row["job_title"],
                row["semantic_similarity"],
                row["skill_match_signal"],
                row["matched_skills"],
                row["skill_gaps"]
            )
        )

        explanations.append(
            explanation
        )

        progress.progress(
            (i + 1) / len(top)
        )

    top["explanation"] = explanations

    status.empty()

    return {
        "resume_text": resume_text,
        "resume_skills": resume_skills,
        "all_recommendations": ranked,
        "top_recommendations": top,
    }


# ======================================================================
# DISPLAY RESULTS
# ======================================================================

def display_results(
    result: dict
):

    resume_text = result[
        "resume_text"
    ]

    resume_skills = result[
        "resume_skills"
    ]

    top = result[
        "top_recommendations"
    ]

    all_jobs = result[
        "all_recommendations"
    ]

    # --------------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Resume Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Resume Characters",
            f"{len(resume_text):,}"
        )

    with c2:
        st.metric(
            "Skills Detected",
            len(resume_skills)
        )

    with c3:
        st.metric(
            "Jobs Evaluated",
            len(all_jobs)
        )

    with c4:
        st.metric(
            "Top Match",
            score_percent(
                top.iloc[0][
                    "hybrid_match_score"
                ]
            )
            if len(top)
            else "N/A"
        )

    # --------------------------------------------------------------
    # SKILLS
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-title">🔎 Skills Detected</div>',
        unsafe_allow_html=True
    )

    render_skills(
        resume_skills
    )

    # --------------------------------------------------------------
    # JOBS
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-title">🎯 Recommended Jobs</div>',
        unsafe_allow_html=True
    )

    for rank, (_, row) in enumerate(
        top.iterrows(),
        start=1
    ):

        with st.expander(
            f"#{rank}  {row['job_title']}   "
            f"• Match {score_percent(row['hybrid_match_score'])}",
            expanded=(rank == 1)
        ):

            st.markdown(
                '<div class="job-card">',
                unsafe_allow_html=True
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Overall Match",
                    score_percent(
                        row[
                            "hybrid_match_score"
                        ]
                    )
                )

            with c2:

                st.metric(
                    "Semantic Match",
                    score_percent(
                        row[
                            "semantic_similarity"
                        ]
                    )
                )

            with c3:

                st.metric(
                    "Skill Match",
                    score_percent(
                        row[
                            "skill_match_signal"
                        ]
                    )
                )

            if row["company"]:

                st.write(
                    f"**Company:** {row['company']}"
                )

            if row["location"]:

                st.write(
                    f"**Location:** {row['location']}"
                )

            st.markdown(
                "### ✅ Skills You Already Have"
            )

            render_skills(
                row["matched_skills"]
            )

            st.markdown(
                "### 📚 Skills You Could Develop"
            )

            render_skills(
                row["skill_gaps"],
                gap=True
            )

            st.markdown(
                "### 💡 Why This Job Matches"
            )

            st.markdown(
                f'<div class="explanation">'
                f'{row["explanation"]}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                "### 📝 Job Description"
            )

            description = row[
                "description"
            ]

            if len(description) > 1200:

                description = (
                    description[:1200]
                    + "..."
                )

            st.write(
                description
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

    # --------------------------------------------------------------
    # AGGREGATE SKILL GAPS
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-title">📈 Overall Skill Gap Analysis</div>',
        unsafe_allow_html=True
    )

    gap_frequency = {}

    for _, row in top.iterrows():

        for gap in row["skill_gaps"]:

            gap_frequency[gap] = (
                gap_frequency.get(
                    gap,
                    0
                )
                + 1
            )

    sorted_gaps = sorted(
        gap_frequency.items(),
        key=lambda x: x[1],
        reverse=True
    )

    if sorted_gaps:

        render_skills(
            [
                gap
                for gap, _ in sorted_gaps[:20]
            ],
            gap=True
        )

    else:

        st.success(
            "No major skill gaps were identified "
            "across the recommended jobs."
        )

    # --------------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-title">💾 Export Results</div>',
        unsafe_allow_html=True
    )

    export_df = top.copy()

    export_df[
        "matched_skills"
    ] = export_df[
        "matched_skills"
    ].apply(
        lambda x: ", ".join(x)
    )

    export_df[
        "skill_gaps"
    ] = export_df[
        "skill_gaps"
    ].apply(
        lambda x: ", ".join(x)
    )

    export_df = export_df.drop(
        columns=[
            "row_index"
        ],
        errors="ignore"
    )

    csv_data = export_df.to_csv(
        index=False
    )

    st.download_button(
        "⬇️ Download Recommendations CSV",
        data=csv_data,
        file_name="resume_job_recommendations.csv",
        mime="text/csv",
        use_container_width=True
    )


# ======================================================================
# SIDEBAR
# ======================================================================

def render_sidebar():

    with st.sidebar:

        st.title(
            "📄 Analyzer"
        )

        st.caption(
            "AI-powered resume analysis using "
            "semantic matching and external skill evidence."
        )

        st.divider()

        st.subheader(
            "🔬 Pipeline"
        )

        pipeline = [
            "Resume text extraction",
            "External skill vocabulary",
            "Skill extraction",
            "Semantic embedding",
            "Job similarity",
            "Skill matching",
            "Hybrid ranking",
            "Evidence-based explanation",
        ]

        for item in pipeline:

            st.write(
                f"• {item}"
            )

        st.divider()

        st.subheader(
            "📊 Dataset"
        )

        try:

            jobs = load_jobs()

            st.write(
                f"{len(jobs):,} job postings"
            )

        except Exception:

            st.write(
                "Dataset unavailable"
            )

        st.divider()

        if GROQ_API_KEY:

            st.success(
                f"Groq enabled\n\n{GROQ_MODEL}"
            )

        else:

            st.info(
                "Groq not configured.\n"
                "Deterministic explanations will be used."
            )


# ======================================================================
# APP
# ======================================================================

def main():

    render_sidebar()

    st.markdown(
        '<div class="main-title">📄 AI Resume Analyzer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Upload your resume and discover your skills, '
        'job matches, skill gaps, and evidence-based recommendations.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "📤 Upload your resume as a PDF",
        type=["pdf"],
        help="Upload a text-based PDF resume."
    )

    top_k = st.selectbox(
        "Number of job recommendations",
        options=[5, 10],
        index=0
    )

    analyze = st.button(
        "🚀 Analyze Resume",
        type="primary",
        use_container_width=True
    )

    if uploaded_file is not None:

        st.caption(
            f"Selected: {uploaded_file.name}"
        )

    if analyze:

        if uploaded_file is None:

            st.warning(
                "Please upload a PDF resume first."
            )

            st.stop()

        resume_bytes = uploaded_file.getvalue()

        # ----------------------------------------------------------
        # Prevent unnecessary reruns for same resume
        # ----------------------------------------------------------

        file_hash = hashlib.md5(
            resume_bytes
        ).hexdigest()

        previous_hash = st.session_state.get(
            "resume_hash"
        )

        if (
            previous_hash == file_hash
            and "analysis_result" in st.session_state
        ):

            st.info(
                "This resume has already been analyzed."
            )

        else:

            try:

                with st.status(
                    "Running resume analysis pipeline...",
                    expanded=True
                ):

                    result = analyze_resume(
                        resume_bytes,
                        top_k
                    )

                st.session_state[
                    "analysis_result"
                ] = result

                st.session_state[
                    "resume_hash"
                ] = file_hash

                st.success(
                    "Resume analysis completed successfully."
                )

            except Exception as exc:

                st.error(
                    "Resume analysis failed."
                )

                st.exception(
                    exc
                )

    # --------------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------------

    if (
        "analysis_result"
        in st.session_state
    ):

        st.divider()

        display_results(
            st.session_state[
                "analysis_result"
            ]
        )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()