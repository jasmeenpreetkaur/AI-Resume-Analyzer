"""
======================================================================
AI RESUME ANALYZER - STREAMLIT APPLICATION
======================================================================

USER FLOW
---------
Upload Resume (text-based OR scanned PDF)
    -> Extract text (pdf_text.py; OCR fallback for image-only pages)
    -> Extract skills (esco_vocab.py; ESCO skills table only)
    -> Resume semantic embedding
    -> Job embeddings + job skill sets (job_ranker.py; fingerprinted
       caches, rebuilt automatically when the data or model changes)
    -> Score every job (matching_core.MatchScorer)
    -> Top-K jobs
    -> Groq explanation for Top-K only (explanation only)
    -> User-friendly resume analysis

ONE SCORING DEFINITION
----------------------
Scores come from matching_core.MatchScorer - the same object Step 10
(offline ranking) and Step 13 (command line) use:

  * models/match_regressor.joblib present and valid (Step 10b)
        -> the trained HistGradientBoostingRegressor scores every job;
  * otherwise -> the transparent weighted formula (weights defined once
    in matching_core.WEIGHTS).

The sidebar and the results page always say which one ran.

SKILLS
------
Only the ESCO skills table counts (preferredLabel + altLabels of
KnowledgeSkillCompetence rows). Occupations, skill groups and other
files in data/external are never read as skills.

IMPORTANT
---------
This application does NOT require resume_analyzer_backend.py.

It uses:
    data/jobs/all_job_post.csv
    data/processed/esco_skills_clean.csv   or   data/external/skills_en.csv
    models/match_regressor.joblib          (optional)

Caches:
    data/cache/job_embeddings.npy (+ .meta.json)
    data/cache/job_skill_index.json

Run:
    python -m streamlit run src/14_app.py
======================================================================
"""

from __future__ import annotations

import os
import html
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import matching_core as mc
import pdf_text
from esco_vocab import load_esco_vocabulary
from job_ranker import (
    build_job_corpus,
    rank_jobs,
    resume_skill_uris,
    skill_breakdown,
)


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

JOB_DATA_PATH = DATA_DIR / "jobs" / "all_job_post.csv"

CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


# ======================================================================
# CONFIGURATION
# ======================================================================

MODEL_NAME = mc.SEMANTIC_MODEL_NAME

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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def score_percent(value: float) -> str:
    value = max(0.0, min(1.0, safe_float(value)))
    return f"{value * 100:.1f}%"


class ResumeReadError(ValueError):
    """The uploaded PDF could not be turned into usable text."""


# ======================================================================
# PDF EXTRACTION (shared with Step 3 / Step 13; OCR for scanned pages)
# ======================================================================

@st.cache_data(show_spinner=False)
def extract_resume(file_bytes: bytes) -> pdf_text.PDFExtraction:
    return pdf_text.extract_pdf_text(file_bytes)


# ======================================================================
# CACHED RESOURCES
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


@st.cache_resource(show_spinner=False)
def load_semantic_model():

    return SentenceTransformer(
        MODEL_NAME
    )


@st.cache_resource(show_spinner=False)
def load_vocabulary(signature: str):
    """
    ESCO skills vocabulary (skills table only). Returns
    ``(vocabulary_or_None, error_message)``.
    ``signature`` changes when the skills file changes.
    """

    try:
        return load_esco_vocabulary(), ""

    except (FileNotFoundError, ValueError) as exc:
        return None, str(exc)


def vocabulary_signature() -> str:

    from esco_vocab import find_esco_skills_file

    path = find_esco_skills_file()

    if path is None:
        return "missing"

    stat = path.stat()

    return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"


@st.cache_resource(show_spinner=False)
def load_scorer(signature: str) -> mc.MatchScorer:
    """
    The shared scorer: trained model if one is installed, otherwise the
    transparent formula. ``signature`` changes when the model file does.
    """

    return mc.MatchScorer.load()


def scorer_signature() -> str:

    import os as _os

    path = Path(
        _os.getenv("MATCH_MODEL_PATH") or mc.DEFAULT_MODEL_PATH
    )

    if not path.is_file():
        return f"none:{path}"

    stat = path.stat()

    return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"


@st.cache_resource(
    show_spinner="Preparing job embeddings and skills "
    "(first run only - this can take a few minutes)..."
)
def load_corpus(
    jobs_signature: str,
    vocabulary_sig: str,
    _jobs: pd.DataFrame,
    _vocabulary,
    _model,
):
    """
    Job embeddings, job skill sets and lexical indexes. The caches on
    disk are fingerprinted by job text, embedding model and vocabulary,
    so they are rebuilt whenever any of them change - never reused stale.
    """

    def encode(texts):

        return np.asarray(
            _model.encode(
                list(texts),
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            ),
            dtype=np.float32
        )

    return build_job_corpus(
        _jobs,
        encode=encode,
        model_name=MODEL_NAME,
        vocabulary=_vocabulary,
        cache_dir=CACHE_DIR,
    )


def jobs_signature() -> str:

    if not JOB_DATA_PATH.exists():
        return "missing"

    stat = JOB_DATA_PATH.stat()

    return f"{JOB_DATA_PATH}:{stat.st_size}:{stat.st_mtime_ns}"


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

    # Labels come from data files and a PDF; escape before putting them
    # into raw HTML.
    markup = "".join(
        f'<span class="{css_class}">{html.escape(str(skill))}</span>'
        for skill in skills[:20]
    )

    st.markdown(
        markup,
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
    # 1. TEXT (OCR fallback for scanned / image-only PDFs)
    # --------------------------------------------------------------

    st.write(
        "📄 Extracting resume text..."
    )

    extraction = extract_resume(
        resume_bytes
    )

    if not extraction.usable:

        raise ResumeReadError(
            extraction.user_message()
        )

    resume_text = extraction.text

    if extraction.ocr_used:

        st.info(
            f"Scanned page(s) {extraction.ocr_pages} were read with OCR "
            f"({extraction.ocr_engine}). OCR can make mistakes, so check "
            "the detected skills below."
        )

    for warning in extraction.warnings:

        st.warning(
            warning
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

    st.success(
        f"Loaded {len(jobs):,} job postings"
    )

    # --------------------------------------------------------------
    # 3. MODEL, SKILLS VOCABULARY, SCORER
    # --------------------------------------------------------------

    st.write(
        "🧠 Loading semantic model..."
    )

    model = load_semantic_model()

    st.write(
        "📚 Loading ESCO skill vocabulary..."
    )

    vocabulary, vocabulary_error = load_vocabulary(
        vocabulary_signature()
    )

    if vocabulary is None:

        st.warning(
            "Skill matching is disabled: "
            + vocabulary_error
        )

    else:

        st.success(
            f"ESCO skills: {vocabulary.skill_count:,}"
        )

    scorer = load_scorer(
        scorer_signature()
    )

    st.write(
        f"⚖️ Scoring: {scorer.description}"
    )

    # --------------------------------------------------------------
    # 4. JOB CORPUS (embeddings + job skills, fingerprinted caches)
    # --------------------------------------------------------------

    st.write(
        "⚡ Loading job embeddings and skills..."
    )

    corpus = load_corpus(
        jobs_signature(),
        vocabulary_signature(),
        jobs,
        vocabulary,
        model
    )

    for note in corpus.notes:

        st.info(
            note
        )

    # --------------------------------------------------------------
    # 5. RESUME SKILLS (ESCO skills table only)
    # --------------------------------------------------------------

    st.write(
        "🔎 Extracting resume skills..."
    )

    resume_skill_map = resume_skill_uris(
        resume_text,
        vocabulary
    )

    resume_skills = sorted(
        resume_skill_map.values(),
        key=str.lower
    )

    st.success(
        f"Detected {len(resume_skills)} skills"
    )

    # --------------------------------------------------------------
    # 6. RESUME EMBEDDING
    # --------------------------------------------------------------

    st.write(
        "🧠 Generating resume semantic embedding..."
    )

    resume_embedding = model.encode(
        [resume_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0].astype(np.float32)

    # --------------------------------------------------------------
    # 7. SCORE AND RANK (shared scoring definition)
    # --------------------------------------------------------------

    st.write(
        "📈 Calculating job similarity and ranking..."
    )

    ranked = rank_jobs(
        corpus,
        resume_text=resume_text,
        resume_embedding=resume_embedding,
        resume_skills=resume_skill_map.keys(),
        scorer=scorer
    )

    top = ranked.head(
        top_k
    ).copy()

    # --------------------------------------------------------------
    # 8. ADD JOB METADATA AND SKILL BREAKDOWN FOR TOP-K
    # --------------------------------------------------------------

    columns = corpus.columns

    companies = []
    locations = []
    descriptions = []
    matched_lists = []
    gap_lists = []

    for row_index in top["row_index"].astype(int):

        row = corpus.frame.iloc[
            row_index
        ]

        companies.append(
            clean_text(row[columns["company"]])
            if columns["company"]
            else ""
        )

        locations.append(
            clean_text(row[columns["location"]])
            if columns["location"]
            else ""
        )

        descriptions.append(
            clean_text(row[columns["description"]])
        )

        matched, gaps = skill_breakdown(
            corpus,
            row_index,
            resume_skill_map
        )

        matched_lists.append(matched)
        gap_lists.append(gaps)

    top["company"] = companies
    top["location"] = locations
    top["description"] = descriptions
    top["matched_skills"] = matched_lists
    top["skill_gaps"] = gap_lists

    # --------------------------------------------------------------
    # 9. GROQ EXPLANATIONS
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
        "scoring_method": scorer.method,
        "scoring_description": scorer.description,
        "extraction": extraction.to_record(),
        "skill_matching_enabled": vocabulary is not None,
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

    st.caption(
        "Scoring: "
        + result.get(
            "scoring_description",
            "unknown"
        )
    )

    if result.get("extraction", {}).get("ocr_used"):

        st.caption(
            "Text source: OCR ("
            + str(result["extraction"].get("ocr_engine"))
            + ")"
        )

    if not result.get("skill_matching_enabled", True):

        st.warning(
            "No ESCO skills file was available, so skills were not "
            "detected and jobs were ranked on semantic and lexical "
            "similarity only."
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

            if result.get("scoring_method") == mc.METHOD_MODEL:

                # A trained model's score is not a percentage of
                # requirements met; show the transparent formula next to
                # it so the number can be sanity-checked.
                st.caption(
                    "Trained-model score (similarity to same-category "
                    "pairs, not a hiring probability). Transparent formula "
                    "score for comparison: "
                    + score_percent(
                        row["formula_match_score"]
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

            # LLM text is rendered as Markdown, never as raw HTML.
            st.markdown(
                row["explanation"]
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

    export_df["resume_skills_detected"] = ", ".join(
        resume_skills
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
            "semantic matching and ESCO skill evidence."
        )

        st.divider()

        st.subheader(
            "🔬 Pipeline"
        )

        pipeline = [
            "Resume text extraction (OCR for scans)",
            "ESCO skill vocabulary (skills only)",
            "Skill extraction",
            "Semantic embedding",
            "Job similarity",
            "Skill matching",
            "Hybrid scoring and ranking",
            "Evidence-based explanation",
        ]

        for item in pipeline:

            st.write(
                f"• {item}"
            )

        st.divider()

        st.subheader(
            "⚖️ Scoring"
        )

        try:

            scorer = load_scorer(
                scorer_signature()
            )

            if scorer.uses_model:

                st.success(
                    "Trained model\n\n"
                    + scorer.description
                )

            else:

                st.info(
                    scorer.description
                )

        except Exception as exc:

            st.write(
                f"Scorer unavailable: {exc}"
            )

        st.divider()

        st.subheader(
            "📚 Skills"
        )

        try:

            vocabulary, vocabulary_error = load_vocabulary(
                vocabulary_signature()
            )

            if vocabulary is None:

                st.warning(
                    "No ESCO skills file found. "
                    "Skill matching is off."
                )

            else:

                st.write(
                    f"{vocabulary.skill_count:,} ESCO skills "
                    f"({Path(vocabulary.source).name})"
                )

        except Exception:

            st.write(
                "Skill vocabulary unavailable"
            )

        st.divider()

        st.subheader(
            "🖼️ Scanned PDFs"
        )

        try:

            ocr = pdf_text.ocr_engine_status()

            if ocr["selected"]:

                st.write(
                    f"OCR engine: {ocr['selected']}"
                )

            else:

                st.warning(
                    "No OCR engine installed - scanned (image-only) "
                    "PDFs cannot be read. "
                    + str(ocr.get("hint") or "")
                )

        except Exception:

            st.write(
                "OCR status unavailable"
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
        help="Text-based or scanned PDF. Scanned (image-only) pages are read with OCR when an OCR engine is installed."
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
        ).hexdigest() + f":{top_k}"

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

            except ResumeReadError as exc:

                # Expected user-facing problem (scanned PDF without OCR,
                # empty file, ...): explain it, no stack trace.
                st.error(
                    "This resume could not be read."
                )

                st.warning(
                    str(exc)
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