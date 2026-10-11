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
    -> Keep jobs inside the user's preferences, drop repeated and
       look-alike postings and weak matches
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

This app never re-weights or re-scores. Preferences only decide WHICH
jobs are eligible; the score itself is untouched. The sidebar and the
results page always say which scorer ran.

SKILLS
------
Only the ESCO skills table counts (preferredLabel + altLabels of
KnowledgeSkillCompetence rows). Occupations, skill groups and other
files in data/external are never read as skills.

What the app adds on top of job_ranker (presentation only, never score):
  * "common" skills (found in many postings, e.g. communication) are
    shown but do not count toward the displayed coverage;
  * coverage is reported as "x of y" with an evidence-quality warning,
    never as a bare 100% from a single detected skill;
  * every skill shown can be traced back to the phrase that triggered it.

Run (from the project root):
    python -m streamlit run src/14_app.py
======================================================================
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# The helper modules live next to this file in src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import esco_vocab
import job_ranker
import matching_core as mc
import pdf_text


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

# --- Display-only skill evidence rules (they never change the score) ---

# A skill found in more than this share of ALL postings (for example
# "communication") says little about fit, so it is shown but is not
# counted toward the displayed coverage or listed as a gap.
GENERIC_SKILL_MAX_JOB_SHARE = 0.20

# Fewer specific skills than this means the posting gave thin evidence.
MIN_JOB_SKILLS_FOR_RELIABLE_COVERAGE = 5

# Default for the "Minimum overall match" slider. 0 = off. Scores come from
# whichever scorer is active (trained model or formula), so a sensible
# cut-off depends on your data; use the slider and the score range shown in
# the results to calibrate it.
DEFAULT_MIN_MATCH = 0.0

# Look-alike removal is always on (no UI control). Two recommendations are
# look-alikes when their TITLES describe the same role (see
# titles_are_lookalikes) or their job texts are at least this similar
# (cosine). Real postings from different employers rarely exceed ~0.9, so
# the text rule is deliberately lower than "near identical".
LOOKALIKE_TEXT_SIMILARITY = 0.80


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

</style>
""",
    unsafe_allow_html=True,
)


# ======================================================================
# UTILITY
# ======================================================================

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
# CACHED RESOURCES
# ======================================================================

def get_file_signature(path: Path | None) -> str:
    """Cheap cache-busting signature for a local file."""
    if path is None or not path.exists() or not path.is_file():
        return "missing"
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


@st.cache_data(show_spinner=False)
def read_resume(file_bytes: bytes):
    """PDF bytes -> pdf_text.PDFExtraction (OCR fallback for scanned pages)."""
    return pdf_text.extract_pdf_text(file_bytes)


@st.cache_data(show_spinner=False)
def _load_jobs_cached(file_signature: str) -> pd.DataFrame:
    del file_signature  # exists solely to invalidate Streamlit's cache

    if not JOB_DATA_PATH.exists():
        raise FileNotFoundError(f"Job dataset not found:\n{JOB_DATA_PATH}")

    return pd.read_csv(JOB_DATA_PATH, low_memory=False).fillna("")


def load_jobs() -> pd.DataFrame:
    return _load_jobs_cached(get_file_signature(JOB_DATA_PATH))


@st.cache_resource(show_spinner=False)
def load_semantic_model():
    return SentenceTransformer(MODEL_NAME)


def vocabulary_signature() -> str:
    path = esco_vocab.find_esco_skills_file()
    return f"{path}:{get_file_signature(path)}" if path else "missing"


@st.cache_resource(show_spinner=False)
def load_vocabulary(signature: str):
    """(ESCOVocabulary | None, error message)."""
    del signature

    try:
        return esco_vocab.load_esco_vocabulary(), ""
    except (FileNotFoundError, ValueError) as exc:
        return None, str(exc)


def scorer_signature() -> str:
    path = Path(os.getenv("MATCH_MODEL_PATH") or mc.DEFAULT_MODEL_PATH)
    return f"{path}:{get_file_signature(path)}"


@st.cache_resource(show_spinner=False)
def load_scorer(signature: str):
    del signature
    return mc.MatchScorer.load()


@st.cache_resource(show_spinner=False)
def load_corpus(jobs_signature: str, vocab_signature: str):
    """
    Everything that depends only on the job data: embeddings, skill sets,
    token indexes. job_ranker caches embeddings and skill sets on disk with
    a fingerprint, so this is slow only the first time (or after the data,
    model or vocabulary changes).
    """
    del jobs_signature

    jobs = load_jobs()
    vocabulary, _ = load_vocabulary(vocab_signature)
    model = load_semantic_model()

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    return job_ranker.build_job_corpus(
        jobs,
        encode=encode,
        model_name=MODEL_NAME,
        vocabulary=vocabulary,
        cache_dir=CACHE_DIR,
    )


# ======================================================================
# SKILL VIEW (display layer on top of job_ranker's skill sets)
# ======================================================================

def compute_skill_job_share(corpus: job_ranker.JobCorpus) -> dict[str, float]:
    """Share of all job postings in which each ESCO skill appears."""
    counts = Counter(
        uri for skills in corpus.skill_sets for uri in skills
    )
    total = max(len(corpus), 1)
    return {uri: count / total for uri, count in counts.items()}


def skill_evidence_level(specific_skill_count: int) -> str:
    if specific_skill_count == 0:
        return "insufficient"
    if specific_skill_count < MIN_JOB_SKILLS_FOR_RELIABLE_COVERAGE:
        return "limited"
    return "good"


def build_skill_view(
    corpus: job_ranker.JobCorpus,
    row_index: int,
    resume_uris: dict[str, str],
    job_share: dict[str, float],
) -> dict[str, Any]:
    """
    Matched skills, gaps and evidence quality for one job.

    Gaps are the job's SPECIFIC skills (not common ones) that the resume
    lacks. Coverage counts specific skills only.
    """
    job_uris = corpus.skill_sets[row_index]

    common = {
        uri for uri in job_uris
        if job_share.get(uri, 0.0) > GENERIC_SKILL_MAX_JOB_SHARE
    }
    specific = job_uris - common

    matched_specific = {uri for uri in specific if uri in resume_uris}

    def labels(uris) -> list[str]:
        return sorted((corpus.label(uri) for uri in uris), key=str.lower)

    return {
        "matched_skills": labels(uri for uri in job_uris if uri in resume_uris),
        "skill_gaps": labels(uri for uri in specific if uri not in resume_uris),
        "common_skills": labels(common),
        "specific_skill_count": len(specific),
        "matched_specific_count": len(matched_specific),
        "evidence_level": skill_evidence_level(len(specific)),
    }


# ======================================================================
# SKILL TRACEABILITY
# ======================================================================

def trace_skill_phrases(
    vocabulary: esco_vocab.ESCOVocabulary,
    text: str,
) -> dict[str, str]:
    """
    {skill URI: the exact phrase in ``text`` that triggered it}.

    Walks the same trie, over the same tokens, as
    ``esco_vocab.extract_skills_from_text`` so the phrases explain exactly
    the skills the pipeline reported. The first occurrence wins.
    """
    tokens = esco_vocab.tokenize(text)
    found: dict[str, str] = {}
    root = vocabulary.trie.root

    for start in range(len(tokens)):

        node = root
        position = start

        while position < len(tokens):

            child = node.children.get(tokens[position])

            if child is None:
                break

            node = child

            if node.skill_ids:

                phrase = " ".join(tokens[start:position + 1])

                for skill_id in node.skill_ids:

                    skill = vocabulary.skills[skill_id]
                    uri = skill["concept_uri"] or f"id:{skill_id}"
                    found.setdefault(uri, phrase)

            position += 1

    return found


def _context_snippet(text: str, phrase: str, width: int = 70) -> str:
    """A short window of the posting around the phrase that triggered a skill."""
    if not text or not phrase or phrase.startswith("[skills field]"):
        return ""

    match = re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text)

    if match is None:
        return ""

    start = max(0, match.start() - width)
    end = min(len(text), match.end() + width)

    return (
        ("…" if start > 0 else "")
        + text[start:end]
        + ("…" if end < len(text) else "")
    )


def build_job_skill_rows(
    corpus: job_ranker.JobCorpus,
    row_index: int,
    resume_uris: dict[str, str],
    job_share: dict[str, float],
) -> list[dict[str, str]]:
    """One row per ESCO skill found in a posting, with the phrase behind it."""
    vocabulary = corpus.vocabulary

    if vocabulary is None:
        return []

    text = corpus.texts[row_index]
    phrases = trace_skill_phrases(vocabulary, text)
    normalized = esco_vocab.normalize_text(text)

    # Skills that came from the dataset's own skills column.
    field_entries: dict[str, str] = {}
    skills_column = corpus.columns.get("skills")

    if skills_column:

        cell = corpus.frame.iloc[row_index][skills_column]

        for entry in job_ranker.parse_skill_list(cell):

            skill_id = vocabulary.lookup_term(entry)

            if skill_id is not None:

                skill = vocabulary.skills[skill_id]
                uri = skill["concept_uri"] or f"id:{skill_id}"
                field_entries.setdefault(uri, entry)

    rows = []

    for uri in sorted(
        corpus.skill_sets[row_index],
        key=lambda u: corpus.label(u).lower(),
    ):

        phrase = phrases.get(uri, "")

        rows.append(
            {
                "Skill": corpus.label(uri),
                "In your resume": "Yes" if uri in resume_uris else "No",
                "Counts toward coverage": (
                    "No (common skill)"
                    if job_share.get(uri, 0.0) > GENERIC_SKILL_MAX_JOB_SHARE
                    else "Yes"
                ),
                "Phrase found in posting": (
                    phrase
                    or f"[skills field] {field_entries.get(uri, '')}".strip()
                ),
                "Context": _context_snippet(normalized, phrase),
            }
        )

    return rows


def build_resume_skill_rows(
    vocabulary: esco_vocab.ESCOVocabulary | None,
    resume_text: str,
    resume_uris: dict[str, str],
) -> list[dict[str, str]]:
    """Which phrase in the resume produced each detected skill."""
    if vocabulary is None or not resume_uris:
        return []

    phrases = trace_skill_phrases(vocabulary, resume_text)

    return sorted(
        (
            {
                "Skill": label,
                "Phrase found in resume": phrases.get(uri, ""),
            }
            for uri, label in resume_uris.items()
        ),
        key=lambda row: row["Skill"].lower(),
    )


# ======================================================================
# JOB SCOPE (USER PREFERENCES)
# ======================================================================

INTEREST_OPTIONS = [
    "AI / Machine Learning",
    "Data Science / Data Analyst",
    "Software Developer",
    "Web Developer",
    "Cybersecurity",
    "Cloud / DevOps",
    "Business / Management",
    "Other",
]

EXPERIENCE_OPTIONS = [
    "No preference",
    "Entry-level / Fresher",
    "Internship",
    "Experienced",
]

# Matched against the job TITLE only (lower-cased).
INTEREST_TITLE_PATTERNS = {
    "AI / Machine Learning": (
        r"\b(?:machine learning|ml|ai|artificial intelligence|deep learning|nlp|"
        r"natural language|computer vision|data scientist|mlops)\b"
    ),
    "Data Science / Data Analyst": (
        r"\b(?:data scien\w*|data analy\w*|data engineer\w*|analytics|"
        r"business intelligence|bi|statistic\w*|quantitative)\b"
    ),
    "Software Developer": (
        r"\b(?:software|developer|programmer|backend|back-end|full[- ]stack|"
        r"application engineer|java|python|dotnet)\b"
    ),
    "Web Developer": (
        r"\b(?:web|front[- ]?end|ui|ux|react|angular|javascript|php|wordpress|html)\b"
    ),
    "Cybersecurity": (
        r"\b(?:cyber\w*|infosec|pentest\w*|penetration test\w*|information security|"
        r"security (?:analyst|engineer|architect|specialist|administrator)|soc analyst)\b"
    ),
    "Cloud / DevOps": (
        r"\b(?:cloud|devops|sre|site reliability|infrastructure|aws|azure|gcp|"
        r"kubernetes|platform engineer|systems? (?:engineer|administrator))\b"
    ),
    "Business / Management": (
        r"\b(?:business|management|manager|sales|marketing|operations|"
        r"account (?:executive|manager)|consultant|executive|administrator|project)\b"
    ),
}

# Titles that contain a technical word ("Software Sales Engineer") but are sales roles.
NON_TECHNICAL_TITLE_PATTERN = (
    r"\b(?:sales|marketing|recruit\w*|business development|"
    r"account (?:executive|manager)|customer success)\b"
)

SENIOR_TITLE_PATTERN = (
    r"\b(?:senior|sr|lead|principal|director|head|vp|vice president|chief|"
    r"manager|staff|supervisor)\b"
)
INTERN_TITLE_PATTERN = r"\b(?:intern|internship|trainee|apprentice\w*|co-op)\b"
JUNIOR_TITLE_PATTERN = (
    r"\b(?:intern|internship|trainee|apprentice\w*|junior|jr|entry[- ]level)\b"
)


def build_job_scope_mask(
    job_titles: list[str],
    interests: list[str],
    experience: str,
    top_k: int,
) -> tuple[np.ndarray, list[str]]:
    """
    Boolean mask (by job position) of postings inside the user's
    preferences, plus human-readable notes. A filter that would leave
    nothing is relaxed with a visible note instead of being ignored.
    """
    notes: list[str] = []
    titles = pd.Series(job_titles, dtype=str).str.lower()
    mask = np.ones(len(titles), dtype=bool)

    chosen = [i for i in (interests or []) if i in INTEREST_TITLE_PATTERNS]

    if chosen:

        interest_mask = np.zeros(len(titles), dtype=bool)

        for interest in chosen:
            interest_mask |= titles.str.contains(
                INTEREST_TITLE_PATTERNS[interest], regex=True, na=False
            ).to_numpy()

        if interest_mask.any():

            mask = interest_mask

            notes.append(
                f"Career interests ({', '.join(chosen)}): "
                f"{int(interest_mask.sum()):,} postings have a matching title."
            )

            if "Business / Management" not in chosen:

                technical_only = mask & ~titles.str.contains(
                    NON_TECHNICAL_TITLE_PATTERN, regex=True, na=False
                ).to_numpy()

                removed = int(mask.sum() - technical_only.sum())

                if removed and technical_only.any():

                    mask = technical_only

                    notes.append(
                        f"Excluded {removed:,} sales/marketing-titled postings "
                        f"that only matched on a technical keyword."
                    )

        else:

            notes.append(
                "No job titles matched your selected career interests, so the "
                "interest filter was not applied."
            )

    if experience == "Entry-level / Fresher":
        exp_mask = ~titles.str.contains(
            SENIOR_TITLE_PATTERN, regex=True, na=False
        ).to_numpy()
        label = "senior/management titles excluded"
    elif experience == "Internship":
        exp_mask = titles.str.contains(
            INTERN_TITLE_PATTERN, regex=True, na=False
        ).to_numpy()
        label = "internship/trainee titles only"
    elif experience == "Experienced":
        exp_mask = ~titles.str.contains(
            JUNIOR_TITLE_PATTERN, regex=True, na=False
        ).to_numpy()
        label = "internship/junior titles excluded"
    else:
        exp_mask = None
        label = ""

    if exp_mask is not None:

        combined = mask & exp_mask

        if combined.any():
            mask = combined
            notes.append(f"Experience level ({experience}): {label}.")
        else:
            notes.append(
                f"No postings matched experience level '{experience}' together "
                f"with your other choices, so that filter was not applied."
            )

    if mask.sum() < top_k and mask.sum() < len(titles):
        notes.append(
            f"Only {int(mask.sum())} posting(s) fit your preferences, "
            f"fewer than the {top_k} requested."
        )

    return mask, notes


# ======================================================================
# TOP-K SELECTION: DEDUPLICATION + MINIMUM RELEVANCE
# ======================================================================

def job_dedup_key(corpus: job_ranker.JobCorpus, row_index: int) -> str:
    """
    Two records are the same posting only if title, employer, location AND
    description text all agree. Same-title vacancies that differ in
    employer, location or requirements stay as distinct alternatives.
    """
    columns = corpus.columns
    row = corpus.frame.iloc[row_index]

    def norm(column: str | None) -> str:
        return mc.clean_text(row[column]).lower() if column else ""

    description_hash = hashlib.md5(
        norm(columns["description"]).encode("utf-8", errors="replace")
    ).hexdigest()

    return "|".join(
        (
            norm(columns["title"]),
            norm(columns["company"]),
            norm(columns["location"]),
            description_hash,
        )
    )


# Words that name the same kind of role.
_ROLE_SYNONYMS = {
    "programmer": "developer",
    "engineer": "developer",
    "coder": "developer",
    "dev": "developer",
    "swe": "developer",
}

# Seniority, contract and filler words that do not change the kind of role.
_TITLE_NOISE = {
    "senior", "sr", "junior", "jr", "lead", "principal", "staff", "associate",
    "intern", "internship", "trainee", "entry", "level", "i", "ii", "iii", "iv",
    "remote", "contract", "contractor", "temporary", "full", "time", "part",
    "the", "a", "an", "and", "of", "for", "in", "at", "to", "with",
}

# "Business Development Executive- Carlisle, PA" -> keep only the role part.
_TITLE_TAIL_SPLIT = re.compile(r"\s*[-\u2013\u2014]\s+|\s+[-\u2013\u2014]\s*|[,|(]")


def canonical_title_tokens(title: str) -> frozenset[str]:
    """Role words of a job title, without seniority, location or filler."""
    role_part = _TITLE_TAIL_SPLIT.split(str(title))[0]

    tokens = [
        _ROLE_SYNONYMS.get(token, token)
        for token in esco_vocab.tokenize(role_part)
    ]

    return frozenset(token for token in tokens if token not in _TITLE_NOISE)


def titles_are_lookalikes(a: frozenset[str], b: frozenset[str]) -> bool:
    """
    True when two canonical titles are the same role:
      * identical ("Information Technology Developer" and
        "Information Technology Programmer"),
      * they share at least two role words AND those make up at least
        two-thirds of the shorter title
        ("Salesforce Developer" / "Salesforce Vlocity/OmniStudio Developer",
        "Business Development Executive" / "Business Development Manager
        Healthcare"). One shared word ("Data Analyst" / "Data Engineer")
        is not enough.
    """
    if not a or not b:
        return False

    if a == b:
        return True

    shared = len(a & b)

    if shared < 2:
        return False

    return shared / min(len(a), len(b)) >= 2 / 3


def _is_lookalike(
    corpus: job_ranker.JobCorpus,
    row_index: int,
    selected_rows: list[int],
) -> bool:
    """True when this posting is essentially a role already selected."""
    tokens = canonical_title_tokens(corpus.titles[row_index])
    vector = corpus.embeddings[row_index]
    norm = float(np.linalg.norm(vector))

    for other in selected_rows:

        if titles_are_lookalikes(
            tokens, canonical_title_tokens(corpus.titles[other])
        ):
            return True

        other_vector = corpus.embeddings[other]
        denominator = norm * float(np.linalg.norm(other_vector))

        if (
            denominator > 0
            and float(vector @ other_vector) / denominator
            >= LOOKALIKE_TEXT_SIMILARITY
        ):
            return True

    return False


def select_top_unique_jobs(
    ranked: pd.DataFrame,
    corpus: job_ranker.JobCorpus,
    top_k: int,
    min_score: float,
):
    """
    Best ``top_k`` postings, best first, skipping exact repeats, look-alike
    roles and weak matches. Skipped jobs are replaced by the next-best
    different role.
    """
    seen: set[str] = set()
    selected: list[int] = []
    selected_rows: list[int] = []
    exact_duplicates = 0
    lookalikes = 0

    scores = ranked["hybrid_match_score"].to_numpy()
    below_minimum = int((scores < min_score).sum())

    for position, (row_index, score) in enumerate(
        zip(ranked["row_index"], scores)
    ):

        if len(selected) >= top_k or score < min_score:
            break

        row_index = int(row_index)

        key = job_dedup_key(corpus, row_index)

        if key in seen:
            exact_duplicates += 1
            continue

        if _is_lookalike(corpus, row_index, selected_rows):
            lookalikes += 1
            continue

        seen.add(key)
        selected.append(position)
        selected_rows.append(row_index)

    top = ranked.iloc[selected].copy().reset_index(drop=True)

    return top, exact_duplicates, lookalikes, below_minimum


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


def deterministic_explanation(
    job_title: str,
    matched_skills: list[str],
    skill_gaps: list[str],
    evidence_level: str = "good",
    job_skill_count: int = 0,
    claim_label: str = "experimental_similarity_not_validated",
):
    validated = claim_label == "human_validated_relevance"
    result_phrase = "match" if validated else "similarity result"
    section_title = "Why this role matches" if validated else "Why this result appears similar"

    matched_text = ", ".join(
        matched_skills[:6]
    )

    gap_text = ", ".join(
        skill_gaps[:6]
    )

    if matched_text:

        why = (
            f"Your resume and this {job_title} posting both mention "
            f"{matched_text}."
        )

    else:

        why = (
            f"The text in your resume has some semantic overlap with the "
            f"{job_title} posting, although few direct skill overlaps were detected."
        )

    if evidence_level == "insufficient":
        why += (
            " The posting gave too little specific skill information, "
            f"so this {result_phrase} mostly reflects text similarity."
        )
    elif evidence_level == "limited":
        why += (
            f" Only {job_skill_count} specific skill(s) could be extracted "
            f"from the posting, so skill evidence is limited."
        )

    if matched_text:

        skills = (
            f"Detected in both your resume and the posting: "
            f"{matched_text}."
        )

    else:

        skills = (
            "No direct skill overlap was identified "
            "from the ESCO skill vocabulary."
        )

    if gap_text:

        improvement = (
            f"Skills mentioned in this posting but not detected in "
            f"your resume include {gap_text}."
        )

    elif evidence_level == "insufficient":

        improvement = (
            "Not enough specific skill requirements could be extracted "
            "from this posting to identify gaps reliably."
        )

    elif evidence_level == "limited":

        improvement = (
            f"No gaps among the {job_skill_count} specific skill(s) "
            f"detected, but the posting gave limited skill information, "
            f"so please read the description yourself."
        )

    else:

        improvement = (
            f"No gaps were found among the {job_skill_count} specific "
            f"skills detected for this role."
        )

    return (
        f"**{section_title}**\n"
        f"{why}\n\n"
        f"**Skills you already have**\n"
        f"{skills}\n\n"
        f"**What you could improve**\n"
        f"{improvement}"
    )


def generate_user_friendly_explanation(
    client,
    job_title: str,
    semantic_score: float,
    matched_skills: list[str],
    skill_gaps: list[str],
    evidence_level: str = "good",
    job_skill_count: int = 0,
    claim_label: str = "experimental_similarity_not_validated",
):
    validated = claim_label == "human_validated_relevance"
    role_header = "Why this role matches" if validated else "Why this result appears similar"
    result_phrase = "match" if validated else "similarity result"

    if client is None:
        return deterministic_explanation(
            job_title,
            matched_skills,
            skill_gaps,
            evidence_level,
            job_skill_count,
            claim_label,
        )

    matched_text = ", ".join(
        matched_skills[:12]
    ) or "No direct skill matches identified"

    gaps_text = ", ".join(
        skill_gaps[:12]
    ) or "None detected"

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

Skill evidence quality for this posting:
{evidence_level} ({job_skill_count} specific job skills detected)

Skills found in both resume and job:
{matched_text}

Skills mentioned in the job posting but not detected in the resume:
{gaps_text}

Claim label: {claim_label}

Rules:
- Mention ONLY skills that are listed above. Do not infer or invent other skills or abilities.
- {("You may describe the result as human-judgment-validated relevance, but never as hiring probability." if validated else "Describe this only as experimental text similarity. Do not call the role a match, suitable, or a suitability recommendation.")}
- Do not write placeholders or parenthetical notes about "other" skills.
- The resume evidence is keyword detection only. Do not describe proficiency or
  seniority (avoid words like strong, proven, extensive, expert, demonstrates).
- Do not describe what this kind of role typically requires; refer only to the
  skills listed above. The posting does not say whether a skill is required or optional.
- If the skill evidence quality is "limited" or "insufficient", say plainly that the posting
  gave little skill information and the {result_phrase} mostly reflects text similarity.
  Never say there are no skill gaps in that case.

Write exactly three short sections:

{role_header}:
Explain the strongest evidence without overstating it.

Skills you already have:
Mention the most relevant matching skills.

What you could improve:
Mention the most relevant skills the posting mentions that were not detected, if any.

Keep the explanation concise and useful to the user.
"""

    try:

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        (
                            "You explain evidence from resume and job text. "
                            if not validated else "You explain human-judgment-validated resume-job relevance. "
                        )
                        + "Always respond in normal human-readable English."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=500,
        )

        text = response.choices[0].message.content

        if text:
            return text.strip()

    except Exception:

        return (
            deterministic_explanation(
                job_title,
                matched_skills,
                skill_gaps,
                evidence_level,
                job_skill_count,
                claim_label,
            )
            + "\n\n"
              "*LLM explanation unavailable; evidence-based "
              "fallback used.*"
        )

    return deterministic_explanation(
        job_title,
        matched_skills,
        skill_gaps,
        evidence_level,
        job_skill_count,
        claim_label,
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

    markup = "".join(
        f'<span class="{css_class}">{html.escape(skill)}</span>'
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
    top_k: int,
    preferences: dict | None = None
):

    preferences = preferences or {}

    # --------------------------------------------------------------
    # 1. TEXT (OCR fallback for scanned / image-only PDFs)
    # --------------------------------------------------------------

    st.write(
        "📄 Extracting resume text..."
    )

    extraction = read_resume(
        resume_bytes
    )

    if not extraction.usable:

        raise ResumeReadError(
            extraction.user_message()
            or "No readable text was found in this PDF."
        )

    resume_text = extraction.text

    st.success(
        f"Extracted {len(resume_text):,} characters"
        + (
            f" using OCR ({extraction.ocr_engine})"
            if extraction.ocr_used
            else ""
        )
    )

    for warning in extraction.warnings:
        st.warning(warning)

    # --------------------------------------------------------------
    # 2. MODEL, SKILL VOCABULARY, SCORER
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
            "ESCO skills could not be loaded, so skill matching is off: "
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
    # 3. JOB CORPUS (embeddings + skill sets; cached on disk)
    # --------------------------------------------------------------

    st.write(
        "📊 Preparing job postings (the first run builds caches "
        "and can take a few minutes)..."
    )

    corpus = load_corpus(
        get_file_signature(JOB_DATA_PATH),
        vocabulary_signature()
    )

    st.success(
        f"Loaded {len(corpus):,} job postings"
    )

    # --------------------------------------------------------------
    # 4. RESUME SKILLS (ESCO skills table only)
    # --------------------------------------------------------------

    st.write(
        "🔎 Extracting resume skills..."
    )

    resume_skill_map = job_ranker.resume_skill_uris(
        resume_text,
        vocabulary
    )

    st.success(
        f"Detected {len(resume_skill_map)} skills"
    )

    # --------------------------------------------------------------
    # 5. RESUME EMBEDDING
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
    # 6. SCORE EVERY JOB (matching_core.MatchScorer)
    # --------------------------------------------------------------

    st.write(
        "📈 Scoring and ranking jobs..."
    )

    ranked = job_ranker.rank_resume(
        corpus,
        resume_text=resume_text,
        resume_embedding=resume_embedding,
        resume_skills=resume_skill_map.keys(),
        scorer=scorer
    )

    # --------------------------------------------------------------
    # 7. PREFERENCES, DUPLICATES, MINIMUM MATCH
    # --------------------------------------------------------------

    scope_mask, scope_notes = build_job_scope_mask(
        corpus.titles,
        preferences.get("interests", []),
        preferences.get("experience", "No preference"),
        top_k,
    )

    ranked = ranked[
        ranked["row_index"].isin(np.flatnonzero(scope_mask))
    ].reset_index(drop=True)

    min_match = float(preferences.get("min_match", DEFAULT_MIN_MATCH))

    (
        top,
        duplicates_skipped,
        lookalikes_skipped,
        below_minimum,
    ) = select_top_unique_jobs(
        ranked,
        corpus,
        top_k,
        min_match,
    )

    if duplicates_skipped:
        scope_notes.append(
            f"Skipped {duplicates_skipped} repeated posting(s) "
            f"(same title, employer, location and description)."
        )

    if lookalikes_skipped:
        message = (
            f"Skipped {lookalikes_skipped} look-alike posting(s) that were "
            f"very similar to a higher-ranked recommendation, so the list "
            f"shows different kinds of roles."
        )
        if len(top) < top_k and not below_minimum:
            message += (
                f" Only {len(top)} distinct role(s) remain in your "
                f"preferences; broaden your career interests to see more."
            )
        scope_notes.append(message)

    if len(top) < top_k and min_match > 0 and below_minimum:
        scope_notes.append(
            f"Showing {len(top)} of {top_k} requested jobs. "
            f"{below_minimum:,} in-scope posting(s) scored below your "
            f"{min_match:.0%} minimum overall match, so weaker roles were "
            f"left out instead of filling the list."
        )

    # --------------------------------------------------------------
    # 8. JOB METADATA, SKILL BREAKDOWN AND TRACE FOR TOP-K
    # --------------------------------------------------------------

    columns = corpus.columns
    job_share = compute_skill_job_share(corpus)

    companies = []
    locations = []
    descriptions = []
    views = []
    trace_rows = []

    for row_index in top["row_index"].astype(int):

        row = corpus.frame.iloc[row_index]

        companies.append(
            mc.clean_text(row[columns["company"]])
            if columns["company"]
            else ""
        )

        locations.append(
            mc.clean_text(row[columns["location"]])
            if columns["location"]
            else ""
        )

        descriptions.append(
            mc.clean_text(row[columns["description"]])
        )

        views.append(
            build_skill_view(
                corpus,
                row_index,
                resume_skill_map,
                job_share
            )
        )

        trace_rows.append(
            build_job_skill_rows(
                corpus,
                row_index,
                resume_skill_map,
                job_share
            )
        )

    top["company"] = companies
    top["location"] = locations
    top["description"] = descriptions
    top["skill_trace_rows"] = trace_rows

    for key in (
        "matched_skills",
        "skill_gaps",
        "common_skills",
        "specific_skill_count",
        "matched_specific_count",
        "evidence_level",
    ):
        top[key] = [view[key] for view in views]

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

        explanations.append(
            generate_user_friendly_explanation(
                client,
                row["job_title"],
                row["semantic_similarity"],
                row["matched_skills"],
                row["skill_gaps"],
                row["evidence_level"],
                int(row["specific_skill_count"]),
                scorer.claim_label,
            )
        )

        progress.progress(
            (i + 1) / len(top)
        )

    top["explanation"] = explanations

    status.empty()

    return {
        "resume_text": resume_text,
        "extraction": extraction.to_record(),
        "resume_skills": sorted(
            resume_skill_map.values(),
            key=str.lower
        ),
        "resume_skill_rows": build_resume_skill_rows(
            vocabulary,
            resume_text,
            resume_skill_map
        ),
        "skill_matching_enabled": vocabulary is not None,
        "scoring_method": scorer.method,
        "scoring_description": scorer.description,
        "claim_label": scorer.claim_label,
        "claim_description": scorer.claim_description,
        "suitability_wording_allowed": scorer.suitability_wording_allowed,
        "scorer_fingerprint": scorer.scorer_fingerprint,
        "candidate_policy": "semantic_top_k_union_lexical_top_k_union_esco_skill_top_k",
        "all_recommendations": ranked,
        "top_recommendations": top,
        "total_jobs": len(corpus),
        "scope_notes": scope_notes,
        "preferences": preferences,
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
        + result.get("scoring_description", "unknown")
    )

    if not result.get("suitability_wording_allowed", False):
        st.warning(
            "Experimental similarity scores — not validated against sufficient human relevance judgments. "
            "These scores are not hiring probabilities and should not be interpreted as proof that a role is suitable."
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

    for note in result.get("scope_notes", []):
        st.info(note)

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
            f"{len(all_jobs):,} of {result.get('total_jobs', len(all_jobs)):,}"
        )

    with c4:
        st.metric(
            "Top Suitability" if result.get("suitability_wording_allowed", False) else "Top Similarity",
            score_percent(
                top.iloc[0][
                    "hybrid_match_score"
                ]
            )
            if len(top)
            else "N/A"
        )

    if len(top) == 0:
        st.warning(
            "No postings met the current filters and score threshold. "
            "Try lowering the threshold or broadening your career interests."
        )
        return

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

    if len(resume_skills) > 20:
        st.caption(
            f"Showing 20 of {len(resume_skills)} detected skills."
        )

    if result.get("resume_skill_rows") and st.toggle(
        "🔍 Show which phrase in your resume produced each skill",
        key="resume_skill_trace"
    ):
        st.dataframe(
            pd.DataFrame(result["resume_skill_rows"]),
            hide_index=True
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
            f"• {'Suitability' if result.get('suitability_wording_allowed', False) else 'Similarity'} "
            f"{score_percent(row['hybrid_match_score'])}",
            expanded=(rank == 1)
        ):

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Overall Suitability" if result.get("suitability_wording_allowed", False) else "Overall Similarity",
                    score_percent(
                        row[
                            "hybrid_match_score"
                        ]
                    )
                )

            with c2:

                st.metric(
                    "Semantic Similarity",
                    score_percent(
                        row[
                            "semantic_similarity"
                        ]
                    )
                )

            with c3:

                specific_count = int(row["specific_skill_count"])

                if specific_count:
                    st.metric(
                        "Required-skill coverage",
                        f"{int(row['matched_specific_count'])} of {specific_count}"
                    )
                    if row["evidence_level"] == "limited":
                        st.caption(
                            f"⚠️ Limited evidence: only {specific_count} "
                            f"specific skill(s) detected in this posting."
                        )
                else:
                    st.metric("Required-skill coverage", "N/A")
                    st.caption(
                        "⚠️ Insufficient job-skill evidence for this posting."
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

            st.caption(
                "Skills mentioned in this posting that were not detected in "
                "your resume. The posting may list them as required or "
                "optional; the app cannot tell which."
            )

            if row["skill_trace_rows"] and st.toggle(
                "🔍 Show where these skills came from in the posting",
                key=f"skill_trace_{rank}"
            ):
                st.dataframe(
                    pd.DataFrame(row["skill_trace_rows"]),
                    hide_index=True
                )

            st.markdown(
                "### 💡 Why This Job Matches"
                if row.get("claim_label") == "human_validated_relevance"
                else "### 💡 Why This Result Appears Similar"
            )

            # LLM text is rendered as Markdown, never as raw HTML.
            with st.container(border=True):
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

    # --------------------------------------------------------------
    # AGGREGATE SKILL GAPS
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-title">📈 Overall Skill Gap Analysis</div>',
        unsafe_allow_html=True
    )

    gap_jobs: dict[str, list[str]] = {}

    for _, row in top.iterrows():

        for gap in row["skill_gaps"]:

            gap_jobs.setdefault(gap, []).append(
                row["job_title"]
            )

    sorted_gaps = sorted(
        gap_jobs.items(),
        key=lambda x: (-len(x[1]), x[0].lower())
    )

    jobs_with_evidence = int(
        (top["specific_skill_count"] > 0).sum()
    )
    jobs_with_limited = int(
        (top["evidence_level"] != "good").sum()
    )

    if sorted_gaps:

        render_skills(
            [
                gap
                for gap, _ in sorted_gaps[:20]
            ],
            gap=True
        )

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Skill": gap,
                        "Recommended jobs mentioning it": len(titles),
                        "Jobs": "; ".join(titles),
                    }
                    for gap, titles in sorted_gaps[:20]
                ]
            ),
            hide_index=True
        )

        if jobs_with_limited:
            st.caption(
                f"{jobs_with_limited} of {len(top)} recommended postings "
                f"gave limited skill information, so this list may be incomplete."
            )

    elif jobs_with_evidence == 0:

        st.info(
            "Insufficient job-skill evidence to calculate gaps reliably: "
            "the recommended postings did not contain enough specific skills."
        )

    else:

        st.success(
            "No gaps were found among the specific skills detected "
            "in the recommended jobs."
        )

        if jobs_with_limited:
            st.caption(
                f"{jobs_with_limited} of {len(top)} postings gave limited "
                f"skill information, so this does not guarantee there are no gaps."
            )

    # --------------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------------

    st.markdown(
        '<div class="section-title">💾 Export Results</div>',
        unsafe_allow_html=True
    )

    export_df = top.drop(
        columns=["row_index", "skill_trace_rows"],
        errors="ignore"
    ).copy()

    for column in (
        "matched_skills",
        "skill_gaps",
        "common_skills",
    ):
        export_df[column] = export_df[column].apply(
            lambda x: ", ".join(x)
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
            "Resume text extraction (OCR fallback)",
            "ESCO skill vocabulary",
            "Skill extraction",
            "Semantic embedding",
            "Job similarity",
            "Skill matching",
            "Scoring (trained model or formula)",
            "Preferences, de-duplication, threshold",
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
                    "No ESCO skills loaded. Skill matching is off.\n\n"
                    + vocabulary_error
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
        help="Text-based and scanned PDFs are supported "
             "(scanned PDFs need an OCR engine)."
    )

    st.markdown(
        '<div class="section-title">🎛️ Resume Analysis Preferences</div>',
        unsafe_allow_html=True
    )

    interests = st.multiselect(
        "Which job roles interest you?",
        options=INTEREST_OPTIONS,
        default=[],
        help="Leave empty to consider every role. Matching uses job titles."
    )

    pref_col1, pref_col2, pref_col3 = st.columns(3)

    with pref_col1:
        experience = st.selectbox(
            "Experience level",
            options=EXPERIENCE_OPTIONS,
            index=0
        )

    with pref_col2:
        top_k = st.selectbox(
            "Number of job recommendations",
            options=[5, 10],
            index=0
        )

    with pref_col3:
        active_scorer = load_scorer(scorer_signature())
        threshold_label = (
            "Minimum suitability score (%)" if active_scorer.suitability_wording_allowed
            else "Minimum similarity score (%)"
        )
        min_match_percent = st.slider(
            threshold_label,
            min_value=0,
            max_value=80,
            value=int(DEFAULT_MIN_MATCH * 100),
            step=5,
            help="Jobs scoring below this are left out, so fewer than the "
                 "requested number may be shown. 0 turns the cut-off off."
        )

    preferences = {
        "interests": interests,
        "experience": experience,
        "min_match": min_match_percent / 100,
    }

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
        # Skip re-analysis only when the resume, the preferences and
        # the scoring inputs are all unchanged.
        # ----------------------------------------------------------

        analysis_key = hashlib.md5(
            resume_bytes
        ).hexdigest() + "|" + json.dumps(
            {
                **preferences,
                "top_k": top_k,
                "scorer": scorer_signature(),
                "vocabulary": vocabulary_signature(),
                "jobs": get_file_signature(JOB_DATA_PATH),
            },
            sort_keys=True
        )

        if (
            st.session_state.get("resume_hash") == analysis_key
            and "analysis_result" in st.session_state
        ):

            st.info(
                "This resume has already been analyzed with these preferences."
            )

        else:

            try:

                with st.status(
                    "Running resume analysis pipeline...",
                    expanded=True
                ):

                    result = analyze_resume(
                        resume_bytes,
                        top_k,
                        preferences
                    )

                st.session_state[
                    "analysis_result"
                ] = result

                st.session_state[
                    "resume_hash"
                ] = analysis_key

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