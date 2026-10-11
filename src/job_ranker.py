"""
======================================================================
AI RESUME ANALYZER - SHARED ONLINE JOB RANKER
======================================================================

Ranks every job in the job dataset for ONE resume. Used by

    Step 13 (13_explainable_rag_recommendation.py)   command line
    Step 14 (14_app.py)                              Streamlit app

so both rank identically. They previously each carried their own copy of
the pipeline (different formulas, different skill vocabularies, even a
different job-text builder sharing one embedding cache file).

Pipeline
--------
    build_job_corpus(...)     once per job dataset (cached on disk)
        job text           matching_core.build_job_text
        job embeddings     data/cache/job_embeddings.npy (+ .meta.json)
        job skill sets     ESCO skills only (esco_vocab) +
                           exact ESCO match of the dataset's own
                           ``job_skill_set`` column, if it has one
        token indexes      for title similarity / text containment

    rank_jobs(corpus, ...)    once per resume
        features   matching_core.build_feature_frame
        score      matching_core.MatchScorer  (model or formula)

The caches carry a fingerprint (job text, embedding model, vocabulary),
so they are rebuilt when the data changes - the old caches only checked
the number of rows.
======================================================================
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

import matching_core as mc
import candidate_retrieval
from esco_vocab import ESCOVocabulary

logger = logging.getLogger(__name__)

CACHE_SCHEMA = 2

EMBEDDINGS_FILE = "job_embeddings.npy"
EMBEDDINGS_META_FILE = "job_embeddings.meta.json"
SKILL_INDEX_FILE = "job_skill_index.json"

ProgressFn = Callable[[str, float], None]


# ======================================================================
# JOB COLUMN DETECTION
# ======================================================================

def _find_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    *,
    allow_substring: bool = False,
) -> Optional[str]:

    lowered = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    if allow_substring:
        for column in df.columns:
            for candidate in candidates:
                if candidate.lower() in str(column).lower():
                    return column

    return None


def detect_job_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Map roles to this dataset's column names. title/description required."""

    columns = {
        "title": _find_column(
            df, ["job_title", "title", "jobtitle", "position"],
            allow_substring=True,
        ),
        "description": _find_column(
            df, ["job_description", "description", "jobdescription", "details"],
            allow_substring=True,
        ),
        "id": _find_column(df, ["job_id", "jobid", "id"]),
        "company": _find_column(df, ["company", "company_name", "employer"]),
        "location": _find_column(df, ["location", "job_location", "city"]),
        "skills": _find_column(
            df, ["job_skill_set", "job_skills", "skill_set", "skills"]
        ),
        "category": _find_column(df, ["category", "job_category"]),
    }

    missing = [
        role for role in ("title", "description") if columns[role] is None
    ]

    if missing:
        raise ValueError(
            f"Job dataset is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return columns


def parse_skill_list(value: Any) -> List[str]:
    """``job_skill_set`` cell -> list of strings (list / JSON / Python repr)."""

    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        raw = list(value)

    else:

        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass

        text = str(value).strip()

        if not text:
            return []

        raw = None

        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(text)
            except (ValueError, SyntaxError):
                continue

            if isinstance(parsed, (list, tuple, set)):
                raw = list(parsed)
                break

            if isinstance(parsed, str):
                raw = [parsed]
                break

        if raw is None:
            raw = [part for part in text.replace(";", ",").split(",")]

    return [mc.clean_text(item) for item in raw if mc.clean_text(item)]


# ======================================================================
# CSR INDEX (token sets as compact int arrays)
# ======================================================================

class SetIndex:
    """
    N sets of strings stored as one flat int array.

    ``count_present(items)`` returns, for every set, how many of its
    members are in ``items``; ``fraction_present`` divides by the set's
    size. Both are vectorised, so scoring 10k+ jobs takes milliseconds.
    """

    def __init__(self, sets: Iterable[Iterable[str]]) -> None:

        self.vocab: Dict[str, int] = {}

        indptr = [0]
        indices: List[int] = []

        for members in sets:

            ids = sorted(
                {
                    self.vocab.setdefault(member, len(self.vocab))
                    for member in members
                }
            )

            indices.extend(ids)
            indptr.append(len(indices))

        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indptr) - 1

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.indptr)

    def count_present(self, items: Iterable[str]) -> np.ndarray:

        mask = np.zeros(len(self.vocab), dtype=bool)

        for item in items:

            position = self.vocab.get(item)

            if position is not None:
                mask[position] = True

        cumulative = np.concatenate(
            ([0], np.cumsum(mask[self.indices], dtype=np.int64))
        )

        return cumulative[self.indptr[1:]] - cumulative[self.indptr[:-1]]

    def fraction_present(self, items: Iterable[str]) -> np.ndarray:

        counts = self.count_present(items).astype(float)
        sizes = self.sizes.astype(float)

        out = np.zeros(len(sizes), dtype=float)

        np.divide(counts, sizes, out=out, where=sizes > 0)

        return out


# ======================================================================
# CORPUS
# ======================================================================

@dataclass
class JobCorpus:
    frame: pd.DataFrame
    columns: Dict[str, Optional[str]]
    titles: List[str]
    texts: List[str]
    embeddings: np.ndarray
    title_index: SetIndex
    text_index: SetIndex
    skill_index: SetIndex
    skill_sets: List[frozenset]
    vocabulary: Optional[ESCOVocabulary]
    fingerprint: str
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frame)

    def label(self, uri: str) -> str:

        if self.vocabulary is not None:
            return self.vocabulary.label_for(uri)

        return uri.rsplit("/", 1)[-1]


def _text_fingerprint(texts: Sequence[str]) -> str:

    digest = hashlib.sha1()

    for text in texts:
        digest.update(text.encode("utf-8", "ignore"))
        digest.update(b"\x1f")

    return digest.hexdigest()


def _skill_cells_fingerprint(values: Optional[Sequence[Any]]) -> str:
    """Deterministic fingerprint of dataset-provided skill cells."""
    if values is None:
        return "no-skill-column"
    canonical = [sorted(set(parse_skill_list(value))) for value in values]
    payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8", "ignore")).hexdigest()


# ---- embeddings -------------------------------------------------------------

def load_or_build_embeddings(
    texts: Sequence[str],
    encode: Callable[[List[str]], np.ndarray],
    *,
    model_name: str,
    cache_dir: Path,
    text_digest: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
) -> tuple:
    """
    Job embeddings from the cache when it matches this exact job text and
    embedding model, otherwise computed and cached. Returns
    ``(embeddings, was_cached)``.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)

    array_path = cache_dir / EMBEDDINGS_FILE
    meta_path = cache_dir / EMBEDDINGS_META_FILE

    digest = text_digest or _text_fingerprint(texts)

    expected = {
        "schema": CACHE_SCHEMA,
        "model": model_name,
        "text_sha1": digest,
        "rows": len(texts),
    }

    if array_path.is_file() and meta_path.is_file():

        try:

            meta = json.loads(meta_path.read_text(encoding="utf-8"))

            if all(meta.get(key) == value for key, value in expected.items()):

                cached = np.load(array_path)

                if cached.ndim == 2 and cached.shape[0] == len(texts):
                    return cached.astype(np.float32, copy=False), True

        except Exception as exc:  # corrupt cache -> rebuild
            logger.warning("Ignoring unreadable embedding cache: %s", exc)

    if progress:
        progress("Embedding job postings (first run, cached afterwards)...", 0.0)

    embeddings = np.asarray(encode(list(texts)), dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError(
            f"Embedding function returned shape {embeddings.shape} for "
            f"{len(texts)} texts."
        )

    np.save(array_path, embeddings)

    meta_path.write_text(
        json.dumps({**expected, "dim": int(embeddings.shape[1])}),
        encoding="utf-8",
    )

    return embeddings, False


# ---- skills -----------------------------------------------------------------

def build_job_skill_sets(
    texts: Sequence[str],
    dataset_skill_cells: Optional[Sequence[Any]],
    vocabulary: ESCOVocabulary,
    *,
    cache_dir: Path,
    text_digest: str,
    progress: Optional[ProgressFn] = None,
) -> tuple:
    """
    ESCO skill URIs for every job. Returns ``(skill_sets, was_cached)``.

    A job's skills are the ESCO skills mentioned in its text, plus any
    entry of the dataset's own ``job_skill_set`` that exactly matches an
    ESCO label (the same exact-match rule Step 6 applies first; fuzzy
    matches are not forced).
    """

    cache_path = cache_dir / SKILL_INDEX_FILE

    expected = {
        "schema": CACHE_SCHEMA,
        "vocabulary": vocabulary.fingerprint,
        "text_sha1": text_digest,
        "skill_cells_sha256": _skill_cells_fingerprint(dataset_skill_cells),
        "rows": len(texts),
        "uses_skill_column": dataset_skill_cells is not None,
    }

    if cache_path.is_file():

        try:

            payload = json.loads(cache_path.read_text(encoding="utf-8"))

            if all(payload.get(k) == v for k, v in expected.items()):

                uris = payload["uris"]

                return (
                    [frozenset(uris[i] for i in row) for row in payload["jobs"]],
                    True,
                )

        except Exception as exc:
            logger.warning("Ignoring unreadable job skill cache: %s", exc)

    skill_sets: List[frozenset] = []

    total = max(len(texts), 1)

    for position, text in enumerate(texts):

        found = {
            item["concept_uri"] or f"id:{item['skill_id']}"
            for item in vocabulary.extract(text)
        }

        if dataset_skill_cells is not None:

            for entry in parse_skill_list(dataset_skill_cells[position]):

                skill_id = vocabulary.lookup_term(entry)

                if skill_id is not None:

                    skill = vocabulary.skills[skill_id]

                    found.add(skill["concept_uri"] or f"id:{skill_id}")

        skill_sets.append(frozenset(found))

        if progress and position % 250 == 0:
            progress("Extracting skills from job postings...", position / total)

    unique = sorted({uri for row in skill_sets for uri in row})
    lookup = {uri: i for i, uri in enumerate(unique)}

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_path.write_text(
        json.dumps(
            {
                **expected,
                "uris": unique,
                "jobs": [sorted(lookup[u] for u in row) for row in skill_sets],
            }
        ),
        encoding="utf-8",
    )

    return skill_sets, False


# ---- corpus -----------------------------------------------------------------

def build_job_corpus(
    jobs: pd.DataFrame,
    *,
    encode: Callable[[List[str]], np.ndarray],
    model_name: str = mc.SEMANTIC_MODEL_NAME,
    vocabulary: Optional[ESCOVocabulary],
    cache_dir: Path,
    progress: Optional[ProgressFn] = None,
) -> JobCorpus:
    """
    Everything that depends only on the job dataset.

    ``vocabulary=None`` (no ESCO file) still works: skill sets are empty,
    so ranking falls back to semantic + lexical signals, and ``notes``
    says so.
    """

    jobs = jobs.reset_index(drop=True)

    columns = detect_job_columns(jobs)

    titles = [mc.clean_text(v) for v in jobs[columns["title"]]]

    texts = [
        mc.build_job_text(title, description)
        for title, description in zip(jobs[columns["title"]], jobs[columns["description"]])
    ]

    digest = _text_fingerprint(texts)

    notes: List[str] = []

    embeddings, embeddings_cached = load_or_build_embeddings(
        texts,
        encode,
        model_name=model_name,
        cache_dir=cache_dir,
        text_digest=digest,
        progress=progress,
    )

    if vocabulary is not None:

        skill_cells = (
            jobs[columns["skills"]].tolist() if columns["skills"] else None
        )

        skill_sets, _ = build_job_skill_sets(
            texts,
            skill_cells,
            vocabulary,
            cache_dir=cache_dir,
            text_digest=digest,
            progress=progress,
        )

    else:

        skill_sets = [frozenset() for _ in texts]

        notes.append(
            "No ESCO skills file was found, so skill matching is disabled "
            "and jobs are ranked on semantic and lexical similarity only."
        )

    if progress:
        progress("Indexing job text...", 1.0)

    skill_cells_digest = _skill_cells_fingerprint(
        jobs[columns["skills"]].tolist() if columns["skills"] else None
    )
    corpus_fingerprint = hashlib.sha256(
        "|".join(
            [
                digest,
                skill_cells_digest,
                model_name,
                vocabulary.fingerprint if vocabulary else "no-vocabulary",
                str(CACHE_SCHEMA),
            ]
        ).encode()
    ).hexdigest()

    return JobCorpus(
        frame=jobs,
        columns=columns,
        titles=titles,
        texts=texts,
        embeddings=embeddings,
        title_index=SetIndex(mc.token_set(t) for t in titles),
        text_index=SetIndex(mc.token_set(t) for t in texts),
        skill_index=SetIndex(skill_sets),
        skill_sets=skill_sets,
        vocabulary=vocabulary,
        fingerprint=corpus_fingerprint,
        notes=notes,
    )


# ======================================================================
# RANKING
# ======================================================================

def resume_skill_uris(
    resume_text: str,
    vocabulary: Optional[ESCOVocabulary],
) -> Dict[str, str]:
    """``{uri: label}`` for the ESCO skills found in a resume."""

    if vocabulary is None:
        return {}

    return {
        (item["concept_uri"] or f"id:{item['skill_id']}"): item["preferred_label"]
        for item in vocabulary.extract(resume_text)
    }


def rank_jobs(
    corpus: JobCorpus,
    *,
    resume_text: str,
    resume_embedding: np.ndarray,
    resume_skills: Iterable[str],
    scorer: mc.MatchScorer,
    max_candidates_per_signal: int = candidate_retrieval.DEFAULT_CANDIDATES_PER_SIGNAL,
) -> pd.DataFrame:
    """Retrieve a shared candidate union, then score/rank those job rows.

    Offline Step 7 uses the same semantic + lexical + skill union policy.
    Signals are computed against the entire corpus, but model/formula scores
    are only evaluated for candidate rows. ``row_index`` always points back
    to ``corpus.frame``.
    """
    skills = set(resume_skills)
    vector = np.asarray(resume_embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm

    embeddings = corpus.embeddings
    if embeddings.shape[1] != vector.shape[0]:
        raise ValueError(
            f"Resume embedding has {vector.shape[0]} dimensions but the job "
            f"embeddings have {embeddings.shape[1]}; delete the job embedding "
            f"cache or use the same model."
        )

    cosine = embeddings @ vector
    resume_tokens = mc.token_set(resume_text)
    features = mc.build_feature_frame(
        semantic_cosine=cosine,
        title_sim=corpus.title_index.fraction_present(resume_tokens),
        containment=corpus.text_index.fraction_present(resume_tokens),
        shared=corpus.skill_index.count_present(skills),
        resume_count=len(skills),
        job_count=corpus.skill_index.sizes,
    )

    selection = candidate_retrieval.select_candidate_indices(
        resume_text=resume_text,
        job_texts=corpus.texts,
        resume_skills=skills,
        job_skill_sets=corpus.skill_sets,
        semantic_cosine=cosine,
        lexical_scores=features["text_containment"].to_numpy(),
        shared_skill_counts=features["shared_skill_count"].to_numpy(),
        skill_jaccard_scores=features["jaccard_similarity"].to_numpy(),
        max_candidates_per_signal=max_candidates_per_signal,
    )
    candidate_indices = np.asarray(selection.indices, dtype=np.int64)
    candidate_features = features.iloc[candidate_indices].copy()
    if candidate_features.empty:
        return pd.DataFrame(columns=[
            "row_index", "job_id", "job_title", *mc.MODEL_FEATURES,
            "hybrid_match_score", "scoring_method", "claim_label",
            "candidate_retrieval_sources", "candidate_semantic_cosine",
            "candidate_lexical_overlap", "formula_match_score", "recommendation_rank",
        ])
    result = scorer.score(candidate_features)

    ranked = candidate_features.copy()
    ranked.insert(0, "row_index", candidate_indices)
    id_column = corpus.columns["id"]
    ranked.insert(1, "job_id", (
        corpus.frame.iloc[candidate_indices][id_column].to_numpy()
        if id_column else candidate_indices
    ))
    ranked.insert(2, "job_title", [corpus.titles[i] for i in candidate_indices])
    ranked["hybrid_match_score"] = result.scores
    ranked["scoring_method"] = result.method
    ranked["claim_label"] = getattr(
        scorer, "claim_label", "experimental_similarity_not_validated"
    )
    ranked["scorer_fingerprint"] = getattr(scorer, "scorer_fingerprint", "")
    ranked["candidate_retrieval_sources"] = [
        ",".join(selection.sources[int(i)]) for i in candidate_indices
    ]
    ranked["candidate_semantic_cosine"] = selection.semantic_cosine[candidate_indices]
    ranked["candidate_lexical_overlap"] = selection.lexical_scores[candidate_indices]
    ranked["formula_match_score"] = mc.formula_score(candidate_features)

    ranked = ranked.sort_values(
        by=[
            "hybrid_match_score",
            "formula_match_score",
            "semantic_similarity",
            "title_similarity",
            "skill_match_signal",
            "row_index",
        ],
        ascending=[False, False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["recommendation_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def rank_resume(
    corpus: JobCorpus,
    *,
    resume_text: str,
    resume_embedding: np.ndarray,
    resume_skills: Iterable[str],
    scorer: mc.MatchScorer,
    max_candidates_per_signal: int = candidate_retrieval.DEFAULT_CANDIDATES_PER_SIGNAL,
) -> pd.DataFrame:
    """Canonical public ranking entry point for app and CLI callers."""
    return rank_jobs(
        corpus,
        resume_text=resume_text,
        resume_embedding=resume_embedding,
        resume_skills=resume_skills,
        scorer=scorer,
        max_candidates_per_signal=max_candidates_per_signal,
    )


def skill_breakdown(
    corpus: JobCorpus,
    row_index: int,
    resume_skills: Dict[str, str],
) -> tuple:
    """
    ``(matched_labels, gap_labels)`` for one job: skills the resume has
    that the job asks for, and skills the job asks for that it lacks.
    """

    job_skills = corpus.skill_sets[row_index]

    matched = sorted(
        (corpus.label(uri) for uri in job_skills if uri in resume_skills),
        key=str.lower,
    )

    gaps = sorted(
        (corpus.label(uri) for uri in job_skills if uri not in resume_skills),
        key=str.lower,
    )

    return matched, gaps
