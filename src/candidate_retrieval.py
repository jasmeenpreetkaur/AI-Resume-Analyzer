"""Shared, deterministic resume-to-job candidate retrieval.

Offline Step 7 and the online ranker use the same union policy:
semantic top-K + lexical-overlap top-K + ESCO-skill-overlap top-K.
The returned set is a candidate-generation result, not a suitability verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

import matching_core as mc

DEFAULT_CANDIDATES_PER_SIGNAL = 100


@dataclass(frozen=True)
class CandidateSelection:
    indices: list[int]
    sources: dict[int, tuple[str, ...]]
    semantic_cosine: np.ndarray
    lexical_scores: np.ndarray
    shared_skill_counts: np.ndarray
    diagnostic: dict[str, int]


def _unit_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def select_candidate_indices(
    *,
    resume_text: str,
    job_texts: Sequence[str],
    resume_skills: Iterable[str] = (),
    job_skill_sets: Sequence[Iterable[str]] | None = None,
    resume_embedding: np.ndarray | None = None,
    job_embeddings: np.ndarray | None = None,
    semantic_cosine: np.ndarray | None = None,
    lexical_scores: np.ndarray | None = None,
    shared_skill_counts: np.ndarray | None = None,
    skill_jaccard_scores: np.ndarray | None = None,
    max_candidates_per_signal: int = DEFAULT_CANDIDATES_PER_SIGNAL,
) -> CandidateSelection:
    """Return the deterministic union of semantic, lexical and skill candidates.

    Inputs may include precomputed signals. This lets the online ranker reuse
    its vectorized feature calculations while Step 7 invokes the same policy
    over the offline data. Ties are always broken by original row index.
    """
    texts = [mc.clean_text(text) for text in job_texts]
    n_jobs = len(texts)
    limit = max(0, int(max_candidates_per_signal))
    if job_skill_sets is None:
        skill_sets = [frozenset() for _ in range(n_jobs)]
    else:
        if len(job_skill_sets) != n_jobs:
            raise ValueError("job_skill_sets length must equal job_texts length")
        skill_sets = [frozenset(str(skill) for skill in skills) for skills in job_skill_sets]

    if semantic_cosine is None:
        if resume_embedding is not None and job_embeddings is not None:
            matrix = np.asarray(job_embeddings, dtype=np.float32)
            vector = _unit_vector(np.asarray(resume_embedding))
            if matrix.ndim != 2 or matrix.shape[0] != n_jobs:
                raise ValueError("job_embeddings must have one row per job_text")
            if matrix.shape[1] != vector.shape[0]:
                raise ValueError("Resume and job embedding dimensions do not match")
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            semantic = (matrix / norms) @ vector
        else:
            semantic = np.zeros(n_jobs, dtype=np.float32)
    else:
        semantic = np.asarray(semantic_cosine, dtype=np.float32).reshape(-1)
        if len(semantic) != n_jobs:
            raise ValueError("semantic_cosine length must equal job_texts length")

    if lexical_scores is None:
        resume_tokens = mc.token_set(resume_text)
        lexical = np.zeros(n_jobs, dtype=np.float32)
        if resume_tokens:
            for index, text in enumerate(texts):
                job_tokens = mc.token_set(text)
                if job_tokens:
                    lexical[index] = len(resume_tokens & job_tokens) / len(job_tokens)
    else:
        lexical = np.asarray(lexical_scores, dtype=np.float32).reshape(-1)
        if len(lexical) != n_jobs:
            raise ValueError("lexical_scores length must equal job_texts length")

    resume_skill_set = set(resume_skills)
    if shared_skill_counts is None:
        shared = np.asarray(
            [len(resume_skill_set & skills) for skills in skill_sets],
            dtype=np.int64,
        )
    else:
        shared = np.asarray(shared_skill_counts, dtype=np.int64).reshape(-1)
        if len(shared) != n_jobs:
            raise ValueError("shared_skill_counts length must equal job_texts length")

    if skill_jaccard_scores is None:
        jaccard = np.zeros(n_jobs, dtype=np.float32)
        for index, skills in enumerate(skill_sets):
            union_size = len(resume_skill_set | skills)
            if union_size:
                jaccard[index] = shared[index] / union_size
    else:
        jaccard = np.asarray(skill_jaccard_scores, dtype=np.float32).reshape(-1)
        if len(jaccard) != n_jobs:
            raise ValueError("skill_jaccard_scores length must equal job_texts length")

    source_sets: dict[int, set[str]] = {}

    def add(indices: Iterable[int], source: str) -> None:
        for raw_index in indices:
            index = int(raw_index)
            source_sets.setdefault(index, set()).add(source)

    # Do not fabricate candidates for a completely empty resume.
    has_resume_text = bool(mc.clean_text(resume_text))
    has_semantic_signal = bool(np.any(np.abs(np.nan_to_num(semantic, nan=0.0)) > 1e-12))
    if (
        n_jobs and limit and has_resume_text and has_semantic_signal
        and (resume_embedding is not None or semantic_cosine is not None or job_embeddings is not None)
    ):
        order = np.lexsort((np.arange(n_jobs), -np.nan_to_num(semantic, nan=-1.0)))
        add(order[: min(limit, n_jobs)], "semantic")

    if n_jobs and limit and np.any(lexical > 0):
        hits = np.flatnonzero(lexical > 0)
        order = hits[np.lexsort((hits, -np.nan_to_num(lexical[hits], nan=0.0)))]
        add(order[:limit], "lexical")

    if n_jobs and limit and np.any(shared > 0):
        hits = np.flatnonzero(shared > 0)
        # Prefer count, then Jaccard, then stable source row order.
        order = hits[np.lexsort((hits, -np.nan_to_num(jaccard[hits], nan=0.0), -shared[hits]))]
        add(order[:limit], "skill")

    indices = sorted(source_sets)
    sources = {
        index: tuple(sorted(source_sets[index]))
        for index in indices
    }
    return CandidateSelection(
        indices=indices,
        sources=sources,
        semantic_cosine=np.nan_to_num(semantic, nan=0.0, posinf=0.0, neginf=0.0),
        lexical_scores=np.nan_to_num(lexical, nan=0.0, posinf=0.0, neginf=0.0),
        shared_skill_counts=shared,
        diagnostic={
            "job_count": n_jobs,
            "semantic_count": sum("semantic" in value for value in source_sets.values()),
            "lexical_count": sum("lexical" in value for value in source_sets.values()),
            "skill_count": sum("skill" in value for value in source_sets.values()),
            "union_count": len(indices),
        },
    )


def encode_texts_cached(
    texts: Sequence[str],
    *,
    encode,
    model_name: str,
    cache_dir,
    batch_size: int = 64,
    progress=None,
) -> np.ndarray:
    """Encode unique texts with per-text content-addressed caching.

    Per-text caching makes Step 9 reuse Step 7's vectors even when Step 9's
    candidate pairs refer to only a subset of resumes/jobs. Cache keys include
    model name and exact cleaned text, so text/model changes cannot reuse stale
    embeddings. All returned vectors are L2-normalized.
    """
    import hashlib
    from pathlib import Path

    root = Path(cache_dir) / "text_embeddings"
    root.mkdir(parents=True, exist_ok=True)
    cleaned = [mc.clean_text(text) for text in texts]
    if not cleaned:
        return np.zeros((0, 0), dtype=np.float32)

    keys = [
        hashlib.sha256((str(model_name) + "\0" + text).encode("utf-8", "ignore")).hexdigest()
        for text in cleaned
    ]
    unique_missing: dict[str, str] = {}
    vectors: dict[str, np.ndarray] = {}

    for key, text in zip(keys, cleaned):
        path = root / f"{key}.npy"
        if key in vectors:
            continue
        if path.is_file():
            try:
                vector = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)
                if vector.size and np.isfinite(vector).all():
                    vectors[key] = vector
                    continue
            except Exception:
                pass
        unique_missing[key] = text

    missing_items = list(unique_missing.items())
    for start in range(0, len(missing_items), max(1, batch_size)):
        batch = missing_items[start:start + max(1, batch_size)]
        encoded = np.asarray(encode([text for _, text in batch]), dtype=np.float32)
        if encoded.ndim != 2 or encoded.shape[0] != len(batch):
            raise ValueError(
                f"Encoder returned shape {encoded.shape} for a batch of {len(batch)} texts"
            )
        norms = np.linalg.norm(encoded, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        encoded = encoded / norms
        for (key, _), vector in zip(batch, encoded):
            vector = np.asarray(vector, dtype=np.float32)
            path = root / f"{key}.npy"
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, vector, allow_pickle=False)
            temporary.replace(path)
            vectors[key] = vector
        if progress:
            progress(min(1.0, (start + len(batch)) / max(1, len(missing_items))))

    dimensions = {vectors[key].shape[0] for key in keys}
    if len(dimensions) != 1:
        raise ValueError(f"Embedding dimension mismatch in cache: {sorted(dimensions)}")
    return np.vstack([vectors[key] for key in keys]).astype(np.float32, copy=False)
