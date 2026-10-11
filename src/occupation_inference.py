"""
======================================================================
AI RESUME ANALYZER - ESCO OCCUPATION INFERENCE
======================================================================

Purpose
-------
The offline job dataset covers only five job categories, so ranking it can
never suggest a chef, pilot or teacher. For LIVE job search the app needs
good search phrases for ANY field. This module infers which ESCO
occupations a resume resembles, so their names can be used as live-search
queries (together with the job title the user typed, if any).

How it works (no training, no labels - nothing is claimed to be "learned"):

  semantic   cosine( resume embedding , occupation embedding )
             pretrained all-MiniLM-L6-v2, the same model the project uses
             for resume<->job similarity. Occupation text = preferred label
             + a few alternative labels + the start of its ESCO description.

  skills     essential_matched / max(essential_total, MIN_ESSENTIAL)
             how many of the occupation's ESSENTIAL ESCO skills
             (occupationSkillRelations_en.csv, relationType == essential)
             the resume evidences. The floor stops occupations with only a
             couple of essential skills from scoring 100% by accident.

  score      0.6 * percentile(semantic) + 0.4 * percentile(skills)
             Percentiles make the two signals comparable without tuning
             scales. The 0.6/0.4 split is a DESIGN CHOICE, not a learned or
             validated value. It is only used to order suggestions.

The result is "roles your resume resembles", NOT a prediction of suitability
or hiring chances. Check the suggestions on your own resumes with:

    python src/occupation_inference.py --per-category 2

(prints only occupation names, never resume text).

Files read:   data/external/occupations_en.csv
              data/external/occupationSkillRelations_en.csv
Cache:        data/cache/esco_occupations/  (embeddings; rebuilt when the
              occupation text or the embedding model changes)
======================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "external"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "esco_occupations"

OCCUPATIONS_FILE = "occupations_en.csv"
RELATIONS_FILE = "occupationSkillRelations_en.csv"

CACHE_SCHEMA = 1

SEMANTIC_WEIGHT = 0.6
MIN_ESSENTIAL_FOR_COVERAGE = 8

ALT_LABELS_IN_TEXT = 3
DESCRIPTION_CHARS = 300


class OccupationDataError(RuntimeError):
    """The ESCO occupation files are missing or have an unexpected layout."""


# ======================================================================
# LOADING
# ======================================================================

def _column(df: pd.DataFrame, *names: str) -> Optional[str]:
    lookup = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}

    for name in names:
        key = re.sub(r"[^a-z0-9]", "", name.lower())

        if key in lookup:
            return lookup[key]

    return None


def _clean(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def occupation_text(label: str, alt_labels: str, description: str) -> str:
    """The text that represents one occupation for embedding."""
    alternatives = [
        _clean(part)
        for part in str(alt_labels or "").splitlines()
        if _clean(part)
    ][:ALT_LABELS_IN_TEXT]

    parts = [label]

    if alternatives:
        parts.append(", ".join(alternatives))

    parts.append(_clean(description)[:DESCRIPTION_CHARS])

    return ". ".join(part for part in parts if part)


@dataclass
class OccupationIndex:
    uris: List[str]
    labels: List[str]
    embeddings: np.ndarray
    essential: Dict[str, FrozenSet[str]]
    optional: Dict[str, FrozenSet[str]]
    fingerprint: str

    def __len__(self) -> int:
        return len(self.uris)


def _text_digest(texts: Sequence[str]) -> str:
    digest = hashlib.sha1()

    for text in texts:
        digest.update(text.encode("utf-8", "ignore"))
        digest.update(b"\x1f")

    return digest.hexdigest()


def build_occupation_index(
    encode: Callable[[List[str]], np.ndarray],
    *,
    model_name: str,
    data_dir: Path = DATA_DIR,
    cache_dir: Path = CACHE_DIR,
) -> OccupationIndex:
    """
    ESCO occupations with embeddings and essential/optional skill sets.
    Embeddings are cached on disk; they are recomputed only when the
    occupation text or the embedding model changes.
    """
    occupations_path = data_dir / OCCUPATIONS_FILE
    relations_path = data_dir / RELATIONS_FILE

    for path in (occupations_path, relations_path):
        if not path.is_file():
            raise OccupationDataError(f"ESCO file not found: {path}")

    occupations = pd.read_csv(occupations_path, low_memory=False)
    relations = pd.read_csv(relations_path, low_memory=False)

    uri_col = _column(occupations, "conceptUri")
    label_col = _column(occupations, "preferredLabel")
    alt_col = _column(occupations, "altLabels")
    description_col = _column(occupations, "description")

    if uri_col is None or label_col is None:
        raise OccupationDataError(
            f"{OCCUPATIONS_FILE} needs conceptUri and preferredLabel columns "
            f"(found: {list(occupations.columns)})"
        )

    occupations = occupations.dropna(subset=[uri_col, label_col])
    occupations = occupations[occupations[label_col].astype(str).str.strip() != ""]
    occupations = occupations.drop_duplicates(uri_col).reset_index(drop=True)

    uris = occupations[uri_col].astype(str).tolist()
    labels = [_clean(v) for v in occupations[label_col]]

    texts = [
        occupation_text(
            label,
            occupations[alt_col].iloc[i] if alt_col else "",
            occupations[description_col].iloc[i] if description_col else "",
        )
        for i, label in enumerate(labels)
    ]

    digest = _text_digest(texts)

    # ---- embeddings (cached) ---------------------------------------
    cache_dir.mkdir(parents=True, exist_ok=True)

    array_path = cache_dir / "occupation_embeddings.npy"
    meta_path = cache_dir / "occupation_embeddings.meta.json"

    expected = {
        "schema": CACHE_SCHEMA,
        "model": model_name,
        "text_sha1": digest,
        "rows": len(texts),
    }

    embeddings: Optional[np.ndarray] = None

    if array_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

            if all(meta.get(key) == value for key, value in expected.items()):
                cached = np.load(array_path)

                if cached.ndim == 2 and cached.shape[0] == len(texts):
                    embeddings = cached.astype(np.float32, copy=False)

        except Exception as exc:  # corrupt cache -> rebuild
            logger.warning("Ignoring unreadable occupation cache: %s", exc)

    if embeddings is None:
        embeddings = np.asarray(encode(texts), dtype=np.float32)

        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
            raise OccupationDataError(
                f"Embedding function returned shape {embeddings.shape} "
                f"for {len(texts)} occupations."
            )

        np.save(array_path, embeddings)
        meta_path.write_text(
            json.dumps({**expected, "dim": int(embeddings.shape[1])}),
            encoding="utf-8",
        )

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = (embeddings / norms).astype(np.float32)

    # ---- essential / optional skills per occupation ----------------
    occ_col = _column(relations, "occupationUri")
    skill_col = _column(relations, "skillUri")
    type_col = _column(relations, "relationType")

    if occ_col is None or skill_col is None or type_col is None:
        raise OccupationDataError(
            f"{RELATIONS_FILE} needs occupationUri, skillUri and relationType "
            f"columns (found: {list(relations.columns)})"
        )

    known = set(uris)
    relation_type = relations[type_col].astype(str).str.strip().str.lower()

    def grouped(kind: str) -> Dict[str, FrozenSet[str]]:
        subset = relations[(relation_type == kind) & relations[occ_col].isin(known)]
        return {
            str(uri): frozenset(group[skill_col].astype(str))
            for uri, group in subset.groupby(occ_col)
        }

    fingerprint = hashlib.sha1(
        f"{digest}|{model_name}|{len(relations)}".encode()
    ).hexdigest()

    return OccupationIndex(
        uris=uris,
        labels=labels,
        embeddings=embeddings,
        essential=grouped("essential"),
        optional=grouped("optional"),
        fingerprint=fingerprint,
    )


# ======================================================================
# INFERENCE
# ======================================================================

def infer_occupations(
    index: OccupationIndex,
    resume_embedding: np.ndarray,
    resume_skill_uris: Sequence[str],
    *,
    top_k: int = 5,
    semantic_weight: float = SEMANTIC_WEIGHT,
) -> pd.DataFrame:
    """
    The ``top_k`` ESCO occupations this resume most resembles.

    Columns: occupation_uri, occupation, semantic_similarity (raw cosine),
    essential_total, essential_matched, optional_matched, skill_signal,
    score, matched_essential_uris.
    """
    vector = np.asarray(resume_embedding, dtype=np.float32).reshape(-1)

    norm = float(np.linalg.norm(vector))

    if norm == 0 or vector.shape[0] != index.embeddings.shape[1]:
        raise ValueError("resume embedding is empty or has the wrong dimension")

    cosine = index.embeddings @ (vector / norm)

    skills = set(resume_skill_uris)

    essential_total = np.zeros(len(index), dtype=np.int32)
    essential_matched = np.zeros(len(index), dtype=np.int32)
    optional_matched = np.zeros(len(index), dtype=np.int32)
    matched_uris: List[List[str]] = [[] for _ in range(len(index))]

    for position, uri in enumerate(index.uris):

        essential = index.essential.get(uri, frozenset())
        optional = index.optional.get(uri, frozenset())

        essential_total[position] = len(essential)

        if skills:
            hits = sorted(essential & skills)
            essential_matched[position] = len(hits)
            matched_uris[position] = hits
            optional_matched[position] = len(optional & skills)

    skill_signal = essential_matched / np.maximum(
        essential_total, MIN_ESSENTIAL_FOR_COVERAGE
    )

    semantic_pct = pd.Series(cosine).rank(pct=True, method="average").to_numpy()
    skill_pct = pd.Series(skill_signal).rank(pct=True, method="average").to_numpy()

    score = semantic_weight * semantic_pct + (1.0 - semantic_weight) * skill_pct

    frame = pd.DataFrame(
        {
            "occupation_uri": index.uris,
            "occupation": index.labels,
            "semantic_similarity": cosine.astype(float),
            "essential_total": essential_total,
            "essential_matched": essential_matched,
            "optional_matched": optional_matched,
            "skill_signal": skill_signal.astype(float),
            "score": score,
            "matched_essential_uris": matched_uris,
        }
    )

    return (
        frame.sort_values(
            ["score", "semantic_similarity", "occupation"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .head(top_k)
        .reset_index(drop=True)
    )


def suggest_queries(inferred: pd.DataFrame, n: int = 3) -> List[str]:
    """Occupation names to use as live-search phrases (near-duplicates dropped)."""
    queries: List[str] = []
    seen: List[frozenset] = []

    for label in inferred["occupation"]:

        tokens = frozenset(re.findall(r"[a-z0-9+#]+", str(label).lower()))

        if not tokens or tokens in seen:
            continue

        seen.append(tokens)
        queries.append(str(label))

        if len(queries) >= n:
            break

    return queries


# ======================================================================
# COMMAND LINE: sanity check on your own resumes (prints no resume text)
# ======================================================================

def _guess_text_column(df: pd.DataFrame) -> str:
    for name in ("resume_text", "text", "clean_text", "resume_str", "cleaned_text", "processed_text"):
        column = _column(df, name)

        if column:
            return column

    object_columns = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "str"]

    return max(object_columns, key=lambda c: df[c].astype(str).str.len().median())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show the ESCO occupations inferred for sample resumes "
        "(occupation names only; no resume text is printed)."
    )
    parser.add_argument(
        "--resumes",
        default=str(PROJECT_ROOT / "data" / "processed" / "resumes_text_processed.csv"),
    )
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--category-column", default=None)
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import esco_vocab
    import job_ranker
    import matching_core as mc
    from sentence_transformers import SentenceTransformer

    resumes = pd.read_csv(args.resumes, low_memory=False).fillna("")

    text_column = args.text_column or _guess_text_column(resumes)
    category_column = args.category_column or _column(resumes, "category", "resume_category")

    model = SentenceTransformer(mc.SEMANTIC_MODEL_NAME)

    def encode(texts: List[str]) -> np.ndarray:
        return model.encode(
            texts, batch_size=64, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
        )

    index = build_occupation_index(encode, model_name=mc.SEMANTIC_MODEL_NAME)

    try:
        vocabulary = esco_vocab.load_esco_vocabulary()
    except (FileNotFoundError, ValueError):
        vocabulary = None

    if category_column:
        sample = (
            resumes.groupby(category_column, group_keys=False)
            .apply(lambda g: g.sample(min(len(g), args.per_category), random_state=args.seed))
            .reset_index(drop=True)
        )
    else:
        sample = resumes.sample(min(len(resumes), 20), random_state=args.seed)

    print(f"{len(sample)} resumes, text column '{text_column}'\n")

    for _, row in sample.iterrows():

        text = str(row[text_column])
        skills = job_ranker.resume_skill_uris(text, vocabulary)
        embedding = encode([text])[0]

        top = infer_occupations(index, embedding, list(skills), top_k=args.top)

        label = row[category_column] if category_column else "-"

        print(f"[{label}]")

        for _, found in top.iterrows():
            print(
                f"    {found['occupation']}  "
                f"(essential skills evidenced: {found['essential_matched']}/{found['essential_total']})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
