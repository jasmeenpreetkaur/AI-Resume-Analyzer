"""
======================================================================
AI RESUME ANALYZER - SHARED MATCHING CORE (single source of truth)
======================================================================

Before this module, four places each computed a resume<->job match with
their own formula and their own weights:

    Step 9   evidence_score        35% semantic, 25% skill, 15% job cov ...
    Step 10  hybrid_match_score    10 signals, weights 0.30/0.20/0.15 ...
    Step 13  calculate_hybrid_score 4 signals, weights 0.50/0.30/0.15/0.05
    Step 14  calculate_rankings    75% raw cosine + 25% job-skill coverage

So the same resume could be ranked differently by the offline pipeline,
the command-line tool and the Streamlit app. Now all of them import
THIS module, which owns:

    * the definition of every signal (semantic, title, containment,
      skill overlap, category)
    * the one transparent weighted formula (``WEIGHTS`` / ``formula_score``)
    * the optional trained model and the ONE place that decides whether
      the model or the formula produces a score (``MatchScorer``)

Scoring policy
--------------
``MatchScorer.load()`` looks for a trained HistGradientBoostingRegressor
(``models/match_regressor.joblib``, produced by
``src/10b_train_match_model.py``). If it is present and valid it scores
every pair. Otherwise the transparent weighted formula is used. Either
way every caller gets the same answer for the same features, and every
result says which scorer produced it.

Signal conventions (all in [0, 1]):

    semantic_similarity   (cosine + 1) / 2          NOT the raw cosine
    title_similarity      share of job-title tokens found in the resume
    text_containment      share of job-text tokens found in the resume
    skill signals         from ESCO skill-URI sets (shared / jaccard /
                          coverages)
    category_match_signal 1 only if both categories are known and equal
======================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "match_regressor.joblib"
# Output from the pooled human-relevance study (Step 11b). It is separate
# from the stricter top-10 review report and requires an explicit sign-off.
HUMAN_REPORT_PATH = PROJECT_ROOT / "reports" / "human_relevance_report.json"

SCHEMA_VERSION = 1

SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"


# ======================================================================
# TEXT UTILITIES
# ======================================================================

def clean_text(value: Any) -> str:
    """None / NaN -> "", NULs removed, whitespace collapsed."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).replace("\x00", " ")

    return re.sub(r"\s+", " ", text).strip()


def build_job_text(title: Any, description: Any) -> str:
    """
    The text that represents a job everywhere (embeddings, lexical
    signals, skill extraction): ``"<title>. <description>"``.
    """

    parts = [
        part
        for part in (clean_text(title), clean_text(description))
        if part
    ]

    return ". ".join(parts)


# Words, keeping node.js / scikit-learn / c++ / c# intact but dropping
# sentence punctuation ("developer." -> "developer").
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#]*(?:[.\-][a-z0-9+#]+)*")


def tokenize_words(text: Any) -> List[str]:
    return _TOKEN_RE.findall(clean_text(text).lower())


def token_set(text: Any) -> frozenset:
    return frozenset(tokenize_words(text))


# ======================================================================
# SIGNALS
# ======================================================================

def semantic_signal(cosine: Any) -> Any:
    """
    Map cosine similarity [-1, 1] to [0, 1].

    Step 9/10/13 always did this; the Streamlit app used the raw cosine,
    which made its scores a different scale from everything else.
    """

    values = (np.asarray(cosine, dtype=float) + 1.0) / 2.0

    return np.clip(values, 0.0, 1.0)


def lexical_fraction(reference: frozenset, other: frozenset) -> float:
    """|reference & other| / |reference|."""

    if not reference:
        return 0.0

    return len(reference & other) / len(reference)


def title_similarity(job_title: Any, resume_text: Any) -> float:
    """Share of the job title's tokens that appear in the resume."""

    return lexical_fraction(token_set(job_title), token_set(resume_text))


def text_containment(resume_text: Any, job_text: Any) -> float:
    """Share of the job text's tokens that appear in the resume."""

    return lexical_fraction(token_set(job_text), token_set(resume_text))


def skill_overlap_arrays(
    shared: Any,
    resume_count: Any,
    job_count: Any,
) -> Dict[str, np.ndarray]:
    """
    Skill-overlap features from three counts. Works on scalars or
    arrays; every caller (offline Step 7/8 definitions, Step 13, the
    app) gets identical arithmetic.
    """

    shared, resume_count, job_count = np.broadcast_arrays(
        np.asarray(shared, dtype=float),
        np.asarray(resume_count, dtype=float),
        np.asarray(job_count, dtype=float),
    )

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        out = np.zeros_like(numerator, dtype=float)
        np.divide(numerator, denominator, out=out, where=denominator > 0)
        return out

    union = resume_count + job_count - shared

    return {
        "shared_skill_count": shared,
        "resume_skill_count": resume_count,
        "job_skill_count": job_count,
        "jaccard_similarity": ratio(shared, union),
        "resume_skill_coverage": ratio(shared, resume_count),
        "job_skill_coverage": ratio(shared, job_count),
        "maximum_skill_overlap_ratio": ratio(
            shared, np.maximum(resume_count, job_count)
        ),
        "minimum_skill_coverage": ratio(
            shared, np.minimum(resume_count, job_count)
        ),
        "skill_count_difference": np.abs(resume_count - job_count),
    }


def skill_match_signal(jaccard: Any, resume_coverage: Any, job_coverage: Any) -> Any:
    """Step 9's structured skill signal."""

    return (
        0.40 * np.clip(np.asarray(jaccard, dtype=float), 0.0, 1.0)
        + 0.30 * np.clip(np.asarray(resume_coverage, dtype=float), 0.0, 1.0)
        + 0.30 * np.clip(np.asarray(job_coverage, dtype=float), 0.0, 1.0)
    )


# Shared-skill count saturates here. Step 10 used log1p(x) / log1p(max in
# the dataset), which depends on whatever data happens to be loaded and
# therefore cannot be reproduced for a single uploaded resume.
SHARED_SKILL_CAP = 20


def shared_skill_count_signal(shared: Any) -> Any:

    capped = np.clip(np.asarray(shared, dtype=float), 0.0, SHARED_SKILL_CAP)

    return np.log1p(capped) / np.log1p(SHARED_SKILL_CAP)


def normalize_category(value: Any) -> str:
    """"INFORMATION-TECHNOLOGY" and "Information Technology" compare equal."""

    text = clean_text(value).lower()

    text = re.sub(r"[-_/&]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def category_signal(resume_category: Any, job_category: Any) -> tuple:
    """
    (signal, available). When either category is missing the signal is
    0 AND ``available`` is False (so the formula ignores the term
    instead of treating "", "" as a match).
    """

    resume = normalize_category(resume_category)
    job = normalize_category(job_category)

    available = bool(resume) and bool(job)

    return (1.0 if available and resume == job else 0.0), available


def category_signal_series(
    resume_categories: pd.Series,
    job_categories: pd.Series,
) -> tuple:
    """Vectorised ``category_signal``: (signal Series, available Series)."""

    resume = resume_categories.map(normalize_category)
    job = job_categories.map(normalize_category)

    available = resume.ne("") & job.ne("")

    signal = (available & resume.eq(job)).astype(float)

    return signal, available


# ======================================================================
# FEATURE FRAME
# ======================================================================

# Columns a trained model may use. Every one can be computed for a
# single uploaded resume. Category features are deliberately NOT here: an
# uploaded resume has no category, so a model that used it could not be
# applied online. (They are also the proxy label the model is trained
# against - see 10b_train_match_model.py.)
MODEL_FEATURES: List[str] = [
    "semantic_similarity",
    "title_similarity",
    "text_containment",
    "shared_skill_count",
    "jaccard_similarity",
    "resume_skill_coverage",
    "job_skill_coverage",
    "maximum_skill_overlap_ratio",
    "skill_match_signal",
]

# More evidence of a match never lowers the score.
MODEL_MONOTONIC_CST: List[int] = [1] * len(MODEL_FEATURES)

COMPUTABLE_FEATURES = set(MODEL_FEATURES) | {
    "resume_skill_count",
    "job_skill_count",
    "minimum_skill_coverage",
    "skill_count_difference",
}


def build_feature_frame(
    *,
    semantic_cosine: Any,
    title_sim: Any,
    containment: Any,
    shared: Any,
    resume_count: Any,
    job_count: Any,
    index: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """
    Canonical feature frame for N resume-job pairs (no category columns).

    This is the single place the online callers (Step 13, Streamlit app)
    assemble features, using the same definitions the offline pipeline
    uses.
    """

    skills = skill_overlap_arrays(shared, resume_count, job_count)

    frame = pd.DataFrame(
        {
            "semantic_similarity": semantic_signal(semantic_cosine),
            "title_similarity": np.asarray(title_sim, dtype=float),
            "text_containment": np.asarray(containment, dtype=float),
            **skills,
        },
        index=index,
    )

    frame["skill_match_signal"] = skill_match_signal(
        frame["jaccard_similarity"],
        frame["resume_skill_coverage"],
        frame["job_skill_coverage"],
    )

    return frame


# ======================================================================
# THE TRANSPARENT FORMULA
# ======================================================================

# Taken verbatim from Step 10 (the most complete of the four formulas
# and the one Step 11/12 validate). These are design parameters, not
# learned weights.
WEIGHTS: Dict[str, float] = {
    "semantic_similarity": 0.30,
    "skill_match_signal": 0.20,
    "title_similarity": 0.15,
    "category_match_signal": 0.10,
    "job_skill_coverage": 0.10,
    "resume_skill_coverage": 0.05,
    "jaccard_similarity": 0.04,
    "text_containment": 0.03,
    "maximum_skill_overlap_ratio": 0.02,
    "shared_skill_count_signal": 0.01,
}

if not np.isclose(sum(WEIGHTS.values()), 1.0):  # pragma: no cover
    raise ValueError(
        f"Hybrid weights must sum to 1.0, got {sum(WEIGHTS.values())}"
    )


def _numeric(features: pd.DataFrame, column: str) -> np.ndarray:

    if column not in features.columns:
        raise KeyError(
            f"Missing feature column '{column}' required by the match score."
        )

    return (
        pd.to_numeric(features[column], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def _category_available(features: pd.DataFrame) -> np.ndarray:
    """
    Whether the category term applies to each row.

    * ``category_available`` column present -> use it (Step 8 writes it).
    * only ``category_match_signal`` present -> assume it is meaningful.
    * neither (uploaded resume) -> unavailable.
    """

    if "category_available" in features.columns:
        return (
            pd.to_numeric(features["category_available"], errors="coerce")
            .fillna(0)
            .astype(bool)
            .to_numpy()
        )

    if "category_match_signal" in features.columns:
        return np.ones(len(features), dtype=bool)

    return np.zeros(len(features), dtype=bool)


def signal_frame(features: pd.DataFrame) -> pd.DataFrame:
    """The ten normalised [0, 1] signals the formula weighs."""

    clip = lambda values: np.clip(values, 0.0, 1.0)  # noqa: E731

    available = _category_available(features)

    category = (
        clip(_numeric(features, "category_match_signal"))
        if "category_match_signal" in features.columns
        else np.zeros(len(features))
    ) * available

    return pd.DataFrame(
        {
            "semantic_similarity": clip(_numeric(features, "semantic_similarity")),
            "skill_match_signal": clip(_numeric(features, "skill_match_signal")),
            "title_similarity": clip(_numeric(features, "title_similarity")),
            "category_match_signal": category,
            "job_skill_coverage": clip(_numeric(features, "job_skill_coverage")),
            "resume_skill_coverage": clip(_numeric(features, "resume_skill_coverage")),
            "jaccard_similarity": clip(_numeric(features, "jaccard_similarity")),
            "text_containment": clip(_numeric(features, "text_containment")),
            "maximum_skill_overlap_ratio": clip(
                _numeric(features, "maximum_skill_overlap_ratio")
            ),
            "shared_skill_count_signal": clip(
                shared_skill_count_signal(_numeric(features, "shared_skill_count"))
            ),
        },
        index=features.index,
    )


def formula_score(features: pd.DataFrame) -> np.ndarray:
    """
    The transparent weighted score in [0, 1].

    When a row's category is unknown (always the case for an uploaded
    resume) the category term is dropped and the other weights are
    rescaled to sum to 1, so offline and online scores stay on the same
    scale instead of online scores being silently capped at 0.90.
    """

    signals = signal_frame(features)

    category_weight = WEIGHTS["category_match_signal"]

    without_category = sum(
        weight * signals[name].to_numpy()
        for name, weight in WEIGHTS.items()
        if name != "category_match_signal"
    )

    with_category = (
        without_category
        + category_weight * signals["category_match_signal"].to_numpy()
    )

    available = _category_available(features)

    scores = np.where(
        available,
        with_category,
        without_category / (1.0 - category_weight),
    )

    return np.clip(scores, 0.0, 1.0)


# ======================================================================
# SCORER (model or formula)
# ======================================================================

METHOD_MODEL = "hist_gradient_boosting_model"
METHOD_FORMULA = "weighted_formula"


@dataclass
class ScoreResult:
    scores: np.ndarray
    method: str
    description: str
    model_path: Optional[str] = None


# ======================================================================
# HUMAN-RELEVANCE CLAIMS (pooled review workflow)
# ======================================================================

CLAIM_MODEL_PROXY = "model_category_proxy_unvalidated"
CLAIM_MODEL_VALIDATED = "model_human_validated"
CLAIM_FORMULA = "formula_unvalidated"
CLAIM_FORMULA_VALIDATED = "formula_human_validated"


@dataclass
class ModelClaim:
    level: str
    validated: bool
    label: str
    statement: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "validated_against_human_judgments": self.validated,
            "label": self.label,
            "statement": self.statement,
            "evidence": self.evidence,
        }


def load_human_report(path: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """Load the pooled relevance report, returning None if missing/invalid."""
    report_path = Path(path or os.getenv("HUMAN_REPORT_PATH") or HUMAN_REPORT_PATH)
    if not report_path.is_file():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Ignoring unreadable human-relevance report %s: %s", report_path, exc)
        return None


def resolve_claim(
    uses_model: bool,
    metadata: Optional[Dict[str, Any]] = None,
    report: Optional[Dict[str, Any]] = None,
    scorer_fingerprint: Optional[str] = None,
) -> ModelClaim:
    """
    Resolve what an evaluated pooled study supports. Evidence alone is not
    sufficient: the report must match the active scorer fingerprint and the
    operator must explicitly approve wording after reviewing the metrics.
    A trained model report must also refer to the same trained model artifact.
    """
    metadata = metadata or {}
    if report is None:
        report = load_human_report()

    status = (report or {}).get("status")
    claims = (report or {}).get("claims", {}) if status == "evaluated" else {}
    report_fp = (report or {}).get("scorer_fingerprint")
    fingerprint_ok = bool(scorer_fingerprint) and report_fp == scorer_fingerprint
    approved = (report or {}).get("suitability_wording_approved") is True
    model_matches = (
        not uses_model
        or (report is not None
            and report.get("model_trained_at") is not None
            and report.get("model_trained_at") == metadata.get("trained_at"))
    )
    evidence = {
        "status": status,
        "judged_pairs": (report or {}).get("judged_pairs"),
        "judged_resumes": (report or {}).get("judged_resumes"),
        "scorer_fingerprint_matches": fingerprint_ok,
        "explicit_operator_approval": approved,
        "model_artifact_matches": model_matches,
    }
    gates = fingerprint_ok and approved and model_matches and status == "evaluated"

    if uses_model:
        supported = bool(gates and claims.get("model_ranks_by_human_relevance"))
        if supported:
            return ModelClaim(
                CLAIM_MODEL_VALIDATED, True,
                "Trained model (checked against human judgments)",
                "The ranking passed the pooled human-relevance evaluation for this exact scorer version. It remains a relevance ranking, not a hiring probability.",
                evidence,
            )
        reason = (
            "the report is for a different scorer version" if report and not fingerprint_ok
            else "the report was generated for a different model file" if report and not model_matches
            else "explicit operator approval is missing" if report and not approved
            else "the study does not show agreement with human judgments" if report and status == "evaluated"
            else "the human-relevance study has not been run or lacks sufficient data"
        )
        return ModelClaim(
            CLAIM_MODEL_PROXY, False,
            "Experimental trained model (category proxy, not validated)",
            "This model was trained on a same-category proxy rather than hiring outcomes. " + reason + ". Treat the score as a similarity ranking, not a measure of suitability, whether you will be hired, or hiring probability.",
            evidence,
        )

    supported = bool(gates and claims.get("formula_ranks_by_human_relevance"))
    if supported:
        return ModelClaim(
            CLAIM_FORMULA_VALIDATED, True,
            "Transparent weighted formula (checked against human judgments)",
            "The weighted ranking passed the pooled human-relevance evaluation for this exact scorer version. It remains a relevance ranking, not a hiring probability.",
            evidence,
        )
    return ModelClaim(
        CLAIM_FORMULA, False,
        "Transparent weighted formula (not validated)",
        "This score uses a hand-weighted combination of text, skills and title signals. Human relevance is not validated for the active scorer; treat it as similarity, not as a measure of suitability or hiring probability.",
        evidence,
    )


@dataclass
class MatchScorer:
    """
    Scores resume-job pairs with the trained model when one is available,
    otherwise with ``formula_score``. Build it with ``MatchScorer.load()``.
    """

    estimator: Any = None
    feature_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    model_path: Optional[str] = None
    fallback_reason: str = ""
    _scorer_fingerprint_cache: Optional[str] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------

    @property
    def uses_model(self) -> bool:
        return self.estimator is not None

    @property
    def method(self) -> str:
        return METHOD_MODEL if self.uses_model else METHOD_FORMULA

    @property
    def claim(self) -> ModelClaim:
        """Claim supported by the pooled human-relevance study, if approved."""
        return resolve_claim(
            self.uses_model,
            self.metadata,
            scorer_fingerprint=self.scorer_fingerprint,
        )

    @property
    def scorer_fingerprint(self) -> str:
        """Fingerprint scoring code, retrieval policy and active model artifact."""
        if self._scorer_fingerprint_cache is not None:
            return self._scorer_fingerprint_cache
        digest = hashlib.sha256()
        for path in (
            Path(__file__),
            PROJECT_ROOT / "src" / "candidate_retrieval.py",
            PROJECT_ROOT / "src" / "job_ranker.py",
        ):
            if path.is_file():
                digest.update(path.name.encode("utf-8"))
                digest.update(path.read_bytes())
        digest.update(json.dumps(WEIGHTS, sort_keys=True).encode("utf-8"))
        digest.update(self.method.encode("utf-8"))
        model_path = Path(self.model_path) if self.uses_model and self.model_path else None
        if model_path and model_path.is_file():
            digest.update(hashlib.sha256(model_path.read_bytes()).digest())
        else:
            digest.update(b"weighted-formula")
        self._scorer_fingerprint_cache = digest.hexdigest()
        return self._scorer_fingerprint_cache

    @property
    def claim_label(self) -> str:
        """Use suitability wording only when a matching human-review gate passed."""
        report_path = Path(
            os.getenv("MATCH_VALIDATION_REPORT_PATH")
            or (PROJECT_ROOT / "reports" / "human_judgment_evaluation.json")
        )
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if (
                    report.get("suitability_wording_allowed") is True
                    and report.get("scorer_fingerprint") == self.scorer_fingerprint
                    and report.get("human_validation_status") == "approved"
                ):
                    return "human_validated_relevance"
            except (OSError, ValueError, TypeError):
                pass
        # The pooled study is a second valid path only after its own metrics,
        # exact fingerprint, matching model artifact and explicit sign-off pass.
        if self.claim.validated:
            return "human_validated_relevance"
        return "experimental_similarity_not_validated"

    @property
    def suitability_wording_allowed(self) -> bool:
        return self.claim_label == "human_validated_relevance"

    @property
    def claim_description(self) -> str:
        if self.suitability_wording_allowed:
            return "Human-judgment validation gate passed for this scorer version."
        return (
            "Experimental similarity only; relevance has not been validated "
            "against sufficient independent human judgments."
        )

    @property
    def description(self) -> str:

        if self.uses_model:

            target = self.metadata.get("target", "unknown target")

            validation_note = (
                "human-relevance validated" if self.claim.validated
                else "NOT validated against human judgments"
            )
            return (
                f"Trained {type(self.estimator).__name__} "
                f"({Path(self.model_path or '').name}; target: {target}; "
                f"{validation_note})"
            )

        reason = f" - {self.fallback_reason}" if self.fallback_reason else ""

        return f"Transparent weighted formula{reason}"

    # ------------------------------------------------------------------

    @classmethod
    def formula_only(cls, reason: str = "") -> "MatchScorer":
        return cls(fallback_reason=reason)

    @classmethod
    def load(cls, path: Optional[Any] = None) -> "MatchScorer":
        """
        Never raises. If there is no usable model the returned scorer
        uses the formula and ``fallback_reason`` says why.

        The model file is a pickle: only load model files you created.
        """

        model_path = Path(
            path
            or os.getenv("MATCH_MODEL_PATH")
            or DEFAULT_MODEL_PATH
        )

        if not model_path.is_file():
            return cls.formula_only(f"no trained model at {model_path}")

        try:
            import joblib

            loaded = joblib.load(model_path)
        except Exception as exc:
            logger.warning("Could not load model %s: %s", model_path, exc)
            return cls.formula_only(
                f"could not load {model_path.name}: {exc}"
            )

        if isinstance(loaded, dict) and "estimator" in loaded:

            estimator = loaded["estimator"]
            feature_names = list(loaded.get("feature_names") or [])
            metadata = {
                key: value
                for key, value in loaded.items()
                if key not in ("estimator",)
            }

            if loaded.get("schema_version") != SCHEMA_VERSION:
                return cls.formula_only(
                    f"{model_path.name} has schema_version "
                    f"{loaded.get('schema_version')!r}, expected "
                    f"{SCHEMA_VERSION}"
                )

        else:  # a bare scikit-learn estimator fitted on a DataFrame

            estimator = loaded
            feature_names = list(getattr(loaded, "feature_names_in_", []))
            metadata = {"target": "unknown (bare estimator)"}

        if not hasattr(estimator, "predict"):
            return cls.formula_only(f"{model_path.name} is not a model")

        if not feature_names:
            return cls.formula_only(
                f"{model_path.name} does not record its feature names"
            )

        unknown = [
            name for name in feature_names if name not in COMPUTABLE_FEATURES
        ]

        if unknown:
            return cls.formula_only(
                f"{model_path.name} needs features that cannot be computed "
                f"for an uploaded resume: {unknown}"
            )

        try:
            probe = pd.DataFrame([{name: 0.0 for name in feature_names}])
            value = float(np.asarray(estimator.predict(probe))[0])

            if not np.isfinite(value):
                raise ValueError("model returned a non-finite value")

        except Exception as exc:
            logger.warning("Model %s failed its self-check: %s", model_path, exc)
            return cls.formula_only(
                f"{model_path.name} failed its self-check: {exc}"
            )

        return cls(
            estimator=estimator,
            feature_names=feature_names,
            metadata=metadata,
            model_path=str(model_path),
        )

    # ------------------------------------------------------------------

    def score(self, features: pd.DataFrame) -> ScoreResult:
        """Scores in [0, 1], one per row of ``features``."""

        if not self.uses_model:
            return ScoreResult(
                scores=formula_score(features),
                method=METHOD_FORMULA,
                description=self.description,
            )

        missing = [
            name for name in self.feature_names if name not in features.columns
        ]

        if missing:
            raise KeyError(
                f"Features required by the trained model are missing: {missing}"
            )

        matrix = features[self.feature_names].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0.0)

        predictions = np.asarray(self.estimator.predict(matrix), dtype=float)

        predictions = np.clip(
            np.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )

        return ScoreResult(
            scores=predictions,
            method=METHOD_MODEL,
            description=self.description,
            model_path=self.model_path,
        )


def save_model_bundle(
    path: Any,
    estimator: Any,
    feature_names: Iterable[str],
    **metadata: Any,
) -> Path:
    """Write the artifact format ``MatchScorer.load`` expects."""

    import joblib
    import sklearn

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "estimator": estimator,
        "feature_names": list(feature_names),
        "schema_version": SCHEMA_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        **metadata,
    }

    joblib.dump(bundle, path)

    return path
