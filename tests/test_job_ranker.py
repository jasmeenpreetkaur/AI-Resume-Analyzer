"""Tests for src/job_ranker.py - the shared online ranker."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
import pytest

import job_ranker as jr
import matching_core as mc
from esco_vocab import ESCOVocabulary

SKILL = "http://data.europa.eu/esco/skill/"

DIM = 64


def fake_encode(texts):
    """Deterministic bag-of-words hashing encoder (unit vectors)."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)

    for row, text in enumerate(texts):
        for token in re.findall(r"[a-z0-9+#]+", text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM
            out[row, bucket] += 1.0

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return out / norms


@pytest.fixture()
def vocabulary():
    frame = pd.DataFrame(
        [
            {"conceptType": "KnowledgeSkillCompetence", "conceptUri": SKILL + "py",
             "preferredLabel": "Python", "altLabels": "python programming"},
            {"conceptType": "KnowledgeSkillCompetence", "conceptUri": SKILL + "sql",
             "preferredLabel": "SQL", "altLabels": ""},
            {"conceptType": "KnowledgeSkillCompetence", "conceptUri": SKILL + "ml",
             "preferredLabel": "machine learning", "altLabels": ""},
            {"conceptType": "KnowledgeSkillCompetence", "conceptUri": SKILL + "acct",
             "preferredLabel": "bookkeeping", "altLabels": ""},
            {"conceptType": "KnowledgeSkillCompetence", "conceptUri": SKILL + "comm",
             "preferredLabel": "communication", "altLabels": ""},
        ]
    )

    vocab = ESCOVocabulary()
    vocab.build_from_dataframe(frame)
    vocab.source_signature = "test:1:1"

    return vocab


@pytest.fixture()
def jobs():
    return pd.DataFrame(
        {
            "job_id": [101, 102, 103, 104],
            "job_title": [
                "Data Scientist",
                "Accountant",
                "Python Developer",
                "Office Manager",
            ],
            "job_description": [
                "Build machine learning models using Python and SQL.",
                "Prepare ledgers, bookkeeping and tax filings.",
                "Write Python services and SQL queries. Strong communication.",
                "Coordinate the office and supplies.",
            ],
            "job_skill_set": [
                "['communication']",
                "[]",
                "['Python', 'not an esco skill']",
                None,
            ],
            "company": ["A", "B", "C", "D"],
        }
    )


RESUME = (
    "Data scientist with five years of Python, SQL and machine learning "
    "experience building models."
)


def build(jobs, vocabulary, tmp_path):
    return jr.build_job_corpus(
        jobs,
        encode=fake_encode,
        model_name="fake",
        vocabulary=vocabulary,
        cache_dir=tmp_path,
    )


# ----------------------------------------------------------------------
# SetIndex
# ----------------------------------------------------------------------

def test_set_index_counts_and_fractions():

    index = jr.SetIndex([{"a", "b", "c"}, {"b"}, set(), {"x", "a"}])

    assert list(index.sizes) == [3, 1, 0, 2]
    assert list(index.count_present({"a", "b"})) == [2, 1, 0, 1]
    assert list(index.fraction_present({"a", "b"})) == pytest.approx(
        [2 / 3, 1.0, 0.0, 0.5]
    )
    assert list(index.count_present({"zzz"})) == [0, 0, 0, 0]


def test_vectorised_signals_equal_the_scalar_definitions(
    jobs, vocabulary, tmp_path
):
    """Offline (Step 9, scalar) and online (vectorised) must agree."""

    corpus = build(jobs, vocabulary, tmp_path)

    tokens = mc.token_set(RESUME)

    titles = corpus.title_index.fraction_present(tokens)
    texts = corpus.text_index.fraction_present(tokens)

    for i in range(len(jobs)):
        assert titles[i] == pytest.approx(
            mc.title_similarity(corpus.titles[i], RESUME)
        )
        assert texts[i] == pytest.approx(
            mc.text_containment(RESUME, corpus.texts[i])
        )


# ----------------------------------------------------------------------
# Skills
# ----------------------------------------------------------------------

def test_parse_skill_list_handles_every_format():

    assert jr.parse_skill_list("['a', 'b']") == ["a", "b"]
    assert jr.parse_skill_list('["a", "b"]') == ["a", "b"]
    assert jr.parse_skill_list("a, b; c") == ["a", "b", "c"]
    assert jr.parse_skill_list(["x", " y "]) == ["x", "y"]
    assert jr.parse_skill_list(None) == []
    assert jr.parse_skill_list(float("nan")) == []
    assert jr.parse_skill_list("") == []


def test_job_skills_come_from_esco_only(jobs, vocabulary, tmp_path):

    corpus = build(jobs, vocabulary, tmp_path)

    # text matches + exact ESCO match of job_skill_set; the non-ESCO
    # entry "not an esco skill" is ignored, not forced.
    assert corpus.skill_sets[0] == {SKILL + "ml", SKILL + "py", SKILL + "sql", SKILL + "comm"}
    assert corpus.skill_sets[1] == {SKILL + "acct"}
    assert corpus.skill_sets[2] == {SKILL + "py", SKILL + "sql", SKILL + "comm"}
    assert corpus.skill_sets[3] == frozenset()


def test_resume_skill_uris(vocabulary):

    found = jr.resume_skill_uris(RESUME, vocabulary)

    assert found == {
        SKILL + "py": "Python",
        SKILL + "sql": "SQL",
        SKILL + "ml": "machine learning",
    }
    assert jr.resume_skill_uris(RESUME, None) == {}


def test_skill_breakdown_gives_labels(jobs, vocabulary, tmp_path):

    corpus = build(jobs, vocabulary, tmp_path)

    resume = jr.resume_skill_uris(RESUME, vocabulary)

    matched, gaps = jr.skill_breakdown(corpus, 2, resume)

    assert matched == ["Python", "SQL"]
    assert gaps == ["communication"]


# ----------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------

def rank(corpus, vocabulary, scorer=None):
    resume_skills = jr.resume_skill_uris(RESUME, vocabulary)

    return jr.rank_jobs(
        corpus,
        resume_text=RESUME,
        resume_embedding=fake_encode([RESUME])[0],
        resume_skills=resume_skills,
        scorer=scorer or mc.MatchScorer.formula_only(),
    )


def test_ranking_puts_the_relevant_jobs_first(jobs, vocabulary, tmp_path):

    ranked = rank(build(jobs, vocabulary, tmp_path), vocabulary)

    assert list(ranked["job_title"][:2]) == ["Data Scientist", "Python Developer"]
    assert ranked["job_title"].iloc[-1] in {"Accountant", "Office Manager"}
    assert list(ranked["recommendation_rank"]) == [1, 2, 3, 4]
    assert ranked["hybrid_match_score"].is_monotonic_decreasing
    assert set(ranked["scoring_method"]) == {mc.METHOD_FORMULA}
    assert ranked["hybrid_match_score"].between(0, 1).all()


def test_ranking_matches_matching_core_formula(jobs, vocabulary, tmp_path):
    """The ranker adds no scoring logic of its own."""

    ranked = rank(build(jobs, vocabulary, tmp_path), vocabulary)

    expected = mc.formula_score(ranked)

    assert ranked["hybrid_match_score"].to_numpy() == pytest.approx(expected)


def test_ranking_is_deterministic(jobs, vocabulary, tmp_path):

    corpus = build(jobs, vocabulary, tmp_path)

    first = rank(corpus, vocabulary)
    second = rank(corpus, vocabulary)

    pd.testing.assert_frame_equal(first, second)


def test_ranker_uses_the_trained_model_when_one_exists(
    jobs, vocabulary, tmp_path
):
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(1)

    X = pd.DataFrame(rng.random((200, 9)), columns=mc.MODEL_FEATURES)
    y = X["semantic_similarity"] * 0.5 + X["job_skill_coverage"] * 0.5

    model = HistGradientBoostingRegressor(max_iter=25, random_state=0).fit(X, y)

    path = mc.save_model_bundle(
        tmp_path / "m.joblib", model, mc.MODEL_FEATURES, target="unit-test"
    )

    scorer = mc.MatchScorer.load(path)

    ranked = rank(build(jobs, vocabulary, tmp_path / "cache"), vocabulary, scorer)

    assert set(ranked["scoring_method"]) == {mc.METHOD_MODEL}

    formula = mc.formula_score(ranked)

    assert not np.allclose(
        np.sort(formula)[::-1], ranked["hybrid_match_score"].to_numpy()
    )


class ConstantScorer:
    """A model that gives every job the same (saturated) score."""

    uses_model = True
    method = mc.METHOD_MODEL
    description = "constant test model"

    def score(self, features):
        return mc.ScoreResult(
            scores=np.ones(len(features)),
            method=mc.METHOD_MODEL,
            description=self.description,
        )


def test_saturated_model_scores_are_ordered_by_the_formula(
    jobs, vocabulary, tmp_path
):
    """
    A model trained on a 0/1 proxy can score many jobs exactly 1.0. Ties
    must fall back to the transparent formula, not to row order.
    """

    ranked = rank(
        build(jobs, vocabulary, tmp_path), vocabulary, ConstantScorer()
    )

    assert set(ranked["hybrid_match_score"]) == {1.0}
    assert ranked["formula_match_score"].is_monotonic_decreasing
    assert list(ranked["job_title"][:2]) == ["Data Scientist", "Python Developer"]


def test_without_a_vocabulary_ranking_degrades_to_text_signals(jobs, tmp_path):

    corpus = jr.build_job_corpus(
        jobs, encode=fake_encode, model_name="fake",
        vocabulary=None, cache_dir=tmp_path,
    )

    assert corpus.notes and "ESCO" in corpus.notes[0]

    ranked = jr.rank_jobs(
        corpus,
        resume_text=RESUME,
        resume_embedding=fake_encode([RESUME])[0],
        resume_skills=[],
        scorer=mc.MatchScorer.formula_only(),
    )

    assert (ranked["shared_skill_count"] == 0).all()
    assert ranked["job_title"].iloc[0] == "Data Scientist"


def test_embedding_dimension_mismatch_is_a_clear_error(jobs, vocabulary, tmp_path):

    corpus = build(jobs, vocabulary, tmp_path)

    with pytest.raises(ValueError, match="dimensions"):
        jr.rank_jobs(
            corpus,
            resume_text=RESUME,
            resume_embedding=np.ones(DIM + 1, dtype=np.float32),
            resume_skills=[],
            scorer=mc.MatchScorer.formula_only(),
        )


def test_missing_title_column_is_reported(tmp_path, vocabulary):

    with pytest.raises(ValueError, match="title"):
        jr.build_job_corpus(
            pd.DataFrame({"job_description": ["x"]}),
            encode=fake_encode, model_name="fake",
            vocabulary=vocabulary, cache_dir=tmp_path,
        )


# ----------------------------------------------------------------------
# Caches
# ----------------------------------------------------------------------

def test_embedding_cache_is_reused_then_invalidated_by_content(
    jobs, vocabulary, tmp_path
):
    calls = []

    def counting_encode(texts):
        calls.append(len(texts))
        return fake_encode(texts)

    def make(frame):
        return jr.build_job_corpus(
            frame, encode=counting_encode, model_name="fake",
            vocabulary=vocabulary, cache_dir=tmp_path,
        )

    make(jobs)
    make(jobs)

    assert calls == [4]  # second build reused the cache

    changed = jobs.copy()
    changed.loc[3, "job_description"] = "Entirely different text now."

    make(changed)

    # Same row count, different content: the old shape-only check would
    # have silently reused stale embeddings.
    assert calls == [4, 4]


def test_embedding_cache_is_invalidated_by_model_name(jobs, vocabulary, tmp_path):

    calls = []

    def counting_encode(texts):
        calls.append(1)
        return fake_encode(texts)

    for name in ("model-a", "model-b"):
        jr.build_job_corpus(
            jobs, encode=counting_encode, model_name=name,
            vocabulary=vocabulary, cache_dir=tmp_path,
        )

    assert len(calls) == 2


def test_skill_cache_is_invalidated_by_vocabulary_change(jobs, tmp_path):

    def vocab_with(labels):
        frame = pd.DataFrame(
            [
                {"conceptType": "KnowledgeSkillCompetence",
                 "conceptUri": SKILL + label, "preferredLabel": label,
                 "altLabels": ""}
                for label in labels
            ]
        )
        vocab = ESCOVocabulary()
        vocab.build_from_dataframe(frame)
        vocab.source_signature = "test:1:1"
        return vocab

    first = jr.build_job_corpus(
        jobs, encode=fake_encode, model_name="fake",
        vocabulary=vocab_with(["python"]), cache_dir=tmp_path,
    )

    second = jr.build_job_corpus(
        jobs, encode=fake_encode, model_name="fake",
        vocabulary=vocab_with(["python", "sql"]), cache_dir=tmp_path,
    )

    assert SKILL + "sql" not in first.skill_sets[2]
    assert SKILL + "sql" in second.skill_sets[2]
