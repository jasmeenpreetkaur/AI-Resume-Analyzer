from __future__ import annotations

import numpy as np
import pytest

import candidate_retrieval as cr


def test_candidate_retrieval_unions_all_signal_types_deterministically():
    selection = cr.select_candidate_indices(
        resume_text="python developer data science",
        job_texts=["remote backend role", "python sql engineer", "bookkeeping assistant"],
        resume_skills={"esco:python"},
        job_skill_sets=[set(), set(), {"esco:python"}],
        semantic_cosine=np.array([0.9, 0.1, 0.2]),
        lexical_scores=np.array([0.0, 0.8, 0.3]),
        shared_skill_counts=np.array([0, 0, 1]),
        skill_jaccard_scores=np.array([0.0, 0.0, 0.5]),
        max_candidates_per_signal=1,
    )

    assert selection.indices == [0, 1, 2]
    assert selection.sources[0] == ("semantic",)
    assert selection.sources[1] == ("lexical",)
    assert selection.sources[2] == ("skill",)
    assert selection.diagnostic["union_count"] == 3


def test_candidate_retrieval_breaks_ties_by_row_index():
    selection = cr.select_candidate_indices(
        resume_text="python analyst",
        job_texts=["same role", "same role", "same role"],
        semantic_cosine=np.array([0.5, 0.5, 0.5]),
        lexical_scores=np.zeros(3),
        shared_skill_counts=np.zeros(3),
        max_candidates_per_signal=2,
    )
    assert selection.indices == [0, 1]


def test_empty_resume_does_not_create_semantic_candidates():
    selection = cr.select_candidate_indices(
        resume_text="   ",
        job_texts=["python developer", "data analyst"],
        semantic_cosine=np.array([0.9, 0.8]),
        lexical_scores=np.array([0.0, 0.0]),
        shared_skill_counts=np.array([0, 0]),
    )
    assert selection.indices == []


def test_text_embedding_cache_reuses_duplicate_and_previous_vectors(tmp_path):
    calls = []

    def encode(texts):
        calls.append(list(texts))
        return np.array([[len(texts[i]), i + 1] for i in range(len(texts))], dtype=np.float32)

    first = cr.encode_texts_cached(
        ["alpha", "beta", "alpha"], encode=encode, model_name="test-encoder", cache_dir=tmp_path
    )
    second = cr.encode_texts_cached(
        ["alpha", "beta"], encode=encode, model_name="test-encoder", cache_dir=tmp_path
    )

    assert len(calls) == 1
    assert calls[0] == ["alpha", "beta"]
    assert first.shape == (3, 2)
    np.testing.assert_allclose(first[0], first[2])
    np.testing.assert_allclose(second, first[:2])


def test_text_embedding_cache_is_scoped_by_model_name(tmp_path):
    calls = []

    def encode(texts):
        calls.append(tuple(texts))
        return np.ones((len(texts), 3), dtype=np.float32)

    cr.encode_texts_cached(["alpha"], encode=encode, model_name="encoder-a", cache_dir=tmp_path)
    cr.encode_texts_cached(["alpha"], encode=encode, model_name="encoder-b", cache_dir=tmp_path)
    assert len(calls) == 2
