"""Tests for src/matching_core.py - the single scoring definition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matching_core as mc
from matching_core import MatchScorer


def features(**overrides) -> pd.DataFrame:
    """One-row feature frame with every column the formula reads."""
    row = {
        "semantic_similarity": 0.6,
        "title_similarity": 0.5,
        "text_containment": 0.4,
        "shared_skill_count": 4,
        "jaccard_similarity": 0.3,
        "resume_skill_coverage": 0.5,
        "job_skill_coverage": 0.4,
        "maximum_skill_overlap_ratio": 0.3,
        "skill_match_signal": 0.41,
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ----------------------------------------------------------------------
# Text / lexical signals
# ----------------------------------------------------------------------

def test_tokenizer_keeps_tech_tokens_and_drops_sentence_punctuation():

    assert mc.tokenize_words("Node.js, scikit-learn, C++ & C#; developer.") == [
        "node.js", "scikit-learn", "c++", "c#", "developer",
    ]


def test_title_similarity_is_fraction_of_title_tokens_in_resume():

    resume = "Senior Python developer with machine learning experience."

    assert mc.title_similarity("Python Developer", resume) == 1.0
    assert mc.title_similarity("Python Data Engineer", resume) == pytest.approx(1 / 3)
    assert mc.title_similarity("", resume) == 0.0


def test_text_containment_is_fraction_of_job_tokens_in_resume():

    assert mc.text_containment("a b c d", "a b x y") == 0.5
    assert mc.text_containment("a b", "") == 0.0


def test_build_job_text_is_the_same_everywhere():

    assert mc.build_job_text("  Data  Analyst ", "Uses\nSQL.") == "Data Analyst. Uses SQL."
    assert mc.build_job_text("", "Only description") == "Only description"
    assert mc.build_job_text(None, float("nan")) == ""


def test_semantic_signal_maps_cosine_to_unit_interval():

    assert mc.semantic_signal(-1.0) == 0.0
    assert mc.semantic_signal(0.0) == 0.5
    assert mc.semantic_signal(1.0) == 1.0
    assert mc.semantic_signal(1.2) == 1.0


# ----------------------------------------------------------------------
# Skill signals
# ----------------------------------------------------------------------

def test_skill_overlap_matches_step7_definitions():

    out = mc.skill_overlap_arrays(shared=3, resume_count=6, job_count=4)

    assert out["jaccard_similarity"] == pytest.approx(3 / 7)
    assert out["resume_skill_coverage"] == pytest.approx(3 / 6)
    assert out["job_skill_coverage"] == pytest.approx(3 / 4)
    assert out["maximum_skill_overlap_ratio"] == pytest.approx(3 / 6)
    assert out["minimum_skill_coverage"] == pytest.approx(3 / 4)
    assert out["skill_count_difference"] == 2


def test_skill_overlap_handles_empty_sets_without_nan():

    out = mc.skill_overlap_arrays(
        shared=np.array([0, 0]),
        resume_count=np.array([0, 5]),
        job_count=np.array([0, 0]),
    )

    for name, values in out.items():
        assert np.isfinite(values).all(), name

    assert (out["jaccard_similarity"] == 0).all()


def test_shared_skill_signal_does_not_depend_on_the_dataset():

    # Step 10 divided by the dataset maximum, so the same pair scored
    # differently depending on what else was loaded.
    assert mc.shared_skill_count_signal(0) == 0.0
    assert mc.shared_skill_count_signal(mc.SHARED_SKILL_CAP) == pytest.approx(1.0)
    assert mc.shared_skill_count_signal(500) == pytest.approx(1.0)
    assert 0 < mc.shared_skill_count_signal(5) < mc.shared_skill_count_signal(10)


# ----------------------------------------------------------------------
# Category
# ----------------------------------------------------------------------

def test_category_equality_ignores_case_and_separators():

    assert mc.category_signal("INFORMATION-TECHNOLOGY", "Information Technology") == (1.0, True)
    assert mc.category_signal("HR", "Accounting") == (0.0, True)


def test_two_missing_categories_are_not_a_match():

    # Step 9 compared "" == "" and called it a match.
    assert mc.category_signal("", "") == (0.0, False)
    assert mc.category_signal(None, "IT") == (0.0, False)

    signal, available = mc.category_signal_series(
        pd.Series(["", "IT", None]), pd.Series(["", "it", "IT"])
    )

    assert list(signal) == [0.0, 1.0, 0.0]
    assert list(available) == [False, True, False]


# ----------------------------------------------------------------------
# Formula
# ----------------------------------------------------------------------

def test_weights_sum_to_one():

    assert sum(mc.WEIGHTS.values()) == pytest.approx(1.0)


def test_perfect_features_score_one_with_and_without_category():

    perfect = features(
        semantic_similarity=1, title_similarity=1, text_containment=1,
        shared_skill_count=mc.SHARED_SKILL_CAP, jaccard_similarity=1,
        resume_skill_coverage=1, job_skill_coverage=1,
        maximum_skill_overlap_ratio=1, skill_match_signal=1,
    )

    assert mc.formula_score(perfect)[0] == pytest.approx(1.0)

    with_cat = perfect.assign(category_match_signal=1, category_available=1)

    assert mc.formula_score(with_cat)[0] == pytest.approx(1.0)


def test_unknown_category_is_rescaled_not_penalised():
    """
    An uploaded resume has no category. Its perfect match must still be
    1.0, not capped at 0.90 by an always-zero category term.
    """

    perfect = features(
        semantic_similarity=1, title_similarity=1, text_containment=1,
        shared_skill_count=mc.SHARED_SKILL_CAP, jaccard_similarity=1,
        resume_skill_coverage=1, job_skill_coverage=1,
        maximum_skill_overlap_ratio=1, skill_match_signal=1,
    )

    assert mc.formula_score(perfect)[0] == pytest.approx(1.0)

    known_mismatch = perfect.assign(category_match_signal=0, category_available=1)

    assert mc.formula_score(known_mismatch)[0] == pytest.approx(0.9)


def test_formula_matches_step10_weights_when_category_is_known():

    frame = features(category_match_signal=1, category_available=1)

    expected = (
        0.30 * 0.6 + 0.20 * 0.41 + 0.15 * 0.5 + 0.10 * 1.0 + 0.10 * 0.4
        + 0.05 * 0.5 + 0.04 * 0.3 + 0.03 * 0.4 + 0.02 * 0.3
        + 0.01 * float(mc.shared_skill_count_signal(4))
    )

    assert mc.formula_score(frame)[0] == pytest.approx(expected)


def test_formula_is_monotonic_in_semantic_similarity():

    low = mc.formula_score(features(semantic_similarity=0.2))[0]
    high = mc.formula_score(features(semantic_similarity=0.9))[0]

    assert high > low


def test_formula_reports_missing_columns():

    with pytest.raises(KeyError):
        mc.formula_score(features().drop(columns=["title_similarity"]))


def test_build_feature_frame_is_consistent_with_scalar_definitions():

    frame = mc.build_feature_frame(
        semantic_cosine=np.array([0.2, 0.9]),
        title_sim=np.array([0.0, 1.0]),
        containment=np.array([0.1, 0.8]),
        shared=np.array([0, 5]),
        resume_count=np.array([8, 8]),
        job_count=np.array([6, 10]),
    )

    assert list(frame["semantic_similarity"]) == pytest.approx([0.6, 0.95])
    assert frame.loc[1, "jaccard_similarity"] == pytest.approx(5 / 13)

    expected = (
        0.4 * (5 / 13) + 0.3 * (5 / 8) + 0.3 * (5 / 10)
    )

    assert frame.loc[1, "skill_match_signal"] == pytest.approx(expected)

    for column in mc.MODEL_FEATURES:
        assert column in frame.columns


# ----------------------------------------------------------------------
# Scorer: model vs formula
# ----------------------------------------------------------------------

def _fit_model(feature_names=None):
    from sklearn.ensemble import HistGradientBoostingRegressor

    names = feature_names or mc.MODEL_FEATURES

    rng = np.random.default_rng(0)

    X = pd.DataFrame(rng.random((300, len(names))), columns=names)
    y = (X.iloc[:, 0] > 0.5).astype(float)

    model = HistGradientBoostingRegressor(max_iter=30, random_state=0)
    model.fit(X, y)

    return model


def test_scorer_without_model_uses_formula_and_says_why(tmp_path):

    scorer = MatchScorer.load(tmp_path / "missing.joblib")

    assert not scorer.uses_model
    assert scorer.method == mc.METHOD_FORMULA
    assert "no trained model" in scorer.fallback_reason

    result = scorer.score(features())

    assert result.method == mc.METHOD_FORMULA
    assert result.scores[0] == pytest.approx(mc.formula_score(features())[0])


def test_scorer_uses_trained_model_when_present(tmp_path):

    path = mc.save_model_bundle(
        tmp_path / "m.joblib",
        _fit_model(),
        mc.MODEL_FEATURES,
        target="unit-test",
    )

    scorer = MatchScorer.load(path)

    assert scorer.uses_model
    assert scorer.method == mc.METHOD_MODEL
    assert "HistGradientBoostingRegressor" in scorer.description

    high = scorer.score(features(semantic_similarity=0.95))
    low = scorer.score(features(semantic_similarity=0.05))

    assert high.method == mc.METHOD_MODEL
    assert 0.0 <= low.scores[0] <= high.scores[0] <= 1.0
    assert high.scores[0] != pytest.approx(mc.formula_score(features())[0])


def test_scorer_accepts_a_bare_estimator_fitted_on_a_dataframe(tmp_path):

    import joblib

    path = tmp_path / "bare.joblib"
    joblib.dump(_fit_model(), path)

    scorer = MatchScorer.load(path)

    assert scorer.uses_model
    assert scorer.feature_names == mc.MODEL_FEATURES


def test_scorer_rejects_a_model_needing_uncomputable_features(tmp_path):

    names = ["semantic_similarity", "evidence_score"]

    path = mc.save_model_bundle(
        tmp_path / "bad.joblib",
        _fit_model(names),
        names,
    )

    scorer = MatchScorer.load(path)

    assert not scorer.uses_model
    assert "evidence_score" in scorer.fallback_reason


def test_scorer_rejects_wrong_schema_and_corrupt_files(tmp_path):

    import joblib

    path = tmp_path / "old.joblib"
    joblib.dump(
        {"estimator": _fit_model(), "feature_names": mc.MODEL_FEATURES,
         "schema_version": 999},
        path,
    )

    assert not MatchScorer.load(path).uses_model

    junk = tmp_path / "junk.joblib"
    junk.write_bytes(b"not a model")

    scorer = MatchScorer.load(junk)

    assert not scorer.uses_model
    assert "could not load" in scorer.fallback_reason


def test_model_path_can_come_from_the_environment(tmp_path, monkeypatch):

    path = mc.save_model_bundle(
        tmp_path / "env.joblib", _fit_model(), mc.MODEL_FEATURES
    )

    monkeypatch.setenv("MATCH_MODEL_PATH", str(path))

    assert MatchScorer.load().uses_model
