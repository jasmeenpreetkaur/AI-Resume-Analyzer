"""
Tests for the evidence-based model claim:

    src/11b_human_relevance_judgments.py   collects + evaluates judgments
    matching_core.resolve_claim            decides what the app may say

The rule under test: the score is only described as validated when a
human report with ENOUGH judgments supports it, for THIS model file.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matching_core as mc

SRC = Path(__file__).resolve().parents[1] / "src"


def load_step():

    spec = importlib.util.spec_from_file_location(
        "human_relevance_judgments", SRC / "11b_human_relevance_judgments.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


step = load_step()


# ----------------------------------------------------------------------
# synthetic Step 9 dataset
# ----------------------------------------------------------------------

def make_dataset(resumes=40, jobs=20, seed=0) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    rows = []

    for resume in range(resumes):
        for job in range(jobs):

            semantic = rng.random()
            coverage = rng.random()
            shared = int(rng.integers(0, 6))

            rows.append(
                {
                    "resume_id": resume,
                    "job_id": job,
                    "job_title": f"job {job}",
                    "resume_text_representation": f"resume {resume} text",
                    "job_text_representation": f"job {job} text",
                    "semantic_similarity": semantic,
                    "title_similarity": rng.random(),
                    "text_containment": rng.random(),
                    "shared_skill_count": shared,
                    "jaccard_similarity": rng.random(),
                    "resume_skill_coverage": rng.random(),
                    "job_skill_coverage": coverage,
                    "maximum_skill_overlap_ratio": rng.random(),
                    "skill_match_signal": min(shared / 5, 1.0),
                    "category_match_signal": int(rng.random() < 0.3),
                    "category_available": 1,
                }
            )

    return pd.DataFrame(rows)


def judgments_from(dataset, score, rng, noise=0.0, annotator="a", per_resume=12):
    """Judge ``per_resume`` random jobs per resume; relevance follows ``score``."""

    rows = []

    for resume_id, group in dataset.groupby("resume_id"):

        picked = group.sample(per_resume, random_state=int(resume_id))

        value = score.loc[picked.index].to_numpy(float)
        value = value + rng.normal(0, noise, len(value))

        edges = np.quantile(value, [0.4, 0.65, 0.85])

        relevance = np.digitize(value, edges)

        for (_, row), rel in zip(picked.iterrows(), relevance):
            rows.append(
                {
                    "resume_id": row["resume_id"],
                    "job_id": row["job_id"],
                    "relevance": int(rel),
                    "annotator": annotator,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def dataset():
    return make_dataset()


@pytest.fixture
def formula_scorer():
    return mc.MatchScorer.formula_only("test")


# ----------------------------------------------------------------------
# export
# ----------------------------------------------------------------------

def test_template_is_pooled_shuffled_and_score_free(dataset, formula_scorer):

    sheet = step.build_template(
        dataset, resumes=10, per_ranker_k=3, random_extra=2, scorer=formula_scorer
    )

    assert sheet["resume_id"].nunique() == 10

    # no ranker's output leaks into what the annotator sees
    leaked = {
        "semantic_similarity", "evidence_score", "match_score", "score",
        "category_match_signal", "formula_without_category",
    }

    assert leaked.isdisjoint(sheet.columns)
    assert {"relevance", "annotator", "notes"} <= set(sheet.columns)
    assert (sheet["relevance"].astype(str) == "nan").all() or sheet["relevance"].isna().all() \
        or (sheet["relevance"] == "").all()

    # pooled: 3 rankers' top-3 (3 distinct here: model absent) + random
    per_resume = sheet.groupby("resume_id").size()

    assert per_resume.max() <= 3 * 3 + 2
    assert per_resume.min() >= 3

    # pairs are unique, and not left in resume order
    assert not sheet.duplicated(["resume_id", "job_id"]).any()
    assert not sheet["resume_id"].is_monotonic_increasing


def test_template_is_reproducible(dataset, formula_scorer):

    first = step.build_template(dataset, resumes=8, scorer=formula_scorer, seed=7)
    second = step.build_template(dataset, resumes=8, scorer=formula_scorer, seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_template_needs_resumes_with_enough_candidates(formula_scorer):

    tiny = make_dataset(resumes=3, jobs=4)

    with pytest.raises(ValueError, match="candidate"):
        step.build_template(tiny, scorer=formula_scorer)


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------

def test_ndcg_is_one_for_a_perfect_ranking_and_lower_when_reversed():

    relevance = np.array([3, 2, 1, 0, 0], dtype=float)

    assert step.ndcg_at_k(relevance, np.array([5, 4, 3, 2, 1.0])) == pytest.approx(1.0)
    assert step.ndcg_at_k(relevance, np.array([1, 2, 3, 4, 5.0])) < 0.7

    # nothing relevant -> undefined, not 0 or 1
    assert step.ndcg_at_k(np.zeros(4), np.arange(4.0)) is None


def test_ties_are_broken_by_row_order_not_randomly():

    relevance = np.array([0, 3, 0, 0], dtype=float)

    scores = np.ones(4)

    assert step.ndcg_at_k(relevance, scores, k=1) == step.ndcg_at_k(
        relevance, scores, k=1
    )


def test_blank_and_invalid_ratings_are_ignored_and_counted():

    raw = pd.DataFrame(
        {
            "resume_id": [1, 1, 1, 1, 1],
            "job_id": [1, 2, 3, 4, 5],
            "relevance": [0, 3, "", "abc", 7],
            "annotator": ["a"] * 5,
        }
    )

    cleaned = step.clean_judgments(raw)

    assert len(cleaned["usable"]) == 2
    assert cleaned["blank_rows"] == 1
    assert cleaned["invalid_rows"] == 2


def test_kappa_is_none_for_a_single_annotator(dataset):

    rng = np.random.default_rng(1)

    judged = judgments_from(dataset, dataset["semantic_similarity"], rng)

    assert step.cohen_kappa(judged)["kappa"] is None


def test_kappa_is_high_for_agreeing_annotators_and_low_for_random(dataset):

    rng = np.random.default_rng(2)

    first = judgments_from(dataset, dataset["semantic_similarity"], rng, annotator="a")

    same = first.copy()
    same["annotator"] = "b"

    agree = step.cohen_kappa(pd.concat([first, same]))

    assert agree["kappa"] == pytest.approx(1.0)
    assert agree["overlap_pairs"] == len(first)

    shuffled = first.copy()
    shuffled["annotator"] = "b"
    shuffled["relevance"] = rng.permutation(shuffled["relevance"].to_numpy())

    assert step.cohen_kappa(pd.concat([first, shuffled]))["kappa"] < 0.2


# ----------------------------------------------------------------------
# evaluate -> claims
# ----------------------------------------------------------------------

def test_formula_claim_is_supported_only_when_humans_agree_with_it(
    dataset, formula_scorer
):

    rng = np.random.default_rng(3)

    formula = pd.Series(mc.formula_score(dataset[list(mc.MODEL_FEATURES)]), index=dataset.index)

    agreeing = judgments_from(dataset, formula, rng, noise=0.02)

    report = step.evaluate(dataset, agreeing, scorer=formula_scorer)

    assert report["status"] == "evaluated"
    assert report["claims"]["formula_ranks_by_human_relevance"] is True

    # judges who ignore the formula entirely
    unrelated = judgments_from(dataset, pd.Series(rng.random(len(dataset)), index=dataset.index), rng)

    report = step.evaluate(dataset, unrelated, scorer=formula_scorer)

    assert report["status"] == "evaluated"
    assert report["claims"]["formula_ranks_by_human_relevance"] is False


def test_too_few_judgments_never_support_a_claim(dataset, formula_scorer):

    rng = np.random.default_rng(4)

    formula = pd.Series(mc.formula_score(dataset[list(mc.MODEL_FEATURES)]), index=dataset.index)

    few = judgments_from(dataset[dataset["resume_id"] < 5], formula, rng, noise=0.0)

    report = step.evaluate(dataset, few, scorer=formula_scorer)

    assert report["status"] == "insufficient_data"
    assert not any(report["claims"].values())

    # the metrics are still reported so the first pilot is not wasted
    assert report["metrics"]["formula_without_category"]["resumes_evaluated"] > 0


def test_low_annotator_agreement_blocks_the_claim(dataset, formula_scorer):

    rng = np.random.default_rng(5)

    formula = pd.Series(mc.formula_score(dataset[list(mc.MODEL_FEATURES)]), index=dataset.index)

    first = judgments_from(dataset, formula, rng, annotator="a")

    second = first.copy()
    second["annotator"] = "b"
    second["relevance"] = rng.permutation(second["relevance"].to_numpy())

    report = step.evaluate(
        dataset, pd.concat([first, second]), scorer=formula_scorer
    )

    assert report["agreement_gate_passed"] is False
    assert report["claims"]["formula_ranks_by_human_relevance"] is False


def test_category_proxy_is_tested_against_the_judges(dataset, formula_scorer):

    rng = np.random.default_rng(6)

    # judges who think "same category" IS relevance
    judged = judgments_from(
        dataset,
        dataset["category_match_signal"] + rng.random(len(dataset)) * 0.01,
        rng,
    )

    report = step.evaluate(dataset, judged, scorer=formula_scorer)

    assert report["claims"]["category_proxy_tracks_human_relevance"] is True

    # judges who think it has nothing to do with relevance
    judged = judgments_from(dataset, pd.Series(rng.random(len(dataset)), index=dataset.index), rng)

    report = step.evaluate(dataset, judged, scorer=formula_scorer)

    assert report["claims"]["category_proxy_tracks_human_relevance"] is False
    assert report["category_proxy_vs_human_relevant"]["auc"] is not None


def test_model_claim_needs_the_model_to_match_the_judges_too(dataset, tmp_path):

    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(7)

    features = dataset[list(mc.MODEL_FEATURES)]

    # a model that is just the category proxy, as in the real pipeline
    proxy_model = HistGradientBoostingRegressor(max_iter=20, random_state=0).fit(
        features, dataset["category_match_signal"]
    )

    path = mc.save_model_bundle(
        tmp_path / "m.joblib", proxy_model, mc.MODEL_FEATURES, target="category proxy"
    )

    scorer = mc.MatchScorer.load(path)

    assert scorer.uses_model

    formula = pd.Series(mc.formula_score(features), index=dataset.index)

    # humans agree with the formula, not with a model that learned the proxy
    judged = judgments_from(dataset, formula, rng, noise=0.02)

    report = step.evaluate(dataset, judged, scorer=scorer)

    assert report["model_in_use"] is True
    assert report["model_trained_at"] == scorer.metadata["trained_at"]
    assert report["claims"]["formula_ranks_by_human_relevance"] is True
    assert report["claims"]["model_ranks_by_human_relevance"] is False

    # ... while a model trained on the humans' own judgments passes
    good_model = HistGradientBoostingRegressor(max_iter=60, random_state=0).fit(
        features, formula
    )

    good_path = mc.save_model_bundle(
        tmp_path / "good.joblib", good_model, mc.MODEL_FEATURES, target="human"
    )

    good = mc.MatchScorer.load(good_path)

    report = step.evaluate(dataset, judged, scorer=good)

    assert report["claims"]["model_ranks_by_human_relevance"] is True


def test_cli_export_then_evaluate_round_trip(dataset, tmp_path, monkeypatch):

    monkeypatch.setenv("MATCH_MODEL_PATH", str(tmp_path / "absent.joblib"))

    data_path = tmp_path / "dataset.csv"
    dataset.to_csv(data_path, index=False)

    template = tmp_path / "template.csv"

    assert step.main(
        ["export", "--input", str(data_path), "--output", str(template),
         "--resumes", "35", "--per-ranker-k", "5"]
    ) == 0

    sheet = pd.read_csv(template)

    formula = pd.Series(mc.formula_score(dataset[list(mc.MODEL_FEATURES)]), index=dataset.index)

    lookup = dataset.set_index(["resume_id", "job_id"]).index

    values = formula.to_numpy()[lookup.get_indexer(
        pd.MultiIndex.from_frame(sheet[["resume_id", "job_id"]])
    )]

    sheet["relevance"] = np.digitize(values, np.quantile(values, [0.4, 0.65, 0.85]))
    sheet["annotator"] = "me"

    filled = tmp_path / "filled.csv"
    sheet.to_csv(filled, index=False)

    report_path = tmp_path / "report.json"

    assert step.main(
        ["evaluate", "--input", str(data_path), "--judgments", str(filled),
         "--report", str(report_path)]
    ) == 0

    report = json.loads(report_path.read_text())

    assert report["judged_resumes"] == 35
    assert report["pairs_not_in_dataset"] == 0


# ----------------------------------------------------------------------
# resolve_claim: what the app / CLI / reports may say
# ----------------------------------------------------------------------

def evaluated(*, scorer_fingerprint="FP", suitability_wording_approved=True, model_trained_at="T1", **claims):

    return {
        "status": "evaluated",
        "judged_pairs": 500,
        "judged_resumes": 40,
        "model_trained_at": model_trained_at,
        "scorer_fingerprint": scorer_fingerprint,
        "suitability_wording_approved": suitability_wording_approved,
        "claims": claims,
    }


def test_model_without_a_human_report_is_labelled_a_proxy():

    claim = mc.resolve_claim(True, {"trained_at": "T1"}, report=None)

    assert claim.level == mc.CLAIM_MODEL_PROXY
    assert claim.validated is False
    assert "proxy" in claim.label.lower()
    assert "not validated" in claim.label.lower()
    assert "hired" in claim.statement.lower()


def test_model_is_validated_only_for_the_model_file_that_was_evaluated():

    report = evaluated(model_ranks_by_human_relevance=True)

    same = mc.resolve_claim(True, {"trained_at": "T1"}, report=report, scorer_fingerprint="FP")

    assert same.validated is True
    assert same.level == mc.CLAIM_MODEL_VALIDATED

    retrained = mc.resolve_claim(True, {"trained_at": "T2"}, report=report, scorer_fingerprint="FP")

    assert retrained.validated is False
    assert "different model file" in retrained.statement


def test_an_unsupported_model_claim_stays_unvalidated_even_with_a_report():

    report = evaluated(model_ranks_by_human_relevance=False)

    claim = mc.resolve_claim(True, {"trained_at": "T1"}, report=report, scorer_fingerprint="FP")

    assert claim.validated is False
    assert "does not show agreement" in claim.statement


def test_insufficient_data_never_validates():

    report = evaluated(model_ranks_by_human_relevance=True)
    report["status"] = "insufficient_data"

    assert mc.resolve_claim(True, {"trained_at": "T1"}, report=report, scorer_fingerprint="FP").validated is False

    report["claims"] = {"formula_ranks_by_human_relevance": True}

    assert mc.resolve_claim(False, {}, report=report, scorer_fingerprint="FP").validated is False


def test_formula_claim_follows_the_report():

    assert mc.resolve_claim(False, {}, report=None).level == mc.CLAIM_FORMULA

    supported = mc.resolve_claim(
        False, {}, report=evaluated(formula_ranks_by_human_relevance=True), scorer_fingerprint="FP"
    )

    assert supported.level == mc.CLAIM_FORMULA_VALIDATED
    assert supported.validated is True


def test_scorer_claim_reads_the_report_named_by_the_environment(
    tmp_path, monkeypatch
):

    report = tmp_path / "r.json"
    scorer = mc.MatchScorer.formula_only()
    report.write_text(json.dumps(evaluated(
        formula_ranks_by_human_relevance=True,
        scorer_fingerprint=scorer.scorer_fingerprint,
    )))

    monkeypatch.setenv("HUMAN_REPORT_PATH", str(report))

    assert scorer.claim.validated is True
    assert "NOT validated" not in scorer.description

    monkeypatch.setenv("HUMAN_REPORT_PATH", str(tmp_path / "missing.json"))

    assert scorer.claim.validated is False


def test_corrupt_report_is_ignored(tmp_path, monkeypatch):

    report = tmp_path / "r.json"
    report.write_text("{not json")

    monkeypatch.setenv("HUMAN_REPORT_PATH", str(report))

    assert mc.load_human_report() is None
    assert mc.MatchScorer.formula_only().claim.validated is False


def test_claim_requires_matching_fingerprint_and_explicit_approval():

    report = evaluated(formula_ranks_by_human_relevance=True)
    report["suitability_wording_approved"] = False
    unapproved = mc.resolve_claim(False, {}, report=report, scorer_fingerprint="FP")
    assert unapproved.validated is False

    report["suitability_wording_approved"] = True
    mismatched = mc.resolve_claim(False, {}, report=report, scorer_fingerprint="OTHER")
    assert mismatched.validated is False
