"""Tests for src/10b_train_match_model.py (the optional model trainer)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matching_core as mc

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(scope="module")
def trainer():
    """10b_... cannot be imported by name, so load it from its path."""

    spec = importlib.util.spec_from_file_location(
        "train_match_model", SRC / "10b_train_match_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


CATEGORIES = ["HR", "IT", "FINANCE", "SALES", "HEALTH", "ARTS"]


def synthetic_pairs(
    *,
    resumes: int = 120,
    jobs_per_resume: int = 40,
    informative: str = "text_containment",
    job_categories=None,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Step-9-shaped pairs where ONE feature (default text_containment, which
    the formula weighs at only 3%) carries the same-category signal and the
    formula's favourite feature (semantic similarity) is noise.
    """

    rng = np.random.default_rng(seed)

    rows = []

    for resume in range(resumes):

        resume_category = CATEGORIES[resume % len(CATEGORIES)]

        for job in range(jobs_per_resume):

            job_category = (job_categories or CATEGORIES)[
                rng.integers(len(job_categories or CATEGORIES))
            ]

            same = float(
                mc.normalize_category(resume_category)
                == mc.normalize_category(job_category)
            )

            row = {
                "resume_id": resume,
                "job_id": 10_000 + job,
                "resume_category": resume_category,
                "job_category": job_category,
            }

            for name in mc.MODEL_FEATURES:
                row[name] = float(rng.random())

            row[informative] = float(
                np.clip(0.25 + 0.5 * same + rng.normal(0, 0.1), 0, 1)
            )

            row["shared_skill_count"] = float(rng.integers(0, 12))

            rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def test_ranking_metrics_on_a_hand_checked_example(trainer):

    frame = pd.DataFrame(
        {
            "resume_id": ["a"] * 4 + ["b"] * 3 + ["c"] * 2,
            "job_id": [1, 2, 3, 4, 1, 2, 3, 1, 2],
            "target": [0, 1, 0, 0, 1, 0, 0, 1, 1],
        }
    )

    scores = [0.9, 0.8, 0.1, 0.0, 0.9, 0.5, 0.1, 0.5, 0.4]

    out = trainer.ranking_metrics(frame, scores, k=2)

    # resume c has no negatives -> skipped. a: positive at rank 2, b: rank 1.
    assert out["resumes_evaluated"] == 2
    assert out["mrr"] == pytest.approx((1 / 2 + 1) / 2)
    assert out["precision_at_2"] == pytest.approx((1 / 2 + 1 / 2) / 2)


def test_ranking_metrics_breaks_ties_by_job_id(trainer):

    frame = pd.DataFrame(
        {
            "resume_id": ["a", "a", "a"],
            "job_id": [3, 1, 2],
            "target": [1, 0, 0],
        }
    )

    out = trainer.ranking_metrics(frame, [0.5, 0.5, 0.5], k=1)

    # equal scores: job 1 first, job 3 (the positive) last -> rank 3
    assert out["mrr"] == pytest.approx(1 / 3)


def test_ranking_metrics_with_nothing_evaluable(trainer):

    frame = pd.DataFrame(
        {"resume_id": ["a", "a"], "job_id": [1, 2], "target": [0, 0]}
    )

    out = trainer.ranking_metrics(frame, [0.1, 0.2])

    assert out["resumes_evaluated"] == 0
    assert out["mrr"] is None


# ----------------------------------------------------------------------
# Label guard rails
# ----------------------------------------------------------------------

def test_missing_categories_are_dropped_not_labelled_negative(trainer):

    frame = synthetic_pairs(resumes=12, jobs_per_resume=5)
    frame.loc[:9, "job_category"] = ""

    labelled = trainer.make_target(frame)

    assert (~labelled["category_known"]).sum() == 10
    assert labelled.loc[:9, "target"].eq(0).all()  # but flagged unknown


def test_disjoint_category_vocabularies_stop_training(trainer):

    frame = synthetic_pairs(
        job_categories=["Accountancy", "Software", "Nursing", "Marketing"]
    )

    labelled = trainer.make_target(frame)

    with pytest.raises(trainer.LabelAlignmentError) as caught:
        trainer.check_label_alignment(labelled)

    assert "barely overlap" in str(caught.value)
    assert "accountancy" in str(caught.value)  # names are shown normalised


def test_alignment_ignores_case_and_separators(trainer):

    frame = synthetic_pairs(
        job_categories=[c.lower().replace("-", " ") for c in CATEGORIES]
    )

    summary = trainer.check_label_alignment(trainer.make_target(frame))

    assert summary["same_category_pairs"] > 0
    assert len(summary["shared_categories"]) == len(CATEGORIES)


# ----------------------------------------------------------------------
# Save / do-not-save decision
# ----------------------------------------------------------------------

def metrics(precision, mrr):
    return {"precision_at_10": precision, "mrr": mrr, "roc_auc": 0.7}


def test_decision_requires_a_clear_gain(trainer):

    assert trainer.decide_to_save(metrics(0.40, 0.60), metrics(0.30, 0.50))[0]

    # equal -> not worth replacing a transparent formula
    assert not trainer.decide_to_save(metrics(0.30, 0.50), metrics(0.30, 0.50))[0]

    # better on one metric, clearly worse on the other
    assert not trainer.decide_to_save(metrics(0.60, 0.40), metrics(0.30, 0.50))[0]

    assert not trainer.decide_to_save(
        metrics(None, None), metrics(0.30, 0.50)
    )[0]


# ----------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------

def test_training_saves_a_model_the_scorer_then_uses(trainer, tmp_path):

    data = tmp_path / "pairs.csv"
    synthetic_pairs().to_csv(data, index=False)

    model_path = tmp_path / "models" / "match_regressor.joblib"
    report_path = tmp_path / "report.json"

    code = trainer.main(
        [
            "--input", str(data),
            "--output", str(model_path),
            "--report", str(report_path),
        ]
    )

    assert code == 0
    assert model_path.is_file()

    report = json.loads(report_path.read_text())

    assert report["model_saved"] is True
    assert report["target_is_proxy"] is True
    assert report["category_used_as_feature"] is False
    assert (
        report["holdout_metrics"]["model"]["mrr"]
        > report["holdout_metrics"]["formula_without_category"]["mrr"]
    )

    # The shared scorer picks the artefact up and scores with it.
    scorer = mc.MatchScorer.load(model_path)

    assert scorer.uses_model
    assert scorer.feature_names == mc.MODEL_FEATURES
    assert "HistGradientBoostingRegressor" in scorer.description

    probe = pd.DataFrame(
        [
            {name: 0.5 for name in mc.MODEL_FEATURES},
            {**{name: 0.5 for name in mc.MODEL_FEATURES}, "text_containment": 0.9},
        ]
    )

    low, high = scorer.score(probe).scores

    # The model learned the informative, monotonic signal.
    assert high > low


def test_training_stops_cleanly_when_categories_do_not_line_up(trainer, tmp_path):

    data = tmp_path / "pairs.csv"

    synthetic_pairs(job_categories=["Accountancy", "Nursing"]).to_csv(
        data, index=False
    )

    model_path = tmp_path / "m.joblib"
    report_path = tmp_path / "report.json"

    code = trainer.main(
        [
            "--input", str(data),
            "--output", str(model_path),
            "--report", str(report_path),
            "--force",  # --force must NOT bypass the label check
        ]
    )

    assert code == 2
    assert not model_path.exists()

    report = json.loads(report_path.read_text())

    assert report["status"] == "stopped_label_not_usable"
    assert report["model_saved"] is False


def test_model_is_not_saved_when_it_cannot_beat_the_formula(trainer, tmp_path):

    # Pure noise: nothing to learn, so no clear gain over the formula.
    frame = synthetic_pairs(informative="semantic_similarity", seed=3)

    for name in mc.MODEL_FEATURES:
        frame[name] = np.random.default_rng(5).random(len(frame))

    data = tmp_path / "noise.csv"
    frame.to_csv(data, index=False)

    model_path = tmp_path / "m.joblib"

    code = trainer.main(
        [
            "--input", str(data),
            "--output", str(model_path),
            "--report", str(tmp_path / "r.json"),
        ]
    )

    assert code == 0
    assert not model_path.exists()


def test_missing_input_is_a_clear_error(trainer, tmp_path):

    assert trainer.main(["--input", str(tmp_path / "nope.csv")]) == 1
