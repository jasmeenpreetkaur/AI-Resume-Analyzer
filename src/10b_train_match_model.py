"""
======================================================================
AI RESUME ANALYZER - STEP 10b TRAIN THE MATCH MODEL (OPTIONAL)
======================================================================

PURPOSE
-------
Train the HistGradientBoostingRegressor that the Streamlit app, Step 10
and Step 13 use through ``matching_core.MatchScorer`` when the file
``models/match_regressor.joblib`` exists. Without that file every caller
falls back to the transparent weighted formula - so this step is
optional, and the project works either way.

READ THIS FIRST - WHAT THE MODEL IS TRAINED AGAINST
---------------------------------------------------
There are NO human hiring labels in this project. The only label-like
information is each resume's occupation category and each job's
category. This script therefore trains against a PROXY:

    target = 1 if the resume's category equals the job's category
             0 if both categories are known and differ

and reports it as a proxy everywhere (report, model metadata, UI text).
A model trained on a proxy learns "which pairs look like same-category
pairs", which is a reasonable relevance signal but NOT a measure of hiring
success.

Guard rails that keep this honest:

* Category columns are NOT model inputs. Uploaded resumes have no
  category, so a model that used it could not run in the app, and it
  would just memorise its own target.
* Inputs are the nine features in ``matching_core.MODEL_FEATURES``,
  every one of which can be computed for a single uploaded resume with
  the same code that built this dataset (train/serve consistency).
* All nine are constrained to be monotonic: more evidence of a match can
  never lower the score.
* Train/test split is grouped by resume (no resume appears on both sides).
* The model is compared on the held-out resumes against the formula
  (evaluated WITHOUT the category term - exactly as it runs online) and
  against semantic-similarity-only. It is saved only if it is at least
  as good as the formula on precision@10 and MRR and better by a clear
  margin overall (use --force to save anyway).
* If the category names of resumes and jobs do not line up (so the
  proxy label would be meaningless) training stops and says why.

INPUT
-----
data/processed/resume_job_matching_dataset.csv   (Step 9)

OUTPUT
------
models/match_regressor.joblib            (only when the model qualifies)
reports/match_model_report.json          (always)

USAGE
-----
python src/10b_train_match_model.py [--force] [--input PATH] [--seed N]
======================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matching_core as mc


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "processed" / "resume_job_matching_dataset.csv"
)

DEFAULT_MODEL_OUTPUT = mc.DEFAULT_MODEL_PATH

REPORT_PATH = PROJECT_ROOT / "reports" / "match_model_report.json"

TARGET_DESCRIPTION = (
    "proxy: resume category == job category "
    "(no human hiring labels exist)"
)

ID_COLUMNS = ["resume_id", "job_id"]

CATEGORY_COLUMNS = ["resume_category", "job_category"]

RANK_K = 10

# Conditions the proxy label must satisfy before training is allowed.
MIN_ROWS = 1_000
MIN_POSITIVES = 200
MIN_POSITIVE_RATE = 0.005
MAX_POSITIVE_RATE = 0.60
MIN_SHARED_CATEGORIES = 2
MIN_EVALUABLE_RESUMES = 20

# A model replaces the formula only if it is not worse than the formula
# on either ranking metric and better by this much on the two combined.
MAX_ALLOWED_LOSS = 0.002
MIN_COMBINED_GAIN = 0.01


class LabelAlignmentError(RuntimeError):
    """The proxy label is unusable (categories do not line up)."""


# =====================================================================
# DATA
# =====================================================================

def load_training_frame(path: Path) -> pd.DataFrame:
    """Only the columns training needs - the Step 9 CSV also holds text."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Step 9 dataset not found: {path}\n"
            "Run steps 2-9 first."
        )

    needed = ID_COLUMNS + CATEGORY_COLUMNS + list(mc.MODEL_FEATURES)

    header = pd.read_csv(path, nrows=0).columns

    missing = [column for column in needed if column not in header]

    if missing:
        raise ValueError(
            f"{path.name} is missing columns needed for training: "
            f"{missing}. Re-run Step 9."
        )

    frame = pd.read_csv(path, usecols=needed)

    for column in mc.MODEL_FEATURES:
        frame[column] = (
            pd.to_numeric(frame[column], errors="coerce")
            .fillna(0.0)
            .astype("float32")
        )

    return frame


def make_target(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``category_known`` and the proxy ``target`` (0/1).

    Pairs where either category is missing have no label and are
    dropped by the caller - they are NOT treated as negatives.
    """

    signal, known = mc.category_signal_series(
        frame["resume_category"],
        frame["job_category"],
    )

    out = frame.copy()

    out["category_known"] = known.to_numpy()
    out["target"] = signal.to_numpy().astype(int)

    return out


def category_alignment_summary(frame: pd.DataFrame) -> Dict[str, Any]:
    """How well resume category names line up with job category names."""

    resume_categories = set(
        frame["resume_category"].map(mc.normalize_category)
    ) - {""}

    job_categories = set(
        frame["job_category"].map(mc.normalize_category)
    ) - {""}

    return {
        "resume_categories": len(resume_categories),
        "job_categories": len(job_categories),
        "shared_categories": sorted(resume_categories & job_categories),
        "resume_only_categories": sorted(resume_categories - job_categories)[:30],
        "job_only_categories": sorted(job_categories - resume_categories)[:30],
    }


def check_label_alignment(labelled: pd.DataFrame) -> Dict[str, Any]:
    """
    Raise LabelAlignmentError unless the proxy label is usable.
    Returns a summary dict for the report.
    """

    rows = int(len(labelled))
    positives = int(labelled["target"].sum())
    rate = positives / rows if rows else 0.0

    alignment = category_alignment_summary(labelled)

    problems: List[str] = []

    if rows < MIN_ROWS:
        problems.append(
            f"only {rows:,} pairs have a category on both sides "
            f"(need at least {MIN_ROWS:,})"
        )

    if positives < MIN_POSITIVES:
        problems.append(
            f"only {positives:,} same-category pairs "
            f"(need at least {MIN_POSITIVES:,})"
        )

    if rows and rate < MIN_POSITIVE_RATE:
        problems.append(
            f"same-category rate is {rate:.2%}, below {MIN_POSITIVE_RATE:.1%}"
        )

    if rows and rate > MAX_POSITIVE_RATE:
        problems.append(
            f"same-category rate is {rate:.2%}, above {MAX_POSITIVE_RATE:.0%}"
        )

    if len(alignment["shared_categories"]) < MIN_SHARED_CATEGORIES:
        problems.append(
            "resume and job category names barely overlap "
            f"({len(alignment['shared_categories'])} shared name(s)); "
            "they probably use different vocabularies"
        )

    summary = {
        "pairs_with_both_categories": rows,
        "same_category_pairs": positives,
        "same_category_rate": rate,
        **alignment,
    }

    if problems:

        raise LabelAlignmentError(
            "The proxy label (same category) is not usable:\n  - "
            + "\n  - ".join(problems)
            + "\n\nResume categories that never appear among job "
            f"categories: {alignment['resume_only_categories']}\n"
            f"Job categories that never appear among resume categories: "
            f"{alignment['job_only_categories']}\n"
            "Map both to one vocabulary before training (Step 2 / Step 7) "
            "or keep using the transparent formula."
        )

    return summary


# =====================================================================
# EVALUATION
# =====================================================================

def ranking_metrics(
    frame: pd.DataFrame,
    scores: Any,
    *,
    k: int = RANK_K,
    group: str = "resume_id",
    label: str = "target",
    tiebreak: str = "job_id",
) -> Dict[str, Any]:
    """
    Per-resume ranking quality, averaged over resumes that have at least
    one positive and one negative candidate.

    precision@k = positives among the top min(k, n) / min(k, n)
    MRR         = mean of 1 / rank of the first positive
    """

    work = pd.DataFrame(
        {
            "g": frame[group].to_numpy(),
            "s": np.asarray(scores, dtype=float),
            "y": frame[label].to_numpy().astype(int),
            "t": frame[tiebreak].astype(str).to_numpy(),
        }
    )

    work = work.sort_values(
        ["g", "s", "t"],
        ascending=[True, False, True],
        kind="mergesort",
    )

    grouped = work.groupby("g", sort=False)

    work["rank"] = grouped.cumcount() + 1
    work["n"] = grouped["y"].transform("size")
    work["pos"] = grouped["y"].transform("sum")

    work = work[(work["pos"] > 0) & (work["pos"] < work["n"])]

    resumes = int(work["g"].nunique())

    auc: Optional[float] = None

    try:
        from sklearn.metrics import roc_auc_score

        if work["y"].nunique() == 2:
            auc = float(roc_auc_score(work["y"], work["s"]))
    except Exception:  # pragma: no cover - sklearn always present here
        auc = None

    if resumes == 0:
        return {
            "resumes_evaluated": 0,
            f"precision_at_{k}": None,
            "mrr": None,
            "roc_auc": auc,
        }

    top = work[work["rank"] <= k]

    hits = top.groupby("g")["y"].sum()
    sizes = work.groupby("g")["n"].first().clip(upper=k)

    precision = float((hits.reindex(sizes.index).fillna(0) / sizes).mean())

    first_positive = work[work["y"] == 1].groupby("g")["rank"].min()

    mrr = float((1.0 / first_positive).mean())

    return {
        "resumes_evaluated": resumes,
        f"precision_at_{k}": precision,
        "mrr": mrr,
        "roc_auc": auc,
    }


def new_regressor(seed: int):
    """Monotonic HistGradientBoostingRegressor with conservative settings."""

    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=1.0,
        monotonic_cst=list(mc.MODEL_MONOTONIC_CST),
        early_stopping=False,
        random_state=seed,
    )


def decide_to_save(
    model_metrics: Dict[str, Any],
    formula_metrics: Dict[str, Any],
    *,
    k: int = RANK_K,
) -> Tuple[bool, str]:
    """Whether the model is good enough to replace the formula."""

    key = f"precision_at_{k}"

    if model_metrics[key] is None or formula_metrics[key] is None:
        return False, "too few evaluable held-out resumes to compare"

    p_gain = model_metrics[key] - formula_metrics[key]
    m_gain = model_metrics["mrr"] - formula_metrics["mrr"]

    if p_gain < -MAX_ALLOWED_LOSS or m_gain < -MAX_ALLOWED_LOSS:
        return False, (
            f"model is worse than the formula (precision@{k} "
            f"{p_gain:+.4f}, MRR {m_gain:+.4f})"
        )

    if p_gain + m_gain < MIN_COMBINED_GAIN:
        return False, (
            f"model is not clearly better than the formula (precision@{k} "
            f"{p_gain:+.4f}, MRR {m_gain:+.4f}; need a combined gain of "
            f"{MIN_COMBINED_GAIN})"
        )

    return True, (
        f"model beats the formula (precision@{k} {p_gain:+.4f}, "
        f"MRR {m_gain:+.4f})"
    )


def online_formula_scores(frame: pd.DataFrame) -> np.ndarray:
    """
    The formula exactly as it runs for an uploaded resume: no category
    columns, so the category weight is redistributed.
    """

    return mc.formula_score(frame[list(mc.MODEL_FEATURES)])


def train_and_evaluate(
    labelled: pd.DataFrame,
    *,
    seed: int = 42,
    test_size: float = 0.25,
) -> Dict[str, Any]:
    """
    Grouped hold-out evaluation, then a refit on all labelled pairs.

    Returns ``{"estimator", "holdout_estimator", "metrics", ...}``.
    """

    from sklearn.model_selection import GroupShuffleSplit

    features = list(mc.MODEL_FEATURES)

    resumes = labelled["resume_id"].nunique()

    if resumes < MIN_EVALUABLE_RESUMES:
        raise LabelAlignmentError(
            f"only {resumes} distinct resumes; need at least "
            f"{MIN_EVALUABLE_RESUMES} for a grouped hold-out evaluation."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )

    train_index, test_index = next(
        splitter.split(labelled, groups=labelled["resume_id"])
    )

    train = labelled.iloc[train_index]
    test = labelled.iloc[test_index]

    holdout = new_regressor(seed)
    holdout.fit(train[features], train["target"])

    model_scores = np.clip(holdout.predict(test[features]), 0.0, 1.0)

    candidates = {
        "model": model_scores,
        "formula_without_category": online_formula_scores(test),
        "semantic_similarity_only": test["semantic_similarity"].to_numpy(),
        "skill_match_only": test["skill_match_signal"].to_numpy(),
    }

    metrics = {
        name: ranking_metrics(test, scores)
        for name, scores in candidates.items()
    }

    final = new_regressor(seed)
    final.fit(labelled[features], labelled["target"])

    return {
        "estimator": final,
        "holdout_estimator": holdout,
        "metrics": metrics,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_resumes": int(train["resume_id"].nunique()),
        "test_resumes": int(test["resume_id"].nunique()),
        "test_frame": test,
    }


def permutation_importance_summary(
    estimator: Any,
    test: pd.DataFrame,
    seed: int,
    sample: int = 20_000,
) -> Optional[Dict[str, float]]:

    try:
        from sklearn.inspection import permutation_importance

        features = list(mc.MODEL_FEATURES)

        subset = test.sample(
            n=min(sample, len(test)),
            random_state=seed,
        )

        result = permutation_importance(
            estimator,
            subset[features],
            subset["target"],
            n_repeats=3,
            random_state=seed,
            scoring="neg_mean_squared_error",
        )

        return {
            name: float(value)
            for name, value in sorted(
                zip(features, result.importances_mean),
                key=lambda item: -item[1],
            )
        }

    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"Permutation importance skipped: {exc}")
        return None


# =====================================================================
# MAIN
# =====================================================================

def write_report(report: Dict[str, Any], path: Path = REPORT_PATH) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"Report saved: {path}")


def main(argv: Optional[List[str]] = None) -> int:

    parser = argparse.ArgumentParser(
        description="Train the optional match model (Step 10b)."
    )

    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--force",
        action="store_true",
        help="save the model even if it does not beat the formula "
        "(never bypasses the label-alignment check)",
    )

    args = parser.parse_args(argv)

    print("=" * 70)
    print("AI RESUME ANALYZER - STEP 10b TRAIN MATCH MODEL")
    print("=" * 70)
    print(f"Target: {TARGET_DESCRIPTION}")
    print(f"Features ({len(mc.MODEL_FEATURES)}): {', '.join(mc.MODEL_FEATURES)}")

    started = time.time()

    report: Dict[str, Any] = {
        "step": "10b",
        "name": "Train match model",
        "status": "started",
        "input": str(args.input),
        "target": TARGET_DESCRIPTION,
        "target_is_proxy": True,
        "hiring_labels_used": False,
        "features": list(mc.MODEL_FEATURES),
        "category_used_as_feature": False,
        "monotonic_constraints": list(mc.MODEL_MONOTONIC_CST),
        "seed": args.seed,
    }

    try:

        frame = load_training_frame(args.input)

        print(f"Loaded {len(frame):,} pairs, "
              f"{frame['resume_id'].nunique():,} resumes, "
              f"{frame['job_id'].nunique():,} jobs")

        labelled = make_target(frame)

        unlabeled = int((~labelled["category_known"]).sum())

        labelled = labelled[labelled["category_known"]].reset_index(drop=True)

        report["pairs_without_category_dropped"] = unlabeled

        report["label"] = check_label_alignment(labelled)

        print(
            f"Same-category pairs: {report['label']['same_category_pairs']:,} "
            f"of {report['label']['pairs_with_both_categories']:,} "
            f"({report['label']['same_category_rate']:.2%})"
        )

        outcome = train_and_evaluate(
            labelled,
            seed=args.seed,
            test_size=args.test_size,
        )

    except LabelAlignmentError as exc:

        print(f"\nSTOPPED: {exc}")

        report["status"] = "stopped_label_not_usable"
        report["reason"] = str(exc)
        report["model_saved"] = False

        write_report(report, args.report)

        return 2

    except (FileNotFoundError, ValueError) as exc:

        print(f"\nERROR: {exc}")

        return 1

    metrics = outcome["metrics"]

    print("\nHeld-out results (resumes the model never saw):")

    for name, values in metrics.items():

        precision = values[f"precision_at_{RANK_K}"]

        print(
            f"  {name:28s} "
            f"precision@{RANK_K}="
            + ("n/a" if precision is None else f"{precision:.4f}")
            + "  MRR="
            + ("n/a" if values["mrr"] is None else f"{values['mrr']:.4f}")
            + "  AUC="
            + ("n/a" if values["roc_auc"] is None else f"{values['roc_auc']:.4f}")
        )

    qualifies, reason = decide_to_save(
        metrics["model"],
        metrics["formula_without_category"],
    )

    importance = permutation_importance_summary(
        outcome["holdout_estimator"],
        outcome["test_frame"],
        args.seed,
    )

    saved_path: Optional[str] = None

    if qualifies or args.force:

        mc.save_model_bundle(
            args.output,
            outcome["estimator"],
            mc.MODEL_FEATURES,
            target=TARGET_DESCRIPTION,
            holdout_metrics=metrics,
            train_pairs=int(len(labelled)),
            monotonic_constraints=list(mc.MODEL_MONOTONIC_CST),
            claim_label="experimental_similarity_not_validated",
            human_validation_status="not_validated",
            target_is_proxy=True,
            human_validated=False,
        )

        saved_path = str(args.output)

        print(
            f"\nMODEL SAVED: {saved_path}"
            + ("" if qualifies else "  (--force: did not beat the formula)")
        )

    else:

        print(f"\nMODEL NOT SAVED: {reason}")
        print("The app and Steps 10/13 keep using the transparent formula.")

    report.update(
        {
            "status": "completed",
            "model_saved": saved_path is not None,
            "model_path": saved_path,
            "target_is_proxy": True,
            "human_validated": False,
            "saved_because": (
                reason if qualifies
                else ("forced" if args.force else None)
            ),
            "not_saved_because": (
                None if (qualifies or args.force) else reason
            ),
            "split": {
                "method": "GroupShuffleSplit by resume_id",
                "test_size": args.test_size,
                "train_pairs": outcome["train_rows"],
                "test_pairs": outcome["test_rows"],
                "train_resumes": outcome["train_resumes"],
                "test_resumes": outcome["test_resumes"],
            },
            "holdout_metrics": metrics,
            "decision_rule": {
                "max_allowed_loss_per_metric": MAX_ALLOWED_LOSS,
                "min_combined_gain": MIN_COMBINED_GAIN,
            },
            "final_model_fit_on": "all labelled pairs (hyper-parameters "
            "fixed before the hold-out evaluation)",
            "permutation_importance_holdout": importance,
            "elapsed_seconds": round(time.time() - started, 2),
            "interpretation": (
                "Metrics measure how well the scorer recovers "
                "same-category pairs on unseen resumes. They are NOT hiring "
                "outcomes. The formula is evaluated without its category "
                "term because uploaded resumes have no category."
            ),
        }
    )

    active_scorer = mc.MatchScorer.load()
    report["claim"] = active_scorer.claim.to_dict()
    report["scoring"] = {
        "scoring_method": active_scorer.method,
        "model_trained_at": active_scorer.metadata.get("trained_at") if active_scorer.uses_model else None,
        "scorer_fingerprint": active_scorer.scorer_fingerprint,
        "claim": active_scorer.claim.to_dict(),
    }

    write_report(report, args.report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
