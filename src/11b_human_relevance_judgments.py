"""
======================================================================
AI RESUME ANALYZER - STEP 11b HUMAN RELEVANCE JUDGMENTS
======================================================================

WHY THIS STEP EXISTS
--------------------
Step 11 ("automatic validation") only checks that scores are consistent
with each other. Nothing in the project says whether a high score means a
resume genuinely suits a job: the trained model's target is a proxy
(same resume/job category, Step 10b) and the formula's weights are
hand-chosen. Until people judge real recommendations, the app and reports
call the score a similarity ranking and label the model experimental
(``matching_core.resolve_claim``).

This step collects those judgments and tests the claim.

    python src/11b_human_relevance_judgments.py export   [--resumes 40]
        -> data/evaluation/human_judgments_template.csv
           Fill the ``relevance`` column (and your name in ``annotator``).

    python src/11b_human_relevance_judgments.py evaluate --judgments FILE...
        -> reports/human_relevance_report.json
           Compares every scorer with the judgments and records which
           claims the evidence supports. The app/CLI read that file.

RUBRIC (relevance, integer 0-3)
-------------------------------
    0  not relevant     different field; this resume would not apply
    1  weakly relevant  adjacent field or seniority mismatch; unlikely
    2  relevant         could reasonably apply; main requirements met/close
    3  strong fit       clearly the kind of role this resume is built for

DESIGN
------
* Pooled judging: the pairs to judge are the top-k of SEVERAL rankers
  (formula, trained model if present, semantic-only, skill-only) plus a few
  random pairs, shuffled, with no scores shown - so judges are not anchored
  and no single ranker picks its own test set.
* More than one annotator on the same pairs is strongly recommended;
  agreement (quadratic-weighted kappa) is reported and gates the claim.
* A claim is only recorded as supported when there are enough judged
  resumes/pairs, the scorer's per-resume rank correlation with the judges
  clears a threshold, and (for the model) it is not worse than the
  formula or semantic-only baselines (bootstrap over resumes).
* The category proxy itself is tested against the judges (AUC), so it is
  visible whether "same category" ever meant "relevant".
======================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import matching_core as mc

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "resume_job_matching_dataset.csv"
TEMPLATE_PATH = PROJECT_ROOT / "data" / "evaluation" / "human_judgments_template.csv"
REPORT_PATH = mc.HUMAN_REPORT_PATH

# ---- evidence thresholds (deliberately explicit) ---------------------------
MIN_JUDGED_RESUMES = 30
MIN_JUDGED_PAIRS = 300
MIN_PAIRS_PER_RESUME_FOR_CORRELATION = 5
MIN_MEAN_SPEARMAN = 0.30          # score vs judges, averaged over resumes
NON_INFERIORITY_MARGIN = 0.02     # model may be at most this far below a baseline (nDCG@10)
MIN_KAPPA = 0.40                  # only enforced when >= MIN_OVERLAP pairs doubly judged
MIN_OVERLAP = 30
PROXY_MIN_AUC = 0.65
RELEVANT_AT = 2
BOOTSTRAPS = 1000
NDCG_K = 10

RELEVANCE_VALUES = (0, 1, 2, 3)

TEXT_EXCERPT = 1500

SCORER_COLUMNS = {
    "formula_without_category": "online formula (no category term)",
    "model": "trained model",
    "semantic_similarity_only": "semantic similarity only",
    "skill_match_only": "skill match only",
    "category_proxy": "same-category proxy (the model's training target)",
}


# ======================================================================
# SCORERS (all on the Step 9 feature columns)
# ======================================================================

def score_pairs(frame: pd.DataFrame, scorer: Optional[mc.MatchScorer] = None) -> pd.DataFrame:
    """One column per scorer, aligned with ``frame``."""

    scorer = scorer or mc.MatchScorer.load()

    features = frame[list(mc.MODEL_FEATURES)]

    out = pd.DataFrame(index=frame.index)

    out["formula_without_category"] = mc.formula_score(features)
    out["semantic_similarity_only"] = frame["semantic_similarity"].to_numpy(float)
    out["skill_match_only"] = frame["skill_match_signal"].to_numpy(float)

    if "category_match_signal" in frame.columns:
        out["category_proxy"] = frame["category_match_signal"].to_numpy(float)

    if scorer.uses_model:
        out["model"] = scorer.score(frame).scores

    return out


# ======================================================================
# EXPORT
# ======================================================================

def build_template(
    dataset: pd.DataFrame,
    *,
    resumes: int = 40,
    per_ranker_k: int = 5,
    random_extra: int = 3,
    seed: int = 42,
    scorer: Optional[mc.MatchScorer] = None,
    min_candidates: int = 15,
) -> pd.DataFrame:
    """Pooled, shuffled, score-free annotation sheet."""

    rng = np.random.default_rng(seed)

    scores = score_pairs(dataset, scorer)

    counts = dataset.groupby("resume_id").size()

    eligible = counts[counts >= min_candidates].index.to_numpy()

    if len(eligible) == 0:
        raise ValueError(
            f"No resume has at least {min_candidates} candidate jobs."
        )

    chosen = rng.choice(eligible, size=min(resumes, len(eligible)), replace=False)

    text_resume = (
        "resume_text_representation"
        if "resume_text_representation" in dataset.columns
        else None
    )

    text_job = (
        "job_text_representation"
        if "job_text_representation" in dataset.columns
        else None
    )

    rows = []

    for resume_id in sorted(chosen.tolist()):

        index = dataset.index[dataset["resume_id"] == resume_id]

        pool = set()

        for column in ("formula_without_category", "model",
                       "semantic_similarity_only", "skill_match_only"):

            if column not in scores.columns:
                continue

            ordered = scores.loc[index, column].sort_values(
                ascending=False, kind="mergesort"
            )

            pool.update(ordered.index[:per_ranker_k])

        remaining = [i for i in index if i not in pool]

        if remaining and random_extra:
            pool.update(rng.choice(remaining, size=min(random_extra, len(remaining)),
                                   replace=False).tolist())

        for i in pool:

            row = dataset.loc[i]

            rows.append(
                {
                    "resume_id": row["resume_id"],
                    "job_id": row["job_id"],
                    "job_title": row.get("job_title", ""),
                    "resume_excerpt": mc.clean_text(row[text_resume])[:TEXT_EXCERPT]
                    if text_resume else "",
                    "job_excerpt": mc.clean_text(row[text_job])[:TEXT_EXCERPT]
                    if text_job else "",
                    "relevance": "",
                    "annotator": "",
                    "notes": "",
                }
            )

    sheet = pd.DataFrame(rows)

    # Shuffle so row order carries no ranking information.
    order = rng.permutation(len(sheet))

    return sheet.iloc[order].reset_index(drop=True)


# ======================================================================
# EVALUATION
# ======================================================================

def load_judgments(paths: List[Path]) -> pd.DataFrame:

    frames = []

    for path in paths:

        frame = pd.read_csv(path, dtype={"annotator": str})

        needed = {"resume_id", "job_id", "relevance"}

        if not needed <= set(frame.columns):
            raise ValueError(f"{path} is missing columns {sorted(needed - set(frame.columns))}")

        if "annotator" not in frame.columns:
            frame["annotator"] = ""

        frame["annotator"] = frame["annotator"].fillna("").astype(str).str.strip()

        frame.loc[frame["annotator"] == "", "annotator"] = Path(path).stem

        frames.append(frame[["resume_id", "job_id", "relevance", "annotator"]])

    return pd.concat(frames, ignore_index=True)


def clean_judgments(raw: pd.DataFrame) -> Dict[str, Any]:
    """Validate; returns the usable rows and what was dropped."""

    relevance = pd.to_numeric(raw["relevance"], errors="coerce")

    # Blank = nothing was typed. Text that is not a number ("abc") or a
    # rating outside 0-3 is INVALID, not blank, so typos are reported.
    blank = raw["relevance"].isna() | (
        raw["relevance"].astype(str).str.strip() == ""
    )
    invalid = ~blank & ~relevance.isin(RELEVANCE_VALUES)

    usable = raw[~blank & ~invalid].copy()
    usable["relevance"] = relevance[~blank & ~invalid].astype(int)

    return {
        "usable": usable,
        "blank_rows": int(blank.sum()),
        "invalid_rows": int(invalid.sum()),
    }


def cohen_kappa(usable: pd.DataFrame) -> Dict[str, Any]:
    """Quadratic-weighted kappa between the first two annotators that overlap."""

    annotators = sorted(usable["annotator"].unique())

    if len(annotators) < 2:
        return {"annotators": len(annotators), "kappa": None, "overlap_pairs": 0}

    best = None

    for i, a in enumerate(annotators):
        for b in annotators[i + 1:]:

            left = usable[usable["annotator"] == a].set_index(["resume_id", "job_id"])["relevance"]
            right = usable[usable["annotator"] == b].set_index(["resume_id", "job_id"])["relevance"]

            both = left.index.intersection(right.index)

            if best is None or len(both) > best[0]:
                best = (len(both), a, b, left.loc[both], right.loc[both])

    overlap, a, b, left, right = best

    if overlap < 2 or left.nunique() < 2 and right.nunique() < 2:
        return {"annotators": len(annotators), "kappa": None, "overlap_pairs": int(overlap)}

    from sklearn.metrics import cohen_kappa_score

    kappa = float(
        cohen_kappa_score(left.to_numpy(), right.to_numpy(),
                          weights="quadratic", labels=list(RELEVANCE_VALUES))
    )

    return {"annotators": len(annotators), "annotator_pair": [a, b],
            "kappa": kappa, "overlap_pairs": int(overlap)}


def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int = NDCG_K) -> Optional[float]:

    order = np.lexsort((np.arange(len(scores)), -scores))
    gains = (2.0 ** relevance[order][:k] - 1.0)
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))

    ideal = np.sort(2.0 ** relevance - 1.0)[::-1][:k]

    denominator = float((ideal * (1.0 / np.log2(np.arange(2, len(ideal) + 2)))).sum())

    if denominator == 0:
        return None

    return float((gains * discounts).sum() / denominator)


def per_resume_metrics(judged: pd.DataFrame, column: str) -> pd.DataFrame:
    """nDCG@k, precision@5 and Spearman for each resume with judgments."""

    from scipy.stats import spearmanr

    rows = []

    for resume_id, group in judged.groupby("resume_id"):

        relevance = group["relevance"].to_numpy(float)
        scores = group[column].to_numpy(float)

        ndcg = ndcg_at_k(relevance, scores)

        if ndcg is None:
            continue

        top = np.lexsort((np.arange(len(scores)), -scores))[:5]

        rho = np.nan

        if (
            len(group) >= MIN_PAIRS_PER_RESUME_FOR_CORRELATION
            and np.ptp(relevance) > 0
            and np.ptp(scores) > 0
        ):
            rho = float(spearmanr(scores, relevance).statistic)

        rows.append(
            {
                "resume_id": resume_id,
                "ndcg": ndcg,
                "precision_at_5": float((relevance[top] >= RELEVANT_AT).mean()),
                "spearman": rho,
            }
        )

    return pd.DataFrame(rows)


def bootstrap_difference(a: pd.DataFrame, b: pd.DataFrame, seed: int) -> Dict[str, float]:
    """95% CI of mean(ndcg_a - ndcg_b), resampling resumes."""

    merged = a[["resume_id", "ndcg"]].merge(
        b[["resume_id", "ndcg"]], on="resume_id", suffixes=("_a", "_b")
    )

    diff = (merged["ndcg_a"] - merged["ndcg_b"]).to_numpy()

    if len(diff) == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "resumes": 0}

    rng = np.random.default_rng(seed)

    means = np.array(
        [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(BOOTSTRAPS)]
    )

    return {
        "mean": float(diff.mean()),
        "low": float(np.percentile(means, 2.5)),
        "high": float(np.percentile(means, 97.5)),
        "resumes": int(len(diff)),
    }


def proxy_validity(judged: pd.DataFrame) -> Dict[str, Any]:
    """Does 'same category' predict human relevance? (AUC)"""

    if "category_proxy" not in judged.columns or "category_available" not in judged.columns:
        return {"auc": None, "pairs": 0}

    known = judged[judged["category_available"].astype(bool)]

    positive = (known["relevance"] >= RELEVANT_AT).astype(int)

    if len(known) < 20 or positive.nunique() < 2:
        return {"auc": None, "pairs": int(len(known))}

    from sklearn.metrics import roc_auc_score

    return {
        "auc": float(roc_auc_score(positive, known["category_proxy"])),
        "pairs": int(len(known)),
    }


def evaluate(
    dataset: pd.DataFrame,
    raw_judgments: pd.DataFrame,
    *,
    scorer: Optional[mc.MatchScorer] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """The full report (see the module docstring)."""

    scorer = scorer or mc.MatchScorer.load()

    cleaned = clean_judgments(raw_judgments)

    usable = cleaned["usable"]

    agreement = cohen_kappa(usable)

    # Average multiple annotators per pair.
    per_pair = (
        usable.groupby(["resume_id", "job_id"], as_index=False)["relevance"].mean()
    )

    judged = per_pair.merge(
        dataset.drop_duplicates(["resume_id", "job_id"]),
        on=["resume_id", "job_id"], how="inner",
    ).reset_index(drop=True)

    unmatched = int(len(per_pair) - len(judged))

    scores = score_pairs(judged, scorer)

    judged = pd.concat([judged, scores], axis=1)

    resumes = int(judged["resume_id"].nunique())

    sufficient = resumes >= MIN_JUDGED_RESUMES and len(judged) >= MIN_JUDGED_PAIRS

    metrics: Dict[str, Any] = {}
    tables: Dict[str, pd.DataFrame] = {}

    for column in scores.columns:

        table = per_resume_metrics(judged, column)

        tables[column] = table

        metrics[column] = {
            "label": SCORER_COLUMNS.get(column, column),
            "resumes_evaluated": int(len(table)),
            f"ndcg_at_{NDCG_K}": float(table["ndcg"].mean()) if len(table) else None,
            "precision_at_5": float(table["precision_at_5"].mean()) if len(table) else None,
            "mean_spearman": float(table["spearman"].mean(skipna=True))
            if len(table) and table["spearman"].notna().any() else None,
        }

    comparisons = {}

    if "model" in tables:
        for baseline in ("formula_without_category", "semantic_similarity_only"):
            comparisons[f"model_minus_{baseline}"] = bootstrap_difference(
                tables["model"], tables[baseline], seed
            )

    kappa_ok = (
        agreement["kappa"] is None
        or agreement["overlap_pairs"] < MIN_OVERLAP
        or agreement["kappa"] >= MIN_KAPPA
    )

    def ranks_by_humans(name: str) -> bool:

        rho = (metrics.get(name) or {}).get("mean_spearman")

        return bool(sufficient and kappa_ok and rho is not None and rho >= MIN_MEAN_SPEARMAN)

    model_ok = ranks_by_humans("model")

    if model_ok:
        for key, comparison in comparisons.items():
            if not (comparison["low"] > -NON_INFERIORITY_MARGIN):
                model_ok = False

    proxy = proxy_validity(judged)

    claims = {
        "model_ranks_by_human_relevance": bool(scorer.uses_model and model_ok),
        "formula_ranks_by_human_relevance": ranks_by_humans("formula_without_category"),
        "category_proxy_tracks_human_relevance": bool(
            proxy["auc"] is not None and proxy["auc"] >= PROXY_MIN_AUC
        ),
    }

    return {
        "step": "11b",
        "name": "Human relevance judgments",
        "status": "evaluated" if sufficient else "insufficient_data",
        "judged_pairs": int(len(judged)),
        "judged_resumes": resumes,
        "annotator_rows": int(len(usable)),
        "blank_rows_ignored": cleaned["blank_rows"],
        "invalid_rows_ignored": cleaned["invalid_rows"],
        "pairs_not_in_dataset": unmatched,
        "agreement": agreement,
        "agreement_gate_passed": bool(kappa_ok),
        "model_in_use": bool(scorer.uses_model),
        "model_path": scorer.model_path,
        "model_trained_at": scorer.metadata.get("trained_at") if scorer.uses_model else None,
        "scorer_fingerprint": scorer.scorer_fingerprint,
        "suitability_wording_approved": False,
        "operator_approval_required": True,
        "criteria": {
            "min_judged_resumes": MIN_JUDGED_RESUMES,
            "min_judged_pairs": MIN_JUDGED_PAIRS,
            "min_mean_spearman": MIN_MEAN_SPEARMAN,
            "non_inferiority_margin_ndcg": NON_INFERIORITY_MARGIN,
            "min_kappa_when_overlap_at_least": [MIN_KAPPA, MIN_OVERLAP],
            "proxy_min_auc": PROXY_MIN_AUC,
            "relevant_means_rating_at_least": RELEVANT_AT,
        },
        "metrics": metrics,
        "model_vs_baselines_ndcg_difference_95ci": comparisons,
        "category_proxy_vs_human_relevant": proxy,
        "claims": claims if sufficient else {k: False for k in claims},
        "interpretation": (
            "Metrics are computed only over judged (pooled) pairs, so they "
            "compare scorers with each other and with the judges; they are "
            "not estimates of hiring outcomes. A claim is True only when the "
            "evidence thresholds in 'criteria' are met."
        ),
    }


# ======================================================================
# CLI
# ======================================================================

def main(argv: Optional[List[str]] = None) -> int:

    parser = argparse.ArgumentParser(description=__doc__.split("WHY")[0])

    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="write the annotation sheet")
    export.add_argument("--input", type=Path, default=DATASET_PATH)
    export.add_argument("--output", type=Path, default=TEMPLATE_PATH)
    export.add_argument("--resumes", type=int, default=40)
    export.add_argument("--per-ranker-k", type=int, default=5)
    export.add_argument("--seed", type=int, default=42)

    evaluate_parser = sub.add_parser("evaluate", help="score the judgments")
    evaluate_parser.add_argument("--input", type=Path, default=DATASET_PATH)
    evaluate_parser.add_argument("--judgments", type=Path, nargs="+", required=True)
    evaluate_parser.add_argument("--report", type=Path, default=REPORT_PATH)
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.add_argument(
        "--approve-suitability-wording", action="store_true",
        help="explicitly approve the wording gate only if all evaluation criteria pass",
    )

    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"ERROR: Step 9 dataset not found: {args.input}")
        return 1

    dataset = pd.read_csv(args.input)

    if args.command == "export":

        sheet = build_template(
            dataset, resumes=args.resumes, per_ranker_k=args.per_ranker_k, seed=args.seed
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        sheet.to_csv(args.output, index=False)

        print(f"Wrote {len(sheet)} pairs for {sheet['resume_id'].nunique()} resumes: {args.output}")
        print("Fill 'relevance' (0-3, see the rubric at the top of this file) and "
              "'annotator'. Use 2+ annotators on the same file if you can.")

        return 0

    scorer = mc.MatchScorer.load()
    report = evaluate(
        dataset, load_judgments(args.judgments), scorer=scorer, seed=args.seed
    )

    supported_claim = (
        report["claims"].get("model_ranks_by_human_relevance", False)
        if scorer.uses_model
        else report["claims"].get("formula_ranks_by_human_relevance", False)
    )
    if args.approve_suitability_wording:
        report["suitability_wording_approved"] = bool(
            report["status"] == "evaluated"
            and supported_claim
            and report.get("scorer_fingerprint") == scorer.scorer_fingerprint
            and (
                not scorer.uses_model
                or report.get("model_trained_at") == scorer.metadata.get("trained_at")
            )
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Status: {report['status']}  ({report['judged_pairs']} pairs, "
          f"{report['judged_resumes']} resumes)")
    print(f"Claims supported: {report['claims']}")
    print(f"Suitability wording approved: {report['suitability_wording_approved']}")
    print(f"Report: {args.report}")

    if args.approve_suitability_wording and not report["suitability_wording_approved"]:
        print("Approval denied: the active scorer's relevance criteria did not pass, the data was insufficient, or the scorer fingerprint/model artifact did not match.")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
