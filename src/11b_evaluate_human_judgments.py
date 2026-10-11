"""Export candidate rankings for human review and evaluate relevance labels.

The evaluator is deliberately conservative. Suitability wording is never
unlocked by sample size alone: the operator must explicitly approve after
reviewing the metrics, all minimum gates must pass, and the report must match
the exact scorer/retrieval fingerprint used to produce the exported rankings.

Rating scale:
  0 = not relevant
  1 = partly / potentially relevant
  2 = clearly relevant to this resume

This evaluates relevance judgments, not hiring probability or hiring outcomes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matching_core as mc

PROJECT_ROOT = SRC_DIR.parent
DEFAULT_TEMPLATE = PROJECT_ROOT / "data" / "evaluation" / "human_judgment_template.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "human_judgment_evaluation.json"
DEFAULT_RESUME_TEXT_PATH = PROJECT_ROOT / "data" / "processed" / "resumes_text_processed.csv"
DEFAULT_JOB_TEXT_PATH = PROJECT_ROOT / "data" / "processed" / "jobs_text_processed.csv"

MIN_JUDGMENTS = 100
MIN_REVIEWERS = 2
MIN_DOUBLE_REVIEWED_PAIRS = 50
MIN_REVIEWER_AGREEMENT = 0.70
MIN_CONTEXT_COVERAGE = 1.0
MIN_COMPLETE_TOP10_RESUMES = 20
MIN_PRECISION_AT_10 = 0.70
MIN_NDCG_AT_10 = 0.70

SCORE_ALIASES = ("hybrid_match_score", "similarity_score", "match_score", "score")


def _find_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    by_normalized = {
        "".join(ch for ch in str(column).lower() if ch.isalnum()): column
        for column in frame.columns
    }
    for alias in aliases:
        key = "".join(ch for ch in alias.lower() if ch.isalnum())
        if key in by_normalized:
            return by_normalized[key]
    return None


def _normalize_identifier(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            number = float(text)
            if number.is_integer():
                return str(int(number))
        except (TypeError, ValueError):
            pass
    return text


def _fill_text_from_source(
    output: pd.DataFrame,
    *,
    id_column: str,
    text_column: str,
    source_path: Path,
    id_aliases: tuple[str, ...],
    text_aliases: tuple[str, ...],
) -> None:
    if output.empty or not source_path.is_file():
        return
    try:
        header = pd.read_csv(source_path, nrows=0).columns
        source_columns = pd.DataFrame(columns=header)
        source_id = _find_column(source_columns, id_aliases)
        source_text = _find_column(source_columns, text_aliases)
        if source_id is None or source_text is None:
            return
        source = pd.read_csv(source_path, usecols=[source_id, source_text], low_memory=False)
        source["__normalized_id"] = source[source_id].map(_normalize_identifier)
        source = source.drop_duplicates("__normalized_id", keep="first")
        lookup = source.set_index("__normalized_id")[source_text].to_dict()
        existing = output[text_column].fillna("").astype(str)
        missing_text = existing.str.strip().eq("")
        output.loc[missing_text, text_column] = output.loc[missing_text, id_column].map(lookup).fillna("").astype(str)
    except (OSError, ValueError, pd.errors.ParserError):
        # Don't fabricate text or block template export if an optional source
        # table is malformed; the reviewer template will show blanks instead.
        return


def export_review_template(
    input_path: str | Path,
    output_path: str | Path = DEFAULT_TEMPLATE,
    *,
    top_k: int = 20,
) -> Path:
    """Write a ranked CSV with blank reviewer_id and human_relevance cells."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Ranking input not found: {input_path}")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    frame = pd.read_csv(input_path, low_memory=False)
    job_id = _find_column(frame, ("job_id", "jobid", "id"))
    if job_id is None:
        raise ValueError(f"Ranking input needs a job ID column; found {list(frame.columns)}")
    resume_id = _find_column(frame, ("resume_id", "resumeid"))
    score_col = _find_column(frame, SCORE_ALIASES)
    rank_col = _find_column(frame, ("recommendation_rank", "rank", "position"))
    title_col = _find_column(frame, ("job_title", "title", "position_title"))
    description_col = _find_column(frame, ("job_description", "description", "details"))
    source_col = _find_column(frame, ("candidate_retrieval_sources", "retrieval_sources"))
    fingerprint_col = _find_column(frame, ("scorer_fingerprint", "scorerfingerprint"))
    if fingerprint_col is None:
        raise ValueError(
            "Ranking input must contain scorer_fingerprint. Rerun Step 10 with "
            "the current code before exporting human judgments."
        )
    current_fingerprint = mc.MatchScorer.load().scorer_fingerprint
    input_fingerprints = sorted({
        str(value).strip() for value in frame[fingerprint_col].dropna()
        if str(value).strip()
    })
    if input_fingerprints != [current_fingerprint]:
        raise ValueError(
            "Ranking input scorer fingerprint is missing, mixed, or stale. "
            "Rerun Step 10 using the active scorer before exporting a review template. "
            f"Input fingerprint(s): {input_fingerprints}; current: {current_fingerprint}"
        )

    out = pd.DataFrame()
    out["resume_id"] = frame[resume_id].map(_normalize_identifier) if resume_id else "resume_1"
    out["job_id"] = frame[job_id].map(_normalize_identifier)
    out["job_title"] = frame[title_col].fillna("").astype(str) if title_col else ""
    out["resume_text"] = frame[_find_column(frame, ("resume_text", "resume", "resume_str", "text", "cv_text"))].fillna("").astype(str) if _find_column(frame, ("resume_text", "resume", "resume_str", "text", "cv_text")) else ""
    out["job_description"] = frame[description_col].fillna("").astype(str) if description_col else ""
    if score_col:
        out["similarity_score"] = pd.to_numeric(frame[score_col], errors="coerce")
    else:
        out["similarity_score"] = np.nan
    if source_col:
        out["candidate_retrieval_sources"] = frame[source_col].fillna("").astype(str)

    if rank_col:
        out["recommendation_rank"] = pd.to_numeric(frame[rank_col], errors="coerce")
    elif score_col:
        out["recommendation_rank"] = out.groupby("resume_id")["similarity_score"].rank(
            method="first", ascending=False
        )
    else:
        out["recommendation_rank"] = out.groupby("resume_id").cumcount() + 1

    # Normalize ranking and remove duplicate resume/job pairs deterministically.
    out = out.sort_values(["resume_id", "recommendation_rank", "job_id"], kind="mergesort")
    out = out.drop_duplicates(["resume_id", "job_id"], keep="first")
    out = out.groupby("resume_id", sort=False, group_keys=False).head(top_k).copy()

    # Step 10 intentionally carries features/IDs rather than full resume/job
    # text. Pull that context from Step 4's processed text tables so reviewers
    # judge the actual pair rather than only the title or model score.
    _fill_text_from_source(
        out, id_column="resume_id", text_column="resume_text",
        source_path=DEFAULT_RESUME_TEXT_PATH,
        id_aliases=("resume_id", "resumeid", "id"),
        text_aliases=("resume_text", "resume", "resume_str", "text", "cv_text"),
    )
    _fill_text_from_source(
        out, id_column="job_id", text_column="job_description",
        source_path=DEFAULT_JOB_TEXT_PATH,
        id_aliases=("job_id", "jobid", "id"),
        text_aliases=("job_description", "description", "details"),
    )
    out["scorer_fingerprint"] = frame.loc[out.index, fingerprint_col].astype(str)
    out["reviewer_id"] = ""
    out["human_relevance"] = ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return output_path


def _dcg(relevances: list[float], k: int) -> float:
    return sum((2.0 ** rel - 1.0) / math.log2(position + 2) for position, rel in enumerate(relevances[:k]))


def evaluate_judgments(
    input_path: str | Path,
    output_path: str | Path = DEFAULT_REPORT,
    *,
    approve_suitability_wording: bool = False,
) -> dict[str, Any]:
    """Evaluate an annotated template and write a scorer-fingerprinted report."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Human-judgment CSV not found: {input_path}")
    frame = pd.read_csv(input_path, low_memory=False, dtype={"human_relevance": "string"})
    required = {
        "resume_id", "job_id", "resume_text", "job_description",
        "human_relevance", "reviewer_id", "scorer_fingerprint",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Human-judgment file missing required columns: {missing}")

    raw_labels = frame["human_relevance"].astype("string").str.strip()
    nonblank = raw_labels.notna() & raw_labels.ne("")
    labels = pd.to_numeric(raw_labels, errors="coerce")
    invalid = nonblank & (~labels.isin([0, 1, 2]))
    if invalid.any():
        examples = sorted(set(raw_labels.loc[invalid].astype(str).tolist()))[:5]
        raise ValueError(
            "Invalid human_relevance labels must be blank or one of 0, 1, 2; "
            f"found {int(invalid.sum())} invalid value(s), examples={examples}."
        )

    judged_mask = labels.isin([0, 1, 2])
    judged = frame.loc[judged_mask].copy()
    judged["human_relevance"] = labels.loc[judged_mask].astype(float)
    if judged.empty:
        raise ValueError("No valid human_relevance labels found. Use 0, 1, or 2.")
    missing_reviewer = judged["reviewer_id"].isna() | judged["reviewer_id"].astype(str).str.strip().eq("")
    if missing_reviewer.any():
        raise ValueError(
            f"{int(missing_reviewer.sum())} judged row(s) are missing reviewer_id. "
            "Each rating must identify the reviewer."
        )
    missing_fingerprint = (
        judged["scorer_fingerprint"].isna()
        | judged["scorer_fingerprint"].astype(str).str.strip().eq("")
    )
    if missing_fingerprint.any():
        raise ValueError(
            f"{int(missing_fingerprint.sum())} judged row(s) are missing scorer_fingerprint. "
            "Re-export a template from the ranking file."
        )
    missing_ids = (
        judged["resume_id"].isna() | judged["resume_id"].astype(str).str.strip().eq("")
        | judged["job_id"].isna() | judged["job_id"].astype(str).str.strip().eq("")
    )
    if missing_ids.any():
        raise ValueError(f"{int(missing_ids.sum())} judged row(s) are missing resume_id or job_id.")
    duplicate_ratings = judged.duplicated(["reviewer_id", "resume_id", "job_id"], keep=False)
    if duplicate_ratings.any():
        raise ValueError(
            f"Found {int(duplicate_ratings.sum())} duplicate reviewer/resume/job rating row(s). "
            "Each reviewer may rate a resume-job pair only once."
        )
    context_complete = (
        judged["resume_text"].fillna("").astype(str).str.strip().ne("")
        & judged["job_description"].fillna("").astype(str).str.strip().ne("")
    )
    context_coverage = float(context_complete.mean()) if len(judged) else 0.0
    context_complete_count = int(context_complete.sum())

    fingerprints = sorted({str(v).strip() for v in judged["scorer_fingerprint"].dropna() if str(v).strip()})
    current_scorer = mc.MatchScorer.load()
    current_fingerprint = current_scorer.scorer_fingerprint
    fingerprint_matches = len(fingerprints) == 1 and fingerprints[0] == current_fingerprint
    unique_reviewers = {
        str(value).strip() for value in judged["reviewer_id"]
        if pd.notna(value) and str(value).strip()
    }

    # Different reviewers may rate the same pair. Average those labels for
    # ranking metrics while retaining reviewer-level rating count for the gate.
    aggregation = {"human_relevance": "mean"}
    if "recommendation_rank" in judged.columns:
        judged["recommendation_rank"] = pd.to_numeric(judged["recommendation_rank"], errors="coerce")
        aggregation["recommendation_rank"] = "min"
    elif "similarity_score" in judged.columns:
        judged["similarity_score"] = pd.to_numeric(judged["similarity_score"], errors="coerce")
        aggregation["similarity_score"] = "max"
    pair_rows = judged.groupby(["resume_id", "job_id"], as_index=False, dropna=False).agg(aggregation)

    # Independent reviewers must overlap on enough pairs to support a basic
    # agreement check. This is exact ordinal-label agreement, not a causal or
    # population-validity metric.
    pair_reviewer_counts = judged.groupby(["resume_id", "job_id"])["reviewer_id"].nunique()
    double_reviewed_pairs = int((pair_reviewer_counts >= 2).sum())
    agreement_by_pair: list[float] = []
    for _, pair_group in judged.groupby(["resume_id", "job_id"], sort=False):
        if pair_group["reviewer_id"].nunique() < 2:
            continue
        label_counts = pair_group["human_relevance"].value_counts()
        agreement_by_pair.append(float(label_counts.max() / label_counts.sum()))
    reviewer_agreement = float(np.mean(agreement_by_pair)) if agreement_by_pair else None

    complete_top10_groups: list[pd.DataFrame] = []
    ndcg_values: list[float] = []
    precision_values: list[float] = []
    for _, group in pair_rows.groupby("resume_id", sort=False):
        if "recommendation_rank" in group.columns:
            ordered = group.sort_values(["recommendation_rank", "job_id"], kind="mergesort")
        elif "similarity_score" in group.columns:
            ordered = group.sort_values(["similarity_score", "job_id"], ascending=[False, True], kind="mergesort")
        else:
            ordered = group.sort_values("job_id", kind="mergesort")
        top10 = ordered.head(10)
        if len(top10) == 10:
            complete_top10_groups.append(top10)
            precision_values.append(float((top10["human_relevance"] >= 1).mean()))
            actual = top10["human_relevance"].astype(float).tolist()
            ideal = sorted(ordered["human_relevance"].astype(float).tolist(), reverse=True)[:10]
            ideal_dcg = _dcg(ideal, 10)
            ndcg_values.append(_dcg(actual, 10) / ideal_dcg if ideal_dcg > 0 else 0.0)

    mean_precision = float(np.mean(precision_values)) if precision_values else None
    mean_ndcg = float(np.mean(ndcg_values)) if ndcg_values else None
    labels_seen = sorted({int(v) for v in judged["human_relevance"].tolist()})
    reasons: list[str] = []
    if len(judged) < MIN_JUDGMENTS:
        reasons.append(f"Need at least {MIN_JUDGMENTS} reviewer-level judgments; found {len(judged)}.")
    if len(unique_reviewers) < MIN_REVIEWERS:
        reasons.append(f"Need at least {MIN_REVIEWERS} distinct reviewer IDs; found {len(unique_reviewers)}.")
    if context_coverage < MIN_CONTEXT_COVERAGE:
        reasons.append(
            f"Every judged row must include both resume_text and job_description; "
            f"complete context coverage was {context_coverage:.3f}."
        )
    if double_reviewed_pairs < MIN_DOUBLE_REVIEWED_PAIRS:
        reasons.append(
            f"Need at least {MIN_DOUBLE_REVIEWED_PAIRS} resume-job pairs rated by 2+ reviewers; "
            f"found {double_reviewed_pairs}."
        )
    if reviewer_agreement is None or reviewer_agreement < MIN_REVIEWER_AGREEMENT:
        reasons.append(
            f"Mean reviewer agreement must be at least {MIN_REVIEWER_AGREEMENT:.2f}; "
            f"found {reviewer_agreement}."
        )
    if len(complete_top10_groups) < MIN_COMPLETE_TOP10_RESUMES:
        reasons.append(
            f"Need at least {MIN_COMPLETE_TOP10_RESUMES} resumes with all top-10 jobs judged; "
            f"found {len(complete_top10_groups)}."
        )
    if len(labels_seen) < 2 or 0 not in labels_seen or 2 not in labels_seen:
        reasons.append("Judgments must include both 0 (not relevant) and 2 (clearly relevant).")
    if not fingerprint_matches:
        reasons.append("Judgment export scorer fingerprint does not match the current scorer version.")
    if mean_precision is None or mean_precision < MIN_PRECISION_AT_10:
        reasons.append(f"Mean precision@10 must be at least {MIN_PRECISION_AT_10:.2f}.")
    if mean_ndcg is None or mean_ndcg < MIN_NDCG_AT_10:
        reasons.append(f"Mean nDCG@10 must be at least {MIN_NDCG_AT_10:.2f}.")
    if not approve_suitability_wording:
        reasons.append("Explicit --approve-suitability-wording sign-off was not supplied.")

    allowed = not reasons
    report: dict[str, Any] = {
        "schema_version": 1,
        "human_validation_status": "approved" if allowed else "not_validated",
        "suitability_wording_allowed": allowed,
        "scorer_fingerprint": fingerprints[0] if len(fingerprints) == 1 else None,
        "current_scorer_fingerprint": current_fingerprint,
        "scorer_fingerprint_matches": bool(fingerprint_matches),
        "judgment_count": int(len(judged)),
        "judged_pair_count": int(pair_rows[["resume_id", "job_id"]].drop_duplicates().shape[0]),
        "resume_count": int(judged["resume_id"].nunique()),
        "reviewer_count": int(len(unique_reviewers)),
        "double_reviewed_pair_count": double_reviewed_pairs,
        "mean_reviewer_agreement": reviewer_agreement,
        "context_complete_judgment_count": context_complete_count,
        "context_coverage": context_coverage,
        "labels_seen": labels_seen,
        "complete_top10_resume_count": int(len(complete_top10_groups)),
        "precision_at_10": mean_precision,
        "ndcg_at_10": mean_ndcg,
        "gates": {
            "minimum_judgments": MIN_JUDGMENTS,
            "minimum_reviewers": MIN_REVIEWERS,
            "minimum_double_reviewed_pairs": MIN_DOUBLE_REVIEWED_PAIRS,
            "minimum_reviewer_agreement": MIN_REVIEWER_AGREEMENT,
            "minimum_context_coverage": MIN_CONTEXT_COVERAGE,
            "minimum_complete_top10_resumes": MIN_COMPLETE_TOP10_RESUMES,
            "minimum_precision_at_10": MIN_PRECISION_AT_10,
            "minimum_ndcg_at_10": MIN_NDCG_AT_10,
            "explicit_operator_signoff": bool(approve_suitability_wording),
        },
        "reasons_not_approved": reasons,
        "limitations": [
            "These metrics evaluate judged relevance in this sample, not hiring probability or hiring outcomes.",
            "Thresholds are operational safeguards, not proof of generalization to all roles or populations.",
            "Suitability wording remains disabled unless the reviewer evaluates the report and explicitly signs off.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--export", type=Path, help="Ranking CSV to turn into a review template")
    modes.add_argument("--evaluate", type=Path, help="Annotated human-judgment CSV to evaluate")
    parser.add_argument("--output", type=Path, help="Output template path (for --export)")
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT, help="Evaluation report path")
    parser.add_argument("--top-k", type=int, default=20, help="Candidates per resume to export")
    parser.add_argument(
        "--approve-suitability-wording", action="store_true",
        help="Explicitly approve suitability wording only if every validation gate passes",
    )
    args = parser.parse_args()

    if args.export:
        path = export_review_template(args.export, args.output or DEFAULT_TEMPLATE, top_k=args.top_k)
        print(f"Review template written: {path}")
        print("Have at least two reviewers label human_relevance as 0, 1, or 2 and fill reviewer_id.")
    else:
        report = evaluate_judgments(
            args.evaluate,
            args.report_output,
            approve_suitability_wording=args.approve_suitability_wording,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Evaluation report written: {args.report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
