"""
======================================================================
AI RESUME ANALYZER - DEPLOYMENT-PATH VERIFICATION
======================================================================

Runs AFTER the pipeline (see run_pipeline.py) against the real data and
checks that what was trained/regenerated offline is what the app and the
CLI will actually run. Writes reports/deployment_verification_report.json
and exits non-zero if any check FAILS (warnings do not fail).

Checks
------
1  model_decision        model file present/absent == what Steps 10 and 12
                         recorded; same trained_at; feature names == code
2  step10_reproduces     the scorer, run again on Step 10's own feature
                         columns, gives Step 10's scores
3  offline_vs_live       for sampled resumes, the live path (job_ranker,
                         as used by the app/CLI) vs the offline Step 7/9
                         path: candidate sets, feature values, ESCO skill
                         sets, and how much of the full-scan top-10 the
                         candidate pool keeps
4  cli_and_app           CLI (Step 13), job_ranker.rank_resume and the
                         app's own analyze_resume return the same jobs and
                         scores for the same resume PDF

Hard failures are reserved for things that must be identical (semantic,
title and text signals; the scorer; the model decision; CLI/app/ranker
agreement). Differences that come from how skills are extracted (offline
Step 5/6 vs live) are MEASURED and reported as warnings with numbers.

USAGE
-----
python src/verify_deployment.py [--resumes 10] [--resume-pdf PDF]
                                [--expect-model present|absent|any]
                                [--skip-cli] [--skip-live]
======================================================================
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import matching_core as mc
from candidate_retrieval import DEFAULT_POLICY, FULL_SCAN_POLICY

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"

STEP9_PATH = PROCESSED / "resume_job_matching_dataset.csv"
STEP10_PREDICTIONS = PROJECT_ROOT / "data" / "evaluation" / "step10_matching_predictions.csv"
STEP10_REPORT = REPORTS / "hybrid_matching_model_report.json"
STEP12_REPORT = REPORTS / "final_recommendation_report.json"
STEP12_OUTPUT = PROJECT_ROOT / "data" / "recommendations" / "final_job_recommendations.csv"
TRAIN_REPORT = REPORTS / "match_model_report.json"
CLI_ANALYSIS = PROJECT_ROOT / "data" / "recommendations" / "test_resume_analysis.json"
DEFAULT_PDF = PROJECT_ROOT / "data" / "test_resume.pdf"
REPORT_PATH = REPORTS / "deployment_verification_report.json"

# Signals that do not depend on ESCO skill extraction must agree exactly
# (up to float32 rounding in the stored embeddings).
EXACT_FEATURES = ("semantic_similarity", "title_similarity", "text_containment")
EXACT_TOLERANCE = 1e-3

SKILL_FEATURES = [f for f in mc.MODEL_FEATURES if f not in EXACT_FEATURES]

WARN_MEAN_SKILL_DIFF = 0.02
WARN_MIN_CANDIDATE_JACCARD = 0.90
WARN_MIN_TOP10_RECALL = 0.90

SCORE_TOLERANCE = 1e-6


@dataclass
class Check:
    name: str
    status: str            # pass | warn | fail | skip
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "status": self.status,
                "detail": self.detail, "data": self.data}


def load_module(path: Path, name: str):

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def read_json(path: Path) -> Optional[Dict[str, Any]]:

    if not path.is_file():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def ids(values) -> List[str]:
    return [str(v).strip() for v in values]


# ======================================================================
# 1. MODEL DECISION
# ======================================================================

def check_model_decision(expect_model: str = "any") -> Check:

    scorer = mc.MatchScorer.load()

    model_path = Path(os.getenv("MATCH_MODEL_PATH") or mc.DEFAULT_MODEL_PATH)

    file_present = model_path.is_file()

    problems: List[str] = []

    data: Dict[str, Any] = {
        "model_file": str(model_path),
        "model_file_present": file_present,
        "scorer_method": scorer.method,
        "scorer_uses_model": scorer.uses_model,
        "claim": scorer.claim.to_dict(),
    }

    if expect_model == "present" and not scorer.uses_model:
        problems.append(
            "a model was expected but the scorer falls back to the formula"
            + (f" ({scorer.fallback_reason})" if scorer.fallback_reason else "")
        )

    if expect_model == "absent" and file_present:
        problems.append(
            f"no model was expected but {model_path} exists (the app would "
            "use it) - run_pipeline.py --model omit sets it aside"
        )

    if file_present and not scorer.uses_model:
        problems.append(
            "the model file exists but cannot be used: "
            + (scorer.fallback_reason or "unknown reason")
        )

    if scorer.uses_model and list(scorer.feature_names) != list(mc.MODEL_FEATURES):
        problems.append(
            f"model features {list(scorer.feature_names)} differ from the "
            f"code's {list(mc.MODEL_FEATURES)}"
        )

    # ---- what the steps recorded -------------------------------------
    step10 = (read_json(STEP10_REPORT) or {}).get("scoring")
    step12 = (read_json(STEP12_REPORT) or {}).get("scoring")
    train = read_json(TRAIN_REPORT)

    for label, recorded in (("Step 10", step10), ("Step 12", step12)):

        if recorded is None:
            problems.append(f"{label} report missing - run the step")
            continue

        method = recorded.get("scoring_method")

        if method != scorer.method:
            problems.append(
                f"{label} scored with '{method}' but the scorer in use is "
                f"'{scorer.method}' - re-run Steps 10-12"
            )

        if scorer.uses_model and recorded.get("model_trained_at") != scorer.metadata.get("trained_at"):
            problems.append(
                f"{label} used a different model file than the one in "
                f"{model_path.name} (trained_at "
                f"{recorded.get('model_trained_at')} vs "
                f"{scorer.metadata.get('trained_at')})"
            )

        data[f"{label.lower().replace(' ', '')}_method"] = method

    if STEP10_PREDICTIONS.is_file():

        methods = sorted(
            pd.read_csv(STEP10_PREDICTIONS, usecols=["scoring_method"])["scoring_method"]
            .dropna().astype(str).unique()
        )

        data["step10_prediction_methods"] = methods

        if methods != [scorer.method]:
            problems.append(f"Step 10 predictions hold methods {methods}, expected [{scorer.method}]")

    else:
        problems.append(f"Step 10 predictions not found: {STEP10_PREDICTIONS}")

    if STEP12_OUTPUT.is_file():

        header = pd.read_csv(STEP12_OUTPUT, nrows=0).columns

        if "scoring_method" in header:

            methods = sorted(
                pd.read_csv(STEP12_OUTPUT, usecols=["scoring_method"])["scoring_method"]
                .dropna().astype(str).unique()
            )

            data["step12_output_methods"] = methods

            if methods != [scorer.method]:
                problems.append(f"Step 12 output holds methods {methods}, expected [{scorer.method}]")
        else:
            problems.append("Step 12 output has no scoring_method column - re-run Step 12")

    else:
        problems.append(f"Step 12 output not found: {STEP12_OUTPUT}")

    if train is not None:

        data["step10b_model_saved"] = train.get("model_saved")

        if train.get("model_saved") is True and not file_present:
            problems.append("Step 10b saved a model but the file is absent now")

        data["step10b_target_is_proxy"] = train.get("target_is_proxy")

    if problems:
        return Check("model_decision", "fail", "; ".join(problems), data)

    return Check(
        "model_decision", "pass",
        f"{scorer.method}; {scorer.claim.label}", data,
    )


# ======================================================================
# 2. STEP 10 REPRODUCES FROM THE SCORER
# ======================================================================

def check_step10_reproduces() -> Check:

    if not STEP10_PREDICTIONS.is_file():
        return Check("step10_reproduces", "skip", "Step 10 predictions not found")

    frame = pd.read_csv(STEP10_PREDICTIONS)

    missing = [c for c in mc.MODEL_FEATURES if c not in frame.columns]

    if missing:
        return Check(
            "step10_reproduces", "fail",
            f"Step 10 predictions lack feature columns {missing}, so they "
            "cannot be re-scored with the shared definition",
        )

    scorer = mc.MatchScorer.load()

    again = scorer.score(frame).scores

    diff = float(np.max(np.abs(again - frame["hybrid_match_score"].to_numpy(float))))

    # Step 10's formula column keeps the category term for rows whose
    # category is known, so it is recomputed from the whole row.
    formula = mc.formula_score(frame)

    formula_diff = (
        float(np.max(np.abs(formula - frame["formula_match_score"].to_numpy(float))))
        if "formula_match_score" in frame.columns else None
    )

    data = {"rows": int(len(frame)), "max_abs_score_diff": diff,
            "max_abs_formula_diff": formula_diff}

    if diff > SCORE_TOLERANCE or (formula_diff is not None and formula_diff > SCORE_TOLERANCE):
        return Check(
            "step10_reproduces", "fail",
            f"re-scoring Step 10's rows differs from its stored scores "
            f"(scorer {diff:.3g}, formula {formula_diff}); Step 10 and the "
            "shared scorer disagree", data,
        )

    return Check("step10_reproduces", "pass",
                 f"{len(frame):,} rows re-scored identically (max diff {diff:.2g})", data)


# ======================================================================
# APP + LIVE CORPUS
# ======================================================================

def load_app():
    """The real 14_app.py, with a no-op streamlit and no LLM calls."""

    os.environ["GROQ_API_KEY"] = ""      # explanations stay deterministic

    import streamlit_headless

    streamlit_headless.install()

    return load_module(SRC / "14_app.py", "resume_app_under_test")


def build_live(app):
    """Build the exact cached app corpus using the app's real helpers."""
    jobs = app.load_jobs()
    model = app.load_semantic_model()
    vocab_signature = app.vocabulary_signature()
    vocabulary, error = app.load_vocabulary(vocab_signature)
    scorer = app.load_scorer(app.scorer_signature())
    corpus = app.load_corpus(
        app.get_file_signature(app.JOB_DATA_PATH), vocab_signature
    )

    def encode(texts):
        return model.encode(
            list(texts), batch_size=32, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
        )

    return {
        "jobs": jobs, "model": model, "vocabulary": vocabulary,
        "vocabulary_error": error, "scorer": scorer, "corpus": corpus,
        "encode": encode,
    }


def jaccard(a, b) -> float:

    a, b = set(a), set(b)

    if not a and not b:
        return 1.0

    return len(a & b) / len(a | b)


# ======================================================================
# 3. OFFLINE vs LIVE
# ======================================================================

def check_offline_vs_live(live: Dict[str, Any], resumes: int, seed: int = 42) -> Check:
    """Compare Step 9's offline candidate pool with the live shared ranker."""
    import job_ranker

    if not STEP9_PATH.is_file():
        return Check("offline_vs_live", "skip", "Step 9 dataset not found")

    resume_source = PROCESSED / "resumes_esco_mapped.csv"
    if not resume_source.is_file():
        return Check("offline_vs_live", "skip", "mapped resume table not found")

    step7 = load_module(SRC / "07_build_matching_dataset.py", "step7_for_verify")
    dataset = pd.read_csv(STEP9_PATH, low_memory=False)
    resumes_frame = pd.read_csv(resume_source, low_memory=False)

    def norm(value):
        if pd.isna(value):
            return ""
        value = str(value).strip()
        if value.endswith(".0"):
            try:
                number = float(value)
                if number.is_integer():
                    return str(int(number))
            except (TypeError, ValueError):
                pass
        return value

    resume_id_col = step7.detect_id_column(resumes_frame, "resume")
    text_col = next(
        (name for name in ("resume_text", "resume_text_representation", "text")
         if name in resumes_frame.columns), None
    )
    if text_col is None:
        return Check("offline_vs_live", "fail", "mapped resumes have no resume text column")
    if not {"resume_id", "job_id"}.issubset(dataset.columns):
        return Check("offline_vs_live", "fail", "Step 9 dataset lacks resume_id/job_id columns")

    resumes_frame["__rid"] = resumes_frame[resume_id_col].map(norm)
    resume_text = {
        norm(row["__rid"]): mc.clean_text(row[text_col])
        for _, row in resumes_frame.iterrows()
        if norm(row["__rid"]) and mc.clean_text(row[text_col])
    }
    dataset["__rid"] = dataset["resume_id"].map(norm)
    dataset["__jid"] = dataset["job_id"].map(norm)

    available = sorted(set(dataset["__rid"]) & set(resume_text))
    if not available:
        return Check("offline_vs_live", "fail", "no Step 9 resume IDs have mapped text")

    rng = np.random.default_rng(seed)
    sample = sorted(rng.choice(available, size=min(max(1, resumes), len(available)), replace=False).tolist())
    corpus = live["corpus"]
    scorer = live["scorer"]
    vocabulary = live["vocabulary"]

    candidate_jaccards, pool_sizes, recalls, top1_kept = [], [], [], []
    feature_diffs = {name: [] for name in EXACT_FEATURES}
    skill_diffs = {name: [] for name in SKILL_FEATURES if name in mc.MODEL_FEATURES}
    pairs_compared = 0

    for rid in sample:
        text = resume_text[rid]
        skill_map = job_ranker.resume_skill_uris(text, vocabulary)
        vector = np.asarray(live["encode"]([text]), dtype=np.float32)[0]

        full = job_ranker.rank_resume(
            corpus, resume_text=text, resume_embedding=vector,
            resume_skills=skill_map.keys(), scorer=scorer,
            max_candidates_per_signal=FULL_SCAN_POLICY.max_candidates_per_signal,
        )
        pool = job_ranker.rank_resume(
            corpus, resume_text=text, resume_embedding=vector,
            resume_skills=skill_map.keys(), scorer=scorer,
            max_candidates_per_signal=DEFAULT_POLICY.max_candidates_per_signal,
        )

        full_ids = ids(full["job_id"])
        pool_ids = set(ids(pool["job_id"]))
        offline = dataset[dataset["__rid"] == rid].drop_duplicates("__jid")
        offline_ids = set(offline["__jid"])

        candidate_jaccards.append(jaccard(offline_ids, pool_ids))
        pool_sizes.append(len(pool_ids))
        top10_ids = full_ids[:10]
        recalls.append(len(set(top10_ids) & pool_ids) / max(len(top10_ids), 1))
        top1_kept.append(bool(top10_ids and top10_ids[0] in pool_ids))

        live_rows = full.copy()
        live_rows["__jid"] = live_rows["job_id"].map(norm)
        live_rows = live_rows.drop_duplicates("__jid").set_index("__jid")
        for _, row in offline.iterrows():
            jid = row["__jid"]
            if jid not in live_rows.index:
                continue
            pairs_compared += 1
            for feature in EXACT_FEATURES:
                if feature in row.index and feature in live_rows.columns:
                    feature_diffs[feature].append(abs(float(row[feature]) - float(live_rows.at[jid, feature])))
            for feature in skill_diffs:
                if feature in row.index and feature in live_rows.columns:
                    skill_diffs[feature].append(abs(float(row[feature]) - float(live_rows.at[jid, feature])))

    feature_summary = {
        feature: {
            "max_abs_diff": float(np.max(vals)) if vals else None,
            "mean_abs_diff": float(np.mean(vals)) if vals else None,
        }
        for feature, vals in {**feature_diffs, **skill_diffs}.items()
    }

    # Compare the ESCO URIs preserved by Step 6 against the skill sets the
    # online ranker constructs from the same raw jobs + job text. This makes
    # it visible when text-derived ESCO skills are lost between steps.
    mapped_jobs_path = PROCESSED / "jobs_esco_mapped.csv"
    job_skill_summary = {
        "jobs_identical_skill_set": 0,
        "jobs_compared": 0,
        "mean_jaccard": None,
    }
    if mapped_jobs_path.is_file() and "job_id" in corpus.frame.columns:
        mapped_jobs = pd.read_csv(mapped_jobs_path, low_memory=False)
        if {"job_id", "esco_mapped_skill_uris"}.issubset(mapped_jobs.columns):
            def parse_uri_set(value):
                if pd.isna(value):
                    return set()
                try:
                    parsed = json.loads(value) if isinstance(value, str) else value
                    if isinstance(parsed, (list, tuple, set)):
                        return {str(item).strip() for item in parsed if str(item).strip()}
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
                return set()

            def norm_job_id(value):
                return norm(value)

            mapped_lookup = {}
            for _, mapped_row in mapped_jobs.iterrows():
                mapped_lookup[norm_job_id(mapped_row["job_id"])] = parse_uri_set(
                    mapped_row["esco_mapped_skill_uris"]
                )
            live_job_id_col = job_ranker.detect_job_columns(corpus.frame)["id"]
            comparisons = []
            for position, raw_id in enumerate(corpus.frame[live_job_id_col]):
                jid = norm_job_id(raw_id)
                if jid not in mapped_lookup or position >= len(corpus.skill_sets):
                    continue
                mapped_set = mapped_lookup[jid]
                live_set = set(corpus.skill_sets[position])
                comparisons.append((mapped_set, live_set))
            if comparisons:
                job_skill_summary = {
                    "jobs_identical_skill_set": int(sum(a == b for a, b in comparisons)),
                    "jobs_compared": int(len(comparisons)),
                    "mean_jaccard": float(np.mean([jaccard(a, b) for a, b in comparisons])),
                }

    data = {
        "job_skill_sets": job_skill_summary,
        "resumes_sampled": len(sample),
        "pairs_compared": pairs_compared,
        "features": feature_summary,
        "candidate_pool": {
            "mean_jaccard_offline_vs_live": float(np.mean(candidate_jaccards)),
            "policy": DEFAULT_POLICY.to_dict(),
            "policy_fingerprint": DEFAULT_POLICY.fingerprint,
            "mean_live_pool_size": float(np.mean(pool_sizes)),
            "jobs_in_corpus": int(len(corpus)),
        },
        "full_scan_top10_kept_by_pool": {
            "mean_recall": float(np.mean(recalls)),
            "top1_kept_share": float(np.mean(top1_kept)),
        },
        "interpretation": (
            "Candidate-pool differences are measured rather than treated as a hard failure, "
            "because offline ESCO skill mapping and live text/structured skill extraction can differ."
        ),
    }

    hard = [
        f"{feature}: max diff {feature_summary[feature]['max_abs_diff']:.3g}"
        for feature in EXACT_FEATURES
        if feature_summary[feature]["max_abs_diff"] is not None
        and feature_summary[feature]["max_abs_diff"] > EXACT_TOLERANCE
    ]
    if pairs_compared == 0:
        return Check("offline_vs_live", "fail", "no offline/live resume-job pairs overlap", data)
    if hard:
        return Check("offline_vs_live", "fail", "shared text signals differ: " + "; ".join(hard), data)

    warnings = []
    pool_overlap = float(np.mean(candidate_jaccards))
    top10_recall = float(np.mean(recalls))
    if pool_overlap < WARN_MIN_CANDIDATE_JACCARD:
        warnings.append(f"offline/live candidate pool overlap is {pool_overlap:.0%}")
    if top10_recall < WARN_MIN_TOP10_RECALL:
        warnings.append(f"candidate pool retains {top10_recall:.0%} of full-scan top-10 jobs")
    skill_means = [v["mean_abs_diff"] for key, v in feature_summary.items()
                   if key in skill_diffs and v["mean_abs_diff"] is not None]
    if skill_means and max(skill_means) > WARN_MEAN_SKILL_DIFF:
        warnings.append(f"skill feature drift max mean diff {max(skill_means):.3f}")
    if warnings:
        return Check("offline_vs_live", "warn", "; ".join(warnings), data)
    return Check("offline_vs_live", "pass", f"{pairs_compared} pairs checked; key text signals align", data)


# ======================================================================
# 4. CLI vs ranker vs app
# ======================================================================

def check_cli_and_app(app, live: Dict[str, Any], pdf: Path, run_cli: bool = True) -> Check:
    """Confirm the Streamlit app, shared ranker and CLI use the same scorer/ranking."""
    import job_ranker

    if not pdf.is_file():
        return Check("cli_and_app", "skip", f"resume PDF not found: {pdf}")

    pdf_bytes = pdf.read_bytes()
    app_result = app.analyze_resume(pdf_bytes, 5)
    ranked_app = app_result["all_recommendations"]
    extraction = app.read_resume(pdf_bytes)
    text = extraction.text
    skill_map = job_ranker.resume_skill_uris(text, live["vocabulary"])
    vector = np.asarray(live["encode"]([text]), dtype=np.float32)[0]
    direct = job_ranker.rank_resume(
        live["corpus"], resume_text=text, resume_embedding=vector,
        resume_skills=skill_map.keys(), scorer=live["scorer"],
        max_candidates_per_signal=DEFAULT_POLICY.max_candidates_per_signal,
    )

    problems: List[str] = []
    app_ids = ids(ranked_app["job_id"])
    direct_ids = ids(direct["job_id"])
    if app_ids != direct_ids:
        problems.append("the app's ranking differs from job_ranker.rank_resume")
    if not np.allclose(
        ranked_app["hybrid_match_score"].to_numpy(float),
        direct["hybrid_match_score"].to_numpy(float), atol=SCORE_TOLERANCE,
    ):
        problems.append("the app's scores differ from job_ranker.rank_resume")
    if app_result.get("candidate_policy_fingerprint") != DEFAULT_POLICY.fingerprint:
        problems.append("the app reports a different candidate retrieval policy")

    data: Dict[str, Any] = {
        "resume_pdf": str(pdf),
        "candidates_scored": int(len(direct)),
        "app_scoring_method": app_result["scoring_method"],
        "app_claim": app_result.get("claim", {}).get("level"),
        "top_job_ids": app_ids[:5],
    }
    if app_result["scoring_method"] != live["scorer"].method:
        problems.append("the app reports a different scoring method than its scorer")

    cli_note = "CLI not run"
    if run_cli:
        before = CLI_ANALYSIS.stat().st_mtime_ns if CLI_ANALYSIS.is_file() else None
        done = subprocess.run(
            [sys.executable, str(SRC / "13_explainable_rag_recommendation.py"), str(pdf)],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=1800,
            env={**os.environ, "GROQ_API_KEY": ""},
        )
        analysis = read_json(CLI_ANALYSIS)
        if done.returncode != 0 or analysis is None or (
            before is not None and CLI_ANALYSIS.stat().st_mtime_ns == before
        ):
            problems.append(
                f"the CLI did not produce a fresh analysis (exit {done.returncode}): "
                f"{(done.stderr or done.stdout)[-300:]}"
            )
        else:
            matching = analysis.get("matching", {})
            top = analysis.get("recommendations", {}).get("top_jobs", [])
            cli_ids = ids(job.get("job_id") for job in top)
            data["cli_top_job_ids"] = cli_ids
            if cli_ids != direct_ids[:len(cli_ids)]:
                problems.append("the CLI's top jobs differ from job_ranker.rank_resume")
            if matching.get("scoring_method") != live["scorer"].method:
                problems.append("the CLI used a different scorer than the app")
            if matching.get("candidate_policy_fingerprint") != DEFAULT_POLICY.fingerprint:
                problems.append("the CLI used a different candidate policy")
            if matching.get("candidates_scored") != len(direct):
                problems.append(
                    f"the CLI scored {matching.get('candidates_scored')} candidates, "
                    f"the app {len(direct)}"
                )
            if matching.get("claim", {}).get("level") != app_result.get("claim", {}).get("level"):
                problems.append("the CLI and the app make different claims about the score")
            cli_note = "CLI matches"

    if problems:
        return Check("cli_and_app", "fail", "; ".join(problems), data)
    return Check("cli_and_app", "pass", f"app, ranker and {cli_note.lower()}", data)


# ======================================================================
# MAIN
# ======================================================================

def main(argv: Optional[List[str]] = None) -> int:

    parser = argparse.ArgumentParser(description=__doc__.split("USAGE")[0])
    parser.add_argument("--resumes", type=int, default=10)
    parser.add_argument("--resume-pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--expect-model", choices=("present", "absent", "any"), default="any")
    parser.add_argument("--skip-cli", action="store_true")
    parser.add_argument("--skip-live", action="store_true",
                        help="only the file-level checks (no embeddings are computed)")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)

    args = parser.parse_args(argv)

    checks = [check_model_decision(args.expect_model), check_step10_reproduces()]

    if args.skip_live:

        checks.append(Check("offline_vs_live", "skip", "--skip-live"))
        checks.append(Check("cli_and_app", "skip", "--skip-live"))

    else:

        app = load_app()

        live = build_live(app)

        checks.append(check_offline_vs_live(live, args.resumes))
        checks.append(check_cli_and_app(app, live, args.resume_pdf, run_cli=not args.skip_cli))

    scorer = mc.MatchScorer.load()

    report = {
        "scorer": scorer.description,
        "claim": scorer.claim.to_dict(),
        "candidate_policy": DEFAULT_POLICY.to_dict(),
        "feature_definition": list(mc.MODEL_FEATURES),
        "overall": (
            "fail" if any(c.status == "fail" for c in checks)
            else "warn" if any(c.status == "warn" for c in checks)
            else "pass"
        ),
        "checks": [c.to_dict() for c in checks],
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    for check in checks:
        print(f"[{check.status.upper():4s}] {check.name}: {check.detail}")

    print(f"\nOverall: {report['overall']}   Report: {args.report}")

    return 1 if report["overall"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
