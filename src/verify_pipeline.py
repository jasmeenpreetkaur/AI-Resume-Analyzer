"""Source, test-suite, and real-data verification for AI Resume Analyzer.

Usage:
    python src/verify_pipeline.py --mode source
    python src/verify_pipeline.py --mode tests
    python src/verify_pipeline.py --mode deployment

`deployment` validates local source inputs and the materialized output schemas;
it does not claim that the recommendations themselves are accurate. Accuracy
requires human judgments (Step 11b) and live input datasets.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"

REQUIRED_INPUTS = [
    Path("data/resume_dataset/Resume/Resume.csv"),
    Path("data/jobs/all_job_post.csv"),
    Path("data/external/skills_en.csv"),
]
REQUIRED_OUTPUTS = {
    Path("data/processed/resume_job_matching.csv"): {"resume_id", "job_id", "candidate_retrieval_sources"},
    Path("data/processed/resume_job_features.csv"): {"resume_id", "job_id"},
    Path("data/processed/resume_job_matching_dataset.csv"): {"resume_id", "job_id"},
    Path("data/evaluation/step10_matching_predictions.csv"): {"resume_id", "job_id", "hybrid_match_score", "scorer_fingerprint", "claim_label"},
}


def check_source() -> list[str]:
    errors: list[str] = []
    source_files = sorted(SRC_DIR.glob("*.py"))
    if not source_files:
        errors.append(f"No Python source files found in {SRC_DIR}")
    for path in source_files + sorted(TESTS_DIR.glob("*.py")):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (OSError, SyntaxError) as exc:
            errors.append(f"Compile error in {path.relative_to(PROJECT_ROOT)}: {exc}")
    for expected in (SRC_DIR / "candidate_retrieval.py", SRC_DIR / "11b_evaluate_human_judgments.py", SRC_DIR / "run_pipeline.py"):
        if not expected.is_file():
            errors.append(f"Required module is missing: {expected.relative_to(PROJECT_ROOT)}")
    return errors


def check_deployment() -> list[str]:
    errors: list[str] = []
    missing = [path for path in REQUIRED_INPUTS if not (PROJECT_ROOT / path).is_file()]
    if missing:
        errors.append("Missing source dataset(s): " + ", ".join(str(p) for p in missing))
    outputs_checked = 0
    for relative, required_columns in REQUIRED_OUTPUTS.items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            errors.append(f"Missing pipeline output: {relative} (run the pipeline first)")
            continue
        try:
            import pandas as pd
            columns = set(pd.read_csv(path, nrows=0).columns)
        except Exception as exc:
            errors.append(f"Cannot read output schema for {relative}: {exc}")
            continue
        absent = sorted(required_columns - columns)
        if absent:
            errors.append(f"Output {relative} missing columns: {absent}")
        else:
            outputs_checked += 1
    prediction_path = PROJECT_ROOT / "data/evaluation/step10_matching_predictions.csv"
    if prediction_path.is_file():
        import pandas as pd
        try:
            frame = pd.read_csv(prediction_path, usecols=["scorer_fingerprint", "claim_label"])
            fingerprints = {str(value).strip() for value in frame["scorer_fingerprint"].dropna() if str(value).strip()}
            labels = set(frame["claim_label"].dropna().astype(str))
            if len(fingerprints) != 1:
                errors.append("Step 10 predictions must contain exactly one non-empty scorer_fingerprint")
            if not labels or not labels.issubset({"experimental_similarity_not_validated", "human_validated_relevance"}):
                errors.append(f"Unexpected claim_label values in Step 10 predictions: {sorted(labels)}")
            from matching_core import MatchScorer
            current = MatchScorer.load().scorer_fingerprint
            if fingerprints != {current}:
                errors.append("Step 10 scorer_fingerprint is stale; rerun Step 10 before deployment")
        except Exception as exc:
            errors.append(f"Cannot validate Step 10 claim metadata: {exc}")
    if not errors:
        print(f"Deployment artifact checks passed ({outputs_checked} output schemas checked).")
        print("This verifies file integrity/schema only, not recommendation accuracy or fairness.")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("source", "tests", "deployment"), default="source")
    parser.add_argument("--quiet", action="store_true", help="Reduce source-check success output")
    args = parser.parse_args(argv)

    if args.mode == "tests":
        return subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT).returncode

    errors = check_source()
    if args.mode == "deployment":
        errors.extend(check_deployment())
    if errors:
        print("VERIFICATION FAILED", file=sys.stderr)
        for error in errors:
            print(f" - {error}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"Source verification passed: compiled {len(list(SRC_DIR.glob('*.py')))} source modules and test modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
