"""
======================================================================
AI RESUME ANALYZER - FULL PIPELINE RUNNER (real deployment path)
======================================================================

Runs the numbered steps in the only order that gives a consistent result:

    2 3 4 5 6 7 8 9        data -> candidate pairs -> features -> Step 9 dataset
    model decision         --model train : Step 10b trains on THIS run's
                                           Step 9 data (an older model file is
                                           set aside first, never reused)
                           --model omit  : any model file is set aside; every
                                           caller uses the transparent formula
    10 11 12               regenerated AFTER the decision, so predictions,
                           validation and recommendations all describe the
                           scorer that is actually installed
    verify                 verify_deployment.py (unless --skip-verify)

Nothing is deleted: a model file that is in the way is renamed to
``<name>.superseded-<timestamp>`` / ``.omitted-<timestamp>``.

It chooses the model decision for you ONLY if you say so: ``--model`` is
required, so "train" or "omit" is always a deliberate choice that is
recorded in reports/pipeline_run_report.json.

USAGE
-----
python src/run_pipeline.py [--model auto|train|omit] [--dry-run]
python src/run_pipeline.py --model train [--force-model] [--require-model]
python src/run_pipeline.py --model omit
Options: --from-step N   start at step N (2-9; earlier outputs are reused)
         --skip-verify   --resume-pdf PDF   --resumes N
======================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matching_core as mc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
REPORTS = PROJECT_ROOT / "reports"
LOG_DIR = REPORTS / "pipeline_run"
RUN_REPORT = REPORTS / "pipeline_run_report.json"
TRAIN_REPORT = REPORTS / "match_model_report.json"

DATA_STEPS = [
    (2, "02_clean_datasets.py"),
    (3, "03_resume_parser.py"),
    (4, "04_prepare_text_data.py"),
    (5, "05_extract_skills.py"),
    (6, "06_map_skills_to_esco.py"),
    (7, "07_build_matching_dataset.py"),
    (8, "08_build_match_features.py"),
    (9, "09_create_training_labels.py"),
]

REGENERATED_STEPS = [
    (10, "10_prepare_training_data.py"),
    (11, "11_human_validation.py"),
    (12, "12_final_recommendation_pipeline.py"),
]


def model_path() -> Path:
    return Path(os.getenv("MATCH_MODEL_PATH") or mc.DEFAULT_MODEL_PATH)


def set_aside(path: Path, reason: str) -> Optional[str]:
    """Rename a model file out of the way (reversible). Returns the new name."""

    if not path.is_file():
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    target = path.with_name(f"{path.name}.{reason}-{stamp}")

    path.rename(target)

    return str(target)


def run_step(label: str, script: str, *args: str) -> Dict[str, Any]:

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log = LOG_DIR / f"{Path(script).stem}.log"

    started = time.time()

    print(f"\n>>> {label}: {script} {' '.join(args)}".rstrip(), flush=True)

    done = subprocess.run(
        [sys.executable, str(SRC / script), *args],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )

    log.write_text(
        f"$ {script} {' '.join(args)}\n\n--- stdout ---\n{done.stdout}\n"
        f"--- stderr ---\n{done.stderr}\n",
        encoding="utf-8",
    )

    seconds = round(time.time() - started, 1)

    print(f"    exit {done.returncode} in {seconds}s  (log: {log})", flush=True)

    if done.returncode not in (0,):
        print((done.stderr or done.stdout)[-1500:])

    return {"step": label, "script": script, "args": list(args),
            "returncode": done.returncode, "seconds": seconds, "log": str(log)}


def main(argv: Optional[List[str]] = None) -> int:

    parser = argparse.ArgumentParser(description=__doc__.split("USAGE")[0])
    parser.add_argument("--model", choices=("auto", "train", "omit"), default=None,
                        help="required decision: auto reuses an existing model (or formula if absent); train/omit are deliberate")
    parser.add_argument("--with-model", action="store_true",
                        help="legacy alias for --model train")
    parser.add_argument("--with-final-recommendations", action="store_true",
                        help="legacy compatibility flag; a full run always generates Step 12 recommendations")
    parser.add_argument("--verify-deployment", action="store_true",
                        help="explicitly request deployment verification (the default unless --skip-verify is used)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the planned command sequence without executing steps")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="skip checks for the three core source datasets")
    parser.add_argument("--force-model", action="store_true",
                        help="pass --force to Step 10b (save even if it does not beat the formula)")
    parser.add_argument("--require-model", action="store_true",
                        help="with --model train: fail if no model ends up installed")
    parser.add_argument("--from-step", type=int, default=2, choices=range(2, 10))
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--resume-pdf", type=Path, default=None)
    parser.add_argument("--resumes", type=int, default=10)

    args = parser.parse_args(argv)

    if args.with_model:
        if args.model not in (None, "train"):
            parser.error("--with-model is an alias for --model train and cannot be combined with another model decision")
        args.model = "train"
    if args.model is None:
        parser.error("--model is required (choose auto, train, or omit); --with-model is a legacy alias for train")
    if args.verify_deployment and args.skip_verify:
        parser.error("--verify-deployment cannot be combined with --skip-verify")

    if args.force_model and args.model != "train":
        parser.error("--force-model requires --model train")
    if args.require_model and args.model != "train":
        parser.error("--require-model requires --model train")

    required_sources = [
        PROJECT_ROOT / "data" / "resume_dataset" / "Resume" / "Resume.csv",
        PROJECT_ROOT / "data" / "jobs" / "all_job_post.csv",
        PROJECT_ROOT / "data" / "external" / "skills_en.csv",
    ]
    missing = [path for path in required_sources if not path.is_file()]
    if missing and not args.skip_preflight and not args.dry_run:
        print("Preflight failed; required source files are missing:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        print("Add the datasets to data/ or use --skip-preflight only when you know the step inputs are already present.", file=sys.stderr)
        return 2

    if args.dry_run:
        print("AI Resume Analyzer pipeline plan")
        print(f"Project root: {PROJECT_ROOT}")
        print("Data steps:")
        for number, script in DATA_STEPS:
            if number >= args.from_step:
                print(f"  Step {number}: {script}")
        print(f"Model decision: {args.model}")
        if args.model == "train":
            print("  Step 10b: train model on this run's Step 9 dataset")
        elif args.model == "omit":
            print("  Move any installed model aside and use the formula scorer")
        else:
            print("  Reuse an installed model if present; otherwise use the formula scorer")
        for number, script in REGENERATED_STEPS:
            print(f"  Step {number}: {script}")
        if not args.skip_verify or args.verify_deployment:
            print("  Verify: verify_deployment.py")
        if args.with_final_recommendations:
            print("  Compatibility flag accepted; Step 12 already runs in every full pipeline")
        print("Dry run complete; no steps were executed and no model files were changed.")
        return 0

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    steps: List[Dict[str, Any]] = []

    report: Dict[str, Any] = {
        "started_at": started,
        "model_decision_requested": args.model,
        "steps": steps,
    }

    def finish(code: int, status: str) -> int:

        report["status"] = status
        report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        RUN_REPORT.parent.mkdir(parents=True, exist_ok=True)
        RUN_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"\nPipeline {status}. Report: {RUN_REPORT}")

        return code

    # ---- data steps -----------------------------------------------------
    for number, script in DATA_STEPS:

        if number < args.from_step:
            continue

        result = run_step(f"Step {number}", script)
        steps.append(result)

        if result["returncode"] != 0:
            return finish(1, f"failed_at_step_{number}")

    # ---- the model decision ---------------------------------------------
    path = model_path()

    decision: Dict[str, Any] = {"requested": args.model, "model_file": str(path)}

    report["model_decision"] = decision

    if args.model == "omit":

        moved = set_aside(path, "omitted")
        decision.update({"model_installed": False, "existing_model_moved_to": moved,
                         "outcome": "omitted_deliberately"})
        print(f"\n>>> Model: omitted deliberately"
              + (f" (existing file moved to {moved})" if moved else ""))

    elif args.model == "auto" and path.is_file():

        decision.update({"model_installed": True, "outcome": "reused_existing_model"})
        print("\n>>> Model: reusing the existing model (explicit --model train is needed to retrain)")

    elif args.model == "auto":

        decision.update({"model_installed": False, "outcome": "formula_no_existing_model"})
        print("\n>>> Model: no installed model found; using the transparent formula scorer")

    else:

        # Never let a model trained on older data survive into a requested
        # training run. Preserve it with a timestamped name instead of delete.
        moved = set_aside(path, "superseded")
        decision["previous_model_moved_to"] = moved

        extra = ["--force"] if args.force_model else []
        result = run_step("Step 10b", "10b_train_match_model.py", *extra)
        steps.append(result)

        trained = None
        try:
            trained = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
        except Exception:
            pass

        installed = path.is_file()
        decision.update({
            "model_installed": installed,
            "step10b_returncode": result["returncode"],
            "step10b_status": (trained or {}).get("status"),
            "not_saved_because": (trained or {}).get("not_saved_because")
            or (trained or {}).get("reason"),
            "outcome": "trained_and_installed" if installed
            else "trained_but_formula_in_use",
        })
        print(f"\n>>> Model: {decision['outcome']}")
        if not installed and decision["not_saved_because"]:
            print(f"    reason: {decision['not_saved_because']}")
        if not installed and args.require_model:
            return finish(1, "model_required_but_not_installed")

    scorer = mc.MatchScorer.load()

    decision["scoring_method"] = scorer.method
    decision["claim"] = scorer.claim.to_dict()
    decision["model_trained_at"] = scorer.metadata.get("trained_at") if scorer.uses_model else None

    # ---- Steps 10-12 regenerated for that decision ------------------------
    for number, script in REGENERATED_STEPS:

        result = run_step(f"Step {number}", script)
        steps.append(result)

        if result["returncode"] != 0:
            return finish(1, f"failed_at_step_{number}")

    # ---- verification -----------------------------------------------------
    if not args.skip_verify or args.verify_deployment:

        verify_args = ["--expect-model", "present" if scorer.uses_model else "absent",
                       "--resumes", str(args.resumes)]

        if args.resume_pdf:
            verify_args += ["--resume-pdf", str(args.resume_pdf)]

        result = run_step("Verify", "verify_deployment.py", *verify_args)
        steps.append(result)

        try:
            verification = json.loads(
                (REPORTS / "deployment_verification_report.json").read_text(encoding="utf-8")
            )
            report["verification"] = {
                "overall": verification["overall"],
                "checks": {c["name"]: [c["status"], c["detail"]] for c in verification["checks"]},
            }
        except Exception:
            report["verification"] = {"overall": "not_available"}

        if result["returncode"] != 0:
            return finish(1, "verification_failed")

    return finish(0, "completed")


if __name__ == "__main__":
    sys.exit(main())
