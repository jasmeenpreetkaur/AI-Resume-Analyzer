"""Run the AI Resume Analyzer pipeline in its supported step order.

Examples (from the project root):
    python src/run_pipeline.py --dry-run
    python src/run_pipeline.py
    python src/run_pipeline.py --with-model
    python src/run_pipeline.py --with-model --with-final-recommendations

The standard run requires the original datasets under data/. Model training is
optional; without it Step 10 and the app use the transparent formula scorer.
Suitability wording remains locked until 11b's independent human-judgment gate
is explicitly approved for the active scorer fingerprint.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

PIPELINE_STEPS = [
    ("02_clean_datasets.py", "Clean source datasets"),
    ("03_resume_parser.py", "Parse resume PDFs"),
    ("04_prepare_text_data.py", "Prepare resume/job text"),
    ("05_extract_skills.py", "Extract skills"),
    ("06_map_skills_to_esco.py", "Map skills to ESCO"),
    ("07_build_matching_dataset.py", "Retrieve resume-job candidate pairs"),
    ("08_build_match_features.py", "Build matching features"),
    ("09_create_training_labels.py", "Create proxy training labels"),
]

REQUIRED_SOURCE_FILES = [
    Path("data/resume_dataset/Resume/Resume.csv"),
    Path("data/jobs/all_job_post.csv"),
    Path("data/external/skills_en.csv"),
]


def build_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    commands = [(label, [sys.executable, str(SRC_DIR / script)]) for script, label in PIPELINE_STEPS]
    if args.with_model:
        command = [sys.executable, str(SRC_DIR / "10b_train_match_model.py")]
        if args.force_model:
            command.append("--force")
        commands.append(("Train optional proxy model", command))
    commands.append(("Score/rank resume-job pairs (Step 10)", [sys.executable, str(SRC_DIR / "10_prepare_training_data.py")]))
    if args.with_final_recommendations:
        commands.append(("Build final recommendation tables (Step 12)", [sys.executable, str(SRC_DIR / "12_final_recommendation_pipeline.py")]))
    return commands


def check_sources() -> list[Path]:
    return [path for path in REQUIRED_SOURCE_FILES if not (PROJECT_ROOT / path).is_file()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-model", action="store_true", help="Run Step 10b model training before Step 10")
    parser.add_argument("--force-model", action="store_true", help="Pass --force to Step 10b (explicitly bypasses its model-quality gate)")
    parser.add_argument("--with-final-recommendations", action="store_true", help="Run Step 12 after Step 10")
    parser.add_argument("--dry-run", action="store_true", help="Print the command sequence without running it")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip checks for the three core source datasets")
    args = parser.parse_args(argv)

    if args.force_model and not args.with_model:
        parser.error("--force-model requires --with-model")

    commands = build_commands(args)
    print("AI RESUME ANALYZER — PIPELINE RUNNER")
    print(f"Project root: {PROJECT_ROOT}")
    print("Suitability claim gate: human-review approval required")

    if not args.skip_preflight:
        missing = check_sources()
        if missing and not args.dry_run:
            display = "\n".join(f"  - {PROJECT_ROOT / path}" for path in missing)
            print("Preflight failed; required source file(s) are missing:\n" + display, file=sys.stderr)
            print("Add the datasets to their expected data/ paths or run this from the correct project copy.", file=sys.stderr)
            return 2

    for index, (label, command) in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {label}")
        print("  " + " ".join(f'"{part}"' if " " in part else part for part in command))
        if args.dry_run:
            continue
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        try:
            subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Pipeline stopped at '{label}' (exit code {exc.returncode}).", file=sys.stderr)
            return int(exc.returncode or 1)

    if args.dry_run:
        print("\nDry run complete; no steps were executed.")
    else:
        print("\nPipeline steps completed.")
        print("Human-review workflow (optional, but required for suitability wording):")
        print("  python src/11b_evaluate_human_judgments.py --export data/evaluation/step10_matching_predictions.csv")
        print("  # collect independent 0/1/2 labels, then run --evaluate with explicit sign-off")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
