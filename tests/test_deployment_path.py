"""
End-to-end tests of the REAL deployment path (src/run_pipeline.py +
src/verify_deployment.py) on a synthetic project, for both model
decisions:

  * model INSTALLED  - trained by Step 10b, then Steps 10-12 regenerated
  * model OMITTED    - formula everywhere

and the failure that matters in practice: a model that changes after
Steps 10-12 ran must be refused, not silently mixed with old outputs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matching_core as mc
import synthetic_project

TESTS = Path(__file__).resolve().parent
SRC = TESTS.parent / "src"


def run(root: Path, script: str, *args: str, env_extra=None):

    env = dict(os.environ)
    env["PYTHONPATH"] = str(TESTS / "stubs")
    env.pop("MATCH_MODEL_PATH", None)
    env.pop("HUMAN_REPORT_PATH", None)
    env.update(env_extra or {})

    return subprocess.run(
        [sys.executable, str(root / "src" / script), *args],
        cwd=root, env=env, capture_output=True, text=True, timeout=900,
    )


def make_project(tmp_path_factory, name: str, jobs_per_category: int) -> Path:

    root = tmp_path_factory.mktemp(name)

    shutil.copytree(SRC, root / "src", ignore=shutil.ignore_patterns("__pycache__"))

    original = synthetic_project.JOBS_PER_CATEGORY

    synthetic_project.JOBS_PER_CATEGORY = jobs_per_category

    try:
        synthetic_project.build_project(root)
    finally:
        synthetic_project.JOBS_PER_CATEGORY = original

    return root


def read(root: Path, relative: str):
    return json.loads((root / relative).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Small corpus (pool == corpus, so the proxy label is usable)."""

    root = make_project(tmp_path_factory, "deploy_trained", 10)

    done = run(root, "run_pipeline.py", "--model", "train", "--force-model",
               "--require-model", "--resumes", "6")

    return root, done


@pytest.fixture(scope="module")
def omitted(tmp_path_factory):
    """200 jobs, so the candidate pool is a strict subset of the corpus."""

    root = make_project(tmp_path_factory, "deploy_omitted", 50)

    done = run(root, "run_pipeline.py", "--model", "omit", "--resumes", "6")

    return root, done


def assert_ok(done):

    assert done.returncode == 0, (
        f"stdout:\n{done.stdout[-3000:]}\nstderr:\n{done.stderr[-2000:]}"
    )


# ----------------------------------------------------------------------
# model installed
# ----------------------------------------------------------------------

def test_trained_model_pipeline_completes_and_verifies(trained):

    root, done = trained

    assert_ok(done)

    run_report = read(root, "reports/pipeline_run_report.json")

    assert run_report["status"] == "completed"
    assert run_report["model_decision"]["outcome"] == "trained_and_installed"
    assert run_report["verification"]["overall"] in ("pass", "warn")

    verification = read(root, "reports/deployment_verification_report.json")

    assert verification["overall"] != "fail"

    assert {c["name"]: c["status"] for c in verification["checks"]}[
        "model_decision"
    ] == "pass"


def test_steps_10_to_12_were_regenerated_after_training(trained):

    root, done = trained

    assert_ok(done)

    trained_at = mc.MatchScorer.load(root / "models" / "match_regressor.joblib").metadata[
        "trained_at"
    ]

    for report, key in (
        ("reports/hybrid_matching_model_report.json", "scoring"),
        ("reports/final_recommendation_report.json", "scoring"),
    ):

        scoring = read(root, report)[key]

        assert scoring["scoring_method"] == mc.METHOD_MODEL
        assert scoring["model_trained_at"] == trained_at
        assert scoring["claim"]["validated_against_human_judgments"] is False
        assert scoring["claim"]["level"] == mc.CLAIM_MODEL_PROXY

    predictions = pd.read_csv(root / "data/evaluation/step10_matching_predictions.csv")

    assert set(predictions["scoring_method"]) == {mc.METHOD_MODEL}

    final = pd.read_csv(root / "data/recommendations/final_job_recommendations.csv")

    assert set(final["scoring_method"]) == {mc.METHOD_MODEL}


def test_model_bundle_and_training_report_say_it_is_a_proxy(trained):

    root, done = trained

    assert_ok(done)

    bundle = mc.MatchScorer.load(root / "models" / "match_regressor.joblib")

    assert bundle.metadata["target_is_proxy"] is True
    assert bundle.metadata["human_validated"] is False
    assert list(bundle.feature_names) == list(mc.MODEL_FEATURES)

    report = read(root, "reports/match_model_report.json")

    assert report["target_is_proxy"] is True
    assert report["claim"]["validated_against_human_judgments"] is False
    assert "NOT validated" in bundle.description


def test_cli_and_app_agree_when_a_model_is_installed(trained):

    root, done = trained

    assert_ok(done)

    checks = {
        c["name"]: c for c in read(root, "reports/deployment_verification_report.json")["checks"]
    }

    assert checks["cli_and_app"]["status"] == "pass"
    assert checks["step10_reproduces"]["status"] == "pass"

    analysis = read(root, "data/recommendations/test_resume_analysis.json")

    assert analysis["matching"]["scoring_method"] == mc.METHOD_MODEL
    assert analysis["matching"]["claim"]["validated_against_human_judgments"] is False
    assert analysis["matching"]["candidate_policy_fingerprint"]


def test_a_model_retrained_after_step12_is_refused(trained):
    """Mixing a new model with old Step 10-12 outputs must not pass silently."""

    root, done = trained

    assert_ok(done)

    path = root / "models" / "match_regressor.joblib"

    original = path.read_bytes()

    try:

        old = mc.MatchScorer.load(path)

        import time

        time.sleep(1.1)       # trained_at has one-second resolution

        mc.save_model_bundle(
            path, old.estimator, mc.MODEL_FEATURES,
            target=old.metadata.get("target"),
            target_is_proxy=True, human_validated=False,
        )

        # Step 12 refuses to describe scores another model produced ...
        stale = run(root, "12_final_recommendation_pipeline.py")

        assert stale.returncode != 0
        assert "retrained" in (stale.stderr + stale.stdout)

        # ... and the verifier reports it.
        verify = run(root, "verify_deployment.py", "--skip-live")

        assert verify.returncode == 1
        assert "model_decision" in verify.stdout and "FAIL" in verify.stdout

    finally:

        path.write_bytes(original)


def test_verifier_flags_a_missing_model_when_one_was_expected(trained):

    root, done = trained

    assert_ok(done)

    path = root / "models" / "match_regressor.joblib"

    hidden = path.with_suffix(".hidden")

    path.rename(hidden)

    try:

        verify = run(root, "verify_deployment.py", "--skip-live", "--expect-model", "present")

        assert verify.returncode == 1

    finally:

        hidden.rename(path)


# ----------------------------------------------------------------------
# model omitted
# ----------------------------------------------------------------------

def test_omitted_model_pipeline_uses_the_formula_everywhere(omitted):

    root, done = omitted

    assert_ok(done)

    assert not (root / "models" / "match_regressor.joblib").exists()

    run_report = read(root, "reports/pipeline_run_report.json")

    assert run_report["model_decision"]["outcome"] == "omitted_deliberately"

    for report in ("reports/hybrid_matching_model_report.json",
                   "reports/final_recommendation_report.json"):

        scoring = read(root, report)["scoring"]

        assert scoring["scoring_method"] == mc.METHOD_FORMULA
        assert scoring["claim"]["level"] == mc.CLAIM_FORMULA
        assert scoring["claim"]["validated_against_human_judgments"] is False

    analysis = read(root, "data/recommendations/test_resume_analysis.json")

    assert analysis["matching"]["scoring_method"] == mc.METHOD_FORMULA


def test_an_existing_model_is_set_aside_not_deleted_when_omitting(tmp_path_factory):

    root = make_project(tmp_path_factory, "deploy_omit_existing", 10)

    # a model left over from some earlier run
    from sklearn.dummy import DummyRegressor

    X = pd.DataFrame(np.random.default_rng(0).random((20, 9)), columns=mc.MODEL_FEATURES)

    stray = DummyRegressor().fit(X, np.zeros(20))

    mc.save_model_bundle(root / "models" / "match_regressor.joblib", stray, mc.MODEL_FEATURES)

    assert_ok(run(root, "run_pipeline.py", "--model", "omit", "--skip-verify"))

    assert not (root / "models" / "match_regressor.joblib").exists()

    assert list((root / "models").glob("match_regressor.joblib.omitted-*"))

    scoring = read(root, "reports/hybrid_matching_model_report.json")["scoring"]

    assert scoring["scoring_method"] == mc.METHOD_FORMULA


def test_offline_and_live_paths_agree_on_features_skills_and_pool(omitted):

    root, done = omitted

    assert_ok(done)

    checks = {
        c["name"]: c for c in read(root, "reports/deployment_verification_report.json")["checks"]
    }

    check = checks["offline_vs_live"]

    assert check["status"] == "pass", check["detail"]

    data = check["data"]

    # job skills: Step 5's text extraction is no longer dropped by Step 6
    assert data["job_skill_sets"]["jobs_identical_skill_set"] == data["job_skill_sets"]["jobs_compared"]

    # the policy is actually exercised: the pool is a strict subset
    assert data["candidate_pool"]["mean_live_pool_size"] < data["candidate_pool"]["jobs_in_corpus"]
    assert data["candidate_pool"]["mean_jaccard_offline_vs_live"] == pytest.approx(1.0)

    for feature in ("semantic_similarity", "title_similarity", "text_containment"):
        assert data["features"][feature]["max_abs_diff"] < 1e-3

    assert data["full_scan_top10_kept_by_pool"]["mean_recall"] >= 0.9


def test_runner_requires_an_explicit_model_decision(tmp_path):

    done = subprocess.run(
        [sys.executable, str(SRC / "run_pipeline.py")],
        capture_output=True, text=True,
    )

    assert done.returncode != 0
    assert "--model" in done.stderr


def test_runner_accepts_legacy_flags_without_weakening_model_decision():

    legacy_model = subprocess.run(
        [sys.executable, str(SRC / "run_pipeline.py"), "--with-model", "--dry-run"],
        capture_output=True, text=True,
    )
    assert legacy_model.returncode == 0, legacy_model.stderr
    assert "Model decision: train" in legacy_model.stdout

    legacy_recommendations = subprocess.run(
        [sys.executable, str(SRC / "run_pipeline.py"), "--model", "auto",
         "--with-final-recommendations", "--verify-deployment", "--dry-run"],
        capture_output=True, text=True,
    )
    assert legacy_recommendations.returncode == 0, legacy_recommendations.stderr
    assert "Step 12: 12_final_recommendation_pipeline.py" in legacy_recommendations.stdout
    assert "Verify: verify_deployment.py" in legacy_recommendations.stdout

    missing_decision = subprocess.run(
        [sys.executable, str(SRC / "run_pipeline.py"), "--dry-run"],
        capture_output=True, text=True,
    )
    assert missing_decision.returncode != 0
    assert "--model is required" in missing_decision.stderr
