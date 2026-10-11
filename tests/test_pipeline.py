"""
End-to-end regression test: runs the REAL pipeline scripts in a temporary
copy of the project on a tiny synthetic dataset.

It guards the five problems this change fixed:

  1. the trained-model path is used by Step 10 / Step 13 when a model exists
  2. Steps 9, 10 and 13 score with the same definition
  3. Step 4 consumes the Step 3 PDF parser output
  4. scanned / image-only resumes are read with OCR (or fail loudly)
  5. only ESCO *skills* become skills

A stand-in for sentence-transformers (tests/stubs) keeps it offline.
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
import pdf_text

from synthetic_project import (
    CSV_ONLY_IDS,
    GENERIC,
    PDF_ONLY_DIGITAL_ID,
    PDF_ONLY_SCANNED_ID,
    SCANNED_CSV_IDS,
    SKILLS,
    build_project,
)

TESTS = Path(__file__).resolve().parent
SRC = TESTS.parent / "src"

needs_ocr = pytest.mark.skipif(
    pdf_text.select_ocr_engine() is None,
    reason="no OCR engine installed (pip install pytesseract / rapidocr-onnxruntime)",
)


def run_script(root: Path, script: str, *args: str, env_extra=None):

    env = dict(os.environ)
    env["PYTHONPATH"] = str(TESTS / "stubs")
    env.pop("MATCH_MODEL_PATH", None)
    env.update(env_extra or {})

    return subprocess.run(
        [sys.executable, str(root / "src" / script), *args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def must_run(root, script, *args, **kwargs):

    done = run_script(root, script, *args, **kwargs)

    assert done.returncode == 0, (
        f"{script} failed ({done.returncode})\n--- stdout ---\n"
        f"{done.stdout[-2500:]}\n--- stderr ---\n{done.stderr[-2500:]}"
    )

    return done


@pytest.fixture(scope="module")
def project(tmp_path_factory):

    root = tmp_path_factory.mktemp("pipeline_project")

    shutil.copytree(
        SRC, root / "src", ignore=shutil.ignore_patterns("__pycache__")
    )

    build_project(root)

    for script in (
        "02_clean_datasets.py",
        "03_resume_parser.py",
        "04_prepare_text_data.py",
        "05_extract_skills.py",
        "06_map_skills_to_esco.py",
        "07_build_matching_dataset.py",
        "08_build_match_features.py",
        "09_create_training_labels.py",
        "10_prepare_training_data.py",
    ):
        must_run(root, script)

    return root


def processed(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(root / "data" / "processed" / name)


# ----------------------------------------------------------------------
# 3. Step 4 consumes the Step 3 parser output
# ----------------------------------------------------------------------

def test_step4_uses_the_pdf_parser_output(project):

    resumes = processed(project, "resumes_text_processed.csv")

    sources = resumes.set_index("resume_id")["text_source"]

    assert sources[1] == "pdf_parser"
    assert sources[CSV_ONLY_IDS[0]] == "csv"          # no PDF -> CSV text

    # PDF-only resumes (no CSV row) are not lost.
    assert PDF_ONLY_DIGITAL_ID in sources.index
    assert PDF_ONLY_SCANNED_ID in sources.index

    # ... and their category comes from the PDF folder name.
    categories = resumes.set_index("resume_id")["category"]

    assert categories[PDF_ONLY_DIGITAL_ID] == "INFORMATION-TECHNOLOGY"

    report = json.loads(
        (project / "reports" / "text_processing_report.json").read_text()
    )

    assert "resumes_parsed.jsonl" in json.dumps(report)


def test_step12_checks_step10_scorer_fingerprint(project):
    # The synthetic fixture ran Steps 2-10, so Step 12 should accept the current
    # fingerprint and materialize recommendations without retraining anything.
    must_run(project, "12_final_recommendation_pipeline.py")

    report_path = project / "reports" / "final_recommendation_report.json"
    report = json.loads(report_path.read_text())
    assert report["validation"]["scorer_fingerprint_matches_step10"] is True
    assert report["score_presentation"]["display_name"] == "Similarity Score"

    predictions_path = project / "data" / "evaluation" / "step10_matching_predictions.csv"
    predictions = pd.read_csv(predictions_path)
    predictions.loc[0, "scorer_fingerprint"] = "stale-fingerprint"
    predictions.to_csv(predictions_path, index=False)
    done = run_script(project, "12_final_recommendation_pipeline.py")
    assert done.returncode != 0
    assert "fingerprint" in (done.stdout + done.stderr).lower()


# ----------------------------------------------------------------------
# 4. Scanned resumes
# ----------------------------------------------------------------------

@needs_ocr
def test_scanned_pdfs_are_ocrd_and_flow_into_the_dataset(project):

    resumes = processed(project, "resumes_skills_extracted.csv")

    row = resumes[resumes["resume_id"] == PDF_ONLY_SCANNED_ID].iloc[0]

    assert bool(row["ocr_used"]) is True
    assert row["text_source"] == "pdf_parser"
    assert row["skill_count"] >= 1

    parsing = json.loads(
        (project / "reports" / "resume_parsing_report.json").read_text()
    )

    assert parsing["ocr"]["pdfs_read_with_ocr"] >= 1
    assert parsing["ocr"]["scanned_pdfs_without_ocr_engine"] == 0


def test_step3_reports_scanned_pdfs_when_no_ocr_engine_exists(project):

    done = must_run(
        project, "03_resume_parser.py",
        env_extra={"RESUME_OCR_ENGINE": "none"},
    )

    report = json.loads(
        (project / "reports" / "resume_parsing_report.json").read_text()
    )

    assert report["ocr"]["scanned_pdfs_without_ocr_engine"] >= 1
    assert report["pdf_status_counts"].get("ocr_unavailable", 0) >= 1

    # restore the normal outputs for the other tests
    must_run(project, "03_resume_parser.py")

    assert "WARNING" in done.stdout or "OCR" in done.stdout


# ----------------------------------------------------------------------
# 5. Only ESCO skills count
# ----------------------------------------------------------------------

def test_only_skills_become_skills(project):

    found = set()

    for column in ("skill_labels",):
        for cell in processed(project, "resumes_skills_extracted.csv")[column]:
            found |= set(json.loads(cell))

    allowed = set(sum(SKILLS.values(), [])) | set(GENERIC)

    assert found <= allowed
    assert found.isdisjoint({"teacher", "nurse", "c"})
    assert "python" in found

    report = json.loads(
        (project / "reports" / "skill_extraction_report.json").read_text()
    )

    assert report["vocabulary"]["non_skill_rows_ignored"] == 2
    assert report["method"]["skills_table_only"] is True
    assert report["method"]["uses_hidden_labels"] is False


def test_step7_union_keeps_candidates_without_shared_skills(project):
    matching = processed(project, "resume_job_matching.csv")
    assert "candidate_retrieval_sources" in matching.columns
    assert matching["candidate_retrieval_sources"].fillna("").str.len().gt(0).all()
    assert (matching["shared_skill_count"] == 0).any()

    report = json.loads((project / "reports" / "matching_dataset_report.json").read_text())
    assert report["matching_statistics"]["candidate_policy"] == (
        "semantic_top_k_union_lexical_top_k_union_esco_skill_top_k"
    )


# ----------------------------------------------------------------------
# 2. One scoring definition across steps 9, 10 and 13
# ----------------------------------------------------------------------

def test_step9_evidence_equals_the_shared_formula(project):

    matching = processed(project, "resume_job_matching_dataset.csv")

    expected = mc.formula_score(matching)

    assert matching["evidence_score"].to_numpy() == pytest.approx(
        expected, abs=1e-9
    )

    # Missing categories are not a match any more.
    assert set(matching["category_match_signal"].unique()) <= {0, 1}


def test_step10_without_a_model_uses_the_formula(project):

    predictions = pd.read_csv(
        project / "data" / "evaluation" / "step10_matching_predictions.csv"
    )

    assert set(predictions["scoring_method"]) == {mc.METHOD_FORMULA}

    assert predictions["hybrid_match_score"].to_numpy() == pytest.approx(
        predictions["formula_match_score"].to_numpy()
    )

    # Step 9's evidence_score and Step 10's formula are the same number.
    assert predictions["hybrid_match_score"].to_numpy() == pytest.approx(
        predictions["evidence_score"].to_numpy(), abs=1e-9
    )

    report = json.loads(
        (project / "reports" / "hybrid_matching_model_report.json").read_text()
    )

    assert report["scoring"]["trained_model_in_use"] is False
    assert report["scoring"]["formula_weights"] == mc.WEIGHTS
    assert report["step9_evidence_score"]["equals_shared_formula"] is True


def test_step13_formula_matches_what_the_app_would_compute(project):

    must_run(project, "13_explainable_rag_recommendation.py")

    analysis = json.loads(
        (
            project / "data" / "recommendations" / "test_resume_analysis.json"
        ).read_text()
    )

    assert analysis["matching"]["scoring_method"] == mc.METHOD_FORMULA
    assert analysis["matching"]["formula_weights"] == mc.WEIGHTS

    # skills come from the ESCO skills table
    assert set(analysis["input"]["skill_list"]) <= (
        set(sum(SKILLS.values(), [])) | set(GENERIC)
    )
    assert "python" in analysis["input"]["skill_list"]

    scores = [
        job["match_score"] for job in analysis["recommendations"]["top_jobs"]
    ]

    assert scores == sorted(scores, reverse=True)


@needs_ocr
def test_step13_reads_a_scanned_resume(project):

    must_run(
        project, "13_explainable_rag_recommendation.py",
        "data/test_resume_scanned.pdf",
    )

    analysis = json.loads(
        (
            project / "data" / "recommendations" / "test_resume_analysis.json"
        ).read_text()
    )

    assert analysis["input"]["pdf_extraction"]["ocr_used"] is True
    assert "python" in analysis["input"]["skill_list"]


def test_step13_stops_with_a_clear_message_for_a_scanned_resume_without_ocr(
    project,
):

    done = run_script(
        project, "13_explainable_rag_recommendation.py",
        "data/test_resume_scanned.pdf",
        env_extra={"RESUME_OCR_ENGINE": "none"},
    )

    assert done.returncode != 0
    assert "scanned" in (done.stderr + done.stdout).lower()


# ----------------------------------------------------------------------
# 1. The trained model is used once it exists
# ----------------------------------------------------------------------

def test_step10_and_step13_use_a_trained_model_when_present(project):

    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(0)

    X = pd.DataFrame(rng.random((400, 9)), columns=mc.MODEL_FEATURES)
    y = 0.6 * X["semantic_similarity"] + 0.4 * X["job_skill_coverage"]

    estimator = HistGradientBoostingRegressor(max_iter=30, random_state=0)
    estimator.fit(X, y)

    mc.save_model_bundle(
        project / "models" / "match_regressor.joblib",
        estimator,
        mc.MODEL_FEATURES,
        target="unit-test",
    )

    try:

        must_run(project, "10_prepare_training_data.py")

        predictions = pd.read_csv(
            project / "data" / "evaluation" / "step10_matching_predictions.csv"
        )

        assert set(predictions["scoring_method"]) == {mc.METHOD_MODEL}
        assert not np.allclose(
            predictions["hybrid_match_score"], predictions["formula_match_score"]
        )

        must_run(project, "13_explainable_rag_recommendation.py")

        analysis = json.loads(
            (
                project / "data" / "recommendations" / "test_resume_analysis.json"
            ).read_text()
        )

        assert analysis["matching"]["scoring_method"] == mc.METHOD_MODEL
        assert analysis["matching"]["trained_model_in_use"] is True

    finally:

        (project / "models" / "match_regressor.joblib").unlink()
        must_run(project, "10_prepare_training_data.py")


def test_corrupt_model_file_falls_back_to_the_formula(project):

    model = project / "models" / "match_regressor.joblib"
    model.parent.mkdir(exist_ok=True)
    model.write_bytes(b"corrupt")

    try:

        must_run(project, "10_prepare_training_data.py")

        report = json.loads(
            (project / "reports" / "hybrid_matching_model_report.json").read_text()
        )

        assert report["scoring"]["trained_model_in_use"] is False
        assert "could not load" in report["scoring"]["fallback_reason"]

    finally:

        model.unlink()
        must_run(project, "10_prepare_training_data.py")
