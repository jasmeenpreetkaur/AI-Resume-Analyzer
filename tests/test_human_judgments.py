from __future__ import annotations

import json

import pandas as pd
import pytest

import matching_core as mc
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "11b_evaluate_human_judgments.py"
SPEC = importlib.util.spec_from_file_location("human_judgments", SCRIPT)
hj = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(hj)


def test_export_template_adds_blank_judgment_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("MATCH_MODEL_PATH", str(tmp_path / "missing-model.joblib"))
    rankings = tmp_path / "rankings.csv"
    pd.DataFrame({
        "resume_id": ["r1", "r1", "r2"],
        "job_id": ["j1", "j2", "j1"],
        "job_title": ["Developer", "Analyst", "Developer"],
        "resume_text": ["python engineer"] * 3,
        "job_description": ["build python systems"] * 3,
        "hybrid_match_score": [0.9, 0.8, 0.7],
        "scorer_fingerprint": [mc.MatchScorer.load().scorer_fingerprint] * 3,
    }).to_csv(rankings, index=False)

    out_path = hj.export_review_template(rankings, tmp_path / "review.csv", top_k=1)
    out = pd.read_csv(out_path, dtype={"human_relevance": "string"})

    assert len(out) == 2
    assert set(out["resume_id"]) == {"r1", "r2"}
    assert out["human_relevance"].isna().all()
    assert out["reviewer_id"].isna().all()
    assert out["scorer_fingerprint"].str.len().gt(0).all()


def test_export_includes_source_resume_and_job_text(tmp_path, monkeypatch):
    monkeypatch.setenv("MATCH_MODEL_PATH", str(tmp_path / "missing-model.joblib"))
    monkeypatch.setattr(hj, "DEFAULT_RESUME_TEXT_PATH", tmp_path / "resume_text.csv")
    monkeypatch.setattr(hj, "DEFAULT_JOB_TEXT_PATH", tmp_path / "job_text.csv")
    scorer = mc.MatchScorer.load()
    pd.DataFrame({"resume_id": ["r1"], "resume_text": ["Python developer with SQL experience"]}).to_csv(
        tmp_path / "resume_text.csv", index=False
    )
    pd.DataFrame({"job_id": ["j1"], "job_description": ["Build Python and SQL services"]}).to_csv(
        tmp_path / "job_text.csv", index=False
    )
    rankings = tmp_path / "rankings.csv"
    pd.DataFrame({
        "resume_id": ["r1"], "job_id": ["j1"], "job_title": ["Developer"],
        "hybrid_match_score": [0.9], "scorer_fingerprint": [scorer.scorer_fingerprint],
    }).to_csv(rankings, index=False)

    out = pd.read_csv(hj.export_review_template(rankings, tmp_path / "review.csv"))
    assert out.loc[0, "resume_text"] == "Python developer with SQL experience"
    assert out.loc[0, "job_description"] == "Build Python and SQL services"


def test_export_rejects_stale_or_unfingerprinted_rankings(tmp_path, monkeypatch):
    monkeypatch.setenv("MATCH_MODEL_PATH", str(tmp_path / "missing-model.joblib"))
    rankings = tmp_path / "rankings.csv"
    pd.DataFrame({
        "resume_id": ["r1"],
        "job_id": ["j1"],
        "hybrid_match_score": [0.9],
        "scorer_fingerprint": ["old-code-version"],
    }).to_csv(rankings, index=False)

    with pytest.raises(ValueError, match="mixed, or stale"):
        hj.export_review_template(rankings, tmp_path / "review.csv")

    pd.DataFrame({"job_id": ["j1"], "hybrid_match_score": [0.9]}).to_csv(rankings, index=False)
    with pytest.raises(ValueError, match="must contain scorer_fingerprint"):
        hj.export_review_template(rankings, tmp_path / "review.csv")


def test_invalid_nonblank_label_is_not_silently_treated_as_blank(tmp_path, monkeypatch):
    monkeypatch.setenv("MATCH_MODEL_PATH", str(tmp_path / "missing-model.joblib"))
    scorer = mc.MatchScorer.load()
    annotated = tmp_path / "annotated.csv"
    pd.DataFrame({
        "resume_id": ["r1"],
        "job_id": ["j1"],
        "resume_text": ["resume context"],
        "job_description": ["job context"],
        "human_relevance": ["abc"],
        "reviewer_id": ["reviewer-a"],
        "scorer_fingerprint": [scorer.scorer_fingerprint],
        "recommendation_rank": [1],
    }).to_csv(annotated, index=False)

    with pytest.raises(ValueError, match="Invalid human_relevance labels"):
        hj.evaluate_judgments(annotated, tmp_path / "report.json")


def test_duplicate_rating_from_same_reviewer_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("MATCH_MODEL_PATH", str(tmp_path / "missing-model.joblib"))
    scorer = mc.MatchScorer.load()
    annotated = tmp_path / "duplicate.csv"
    row = {
        "resume_id": "r1", "job_id": "j1", "resume_text": "resume context",
        "job_description": "job context", "human_relevance": 2,
        "reviewer_id": "reviewer-a", "scorer_fingerprint": scorer.scorer_fingerprint,
        "recommendation_rank": 1,
    }
    pd.DataFrame([row, row]).to_csv(annotated, index=False)

    with pytest.raises(ValueError, match="duplicate reviewer/resume/job"):
        hj.evaluate_judgments(annotated, tmp_path / "report.json")


def test_small_sample_cannot_unlock_suitability_wording(tmp_path, monkeypatch):
    monkeypatch.setenv("MATCH_MODEL_PATH", str(tmp_path / "missing-model.joblib"))
    scorer = mc.MatchScorer.load()
    annotated = tmp_path / "annotated.csv"
    pd.DataFrame({
        "resume_id": ["r1", "r1"],
        "job_id": ["j1", "j2"],
        "resume_text": ["resume context"] * 2,
        "job_description": ["job context"] * 2,
        "human_relevance": [0, 2],
        "reviewer_id": ["reviewer-a", "reviewer-b"],
        "scorer_fingerprint": [scorer.scorer_fingerprint] * 2,
        "recommendation_rank": [1, 2],
        "similarity_score": [0.9, 0.7],
    }).to_csv(annotated, index=False)

    report = hj.evaluate_judgments(
        annotated, tmp_path / "report.json", approve_suitability_wording=True
    )
    saved = json.loads((tmp_path / "report.json").read_text())

    assert report["suitability_wording_allowed"] is False
    assert report["human_validation_status"] == "not_validated"
    assert saved["reasons_not_approved"]


def test_approved_sufficient_judgments_unlock_only_matching_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setenv("MATCH_MODEL_PATH", str(tmp_path / "missing-model.joblib"))
    report_path = tmp_path / "validation.json"
    monkeypatch.setenv("MATCH_VALIDATION_REPORT_PATH", str(report_path))
    scorer = mc.MatchScorer.load()
    rows = []
    for resume_number in range(20):
        for rank in range(1, 11):
            # Each resume has eight relevant and two non-relevant top-10 jobs.
            label = 2 if rank <= 8 else 0
            for reviewer in ("reviewer-a", "reviewer-b"):
                rows.append({
                    "resume_id": f"r{resume_number:02d}",
                    "job_id": f"j{resume_number:02d}-{rank:02d}",
                    "resume_text": f"resume {resume_number} engineering experience",
                    "job_description": f"job {rank} engineering responsibilities",
                    "human_relevance": label,
                    "reviewer_id": reviewer,
                    "scorer_fingerprint": scorer.scorer_fingerprint,
                    "recommendation_rank": rank,
                    "similarity_score": 1.0 - rank / 100,
                })
    annotated = tmp_path / "annotated.csv"
    pd.DataFrame(rows).to_csv(annotated, index=False)

    report = hj.evaluate_judgments(
        annotated, report_path, approve_suitability_wording=True
    )
    assert report["suitability_wording_allowed"] is True
    assert report["context_coverage"] == 1.0
    assert scorer.claim_label == "human_validated_relevance"

    # The approval must not survive a different active scorer fingerprint.
    payload = json.loads(report_path.read_text())
    payload["scorer_fingerprint"] = "stale-version"
    report_path.write_text(json.dumps(payload))
    fresh_scorer = mc.MatchScorer.load()
    assert fresh_scorer.claim_label == "experimental_similarity_not_validated"
