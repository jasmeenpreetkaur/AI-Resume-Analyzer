"""Tests for src/occupation_inference.py (synthetic ESCO files, fake embedder)."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import occupation_inference as oi  # noqa: E402


DIM = 64


def embed_text(text: str) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    for word in re.findall(r"[a-z0-9+#]+", text.lower()):
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


class CountingEncoder:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        return np.stack([embed_text(t) for t in texts])


def skill(n):
    return f"http://data.europa.eu/esco/skill/{n}"


def occ(n):
    return f"http://data.europa.eu/esco/occupation/{n}"


@pytest.fixture()
def esco_dir(tmp_path):
    occupations = pd.DataFrame(
        {
            "conceptUri": [occ(1), occ(2), occ(3), occ(4)],
            "preferredLabel": ["software developer", "data analyst", "chef", "pastry chef"],
            "altLabels": ["programmer\nsoftware engineer", "analyst\nstatistician", "cook\nhead cook", "baker"],
            "description": [
                "Writes python code and builds software applications.",
                "Analyses data with sql and statistics.",
                "Prepares meals in a restaurant kitchen.",
                "Prepares pastries and desserts in a bakery kitchen.",
            ],
        }
    )
    relations = pd.DataFrame(
        [
            # software developer: 10 essential skills (1..10), 1 optional
            *[(occ(1), "essential", skill(i)) for i in range(1, 11)],
            (occ(1), "optional", skill(50)),
            # data analyst: 2 essential skills only (small profile -> floor matters)
            (occ(2), "essential", skill(1)),
            (occ(2), "essential", skill(2)),
            # chef: 3 essential
            *[(occ(3), "essential", skill(i)) for i in (60, 61, 62)],
            (occ(4), "essential", skill(60)),
            # a relation for an occupation that is not in the occupations file
            ("http://data.europa.eu/esco/occupation/999", "essential", skill(1)),
        ],
        columns=["occupationUri", "relationType", "skillUri"],
    )
    directory = tmp_path / "external"
    directory.mkdir()
    occupations.to_csv(directory / oi.OCCUPATIONS_FILE, index=False)
    relations.to_csv(directory / oi.RELATIONS_FILE, index=False)
    return directory


@pytest.fixture()
def index(esco_dir, tmp_path):
    return oi.build_occupation_index(
        CountingEncoder(), model_name="fake", data_dir=esco_dir, cache_dir=tmp_path / "cache"
    )


# ---- index building and caching ----------------------------------------------

def test_index_contents(index):
    assert len(index) == 4 and index.embeddings.shape == (4, DIM)
    assert np.allclose(np.linalg.norm(index.embeddings, axis=1), 1.0, atol=1e-5)
    assert len(index.essential[occ(1)]) == 10
    assert index.optional[occ(1)] == frozenset({skill(50)})
    assert "http://data.europa.eu/esco/occupation/999" not in index.essential


def test_occupation_text_includes_alternatives_and_description():
    text = oi.occupation_text("chef", "cook\nhead cook\nkitchen lead\nfourth", "Prepares meals.")
    assert text.startswith("chef. cook, head cook, kitchen lead.")
    assert "fourth" not in text and text.endswith("Prepares meals.")


def test_embeddings_are_cached_and_invalidated(esco_dir, tmp_path):
    cache = tmp_path / "cache"
    encoder = CountingEncoder()
    kwargs = dict(data_dir=esco_dir, cache_dir=cache)

    oi.build_occupation_index(encoder, model_name="m1", **kwargs)
    oi.build_occupation_index(encoder, model_name="m1", **kwargs)
    assert encoder.calls == 1  # second build came from the cache

    oi.build_occupation_index(encoder, model_name="m2", **kwargs)
    assert encoder.calls == 2  # different embedding model -> rebuilt

    frame = pd.read_csv(esco_dir / oi.OCCUPATIONS_FILE)
    frame.loc[0, "description"] = "completely different description"
    frame.to_csv(esco_dir / oi.OCCUPATIONS_FILE, index=False)
    oi.build_occupation_index(encoder, model_name="m2", **kwargs)
    assert encoder.calls == 3  # changed occupation text -> rebuilt


def test_corrupt_cache_is_rebuilt(esco_dir, tmp_path):
    cache = tmp_path / "cache"
    encoder = CountingEncoder()
    oi.build_occupation_index(encoder, model_name="m", data_dir=esco_dir, cache_dir=cache)
    (cache / "occupation_embeddings.npy").write_bytes(b"garbage")
    oi.build_occupation_index(encoder, model_name="m", data_dir=esco_dir, cache_dir=cache)
    assert encoder.calls == 2


def test_missing_or_malformed_files_raise_clear_errors(esco_dir, tmp_path):
    with pytest.raises(oi.OccupationDataError, match="not found"):
        oi.build_occupation_index(CountingEncoder(), model_name="m", data_dir=tmp_path / "nope", cache_dir=tmp_path / "c")

    pd.DataFrame({"x": [1]}).to_csv(esco_dir / oi.RELATIONS_FILE, index=False)
    with pytest.raises(oi.OccupationDataError, match="needs occupationUri"):
        oi.build_occupation_index(CountingEncoder(), model_name="m", data_dir=esco_dir, cache_dir=tmp_path / "c2")


def test_wrong_encoder_shape_is_rejected(esco_dir, tmp_path):
    with pytest.raises(oi.OccupationDataError, match="shape"):
        oi.build_occupation_index(lambda texts: np.zeros((1, 3)), model_name="m", data_dir=esco_dir, cache_dir=tmp_path / "c")


# ---- inference ----------------------------------------------------------------

def test_semantic_signal_alone_finds_the_right_field(index):
    resume = embed_text("experienced chef preparing meals in a restaurant kitchen")
    top = oi.infer_occupations(index, resume, [], top_k=2)
    assert top.loc[0, "occupation"] in {"chef", "pastry chef"}
    assert (top["essential_matched"] == 0).all()  # no skills -> no skill evidence, no crash


@pytest.fixture()
def flat_index(esco_dir, tmp_path):
    """Every occupation has the SAME embedding, so semantic similarity is a true tie."""
    return oi.build_occupation_index(
        lambda texts: np.ones((len(texts), DIM), dtype=np.float32),
        model_name="flat", data_dir=esco_dir, cache_dir=tmp_path / "flat_cache",
    )


def test_skill_evidence_breaks_a_semantic_tie(flat_index):
    resume = np.ones(DIM, dtype=np.float32)
    with_skills = oi.infer_occupations(flat_index, resume, [skill(i) for i in range(1, 9)], top_k=4)
    assert with_skills.loc[0, "occupation"] == "software developer"
    assert with_skills.loc[0, "essential_matched"] == 8 and with_skills.loc[0, "essential_total"] == 10


def test_small_profiles_cannot_win_with_one_or_two_skills(flat_index):
    # data analyst has only 2 essential skills; matching both must NOT read as 100%
    resume = np.ones(DIM, dtype=np.float32)
    top = oi.infer_occupations(flat_index, resume, [skill(1), skill(2)], top_k=4)
    analyst = top[top["occupation"] == "data analyst"].iloc[0]
    assert analyst["essential_matched"] == 2 and analyst["essential_total"] == 2
    assert analyst["skill_signal"] == pytest.approx(2 / oi.MIN_ESSENTIAL_FOR_COVERAGE)


def test_matched_essential_uris_are_reported(index):
    resume = embed_text("python developer")
    top = oi.infer_occupations(index, resume, [skill(3), skill(4), skill(77)], top_k=4)
    developer = top[top["occupation"] == "software developer"].iloc[0]
    assert developer["matched_essential_uris"] == [skill(3), skill(4)]


def test_results_are_deterministic_and_sorted(index):
    resume = embed_text("data analysis with sql")
    first = oi.infer_occupations(index, resume, [skill(1)], top_k=4)
    second = oi.infer_occupations(index, resume, [skill(1)], top_k=4)
    pd.testing.assert_frame_equal(first, second)
    assert first["score"].is_monotonic_decreasing


def test_bad_embeddings_raise(index):
    with pytest.raises(ValueError):
        oi.infer_occupations(index, np.zeros(DIM), [])
    with pytest.raises(ValueError):
        oi.infer_occupations(index, np.ones(DIM + 5), [])


def test_suggest_queries_drops_near_duplicates():
    frame = pd.DataFrame({"occupation": ["software developer", "Developer software", "data analyst", "chef", "pilot"]})
    assert oi.suggest_queries(frame, n=3) == ["software developer", "data analyst", "chef"]
    assert oi.suggest_queries(pd.DataFrame({"occupation": []})) == []


def test_floor_limits_the_advantage_of_tiny_profiles(flat_index):
    # resume evidences 2 skills shared by both profiles.
    # analyst: 2 of 2 essential (raw 100%) -> floored to 2/8; developer: 2 of 10 -> 0.2
    resume = np.ones(DIM, dtype=np.float32)
    top = oi.infer_occupations(flat_index, resume, [skill(1), skill(2)], top_k=4)
    signals = top.set_index("occupation")["skill_signal"]
    assert signals["data analyst"] == pytest.approx(0.25)  # not 1.0
    assert signals["software developer"] == pytest.approx(0.20)
